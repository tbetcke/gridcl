//! PT-Scotch (docs/design.md, "Partitioning"; ADR-0029, ADR-0031).
//!
//! The distributed graph, strategy and context are `ptscotch_sys` objects owned by [`Owned`].
//! Every step that can fail on some ranks only is agreed before the next collective library
//! call, so no rank waits in PT-Scotch for a rank that has already returned.

use std::os::raw::c_int;
use std::ptr;
use std::sync::PoisonError;

use gridcl_parallel::{CommDyn, GridError, Result};
use ptscotch_sys::{self as s, SCOTCH_NUM_BITS, SCOTCH_Num as Num};

use crate::graph::{Checked, agree, local_weights, signed_max, validate};
use crate::owned::{self, Owned};
use crate::rank0::{RANDOM_STATE, narrow};
use crate::{DistGraph, Partitioner};

const NAME: &str = "ptscotch";

/// PT-Scotch, the distributed graph partitioner (`SCOTCH_dgraphPart`), on the communicator's
/// MPI handle (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count. Each call
/// runs in a deterministic Scotch context with a private random generator seeded with the seed
/// (ADR-0031). The strategy is `SCOTCH_STRATQUALITY` with a balance target of 3%; vertex
/// weights are used, edge weights are not. With `nparts == 1` every vertex goes to part 0
/// without a library call.
///
/// # Errors
///
/// The validation errors of ADR-0029, item 9 (`TooLarge` for totals beyond `SCOTCH_Num`);
/// `NeedsMpiCommunicator("ptscotch")` if the communicator has no MPI handle
/// ([`CommDyn::raw`] is `None`, as for `Serial` and `ThreadComm`) and `nparts > 1`;
/// `PartitionerFailed { name: "ptscotch", code }` with PT-Scotch's status if the library fails.
/// Agreed over the ranks.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommDyn, Result};
/// use gridcl_partition::{DistGraph, Partitioner, PtScotch};
///
/// /// One part per rank, on an MPI communicator.
/// fn parts(comm: &dyn CommDyn, g: &DistGraph) -> Result<Vec<i32>> {
///     PtScotch::default().seed(7).partition(comm, g, comm.size())
/// }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct PtScotch {
    seed: u32,
}

impl PtScotch {
    /// The partitioner with random seed `seed`.
    pub fn seed(self, seed: u32) -> Self {
        PtScotch { seed }
    }

    /// `SCOTCH_dgraphPart` of the validated graph `g` on the MPI handle of `comm`, into
    /// `nparts ≥ 2` parts. Collective; the result is agreed.
    fn run(
        &self,
        comm: &dyn CommDyn,
        g: &DistGraph,
        checked: &Checked,
        nparts: usize,
    ) -> Result<Vec<i32>> {
        let raw = agree(
            comm,
            comm.raw().ok_or(GridError::NeedsMpiCommunicator(NAME)),
        )?;
        let n = (checked.end - checked.begin) as usize;
        // Declared before the graph, which borrows them, so they outlive it.
        let arrays = (|| -> Result<_> {
            let verttab: Vec<Num> = narrow(&g.xadj)?;
            let edgetab: Vec<Num> = narrow(&g.adjncy)?;
            let velotab: Option<Vec<Num>> = local_weights(g, checked.weighted)
                .map(|w| narrow(&w))
                .transpose()?;
            let num = |c: usize| Num::try_from(c).map_err(|_| GridError::TooLarge);
            let (vertnbr, edgenbr) = (num(n)?, num(edgetab.len())?);
            let (procnbr, partnbr) = (num(comm.size())?, num(nparts)?);
            Ok((
                verttab, edgetab, velotab, vertnbr, edgenbr, procnbr, partnbr,
            ))
        })();
        let (mut verttab, mut edgetab, mut velotab, vertnbr, edgenbr, procnbr, partnbr) =
            agree(comm, arrays)?;

        // The local objects; their creation is not collective.
        let objects = (|| -> Result<_> {
            // SAFETY: `SCOTCH_dgraphInit` initialises the graph it is given on raw, a valid
            // communicator for the lifetime of comm, which outlives the graph;
            // `SCOTCH_dgraphExit` releases it.
            let dgraph = unsafe {
                Owned::new(NAME, |p| s::SCOTCH_dgraphInit(p, raw), s::SCOTCH_dgraphExit)?
            };
            // SAFETY: as for the graph, with the strategy's functions.
            let mut strat =
                unsafe { Owned::new(NAME, |p| s::SCOTCH_stratInit(p), s::SCOTCH_stratExit)? };
            // SAFETY: strat is initialised and empty; procnbr is the size of comm.
            status(unsafe {
                s::SCOTCH_stratDgraphMapBuild(
                    strat.as_mut_ptr(),
                    s::SCOTCH_STRATQUALITY,
                    procnbr,
                    partnbr,
                    0.03,
                )
            })?;
            // A threaded Scotch uses non-deterministic algorithms unless told otherwise, so the
            // same seed alone does not give the same parts (ADR-0031).
            // SAFETY: as for the graph, with the context's functions.
            let mut context =
                unsafe { Owned::new(NAME, |p| s::SCOTCH_contextInit(p), s::SCOTCH_contextExit)? };
            // SAFETY: context is initialised; the option number is one of the header's.
            status(unsafe {
                s::SCOTCH_contextOptionSetNum(
                    context.as_mut_ptr(),
                    s::SCOTCH_OPTIONNUMDETERMINISTIC,
                    1,
                )
            })?;
            Ok((dgraph, strat, context))
        })();
        let (mut dgraph, mut strat, mut context) = agree(comm, objects)?;

        // SAFETY: dgraph is initialised; verttab holds vertnbr + 1 offsets into edgetab's
        // edgenbr entries, velotab (if any) vertnbr loads, all validated (symmetric, in range,
        // no self-loops, non-negative loads, totals within SCOTCH_Num) and declared before
        // dgraph, so they outlive it; PT-Scotch does not write them. Null vendloctab,
        // vlblloctab, edgegsttab and edloloctab mean compact rows, no labels, no ghost
        // numbering and unit edge loads. Every rank calls it, with weights on all or none.
        let built = status(unsafe {
            s::SCOTCH_dgraphBuild(
                dgraph.as_mut_ptr(),
                0,
                vertnbr,
                vertnbr,
                verttab.as_mut_ptr(),
                ptr::null_mut(),
                velotab.as_mut().map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
                ptr::null_mut(),
                edgenbr,
                edgenbr,
                edgetab.as_mut_ptr(),
                ptr::null_mut(),
                ptr::null_mut(),
            )
        });
        agree(comm, built)?;
        // SAFETY: `SCOTCH_contextBindDgraph` initialises the container it is given to refer to
        // dgraph and context, both built above and declared before it, so the container is
        // released first; `SCOTCH_dgraphExit` releases a container.
        let bound = unsafe {
            Owned::new(
                NAME,
                |p| s::SCOTCH_contextBindDgraph(context.as_mut_ptr(), dgraph.as_mut_ptr(), p),
                s::SCOTCH_dgraphExit,
            )
        };
        let mut bound = agree(comm, bound)?;

        let mut parttab: Vec<Num> = vec![0; n];
        // Scotch's random state is process-global (ADR-0029, item 8): the context's private
        // generator is cloned from it, then seeded.
        let lock = RANDOM_STATE.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: context is initialised; the global random state is only read, by the clone,
        // under the lock.
        let cloned = status(unsafe { s::SCOTCH_contextRandomClone(context.as_mut_ptr()) });
        let parted = agree(comm, cloned).and_then(|()| {
            // SAFETY: context, bound and strat are initialised and built above; parttab holds
            // one entry per local vertex; every rank of raw calls `SCOTCH_dgraphPart`.
            status(unsafe {
                // The bits of the seed.
                s::SCOTCH_contextRandomSeed(context.as_mut_ptr(), self.seed as Num);
                s::SCOTCH_contextRandomReset(context.as_mut_ptr());
                s::SCOTCH_dgraphPart(
                    bound.as_mut_ptr(),
                    partnbr,
                    strat.as_mut_ptr(),
                    parttab.as_mut_ptr(),
                )
            })
        });
        drop(lock);
        let parts = parted.and_then(|()| {
            // `SCOTCH_Num` is 32 or 64 bits, as Scotch was built.
            #[allow(clippy::useless_conversion)]
            parttab
                .into_iter()
                .map(|p| i32::try_from(p).map_err(|_| GridError::TooLarge))
                .collect()
        });
        agree(comm, parts)
    }
}

/// `PartitionerFailed { name: "ptscotch", code }` with a PT-Scotch status other than 0.
fn status(code: c_int) -> Result<()> {
    owned::status(NAME, code)
}

impl Partitioner for PtScotch {
    fn name(&self) -> &str {
        NAME
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let checked = validate(comm, g, nparts, NAME, signed_max(SCOTCH_NUM_BITS))?;
        if nparts == 1 {
            return Ok(vec![0; (checked.end - checked.begin) as usize]);
        }
        self.run(comm, g, &checked, nparts)
    }
}
