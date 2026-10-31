//! Scotch on rank 0 (docs/design.md, "Partitioning"; ADR-0029).
//!
//! The `scotch` crate keeps its graph and strategy handles private and offers no strategy
//! builder, so the graph, architecture, strategy and context are `scotch_sys` objects owned by
//! [`Owned`] (ADR-0029, item 1).

use std::os::raw::c_int;
use std::ptr;
use std::sync::PoisonError;

use gridcl_parallel::{CommDyn, GridError, Result};
use scotch::Num;
use scotch_sys as s;

use crate::graph::validate;
use crate::owned::{self, Owned};
use crate::rank0::{Csr, RANDOM_STATE, gather_to_root, narrow, scatter_parts};
use crate::{DistGraph, Partitioner};

/// Scotch (`SCOTCH_graphMap` onto a complete graph of `nparts` vertices) on rank 0: the graph
/// is gathered on rank 0, partitioned there and the destinations scattered back (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count. The
/// strategy is `SCOTCH_STRATQUALITY` with a balance target of 3%; vertex weights are used, edge
/// weights are not.
///
/// # Errors
///
/// The validation errors of ADR-0029, item 9 (`TooLarge` for totals beyond `SCOTCH_Num`), and
/// `PartitionerFailed { name: "scotch", code }` with Scotch's status if the library fails;
/// agreed over the ranks.
///
/// # Example
///
/// ```
/// use gridcl_parallel::Serial;
/// use gridcl_partition::{DistGraph, Partitioner, Scotch};
///
/// // A path of four vertices with weights 1, 1, 1, 3.
/// let g = DistGraph {
///     vtxdist: vec![0, 4],
///     xadj: vec![0, 1, 3, 5, 6],
///     adjncy: vec![1, 0, 2, 1, 3, 2],
///     vwgt: Some(vec![1, 1, 1, 3]),
/// };
/// let parts = Scotch::default().seed(7).partition(&Serial, &g, 2)?;
/// assert_eq!(parts.len(), 4);
/// assert!(parts.iter().all(|&p| p == 0 || p == 1));
/// assert_ne!(parts[0], parts[3]);
/// # Ok::<(), gridcl_parallel::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Scotch {
    seed: u32,
}

impl Scotch {
    /// The partitioner with random seed `seed`.
    pub fn seed(self, seed: u32) -> Self {
        Scotch { seed }
    }

    /// `SCOTCH_graphMap` of the whole graph onto the complete graph of `nparts` vertices; rank
    /// 0 only.
    fn run(&self, csr: &Csr, nparts: usize) -> Result<Vec<i32>> {
        let n = csr.xadj.len() - 1;
        if n == 0 {
            return Ok(Vec::new());
        }
        // Declared before the graph, which borrows them, so they outlive it.
        let verttab: Vec<Num> = narrow(&csr.xadj)?;
        let edgetab: Vec<Num> = narrow(&csr.adjncy)?;
        let velotab: Option<Vec<Num>> = csr.vwgt.as_deref().map(narrow).transpose()?;
        let vertnbr = Num::try_from(n).map_err(|_| GridError::TooLarge)?;
        let edgenbr = Num::try_from(edgetab.len()).map_err(|_| GridError::TooLarge)?;
        let partnbr = Num::try_from(nparts).map_err(|_| GridError::TooLarge)?;

        // SAFETY: `SCOTCH_graphInit` initialises the graph it is given; `SCOTCH_graphExit`
        // releases it.
        let mut graph =
            unsafe { Owned::new("scotch", |p| s::SCOTCH_graphInit(p), s::SCOTCH_graphExit)? };
        // SAFETY: graph is initialised; verttab holds vertnbr + 1 offsets into edgetab's
        // edgenbr entries, velotab (if any) vertnbr loads, all validated (symmetric, in range,
        // no self-loops, non-negative loads) and declared before graph, so they outlive it;
        // null vendtab, vlbltab and edlotab mean compact rows, no labels and unit edge loads.
        status(unsafe {
            s::SCOTCH_graphBuild(
                graph.as_mut_ptr(),
                0,
                vertnbr,
                verttab.as_ptr(),
                ptr::null(),
                velotab.as_ref().map_or(ptr::null(), |v| v.as_ptr()),
                ptr::null(),
                edgenbr,
                edgetab.as_ptr(),
                ptr::null(),
            )
        })?;
        // SAFETY: as for the graph, with the architecture's functions.
        let mut arch =
            unsafe { Owned::new("scotch", |p| s::SCOTCH_archInit(p), s::SCOTCH_archExit)? };
        // SAFETY: arch is initialised; partnbr ≥ 2.
        status(unsafe { s::SCOTCH_archCmplt(arch.as_mut_ptr(), partnbr) })?;
        // SAFETY: as for the graph, with the strategy's functions.
        let mut strat =
            unsafe { Owned::new("scotch", |p| s::SCOTCH_stratInit(p), s::SCOTCH_stratExit)? };
        // SAFETY: strat is initialised and empty.
        status(unsafe {
            s::SCOTCH_stratGraphMapBuild(
                strat.as_mut_ptr(),
                s::SCOTCH_STRATQUALITY as Num,
                partnbr,
                0.03,
            )
        })?;
        // A threaded Scotch uses non-deterministic algorithms unless told otherwise, so the
        // same seed alone does not give the same parts; the context makes them deterministic.
        // SAFETY: as for the graph, with the context's functions.
        let mut context = unsafe {
            Owned::new(
                "scotch",
                |p| s::SCOTCH_contextInit(p),
                s::SCOTCH_contextExit,
            )?
        };
        // SAFETY: context is initialised; the option number is one of the header's.
        status(unsafe {
            s::SCOTCH_contextOptionSetNum(
                context.as_mut_ptr(),
                s::SCOTCH_OPTIONNUMDETERMINISTIC as c_int,
                1,
            )
        })?;
        // SAFETY: `SCOTCH_contextBindGraph` initialises the container it is given to refer to
        // graph and context, both built above and declared before it, so the container is
        // released first; `SCOTCH_graphExit` releases a container.
        let mut bound = unsafe {
            Owned::new(
                "scotch",
                |p| s::SCOTCH_contextBindGraph(context.as_mut_ptr(), graph.as_mut_ptr(), p),
                s::SCOTCH_graphExit,
            )?
        };

        let mut parttab: Vec<Num> = vec![0; n];
        // Scotch's random state is process-global (ADR-0029, item 8): the context's private
        // generator is cloned from it, then seeded.
        let lock = RANDOM_STATE.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: context, bound, arch and strat are initialised and built above; the global
        // random state is only read, by the clone, under the lock; parttab holds one entry per
        // vertex.
        let code = unsafe {
            let code = s::SCOTCH_contextRandomClone(context.as_mut_ptr());
            if code == 0 {
                // The bits of the seed.
                s::SCOTCH_contextRandomSeed(context.as_mut_ptr(), self.seed as Num);
                s::SCOTCH_contextRandomReset(context.as_mut_ptr());
                s::SCOTCH_graphMap(
                    bound.as_mut_ptr(),
                    arch.as_mut_ptr(),
                    strat.as_mut_ptr(),
                    parttab.as_mut_ptr(),
                )
            } else {
                code
            }
        };
        drop(lock);
        status(code)?;
        // `SCOTCH_Num` is 32 or 64 bits, as Scotch was built.
        #[allow(clippy::useless_conversion)]
        parttab
            .into_iter()
            .map(|p| i32::try_from(p).map_err(|_| GridError::TooLarge))
            .collect()
    }
}

/// `PartitionerFailed { name: "scotch", code }` with a Scotch status other than 0.
fn status(code: c_int) -> Result<()> {
    owned::status("scotch", code)
}

impl Partitioner for Scotch {
    fn name(&self) -> &str {
        "scotch"
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let checked = validate(comm, g, nparts, "scotch", i64::from(Num::MAX))?;
        if nparts == 1 {
            return Ok(vec![0; (checked.end - checked.begin) as usize]);
        }
        let parts = gather_to_root(comm, g, &checked)?.map(|csr| self.run(&csr, nparts));
        scatter_parts(comm, &g.vtxdist, parts)
    }
}
