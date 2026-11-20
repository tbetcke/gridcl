//! ParMETIS (docs/design.md, "Partitioning"; ADR-0029).

use std::ptr;
use std::sync::PoisonError;

use parmetis_sys::{self as p, IDX_BITS, idx_t as Idx, real_t as Real};
use tetraloma_parallel::{CommDyn, GridError, Result};

use crate::graph::{Checked, agree, local_weights, signed_max, validate};
use crate::rank0::{RANDOM_STATE, narrow};
use crate::{DistGraph, Partitioner};

const NAME: &str = "parmetis";

// Design: "Partitioning"; ADR-0029 (item 9: the validation; item 12: no empty ranks),
// ADR-0033 (`parmetis` excludes `metis`).
/// ParMETIS (`ParMETIS_V3_PartKway`), the distributed graph partitioner of the METIS family,
/// with the feature `parmetis`: every rank takes part, on the communicator's MPI handle.
///
/// No rank holds the whole graph. The partition is seeded (default 0): a fixed seed gives the
/// same partition for the same graph and number of ranks. The balance target is 3% (`ubvec`
/// 1.03) with equal target part weights; vertex weights are used, edge weights are not. With
/// `nparts == 1` every vertex goes to part 0 without a library call.
///
/// ParMETIS 4 fails when a rank holds no vertices, so every rank must hold at least one.
///
/// The feature `parmetis` excludes `metis`: the METIS that the crate `metis` builds would replace
/// ParMETIS's own at run time. ParMETIS's licence restricts redistribution; check its terms before
/// you redistribute binaries built with the feature `parmetis`. See [Cargo features] and the
/// tutorial [Distributed meshes].
///
/// # Errors
///
/// Before any library call, every rank checks its part of the graph:
///
/// - `LengthMismatch` for `nparts` 0, for inconsistent arrays, or for a `vtxdist` that
///   disagrees with the ranks' vertex counts;
/// - `IndexOutOfRange` for a neighbour outside the graph;
/// - `PartitionerFailed { name: "parmetis", code: -2 }` for a self-loop, and `code: -3` for an
///   edge whose reverse edge is missing;
/// - `TooLarge` for a negative weight, or for totals beyond ParMETIS's integer type `idx_t`;
/// - `MessageTooLarge` if the check of the reverse edges would send a rank more than
///   `i32::MAX` values.
///
/// With `nparts > 1`, `NeedsMpiCommunicator("parmetis")` if the communicator has no MPI handle
/// ([`CommDyn::raw`] is `None`, as for `Serial`), `PartitionerFailed { name: "parmetis",
/// code: -1 }` if a rank holds no vertices, and `PartitionerFailed { name: "parmetis", code }`
/// with ParMETIS's status if the library fails. Errors are agreed over the ranks: the ranks
/// without a local error return `RemoteFailure` with the lowest failing rank.
///
/// # Example
///
/// The example runs as one process without MPI. One part needs no library call; two parts need
/// an MPI communicator. Under `mpirun`, pass the MPI communicator `Mpi` of tetraloma-parallel
/// (for example `Mpi::world(&universe)`) instead of `Serial`, with `nparts` set to
/// `comm.size()`.
///
/// ```
/// use tetraloma_parallel::{GridError, Serial};
/// use tetraloma_partition::{DistGraph, ParMetis, Partitioner};
///
/// // A path of four vertices with weights 1, 1, 1, 3, all on this rank.
/// let g = DistGraph {
///     vtxdist: vec![0, 4],
///     xadj: vec![0, 1, 3, 5, 6],
///     adjncy: vec![1, 0, 2, 1, 3, 2],
///     vwgt: Some(vec![1, 1, 1, 3]),
/// };
/// let partitioner = ParMetis::default().seed(7);
/// assert_eq!(partitioner.partition(&Serial, &g, 1)?, [0, 0, 0, 0]);
/// assert!(matches!(
///     partitioner.partition(&Serial, &g, 2),
///     Err(GridError::NeedsMpiCommunicator("parmetis"))
/// ));
/// # Ok::<(), GridError>(())
/// ```
///
/// [Cargo features]: https://tetracl.github.io/tetraloma/features.html#partitioners
/// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
#[derive(Clone, Copy, Debug, Default)]
pub struct ParMetis {
    seed: u32,
}

impl ParMetis {
    /// The partitioner with random seed `seed`; the default is 0. The same seed gives the same
    /// partition for the same graph and number of ranks. See [Distributed meshes].
    ///
    /// # Example
    ///
    /// The example runs as one process without MPI, so it can only ask for one part, which needs
    /// no library call. Under `mpirun`, pass the MPI communicator `Mpi` of tetraloma-parallel
    /// instead of `Serial`.
    ///
    /// ```
    /// use tetraloma_parallel::Serial;
    /// use tetraloma_partition::{DistGraph, ParMetis, Partitioner};
    ///
    /// // A ring of six vertices, all on this rank.
    /// let g = DistGraph {
    ///     vtxdist: vec![0, 6],
    ///     xadj: vec![0, 2, 4, 6, 8, 10, 12],
    ///     adjncy: vec![1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0],
    ///     vwgt: None,
    /// };
    /// let partitioner = ParMetis::default().seed(42);
    /// assert_eq!(partitioner.name(), "parmetis");
    /// assert_eq!(partitioner.partition(&Serial, &g, 1)?, [0; 6]);
    /// # Ok::<(), tetraloma_parallel::GridError>(())
    /// ```
    ///
    /// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
    pub fn seed(self, seed: u32) -> Self {
        ParMetis { seed }
    }

    /// `ParMETIS_V3_PartKway` of the validated graph `g` on the MPI handle of `comm`, into
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
        // vtxdist is the same on every rank (validated), so so is this.
        if g.vtxdist.windows(2).any(|w| w[0] == w[1]) {
            return Err(failed(-1));
        }
        let n = (checked.end - checked.begin) as usize;
        let arrays = (|| -> Result<_> {
            let vtxdist: Vec<Idx> = narrow(&g.vtxdist)?;
            let xadj: Vec<Idx> = narrow(&g.xadj)?;
            let adjncy: Vec<Idx> = narrow(&g.adjncy)?;
            let vwgt: Option<Vec<Idx>> = local_weights(g, checked.weighted)
                .map(|w| narrow(&w))
                .transpose()?;
            let nparts = Idx::try_from(nparts).map_err(|_| GridError::TooLarge)?;
            Ok((vtxdist, xadj, adjncy, vwgt, nparts))
        })();
        let (mut vtxdist, mut xadj, mut adjncy, mut vwgt, mut nparts) = agree(comm, arrays)?;

        // Vertex weights only (2) or none (0); C numbering; one constraint.
        let mut wgtflag: Idx = if vwgt.is_some() { 2 } else { 0 };
        let (mut numflag, mut ncon): (Idx, Idx) = (0, 1);
        let mut tpwgts: Vec<Real> = vec![1.0 / nparts as Real; nparts as usize];
        let mut ubvec: [Real; 1] = [1.03];
        // The bits of the seed in options[2]; options[0] = 1 makes ParMETIS read them.
        let mut options: [Idx; 3] = [1, 0, self.seed as Idx];
        let mut edgecut: Idx = 0;
        let mut part: Vec<Idx> = vec![0; n];
        let mut raw = raw;

        // ParMETIS seeds the C library's process-global rand() (ADR-0029, item 8).
        let lock = RANDOM_STATE.lock().unwrap_or_else(PoisonError::into_inner);
        // SAFETY: vtxdist holds nranks + 1 offsets, xadj n + 1 offsets into adjncy, vwgt (if
        // any) n weights, all validated (symmetric, in range, no self-loops, non-negative
        // weights, totals within idx_t) and the same vtxdist on every rank, with no empty rank;
        // a null adjwgt with wgtflag 0 or 2 means no edge weights. tpwgts holds ncon · nparts
        // entries, ubvec ncon, options 3, part n; ParMETIS reads the inputs and writes edgecut
        // and part only. raw is a valid communicator for the lifetime of comm; every rank of
        // it makes this call.
        let status = unsafe {
            p::ParMETIS_V3_PartKway(
                vtxdist.as_mut_ptr(),
                xadj.as_mut_ptr(),
                adjncy.as_mut_ptr(),
                vwgt.as_mut().map_or(ptr::null_mut(), |v| v.as_mut_ptr()),
                ptr::null_mut(),
                &mut wgtflag,
                &mut numflag,
                &mut ncon,
                &mut nparts,
                tpwgts.as_mut_ptr(),
                ubvec.as_mut_ptr(),
                options.as_mut_ptr(),
                &mut edgecut,
                part.as_mut_ptr(),
                &mut raw,
            )
        };
        drop(lock);
        let parts = if status == p::METIS_OK {
            // `idx_t` is 32 or 64 bits, as ParMETIS was built.
            #[allow(clippy::useless_conversion)]
            part.into_iter()
                .map(|p| i32::try_from(p).map_err(|_| GridError::TooLarge))
                .collect()
        } else {
            Err(failed(status))
        };
        agree(comm, parts)
    }
}

/// `PartitionerFailed { name: "parmetis", code }`.
fn failed(code: i32) -> GridError {
    GridError::PartitionerFailed { name: NAME, code }
}

impl Partitioner for ParMetis {
    fn name(&self) -> &str {
        NAME
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let checked = validate(comm, g, nparts, NAME, signed_max(IDX_BITS))?;
        if nparts == 1 {
            return Ok(vec![0; (checked.end - checked.begin) as usize]);
        }
        self.run(comm, g, &checked, nparts)
    }
}
