//! METIS on rank 0 (docs/design.md, "Partitioning"; ADR-0029).

use std::sync::PoisonError;

use metis::Idx;
use metis::option::{Seed, UFactor};
use tetraloma_parallel::{CommDyn, GridError, Result};

use crate::graph::validate;
use crate::rank0::{Csr, RANDOM_STATE, gather_to_root, narrow, scatter_parts};
use crate::{DistGraph, Partitioner};

// Design: "Partitioning"; ADR-0029 (item 9: the validation), ADR-0033 (`metis` excludes
// `parmetis`).
/// METIS (`METIS_PartGraphKway`) on rank 0, with the feature `metis`: the graph is gathered on
/// rank 0, partitioned there, and the destinations are scattered back.
///
/// Rank 0 holds the whole graph, which suits meshes of up to about 10⁷ cells. The partition is
/// seeded (default 0): a fixed seed gives the same partition for the same graph and number of
/// ranks. The balance target is 3% (`ufactor` 30); vertex weights are used, edge weights are not.
/// METIS is built from source by the crate `metis`, so it needs no system library. The feature
/// `metis` excludes `parmetis`. See [Cargo features] and the tutorial [Distributed meshes].
///
/// # Errors
///
/// Before any library call, every rank checks its part of the graph:
///
/// - `LengthMismatch` for `nparts` 0, for inconsistent arrays, or for a `vtxdist` that
///   disagrees with the ranks' vertex counts;
/// - `IndexOutOfRange` for a neighbour outside the graph;
/// - `PartitionerFailed { name: "metis", code: -2 }` for a self-loop, and `code: -3` for an
///   edge whose reverse edge is missing;
/// - `TooLarge` for a negative weight, or for totals beyond METIS's 32-bit `idx_t`;
/// - `MessageTooLarge` if the check of the reverse edges would send a rank more than
///   `i32::MAX` values.
///
/// If the library fails, `PartitionerFailed { name: "metis", code }` carries METIS's status
/// (−2 input, −3 memory, −4 other). Errors are agreed over the ranks: the ranks without a local
/// error return `RemoteFailure` with the lowest failing rank.
///
/// # Example
///
/// ```
/// use tetraloma_parallel::Serial;
/// use tetraloma_partition::{DistGraph, Metis, Partitioner};
///
/// // A path of four vertices with weights 1, 1, 1, 3.
/// let g = DistGraph {
///     vtxdist: vec![0, 4],
///     xadj: vec![0, 1, 3, 5, 6],
///     adjncy: vec![1, 0, 2, 1, 3, 2],
///     vwgt: Some(vec![1, 1, 1, 3]),
/// };
/// let parts = Metis::default().seed(7).partition(&Serial, &g, 2)?;
/// assert_eq!(parts.len(), 4);
/// assert!(parts.iter().all(|&p| p == 0 || p == 1));
/// assert_ne!(parts[0], parts[3]);
/// # Ok::<(), tetraloma_parallel::GridError>(())
/// ```
///
/// [Cargo features]: https://tetracl.github.io/tetraloma/features.html#partitioners
/// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
#[derive(Clone, Copy, Debug, Default)]
pub struct Metis {
    seed: u32,
}

impl Metis {
    /// The partitioner with random seed `seed`; the default is 0. The same seed gives the same
    /// partition for the same graph and number of ranks. See [Distributed meshes].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_parallel::Serial;
    /// use tetraloma_partition::{DistGraph, Metis, Partitioner};
    ///
    /// // A ring of six vertices.
    /// let g = DistGraph {
    ///     vtxdist: vec![0, 6],
    ///     xadj: vec![0, 2, 4, 6, 8, 10, 12],
    ///     adjncy: vec![1, 5, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0],
    ///     vwgt: None,
    /// };
    /// let first = Metis::default().seed(42).partition(&Serial, &g, 3)?;
    /// let again = Metis::default().seed(42).partition(&Serial, &g, 3)?;
    /// assert_eq!(first, again);
    /// assert!(first.iter().all(|&p| (0..3).contains(&p)));
    /// # Ok::<(), tetraloma_parallel::GridError>(())
    /// ```
    ///
    /// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
    pub fn seed(self, seed: u32) -> Self {
        Metis { seed }
    }

    /// `METIS_PartGraphKway` on the whole graph; rank 0 only.
    fn run(&self, csr: &Csr, nparts: usize) -> Result<Vec<i32>> {
        let n = csr.xadj.len() - 1;
        if n == 0 {
            return Ok(Vec::new());
        }
        let xadj: Vec<Idx> = narrow(&csr.xadj)?;
        let adjncy: Vec<Idx> = narrow(&csr.adjncy)?;
        let vwgt: Option<Vec<Idx>> = csr.vwgt.as_deref().map(narrow).transpose()?;
        let nparts = Idx::try_from(nparts).map_err(|_| GridError::TooLarge)?;
        let failed = |code| GridError::PartitionerFailed {
            name: "metis",
            code,
        };
        let mut graph = metis::Graph::new(1, nparts, &xadj, &adjncy)
            .map_err(|_| failed(METIS_ERROR_INPUT))?
            .set_option(UFactor(30))
            // The bits of the seed; u32::MAX gives −1, METIS's default seed.
            .set_option(Seed(self.seed as Idx));
        if let Some(vwgt) = &vwgt {
            graph = graph.set_vwgt(vwgt);
        }
        let mut part = vec![0; n];
        // GKlib's random state is process-global (ADR-0029, item 8).
        let lock = RANDOM_STATE.lock().unwrap_or_else(PoisonError::into_inner);
        let status = graph.part_kway(&mut part);
        drop(lock);
        status.map_err(|e| {
            failed(match e {
                metis::Error::Input => METIS_ERROR_INPUT,
                metis::Error::Memory => METIS_ERROR_MEMORY,
                metis::Error::Other => METIS_ERROR,
            })
        })?;
        // `idx_t` is 32 bits vendored, 32 or 64 with `use-system` (ADR-0029, item 14).
        #[allow(clippy::useless_conversion)]
        part.into_iter()
            .map(|p| i32::try_from(p).map_err(|_| GridError::TooLarge))
            .collect()
    }
}

/// METIS's `rstatus_et` codes.
const METIS_ERROR_INPUT: i32 = -2;
const METIS_ERROR_MEMORY: i32 = -3;
const METIS_ERROR: i32 = -4;

impl Partitioner for Metis {
    fn name(&self) -> &str {
        "metis"
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let checked = validate(comm, g, nparts, "metis", i64::from(Idx::MAX))?;
        if nparts == 1 {
            return Ok(vec![0; (checked.end - checked.begin) as usize]);
        }
        let parts = gather_to_root(comm, g, &checked)?.map(|csr| self.run(&csr, nparts));
        scatter_parts(comm, &g.vtxdist, parts)
    }
}
