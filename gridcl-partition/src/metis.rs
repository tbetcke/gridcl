//! METIS on rank 0 (docs/design.md, "Partitioning"; ADR-0029).

use std::sync::PoisonError;

use gridcl_parallel::{CommDyn, GridError, Result};
use metis::Idx;
use metis::option::{Seed, UFactor};

use crate::graph::validate;
use crate::rank0::{Csr, RANDOM_STATE, gather_to_root, narrow, scatter_parts};
use crate::{DistGraph, Partitioner};

/// METIS (`METIS_PartGraphKway`) on rank 0: the graph is gathered on rank 0, partitioned there
/// and the destinations scattered back (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count. The
/// balance target is 3% (`ufactor` 30); vertex weights are used, edge weights are not. The
/// feature `metis` excludes `parmetis` (ADR-0033).
///
/// # Errors
///
/// The validation errors of ADR-0029, item 9 (`TooLarge` for totals beyond the 32-bit
/// `idx_t`), and `PartitionerFailed { name: "metis", code }` with METIS's status (−2 input,
/// −3 memory, −4 other) if the library fails; agreed over the ranks.
///
/// # Example
///
/// ```
/// use gridcl_parallel::Serial;
/// use gridcl_partition::{DistGraph, Metis, Partitioner};
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
/// # Ok::<(), gridcl_parallel::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Metis {
    seed: u32,
}

impl Metis {
    /// The partitioner with random seed `seed`.
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
