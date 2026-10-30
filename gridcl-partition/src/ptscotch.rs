//! PT-Scotch (docs/design.md, "Partitioning"; ADR-0029).

use gridcl_parallel::{CommDyn, Result};

use crate::{DistGraph, Partitioner};

/// PT-Scotch, the distributed graph partitioner (`SCOTCH_dgraphPart`), on the communicator's MPI handle (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count.
#[derive(Clone, Copy, Debug, Default)]
pub struct PtScotch {
    seed: u32,
}

impl PtScotch {
    /// The partitioner with random seed `seed`.
    pub fn seed(self, seed: u32) -> Self {
        PtScotch { seed }
    }
}

impl Partitioner for PtScotch {
    fn name(&self) -> &str {
        "ptscotch"
    }
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let _ = (self.seed, comm, g, nparts);
        todo!("PtScotch::partition (M7)")
    }
}
