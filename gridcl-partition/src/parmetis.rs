//! ParMETIS (docs/design.md, "Partitioning"; ADR-0029).

use gridcl_parallel::{CommDyn, Result};

use crate::{DistGraph, Partitioner};

/// ParMETIS (`ParMETIS_V3_PartKway`) on the communicator's MPI handle (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParMetis {
    seed: u32,
}

impl ParMetis {
    /// The partitioner with random seed `seed`.
    pub fn seed(self, seed: u32) -> Self {
        ParMetis { seed }
    }
}

impl Partitioner for ParMetis {
    fn name(&self) -> &str {
        "parmetis"
    }
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let _ = (self.seed, comm, g, nparts);
        todo!("ParMetis::partition (M7)")
    }
}
