//! Scotch on rank 0 (docs/design.md, "Partitioning"; ADR-0029).

use gridcl_parallel::{CommDyn, Result};

use crate::{DistGraph, Partitioner};

/// Scotch (`SCOTCH_graphMap` onto a complete graph of `nparts` vertices) on rank 0: the graph is gathered on rank 0, partitioned there and the destinations scattered back (ADR-0029).
///
/// Seeded (default 0); a fixed seed reproduces the partition for a fixed rank count.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scotch {
    seed: u32,
}

impl Scotch {
    /// The partitioner with random seed `seed`.
    pub fn seed(self, seed: u32) -> Self {
        Scotch { seed }
    }
}

impl Partitioner for Scotch {
    fn name(&self) -> &str {
        "scotch"
    }
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let _ = (self.seed, comm, g, nparts);
        todo!("Scotch::partition (M7)")
    }
}
