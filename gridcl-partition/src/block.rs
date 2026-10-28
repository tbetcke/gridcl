//! The built-in block partitioner (ADR-0026, item 2).

use gridcl_parallel::{CommDyn, Result};

use crate::{DistGraph, Partitioner};

/// Built-in block partitioner: global graph vertex i goes to rank ⌊i · nparts / N⌋, with
/// N = `vtxdist[nranks]` (ADR-0026).
#[derive(Clone, Copy, Debug, Default)]
pub struct Block;

impl Partitioner for Block {
    fn name(&self) -> &str {
        "block"
    }
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let _ = (comm, g, nparts);
        todo!("M6 T8: Block::partition")
    }
}
