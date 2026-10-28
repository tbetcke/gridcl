//! `DistGraph` and `Partitioner` (Listing S3).

use gridcl_parallel::{CommDyn, Result};

/// A distributed CSR graph in the ParMETIS and PT-Scotch convention: rank r holds the global
/// vertices `vtxdist[r]..vtxdist[r + 1]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DistGraph {
    /// The first global vertex of every rank, and the total; `nranks + 1` entries.
    pub vtxdist: Vec<i64>,
    /// Row offsets of the local vertices; `num_local + 1` entries.
    pub xadj: Vec<i64>,
    /// Global ids of the neighbours (global cell ids).
    pub adjncy: Vec<i64>,
    /// Vertex weights, from the cost model per (cell type, degree); `None` for unit weights.
    pub vwgt: Option<Vec<i64>>,
}

/// A graph partitioner.
pub trait Partitioner {
    /// The partitioner's name, for logs.
    fn name(&self) -> &str;
    /// Destination rank of every local graph vertex (cell). Collective.
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>>;
}
