//! `DistGraph`, `Partitioner` and the block partitioner; METIS, Scotch, PT-Scotch and ParMETIS
//! behind features.
//!
//! M6 adds the graph, the trait and [`Block`] (Listing S3; ADR-0026); milestone M7 adds the
//! external partitioners.

mod block;
mod graph;

pub use block::Block;
pub use graph::{DistGraph, Partitioner};
