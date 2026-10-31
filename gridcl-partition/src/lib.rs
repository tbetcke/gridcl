//! `DistGraph`, `Partitioner` and the block partitioner; METIS, Scotch, PT-Scotch and ParMETIS
//! behind features.
//!
//! M6 adds the graph, the trait and [`Block`] (Listing S3; ADR-0026); milestone M7 adds the
//! external partitioners (ADR-0029).

mod block;
mod graph;
#[cfg(feature = "metis")]
mod metis;
#[cfg(any(feature = "scotch", feature = "ptscotch"))]
mod owned;
#[cfg(feature = "parmetis")]
mod parmetis;
#[cfg(feature = "ptscotch")]
mod ptscotch;
mod rank0;
#[cfg(feature = "scotch")]
mod scotch;

pub use block::Block;
pub use graph::{DistGraph, Partitioner};
#[cfg(feature = "metis")]
pub use metis::Metis;
#[cfg(feature = "parmetis")]
pub use parmetis::ParMetis;
#[cfg(feature = "ptscotch")]
pub use ptscotch::PtScotch;
#[cfg(feature = "scotch")]
pub use scotch::Scotch;
