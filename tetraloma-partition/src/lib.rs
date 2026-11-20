// Design: "Partitioning", Listing S3; ADR-0026 (the graph, the trait and `Block`, M6) and
// ADR-0029 (the external partitioners, M7).

//! Distributed graphs and the partitioners that assign the cells of a mesh to MPI ranks.
//!
//! When you build a mesh on several ranks, tetraloma forms its dual graph: one vertex per cell,
//! and an edge between two cells that share a facet. The graph is spread over the ranks as a
//! [`DistGraph`] in the format of ParMETIS and PT-Scotch. A [`Partitioner`] then gives every
//! vertex its destination rank, and the cells move there. Vertex weights estimate the work of
//! each cell from its type and degree.
//!
//! - [`Block`] is built in and needs no library: it splits the cells into contiguous ranges in
//!   input order and ignores the edges.
//! - `Metis` (feature `metis`) and `Scotch` (feature `scotch`) gather the graph on rank 0 and
//!   partition it there.
//! - `PtScotch` (feature `ptscotch`) and `ParMetis` (feature `parmetis`) partition in parallel
//!   on the MPI communicator.
//!
//! The graph partitioners cut fewer facets between ranks than `Block`, so the ghost layers and
//! halo exchanges stay small. Each takes a seed and gives the same partition for the same
//! graph, seed and number of ranks. The features `metis` and `parmetis` exclude each other: the
//! two METIS libraries they link clash at run time.
//!
//! You usually reach these types through the crate `tetraloma`, which re-exports them, and pass a
//! partitioner to its mesh builder or Gmsh reader. The tutorial [Distributed meshes] shows them at
//! work, and [Cargo features] says which system libraries each feature needs.
//!
//! # Example
//!
//! ```
//! use tetraloma_parallel::Serial;
//! use tetraloma_partition::{Block, DistGraph, Partitioner};
//!
//! // A path of four cells, 0 - 1 - 2 - 3, all on this rank.
//! let g = DistGraph {
//!     vtxdist: vec![0, 4],
//!     xadj: vec![0, 1, 3, 5, 6],
//!     adjncy: vec![1, 0, 2, 1, 3, 2],
//!     vwgt: None,
//! };
//! assert_eq!(Block.partition(&Serial, &g, 2)?, [0, 0, 1, 1]);
//! # Ok::<(), tetraloma_parallel::GridError>(())
//! ```
//!
//! [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
//! [Cargo features]: https://tetracl.github.io/tetraloma/features.html#partitioners

// The vendored METIS and ParMETIS's METIS define the same symbols; on Linux one replaces the
// other at run time and ParMETIS crashes (ADR-0033).
#[cfg(all(feature = "metis", feature = "parmetis"))]
compile_error!(
    "the features `metis` and `parmetis` are mutually exclusive: the vendored METIS and \
     ParMETIS's METIS clash at run time (docs/adr/0033-metis-and-parmetis-are-mutually-exclusive.md)"
);

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
