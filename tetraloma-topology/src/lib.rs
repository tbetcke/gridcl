// Design: "Topology construction on the device"; ADR-0012.
//! The topology of a mesh on the device: cell and entity blocks, sort-based entity creation,
//! connectivity, the owners of shared entities and permutation bits.
//!
//! [`Topology`] holds the cell blocks of a mesh with its vertex `IndexMap` and vertex origin
//! ids. It computes the permutation bits of every cell when it is created; edges and faces are
//! created on demand by [`Topology::create_entities`], and upward connectivity by
//! [`Topology::create_connectivity`]. Most programs never build a `Topology` themselves: a
//! `tetraloma::Mesh` owns one and offers these operations as its own methods, and the facade
//! crate `tetraloma` re-exports [`CellBlock`], [`EntityBlock`] and [`Connectivity`].
//!
//! Origin ids live on the device as interleaved (hi ^ 0x8000_0000, lo) u32 pairs
//! ([`encode_origin_ids`]), so unsigned pair order equals i64 order.
//!
//! The book explains cells, entities, connectivity and ownership in [Meshes and topology], and
//! shows how to read them in [Iterating over a mesh].
//!
//! # Example
//!
//! ```
//! use tetraloma::prelude::*;
//!
//! // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
//! # let client = tetraloma_testing::test_client();
//! // mesh: a `PlanarMesh<f32>`, the unit square in two triangles; the tutorial "Building a
//! // mesh" shows how to make a mesh.
//! # let mut mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build(&client)?;
//! // The square's four sides and its diagonal.
//! mesh.create_entities(1)?;
//! assert_eq!(mesh.entity_blocks(1)?[0].index_map.num_local(), 5);
//! # Ok::<(), GridError>(())
//! ```
//!
//! [Meshes and topology]: https://tetracl.github.io/tetraloma/concepts/meshes.html
//! [Iterating over a mesh]: https://tetracl.github.io/tetraloma/tutorials/iterate.html

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench;
mod cell_block;
mod connectivity;
mod emission;
mod entity;
mod facet_cells;
mod keys;
mod matching;
mod origin;
mod ownership;
mod permutation;
mod renumber;
mod sharers;
mod topology;
mod transpose;

pub use cell_block::CellBlock;
pub use entity::{Connectivity, EntityBlock};
#[doc(hidden)]
pub use facet_cells::{FacetCells, facet_cell_counts, facet_cells_of_parts, facet_rows};
pub use origin::{decode_origin_ids, encode_origin_ids, origin_ids_fit_u32};
pub use sharers::VertexSharers;
pub use tetraloma_device::ConnKey;
pub use topology::Topology;
