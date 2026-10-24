//! Cell and entity blocks, sort-based entity creation, connectivity and permutation info
//! (docs/design.md, "Topology construction on the device"; ADR-0012).
//!
//! [`Topology`] holds the cell blocks of a mesh with its vertex `IndexMap` and vertex origin
//! ids. Permutation info is computed when it is created; edges and faces are created on demand
//! by [`Topology::create_entities`], and upward connectivity by
//! [`Topology::create_connectivity`].
//!
//! Origin ids live on the device as interleaved (hi ^ 0x8000_0000, lo) u32 pairs
//! ([`encode_origin_ids`]), so unsigned pair order equals i64 order.

#[cfg(feature = "bench")]
#[doc(hidden)]
pub mod bench;
mod cell_block;
mod connectivity;
mod emission;
mod entity;
mod keys;
mod origin;
mod permutation;
mod renumber;
mod topology;

pub use cell_block::CellBlock;
pub use entity::{Connectivity, EntityBlock};
pub use gridcl_device::ConnKey;
pub use origin::{decode_origin_ids, encode_origin_ids, origin_ids_fit_u32};
pub use topology::Topology;
