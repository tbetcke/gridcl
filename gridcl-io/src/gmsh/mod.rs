//! Gmsh MSH 4.1 on the host (docs/design.md, "Import and export / Gmsh MSH 4.1"; ADR-0022).
//!
//! - [`MshFile`]: the sections of a file, as stored ([`parse`], [`read_file`],
//!   [`MshFile::write_to`]), ASCII or binary, in either byte order and with either `size_t`
//!   width.
//! - [`ElementType`], [`element_type`], [`lagrange_element_type`]: the element table of A.7.
//! - [`GmshMesh`]: the mesh a file describes, with cell blocks in A.5 node order, tag elements
//!   and names ([`to_gmsh_mesh`], [`from_gmsh_mesh`]).

mod convert;
mod element_type;
mod model;
mod parse;
mod write;

pub use convert::{GmshCellBlock, GmshMesh, GmshTagGroup, from_gmsh_mesh, to_gmsh_mesh};
pub use element_type::{ElementType, element_type, lagrange_element_type};
pub use model::{
    ElementBlock, Entities, Entity, MshFile, NodeBlock, PartitionedEntities, PartitionedEntity,
    Periodic, PhysicalName,
};
pub use parse::{parse, read_file};
pub use write::Encoding;

/// The format name in `GridError::Parse` and `GridError::UnsupportedElementType`.
pub(crate) const FORMAT: &str = "Gmsh";
