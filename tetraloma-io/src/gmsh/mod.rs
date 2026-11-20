// Design: "Import and export / Gmsh MSH 4.1"; ADR-0022 (the reader, writer and conversion),
// ADR-0035 (chunked reading on several ranks); Appendix A.5 (node order), A.7 (element table).

//! Gmsh MSH 4.1 files on the host.
//!
//! - [`MshFile`]: the sections of a file, as stored. [`parse`] and [`read_file`] read ASCII and
//!   binary files in either byte order and with either `size_t` width; [`MshFile::write_to`]
//!   writes ASCII or binary.
//! - [`ElementType`], [`element_type`] and [`lagrange_element_type`]: the Gmsh element types
//!   that tetraloma reads, by type id.
//! - [`GmshMesh`]: the mesh a file describes, with cell blocks in tetraloma's node order, the
//!   lower-dimensional elements as tags, and the physical names. [`to_gmsh_mesh`] makes it from
//!   a file and [`from_gmsh_mesh`] turns it into one.
//! - [`MshIndex`]: the layout of a binary file, for reading ranges of nodes and elements
//!   ([`NodeChunk`], [`ElementChunk`]) on every rank.
//!
//! The crate `tetraloma` reads files into device meshes with `gmsh::read` and writes them with
//! `GmshWriter`; the tutorial [Reading and writing files] shows both. The chapter [File formats]
//! lists the element types and Gmsh's node orders.
//!
//! # Example
//!
//! ```
//! use tetraloma_dims::CellType;
//! use tetraloma_io::gmsh::{from_gmsh_mesh, parse, to_gmsh_mesh};
//! use tetraloma_parallel::Serial;
//!
//! // Two triangles of the unit square.
//! let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
//!     $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
//!     $Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
//! let mesh = to_gmsh_mesh(&parse(text)?, &Serial, None)?;
//! assert_eq!(mesh.blocks[0].cell, CellType::Triangle);
//! assert_eq!(mesh.blocks[0].nodes, [0, 1, 2, 0, 2, 3]);
//!
//! // Back to a file: the same nodes and elements.
//! let file = from_gmsh_mesh(&mesh)?;
//! assert_eq!(file.node_blocks[0].tags, [1, 2, 3, 4]);
//! assert_eq!(file.element_blocks[0].nodes, [1, 2, 3, 1, 3, 4]);
//! # Ok::<(), tetraloma_device::GridError>(())
//! ```
//!
//! [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#reading-gmsh
//! [File formats]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh

mod chunk;
mod convert;
mod element_type;
mod index;
mod model;
mod parse;
mod write;

#[doc(hidden)]
pub use chunk::{
    BlockHeader, CellChunk, Layout, complete, element_range, geometric_dim, node_range,
};
pub use convert::{
    FileNumbering, GmshCellBlock, GmshMesh, GmshTagGroup, from_gmsh_mesh, from_numbered_gmsh_mesh,
    to_gmsh_mesh,
};
pub use element_type::{ElementType, element_type, lagrange_element_type};
pub use index::{ElementChunk, MshIndex, NodeChunk};
pub use model::{
    ElementBlock, Entities, Entity, MshFile, NodeBlock, PartitionedEntities, PartitionedEntity,
    Periodic, PhysicalName,
};
pub use parse::{parse, read_file};
pub use write::Encoding;

/// The format name in `GridError::Parse` and `GridError::UnsupportedElementType`.
pub(crate) const FORMAT: &str = "Gmsh";
