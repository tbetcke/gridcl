//! VTU and PVTU on the host (docs/design.md, "Import and export / VTK"; ADR-0024).
//!
//! - [`vtk_cell_type`]: the cell-type table of A.8.
//! - [`VtkMesh`]: the mesh a file describes, with cell blocks in A.5 node order; [`to_vtu`]
//!   turns it into a [`VtuFile`].
//! - [`VtuFile`]: the arrays of one `UnstructuredGrid` piece, as stored
//!   ([`VtuFile::write_to`], [`parse_vtu`]); [`write_pvtu`] writes the parallel header.
//!
//! `gridcl::vtk::VtkWriter` downloads meshes.

mod cell_type;
mod convert;
mod model;
mod parse;
mod write;

pub use cell_type::vtk_cell_type;
pub use convert::to_vtu;
pub use model::{DataArray, DataValues, VtkCellBlock, VtkMesh, VtuFile};
pub use parse::parse_vtu;
pub use write::{Binary, write_pvtu};

/// The format name in `GridError::Parse`.
#[allow(dead_code)] // used by the parser (T4)
pub(crate) const FORMAT: &str = "VTU";
