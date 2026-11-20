// Design: "Import and export / VTK"; ADR-0024 (VTU), ADR-0035 (PVTU of distributed meshes,
// VTKHDF); Appendix A.5 (node order), A.8 (cell types), A.9 (VTK node order).

//! VTU, PVTU and VTKHDF files on the host.
//!
//! - [`vtk_cell_type`]: the VTK cell type that tetraloma writes for a cell type and degree.
//! - [`VtkMesh`]: the mesh a file describes, with cell blocks in tetraloma's node order;
//!   [`to_vtu`] turns it into a [`VtuFile`].
//! - [`VtuFile`]: the arrays of one `UnstructuredGrid` piece, as stored.
//!   [`VtuFile::write_to`] writes it and [`parse_vtu`] reads it; [`write_pvtu`] writes the
//!   header of a parallel file.
//! - `write_vtkhdf` (feature `hdf5`): the pieces of all ranks in one VTKHDF file.
//!
//! The crate `tetraloma` writes device meshes with `VtkWriter`, which uses this module; the
//! tutorial [Reading and writing files] shows it. The chapter [File formats] lists the cell types
//! and VTK's node orders.
//!
//! # Example
//!
//! ```
//! use tetraloma_dims::CellType;
//! use tetraloma_io::vtk::{Binary, VtkCellBlock, VtkMesh, parse_vtu, to_vtu};
//!
//! // One triangle in the plane.
//! let mesh = VtkMesh {
//!     gdim: 2,
//!     nodes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
//!     blocks: vec![VtkCellBlock { cell: CellType::Triangle, degree: 1, nodes: vec![0, 1, 2] }],
//! };
//! let (file, _) = to_vtu(&mesh, true, false)?;
//! let mut bytes = Vec::new();
//! file.write_to(&mut bytes, Binary::Appended, false)?;
//! assert_eq!(parse_vtu(&bytes)?, file);
//! # Ok::<(), tetraloma_device::GridError>(())
//! ```
//!
//! [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#writing-vtu
//! [File formats]: https://tetracl.github.io/tetraloma/elements/file-formats.html#vtk

mod base64;
mod cell_type;
mod convert;
#[cfg(feature = "hdf5")]
mod hdf;
mod model;
mod parse;
mod write;

pub use cell_type::vtk_cell_type;
pub use convert::to_vtu;
#[cfg(feature = "hdf5")]
pub use hdf::write_vtkhdf;
pub use model::{DataArray, DataValues, VtkCellBlock, VtkMesh, VtuFile};
pub use parse::parse_vtu;
pub use write::{Binary, write_pvtu};

/// The format name in `GridError::Parse`.
pub(crate) const FORMAT: &str = "VTU";
