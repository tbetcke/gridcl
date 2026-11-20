// Design: "Import and export"; ADR-0022 (Gmsh, M4), ADR-0024 (VTU, M5), ADR-0035 (parallel I/O
// and VTKHDF, M8).

//! The host side of tetraloma's file input and output: Gmsh MSH 4.1 files, and VTU, PVTU and
//! VTKHDF output.
//!
//! This crate works on the host only. It parses and writes the files as plain Rust data, and
//! converts between a file and a host description of the mesh it holds. The crate `tetraloma`
//! builds device meshes from these descriptions and downloads meshes into them; most programs
//! use its `gmsh::read`, `GmshWriter` and `VtkWriter` instead of this crate. Use this crate
//! directly to inspect a file, to convert one without a device, or to write files from your own
//! arrays.
//!
//! - [`gmsh`]: the file model [`gmsh::MshFile`] with its parser and writer, the table of Gmsh
//!   element types, the conversion between a file and the host mesh [`gmsh::GmshMesh`], and the
//!   chunked reading of binary files on several ranks.
//! - [`vtk`]: the file model [`vtk::VtuFile`] with its writer and parser, the table of VTK cell
//!   types, the conversion from the host mesh [`vtk::VtkMesh`], the PVTU header and, with the
//!   feature `hdf5`, VTKHDF output.
//!
//! The tutorial [Reading and writing files] shows the readers and writers of `tetraloma`, and the
//! chapter [File formats] lists the cell types and node orders of both formats.
//!
//! # Example
//!
//! ```
//! use tetraloma_io::gmsh::{Encoding, parse, to_gmsh_mesh};
//! use tetraloma_parallel::Serial;
//!
//! // One triangle in the plane z = 0.
//! let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
//!     $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
//!     $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
//! let file = parse(text)?;
//! let mesh = to_gmsh_mesh(&file, &Serial, None)?;
//! assert_eq!((mesh.tdim, mesh.gdim), (2, 2));
//!
//! // Write the file as binary MSH 4.1 and read it back.
//! let mut bytes = Vec::new();
//! file.write_to(&mut bytes, Encoding::Binary)?;
//! assert_eq!(parse(&bytes)?, file);
//! # Ok::<(), tetraloma_device::GridError>(())
//! ```
//!
//! [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html
//! [File formats]: https://tetracl.github.io/tetraloma/elements/file-formats.html

pub mod gmsh;
pub mod vtk;
