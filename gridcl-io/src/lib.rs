//! Gmsh MSH 4.1 reader and writer, VTU/PVTU and VTKHDF output.
//!
//! - [`gmsh`]: the host side of Gmsh I/O (M4, ADR-0022): the file model [`gmsh::MshFile`], its
//!   parser and writer, the element table of A.7, and the conversion between a file and the
//!   host mesh description [`gmsh::GmshMesh`]. `gridcl::gmsh` builds and downloads meshes.
//! - [`vtk`]: the host side of VTU output (M5, ADR-0024): the file model [`vtk::VtuFile`], its
//!   writer and parser, the cell-type table of A.8 and the conversion from the host mesh
//!   description [`vtk::VtkMesh`]. `gridcl::vtk` downloads meshes.
//!
//! Milestone M8 adds parallel I/O and VTKHDF.

pub mod gmsh;
pub mod vtk;
