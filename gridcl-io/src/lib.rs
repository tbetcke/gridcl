//! Gmsh MSH 4.1 reader and writer, VTU/PVTU and VTKHDF output.
//!
//! - [`gmsh`]: the host side of Gmsh I/O (M4, ADR-0022): the file model [`gmsh::MshFile`], its
//!   parser and writer, the element table of A.7, and the conversion between a file and the
//!   host mesh description [`gmsh::GmshMesh`]. `gridcl::gmsh` builds and downloads meshes.
//!
//! Milestones M5 and M8 add VTK output and parallel I/O.

pub mod gmsh;
