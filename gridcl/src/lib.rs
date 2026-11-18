//! gridcl: distributed unstructured meshes on GPUs and CPUs through CubeCL.
//!
//! The facade crate: [`Mesh`], [`MeshBuilder`], [`AnyMesh`], [`AnyMeshRef`], [`with_mesh!`],
//! [`DeviceGrid`] (re-exported), tags ([`TagRegistry`], [`MeshTags`], [`Codim`]), Gmsh I/O
//! ([`gmsh`]), VTK output ([`vtk`]), and the [`prelude`] (docs/design.md, "Architecture").
//!
//! A mesh is typed by its scalar `F`, its topological and geometric dimensions `TD` and `GD`,
//! and its communicator `C` (default [`Serial`]). Methods exist only where they make sense:
//! inverse Jacobians and facet normals in codimension 0, normals in codimension 1.
//!
//! Distributed meshes come from [`MeshBuilder::build_on`] on any communicator (`Mpi` with
//! the feature `mpi`), which partitions the cells ([`Partitioner`], default [`Block`]), adds
//! ghost cells ([`GhostMode`]) and orders them ([`CellOrdering`]). Their index maps exchange
//! values between ranks through a [`Scatterer`] (docs/design.md, "Distributed mesh model").
//! The external partitioners `Metis`, `Scotch`, `PtScotch` and `ParMetis` are behind the
//! features of the same names in lower case ([`MeshBuilder::partitioner`]).
//!
//! # Examples and walkthrough
//!
//! The API walkthrough of docs/design.md ("API walkthrough") follows one program from a Gmsh
//! file to output: reading the file and fixing the dimensions once (Listing B8), a volume mesh
//! to a boundary surface mesh (Listing B9), and the halo exchange and output (Listing B13).
//! The crate gridcl-examples holds them as programs, each runnable with
//! `cargo run --release -p gridcl-examples --example <name>`:
//!
//! - `read_gmsh`: reads a Gmsh file and fixes the dimensions once (Listing B8);
//! - `surface_submesh`: a boundary surface mesh with its normals, Jacobians and area
//!   (Listing B9);
//! - `geometric_factors`: geometric factors at p = 1, 4 and 8 on a curved sphere;
//! - `vtk_output`: VTU output of a mesh and a point field (Listing B13);
//! - `halo_exchange`: `forward` and `reverse` on a mesh's node index map under MPI
//!   (Listing B13; feature `mpi`, run with `mpirun`).
//!
//! `scripts/run-examples.sh` builds and runs them all.

mod any;
mod builder;
mod entity_map;
pub mod gmsh;
mod io_download;
mod mesh;
mod orient;
mod pipeline;
pub mod prelude;
mod pull_back;
mod submesh;
mod tags;
pub mod vtk;

pub use any::{AnyMesh, AnyMeshRef};
pub use builder::{CellOrdering, GhostMode, MeshBuilder};
pub use entity_map::EntityMap;
pub use gmsh::GmshOptions;
pub use mesh::{CurveMesh, Mesh, PlanarMesh, SurfaceMesh, VolumeMesh};
pub use tags::{Codim, MeshTags, TagRegistry};
pub use vtk::VtkWriter;

pub use gridcl_device::{
    ConnKey, DeviceBuffer, DeviceCsr, DeviceStrided, GridError, Real, Result, VECTOR_PAD,
};
pub use gridcl_dims::{
    Cell1, Cell2, Cell3, CellType, Codim0, Codim1, D0, D1, D2, D3, Dim, EmbeddedIn, TopoDim,
};
pub use gridcl_geometry::{
    BlockFields, CoordinateElement, DeviceGrid, EVector, FacetPointSet, FacetQuadRule, FieldShape,
    GeomLayout, GeometryBlock, JacShape, NodeFamily, OrientationRule, PinvShape, PointSet,
    PullBack, PullBackOptions, QuadChunk, QuadField, QuadRule, ScalarShape, VecShape, detj_block,
    facet_normals_block, jacobians_block, measure_block, normals_block, physical_points_block,
    pseudo_inverse_block, pull_back_block,
};
#[cfg(feature = "mpi")]
pub use gridcl_parallel::Mpi;
pub use gridcl_parallel::{
    Comm, CommDyn, CommExt, CommScalar, DType, IndexMap, Neighbourhood, ReverseOp, ScatterNumeric,
    Scatterer, Serial, Transport, local_gpu_index,
};
#[cfg(feature = "metis")]
pub use gridcl_partition::Metis;
#[cfg(feature = "parmetis")]
pub use gridcl_partition::ParMetis;
#[cfg(feature = "ptscotch")]
pub use gridcl_partition::PtScotch;
#[cfg(feature = "scotch")]
pub use gridcl_partition::Scotch;
pub use gridcl_partition::{Block, DistGraph, Partitioner};
pub use gridcl_topology::{CellBlock, Connectivity, EntityBlock};
