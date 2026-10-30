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

mod any;
mod builder;
mod entity_map;
pub mod gmsh;
mod mesh;
mod pipeline;
pub mod prelude;
mod tags;
#[cfg(test)]
mod test_comm;
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
    GeomLayout, GeometryBlock, JacShape, NodeFamily, PinvShape, PointSet, QuadChunk, QuadField,
    QuadRule, ScalarShape, VecShape, detj_block, facet_normals_block, jacobians_block,
    measure_block, normals_block, physical_points_block, pseudo_inverse_block,
};
#[cfg(feature = "mpi")]
pub use gridcl_parallel::Mpi;
pub use gridcl_parallel::{
    Comm, CommDyn, CommExt, CommScalar, DType, IndexMap, Neighbourhood, ReverseOp, ScatterNumeric,
    Scatterer, Serial, Transport,
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
