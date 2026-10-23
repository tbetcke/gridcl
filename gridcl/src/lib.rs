//! gridcl: distributed unstructured meshes on GPUs and CPUs through CubeCL.
//!
//! The facade crate: [`Mesh`], [`MeshBuilder`], [`AnyMesh`], [`AnyMeshRef`], [`with_mesh!`],
//! [`DeviceGrid`] (re-exported), and the [`prelude`] (docs/design.md, "Architecture").
//!
//! A mesh is typed by its scalar `F`, its topological and geometric dimensions `TD` and `GD`,
//! and its communicator `C` (default [`Serial`]). Methods exist only where they make sense:
//! inverse Jacobians and facet normals in codimension 0, normals in codimension 1.

mod any;
mod builder;
mod entity_map;
mod mesh;
pub mod prelude;

pub use any::{AnyMesh, AnyMeshRef};
pub use builder::MeshBuilder;
pub use entity_map::EntityMap;
pub use mesh::{CurveMesh, Mesh, PlanarMesh, SurfaceMesh, VolumeMesh};

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
pub use gridcl_parallel::{Comm, CommDyn, CommExt, IndexMap, Serial};
pub use gridcl_topology::{CellBlock, Connectivity, EntityBlock};
