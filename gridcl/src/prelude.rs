//! The names most programs need: `use gridcl::prelude::*;`.

pub use crate::{
    AnyMesh, AnyMeshRef, BlockFields, Cell1, Cell2, Cell3, CellBlock, CellType, Codim, Codim0,
    Codim1, Comm, CommExt, ConnKey, Connectivity, CoordinateElement, CurveMesh, D0, D1, D2, D3,
    DeviceBuffer, DeviceGrid, Dim, EmbeddedIn, EntityBlock, EntityMap, FacetQuadRule, GmshOptions,
    GridError, IndexMap, Mesh, MeshBuilder, MeshTags, NodeFamily, PlanarMesh, QuadField, QuadRule,
    Real, Result, Serial, SurfaceMesh, TagRegistry, TopoDim, VolumeMesh, gmsh, with_mesh,
};
