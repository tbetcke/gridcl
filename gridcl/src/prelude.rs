//! The names most programs need: `use gridcl::prelude::*;`.

#[cfg(feature = "mpi")]
pub use crate::Mpi;
pub use crate::{
    AnyMesh, AnyMeshRef, Block, BlockFields, Cell1, Cell2, Cell3, CellBlock, CellOrdering,
    CellType, Codim, Codim0, Codim1, Comm, CommDyn, CommExt, CommScalar, ConnKey, Connectivity,
    CoordinateElement, CurveMesh, D0, D1, D2, D3, DType, DeviceBuffer, DeviceGrid, Dim, DistGraph,
    EmbeddedIn, EntityBlock, EntityMap, FacetQuadRule, GhostMode, GmshOptions, GridError, IndexMap,
    Mesh, MeshBuilder, MeshTags, Neighbourhood, NodeFamily, Partitioner, PlanarMesh, QuadField,
    QuadRule, Real, Result, ReverseOp, ScatterNumeric, Scatterer, Serial, SurfaceMesh, TagRegistry,
    TopoDim, Transport, VolumeMesh, VtkWriter, gmsh, vtk, with_mesh,
};
