// Design: "Architecture", the facade crate.
//! The names most programs need, in one import: `use tetraloma::prelude::*;`.
//!
//! The prelude brings the mesh types and their builder, the Gmsh and VTK modules, the dimension
//! markers and cell types, coordinate elements, quadrature rules, communicators, index maps and
//! partitioners, and the error type into scope. It also imports the traits whose methods you call
//! on them, such as [`DeviceGrid`] for the arrays of a mesh and [`CommExt`] for collectives. With
//! the features `mpi`, `metis`, `scotch`, `ptscotch` and `parmetis` it also brings `Mpi`, `Metis`,
//! `Scotch`, `PtScotch` and `ParMetis`. The program of [A first program] starts with this import,
//! as do the examples of this documentation.
//!
//! [A first program]: https://tetracl.github.io/tetraloma/getting-started/first-program.html
//!
//! # Example
//!
//! ```
//! use tetraloma::prelude::*;
//!
//! // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
//! # let client = tetraloma_testing::test_client();
//! let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
//! let mesh: Mesh<f32, D1, D1> = MeshBuilder::new(&client)
//!     .nodes(&[0.0, 1.0], None)
//!     .cell_block(seg, &[0, 1], None)
//!     .build()?;
//! // `cell_blocks` is a method of the trait `DeviceGrid`, which the prelude imports.
//! assert_eq!(mesh.cell_blocks().len(), 1);
//! # Ok::<(), GridError>(())
//! ```

#[cfg(feature = "metis")]
pub use crate::Metis;
#[cfg(feature = "mpi")]
pub use crate::Mpi;
#[cfg(feature = "parmetis")]
pub use crate::ParMetis;
#[cfg(feature = "ptscotch")]
pub use crate::PtScotch;
#[cfg(feature = "scotch")]
pub use crate::Scotch;
pub use crate::{
    AnyMesh, AnyMeshRef, Block, BlockFields, Cell1, Cell2, Cell3, CellBlock, CellOrdering,
    CellType, Codim, Codim0, Codim1, Comm, CommDyn, CommExt, CommScalar, ConnKey, Connectivity,
    CoordinateElement, CurveMesh, D0, D1, D2, D3, DType, DeviceBuffer, DeviceGrid, Dim, DistGraph,
    EmbeddedIn, EntityBlock, EntityMap, FacetQuadRule, GhostMode, GmshOptions, GridError, IndexMap,
    Mesh, MeshBuilder, MeshTags, Neighbourhood, NodeFamily, OrientationRule, Partitioner,
    PlanarMesh, PullBack, PullBackOptions, QuadField, QuadRule, Real, Result, ReverseOp,
    ScatterNumeric, Scatterer, Serial, SurfaceMesh, TagRegistry, TopoDim, Transport, VolumeMesh,
    VtkWriter, gmsh, local_gpu_index, vtk, with_mesh,
};
