// Design: "Summary", "Architecture", "API walkthrough", "Cargo features"; the user book's
// chapters "Introduction" and "A first program" (ADR-0067).
//! Unstructured meshes on GPUs and CPUs, through CubeCL, for finite and boundary element codes.
//!
//! tetraloma keeps a mesh in the memory of a compute device (a GPU, or the CPU through CubeCL's
//! CPU runtime) and computes its geometry there: physical points, Jacobians, their determinants
//! and pseudo-inverses, normals and measures at the points of a quadrature rule. It reads and
//! writes Gmsh files, writes VTK files, and distributes meshes over the ranks of an MPI program.
//!
//! The [user book] explains the library from the start: [Installation], [A first program], the
//! mathematical concepts and the element definitions, tutorials, and the guides to [compute
//! devices] and [Cargo features]. This page is the entry to the API documentation.
//!
//! # Example
//!
//! Build the unit square from two triangles, compute the Jacobian determinants at the points of
//! a quadrature rule, and check the area:
//!
//! ```
//! use tetraloma::prelude::*;
//!
//! // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
//! # let client = tetraloma_testing::test_client();
//! // Node coordinates [node][2] in f64, and three node indices per triangle.
//! let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
//! let p1 = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
//! let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
//!     .nodes(&coords, None)
//!     .cell_block(p1, &[0, 1, 2, 3, 2, 1], None)
//!     .build()?;
//!
//! // det J at the points of a Gauss rule of degree 2, computed on the device. Each triangle has
//! // the area of the reference triangle, so det J = 1. The copy to the host is for inspection.
//! let rule = QuadRule::for_grid(&mesh, 2)?;
//! let detj = mesh.detj(&rule)?;
//! assert!(detj.blocks[0].to_host_cell_major()?.iter().all(|d| (d - 1.0).abs() < 1e-6));
//! assert!((mesh.measure(2)? - 1.0).abs() < 1e-6);
//! # Ok::<(), GridError>(())
//! ```
//!
//! # Meshes
//!
//! A [`Mesh<F, TD, GD, C>`](Mesh) has a scalar type `F` (`f32` or `f64`), a topological
//! dimension `TD` and a geometric dimension `GD` (the markers [`D1`], [`D2`] and [`D3`]), and a
//! communicator `C`, [`Serial`] by default. The aliases [`CurveMesh`], [`PlanarMesh`],
//! [`SurfaceMesh`] and [`VolumeMesh`] name the common cases. Methods exist only where they make
//! sense: inverse Jacobians and facet normals for a mesh that fills its space, unit normals for
//! a mesh one dimension lower than its space, and `boundary` wherever the boundary is a mesh.
//! Cell types and the geometry's polynomial degree are runtime values, stored per cell block.
//! [Typed dimensions] explains the design.
//!
//! - [`MeshBuilder`] builds a mesh from host arrays ([Building a mesh]).
//! - [`gmsh::read`] reads a Gmsh file into an [`AnyMesh`], whose dimensions the file decides;
//!   `try_into` fixes them, and [`with_mesh!`] runs generic code on any of them. [`gmsh`] also
//!   writes Gmsh files, and [`VtkWriter`] writes VTU, PVTU and VTKHDF files ([Reading and
//!   writing files]).
//! - [`QuadRule`] and [`FacetQuadRule`] hold quadrature points; the geometric data comes back as
//!   [`BlockFields`], one field per cell block, on the device ([Geometric data]).
//! - [`TagRegistry`], [`MeshTags`] and [`Codim`] select tagged cells and facets for submeshes
//!   ([Tags and submeshes]).
//! - [`DeviceGrid`] is the view of a mesh that kernels of your own read ([Writing kernels]).
//!
//! # Distributed meshes
//!
//! [`MeshBuilder::build_on`] builds a mesh on any communicator: `Mpi` with the feature `mpi`.
//! It partitions the cells ([`Partitioner`], by default [`Block`]), adds ghost cells
//! ([`GhostMode`]) and orders the cells ([`CellOrdering`]) ([Distributed meshes]). Index maps
//! ([`IndexMap`]) number the owned and ghost entries of each rank, and a [`Scatterer`] exchanges
//! values between ranks ([Halo exchange]). [`local_gpu_index`] gives each rank of a node its own
//! GPU ([One GPU per MPI rank]).
//!
//! # Cargo features
//!
//! `mpi` adds the MPI communicator `Mpi`; `hdf5` adds VTKHDF output; `metis`, `scotch`,
//! `ptscotch` and `parmetis` add the partitioners `Metis`, `Scotch`, `PtScotch` and `ParMetis`
//! (`metis` and `parmetis` exclude each other); `gpu-aware-cuda` is reserved and enables
//! nothing. The devices come from the runtime features of the `cubecl` dependency of your
//! program, such as `cpu`, `cuda`, `vulkan` or `metal`. [Cargo features] lists the system
//! libraries each needs.
//!
//! [user book]: https://tetracl.github.io/tetraloma/introduction.html
//! [Installation]: https://tetracl.github.io/tetraloma/getting-started/installation.html
//! [A first program]: https://tetracl.github.io/tetraloma/getting-started/first-program.html
//! [compute devices]: https://tetracl.github.io/tetraloma/devices.html
//! [Cargo features]: https://tetracl.github.io/tetraloma/features.html
//! [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html
//! [Building a mesh]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html
//! [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html
//! [Geometric data]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html
//! [Tags and submeshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html
//! [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html
//! [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html
//! [Halo exchange]: https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html
//! [One GPU per MPI rank]: https://tetracl.github.io/tetraloma/devices.html#one-gpu-per-mpi-rank

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

pub use tetraloma_device::{
    ConnKey, DeviceBuffer, DeviceCsr, DeviceStrided, GridError, Real, Result, VECTOR_PAD,
};
pub use tetraloma_dims::{
    Cell1, Cell2, Cell3, CellType, Codim0, Codim1, D0, D1, D2, D3, Dim, EmbeddedIn, TopoDim,
};
pub use tetraloma_geometry::{
    BlockFields, CoordinateElement, DeviceGrid, EVector, FacetPointSet, FacetQuadRule, FieldShape,
    GeomLayout, GeometryBlock, JacShape, NodeFamily, OrientationRule, PinvShape, PointSet,
    PullBack, PullBackOptions, QuadChunk, QuadField, QuadRule, ScalarShape, VecShape, detj_block,
    facet_normals_block, jacobians_block, measure_block, normals_block, physical_points_block,
    pseudo_inverse_block, pull_back_block,
};
#[cfg(feature = "mpi")]
pub use tetraloma_parallel::Mpi;
pub use tetraloma_parallel::{
    Comm, CommDyn, CommExt, CommScalar, DType, IndexMap, Neighbourhood, ReverseOp, ScatterNumeric,
    Scatterer, Serial, Transport, local_gpu_index,
};
#[cfg(feature = "metis")]
pub use tetraloma_partition::Metis;
#[cfg(feature = "parmetis")]
pub use tetraloma_partition::ParMetis;
#[cfg(feature = "ptscotch")]
pub use tetraloma_partition::PtScotch;
#[cfg(feature = "scotch")]
pub use tetraloma_partition::Scotch;
pub use tetraloma_partition::{Block, DistGraph, Partitioner};
pub use tetraloma_topology::{CellBlock, Connectivity, EntityBlock};
