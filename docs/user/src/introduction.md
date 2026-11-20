# Introduction

tetraloma is a Rust library for unstructured meshes whose data lives on a compute device: a GPU,
or the CPU through the CubeCL CPU runtime. It is meant as the foundation for finite element and
boundary element codes. This book explains what a tetraloma mesh is, how to build, read and write
one, how to compute geometric data on it, and how to run it on one or many devices.

## What tetraloma is

A mesh divides a domain, such as a solid body or its surface, into simple cells: intervals,
triangles, quadrilaterals, tetrahedra, hexahedra, prisms and pyramids. Numerical methods for
partial differential equations integrate over these cells. For that they need the cells'
geometry at many points: positions, Jacobian matrices, their determinants and inverses, and
normal vectors. tetraloma stores the mesh in device memory and computes these quantities there,
with kernels written once in [CubeCL](https://github.com/tracel-ai/cubecl) and compiled for
each backend.

A mesh has two dimensions. Its topological dimension is the dimension of its cells: 2 for
triangles. Its geometric dimension is the dimension of the space it lives in: 3 for a triangulated
surface in space. tetraloma puts both into the type of a mesh, `Mesh<F, TD, GD, C>`, together
with the scalar type `F` (`f32` or `f64`) and the communicator `C`. Each mesh type has only the
methods that make sense for it. A surface in space has unit normals, so `normals` exists for it.
A volume mesh has no unit normal field, so calling `normals` on it does not compile. The cell
types and the polynomial degree of the geometry, in contrast, are values that the program reads
at run time, for example from a file.

The example below builds the unit square from two triangles on the CPU runtime and checks its
area. The chapter [A first program](getting-started/first-program.md) explains each line.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

// A CubeCL client of the CPU runtime.
let client = cubecl::Device::cpu()?.client();
// The unit square as two triangles with straight sides.
let p1 = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], None)
    .cell_block(p1, &[0, 1, 2, 3, 2, 1], None)
    .build()?;
assert!((mesh.measure(1)? - 1.0).abs() < 1e-12);
# Ok(())
# }
```

## Features

- **Typed dimensions.** A mesh type fixes its topological and geometric dimensions, which form
  one of six pairs: curves in 1D, 2D and 3D, planar meshes, surfaces in 3D, and volume meshes.
  Methods such as `normals`, `inverse_jacobians` and `boundary` exist only where they make
  sense. [`AnyMesh`](api/tetraloma/enum.AnyMesh.html) holds a mesh whose dimensions a file
  decides ([Typed dimensions](concepts/typed-dimensions.md)).
- **All standard cells.** Intervals, triangles, quadrilaterals, tetrahedra, hexahedra, prisms and
  pyramids, and hybrid meshes that mix them. Each cell type is stored in a block of its own.
- **High-order geometry.** Cells are Lagrange elements of any polynomial degree, with equispaced or
  Gauss–Lobatto–Legendre nodes. The device kernels are verified and tuned up to degree 8, and up to
  degree 3 for pyramids ([Lagrange geometry](concepts/geometry.md)).
- **Geometric data on the device.** Physical points, Jacobians, determinants, pseudo-inverses,
  inverse Jacobians, normals, facet normals and measures at the points of a quadrature rule, and
  the pull-back of physical points to reference coordinates.
- **Topology on the device.** Edges and faces, connectivity between cells and entities, and the
  orientation bits of every cell's edges and faces.
- **Tags and submeshes.** Physical groups from Gmsh files, submeshes of cells, facets and edges,
  boundary meshes, and a consistent orientation of manifold meshes.
- **Many devices.** The CubeCL CPU runtime, CUDA, HIP, Vulkan, Metal and wgpu, chosen at run
  time, in `f32` and `f64`.
- **Distributed meshes.** Meshes spread over the ranks of an MPI program, with partitioners
  (a built-in block partitioner, METIS, Scotch, PT-Scotch and ParMETIS), ghost cells and halo
  exchange ([Distributed meshes](tutorials/distributed.md)).
- **Files.** Gmsh MSH 4.1 reading and writing; VTK output as VTU, PVTU and VTKHDF, with Lagrange
  cells for curved geometry.
- **Reproducible results.** Every kernel gives bitwise identical results from run to run on the
  same device. Kernels that treat cells or points independently give the same bits on the CPU
  runtime at every SIMD vector width ([Determinism](concepts/precision.md#determinism)).

## Status

tetraloma is under active development and has not been released yet. The public API may still
change between commits. The crates will be published on crates.io with the first release; until
then you depend on the Git repository, as [Installation](getting-started/installation.md)
describes.

This book and the API documentation describe the `main` branch of the repository. Versioned
copies of the documentation will start with the first release. Every change is tested on the CPU
runtime and on Metal. Continuous integration on Linux also tests MPICH, Open MPI and a software
Vulkan driver. NVIDIA GPUs are tested through CUDA after every change to a kernel.

## How this documentation is organised

The book has six parts, and the API documentation sits beside it.

- **Getting started.** [Installation](getting-started/installation.md) and
  [A first program](getting-started/first-program.md): what you need on your machine and a
  program that builds a mesh, computes its geometry and writes a file.
- **Mathematical concepts.** What meshes, reference cells, Lagrange geometry, Jacobians,
  orientation, quadrature and the pull-back are, starting from [Meshes and
  topology](concepts/meshes.md). These chapters assume linear algebra and calculus, but no
  knowledge of finite elements.
- **Element definitions.** The exact numbering of vertices, edges, faces and nodes of every
  cell, starting from [Reference cell definitions](elements/reference-cells.md), and how they map
  to Gmsh, VTK, Basix and DOLFINx.
- **Tutorials.** Task-oriented chapters with complete programs, from [Building a
  mesh](tutorials/build-mesh.md) to [Writing kernels](tutorials/kernels.md) of your own.
- **Devices and features.** [Compute devices](devices.md) describes the backends and how to choose
  one; [Cargo features](features.md) lists the optional features and the system libraries they
  need.
- **Help.** [Troubleshooting](troubleshooting.md) and a [Glossary](glossary.md).

The [API documentation](api/tetraloma/index.html) describes every public type and function, with
an example for each function. Start at the crate `tetraloma`, which re-exports everything a
program needs; `use tetraloma::prelude::*;` brings the common names into scope.

Every Rust example in this book is compiled and run as a test. The longer programs live in the
crate `tetraloma-examples` of the repository, and you can run each of them with
`cargo run --release -p tetraloma-examples --example <name>`.
