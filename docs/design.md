# gridcl — Design B: typed dimensions, runtime cells (revised design)

As of 17 October 2026. Markdown export of the revised design document; this file is `docs/design.md`, the design reference for implementation.

## Summary

gridcl is a GPU-resident mesh library in Rust built on CubeCL 0.11. It serves as the foundation for strongly typed FEM and BEM crates. It encodes the topological and geometric dimensions in the type system as `Mesh<F, TD, GD, C>`, and keeps cell types and geometry degree as runtime values stored per block. Methods that make sense only for some embeddings (normals, inverse Jacobians, boundaries) exist only where their trait bounds hold, so misuse is a compile error. CubeCL comptime parameters then recover full kernel specialisation from the runtime values.

- **Scope:** 1D–3D meshes with D_TOP ≤ D_GEO ≤ 3, all standard cells including hybrid meshes, arbitrary geometry degree, and device residency on CUDA, HIP, Vulkan, Metal and the CubeCL CPU runtime. It also covers MPI distribution with partitioning, Gmsh MSH 4.1 import and export, and VTK export.
- **Audience:** library architects and the implementation agent (Claude Code). Convert this document to `docs/design.md`.
- **Status:** revised design, 17 October 2026. It supersedes the draft "gridcl Design B — Typed dimensions, runtime cells" of October 2026 and incorporates the consolidated design review of the same date. The last section maps every review finding to the place where it is fixed. Revision 2 of the same date adds the development environment: the development and test targets and milestone E0. Revision 3 adds SIMD vectorization of kernels through CubeCL vectors, with a benchmark milestone (M3s) that decides where vectors are used.
- **Name:** gridcl is a placeholder.

**Main changes from the draft:**

- the crate graph keeps serial builds free of MPI and lets kernels live where they are used;
- orientation is keyed on partition-independent origin vertex ids;
- the topology algorithm, ownership rules and pipeline order are corrected;
- f32 is a first-class precision and device limits are respected, so wgpu and Metal are real targets;
- prism and pyramid node layouts follow the DefElement conventions;
- coupe and the Python bindings are removed;
- every milestone has measurable acceptance tests.

**API caveat.** Code sketches follow the released CubeCL 0.11.0 API (crates.io, 6 October 2026), checked against its source but not compiled. That API has a runtime-erased `Client` obtained from `cubecl::Device`, slice kernel arguments, `Vector<F, N>` lanes whose size is given at launch, `BufferArg::from_raw_parts(handle, len)`, runtime scalars passed as plain values, and `#[comptime]` parameters. CubeCL is alpha and announces breaking changes between minor versions, so pin `cubecl = "=0.11.0"`.

## Requirements and decisions

The requirements are unchanged from the draft, apart from the development targets and CPU performance. Eleven design decisions are fixed, two topics are out of scope, and the review settled five open questions.

### Requirements

- **Cells:** interval, triangle, quadrilateral, tetrahedron, hexahedron, prism (wedge) and pyramid, including hybrid meshes that mix cell types of the same topological dimension.
- **Embedding:** topological dimension D_TOP embedded in geometric dimension D_GEO with D_TOP ≤ D_GEO ≤ 3. This covers curves in 2D and 3D, surfaces in 3D, and the usual D_TOP = D_GEO case.
- **Device residency:** all large mesh arrays live on the compute device. The host keeps only metadata and communication bookkeeping.
- **Backends:** CUDA, HIP, Vulkan and Metal (through wgpu), and the CubeCL CPU runtime. f64 is used where the device supports it; f32 must work everywhere.
- **CPU performance:** on the CubeCL CPU runtime, kernels use the host's SIMD units through CubeCL vectors where benchmarks show a gain. The same kernel source serves CPUs and GPUs.
- **Development and test targets:** a local Apple M3 Max (Claude Code runs here; CubeCL CPU runtime and Metal), UCL Locust (NVIDIA Grace Hopper GH200) as the CUDA platform for routine runs, milestone reviews and benchmarks, UCL Kathleen (Slurm, Xeon Gold 6248 Cascade Lake with AVX-512) for CPU-only runs and CPU benchmarks (CpuSimd512) and for MPI tests across two nodes, with Myriad's CPU-only D nodes as the fallback, and Myriad's V100 and A100 nodes for multi-GPU tests and architecture comparisons, and as the fallback when Locust is unavailable (ADR-0014, ADR-0016). See "Development environment and test targets".
- **Distribution:** MPI-distributed meshes with ownership, ghost layers and parallel partitioning. PT-Scotch is the production default; other partitioners are pluggable.
- **High order:** geometry of arbitrary polynomial degree, with sub-entity orientation information for later high-order FEM and BEM.
- **I/O:** Gmsh MSH 4.1 import and export, and VTK export (VTU/PVTU, VTKHDF) including VTK Lagrange cells.
- **Generic:** no assumption about the discretisation built on top (FEM, BEM, DG, FV).

### Fixed decisions

1. Design B: compile-time (D_TOP, D_GEO), runtime cell types and degrees. Designs A and C appear only for comparison.
2. CubeCL is pinned to `=0.11.0`.
3. Reference-cell conventions follow ndelement main at commit f8cbffe (codeberg.org/nd-project/nd, mirrored at github.com/bempp/nd). Those conventions are the DefElement conventions. ndelement 0.4.0 on crates.io still uses the older Basix simplex order.
4. gridcl-refcell holds gridcl's own tables. ndelement is used only in the unpublished gridcl-conformance crate.
5. Edge and face vertex lists are stored in canonical orientation as well as per-cell orientation bits, a deliberate difference from DOLFINx.
6. High-order pyramids go to VTK only as linear or subdivided cells.
7. Canonical orientation is keyed on origin vertex ids (decided in review, Q1).
8. wgpu and Metal are real targets (Q2).
9. The CI default partitioner is the built-in block partitioner plus vendored METIS on rank 0; PT-Scotch runs in its own CI job (Q3).
10. Development targets: the M3 Max runs Claude Code and the CPU-runtime and Metal tests; Locust (GH200) is the CUDA platform for routine runs, milestone reviews and benchmarks; Kathleen runs CPU-only runs, the x86-64 AVX-512 CPU benchmarks and MPI tests across two nodes, with Myriad's CPU-only D nodes as the fallback; Myriad V100 and A100 run multi-GPU tests and architecture comparisons, and are the fallback when Locust is unavailable (ADR-0014, ADR-0016).
11. SIMD is written in CubeCL kernels with `Vector<F, N>`, never with intrinsics or `std::simd` on the host. Vector sizes per kernel family and device class come from benchmarks (M3s); until then every kernel runs at vector size 1.

### Out of scope

- **Python bindings.** `AnyMesh` exists for file I/O only.
- **coupe** is not used.

### Decisions taken in review

| Question | Decision | Where it applies |
| --- | --- | --- |
| Q1: which vertex ids define canonical orientation? | Origin vertex ids: Gmsh node tags, builder ids, or ids inherited from a parent mesh | Reference cells and conventions; topology construction; A.6 |
| Q2: are wgpu and Metal real targets? | Yes | Device model and backends; testing; workspace and CI |
| Q3: CI default partitioner | Built-in block partitioner, plus vendored METIS on rank 0 | Partitioning; workspace and CI |
| Q4: prism and pyramid node layouts | Defined by gridcl from the DefElement paper's conventions (Scroggs et al., [arXiv:2506.20188](https://arxiv.org/html/2506.20188)) | Reference cells and conventions; A.5 |
| Q5: orientation scope | Manifold surfaces only. Non-manifold surfaces and interior-interface submeshes are not supported. | Geometry; tags, regions and submeshes |

## Design rationale

Design B types the six valid (D_TOP, D_GEO) pairs and leaves everything else to runtime values. This catches the dimension and codimension errors that matter for FEM and BEM, at small cost. CubeCL compiles kernels just in time and accepts host values as comptime constants, so a kernel launched with cell type, degree and dimensions as comptime parameters is specialised as if they were Rust generics. Moving more information into the host type system therefore buys type safety and ergonomics, but almost no device performance.

### Position on the design spectrum

| Aspect | A: runtime-driven | B: typed dimensions (this design) | C: compile-time first |
| --- | --- | --- | --- |
| Topological dimension | runtime u8 | type marker D1/D2/D3 | derived from cell type |
| Geometric dimension | runtime u8 | type marker | type marker |
| Cell type | runtime enum per block | runtime enum, restricted per D_TOP | zero-sized type; tuples for mixed meshes |
| Geometry degree | runtime u32 | runtime u32 | `Const<P>` or `Dyn` |
| Serial vs distributed | runtime enum | type parameter `C: Comm` | typestate |
| Dimension errors caught | at run time | at compile time | at compile time |
| Mixed-cell meshes | native | native | type-level lists and visitors |
| Host monomorphisations | about 2 | about 24 (6 pairs × 2 scalars × 2 communicators) | 60+ |
| Closest relatives | DOLFINx, MFEM, NGSolve | deal.II, Dune, ndmesh | nalgebra Const/Dyn pattern, static Dune geometries |

### Lessons from existing libraries

| Library | Adopt | Avoid |
| --- | --- | --- |
| FEniCSx / DOLFINx | Topology/geometry split; IndexMap with owned and ghost split; per-cell-type blocks; parallel creation pipeline with graph reordering before numbering; cell permutation info; MeshTags; EntityMap for submeshes | Host-only containers; connectivity objects that cannot live on a device |
| MFEM | Explicit host/device validity; geometric factors precomputed at quadrature points; element-wise (E-vector) layouts; GPU-aware MPI switch | Per-element heap objects and virtual dispatch per element |
| NGSolve / Netgen | Codimension-based addressing (VOL, BND, BBND, BBBND); regions by name; raising geometry degree after loading | Tight coupling to one mesh generator |
| Firedrake | Geometry as a vector-valued field; explicit manifold cell orientation; locality-improving renumbering | Dependence on a CPU-resident external mesh manager |
| nd (ndelement, ndmesh) | DefElement reference-cell numbering, orientation bits and Lagrange layout, so the FEM/BEM layer can use ndelement directly; a small trait vocabulary for downstream crates | Per-entity host iteration as the main access path on GPUs |
| deal.II (matrix-free) | Vectorization over cells: batches of cells in SIMD lanes, with data tiled by cells | Hand-written SIMD wrappers per instruction set; CubeCL's code generation replaces them |

Design B therefore takes the data model of DOLFINx and the device-memory discipline of MFEM. It adds the compile-time (dim, spacedim) pair of deal.II (`Triangulation<dim, spacedim>`) and Dune (dim, dimworld). From ndmesh it takes the idea of a small trait vocabulary that downstream crates program against.

ndelement at f8cbffe has no Lagrange elements on prisms or pyramids. For those cells gridcl defines the layout itself, from the same DefElement conventions.

## Architecture

The pair (D_TOP, D_GEO) is typed; cell type and geometry degree are runtime values stored per block. The pair determines the shape of almost everything a FEM or BEM code computes from a mesh:

- Jacobians are D_GEO × D_TOP;
- normals exist only in codimension 1, and inverse Jacobians only in codimension 0;
- the boundary of a mesh has dimension D_TOP − 1.

There are only six valid pairs. The type system also restricts which cell types may appear: a D2 mesh holds only triangles and quadrilaterals.

### Dimension markers

**Listing B1 — type-level dimensions (gridcl-dims, no dependencies)**

```rust
use core::{fmt::Debug, hash::Hash};

mod sealed { pub trait Sealed {} }

pub trait Dim: sealed::Sealed + Copy + Default + Debug + Eq + Hash + Send + Sync + 'static {
    const N: usize;
    /// One dimension lower. D0::Lower = D0; D0 is not a TopoDim, so it never has a boundary.
    type Lower: Dim;
    /// [T; N] on stable Rust, without generic_const_exprs.
    type Array<T: Copy + Default>: Copy + Default + AsRef<[T]> + AsMut<[T]>;
}

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)] pub struct D0;
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)] pub struct D1;
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)] pub struct D2;
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)] pub struct D3;

impl sealed::Sealed for D0 {}
impl sealed::Sealed for D1 {}
impl sealed::Sealed for D2 {}
impl sealed::Sealed for D3 {}

impl Dim for D0 { const N: usize = 0; type Lower = D0; type Array<T: Copy + Default> = [T; 0]; }
impl Dim for D1 { const N: usize = 1; type Lower = D0; type Array<T: Copy + Default> = [T; 1]; }
impl Dim for D2 { const N: usize = 2; type Lower = D1; type Array<T: Copy + Default> = [T; 2]; }
impl Dim for D3 { const N: usize = 3; type Lower = D2; type Array<T: Copy + Default> = [T; 3]; }

/// TD can be embedded in GD (TD <= GD). Implemented for exactly six pairs.
pub trait EmbeddedIn<GD: Dim>: Dim {
    /// GD x TD Jacobian on the host: TD columns of GD entries (column-major).
    type Jacobian<T: Copy + Default>: Copy + Default;
    /// TD x GD pseudo-inverse K = (J^T J)^-1 J^T: GD columns of TD entries.
    type PseudoInverse<T: Copy + Default>: Copy + Default;
}
impl EmbeddedIn<D1> for D1 { type Jacobian<T: Copy + Default> = [[T; 1]; 1]; type PseudoInverse<T: Copy + Default> = [[T; 1]; 1]; }
impl EmbeddedIn<D2> for D1 { type Jacobian<T: Copy + Default> = [[T; 2]; 1]; type PseudoInverse<T: Copy + Default> = [[T; 1]; 2]; }
impl EmbeddedIn<D3> for D1 { type Jacobian<T: Copy + Default> = [[T; 3]; 1]; type PseudoInverse<T: Copy + Default> = [[T; 1]; 3]; }
impl EmbeddedIn<D2> for D2 { type Jacobian<T: Copy + Default> = [[T; 2]; 2]; type PseudoInverse<T: Copy + Default> = [[T; 2]; 2]; }
impl EmbeddedIn<D3> for D2 { type Jacobian<T: Copy + Default> = [[T; 3]; 2]; type PseudoInverse<T: Copy + Default> = [[T; 2]; 3]; }
impl EmbeddedIn<D3> for D3 { type Jacobian<T: Copy + Default> = [[T; 3]; 3]; type PseudoInverse<T: Copy + Default> = [[T; 3]; 3]; }

pub trait Codim0<GD: Dim>: EmbeddedIn<GD> {}
impl Codim0<D1> for D1 {}
impl Codim0<D2> for D2 {}
impl Codim0<D3> for D3 {}

pub trait Codim1<GD: Dim>: EmbeddedIn<GD> {}
impl Codim1<D2> for D1 {}
impl Codim1<D3> for D2 {}
```

Marker types are used instead of `const N: usize` generics, because stable Rust cannot express N − 1 in a return type, or `[T; N]` from an associated constant in generic code. `Dim` is sealed. The orphan rule then also stops downstream crates from adding `EmbeddedIn` or `Codim` impls for the markers.

### Cell families per topological dimension

**Listing B2 — admissible cells are part of the type (gridcl-dims)**

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CellType { Point, Interval, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum Cell1 { Interval }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum Cell2 { Triangle, Quadrilateral }
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)] pub enum Cell3 { Tetrahedron, Hexahedron, Prism, Pyramid }

/// A cell type used where a different topological dimension is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrongCellDim { pub found: CellType, pub expected_tdim: usize }

pub trait TopoDim: Dim {
    type Cell: Copy + Eq + Hash + Debug + Send + Sync + 'static
        + Into<CellType> + TryFrom<CellType, Error = WrongCellDim>;
}
impl TopoDim for D1 { type Cell = Cell1; }
impl TopoDim for D2 { type Cell = Cell2; }
impl TopoDim for D3 { type Cell = Cell3; }
// From<CellN> for CellType and TryFrom<CellType> for CellN are generated by one macro.
```

`D0` is not a `TopoDim`: point meshes do not exist, so `boundary()` is unavailable on curve meshes. `CellType` lives in gridcl-dims, so gridcl-dims stays dependency-free and gridcl-refcell builds on it.

### Mesh type

**Listing B3 — generic over scalar, dimensions and communicator, not over the backend (gridcl)**

```rust
/// Scalar of mesh geometry (gridcl-device, re-exported by gridcl). Host arithmetic uses widen/narrow (or f64 directly), never method
/// calls on Self: CubeCL's Float and num_traits::Float share method names (sqrt, abs, min, ...),
/// and CubeCL's versions are device-only stubs that panic on the host.
pub trait Real: cubecl::prelude::Float + cubecl::prelude::CubeElement {
    fn widen(self) -> f64;
    fn narrow(x: f64) -> Self;
}
impl Real for f32 { fn widen(self) -> f64 { self as f64 } fn narrow(x: f64) -> Self { x as f32 } }
impl Real for f64 { fn widen(self) -> f64 { self } fn narrow(x: f64) -> Self { x } }

/// The struct carries only the bounds its fields need; struct bounds are not implied on stable
/// Rust. Constructors and methods require TD: EmbeddedIn<GD>. Mesh is not Clone (see try_clone).
pub struct Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial> {
    client: Client,                 // runtime-erased CubeCL client
    comm: C,                        // Listing S2c
    topology: Topology<TD>,         // cell and entity blocks, connectivity, vertex IndexMap, origin ids
    geometry: Geometry<F, TD, GD>,  // geometry blocks, nodes, node IndexMap, cached factors
    tags: TagRegistry,              // from M4, its first user (ADR-0012)
}

pub type CurveMesh<F, C = Serial>   = Mesh<F, D1, D3, C>;
pub type PlanarMesh<F, C = Serial>  = Mesh<F, D2, D2, C>;
pub type SurfaceMesh<F, C = Serial> = Mesh<F, D2, D3, C>;
pub type VolumeMesh<F, C = Serial>  = Mesh<F, D3, D3, C>;

pub struct CellBlock<TD: TopoDim> {
    pub cell: TD::Cell,                 // e.g. Cell3::Prism
    pub index_map: IndexMap,            // [owned | ghost] cells of this block
    pub vertices: DeviceStrided<u32>,   // local vertex indices in reference-cell order
    pub origin_ids: DeviceBuffer<u32>,  // interleaved (hi ^ 0x8000_0000, lo) origin id of each cell
    pub flags: DeviceBuffer<u32>,       // bit 0: manifold orientation flipped (codimension 1)
}
impl<TD: TopoDim> CellBlock<TD> {      // CellBlock lives in gridcl-topology (ADR-0012)
    pub fn new(cell: TD::Cell, index_map: IndexMap, vertices: DeviceStrided<u32>,
               origin_ids: DeviceBuffer<u32>, flags: DeviceBuffer<u32>) -> Self;
}

// CoordinateElement and GeometryBlock live in gridcl-geometry, re-exported by gridcl (ADR-0012).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoordinateElement<TD: TopoDim> { pub cell: TD::Cell, pub degree: u32, pub family: NodeFamily }

pub struct GeometryBlock<TD: TopoDim> {
    pub element: CoordinateElement<TD>,
    pub dofmap: DeviceStrided<u32>,     // cells x nodes-per-cell, into the node array
}
```

The communicator parameter defaults to `Serial`, so tests and single-GPU users never see MPI types. `Serial` implements `Comm` with no-op collectives, and its IndexMaps have no ghosts. `Mesh<…, Mpi>` is deliberately not `Send`: gridcl runs one host thread per rank (see "Streams and threading").

### Methods that exist only where they make sense

**Listing B4 — capability by trait bound (gridcl)**

```rust
impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim, C: Comm,
{
    pub const TDIM: usize = TD::N;
    pub const GDIM: usize = GD::N;
    pub fn tdim(&self) -> usize { TD::N }
    pub fn gdim(&self) -> usize { GD::N }
    pub fn client(&self) -> &Client;
    pub fn try_clone(&self) -> Result<Self>;    // deep copy of all device arrays
    pub fn physical_points(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>>;
    pub fn jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, JacShape<TD, GD>>>;
    /// det J, or sqrt(det(J^T J)) when TD < GD.
    pub fn detj(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, ScalarShape>>;
    /// K = (J^T J)^-1 J^T; equals J^-1 in codimension 0. Needed for surface gradients in BEM.
    pub fn pseudo_inverse_jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, PinvShape<TD, GD>>>;
    /// Sum over owned cells on all ranks, finished in f64 on the host.
    pub fn measure(&self, degree: u32) -> Result<f64>;
    /// One node copied to the host. Tests and debugging only: one device read and one sync per call.
    pub fn read_node(&self, i: usize) -> Result<GD::Array<F>>;
    pub fn create_entities(&mut self, dim: usize) -> Result<()>;
    pub fn create_connectivity(&mut self, from: usize, to: usize) -> Result<()>;
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where F: Real, TD: TopoDim + Codim0<GD>, GD: Dim, C: Comm,
{
    pub fn inverse_jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, PinvShape<TD, GD>>>;
    /// Outward unit normals of every local facet of every cell, at facet quadrature points.
    pub fn facet_normals(&self, rule: &FacetQuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>>;
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where F: Real, TD: TopoDim + Codim1<GD>, GD: Dim, C: Comm,
{
    /// Unit normals; sign from the stored manifold orientation.
    pub fn normals(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>>;
    /// Consistent orientation of a manifold mesh (see "Manifold orientation").
    pub fn orient(&mut self, rule: OrientationRule) -> Result<()>;
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where F: Real, GD: Dim, C: Comm,
      TD: TopoDim + EmbeddedIn<GD>, TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    /// All boundary facets as a mesh one dimension lower, oriented outward.
    /// Returns GridError::NotYetImplemented until M9 (ADR-0012).
    pub fn boundary(&self) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>;
    /// Tagged boundary facets only; interior facets are rejected (GridError::InteriorFacet).
    pub fn facet_submesh(&self, facets: &MeshTags) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>;
}
```

**Listing B4b — quadrature rules and fields (gridcl-geometry)**

```rust
/// One point set per cell type of a dimension, so one rule serves mixed-cell meshes.
pub struct QuadRule<F: Real, TD: TopoDim> { sets: Vec<PointSet<F, TD>> }
impl<F: Real, TD: TopoDim> QuadRule<F, TD> {
    /// Gauss rules exact for polynomials of `degree`, for the given cell types.
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self>;
    /// Gauss rules for every cell type present in `grid`.
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self>;
    pub fn get(&self, cell: TD::Cell) -> Result<&PointSet<F, TD>>;   // CellNotInRule if absent
}

/// Points and weights on one reference cell (host, f64), with device tabulations of coordinate
/// elements: computed in f64, narrowed to F, uploaded on first use and cached per element
/// (ADR-0012). The tabulate calls return share()s of the cached buffers.
pub struct PointSet<F: Real, TD: TopoDim> { /* cell, points, weights, client, tabulation cache */ }
impl<F: Real, TD: TopoDim> PointSet<F, TD> {
    pub fn cell(&self) -> TD::Cell;
    pub fn len(&self) -> usize;
    pub fn points(&self) -> &[f64];                  // reference coordinates, [q][tdim]
    pub fn weights(&self) -> &[f64];
    pub fn tabulate(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>>;             // [q][k]
    pub fn tabulate_derivatives(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>>; // [q][k][d]
}

/// Points on every facet of a cell, in cell coordinates, facet-major (ADR-0012). A cell's facets
/// may differ in point count (prism, pyramid), so a facet field has sum over facets of n_f points per cell.
pub struct FacetPointSet<F: Real, TD: TopoDim> { /* points: PointSet<F, TD>, facet offsets, normals */ }
impl<F: Real, TD: TopoDim> FacetPointSet<F, TD> {
    pub fn points(&self) -> &PointSet<F, TD>;        // the facets' weights
    pub fn facet_offsets(&self) -> &[usize];         // facets + 1 entries
    pub fn reference_normal(&self, q: usize) -> &[f64];
}
pub struct FacetQuadRule<F: Real, TD: TopoDim> { sets: Vec<FacetPointSet<F, TD>> }
impl<F: Real, TD: TopoDim> FacetQuadRule<F, TD> {
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self>;
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self>;
    pub fn get(&self, cell: TD::Cell) -> Result<&FacetPointSet<F, TD>>;
}

pub trait FieldShape { const LEN: usize; type Host<T: Copy + Default>: Copy + Default; }
pub struct ScalarShape;
pub struct VecShape<GD>(PhantomData<GD>);
pub struct JacShape<TD, GD>(PhantomData<(TD, GD)>);    // GD x TD, column-major
pub struct PinvShape<TD, GD>(PhantomData<(TD, GD)>);   // TD x GD, column-major
impl FieldShape for ScalarShape { const LEN: usize = 1; type Host<T: Copy + Default> = T; }
impl<GD: Dim> FieldShape for VecShape<GD> { const LEN: usize = GD::N; type Host<T: Copy + Default> = GD::Array<T>; }
impl<TD: EmbeddedIn<GD>, GD: Dim> FieldShape for JacShape<TD, GD> {
    const LEN: usize = TD::N * GD::N;
    type Host<T: Copy + Default> = <TD as EmbeddedIn<GD>>::Jacobian<T>;
}
impl<TD: EmbeddedIn<GD>, GD: Dim> FieldShape for PinvShape<TD, GD> {
    const LEN: usize = TD::N * GD::N;
    type Host<T: Copy + Default> = <TD as EmbeddedIn<GD>>::PseudoInverse<T>;
}

/// Values at quadrature points of one cell block, in tiles of `tile` cells (1, 2, 4, 8 or 16;
/// VectorPolicy::quad_tile(size_of::<F>()), 16 by default). Component k at point q of cell c is stored at
///     ((c / tile) * num_points * LEN + q * LEN + k) * tile + c % tile,
/// so the lanes of a kernel over consecutive cells read and write contiguous memory. tile = 1 is
/// the plain [cell][point][component] layout. Stored in chunks of whole tiles so that no chunk
/// exceeds the device's largest allocation (device_properties(client).memory.max_page_size,
/// which honours testing::with_properties; ADR-0012); the cell count is padded to whole tiles.
pub struct QuadField<F: Real, S: FieldShape> { chunks: Vec<QuadChunk<F>>, num_cells: usize, num_points: usize, tile: usize, _s: PhantomData<S> }
pub struct QuadChunk<F: Real> { pub first_cell: usize, pub num_cells: usize, pub data: DeviceBuffer<F> }
impl<F: Real, S: FieldShape> QuadField<F, S> {
    pub fn uninit(client: &Client, num_cells: usize, num_points: usize, tile: usize) -> Result<Self>;
    pub fn tile(&self) -> usize;
    pub fn num_cells(&self) -> usize;
    pub fn num_points(&self) -> usize;
    pub fn chunks(&self) -> &[QuadChunk<F>];
    /// Host copy in plain [cell][point] order, whatever the tile width.
    pub fn to_host_cell_major(&self) -> Result<Vec<S::Host<F>>>;
}
/// One QuadField per cell block, indexed like cell_blocks().
pub struct BlockFields<F: Real, S: FieldShape> { pub blocks: Vec<QuadField<F, S>> }

/// The cached E-vector of one geometry block (ADR-0012): chunks of whole tiles, [k][g][cell] with
/// cells fastest, the cell count padded to VECTOR_PAD with copies of the block's last cell. An empty
/// block has none. Built on first use by DeviceGrid::evector.
pub struct EVector<F: Real> { /* chunks */ }
impl<F: Real> EVector<F> {
    pub fn num_cells(&self) -> usize;
    pub fn chunks(&self) -> &[QuadChunk<F>];
}
```

### Kernels receive dimensions from types

The launcher is the only place where type-level dimensions become comptime kernel constants. A mismatch between the host result type (e.g. `JacShape<D2, D3>`) and the kernel layout is therefore impossible by construction. The backend never appears in a type: the launch takes the runtime-erased `&Client`.

**Listing B5 — typed launcher and kernel (gridcl-geometry)**

```rust
/// Comptime parameters need Clone + Debug + PartialEq + Eq + Hash + Send + Sync + 'static.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeomLayout {
    pub tdim: usize, pub gdim: usize, pub nodes_per_cell: usize, pub num_points: usize,
    pub tile: usize,                                   // cells per QuadField tile: 1, 2, 4, 8 or 16
}

impl GeomLayout {
    pub fn new<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(e: &CoordinateElement<TD>, points: usize, tile: usize) -> Self {
        let cell: CellType = e.cell.into();
        GeomLayout { tdim: TD::N, gdim: GD::N,
                     nodes_per_cell: gridcl_refcell::num_lagrange_nodes(cell, e.degree),
                     num_points: points, tile }
    }
}

/// J at quadrature points. The lanes are N consecutive cells of one tile (N divides the tile).
///   xe    E-vector chunk [k][g][cell], cells fastest; row (k, g) starts at (k*G + g) * x_stride.
///   dphi  reference derivatives [q][k][d]; one scalar per (q, k, d), broadcast to every lane.
///   out   tiled QuadField chunk; entry (g, d) of J is component d*G + g (column-major).
/// Offsets and strides count vectors. Each unit handles items_per_unit consecutive cell
/// vectors: 1 on GPUs, a contiguous block on CPUs (LaunchPlan).
#[cube(launch_unchecked)]
fn jacobian_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>], dphi: &[F], out: &mut [Vector<F, N>],
    x_offset: u32, x_stride: u32, out_first_tile: u32,
    num_vectors: u32, items_per_unit: u32,
    #[comptime] l: GeomLayout,
) {
    let vpt = l.tile / N::value();                              // vectors per tile, comptime
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let tile = out_first_tile as usize + v / vpt;
            let lane = v % vpt;
            for q in 0..l.num_points {
                #[unroll]
                for d in 0..l.tdim {
                    #[unroll]
                    for g in 0..l.gdim {
                        let mut acc = Vector::<F, N>::broadcast(F::new(0.0));
                        #[unroll]
                        for k in 0..l.nodes_per_cell {
                            let x = xe[(k * l.gdim + g) * x_stride as usize + x_offset as usize + v];
                            let w = Vector::<F, N>::broadcast(dphi[(q * l.nodes_per_cell + k) * l.tdim + d]);
                            acc += x * w;                       // same order in every lane as for N = 1
                        }
                        let c = d * l.gdim + g;
                        out[((tile * l.num_points + q) * l.tdim * l.gdim + c) * vpt + lane] = acc;
                    }
                }
            }
        }
    }
}

/// A tile-aligned cell range that lies in one E-vector chunk and one field chunk.
struct ChunkPair<'a, F: Real> {
    x: &'a DeviceBuffer<F>, x_first_cell: usize, x_stride_cells: usize,
    out: &'a DeviceBuffer<F>, out_first_cell: usize, padded_cells: usize,
}

pub fn jacobians_block<F, TD, GD, G>(g: &G, b: usize, pts: &PointSet<F, TD>)
    -> Result<QuadField<F, JacShape<TD, GD>>>
where F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim, G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let gb = g.geometry_block(b);
    let tile = VectorPolicy::for_client(g.client()).quad_tile(size_of::<F>());
    let layout = GeomLayout::new::<TD, GD>(&gb.element, pts.len(), tile);
    let num_cells = g.cell_blocks()[b].index_map.num_local();          // owned + ghost
    let xe = g.evector(b)?;                                             // cached; tile-aligned chunks
    let out = QuadField::<F, JacShape<TD, GD>>::uninit(g.client(), num_cells, pts.len(), tile)?;
    let dphi = pts.tabulate_derivatives(&gb.element)?;                  // host f64 tabulation, narrowed
    // Tile-aligned cell ranges that each lie in one E-vector chunk and one field chunk.
    for r in chunk_pairs(&xe, &out) {
        let plan = LaunchPlan::elementwise(g.client(), KernelKind::GeomFactors, r.padded_cells,
                                           size_of::<F>(), /* max_vector */ tile, /* live */ 4)?;
        let n = plan.vector_size;
        unsafe {
            jacobian_kernel::launch_unchecked::<F>(
                g.client(), plan.cube_count, plan.cube_dim, n,
                r.x.vector_arg(n), dphi.arg(), r.out.vector_arg(n),
                (r.x_first_cell / n) as u32, (r.x_stride_cells / n) as u32,
                (r.out_first_cell / tile) as u32,
                plan.num_vectors, plan.items_per_unit, layout,
            );
        }
    }
    Ok(out)
}
```

`LaunchPlan` computes the cube count itself: it fills x up to the device's maximum, then y, then z, so no launch exceeds the 65 535-per-dimension limit of WebGPU-class devices (ADR-0010). On CPUs it gives each thread contiguous blocks of cell vectors. With vector size 1 the same source is the scalar kernel, so the policy can switch sizes without a second implementation. Chunking keeps every buffer within the device's maximum allocation (see "Device limits").

### Runtime erasure for I/O

**Listing B6 — the erased layer (gridcl)**

```rust
pub enum AnyMesh<F: Real, C: Comm = Serial> {
    Curve1(Mesh<F, D1, D1, C>), Curve2(Mesh<F, D1, D2, C>), Curve3(Mesh<F, D1, D3, C>),
    Planar(Mesh<F, D2, D2, C>), Surface(Mesh<F, D2, D3, C>), Volume(Mesh<F, D3, D3, C>),
}

/// Borrowed view, so writers never need to move or clone a mesh.
pub enum AnyMeshRef<'a, F: Real, C: Comm = Serial> {
    Curve1(&'a Mesh<F, D1, D1, C>), Curve2(&'a Mesh<F, D1, D2, C>), Curve3(&'a Mesh<F, D1, D3, C>),
    Planar(&'a Mesh<F, D2, D2, C>), Surface(&'a Mesh<F, D2, D3, C>), Volume(&'a Mesh<F, D3, D3, C>),
}

impl<F: Real, C: Comm> AnyMesh<F, C> {
    pub fn as_ref(&self) -> AnyMeshRef<'_, F, C>;
    pub fn tdim(&self) -> usize;
    pub fn gdim(&self) -> usize;
}

/// Runs `$body` once per variant with `$m: &Mesh<F, TD, GD, C>` of that variant's types.
#[macro_export]
macro_rules! with_mesh {
    ($any:expr, $m:ident => $body:expr) => {
        match $any {
            $crate::AnyMeshRef::Curve1($m) => $body, $crate::AnyMeshRef::Curve2($m) => $body,
            $crate::AnyMeshRef::Curve3($m) => $body, $crate::AnyMeshRef::Planar($m) => $body,
            $crate::AnyMeshRef::Surface($m) => $body, $crate::AnyMeshRef::Volume($m) => $body,
        }
    };
}

// TryFrom<AnyMesh<F, C>> for each of the six Mesh types, generated by one macro:
impl<F: Real, C: Comm> TryFrom<AnyMesh<F, C>> for SurfaceMesh<F, C> {
    type Error = GridError;
    fn try_from(m: AnyMesh<F, C>) -> Result<Self> {
        match m {
            AnyMesh::Surface(s) => Ok(s),
            other => Err(GridError::DimensionMismatch { expected: (2, 3), found: (other.tdim(), other.gdim()) }),
        }
    }
}
```

`Result<T>` is the crate alias `Result<T, E = GridError>`. Call sites pass `any.as_ref()` to `with_mesh!`.

### Trait vocabulary for downstream crates

**Listing B7 — what FEM and BEM crates program against (gridcl-geometry, re-exported by gridcl; ADR-0012)**

```rust
/// Implemented by Mesh and by submeshes; later by extruded or structured grids.
pub trait DeviceGrid {
    type F: Real;
    type TD: TopoDim + EmbeddedIn<Self::GD>;
    type GD: Dim;

    fn client(&self) -> &Client;                       // runtime-erased: no runtime type parameter
    fn comm(&self) -> &dyn CommDyn;                    // Listing S2c
    fn cell_blocks(&self) -> &[CellBlock<Self::TD>];
    fn geometry_block(&self, b: usize) -> &GeometryBlock<Self::TD>;
    fn nodes(&self) -> &DeviceBuffer<Self::F>;         // L-vector [node][gdim], relative to origin()
    /// Centre of the global bounding box; physical_points adds it back (ADR-0012).
    fn origin(&self) -> <Self::GD as Dim>::Array<f64>;
    /// E-vector of geometry block b, built on first use and cached (ADR-0012).
    fn evector(&self, b: usize) -> Result<&EVector<Self::F>>;
    fn node_index_map(&self) -> &IndexMap;
    fn vertex_index_map(&self) -> &IndexMap;
    fn vertex_origin_ids(&self) -> &DeviceBuffer<u32>; // interleaved (hi ^ 0x8000_0000, lo) per local vertex
    /// Err(GridError::EntitiesNotCreated(dim)) until create_entities(dim) has run, and always
    /// for dim = tdim: cells are in cell_blocks() (ADR-0012).
    fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]>;
    /// Err(GridError::ConnectivityNotCreated(key)) until create_connectivity has run;
    /// Err(GridError::UnsupportedConnectivity(key)) for pairs other than (tdim, d) and (d, tdim).
    fn connectivity(&self, key: ConnKey) -> Result<&Connectivity>;
    /// Per-cell orientation bits (A.6); None for interval cells, which have none.
    fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>>;
}
```

This mirrors the mesh traits of ndmesh but returns device views instead of per-entity host iterators. Function spaces, assemblers and BEM operators are generic over `G: DeviceGrid`, so they work unchanged on submeshes and future grid kinds. Accessors create no topology: entities and connectivity are built only by the `&mut self` methods `create_entities` and `create_connectivity`. Derived geometry data (E-vectors) is built on first use and cached (ADR-0012).

`DeviceGrid` lives in gridcl-geometry, because `QuadRule::for_grid` and the block functions of Listing B5 (gridcl-geometry) are generic over it; gridcl re-exports it. `create_entities(d)` accepts 0 ≤ d < tdim, and dimension 0 gives one `Point` block of the vertices. `create_connectivity` supports (tdim, d), which is downward and filled by entity creation, and (d, tdim), which is upward, for 0 ≤ d < tdim; d → 0 is `EntityBlock::vertices` and tdim → 0 is `CellBlock::vertices` (ADR-0012).

## API walkthrough

The file decides the dimensions at run time, the program fixes them once, and the compiler checks everything after that.

### Reading a file and fixing the dimensions once

**Listing B8**

```rust
use cubecl::{Device, device::WgpuDeviceKind};
use gridcl::prelude::*;

let universe = mpi::initialize().expect("MPI already initialised"); // one host thread per rank
let comm = Mpi::world(&universe);
let gpu = local_gpu_index(&comm);                                    // node-local rank
let device = Device::cuda(gpu)
    .or_else(|_| Device::rocm(gpu))
    .or_else(|_| Device::vulkan(WgpuDeviceKind::default()))
    .or_else(|_| Device::metal_msl(WgpuDeviceKind::default()))
    .or_else(|_| Device::cpu())?;
let client = device.client();

// The file decides D_TOP and D_GEO at run time ...
let any: AnyMesh<f64, Mpi> = gmsh::read(&client, comm.clone(), "head.msh",
    GmshOptions::default().partitioner(PtScotch::default()).ghosts(GhostMode::SharedFacet))?;

// ... the program fixes them once.
let mut head: VolumeMesh<f64, Mpi> = any.try_into()?;
let owned: usize = head.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
println!("{owned} owned cells on rank {}, running on {}", comm.rank(), client.name());
```

On a device without f64 support (Metal, some Vulkan adapters), reading into an f64 mesh returns `GridError::UnsupportedPrecision`; use `AnyMesh<f32, _>` there.

### Volume mesh to BEM surface mesh

**Listing B9**

```rust
let scalp_tag = head.tags().named(Codim::Bnd, "scalp")?;             // boundary facets only
let (scalp, emap): (SurfaceMesh<f64, Mpi>, EntityMap) = head.facet_submesh(&scalp_tag)?;
// Each facet has one parent cell; normals inherit outward orientation from it.

let rule = QuadRule::for_grid(&scalp, 6)?;      // one point set per cell type present
let n = scalp.normals(&rule)?;                   // compiles: D2 is Codim1 in D3
let j = scalp.jacobians(&rule)?;                 // per point a [[f64; 3]; 2]: two columns of three
let k = scalp.pseudo_inverse_jacobians(&rule)?;  // per point a [[f64; 2]; 3], for surface gradients
let area = scalp.measure(6)?;                    // pseudo-determinant path, chosen at comptime
```

### Errors the compiler catches

**Listing B10 — verified with trybuild compile-fail tests**

```rust
let n = head.normals(&rule3);
// error[E0599]: the method `normals` exists for struct `Mesh<f64, D3, D3, Mpi>`,
//               but its trait bounds were not satisfied (`D3: Codim1<D3>`)

let block = CellBlock::<D2>::new(Cell3::Tetrahedron, /* ... */);
// error[E0308]: mismatched types: expected `Cell2`, found `Cell3`

let points = curve_in_2d.boundary();
// error[E0599]: the method `boundary` exists for struct `Mesh<f64, D1, D2>`,
//               but its trait bounds were not satisfied (`D0: TopoDim`)
```

The comments show the error codes and the unsatisfied bound, not rustc's exact text. The trybuild `.stderr` files are generated with `TRYBUILD=overwrite`, reviewed, and committed. Note that `volume.boundary().boundary()` compiles: the boundary of a surface is a curve. Only the boundary of a curve is rejected.

### Generic algorithms over any grid

**Listing B11**

```rust
/// Works for volume meshes, surface meshes, submeshes and future grid kinds.
pub fn total_measure<G: DeviceGrid>(g: &G, degree: u32) -> Result<f64> {
    let rule = QuadRule::for_grid(g, degree)?;
    let mut local = 0.0_f64;
    for (b, block) in g.cell_blocks().iter().enumerate() {
        // owned cells only: device reduction per block, finished in f64 on the host
        local += measure_block(g, b, rule.get(block.cell)?)?;
    }
    Ok(g.comm().sum(local))                       // CommExt, Listing S2c
}

/// Codimension-specific algorithms state their requirement in the bound.
pub fn solid_angle_check<G>(g: &G) -> Result<f64>
where G: DeviceGrid, G::TD: Codim1<G::GD> { /* integral of x.n / |x|^3 over the surface */ }
```

### Mixed cells stay runtime

**Listing B12**

```rust
// Exhaustive match over Cell3 lists only 3D cells; no unreachable arms for triangles.
let work: f64 = head.cell_blocks().iter().enumerate().map(|(b, block)| {
    let p = head.geometry_block(b).element.degree as f64;
    let w = match block.cell {
        Cell3::Tetrahedron => 1.0, Cell3::Pyramid => 1.3,
        Cell3::Prism => 1.6, Cell3::Hexahedron => 2.2,
    };
    w * p.powi(3) * block.index_map.num_owned() as f64   // cost model, e.g. partitioner weights
}).sum();

head.create_entities(2)?;                 // triangle and quadrilateral face blocks as needed
let faces = head.entity_blocks(2)?;       // one Triangle block and one Quadrilateral block
```

### Halo exchange and output

**Listing B13**

```rust
let nodes = scalp.node_index_map();
let mut u = DeviceBuffer::<f64>::zeros(scalp.client(), 3 * nodes.num_local())?;
let sc = nodes.scatterer(scalp.client(), 3)?;    // block size 3: a vector per node
sc.forward(&mut u)?;                             // owner -> ghosts (insert)
sc.reverse(&mut u, ReverseOp::Add)?;             // ghosts -> owner, deterministic, no atomics

VtkWriter::vtu("out/scalp").lagrange(true).write(&scalp)?;          // generic over TD, GD
VtkWriter::vtu("out/scalp2").write_any(AnyMeshRef::Surface(&scalp))?; // erased path, no clone
```

## Reference cells and conventions

gridcl uses the DefElement conventions (Conventions 1–2 and Definitions 8–9 of Scroggs et al., [arXiv:2506.20188](https://arxiv.org/html/2506.20188), published 2026). ndelement main at f8cbffe implements them. gridcl-refcell holds its own copy of every table and depends only on gridcl-dims, so the tables are usable on the host, inside `comptime!` blocks and in tests. Agreement with ndelement is checked by the separate gridcl-conformance crate (A.10).

| Cell | D_TOP | V | E | F | Facets | Lagrange nodes N(p) | Gmsh P1/P2 | VTK linear/Lagrange |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Interval | 1 | 2 | 1 | – | point | p+1 | 1 / 8 | 3 / 68 |
| Triangle | 2 | 3 | 3 | 1 | interval | (p+1)(p+2)/2 | 2 / 9 | 5 / 69 |
| Quadrilateral | 2 | 4 | 4 | 1 | interval | (p+1)^2 | 3 / 10 | 9 / 70 |
| Tetrahedron | 3 | 4 | 6 | 4 | triangle | (p+1)(p+2)(p+3)/6 | 4 / 11 | 10 / 71 |
| Hexahedron | 3 | 8 | 12 | 6 | quadrilateral | (p+1)^3 | 5 / 12 | 12 / 72 |
| Prism (wedge) | 3 | 6 | 9 | 5 | 2 triangles + 3 quadrilaterals | (p+1)^2(p+2)/2 | 6 / 13 | 13 / 73 |
| Pyramid | 3 | 5 | 8 | 5 | 1 quadrilateral + 4 triangles | (p+1)(p+2)(2p+3)/6 | 7 / 14 | 14 / – |

- **Vertex numbering (A.1).** Every cell uses the DefElement reference cell, shared with ndelement, Basix and Symfem. Quadrilaterals and hexahedra use tensor (lexicographic) order, e.g. (0,0), (1,0), (0,1), (1,1) for the quadrilateral. This differs from the counter-clockwise order of Gmsh and VTK but keeps the tensor structure that sum-factorisation kernels need. I/O applies the maps in A.9.
- **Sub-entity numbering (A.2, A.3).** Sub-entities of each dimension are numbered by their sorted vertex lists in lexicographic order (DefElement Convention 2). ndelement main and Symfem main use the same rule. For triangles and tetrahedra this reverses the Basix and DOLFINx order (conversion in A.11); for all other cells the two coincide. Every cell stores, per dimension, the local vertex lists of its sub-entities and their cell types, since prism and pyramid faces are mixed.
- **Origin ids.** Every vertex has an origin id, an i64 that does not depend on the partition. It is the Gmsh node tag of the corner node, the id passed to the mesh builder (or the construction index if none is given), or the parent's id for a submesh. Origin ids must be globally unique; readers check this and return `GridError::DuplicateOriginId`. They travel with the vertex through migration and ghosting. On the device they are stored as interleaved (hi ^ 0x8000_0000, lo) u32 pairs, so unsigned pair order equals i64 order and negative ids stay valid; a comptime fast path compares only the lo words when every id lies in [0, 2^32), that is, when every stored hi word is 0x8000_0000 (ADR-0012). Cells carry origin ids too (Gmsh element tags or builder indices); these are used for output ordering and tests. A future refinement must derive new ids independently of the partition, e.g. from the pair of origin ids of the parent edge.
- **Canonical entity orientation (A.6).** An edge or face is stored starting at its smallest origin id and continuing towards the smaller neighbour. This is the low-to-high rule of ndelement, Basix and DOLFINx, applied to origin ids. Every rank sharing an entity therefore agrees on its orientation, and the orientation does not change when the mesh is repartitioned. Storing entity vertex lists in this orientation is a deliberate difference from DOLFINx: device kernels can orient an entity without a lookup.
    - Edges and triangle faces are simply sorted by origin id.
    - For a quadrilateral face the four slots follow its tensor order (boundary cycle 0–1–3–2). Slot 0 holds the lowest vertex, slot 1 its lower neighbour, slot 2 its other neighbour and slot 3 the opposite vertex.
- **Permutation info (A.6).** For each cell, the relation between the cell-local and canonical orientation of every edge and face is packed into one u32 with ndelement's layout: one bit per edge first, then three bits per face. The value equals `ndelement::orientation::compute_orientation` applied to the cell's vertex origin ids. It is computed per cell, independently of entity creation, and is identical on every rank count.
- **Lagrange node layout (A.5).** Blocks are ordered vertices, edges, faces, interior. Inside an entity, nodes follow that entity's own vertex list.
    - Intervals, triangles, quadrilaterals, tetrahedra and hexahedra use exactly ndelement's layout, so the FEM/BEM layer can apply ndelement's DOF transformations without a second convention.
    - ndelement has no Lagrange element on prisms or pyramids. gridcl defines those layouts in A.5, using the DefElement conventions and the same in-entity rule; Basix's equispaced lattices for these cells coincide with it.
    - The pyramid uses DefElement's Lagrange pyramid space (Definition 9), a rational space of dimension (p+1)(p+2)(2p+3)/6. A FEM layer that needs prism or pyramid elements must supply them itself.
- **Node families.** Equispaced points on every cell are required for Gmsh and VTK interoperability. Gauss–Lobatto–Legendre (GLL) points are offered on intervals, quadrilaterals and hexahedra only, matching ndelement. Requesting GLL on any other cell returns `GridError::UnsupportedNodeFamily`. Conversion between families is an interpolation applied on the device; writers always convert to equispaced.
- **DOF transformations.** The base transformations of Scroggs, Dokken, Richardson and Wells (ACM TOMS 48(2), 2022, doi 10.1145/3524456) depend only on the sub-entity type. They are therefore identical in the DefElement and Basix numberings; only the block of the cell transformation that each one lands in moves.

## Device model and backends

Every large array is a typed wrapper around a CubeCL `Handle`, and host access is always explicit. The same code serves every backend; capability and limit checks happen when a mesh is created, not at JIT time.

### Device containers

**Listing S1 — device containers (gridcl-device)**

```rust
use core::marker::PhantomData;
use cubecl::prelude::*;                      // Client, BufferArg, CubeElement, ...
use cubecl::server::Handle;

/// Every allocation is padded to a multiple of VECTOR_PAD elements, the widest vector size any
/// device class uses, so vectorized kernels never need a scalar tail.
pub const VECTOR_PAD: usize = 16;

/// Typed, length-carrying view of one CubeCL allocation. Not Clone: a second view of the same
/// memory is made explicitly with share(), an independent copy with deep_clone().
pub struct DeviceBuffer<T: CubeElement> {
    handle: Handle,                          // keeps the allocation alive
    len: usize,                              // logical length in elements, not bytes
    padded: usize,                           // allocated elements: len rounded up to VECTOR_PAD
    client: Client,
    _t: PhantomData<T>,
}

impl<T: CubeElement> DeviceBuffer<T> {
    /// Uploads `data` and zeroes the padding.
    pub fn from_host(client: &Client, data: &[T]) -> Result<Self> {
        let padded = pad_len(data.len())?;
        let bytes = check_allocation::<T>(client, padded)?;
        let mut host = T::as_bytes(data).to_vec();
        host.resize(bytes, 0);
        let handle = client.create_from_slice(&host);
        Ok(Self { handle, len: data.len(), padded, client: client.clone(), _t: PhantomData })
    }
    /// client.empty() does not clear memory: only for outputs that a kernel fully writes.
    /// The padding stays unspecified.
    pub fn uninit(client: &Client, len: usize) -> Result<Self> {
        let padded = pad_len(len)?;
        let bytes = check_allocation::<T>(client, padded)?;
        Ok(Self { handle: client.empty(bytes), len, padded, client: client.clone(), _t: PhantomData })
    }
    pub fn zeros(client: &Client, len: usize) -> Result<Self>;   // uninit + fill kernel
    pub fn share(&self) -> Self;                                 // same allocation; aliasing is explicit
    pub fn deep_clone(&self) -> Result<Self>;                    // copy kernel (no same-device copy in CubeCL 0.11)
    pub fn len(&self) -> usize { self.len }
    pub fn padded_len(&self) -> usize { self.padded }
    pub fn handle(&self) -> &Handle { &self.handle }
    pub fn client(&self) -> &Client { &self.client }

    /// Explicit, blocking device-to-host copy.
    pub fn to_host(&self) -> Result<Vec<T>> {
        let bytes = self.client.read_one(self.handle.clone())?;
        Ok(T::from_bytes(&bytes)[..self.len].to_vec())
    }
    pub async fn to_host_async(&self) -> Result<Vec<T>> {
        let mut out = self.client.read_async(vec![self.handle.clone()]).await?;
        Ok(T::from_bytes(&out.remove(0))[..self.len].to_vec())
    }
    /// Scalar kernel argument. In CubeCL 0.11 the handle's size, not this length, bounds the
    /// kernel's slice, so every kernel also takes its lengths as explicit arguments.
    /// Unsafe: the kernel must stay within `len` and the padding.
    pub unsafe fn arg(&self) -> BufferArg {
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len) }
    }
    /// Argument for a kernel over Vector<T, n>: the padded length, counted in vectors. Kernels
    /// take `num_vectors` explicitly, as in Listing B5.
    pub unsafe fn vector_arg(&self, n: usize) -> BufferArg {
        debug_assert!(n.is_power_of_two() && VECTOR_PAD % n == 0);
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.padded / n) }
    }
}

/// An empty buffer still gets one block, so no handle has zero size (wgpu rejects zero-size
/// bindings); len() stays 0.
fn pad_len(len: usize) -> Result<usize> {
    len.max(1).checked_next_multiple_of(VECTOR_PAD).ok_or(GridError::TooLarge)
}

/// Bytes for `len` elements, or an error if the device cannot hold them in one allocation.
fn check_allocation<T: CubeElement>(client: &Client, len: usize) -> Result<usize> {
    let bytes = len.checked_mul(core::mem::size_of::<T>()).ok_or(GridError::TooLarge)?;
    let max = client.properties().memory.max_page_size;
    if bytes as u64 > max { return Err(GridError::AllocationTooLarge { bytes, max }); }
    Ok(bytes)
}

/// Fixed number of entries per row (cell -> vertices, etc.).
pub struct DeviceStrided<T: CubeElement> { pub data: DeviceBuffer<T>, pub stride: u32 }

/// Variable-length rows (facet -> cells, vertex -> cells).
pub struct DeviceCsr { pub offsets: DeviceBuffer<u32>, pub indices: DeviceBuffer<u32> }

/// Called whenever a mesh is created on a client, so precision errors surface early.
pub fn check_precision<F: Real>(client: &Client) -> Result<()> {
    if client.properties().supports_type(F::cube_type()) { return Ok(()); }
    Err(GridError::UnsupportedPrecision { scalar: F::type_name(), device: client.name() })
}
```

`GridError` and the alias `Result<T, E = GridError>` live in gridcl-device, the lowest crate that returns them; gridcl-parallel and gridcl re-export both (ADR-0009). `GridError` is `#[non_exhaustive]` and implements `Display`, `std::error::Error` and `From<ServerError>`. Its M2 variants are `TooLarge`, `AllocationTooLarge { bytes: usize, max: u64 }`, `UnsupportedPrecision { scalar: &'static str, device: &'static str }`, `LengthMismatch { expected: usize, found: usize }` and `Server(ServerError)`; later milestones add their variants when they first use them. M3 adds `UnsupportedNodeFamily { cell, family }`, `UnsupportedDegree { cell, degree: u32 }`, `WrongCellDim { found, expected_tdim: usize }`, `DimensionMismatch { expected: (usize, usize), found: (usize, usize) }`, `EntitiesNotCreated(usize)`, `ConnectivityNotCreated(ConnKey)`, `UnsupportedConnectivity(ConnKey)`, `DuplicateOriginId(i64)`, `IndexOutOfRange { index: usize, len: usize }`, `NodeRoleMismatch(usize)`, `DuplicateCellBlock(&'static str)`, `CellNotInRule(&'static str)` and `NotYetImplemented(&'static str)`; cell types and node families are named by `&'static str`, because gridcl-device depends on neither gridcl-dims nor gridcl-refcell. `ConnKey` (Listing T1) lives in gridcl-device for this reason. `pad_len`, `debug_assert_bindings` and `device_properties(&Client) -> Arc<DeviceProperties>` are public: kernel launchers in every crate read device limits through `device_properties`, which honours `testing::with_properties` (ADR-0012). `Mesh` is not `Clone` either; `Mesh::try_clone` copies every array with `deep_clone`.

### Supported backends

The backend is chosen at run time through `cubecl::Device`, and no gridcl type carries a runtime parameter. The table lists what CubeCL 0.11 registers for each backend.

| Backend | cubecl feature | Device | f64 | u64 arithmetic | Float atomic add | GPU-aware MPI | Tested on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU runtime | `cpu` | `Device::cpu()` | yes | yes | f32, f64 | no | M3 Max on every change; hosted Linux CI |
| CUDA | `cuda` | `Device::cuda(i)` | yes | yes | f32; f64 from sm_60 | yes (`gpu-aware-cuda`) | Locust; Myriad V100/A100 for multi-GPU, comparisons and fallback |
| HIP/ROCm | `hip` | `Device::rocm(i)` | yes | yes | f32 | planned (`gpu-aware-hip`) | manual |
| Vulkan via wgpu (SPIR-V) | `vulkan` | `Device::vulkan(kind)` | if the adapter has shaderFloat64 | if the adapter has shaderInt64 | f32 if supported | no | hosted Linux CI (software Vulkan) |
| Metal via wgpu (MSL) | `metal` | `Device::metal_msl(kind)` | no | yes | f32 only | no | M3 Max on every change |
| Metal native | `metal-native` | `Device::metal(kind)` | no | yes | f32 only | no | not gated |
| wgpu WGSL (DX12, WebGPU) | `wgpu` | `Device::wgpu(kind)` | with SHADER_F64 | with SHADER_INT64 | f32 with SHADER_FLOAT32_ATOMIC | no | best effort |

The supported Metal path is wgpu with MSL. The native Metal runtime is optional and not CI-gated. Neither Metal path registers 64-bit atomics or float min/max atomics.

### CUDA toolkit and driver

CubeCL 0.11 compiles each kernel to PTX with its own LLVM (NVPTX backend; feature `cuda`, the default), choosing the PTX ISA version from the node's driver, and the driver JIT-compiles that PTX when it loads the module. The alternative backend `cuda-cpp` compiles CUDA C++ with NVRTC from the toolkit; gridcl uses it only for diagnosis (env-check) (ADR-0001). Two rules follow:

- **No CUDA 13 on V100.** CUDA 13 dropped Volta (sm_70), cudarc takes the CUDA version from the toolkit at build time, and the V100 nodes' driver supports CUDA 12.4, so V100 jobs use a CUDA 12.x toolkit. A100 (sm_80) and GH200 (sm_90, compiled as `sm_90a`) may use CUDA 12 or 13.
- **Toolkit no newer than the driver (`cuda-cpp` only).** On the NVRTC path the driver must understand the PTX the toolkit emits, so the toolkit must not be newer than the CUDA version the node's driver reports in `nvidia-smi`. Otherwise module loading fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.

`CUDA_PATH` and the toolkit module are needed at build time, because cudarc reads the CUDA version from `nvcc`; at run time only `cuda-cpp` (NVRTC) needs them. The env-check probe of milestone E0 records the toolkit and driver versions on every target.

### Precision policy

- f32 is a first-class precision. Every kernel test runs in f32 on every CI backend, and in f64 on the CPU runtime and on any device that supports it.
- `check_precision::<F>` runs whenever a mesh is created. An f64 mesh on Metal fails with `GridError::UnsupportedPrecision`.
- Basis tabulation and quadrature rules are computed in f64 on the host and narrowed to `F` on upload.
- Coordinates are stored relative to a per-mesh origin (the bounding-box centre), kept in f64 on the host and added back by readers, writers and `read_node`. This limits f32 cancellation.
- Reductions such as measures and volumes are summed per block on the device and finished in f64 on the host. `measure` therefore returns f64.
- Reading a Gmsh file into an f32 mesh narrows the coordinates; the reader logs one warning per file.

### Device limits

- **Largest allocation:** `client.properties().memory.max_page_size`. Every `DeviceBuffer` constructor checks it. Per-rank topology and geometry arrays must fit in one allocation each; otherwise mesh creation fails with `GridError::AllocationTooLarge`, and the mesh must be spread over more ranks. Fields at quadrature points and E-vectors are stored in chunks of whole cell tiles (`QuadField`), sized from this limit.
- **Buffers per kernel:** `hardware.max_bindings`. Library kernels use at most six buffers, which fits WebGPU-class devices; id pairs are interleaved in one buffer for this reason. A debug assertion checks each launch.
- **Shared memory:** `hardware.max_shared_memory_size`. Primitives size their tiles from it and assume at least 16 KiB.
- **Cube count:** computed by `LaunchPlan`, which respects `hardware.max_cube_count`; `TooLarge` if no launch fits (ADR-0010). Element counts are multiplied in usize with overflow checks.
- **Units per cube:** `hardware.max_units_per_cube` and `max_cube_dim`. On the CPU runtime both equal the number of hardware threads. Every launch takes its shape from `LaunchPlan`, which respects them (see "Vectorization and CPU execution").
- **Plane size:** read from `hardware.plane_size_min` and `plane_size_max`, never assumed to be 32. Every algorithm must also be correct with plane size 1, as on the CPU runtime.

### Data layouts and index widths

- Local indices are u32 on the device. Origin ids and global indices are i64 on the host and interleaved (hi, lo) u32 pairs on the device; origin ids store hi ^ 0x8000_0000 (ADR-0012).
- Sort keys are arrays of one to four u32 words, stored planar and word-major in one buffer (`SortKeys`, Listing S1b): word w of key i is at `w · pad_len(len) + i`, and word 0 is the most significant. The lanes of a vector are therefore independent keys. Payloads are u32 emission indices, so the total number of emitted keys per launch must stay below 2^32; this is checked.
- Geometry nodes are stored as an L-vector (unique nodes, AoS [node][gdim]). A cached E-vector ([node_in_cell][gdim][cell], cell index fastest) is built on demand for coalesced and vectorized element kernels, following MFEM. It is stored in chunks of whole tiles, and its cell count is padded to a multiple of `VECTOR_PAD` with copies of the block's last cell, so padding lanes compute finite values.
- `QuadField` is tiled by cells (Listing B4b), so the lanes of a vectorized kernel write contiguous memory on CPUs and coalesced memory on GPUs.
- Optional host mirrors carry valid-on-host and valid-on-device flags (MFEM style); any device write invalidates them.

### Device primitives

gridcl-device provides the following primitives (check CubeK first for reusable pieces):

- inclusive and exclusive scan, built as reduce-then-scan;
- stable LSD radix sort over keys of one to four u32 words with 8-bit digits and a u32 payload;
- run-length encoding and unique-by-key;
- histogram, from per-workgroup histograms in shared memory followed by a reduction;
- gather and scatter;
- segmented reduce (sum, min, max) for f32, f64, i32 and u32;
- fill and copy.

They follow eight rules:

- no algorithm relies on forward progress between workgroups, which Vulkan and Metal do not guarantee, so there is no decoupled look-back;
- kernels use no 64-bit atomics and no float atomics; u32 and i32 atomics are allowed;
- results are deterministic, bitwise identical across runs on the same device;
- every primitive passes the same tests on CPU, Vulkan and Metal;
- fill, copy, reduce, the tile-local phase of scan, the digit counting of the sort (SortDigits) and gather are generic over the vector size, and every primitive takes its launch shape from `LaunchPlan` (see "Vectorization and CPU execution");
- one partition per primitive: a primitive with several kernels computes one `LaunchPlan`, for the `KernelKind` of its vectorized phase, or `KernelKind::Scalar` if none is vectorized (ADR-0015), and every kernel takes its ranges from it. A unit owns `items_per_unit · vector_size` consecutive items, and a cube owns `cube_dim` times as many. Scalar phases launch with the same cube count and cube dimension. Recursive levels (the partials of reduce, the tile sums of scan) use one new `LaunchPlan::elementwise` per level, until one cube remains;
- no plane operations: values are combined within a cube through shared memory and `sync_cube`, so every primitive is correct with plane size 1 and on the CPU runtime, which has no plane operations;
- no primitive reads from the device, except that run-length and unique-by-key read their count once (a blocking 4-byte read) and return exactly sized buffers, and that debug builds of gather and scatter read back one value to check their indices (ADR-0009).

**Listing S1b — primitives (gridcl-device)**

```rust
/// The scalars of the typed primitives (sealed): f32, f64, i32, u32.
pub trait DeviceScalar: CubeElement + Numeric + sealed::Sealed {}

/// Sum, min or max. Re-exported by gridcl-parallel, so `gridcl_parallel::ReduceOp` stays valid.
#[derive(Clone, Copy, Debug)] pub enum ReduceOp { Sum, Min, Max }

/// Writes `v` over the padded length.
pub fn fill<T: DeviceScalar>(x: &mut DeviceBuffer<T>, v: T) -> Result<()>;
/// Moves u32 words, so size_of::<T>() must be a multiple of 4; unequal lengths give LengthMismatch.
pub fn copy<T: CubeElement>(src: &DeviceBuffer<T>, dst: &mut DeviceBuffer<T>) -> Result<()>;
/// Length 1; an empty input gives the identity of `op` (0, T::max_value(), T::min_value(); finite
/// for floats), written by the fill kernel (ADR-0011).
pub fn reduce<T: DeviceScalar>(x: &DeviceBuffer<T>, op: ReduceOp) -> Result<DeviceBuffer<T>>;
/// One value per CSR segment, combined in index order. Offsets are clamped to [0, values.len()];
/// a decreasing pair gives an empty segment, whose value is the identity of `op`.
pub fn segmented_reduce<T: DeviceScalar>(values: &DeviceBuffer<T>, offsets: &DeviceBuffer<u32>,
                                         op: ReduceOp) -> Result<DeviceBuffer<T>>;
pub fn inclusive_scan<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>>;
/// n + 1 elements, the last being the total, so the result can serve as CSR offsets. This
/// differs from the usual length-n exclusive scan; the rustdoc says so first.
pub fn exclusive_scan<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>>;
/// out[i] = src[idx[i]]. An index >= src.len() yields 0 through a mask (debug builds panic);
/// the read address is clamped and never out of bounds.
pub fn gather<T: DeviceScalar>(src: &DeviceBuffer<T>, idx: &DeviceBuffer<u32>)
    -> Result<DeviceBuffer<T>>;
/// dst[idx[i]] = src[i]. Out-of-range writes are dropped. Indices must be distinct: duplicates
/// are memory-safe but leave the result unspecified (debug builds panic).
pub fn scatter<T: DeviceScalar>(src: &DeviceBuffer<T>, idx: &DeviceBuffer<u32>,
                                dst: &mut DeviceBuffer<T>) -> Result<()>;
/// Values >= num_bins are not counted.
pub fn histogram(values: &DeviceBuffer<u32>, num_bins: u32) -> Result<DeviceBuffer<u32>>;

/// `len` keys of 1..=4 u32 words, planar: word w of key i at w * pad_len(len) + i; word 0 most
/// significant.
pub struct SortKeys { /* words: DeviceBuffer<u32>, num_words, len */ }
impl SortKeys {
    pub fn uninit(client: &Client, len: usize, num_words: usize) -> Result<Self>;
    pub fn from_host(client: &Client, keys: &[u32], num_words: usize) -> Result<Self>; // key-major
    pub fn to_host(&self) -> Result<Vec<u32>>;                                        // key-major
    pub fn len(&self) -> usize;
    pub fn num_words(&self) -> usize;
    pub fn words(&self) -> &DeviceBuffer<u32>;
    pub fn words_mut(&mut self) -> &mut DeviceBuffer<u32>;
}

/// Stable LSD sort with 8-bit digits. Unequal lengths give LengthMismatch; len >= 2^32 TooLarge.
pub fn radix_sort(keys: &SortKeys, payload: &DeviceBuffer<u32>)
    -> Result<(SortKeys, DeviceBuffer<u32>)>;

/// Runs of consecutive equal keys; offsets has runs + 1 entries.
pub struct Runs { pub keys: SortKeys, pub offsets: DeviceBuffer<u32> }
pub fn run_length(keys: &SortKeys) -> Result<Runs>;

/// first: payload of each group's first key; group: the group of each input position.
pub struct Unique { pub keys: SortKeys, pub first: DeviceBuffer<u32>, pub group: DeviceBuffer<u32> }
pub fn unique_by_key(sorted: &SortKeys, payload: &DeviceBuffer<u32>) -> Result<Unique>;
```

The primitives are re-exported at the root of gridcl-device. An empty input never launches a kernel over that input. Outputs whose size does not depend on it are still written: by the fill kernel (for example reduce, exclusive scan, the offsets of run-length, histogram, gather from an empty source), or by a kernel over another, non-empty input that binds the empty buffer but reads nothing from it (segmented reduce with empty values) (ADR-0011).

### Streams and threading

A CubeCL `Client` uses a thread-local stream by default, so launches from different host threads land on different streams. gridcl's model is one host thread per MPI rank: every launch and transfer for a mesh comes from the thread that created it. `Mesh<…, Mpi>` is not `Send` (the communicator is held in an `Rc`), which enforces this. MPI is initialised with `Threading::Single`. Tests run on separate threads but never share a mesh between threads.

## Vectorization and CPU execution

CubeCL expresses SIMD in the kernel language, so gridcl writes a vectorized kernel once and each backend lowers it: the CPU runtime to the host's SIMD instructions, GPU backends to wide loads and stores. gridcl writes no host-side SIMD code. Which kernels use vectors, and how wide, is decided by the benchmarks of milestone M3s; until then every kernel runs at vector size 1.

### CubeCL's vector model

- **Vectors in kernels.** A kernel generic over `N: Size` takes `&[Vector<F, N>]` arguments. `Vector::broadcast`, lane-wise arithmetic and comparisons, `extract` and `insert` act on all N lanes; `N::value()` is a comptime constant.
- **Size at launch.** The vector size is a launch argument: `kernel::launch_unchecked::<F>(client, count, dim, vector_size, args…)`. CubeCL compiles and caches one kernel per vector size, so every size used costs one JIT compilation. Buffer lengths given to `BufferArg::from_raw_parts` count vectors, not scalars.
- **Sizes the device suggests.** `client.io_optimized_vector_sizes(elem_bytes)` lists sizes widest first, from the device's IO width. `properties().vector_sizes_in_registers(elem_bytes, live)` gives the widest sizes at which `live` vectors stay in registers. `tensor_vector_size_parallel` checks a strided layout against candidate sizes.

The development targets report these widths:

| Device | Register width | Registers | Lanes per register, f32 / f64 | Widest IO-optimized size, f32 / f64 |
| --- | --- | --- | --- | --- |
| M3 Max and Locust's Grace CPU (aarch64, NEON) | 128 bits; LLVM keeps fixed 128-bit vectors even where SVE exists | 32 | 4 / 2 | 16 / 8 |
| Kathleen's and Myriad's Xeon Gold CPUs (x86-64, AVX-512) | 512 bits | 32 | 16 / 8 | 16 / 8 |
| x86-64 with AVX2 only (some hosted CI runners) | 256 bits | 16 | 8 / 4 | 16 / 8 |
| GPUs: CUDA, HIP, Metal, wgpu | 128-bit loads | no register budget | – | 4 / 2 |

On every CPU the runtime sizes IO for 512 bits, because wider vectors amortise each iteration's index arithmetic even past one register. wgpu allows at most four lanes.

### How the CPU runtime runs a kernel

- Every unit of a cube is a separate worker thread, and each unit runs the whole cube loop: unit u visits the absolute positions cube · cube_dim + u for every cube of the launch. The plane size is 1.
- `max_units_per_cube` and each axis of `max_cube_dim` equal the number of hardware threads.
- LLVM compiles each kernel for the host CPU at O3. CubeCL 0.11 fuses a multiply and the add that consumes it into one fused multiply-add (`llvm.fmuladd`) on every target, the CPU included (ADR-0013). The fusion depends only on the kernel source, so each vector lane computes exactly what the same kernel computes at vector size 1.
- Shared memory is sized to the L1 data cache.

Two consequences shape gridcl's kernels. First, with the GPU mapping of one item per unit, neighbouring items go to different threads: threads write to shared cache lines (128 bytes on Apple CPUs) and none streams through memory. Every kernel therefore takes its work distribution from a launch plan that gives each CPU thread contiguous blocks. Second, auto-vectorization is not relied on, because the cube loop strides through memory by cube_dim items; vectors are requested explicitly with `Vector<F, N>`.

### Launch plans and the vector policy

**Listing S2 — launch plans and the vector policy (gridcl-device)**

```rust
/// Kernel families with an entry in the vector policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    Fill, Copy, Reduce, ScanTile, SortDigits, Gather, Pack,
    TopologyKeys, PermutationInfo, EVector, GeomFactors, PullBack,
    /// Kernels not on the candidate list; always vector size 1 (ADR-0015).
    Scalar,
}

/// Device classes the policy distinguishes, derived from Client::properties().
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceClass { CpuSimd128, CpuSimd256, CpuSimd512, Gpu }
impl DeviceClass {
    /// A device with hardware.num_cpu_cores is a CPU, classed by hardware.load_width (512 bits:
    /// CpuSimd512, 256: CpuSimd256, else CpuSimd128); every other device is Gpu. load_width alone
    /// does not tell a CPU from wgpu, which also reports 128 bits.
    pub fn for_client(client: &Client) -> DeviceClass;
}

/// Value of GRIDCL_VECTOR_SIZE (scalar | policy | widest | <n>) and of testing::with_vector_size.
/// Widest is the first of io_optimized_vector_sizes(elem_bytes). Fixed(n) and Widest skip the
/// register budget but stay clamped by max_vector and the device's max_vector_size (ADR-0011);
/// Policy applies the full rule below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSizeOverride { Scalar, Policy, Widest, Fixed(usize) }

/// Vector sizes, QuadField tiles and CPU block sizes chosen from benchmarks
/// (docs/perf/vectorization.md), compiled in from gridcl-device/vector-policy.toml.
/// GRIDCL_VECTOR_SIZE (scalar | policy | widest | <n>) and GRIDCL_QUAD_TILE override it for
/// tests and benchmarks.
pub struct VectorPolicy { /* per (KernelKind, DeviceClass, element size) */ }
impl VectorPolicy {
    pub fn for_client(client: &Client) -> &'static VectorPolicy;
    /// Preferred vector size for one kernel family and scalar size; 1 until benchmarked.
    pub fn vector_size(&self, kind: KernelKind, elem_bytes: usize) -> usize;
    /// Tile width of QuadField and E-vector chunks for one scalar size: 1, 2, 4, 8 or 16.
    pub fn quad_tile(&self, elem_bytes: usize) -> usize;
    /// CPU block of one kind and element size in bytes (vector-policy.toml stores units of
    /// 4 KiB); None on Gpu.
    pub fn cpu_block_bytes(&self, kind: KernelKind, elem_bytes: usize) -> Option<usize>;
    /// CPU units of one kind and element size: all hardware threads or the performance
    /// cores (ADR-0017); None on Gpu.
    pub fn cpu_units(&self, kind: KernelKind, elem_bytes: usize) -> Option<CpuUnits>;
}

/// The CPU units a launch uses: one per hardware thread, or one per performance core where
/// the CPU reports a smaller performance-core count (macOS; ADR-0017).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuUnits { All, Performance }

/// Shape of one launch on the current device. Not Copy: CubeCount is not Copy in CubeCL 0.11.
#[derive(Clone, Debug)]
pub struct LaunchPlan {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    /// Lanes per vector; 1 is a scalar launch.
    pub vector_size: usize,
    /// Consecutive vectors each unit processes: 1 on GPUs, a block on CPUs.
    pub items_per_unit: u32,
    /// Vectors covered, padding included.
    pub num_vectors: u32,
}

impl LaunchPlan {
    /// `items` scalar items; `max_vector` the widest size the data layout allows (a divisor of
    /// VECTOR_PAD); `live` the vectors the kernel keeps live, for the register budget.
    pub fn elementwise(client: &Client, kind: KernelKind, items: usize, elem_bytes: usize,
                       max_vector: usize, live: usize) -> Result<Self>;
}

/// Test hooks, only with gridcl-device's Cargo feature `testing` (off by default, not stable
/// API; gridcl-testing turns it on). Overrides are scoped and thread-local, because gridcl runs
/// one host thread per mesh. Precedence: the scoped per-kind override, then the scoped override,
/// then the environment variable, then vector-policy.toml. An invalid GRIDCL_VECTOR_SIZE or
/// GRIDCL_QUAD_TILE panics.
#[cfg(feature = "testing")]
pub mod testing {
    /// Replaces the properties gridcl-device reads (limits, device class, vector sizes) for
    /// buffers and plans made inside `f`; tests clone client.properties() and modify it.
    pub fn with_properties<R>(props: DeviceProperties, f: impl FnOnce() -> R) -> R;
    pub fn with_vector_size<R>(v: VectorSizeOverride, f: impl FnOnce() -> R) -> R;
    pub fn with_quad_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R;
    /// Vector size of one family; takes precedence over with_vector_size, the environment and
    /// the policy for that kind.
    pub fn with_kind_vector_size<R>(kind: KernelKind, v: VectorSizeOverride,
                                    f: impl FnOnce() -> R) -> R;
    /// CPU block target, a power of two from 4 to 64 KiB, for plans made inside f.
    pub fn with_cpu_block_bytes<R>(bytes: usize, f: impl FnOnce() -> R) -> R;
    /// CPU units for plans made inside f, in place of the policy's (ADR-0017).
    pub fn with_cpu_units<R>(units: CpuUnits, f: impl FnOnce() -> R) -> R;
    /// Replaces the detected performance-core count (None: no split) inside f.
    pub fn with_performance_cores<R>(cores: Option<u32>, f: impl FnOnce() -> R) -> R;
    /// The plans made inside `f`, so a test can assert the vector size it ran at.
    pub fn record_plans<R>(f: impl FnOnce() -> R) -> (R, Vec<(KernelKind, LaunchPlan)>);
}
```

- **Vector size:** the largest power of two not above the policy's size, `max_vector`, the device's `hardware.max_vector_size` (ADR-0011), and the widest of `vector_sizes_in_registers(elem_bytes, live)`.
- **GPUs:** `cube_dim` is 256, or `max_units_per_cube` if that is smaller; `items_per_unit` is 1; the cube count comes from gridcl-device's spread over `max_cube_count` (x first, then y, then z), computed from the properties gridcl-device reads, so injected limits apply (ADR-0010).
- **CPUs:** `cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`, ADR-0011), or the performance-core count where `VectorPolicy::cpu_units` says `Performance` and the device reports a smaller count (macOS); one unit per thread (ADR-0017). The vectors are split into blocks of `items_per_unit` consecutive vectors, `VectorPolicy::cpu_block_bytes` of the widest stream (16 KiB until M3s), rounded to a whole number of 128-byte lines, so block boundaries fall on cache-line boundaries. The block size is chosen per kind and element size by the M3s benchmarks (ADR-0015).
- **One kernel source:** kernels compute `first = ABSOLUTE_POS * items_per_unit` and loop over `items_per_unit` vectors, checking `v < num_vectors` (Listing B5). Every plan is checked against `max_cube_dim`, `max_units_per_cube` and `max_cube_count`.

### Padding and tiles

- `DeviceBuffer` allocations are padded to a multiple of `VECTOR_PAD` = 16 elements (Listing S1). `len()` stays the logical length; `vector_arg(n)` passes the padded length in vectors.
- An empty buffer still allocates one `VECTOR_PAD` block, so no handle has zero size (wgpu rejects zero-size bindings); `len()` stays 0 and `padded_len()` is at least `VECTOR_PAD`. An empty input never launches a kernel over that input. Outputs whose size does not depend on it are still written: by the fill kernel (for example reduce, exclusive scan, the offsets of run-length, histogram, gather from an empty source), or by a kernel over another, non-empty input that binds the empty buffer but reads nothing from it (segmented reduce with empty values) (ADR-0011).
- `from_host` and `zeros` zero the padding; after `uninit` it is unspecified. Element-wise kernels may compute garbage in padding lanes. Reductions and scans mask lanes at index ≥ `len`, so padding never reaches a result.
- Cell data that vectorized kernels read or write is tiled by cells: E-vectors are cell-fastest with their cell count padded by copies of the last cell, and `QuadField` uses tiles of `quad_tile` cells (Listing B4b). Chunks start on tile boundaries. Lanes are consecutive cells, so accesses are contiguous on CPUs and coalesced on GPUs.

### Where vectors apply

Vectors pay off where a kernel does the same arithmetic for many independent items stored contiguously. They do not help indirect gathers and scatters, atomics or data-dependent control flow. Within those limits gridcl uses the candidate list below; M3s measures each family, and the policy keeps vector size 1 wherever no gain is shown.

| Kernel family (`KernelKind`) | Lanes over | Layout prerequisite | Expectation |
| --- | --- | --- | --- |
| Fill, Copy | elements | padded buffer | memory-bound; wider IO saves index arithmetic on CPUs and gives 128-bit accesses on GPUs |
| Reduce (sum, min, max over contiguous ranges) | elements | padded buffer, masked tail | gain on CPUs; float sums are reordered by lane (see "Determinism across vector sizes") |
| ScanTile (tile-local phase of scan) | elements of a tile | padded buffer | to measure: the in-vector prefix needs lane extracts; as a fixed-order prefix it is an allowed cross-lane operation, like a fixed-order reduction |
| SortDigits (digit extraction of the radix sort) | keys | contiguous u32 key words | to measure; the histogram and scatter phases stay scalar |
| Gather, Pack (halo pack and unpack) | output elements | contiguous index list | to measure: indices load as vectors, data is still gathered lane by lane |
| TopologyKeys (sub-entity key emission) | cells of one block | cell-fastest copy of the cell–vertex connectivity | to measure, including the cost of the transpose |
| PermutationInfo | cells of one block | as for TopologyKeys | to measure |
| EVector (E-vector construction) | cells | – | gathers from the L-vector; only the stores are vectors |
| GeomFactors (X, J, det J, K, normals) | cells of one tile | E-vector, tiled `QuadField` | the main expected gain: dense arithmetic with broadcast basis values |
| PullBack (Newton iteration) | points | points tiled like cells | to measure: lanes that have converged keep their value through arithmetic masks while the others iterate |

Histograms, connectivity inversion, segmented reductions over irregular segments and the scatter phase of the sort stay scalar. Kernels in the candidate list are written generic over `N: Size` from the start and run at the policy's size. All other kernels are scalar, plan as `KernelKind::Scalar`, and still take their CPU work distribution from `LaunchPlan`.

### Determinism across vector sizes

- **Lane-independent kernels** (geometric factors, element-wise maps, integer kernels) give bitwise identical results at every vector size on the CPU runtime, because each lane performs the operation sequence of the same kernel at vector size 1 (ADR-0013). On GPUs they agree within the precision tolerances.
- **Float reductions** combine per-lane partial results in a fixed order. The result depends on the vector size but is bitwise reproducible for a fixed size on the same device; the policy fixes the size per kernel family and device class, so the determinism rule holds.
- **Integer reductions and scans** are exact at every size.

### Benchmarks

gridcl-bench, an unpublished workspace crate, measures where vectors help.

- **What it runs:** every candidate family at every vector size the device offers, at QuadField tile widths 1, 4 and 16 for geometry kernels, and on CPUs at block sizes of 4, 16 and 64 KiB. Sizes are 10^5, 10^6 and 10^7 items, in f32, f64 and u32 as the family requires.
- **What it reports:** the median of ten timed runs, each repeating the call until it has lasted at least 20 ms and giving the time per call (ADR-0018), after a warm-up launch that absorbs JIT compilation, the effective bandwidth relative to a copy kernel on the same device, and the compile time per vector size. Results are written as JSON.
- **Where:** the M3 Max (CPU runtime as CpuSimd128, and Metal) and Locust (Grace CPU with 72 cores as CpuSimd128, and H100), plus one socket of a whole Kathleen node (20 cores of a Xeon Gold 6248 with AVX-512, bound with its memory; ADR-0020) for the CPU runtime as CpuSimd512, which Locust cannot provide; a whole CPU-only Myriad D node (36-core Xeon Gold 6240, `-ac allow=D`) only as the fallback (ADR-0016). Myriad's V100 and A100 nodes run the GPU benchmarks only as the fallback when Locust stays busy or is unavailable (ADR-0014). CpuSimd256 uses the CpuSimd512 table unless a hosted runner with AVX2 only is measured.
- **Conditions:** on Myriad, CPU benchmarks request every core of the node (`--cores 36`) so that no other job shares it. Locust has no scheduler and is shared with other users: a benchmark there counts only if the runner's node-load snapshots before and after the run both show the node idle; otherwise it is repeated. On the M3 Max they run on mains power with other applications closed, and the report records the performance- and efficiency-core counts (`sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`). With static blocks, efficiency cores can finish last; M3s therefore also measures `cube_dim` equal to the performance-core count, and the policy chooses between the two per kind and element size (ADR-0017). On a node with several sockets, CPU benchmarks bind to one socket and its memory, one thread per physical core, because the CPU runtime's unpinned threads across sockets make a configuration's time depend on their placement (up to 4×; ADR-0020).
- **Decision rule:** for each kernel family, device class and scalar type, a vector size above 1 is adopted only if it is at least 10% faster than size 1 at both 10^6 and 10^7 items. Among the configurations within 3% of the fastest, the smallest vector size is chosen, because it pads less and keeps fewer registers live. Tile width, CPU block size and CPU units follow the same rule; ties in units go to all hardware threads (ADR-0017). With several devices in one class, the rule uses the geometric mean of the per-device time ratios; the 3% rule chooses among the configurations the 10% rule admits; a 4-byte entry shared by f32 and u32 must be admitted for both (ADR-0015), and for such an entry the 3% rule measures each type against the fastest configuration at a size both admit (ADR-0019). If no admitted configuration is within 3% of the fastest at both sizes, the one with the smallest worst-case ratio is chosen if that ratio is at most 1.10, or at most 1.25 for a 4-byte entry shared by types that admit different sizes or prefer different configurations (ADR-0021), and the report marks it (ADR-0018).
- **Record:** the choices — vector sizes, QuadField tiles per element size, CPU block sizes per kind and element size in units of 4 KiB, and CPU units per kind and element size — go to `gridcl-device/vector-policy.toml`; the measured speedups, choices and reasons go to `docs/perf/vectorization.md`, with the raw JSON under `docs/perf/data/`. The study is re-run in M10 for every kernel family and after any CubeCL pin bump.

## Topology construction on the device

Cells and their vertices come from the input. Edges and faces are created on demand by `create_entities(dim)`, as in DOLFINx, with one sort per entity type across all cell blocks. The algorithm uses only the primitives of gridcl-device, and its result does not depend on the number of ranks.

### Entity creation

1. **Emission plan (host).** For entity dimension d and each entity type T (e.g. triangle faces), list the cell blocks whose cells have sub-entities of type T, with the count k_b per cell from the gridcl-refcell tables. With n̂_b the block's cell count n_b rounded up to `VECTOR_PAD`, prefix sums of k_b · n̂_b give block b the emission range [e_b, e_b + n̂_b · k_b); sub-entity i of cell c has emission index e = e_b + i · n̂_b + c, so the lanes of a vector are aligned consecutive cells. The padded total must stay below 2^32 (ADR-0012).
2. **Emit keys (device).** There is one launch per cell block, with the cell type and sub-entity table as comptime constants. One unit per (cell, local sub-entity i of type T) writes at its emission index e:
    - the key is the sub-entity's local vertex ids, sorted ascending by a sorting network of at most four values (two u32 words for edges, three for triangles, four for quadrilaterals);
    - the payload is e;
    - padding positions (c ≥ n_b) write an all-ones sentinel key, which sorts last and forms one group that is dropped from the entity count and from the decode (ADR-0012).
3. **Sort (device).** Stable LSD radix sort of all keys of type T with their payloads, all blocks together. A face shared by a hexahedron and a prism is therefore found once.
4. **Mark and number.** flag[i] = 1 if i = 0 or key[i] ≠ key[i−1], else 0. The provisional id is id[i] = inclusive_scan(flag)[i] − 1, and the entity count is id[last] + 1. `unique_by_key` computes exactly this: its `group` is the provisional id, and its count read is the one CLAUDE.md allows (ADR-0012).
5. **Provisional connectivity.** Each sorted position i decodes its payload e into (block, cell, local index) using the emission ranges, then writes cell→entity[block][cell][local index] = id[i]. Each head position also writes entity→vertex from its own occurrence, rewritten into canonical orientation using vertex origin ids (A.6 storage slots).
6. **Ownership** (parallel only; see below). Each entity is marked owned or ghost, and every ghost gets its owner rank.
7. **Renumber.** Owned entities come first, ordered by the smallest local cell index that touches them (a segmented minimum over the sorted runs), which preserves locality. Ghosts follow, ordered by (owner rank, global index). The permutation is applied to entity→vertex, and cell→entity is rewritten by a gather.
8. **Global numbers.** An exclusive scan of the owned counts over ranks gives each rank its owned range. Ghost global numbers arrive from their owners by a forward exchange.
9. **Upward connectivity** (on demand). entity→cells is built by counting sort into a `DeviceCsr`: offsets from histogram and exclusive scan, and the rows filled by a stable radix sort of the entity ids with the cell as payload, so each row is in cell order and deterministic without atomics (ADR-0012).

Entities of one dimension are grouped into homogeneous entity blocks by type; a prism mesh, for example, has a triangle face block and a quadrilateral face block. Each entity block has its own IndexMap with [owned | ghost] numbering, so device arrays always have a fixed stride. Connectivity arrays use a per-dimension flat index (block offset + block-local index). The type of the sub-entity at local position i follows from the reference tables, so no per-entry type tag is stored.

**Listing T1 — topology types (gridcl-topology)**

```rust
pub struct EntityBlock {
    pub cell: CellType,                   // entity type, e.g. Triangle
    pub index_map: IndexMap,              // [owned | ghost]
    pub vertices: DeviceStrided<u32>,     // local vertex indices in canonical orientation (A.6)
    pub offset: u32,                      // first flat index of this block in its dimension
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnKey { pub from: u8, pub to: u8 }        // defined in gridcl-device, re-exported (ADR-0012)

pub enum Connectivity {
    /// One strided array per cell block (cell -> edges, cell -> faces, ...), flat indices.
    Downward(Vec<DeviceStrided<u32>>),
    /// Variable-length rows (entity -> cells, vertex -> cells).
    Upward(DeviceCsr),
}
```

### Ownership of vertices and entities

A vertex is *shared* if it appears on more than one rank, whether in owned or ghost cells. Pipeline stage 8 gives each local vertex two sets: its sharer set (all ranks holding it) and its owned-cell sharer set (ranks holding it in an owned cell).

- **Vertex owner.** Chosen among the owned-cell sharers (sorted by rank) as the one at position hash(origin id) mod (number of candidates). A rank that holds the vertex only through ghost cells holds it as a ghost.
- **Purely local entities.** An entity with at least one unshared vertex exists on this rank only, and is owned here.
- **Candidates.** An entity whose vertices are all shared is a sharing candidate. Its key (the sorted origin ids of its vertices) is sent to every rank in the intersection of its vertices' sharer sets. Each receiving rank replies whether it holds the entity, and whether it holds it in an owned cell.
- **Entity owner.** Chosen among the ranks that hold the entity in an owned cell, as the one at position hash(key) mod (number of candidates). There is always at least one such rank, because the owner of any cell containing the entity holds all its vertices. Every other holder keeps the entity as a ghost.
- **Global numbers** flow from owner to ghost holders (forward direction).

All exchanges use the neighbourhood communicator and touch only process-boundary entities, so the volume is small and the exchange runs on the host. Keys are origin ids, so ownership does not depend on local numbering. For a given input and partition the owners are deterministic.

### Permutation info

One unit per cell computes the cell's orientation word (A.6) from its vertices' origin ids, compared as (hi, lo) pairs, and from the comptime sub-entity tables. The word does not depend on entity creation or on the partition. It is computed at mesh creation for every cell with D_TOP ≥ 2, and the conformance suite checks it against `ndelement::orientation::compute_orientation`.

## Geometry

Geometry is a vector-valued Lagrange field of arbitrary degree, evaluated by device kernels from host-tabulated bases. Manifold orientation is supported for manifold surfaces and curves only.

### Coordinate element and storage

- **Coordinate element:** (cell type, degree p, node family). Each cell block has a geometry block: a geometry dofmap (cells × nodes per cell) into the shared node array, as in DOLFINx. Every block of one cell type has one degree; a mesh may use different degrees for different cell types.
- **Geometry as a field** (Firedrake). Raising the degree after loading (NGSolve's `Curve`) is an interpolation, optionally followed by projection onto a CAD or analytic surface given as a callback.
- **Coordinates relative to an origin.** The origin is the centre of the global bounding box, agreed by an allreduce at mesh creation and kept in f64 on the host. Device kernels see coordinates relative to it. `physical_points`, `read_node`, readers and writers add it back.

### Tabulation and quadrature

- **Tabulation** happens on the host in gridcl-refcell, in f64. It uses an orthonormal basis (Legendre, Dubiner, or orthonormal rationomials for pyramids) and the generalised Vandermonde matrix. The matrix is solved with gridcl-refcell's own dense LU with partial pivoting, so the crate needs no linear-algebra dependency.
    - Equispaced points are well conditioned up to about p = 10; above that, GLL points (tensor cells) are recommended.
    - Values and reference derivatives are uploaded once per (element, point set) and narrowed to `F`.
- **Quadrature:** collapsed Gauss–Jacobi rules on triangles, tetrahedra and pyramids. Quadrilaterals and hexahedra use tensor Gauss rules; prisms use a triangle rule times a Gauss interval rule.
- **Rules for mixed meshes:** a `QuadRule<F, TD>` holds one point set per cell type, and a `FacetQuadRule<F, TD>` holds point sets on each reference facet type.

### Geometric factors

Factors are computed on the device, following MFEM's GeometricFactors. Caching by (rule id, factor flags) is deferred to the first milestone that needs it; until then the methods of Listing B4 return fresh fields (ADR-0012). They are stored in cell chunks (`QuadField`). Factor kernels read the E-vector and vectorize across the cells of a tile (Listing B5).

- **Physical points** X(ξ).
- **Jacobian** J (D_GEO × D_TOP, column-major).
- **det J**, or the pseudo-determinant sqrt(det(J^T J)) when D_TOP < D_GEO; the path is chosen at comptime from the layout.
- **Pseudo-inverse** K = (J^T J)^−1 J^T for every pair; it equals J^−1 in codimension 0.
- **Normals** in codimension 1, with s = −1 if the cell's orientation flag is set and +1 otherwise:
    - D2 in D3: n = s (J0 × J1) / |J0 × J1|;
    - D1 in D2: n = s (J0_y, −J0_x) / |J0|, which points outward for counter-clockwise closed curves.
- **Facet normals** of codimension-0 cells, at facet quadrature points: n = K^T n_ref / |K^T n_ref|, using the reference facet normals from gridcl-refcell. The points are facet-major, so a cell has Σ_f n_f points (`FacetPointSet`, Listing B4b; ADR-0012).

### Manifold orientation

`orient(rule)` exists for codimension-1 meshes only (surfaces in 3D, curves in 2D). Following review decision Q5, it supports manifold meshes only:

1. **Manifold check.** Every facet (an edge of a surface, a point of a curve) must have at most two cells; otherwise `GridError::NonManifold`.
2. **Parity per interior facet.** Each of the two cells induces a direction on the shared facet: the edge's direction in the cell's boundary cycle combined with the edge's permutation bit, or for curves whether the point is the cell's start or end. The parity is 0 if the two directions are opposite (consistent), 1 otherwise.
3. **Signs by parity label propagation.** The constraint is s_a · s_b = (−1)^parity.
    - Each cell starts with its own origin id as label and parity 0, and repeatedly adopts the smallest label among its neighbours, accumulating parity. Pointer jumping keeps the number of rounds logarithmic on each rank.
    - Between rounds, (label, parity) of ghost cells is exchanged, until a global allreduce reports no change.
    - Two different parities for the same label give `GridError::NonOrientable` (e.g. a Möbius strip).
4. **Direction of each component**, from `OrientationRule`:
    - `Outward`: closed components (no boundary facets) are flipped where needed so that the enclosed signed measure (1/D_GEO) ∮ x·n is positive. Open components return `GridError::NeedsReferencePoint`.
    - `AwayFrom(p)` and `Towards(p)`: each component is flipped so that ∫ (x − p)·n has the requested sign. These work for open components too.
5. **Result.** Flags are written to bit 0 of `CellBlock::flags`.

Boundary submeshes (`boundary()`, `facet_submesh`) need no propagation. Each facet cell is oriented so that its normal agrees with the outward facet normal of its single parent cell. Tagged interior facets are rejected with `GridError::InteriorFacet`. Non-manifold surfaces and interior-interface submeshes are out of scope.

### Pull-back

Physical-to-reference mapping by Newton iteration, for point location and interpolation between meshes, is part of M9. The Newton kernel is generic over the vector size: lanes that have converged keep their value through arithmetic masks while the others iterate.

## Distributed mesh model

Cells are partitioned without overlap, and each rank also holds a configurable ghost layer: `GhostMode::None`, `SharedFacet` or `SharedVertex { layers }`. Every cell block, entity block and the vertex and node sets carry an IndexMap in the DOLFINx sense. Serial meshes use the same code with `Serial`, whose IndexMaps have no ghosts.

### Communicators, IndexMap and Scatterer

**Listing S2c — communicators, ownership and halo exchange (gridcl-parallel)**

```rust
/// Object-safe communicator interface used by IndexMap, Scatterer and partitioners.
pub trait CommDyn: 'static {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn barrier(&self);
    /// In-place all-reduce of `buf`, read as elements of `dtype`.
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp);
    /// In-place exclusive scan; rank 0 receives the identity of `op` (0, MAX, MIN; finite for
    /// floats, ADR-0011).
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp);
    /// Variable all-to-all; counts in elements.
    fn all_to_all_v_bytes(&self, send: &[u8], send_counts: &[usize],
                          recv: &mut [u8], recv_counts: &[usize], dtype: DType) -> Result<()>;
    /// Neighbourhood variant on a distributed-graph communicator.
    fn neighbour_all_to_all_v_bytes(&self, nb: &Neighbourhood, send: &[u8], send_counts: &[usize],
                                    recv: &mut [u8], recv_counts: &[usize], dtype: DType) -> Result<()>;
    /// Raw handle for PT-Scotch and ParMETIS; None for Serial.
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm>;
}

/// The communicator stored in a Mesh. Not required to be Send or Sync.
pub trait Comm: CommDyn + Clone {}
impl<T: CommDyn + Clone> Comm for T {}

/// Typed helpers on any communicator, including &dyn CommDyn.
pub trait CommExt {
    fn sum<T: CommScalar>(&self, x: T) -> T;
    fn max<T: CommScalar>(&self, x: T) -> T;
    fn exscan_sum<T: CommScalar>(&self, x: T) -> T;
}
impl<C: CommDyn + ?Sized> CommExt for C { /* through all_reduce_bytes and exscan_bytes */ }

#[derive(Clone, Copy, Debug)] pub enum DType { I32, I64, U32, U64, F32, F64 }
pub use gridcl_device::ReduceOp;                              // Sum, Min, Max (Listing S1b)
pub trait CommScalar: bytemuck::Pod { const DTYPE: DType; }   // i32, i64, u32, u64, f32, f64

#[derive(Clone, Copy, Debug, Default)] pub struct Serial;      // no-op collectives
#[cfg(feature = "mpi")]
#[derive(Clone)] pub struct Mpi(Rc<mpi::topology::SimpleCommunicator>); // deliberately not Send

/// Source and destination ranks of a distributed-graph communicator. Opaque; Serial accepts
/// only the empty one. M6 adds construction over a graph communicator.
pub struct Neighbourhood { /* sources, destinations, graph communicator (mpi) */ }
impl Neighbourhood {
    pub fn empty() -> Self;
    pub fn sources(&self) -> &[usize];
    pub fn destinations(&self) -> &[usize];
}

pub struct IndexMap {
    owned_range: (i64, i64),          // global indices [begin, end) owned by this rank
    ghosts: Vec<i64>,                 // global index of each ghost (host)
    ghost_owners: Vec<i32>,           // owning rank of each ghost (host)
    size_global: i64,
    neighbourhood: Neighbourhood,     // source and destination ranks + distributed-graph communicator
    device: OnceCell<SharedIndices>,  // device-side send and receive lists, built on first use
}

impl IndexMap {
    /// Without ghosts: owned range from exscan_sum of num_owned, size from sum; an i64
    /// overflow gives TooLarge. M6 adds the constructor with ghosts.
    pub fn new(comm: &dyn CommDyn, num_owned: usize) -> Result<IndexMap>;
    pub fn owned_range(&self) -> (i64, i64);
    pub fn ghosts(&self) -> &[i64];
    pub fn ghost_owners(&self) -> &[i32];
    pub fn num_owned(&self) -> usize;
    pub fn num_ghosts(&self) -> usize;
    pub fn num_local(&self) -> usize;   // owned + ghosts
    pub fn size_global(&self) -> i64;
    pub fn scatterer(&self, client: &Client, block_size: usize) -> Result<Scatterer>;
}

pub struct Scatterer {
    send_idx: DeviceBuffer<u32>,      // owned local indices to pack, grouped by neighbour
    recv_idx: DeviceBuffer<u32>,      // ghost local indices to unpack
    rev_order: DeviceBuffer<u32>,     // reverse: received entries sorted by target owned index
    rev_segments: DeviceBuffer<u32>,  // reverse: segment offsets, one segment per target
    rev_targets: DeviceBuffer<u32>,   // reverse: owned local index of each segment
    send_counts: Vec<usize>,
    recv_counts: Vec<usize>,
    block_size: usize,
    transport: Transport,             // HostStaged | GpuAware
}

impl Scatterer {
    /// owner -> ghosts (insert). T is moved as u32 words; size_of::<T>() must be a multiple of 4.
    pub fn forward<T: CubeElement>(&self, x: &mut DeviceBuffer<T>) -> Result<()>;
    /// ghosts -> owner by a segmented reduce: deterministic and free of atomics.
    /// ReverseOp::Insert takes the contribution of the lowest neighbour rank.
    pub fn reverse<T: ScatterNumeric>(&self, x: &mut DeviceBuffer<T>, op: ReverseOp) -> Result<()>;
}
```

`ScatterNumeric` is implemented for f32, f64, i32 and u32. Device support for the chosen type is checked when the scatterer is created. The reverse segments are built once, when the scatterer is created, so every reverse exchange is a gather followed by a segmented reduce in a fixed order.

### Transports

- **HostStaged** (default, every backend): a pack kernel fills the send buffer, `Client::read_one` copies it to the host, and the data moves by a neighbourhood all-to-all-v. `Client::write` then uploads the receive buffer and an unpack kernel scatters it. This is the one host–device copy allowed on a library hot path.
- **GpuAware** (CUDA, feature `gpu-aware-cuda`; HIP later as `gpu-aware-hip`): device pointers are handed to a CUDA-aware MPI, using `Client::get_resource` as in Listing S2b. Keep CubeCL's CUDA allocator at its default, `CudaAllocator::Sync`; the stream-ordered `Async` pool may not work with CUDA IPC in MPI libraries.
- **CubeCL collectives:** `Client::all_reduce` and `Client::to_client` (both take `&mut self`) move data between GPUs of one process when `has_device_transport()` is true. They are not used in the one-GPU-per-rank model.

**Listing S2b — GPU-aware path (gridcl-device, feature `gpu-aware-cuda`)**

```rust
// cubecl re-exports the Runtime trait only under cubecl::__private, so depend on
// cubecl-runtime = "=0.11.0". This is the one place outside the launch API that uses runtime types.
use cubecl::{cuda::CudaRuntime, future::block_on, prelude::*};
use cubecl::server::{Handle, ServerError};
use cubecl_runtime::runtime::Runtime;

type CudaServer = <CudaRuntime as Runtime>::Server;   // not exported by name

pub struct PinnedDevicePtr<Pin> {
    pub ptr: u64,                     // CUdeviceptr, already offset to the handle start
    pub bytes: u64,                   // size of the handle's region
    _pin: Pin,                        // the allocation is neither freed nor moved while held
}

pub fn device_ptr(client: &Client, h: &Handle) -> Result<PinnedDevicePtr<impl Send>, ServerError> {
    block_on(client.sync())?;         // producer kernels must finish before MPI reads
    let res = client.get_resource::<CudaServer>(h.clone())?;
    let (ptr, bytes) = (res.resource().ptr, res.resource().size);
    Ok(PinnedDevicePtr { ptr, bytes, _pin: res })
}
// MPI_Isend / MPI_Irecv on ptr; complete the request before any kernel reads the buffer.
```

### MPI bindings

- `mpi` (rsmpi) 0.8.2 provides initialisation, communicators and the basic collectives.
- rsmpi has no distributed-graph communicators or neighbourhood collectives. gridcl-parallel therefore wraps `MPI_Dist_graph_create_adjacent` and `MPI_Neighbor_alltoallv` from `mpi::ffi` in safe functions, each with a SAFETY comment.
- Counts above `i32::MAX` return `GridError::MessageTooLarge`.
- `Mpi` wraps the communicator in an `Rc`. Under Open MPI, `MPI_Comm` is a pointer, so the communicator is not `Send` anyway; with `Rc` the behaviour is the same under every MPI implementation. CI tests both MPICH and Open MPI.

### Parallel creation pipeline

Locality ordering happens before any numbering, so global numbers are assigned once. Each stage is a pure function from documented inputs to documented outputs, so it can be tested in isolation and handed to an implementation agent one stage at a time.

| Stage | Where | What | Output |
| --- | --- | --- | --- |
| 1 Read | host, all ranks | Chunked read: rank r gets cells [rN/P, (r+1)N/P) with origin ids, types and tags, and an independent chunk of nodes (origin node ids and coordinates). For ASCII files, rank 0 reads and scatters. | cell chunk, node chunk |
| 2 Dual graph | host + MPI | Facet keys (sorted origin vertex ids) are hashed to ranks and matched; cells are adjacent if they share a facet. | distributed CSR graph (vtxdist, xadj, adjncy) |
| 3 Partition | host | `Partitioner` trait; vertex weights from the cost model by cell type and degree. | destination rank per cell |
| 4 Migrate | MPI all-to-all-v | Send cells with origin ids and tags to their destination. | owned cells per rank |
| 5 Ghost cells | host + MPI | Cells sharing a vertex or facet with another rank's cells, per GhostMode. | ghost cells with owner ranks |
| 6 Fetch nodes | MPI | Fetch the coordinates of all nodes of owned and ghost cells, by origin node id, from the stage-1 chunk owners. | node coordinates by origin id |
| 7 Order cells | host | Owned cells along a Hilbert curve of their centroids (reverse Cuthill–McKee as an option); ghost cells after, ordered by (owner rank, origin id). | local cell order |
| 8 Number | host + MPI | Sharer sets; owners chosen by hash of origin id among owned-cell sharers. Local numbering is owned-first, with vertices and nodes in order of first touching cell. Global numbering by exclusive scan; ghost numbers come from owners. | cell, vertex and node IndexMaps; local cell-to-vertex arrays and dofmaps |
| 9 Upload | device | Upload cells, dofmaps, nodes (relative to the global origin) and origin ids; compute permutation info. | device-resident mesh |
| 10 Entities | device + small MPI | On demand: sort-based entity creation and boundary ownership exchange. | entity IndexMaps, connectivity |

## Partitioning

Every supported backend partitions the distributed dual graph, so the `Partitioner` trait takes a graph only. PT-Scotch is the production default. The CI default is the built-in block partitioner for pipeline tests, plus vendored METIS on rank 0 for partition-quality tests.

**Listing S3 — partitioner interface (gridcl-partition)**

```rust
pub struct DistGraph {
    pub vtxdist: Vec<i64>,            // len = nranks + 1 (ParMETIS / PT-Scotch convention)
    pub xadj: Vec<i64>,
    pub adjncy: Vec<i64>,             // global cell ids
    pub vwgt: Option<Vec<i64>>,       // cost model per (cell type, degree)
}

pub trait Partitioner {
    fn name(&self) -> &str;
    /// Destination rank of every local graph vertex (cell).
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>>;
}
```

| Backend | Integration | Feature | Role |
| --- | --- | --- | --- |
| Block | built in | – | CI default for pipeline tests. Also honours partition data stored in Gmsh files. |
| METIS on rank 0 | `metis` crate with its `vendored` feature, which builds METIS from source | `metis` | CI default for partition-quality tests; small and medium meshes |
| PT-Scotch | own `ptscotch-sys` (bindgen of `ptscotch.h`, `SCOTCH_Num` width probed at build time) plus a safe wrapper; the existing `scotch-sys` binds only `scotch.h` | `ptscotch` | production default; tested in its own CI job on an image with a known PT-Scotch build |
| Scotch on rank 0 | `scotch` crate | `scotch` | rank-0 alternative; the reference edge cut in M7 |
| ParMETIS | own `parmetis-sys` (no crate exists; `idx_t` and `real_t` widths probed at build time) | `parmetis` | optional; check the licence terms before redistributing binaries |

- **Rank-0 partitioners** gather the graph on rank 0, partition it and scatter the destinations. Rank 0 then needs memory for the whole graph, which limits them to meshes of about 10^7 cells.
- **Default choice** for distributed reads: PT-Scotch when the `ptscotch` feature is enabled, otherwise METIS on rank 0 when `metis` is enabled, otherwise Block with a warning. Serial meshes need no partitioner.
- **Reproducibility:** PT-Scotch is called with a fixed seed (`SCOTCH_randomSeed`, then `SCOTCH_randomReset` before each call), METIS with `METIS_OPTION_SEED`. Results are reproducible for a fixed number of ranks only.

## Import and export

gridcl has its own Gmsh MSH 4.1 reader and writer, a VTU/PVTU writer, and a VTKHDF writer. Node orderings are generated from each format's documented rules and checked against fixtures.

### Gmsh MSH 4.1

- **Own reader and writer.** The mshio crate parses only Entities, Nodes and Elements, reads only MSH 4.1 and cannot write. It is a test oracle, not the implementation.
- **Sections:** $MeshFormat, $PhysicalNames, $Entities, $Nodes and $Elements, optionally $PartitionedEntities and $Periodic. $Periodic is stored but applied only in a later milestone, outside this roadmap. Both ASCII and binary are supported, in either byte order (binary marker), and the data-size field gives the width of `size_t`.
- **Element types** come from a generated table (A.7): complete Lagrange elements at the listed orders, including pyramids 118–124. Serendipity types 16–19 are upgraded to complete Lagrange on read, by evaluating the serendipity map at the missing nodes (8 → 9, 20 → 27, 15 → 18, 13 → 14). Other incomplete types are rejected.
- **Mesh content.** D_TOP is the highest element dimension in the file, and elements of that dimension form the mesh. Lower-dimensional elements become MeshTags:
    - their sorted corner-node tags (origin ids) form keys;
    - after `create_entities(d)`, the keys are matched against the mesh's entities of that dimension by a device sort and merge (distributed: keys hashed to ranks);
    - points match vertices;
    - an element that matches no entity returns `GridError::UnmatchedTagElement`.
- **Tags.** Physical groups become MeshTags per dimension, and the name table is preserved (name → (dimension, value)). Elementary entity tags are kept as secondary tags.
- **Geometric dimension.** The caller may fix D_GEO (`GmshOptions::gdim`). Otherwise it is inferred from a global allreduce over all nodes, relative to the bounding-box diameter L:
    - D_GEO = 3 if any |z| > 10^−12 L;
    - else 2 if any |y| > 10^−12 L;
    - else 1;
    - never below D_TOP. Gmsh always writes three coordinates.
- **Origin ids** are the node tags, and the element tags for cells. Duplicates return `GridError::DuplicateOriginId`.
- **Parallel read.** Binary files are read in byte-range chunks per rank. Block headers are scanned first, since records within a block have a fixed size and offsets are therefore computable. ASCII files are read on rank 0 and scattered.
- **Parallel write:** gather to rank 0, or one partitioned file per rank with $PartitionedEntities.
- **Precision.** Reading into an f32 mesh narrows coordinates and logs one warning per file.

### VTK

- **VTU** (XML with `version="2.2"`, appended raw or base64 binary, optional zlib). Cells are written as linear cells (1, 3, 5, 9, 10, 12, 13, 14) or, for p ≥ 2, Lagrange cells (68–73).
    - The version matters: VTK renumbers higher-order hexahedra in files with a version below 2.1 (`vtkUpdateCellsV8toV9`), and gridcl writes the VTK 9 ordering.
    - Degree is inferred from the point count for isotropic cells.
- **High-order pyramids** are written as vertices only (VTK type 14) with a warning, or as a subdivided output mesh. VTK's type 74 (`VTK_LAGRANGE_PYRAMID`) is a placeholder, and DOLFINx's writer also refuses pyramids with more than five nodes.
- **PVTU** for parallel runs: one piece per rank plus the parallel header. Ghost cells carry a `vtkGhostType` cell array (value 1 = duplicate cell) so that ParaView hides them.
- **VTKHDF** (feature `hdf5`) through hdf5-metno with its `mpio` feature for parallel HDF5: a single file with per-rank offsets, the same cell types and the same ghost array.
- **VTU reading** (linear and Lagrange) has lower priority and is used for round-trip tests.
- **Test oracles:** VTK's own Python reader (pinned VTK ≥ 9.2) checks what ParaView will show. meshio checks the raw arrays; it writes version 0.1 and applies no conversion.

### High-order node orderings

- **Generated, not hand-written.** Each format's ordering is generated from its documented recursive rule, the reference points are computed, and the permutation to the internal layout is derived by coordinate matching.
    - Gmsh: vertices, then edge interiors, then face interiors recursively as lower-order elements, then the volume interior.
    - VTK: vertices, edges and faces with VTK's own edge directions, then the interior; VTK triangle cells and tetrahedron faces and interiors are recursive; quadrilateral, hexahedron and wedge faces and interiors, including the wedge's triangular faces and triangle layers, are lexicographic.
- **Direction convention (A.9):** entry k of a map is the internal index of the file's k-th node, `internal[map[k]] = file[k]`. DOLFINx's arrays use the same direction even though its docstring states the inverse. DOLFINx's triangle and tetrahedron arrays use Basix numbering; convert them with A.11.
- **Checks:**
    - reproduce A.9 for p ≤ 2;
    - reproduce DOLFINx's Gmsh P3 tables: the 20-node tetrahedron after A.11 conversion, and the 16-node quadrilateral, 64-node hexahedron, 40-node prism and 30-node pyramid unchanged;
    - agree with files written by a pinned Gmsh version;
    - check VTK prisms at p = 3, 5 and 6 against VTK-written fixtures, since DOLFINx has no VTK wedge map above p = 2 and the wedge's triangular faces first differ from the recursive order at p = 5 (one level of recursion) and p = 6 (two levels).

## Tags, regions and submeshes

Tags and submeshes connect the volume mesh to boundary problems; the boundary of a volume mesh becomes the BEM surface mesh with outward orientation.

- **MeshTags:** (dimension, sorted local entity indices in flat numbering, i32 values) on the device, covering owned and ghost entities. Ghost values are made consistent by a forward scatter. A host table in `TagRegistry` maps names to (dimension, value), filled from Gmsh physical names.
- **Codimension addressing** (NGSolve): `Codim::Vol`, `Bnd`, `BBnd` and `BBBnd` map to D_TOP, D_TOP−1, D_TOP−2 and D_TOP−3. The same region code then works for curve, surface and volume meshes.
- **Submeshes:** tagged cells (same D_TOP), tagged boundary facets (`facet_submesh`, D_TOP−1), and tagged edges of 3D meshes (curves in 3D). Each comes with an `EntityMap`: child → parent indices on the device, parent → child as a sparse map.
    - Ownership and ghosting are inherited from the parent.
    - Facet submeshes accept boundary facets only (`GridError::InteriorFacet` otherwise), and inherit outward orientation from the single parent cell.
    - Interface submeshes for multi-domain coupling are out of scope.

## Testing and validation

Every acceptance test in the roadmap is automated and runs on the targets below.

| Target | Backends and precisions | When |
| --- | --- | --- |
| M3 Max, local (macOS, arm64) | CubeCL CPU runtime in f32 and f64; Metal via wgpu in f32; MPI | every change |
| Hosted CI, Linux x86-64 | CPU runtime in f32 and f64; Vulkan (software driver) in f32, plus f64 if supported; MPI under MPICH and Open MPI | every push |
| Myriad V100 (sm_70, CUDA 12.x) | CUDA in f32 and f64 | multi-GPU and GPU-aware tests; architecture comparisons; fallback when Locust is unavailable (ADR-0014) |
| Myriad A100 (sm_80) | CUDA in f32 and f64 | when required (see "Development environment and test targets") |
| Locust GH200 (sm_90, aarch64) | CUDA in f32 and f64; CPU runtime on aarch64 Linux | after every task that changes kernels; every milestone review; benchmarks (ADR-0014) |
| Kathleen (Slurm, x86-64, AVX-512, Omni-Path) | CPU runtime in f32 and f64; MPI on one and two nodes | CPU benchmarks; MPI runs beyond 4 ranks; two-node MPI at every review from M6 (ADR-0016) |
| Myriad node with a CUDA-aware MPI | GPU-aware halo exchange | from M10 |

- **Tolerances.** Quantities that are exact in exact arithmetic (unit volume, polynomial reproduction) are checked to 10^−12 relative in f64 and 10^−5 in f32. f32 results are also compared with f64 results to 10^−5 relative. Convergence rates are tested in f64 only.
- **Topology invariants:**
    - closed-form entity counts for structured boxes of every cell type and for the mixed box;
    - Euler characteristic V − E + F − C = 1 for boxes;
    - every facet has one or two cells;
    - neighbouring cells reconstruct the same canonical vertex list from their permutation info.
- **Geometry:**
    - unit-cube volume equals 1 for p = 1–4;
    - sphere surface area converges at a rate of at least p+1 − 0.2 for p = 1–4 over four refinements (even p may superconverge);
    - divergence check: ∮ x·n = D_GEO · V on closed boundaries.
- **Parallel invariance** on 1, 2, 3 and 4 ranks, under both MPICH and Open MPI. The following must be identical to the one-rank run:
    - global entity counts, total volume and tag counts;
    - canonical entity vertex lists, as origin ids;
    - permutation bits, keyed by cell origin id;
    - output files, after sorting by origin id.
- **Determinism.** Two runs on the same device and rank count produce bitwise identical device arrays.
- **Vector-size invariance.** Every kernel that is generic over the vector size runs at size 1 and at every larger power of two up to 16 that the device offers, forced with `GRIDCL_VECTOR_SIZE` or `testing::with_vector_size`, for lengths 0, 1, 15, 16, 17 and 10^6 + 3. Integer results and lane-independent float results are compared bitwise on the CPU runtime and within the tolerances on GPUs; float reductions are compared within the tolerances. Geometry fields are also compared across tile widths 1 and 16. Factor kernels plan with `max_vector` equal to the tile width, so at tile width 1 they run at size 1, and the sweep runs at tile width 16 (ADR-0012). A separately written reference kernel may fuse multiply-adds differently; comparisons with one are bitwise up to the sign of zero on the CPU runtime (ADR-0013).
- **I/O:**
    - read → write → read round trips;
    - a committed fixture corpus of Gmsh files written by a pinned Gmsh version, for every element type up to order 4 (pyramids up to order 3);
    - every VTU fixture read back by VTK's Python reader.
- **Conformance with ndelement:** gridcl-conformance (A.10).
- **Property tests** (proptest) cover node-ordering permutations, primitives and IndexMap scatter consistency. They use sizes up to 10^5 with 256 cases, plus fixed sizes of 10^6 and 10^7 for primitives.
- **Compile-fail tests:** trybuild, as in Listing B10.
- **Test meshes** come from gridcl-testing, with origin ids and any degree p:
    - structured boxes for every cell type: prisms by splitting hexahedra, pyramids by splitting each hexahedron into six around its centre, and a mixed tetrahedron/pyramid/hexahedron box;
    - high-order curved sphere surfaces (icosahedral refinement with nodes projected onto the sphere);
    - curved balls.

## Development environment and test targets

Claude Code runs on the M3 Max and tests there on every change. CUDA work runs on UCL's clusters through a small remote-test runner. Milestone E0 sets this up and records the details in `docs/dev-environment.md`.

| Target | Hardware | Access | Role |
| --- | --- | --- | --- |
| M3 Max (local) | Apple M3 Max, macOS, arm64 | local | Claude Code; CPU runtime (f32, f64) and Metal (f32) on every change; MPI up to 4 ranks |
| Myriad V100 | E and F nodes: 36 cores (Xeon Gold 6140 or 6240, AVX-512) and 2 × V100 (sm_70) each | SGE batch jobs: `-l gpu=N` with `-ac allow=EF` | multi-GPU tests, architecture comparisons and fallback; oldest supported architecture |
| Myriad A100 | L nodes: 36 cores and 4 × A100 40 GB (sm_80) each | SGE batch jobs: `-l gpu=N` with `-ac allow=L` | second CUDA architecture and CUDA 13; comparisons and fallback |
| Myriad CPU | D nodes: 36 cores (Xeon Gold 6240, AVX-512), 192 GB, no GPU | SGE batch jobs with `-ac allow=D` | fallback for CPU-only runs (ADR-0014, ADR-0016) |
| Kathleen | 190 diskless nodes: 2 × 20 cores (Xeon Gold 6248, Cascade Lake, AVX-512), 192 GB, Omni-Path; RHEL 9 | Slurm batch jobs, whole nodes, at most two nodes per job; QoS `singlenode` (1 node) or `test` (2 nodes) | CPU-only runs and CPU benchmarks (CpuSimd512); MPI runs beyond 4 ranks; two-node MPI at every review from M6 (ADR-0016) |
| Locust | NVIDIA GH200 Grace Hopper: Grace aarch64 CPU and Hopper GPU (sm_90) | UCL ARC experimental system; SSH to the node, no scheduler (recorded in E0) | CUDA platform for routine runs, milestone reviews and benchmarks; aarch64 host and Hopper GPU |
| Hosted CI | Linux x86-64 runners | automatic | CPU runtime, Vulkan, both MPI flavours, conformance |

**When the V100 and A100 run.** Locust is the CUDA platform for every run: Claude runs the CUDA tests there after every task that changes kernels and at every milestone review, and runs the benchmarks there (ADR-0014). Claude schedules Myriad V100 or A100 runs only:

- as the fallback, when Locust is unreachable, a run fails because another user occupies its GPU, or a benchmark cannot get an idle node; the run report says why;
- for multi-GPU and GPU-aware MPI tests (`--gpus 2` on V100, `--gpus 4` on A100; Locust has one GPU);
- to compare, when a Locust failure or result looks architecture-dependent (sm_70, sm_80);
- when a change touches CUDA-version- or architecture-specific code, such as the CUDA paths of gridcl-device or the GPU-aware transport;
- when the user asks.

CPU-only runs, such as the CPU benchmarks on x86-64 with AVX-512 (CpuSimd512), which Locust cannot provide, use one whole Kathleen node; benchmarks bind to one socket (ADR-0020). A Myriad D node (`-ac allow=D`, never a GPU node) runs them only as the fallback: when Kathleen is unreachable, a job is still queued after `--max-wait`, or the user asks; the run report says why (ADR-0016). MPI runs beyond the M3 Max's 4 ranks use Kathleen, at most two nodes per job; from M6 on, every milestone review also runs the MPI tests on two Kathleen nodes.

Locust is an experimental system, so its runs are kept short, build with at most half of the node's cores, and never block a milestone when it is unavailable. It is an open multi-user node without a scheduler, so other users' work can share its CPU and GPU at any time.

**Remote-test runner.** Claude and developers run remote tests with one command from the M3 Max:

```text
scripts/hpc/remote-test.sh <target> [--gpus N] [--ranks N] [--cores N] [--nodes N] [--time HH:MM:SS] [--max-wait HH:MM:SS] -- <command>
targets: myriad-v100 | myriad-a100 | myriad-cpu | locust | kathleen
```

1. **Configuration.** It reads `scripts/hpc/targets.env`, which is personal and git-ignored and is created from the committed `targets.example.env`. The file holds the SSH host aliases, the remote base directory, the module lines, `CUDA_PATH` and the scheduler kind (`sge`, `slurm` or `direct`).
2. **Sync.** It copies the working tree to `<base>/<branch>/` with rsync, respecting `.gitignore`, and records the commit and a dirty flag.
3. **Fetch.** It fetches crates on the login node, and the LLVM bundle when needed — on Myriad, whose RHEL 7 is too old for the prebuilt bundle, a Spack-built LLVM of the same version in the bundler's layout; Kathleen's RHEL 9 takes the prebuilt bundle — because Myriad's compute nodes have no internet access, and so that every job runs offline. Jobs run with `CARGO_NET_OFFLINE=true`.
4. **Submit.** It submits `scripts/hpc/jobs/<target>.sh`, with `qsub -terse` on SGE, `sbatch --parsable` on Slurm, or through `ssh` with `timeout` for direct targets. The job loads the modules, records `nvidia-smi` and the toolkit version, builds, and runs the command with `GRIDCL_TEST_DEVICE=cuda`. It writes the exit status and logs to `results/<job id>/`. On direct targets the job also records a node-load snapshot immediately before and after the command: GPU utilisation and memory use, the GPU compute processes and their owners, the 1-minute load average (recorded only), and the CPU share of other users' processes over 5 s. The node counts as idle if no other user has a GPU compute process, GPU utilisation is below 5%, and other users' processes use less than one core in total.
5. **Wait.** It polls the scheduler for up to `--max-wait` (default 3 hours, queue time included). On timeout it deletes the job and exits with status 124.
6. **Report.** It copies the results to `target/hpc/<target>/<job id>/`, prints a summary of at most 40 lines with the first failure first (on direct targets including a `node load:` line, `idle` or `busy` with the reason, for both snapshots), and exits with the command's exit status.

`--cores` requests CPU cores for the job (a whole node for CPU benchmarks); the job script maps it to the scheduler request recorded in E0. `--nodes N` (Kathleen only, 1 or 2) requests whole nodes, and `--ranks N` spreads the ranks evenly over them. `scripts/hpc/status.sh` lists the user's remote jobs and `scripts/hpc/cancel.sh <job id>` deletes one. At most two scheduler jobs (Myriad, Kathleen) run at a time, and one Locust run may run in parallel with them. Myriad jobs are submitted one after the other, because they sync into one remote tree (ADR-0014).

**Myriad rules that shape the runner** (UCL Research Computing documentation):

- From outside the UCL network, SSH needs the UCL VPN or a jump through `ssh-gateway.ucl.ac.uk`, which accepts key-based logins only.
- Login nodes are for managing files, compiling and short tests (under 15 minutes); everything else runs as a job.
- MPI jobs that use GPUs run within a single node. Multi-rank GPU tests therefore use the 2 V100s of an E or F node or the 4 A100s of an L node.
- Home is also scratch, with a 1 TB quota. The runner keeps one tree per branch and removes the trees of merged branches.
- Myriad runs RHEL 7.9 (glibc 2.17), older than CubeCL's prebuilt LLVM bundle allows (glibc ≥ 2.34). The same LLVM version is built with Spack (`scripts/hpc/spack/`, a CPU batch job) and linked with LLD; it is rebuilt when the bundler version changes (ADR-0004).

## Workspace layout

The crate graph keeps serial builds free of MPI and lets kernels live in the crate that uses them, while CubeCL runtime types stay in one crate.

| Crate | Purpose | Depends on |
| --- | --- | --- |
| gridcl-dims | Dimension markers, `EmbeddedIn`/`Codim`, `CellType`, `Cell1`–`Cell3`, `TopoDim` | – |
| gridcl-refcell | Reference cells, sub-entity tables, Lagrange layouts for any p (including the prism and pyramid rules), point families, quadrature, tabulation with its own dense LU, orientation bits, A.6 storage slots, reference facet normals and facet point maps, format permutation generators. Tests read committed JSON fixtures. | gridcl-dims; dev: serde_json |
| gridcl-device | `DeviceBuffer` family, `GridError` and `ConnKey`, capability and limit checks, launch plans and the vector policy, primitives, pack/unpack kernels, GPU-aware pointer access, test hooks (feature `testing`). The only crate that uses CubeCL runtime types. | cubecl; toml; cubecl-runtime (feature `gpu-aware-cuda`); libc (macOS, ADR-0017); dev: proptest |
| gridcl-parallel | `CommDyn`, `Comm`, `CommExt`, `Serial`, `Mpi` (feature `mpi`), `IndexMap`, `Scatterer`, neighbourhood exchange, wrappers over `mpi::ffi` | gridcl-device; bytemuck; mpi (optional) |
| gridcl-topology | Cell and entity blocks, sort-based entity creation, ownership, connectivity, permutation info, benchmark hooks (feature `bench`) | dims, refcell, device, parallel, cubecl; dev: gridcl-testing |
| gridcl-geometry | Coordinate elements, geometry blocks, `DeviceGrid`, point sets and quadrature rules, factor kernels, orientation, E-vectors, pull-back | dims, refcell, device, parallel, topology, cubecl; dev: gridcl-testing |
| gridcl-partition | `DistGraph`, `Partitioner`, block partitioner; METIS, Scotch, PT-Scotch and ParMETIS behind features | parallel; metis, scotch, ptscotch-sys, parmetis-sys (all optional) |
| gridcl-io | Gmsh reader and writer, VTU/PVTU, VTKHDF | topology, geometry, parallel, partition; hdf5-metno (optional) |
| gridcl | Facade: `Mesh`, builder, `AnyMesh`, `AnyMeshRef`, `with_mesh!`, `DeviceGrid` (re-exported), tags, submeshes, prelude | all of the above |
| ptscotch-sys | bindgen FFI to PT-Scotch, with `SCOTCH_Num` width detection | mpi |
| parmetis-sys | bindgen FFI to ParMETIS, with `idx_t` and `real_t` width detection | mpi |
| gridcl-testing (unpublished) | `test_client()` chosen by the `GRIDCL_TEST_DEVICE` environment variable, test-mesh generators, tolerance helpers; forwards runtime features | gridcl; gridcl-device (feature `testing`); gridcl-dims; gridcl-refcell; cubecl with runtime features |
| gridcl-mpi-tests (unpublished) | MPI integration tests as `[[test]]` targets with `harness = false`, run by `scripts/mpi-test.sh` | gridcl, gridcl-testing |
| gridcl-bench (unpublished) | Kernel benchmarks across vector sizes, tile widths and CPU block sizes; end-to-end throughput; JSON output for `docs/perf/` | gridcl, gridcl-device, gridcl-topology (feature `bench`), gridcl-testing; cubecl with runtime features; serde_json, toml |
| conformance/ (gridcl-conformance, outside the workspace) | Compares gridcl-refcell with ndelement at the pinned commit and exports JSON fixtures. The only crate that depends on ndelement and BLAS/LAPACK. | gridcl-refcell (path), ndelement (git), blas-src, lapack-src, openblas-src |
| tools/env-check (outside the workspace) | Probe of a development target: devices, properties, sample kernels in f32 and f64, MPI allreduce; JSON report. Built in E0. | cubecl with the target's runtime features; mpi (optional) |

### Dependency rules

1. Every crate that defines or launches kernels may depend on `cubecl` without runtime features. This covers gridcl-device, gridcl-topology, gridcl-geometry and gridcl.
2. The launch API may be used anywhere: `Client`, `CubeCount`, `CubeDim`, `BufferArg`, `launch_unchecked`, `calculate_cube_count_elemwise` and `Client::properties`.
3. Runtime types appear only in gridcl-device: runtime structs such as `CudaRuntime`, `get_resource`, and `cubecl-runtime`.
4. Runtime features (`cpu`, `cuda`, `hip`, `vulkan`, `metal`, `metal-native`, `wgpu`) are enabled only in gridcl-testing, examples, benchmarks and binaries. The one exception is `gpu-aware-cuda` in gridcl-device, which enables cubecl's `cuda` feature.
5. No crate depends on MPI unless its `mpi` feature is on. `cargo test --workspace` without features needs neither MPI nor a GPU.
6. ndelement appears only in conformance/. The workspace excludes conformance/ and tools/env-check/ (`exclude = ["conformance", "tools/env-check"]`).

### Cargo features

- **Library features** (on gridcl, forwarded to the crates concerned): `mpi`, `ptscotch`, `parmetis`, `metis`, `scotch`, `hdf5`, `gpu-aware-cuda`.
- **Runtime features** (on gridcl-testing and examples): `cpu` (default), `cuda`, `hip`, `vulkan`, `metal`, `metal-native`, `wgpu`. Each maps to the cubecl feature of the same name; cubecl's `vulkan` is wgpu with SPIR-V and `metal` is wgpu with MSL.
- **Test-support feature** (on gridcl-device; not a library feature): `testing` enables `gridcl_device::testing` (Listing S2). gridcl-testing turns it on, so every workspace test has it.
- **Benchmark-support feature** (on gridcl-topology; not a library feature): `bench` enables the hidden `gridcl_topology::bench` module (ADR-0015). gridcl-bench turns it on.

### CI matrix

| Job | Runner | Runs | Trigger |
| --- | --- | --- | --- |
| local | M3 Max (developer machine) | CPU runtime in f32 and f64; Metal in f32; MPI on 1–4 ranks; kernel tests also at `GRIDCL_VECTOR_SIZE=scalar` and `widest` | every change, by Claude (verify skill, test-runner) |
| cpu | hosted Linux | `cargo test --workspace` on the CPU runtime in f32 and f64; clippy; rustdoc | every push |
| mpi-mpich, mpi-openmpi | hosted Linux | gridcl-mpi-tests on 1–4 ranks (`--oversubscribe` for Open MPI) | every push |
| vulkan | hosted Linux with a software Vulkan driver (Mesa lavapipe) | `GRIDCL_TEST_DEVICE=vulkan`, f32, plus f64 if the adapter supports it | every push |
| ptscotch | hosted Linux image with PT-Scotch | partitioner tests | changes to gridcl-partition; nightly |
| conformance | hosted Linux with OpenBLAS, or the M3 Max with Accelerate | `cargo test --manifest-path conformance/Cargo.toml` | nightly; every ndelement pin bump |
| cuda-v100 | Myriad E/F node through `scripts/hpc` | `GRIDCL_TEST_DEVICE=cuda`, f32 and f64 | multi-GPU; fallback |
| cuda-a100 | Myriad L node through `scripts/hpc` | as cuda-v100, with the newest toolkit the driver supports | when required |
| gh200 | Locust through `scripts/hpc` | as cuda-v100, plus the CPU runtime on aarch64 Linux | after every task that changes kernels; every milestone review |
| gpu-aware | Myriad node with a CUDA-aware MPI, through `scripts/hpc` | host-staged vs GPU-aware halo exchange | from M10, at every milestone review |

The hosted CI provider is not fixed. Without one, the M3 Max covers the cpu, MPI and conformance jobs, and Vulkan runs are skipped.

## Implementation roadmap

Each milestone is one Claude Code work package. Every acceptance test is automated and runs on the targets of "Testing and validation", and later milestones depend only on the public interfaces of earlier ones. E0 sets up and checks the development environment before any library code exists. M1a (type layer) precedes M1b (reference cells), because gridcl-refcell builds on gridcl-dims. M3s measures where SIMD vectors pay off once the primitives and mesh kernels exist.

| Milestone | Depends on |
| --- | --- |
| E0 Development environment | – |
| M0 Workspace and CI | E0 |
| M1a Type layer | M0 |
| M1b Reference cells | M1a |
| M2 Device primitives and communication core | M0 |
| M3 Serial mesh core | M1b, M2 |
| M3s SIMD vectorization and benchmarks | M3 |
| M4 Gmsh I/O (serial) | M3 |
| M5 VTK output | M3 |
| M6 Parallel core | M3 |
| M7 Partitioners | M6 |
| M8 Parallel I/O | M4, M5, M7 |
| M9 Submeshes and manifolds | M6 |
| M10 Hardening | all |

### E0 Development environment

**Scope:**

- the targets of "Development environment and test targets": the M3 Max, the Myriad V100 and A100 nodes, and Locust;
- `rust-toolchain.toml` with Rust 1.99 at the repository root, and a `.gitignore` that covers build output, `target/hpc/` and `scripts/hpc/targets.env`;
- `tools/env-check/`, a standalone crate outside the future workspace. For each runtime compiled in (`cpu`, `metal`, `cuda`) it:
    - opens the device and records its name and properties (f64 and u64 support, `max_page_size`, `max_bindings`, `max_cube_count`, plane sizes, `load_width`, `vector_register_count`, `max_units_per_cube`, and `io_optimized_vector_sizes` for f32 and f64);
    - runs an element-wise add over 10^7 elements in f32 and, where supported, f64, at vector size 1 and at the widest size the device offers, with `calculate_cube_count_elemwise`;
    - runs a u32 atomic histogram;
    - with its `mpi` feature, checks an allreduce of rank ids;
    - checks the results and writes a JSON report;
- `scripts/hpc/`: `remote-test.sh` with the interface above, one job script per target, `targets.example.env`, `status.sh` and `cancel.sh`;
- `docs/dev-environment.md`, filled in with everything E0 finds.

**Acceptance:**

- **M3 Max:**
    - env-check passes on the CPU runtime (f32, f64) and on Metal (f32, with f64 reported as unsupported);
    - a 4-rank MPI allreduce passes;
    - `vtk` ≥ 9.2 and `meshio` import in Python;
    - the output of `gmsh --version` and `claude --version` (≥ 2.1.286) is recorded.
- **Myriad V100:** `scripts/hpc/remote-test.sh myriad-v100 -- cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda` runs from the M3 Max without interaction.
    - The report shows sm_70, a CUDA 12.x toolkit no newer than the driver's CUDA version, and passing f32 and f64 kernels.
    - With `--gpus 2`, both GPUs pass and a 2-rank MPI allreduce passes within the node.
    - Whether the MPI is CUDA-aware is recorded.
- **Myriad A100:** the same, with sm_80 and the newest toolkit the driver supports (CUDA 13 if available).
- **Locust:** the same, with sm_90 on an aarch64 host; env-check also passes on the CPU runtime on Grace. If Locust is unavailable, this item is recorded as pending and does not block M0.
- **Runner robustness:**
    - a deliberately failing command exits non-zero and prints its first error;
    - a run that exceeds `--max-wait` is deleted and exits with status 124;
    - no job is left in `qstat` after the acceptance runs.
- **Turnaround:** queue wait and run time are recorded for each remote target, as the median of three runs.
- **Record:** `docs/dev-environment.md` has no open fields left, apart from Locust details that ARC cannot provide.

### M0 Workspace and CI

**Scope:** the workspace of "Workspace layout" as skeleton crates; `rust-toolchain.toml` with Rust 1.99; `cubecl = "=0.11.0"` in `[workspace.dependencies]`; gridcl-testing with `test_client()`; the gridcl-mpi-tests skeleton; the hosted CI jobs cpu, mpi-mpich, mpi-openmpi and vulkan, plus skeletons for the conformance and ptscotch jobs. gridcl-testing selects the device from `GRIDCL_TEST_DEVICE` both locally and inside `scripts/hpc` jobs.

**Acceptance:**

- an element-wise add kernel in the workspace passes on the CPU runtime (f32, f64) and Metal (f32) on the M3 Max, on Vulkan (f32) in hosted CI, and on CUDA (f32, f64) on a Myriad V100 through `scripts/hpc`;
- a 4-rank MPI smoke test (allreduce of rank ids) passes under MPICH and Open MPI;
- the conformance job builds and links ndelement at f8cbffe and compares one table (triangle edges);
- `cargo test --workspace` builds neither ndelement, BLAS nor MPI, checked with `cargo tree`.

### M1a Type layer (gridcl-dims)

**Scope:** Listings B1 and B2. This includes `CellType`, `Cell1`–`Cell3`, `TopoDim` and the `WrongCellDim` conversions, plus a test-only stub `Mesh` with Listing B4's impl bounds.

**Acceptance:**

- **compile-fail tests:**
    - `normals()` on a volume mesh gives E0599;
    - a `Cell3` in a `CellBlock<D2>` gives E0308;
    - `boundary()` on a curve mesh gives E0599;
    - the `.stderr` files are generated with `TRYBUILD=overwrite` and reviewed.
- **compile-pass tests:**
    - `normals()` on surface and planar-curve meshes;
    - `boundary()` on volume and surface meshes;
    - `boundary().boundary()` on a volume mesh.
- `CellType` ↔ `CellN` round trips for every cell; a wrong dimension returns `WrongCellDim`.

### M1b Reference cells (gridcl-refcell, gridcl-conformance)

**Scope:**

- tables A.1–A.4 for the eight cells, generated by the lexicographic rule;
- Lagrange layouts for any p, including the prism and pyramid rules of A.5;
- the DefElement pyramid space, with an orthonormal rationomial basis;
- point families: equispaced on every cell, GLL on tensor cells;
- quadrature, and tabulation with gridcl-refcell's own LU;
- orientation bits (A.6) and the format permutation generators (A.9);
- the conformance crate with fixture export (A.10).

**Acceptance:**

- node counts match A.4 for p ≤ 10;
- partition of unity and reproduction of each element's own space to 10^−12 for p ≤ 8 (f64);
- sub-entity tables equal ndelement's for all eight cells;
- orientation bits equal `compute_orientation` for 10 000 random vertex numberings per cell type with D_TOP ≥ 2;
- point sets equal ndelement's node by node for intervals, triangles, quadrilaterals, tetrahedra and hexahedra, p ≤ 4, equispaced and (tensor cells) GLL;
- the generators reproduce every A.9 row and DOLFINx's Gmsh P3 tables (tet20 after A.11; quad16, hex64, prism40 and pyramid30 unchanged);
- quadrature integrates monomials (rationomials on pyramids) up to its degree to 10^−13;
- `cargo test --workspace` still builds neither ndelement nor BLAS.

### M2 Device primitives and communication core (gridcl-device, gridcl-parallel)

**Scope:**

- `Real` and the `DeviceBuffer` family with limit checks and `VECTOR_PAD` padding, plus `check_precision`;
- `LaunchPlan` and `VectorPolicy`, with vector size 1 and tile width 16 everywhere until M3s, and the `GRIDCL_VECTOR_SIZE` and `GRIDCL_QUAD_TILE` overrides;
- scan, radix sort (u32-word keys with u32 payload), unique-by-key, run-length, histogram, gather/scatter, segmented reduce, fill and copy. Fill, copy, reduce, the tile-local scan phase, the digit counting of the sort (SortDigits) and gather are generic over the vector size (ADR-0009);
- `CommDyn`, `Comm`, `CommExt` and `Serial`, and `IndexMap` without ghosts.

**Acceptance:**

- property tests against CPU references, with sizes up to 10^5 and 256 cases plus fixed sizes of 10^6 and 10^7. They run on the CPU runtime, and on Vulkan and Metal with u32 keys and f32 values;
- results are bitwise identical across two runs;
- the throughput of sorting 10^7 two-word keys is recorded per backend;
- every vector-generic primitive gives the same results at vector size 1 and at every larger power of two up to 16 that the device offers, for lengths 0, 1, 15, 16, 17 and 10^6 + 3: integers bitwise, float reductions within the tolerances;
- on the CPU runtime, `LaunchPlan` uses one unit per hardware thread and blocks of whole 128-byte lines; with injected device properties, no plan exceeds `max_units_per_cube`, `max_cube_dim` or `max_cube_count` (property test);
- with an injected limit, an allocation above `max_page_size` returns `AllocationTooLarge`;
- an f64 request on Metal returns `UnsupportedPrecision`.

### M3 Serial mesh core

**Scope:**

- `Mesh` from host arrays, built with origin ids;
- device entity creation ("Topology construction"), connectivity and permutation info;
- geometry blocks, `QuadRule` and `FacetQuadRule`;
- E-vectors of the geometry nodes, built on demand and cached per geometry block, and the tiled `QuadField`;
- factor kernels (X, J, det J, K, facet normals, normals), generic over the vector size, and `measure`;
- `AnyMesh`, `AnyMeshRef`, `with_mesh!` and `DeviceGrid` (change log, BLK-3); `boundary()` is declared and returns `NotYetImplemented` until M9 (ADR-0012);
- the gridcl-testing mesh generators;
- the real `Mesh` replaces the M1a stub in the compile-fail tests.

**Acceptance:**

- closed-form entity counts and Euler characteristic 1 for boxes of every cell type and the mixed box, at three refinements;
- unit volume for p = 1–4, to 10^−12 in f64 and 10^−5 in f32;
- sphere-area convergence rate ≥ p+1 − 0.2 for p = 1–4 over four refinements (f64);
- f32 results agree with f64 to 10^−5 relative;
- Jacobians on a mixed triangle–quadrilateral surface mesh use one `QuadRule` and match a host reference to 10^−12;
- permutation info equals the conformance fixtures for every cell;
- with an injected limit, `QuadField` results are identical with 1 and 7 chunks;
- factor kernels on E-vectors agree with a scalar L-vector reference kernel at every vector size from 1 up to 16 that the device offers and the tile holds, and at tile widths 1 and 16, bitwise on the CPU runtime, with +0 and −0 equal (ADR-0013), and within the tolerances on GPUs.

### M3s SIMD vectorization and benchmarks

**Scope:**

- gridcl-bench, with the protocol of "Vectorization and CPU execution": vector sizes, tile widths, CPU block sizes, problem sizes, medians, compile times and JSON output;
- benchmarks of every candidate family that exists after M3: Fill, Copy, Reduce, ScanTile, SortDigits, Gather, TopologyKeys, PermutationInfo, EVector and GeomFactors;
- for TopologyKeys and PermutationInfo, the cell-fastest copy of the connectivity they need, built behind the policy and measured including the transpose;
- the decision rule applied for every family, device class and scalar type; `vector-policy.toml` and `docs/perf/vectorization.md` written.

**Acceptance:**

- benchmark JSON committed under `docs/perf/data/` for the M3 Max (CPU runtime, Metal), Locust (CPU runtime on Grace, CUDA) and one socket of a whole Kathleen node (CPU runtime on its AVX-512 Xeon; ADR-0020), plus a Myriad D node if Kathleen was unavailable, and Myriad's V100 and A100 nodes if Locust was unavailable (ADR-0014, ADR-0016);
- `docs/perf/vectorization.md` lists, for every candidate family, device class and scalar type, the measured speedup of each vector size over size 1, the chosen size and the reason;
- every vector size above 1 in `vector-policy.toml` is at least 10% faster than size 1 at 10^6 and 10^7 items on its device class (the geometric mean over its measured devices), and every chosen configuration (vector size, tile width, block size, CPU units) is within 3% of the fastest one the 10% rule admits (ADR-0015; for a shared 4-byte entry, at a size every type admits, ADR-0019), or, where none is, the closest one, at most 10% slower (25% for a shared 4-byte entry, ADR-0021) and marked in the report (ADR-0018);
- a test checks that `vector-policy.toml` covers every `KernelKind` and `DeviceClass`, and that every numeric value is a power of two from 1 to 16 and every `cpu_units` value is `all` or `performance` (ADR-0017);
- with the chosen policy, the whole test suite passes on every target, including the vector-size invariance tests;
- entities created per second and geometric factors per second are recorded on the M3 Max CPU runtime and one socket of a Kathleen node, with the chosen policy and with every kernel scalar (ADR-0016, ADR-0020).

### M4 Gmsh I/O (serial)

**Scope:** reader and writer, the element table (A.7), serendipity upgrade, MeshTags from physical groups by key matching, D_TOP and D_GEO inference, and origin ids from tags.

**Acceptance:**

- read → write → read gives identical topology, and coordinates equal to 10^−15 relative (f64);
- the committed fixture corpus reads without error with correct node counts. It is written by a pinned Gmsh version and covers every element type up to order 4, pyramids up to order 3;
- nodes and elements agree with mshio on every fixture;
- tag counts equal Gmsh's physical-group sizes;
- D_GEO is inferred correctly on 1D, 2D and 3D fixtures, and duplicate node tags are rejected.

### M5 VTK output

**Scope:** VTU (version 2.2) with linear and Lagrange cells, cell and point data, the PVTU skeleton and the pyramid fallback.

**Acceptance:**

- VTK's Python reader (pinned, ≥ 9.2) reads every fixture;
- the integrated volume or area of P2 and P3 meshes (`vtkIntegrateAttributes`) equals gridcl's `measure` to 10^−10 (f64);
- meshio reads identical raw arrays;
- a P2 pyramid mesh is written as linear cells with one warning.

### M6 Parallel core

**Scope:**

- `Mpi` and the `mpi::ffi` wrappers (distributed graph, neighbourhood all-to-all-v);
- `IndexMap` with ghosts, and `Scatterer` (host-staged, deterministic reverse); its pack and unpack kernels take their shape from `LaunchPlan` and are generic over the vector size (`KernelKind::Pack`);
- pipeline stages 1–10 from host arrays with the block partitioner, ghost modes, ownership rules, global numbering and locality ordering.

**Acceptance** on 1–4 ranks under MPICH and Open MPI:

- global entity counts, total volume and tag counts are identical to one rank;
- canonical entity vertex lists (origin ids) and permutation bits per cell origin id are identical to one rank;
- a forward scatter of global ids reproduces the ghost global ids;
- a reverse Add of ones equals the number of ranks holding each index;
- two runs are bitwise identical;
- on a randomly permuted structured box, locality ordering at least halves the mean over cells of (max − min local node index in the cell).

### M7 Partitioners

**Scope:** the distributed dual graph, ptscotch-sys and its wrapper, parmetis-sys and its wrapper (feature-gated), METIS on rank 0 (vendored), Scotch on rank 0, and vertex weights.

**Acceptance** on a 10^6-cell tetrahedral box and a 2 · 10^5-cell mixed hexahedron/prism/pyramid mesh, at 4 and 8 ranks (8-rank runs on one Kathleen node through `scripts/hpc`; ADR-0016):

- weighted imbalance ≤ 5%;
- PT-Scotch's edge cut is within 10% of sequential Scotch on the same graph, and METIS's within 10% of Scotch;
- a fixed seed reproduces the partition for a fixed rank count.

### M8 Parallel I/O

**Scope:** chunked binary Gmsh reading, partitioned writing, PVTU with `vtkGhostType`, and VTKHDF (hdf5-metno with `mpio`).

**Acceptance:**

- the global mesh, compared by origin ids, is identical for 1–8 ranks;
- reading a 10^7-cell binary file on 8 ranks of one Kathleen node keeps peak resident memory per rank below twice the rank's final mesh size plus 256 MiB (measured with getrusage);
- VTKHDF read back by VTK's Python reader equals the PVTU content;
- after VTK removes ghost cells, every cell appears exactly once.

### M9 Submeshes and manifolds

**Scope:** boundary, facet and edge submeshes with `EntityMap` (`boundary()` replaces M3's `NotYetImplemented`); distributed manifold orientation (`orient`); the manifold tests of normals, pseudo-inverse and facet normals, whose kernels exist from M3 (ADR-0012); pull-back, generic over the vector size.

**Acceptance:**

- normals on the boundary of a centred ball satisfy n·x > 0 at every quadrature point;
- ∮ x·n = D_GEO · V to 10^−10 (f64);
- `orient(Outward)` on a closed sphere with randomly flipped cells recovers outward normals on 1–4 ranks;
- error cases:
    - an edge with three cells gives `NonManifold`;
    - interior facets in `facet_submesh` give `InteriorFacet`;
    - `Outward` on an open surface gives `NeedsReferencePoint`;
    - a Möbius strip gives `NonOrientable`;
- pull-back recovers reference points to 10^−12 for p = 1–3;
- pull-back gives the same reference points at vector size 1 and at the widest size the device offers, to 10^−12.

### M10 Hardening

**Scope:** the GPU-aware transport (CUDA), benchmarks (including a re-run of the M3s vectorization study for every kernel family), documentation and examples.

**Acceptance:**

- on a Myriad node with a CUDA-aware MPI, the GPU-aware halo exchange is bitwise identical to the host-staged one on 2 ranks (V100 node) and 4 ranks (A100 node);
- benchmarks (entities created per second, geometric factors per second) are recorded on the M3 Max (CPU runtime, Metal), Locust and one socket of a whole Kathleen node (AVX-512; ADR-0020), plus a Myriad D node if Kathleen was unavailable, and Myriad's V100 and A100 nodes if Locust was unavailable (ADR-0014, ADR-0016);
- the M3s study covers every candidate `KernelKind`, including Pack and PullBack; `vector-policy.toml` and `docs/perf/vectorization.md` are updated from a re-run on every device class;
- every example runs on the CPU runtime and on at least one GPU backend;
- rustdoc builds without warnings, and every public item is documented.

## Risks and mitigations

The largest risks are CubeCL churn, the capability gaps of wgpu and Metal, and the external partitioner and MPI bindings that must be written from scratch.

| Risk | Mitigation |
| --- | --- |
| CubeCL API churn (alpha, breaking minor releases) | Pin `=0.11.0`. Keep runtime types in gridcl-device. Upgrade as a dedicated milestone, with the kernel test suite as safety net. |
| Backend capability gaps: no f64 on Metal; f64 and u64 adapter-dependent on Vulkan and WGSL; no 64-bit or float min/max atomics on Metal | f32 is first-class; sort keys are u32 words; kernels use no 64-bit or float atomics; capabilities are checked at mesh creation; CI runs on Vulkan and Metal. |
| Device limits: allocation size, buffers per kernel, workgroups per dimension, shared memory | Query `Client::properties`; chunk fields by cells; check per-rank array sizes at creation; use `LaunchPlan`; keep kernels to six buffers. |
| No device-wide sort or scan in CubeCL; no forward-progress guarantee on Vulkan and Metal | Implement reduce-then-scan and LSD radix sort in gridcl-device; check CubeK first; benchmark in M2, because topology creation depends on them. |
| f32 accuracy for curved high-order geometry | Tabulate in f64 on the host; store coordinates relative to the global origin; finish reductions in f64; test f32 against f64. |
| GPU-aware MPI relies on `Client::get_resource` and on naming the server type through `cubecl-runtime`; CUDA allocator settings | Host-staged transport stays the default. GPU-aware path behind a feature, with an integration test on a CUDA-aware MPI runner. Keep `CudaAllocator::Sync`. Re-check on every CubeCL upgrade. |
| MPI flavour differences: `MPI_Comm` is a pointer in Open MPI and an integer in MPICH | `Mpi` holds the communicator in an `Rc` and is not `Send` under either; `Comm` requires neither `Send` nor `Sync`; CI runs both MPICH and Open MPI. |
| rsmpi lacks neighbourhood collectives; no Rust bindings exist for PT-Scotch's distributed API or for ParMETIS | Own wrappers over `mpi::ffi`; own ptscotch-sys and parmetis-sys with width probing; vendored METIS as the dependency-free fallback; dedicated CI image; check the ParMETIS licence. |
| Rust 1.95 MSRV is declared only in CubeCL's workspace, not in its published manifests | Pin the toolchain (Rust 1.99) in `rust-toolchain.toml`; CI images with that toolchain. |
| Pyramid geometry: Gmsh's high-order pyramid basis may differ from the DefElement space inside the cell | Use the DefElement Lagrange pyramid space; validate face traces against Gmsh fixtures; treat pyramids with p > 2 as experimental. |
| ndelement's current numbering is unreleased; ndelement has no prism or pyramid elements | Pin commit f8cbffe in gridcl-conformance only. The nightly job and every pin bump re-check the conventions, and a failure forces an explicit decision. Prisms and pyramids are checked against DOLFINx's Gmsh maps and Gmsh fixtures. |
| Node-ordering mistakes in I/O; VTK's legacy renumbering of higher-order hexahedra | Generated orderings with coordinate matching; Gmsh-written fixtures; VTU version 2.2; checks through VTK's own reader. |
| Generic bounds spreading through downstream crates | `DeviceGrid` and the type aliases; minimal struct bounds; cell type and degree stay runtime. |
| Unmaintained dependencies | hdf5-metno instead of hdf5; mshio as a test oracle only; coupe removed. |
| CUDA 13 dropped sm_70, and the driver JIT-compiles CubeCL's PTX | V100 jobs use a CUDA 12.x toolkit; CubeCL's default LLVM path chooses the PTX version for the node's driver, and the NVRTC path (`cuda-cpp`) needs a toolkit no newer than the driver's CUDA version (ADR-0001); env-check records both on every target. |
| CubeCL's LLVM NVPTX backend advertises fewer device features than NVRTC (MMA) | gridcl uses no tensor-core MMA; env-check runs both backends on every remote target (ADR-0001). |
| Remote targets: VPN or SSH-gateway access, queue waits, quota; Locust is an experimental system that may be unavailable | One runner script with SSH connection reuse; turnaround measured in E0; Locust (no queue) for all CUDA runs, reviews included, and the V100 as the fallback (ADR-0014); Locust runs never block a milestone. |
| Compute nodes may have no internet access, and the CPU runtime downloads an LLVM bundle from github.com at build time | Fetch crates and the bundle on the login node; jobs run offline. The prebuilt bundle needs glibc ≥ 2.34; on Myriad the same LLVM version is built with Spack and linked with LLD. |
| Spack-built LLVM relies on undocumented bundler internals (sentinel file, directory name, `llvm-config`) | Rebuild and re-run env-check on Myriad with every CubeCL upgrade; Apptainer image as the fallback (ADR-0004). |
| SIMD gains depend on kernel and hardware, and each vector size is a separate JIT compilation | Benchmark-driven policy (M3s) with scalar defaults; one vector size per kernel family and device class at run time; compile time recorded. |
| Wider vectors raise register pressure and padding memory | Register-aware sizes from `vector_sizes_in_registers`; padding of at most 15 elements per buffer and 15 cells per block. |
| CubeCL's vector API may change between releases | Pinned at `=0.11.0`; vector handling confined to `LaunchPlan`, `DeviceBuffer::vector_arg` and the kernels of the candidate list. |
| A tiled `QuadField` is a less familiar layout for downstream crates | The index formula is documented; `to_host_cell_major` gives the plain layout; tile width 1 stays available. |

## Handing this design to Claude Code

Convert this document to `docs/design.md` in the repository, add the working agreement below as `CLAUDE.md`, and run one milestone per session. Ask for failing tests from the acceptance list first, then the implementation.

**Listing H1 — CLAUDE.md**

```markdown
# gridcl - working agreement

- Design reference: docs/design.md (gridcl Design B, revised design, October 2026).
  Public types in "Architecture" and "API walkthrough" change only via an ADR in docs/adr/.
- Toolchain: stable Rust 1.99, pinned in rust-toolchain.toml.
- CubeCL: cubecl = "=0.11.0" in [workspace.dependencies].
  - Crates that define or launch kernels may depend on cubecl without runtime features.
  - Runtime types (runtime structs, get_resource, cubecl-runtime) only in gridcl-device.
  - Never add a Runtime type parameter to a public type; use cubecl::prelude::Client.
  - Runtime features only in gridcl-testing, examples, benches and binaries
    (exception: gpu-aware-cuda in gridcl-device).
- Kernels:
  - launch shape (cube count, cube dim, vector size, items per unit) only from LaunchPlan;
    usize arithmetic with overflow checks;
  - at most six buffers; no 64-bit atomics; no float atomics;
  - no assumption of forward progress between workgroups (no decoupled look-back);
  - never assume a plane size; correct for plane size 1;
  - results bitwise deterministic.
- SIMD only through CubeCL Vector<F, N> in kernels; never intrinsics or std::simd.
  - kernels on the candidate list are generic over N: Size and correct for N = 1;
  - lanes are independent items (cells, elements); no cross-lane operations except
    fixed-order reductions and fixed-order in-vector prefixes; no branches on lane values
    (use arithmetic masks);
  - vector sizes come from VectorPolicy (vector-policy.toml), changed only with benchmark
    data in docs/perf/vectorization.md.
- Precision: every kernel test runs in f32 and, where supported, f64.
  Tolerances: 1e-12 relative (f64), 1e-5 (f32).
- Host maths generic over F: Real uses widen()/narrow() or f64, never F's own methods.
- Orientation always uses origin ids, never local or post-partition global ids.
- Tests: cargo test --workspace must pass on the CPU runtime.
  GPU runs: GRIDCL_TEST_DEVICE=vulkan | metal | cuda.
- MPI tests: crate gridcl-mpi-tests ([[test]] targets, harness = false),
  run via scripts/mpi-test.sh 1 2 3 4 under MPICH and Open MPI.
- ndelement is a dependency of conformance/ only (outside the workspace). Never add it to a
  library crate, and never add `version =` to its git dependency. After bumping its pin, run
  cargo test --manifest-path conformance/Cargo.toml and re-export the fixtures with
  cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures.
- Host<->device copies in library code only for I/O (including the upload of input arrays
  at mesh creation, and of quadrature points and tabulations when a rule is created or a
  tabulation is first cached), explicit to_host()/read_node(), the count read of run-length
  and unique-by-key, the per-block read that finishes measure in f64, the debug-build index
  checks of gather and scatter, and the host-staged halo exchange.
- FFI declarations only in *-sys crates. FFI calls only in gridcl-parallel (MPI) and
  gridcl-device (GPU-aware; the macOS performance-core count, ADR-0017), behind safe wrappers. Every unsafe block needs a SAFETY comment.
- Each milestone: acceptance tests first (failing), then implementation, unit tests,
  property tests and rustdoc examples.
- If an API deviation is unavoidable, stop and draft an ADR.
```

The repository's `CLAUDE.md` extends this listing with workflow rules (plans, ADRs, commits) and test-target rules (when to use the Myriad V100 and A100 nodes and Locust).

**Listing H2 — example milestone prompt**

```text
Implement milestone M3 ("Serial mesh core") from docs/design.md.
Scope: gridcl-topology, gridcl-geometry, the Mesh facade in gridcl, and the mesh
generators in gridcl-testing. Use the types from the "Architecture" section exactly, and
replace the M1a stub Mesh in the compile-fail tests.
Start by writing the M3 acceptance tests so that they fail: entity counts and Euler
characteristic, unit volume for p = 1..4 in f64 and f32, sphere-area convergence,
f32-vs-f64 agreement, mixed-surface Jacobians, permutation info against the fixtures, and
QuadField chunking. Then implement until they pass on the CPU runtime, and run them with
GRIDCL_TEST_DEVICE=vulkan and GRIDCL_TEST_DEVICE=metal.
If an API deviation is unavoidable, stop and draft an ADR.
```

## Appendix A — Reference cells and element tables

gridcl follows the DefElement conventions as implemented by ndelement on the main branch of nd-project/nd (commit f8cbffe, 1 October 2026; GitHub mirror bempp/nd). Vertex coordinates, and the sub-entity numbering of quadrilaterals, hexahedra, prisms and pyramids, are identical in ndelement, Basix, Symfem and DefElement. For triangles and tetrahedra, ndelement, Symfem and DefElement number edges and faces in sorted (lexicographic) order, while Basix and DOLFINx use the reverse order; A.11 gives the conversion.

**Version pin.** The released ndelement 0.4.0 on crates.io (13 April 2026) still uses the Basix order for simplices. Pin the nd git commit, not the crates.io release, until a release contains the change. The pin lives only in the conformance crate (A.10).

### A.1 Vertex coordinates

| Cell | Vertex coordinates (local index: reference coordinates) |
| --- | --- |
| Point | 0: () |
| Interval | 0: (0)  1: (1) |
| Triangle | 0: (0,0)  1: (1,0)  2: (0,1) |
| Quadrilateral | 0: (0,0)  1: (1,0)  2: (0,1)  3: (1,1) |
| Tetrahedron | 0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (0,0,1) |
| Hexahedron | 0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)  4: (0,0,1)  5: (1,0,1)  6: (0,1,1)  7: (1,1,1) |
| Prism | 0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (0,0,1)  4: (1,0,1)  5: (0,1,1) |
| Pyramid | 0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (1,1,0)  4: (0,0,1) |

Quadrilaterals and hexahedra use tensor (lexicographic) vertex order, so the boundary cycle of a quadrilateral is 0–1–3–2. Gmsh and VTK use counter-clockwise order (0–1–2–3) instead; their vertex maps are in A.9.

### A.2 Edges

**Rule for all eight cells (DefElement Convention 2):** sub-entities are numbered by their sorted vertex lists in lexicographic order. One generator therefore produces every table in A.2 and A.3, and M1b tests it against ndelement.

| Cell | Edges (local index: local vertex pair, oriented from first to second) |
| --- | --- |
| Interval | 0: (0,1) |
| Triangle | 0: (0,1)  1: (0,2)  2: (1,2)  [edge i is opposite vertex 2−i] |
| Quadrilateral | 0: (0,1)  1: (0,2)  2: (1,3)  3: (2,3) |
| Tetrahedron | 0: (0,1)  1: (0,2)  2: (0,3)  3: (1,2)  4: (1,3)  5: (2,3) |
| Hexahedron | 0: (0,1)  1: (0,2)  2: (0,4)  3: (1,3)  4: (1,5)  5: (2,3)  6: (2,6)  7: (3,7)  8: (4,5)  9: (4,6)  10: (5,7)  11: (6,7) |
| Prism | 0: (0,1)  1: (0,2)  2: (0,3)  3: (1,2)  4: (1,4)  5: (2,5)  6: (3,4)  7: (3,5)  8: (4,5) |
| Pyramid | 0: (0,1)  1: (0,2)  2: (0,4)  3: (1,3)  4: (1,4)  5: (2,3)  6: (2,4)  7: (3,4) |

### A.3 Faces of three-dimensional cells

| Cell | Faces (local index: local vertices, type) |
| --- | --- |
| Tetrahedron | 0: (0,1,2)  1: (0,1,3)  2: (0,2,3)  3: (1,2,3); all triangles; face i is opposite vertex 3−i |
| Hexahedron | 0: (0,1,2,3) z=0  1: (0,1,4,5) y=0  2: (0,2,4,6) x=0  3: (1,3,5,7) x=1  4: (2,3,6,7) y=1  5: (4,5,6,7) z=1; all quadrilaterals |
| Prism | 0: (0,1,2) triangle  1: (0,1,3,4) quad  2: (0,2,3,5) quad  3: (1,2,4,5) quad  4: (3,4,5) triangle |
| Pyramid | 0: (0,1,2,3) quad  1: (0,1,4) triangle  2: (0,2,4) triangle  3: (1,3,4) triangle  4: (2,3,4) triangle |

Quadrilateral faces list their vertices in the face's own tensor order, as for the quadrilateral cell. For two-dimensional cells the only face is the cell itself; for one-dimensional cells the facets are the two vertices.

### A.4 Lagrange node counts

| Cell | Nodes N(p) | Interior nodes | N(2) | N(3) |
| --- | --- | --- | --- | --- |
| Interval | p+1 | p−1 | 3 | 4 |
| Triangle | (p+1)(p+2)/2 | (p−1)(p−2)/2 | 6 | 10 |
| Quadrilateral | (p+1)^2 | (p−1)^2 | 9 | 16 |
| Tetrahedron | (p+1)(p+2)(p+3)/6 | (p−1)(p−2)(p−3)/6 | 10 | 20 |
| Hexahedron | (p+1)^3 | (p−1)^3 | 27 | 64 |
| Prism | (p+1)^2(p+2)/2 | (p−1)^2(p−2)/2 | 18 | 40 |
| Pyramid | (p+1)(p+2)(2p+3)/6 | (p−1)(p−2)(2p−3)/6 | 14 | 30 |

The pyramid count equals the dimension of DefElement's Lagrange pyramid space (A.5).

### A.5 Lagrange node ordering

- **Blocks:** all vertices in vertex order, then the interior nodes of each edge in edge order (A.2), then those of each face in face order (A.3), then the cell interior.
- **Inside an edge:** from the edge's first listed vertex towards its second.
- **Inside a triangle (face or cell):** with v0, v1, v2 the first three listed vertices, the points are v0 + (i0/p)(v1 − v0) + (i1/p)(v2 − v0), with i0, i1 ≥ 1 and i0 + i1 ≤ p−1. The loop has i1 outer and i0 inner: lexicographic, first face coordinate fastest. VTK orders the interiors of triangle cells and tetrahedron faces recursively instead (its wedge's triangular faces are lexicographic); the I/O maps handle this.
- **Inside a quadrilateral face:** the same construction on the tensor grid, first face coordinate fastest.
- **Tetrahedron interior:** v0 + (i0/p)(v1−v0) + (i1/p)(v2−v0) + (i2/p)(v3−v0), with all indices ≥ 1 and i0 + i1 + i2 ≤ p−1; i2 outermost, i0 innermost.
- **Hexahedron interior:** the tensor grid along the axes v1−v0, v2−v0, v4−v0, first coordinate fastest.
- **Prism interior** (gridcl rule, DefElement conventions): v0 + (i0/p)(v1−v0) + (i1/p)(v2−v0) + (i2/p)(v3−v0), with i0, i1 ≥ 1, i0 + i1 ≤ p−1 and 1 ≤ i2 ≤ p−1; i2 outermost, i0 innermost.
- **Pyramid interior** (gridcl rule): points (i0/p, i1/p, i2/p) with 1 ≤ i2 ≤ p−2 and 1 ≤ i0, i1 ≤ p−1−i2; i2 outermost, i0 innermost.
- **Pyramid space:** DefElement's Lagrange pyramid space (Definition 9 of the DefElement paper), spanned by x^a y^b z^c / (1−z)^min(a,b) with a + c ≤ p and b + c ≤ p. Tabulate it with an orthonormal rationomial basis.
- **Agreement:** the interval, triangle, quadrilateral, tetrahedron and hexahedron layouts are exactly ndelement's at f8cbffe. The prism and pyramid rules coincide with Basix's equispaced lattices, so DOLFINx's Gmsh maps for those cells apply unchanged.
- **Node families:** equispaced points on all cells. GLL points on intervals, quadrilaterals and hexahedra only, matching ndelement; GLL replaces the 1D equispaced points in the constructions above.

### A.6 Orientation and permutation bits

- **Reference orientation (low-to-high, as in ndelement, Basix and DOLFINx), applied to origin ids:** an edge runs from its lower to its higher origin id. A face starts at its lowest origin id and continues towards the lower of that vertex's two neighbours on the face.
- **Stored per cell in one u32 with ndelement's layout** (`ndelement::orientation::compute_orientation`). From the least significant bit:
    - one bit per edge in edge order, set when the edge's first listed vertex has the higher origin id;
    - then three bits per face in face order: the low two hold the rotation count, the third the reflection.
    - Cells of dimension 1 have no bits; cells of dimension 2 have edge bits only.
- **Rotation count:** the cyclic position of the face's lowest vertex within the cell-local face cycle (triangles: list order; quadrilaterals: tensor order, cycle 0–1–3–2). **Reflection bit:** set when that vertex's successor on the cycle has a higher origin id than its predecessor.
- **Unused bits:** at most 30 bits are needed (hexahedron), so bits 30 and 31 stay zero. ndelement returns an i32; gridcl stores the same bit pattern as u32.
- **Deliberate difference from DOLFINx:** gridcl also stores edge and face vertex lists in the reference orientation, so device kernels can orient an entity without a lookup. DOLFINx keeps orientation only in the bits.
- **Storage slots:** an edge is stored as (lower, higher). A triangle face is stored in ascending order, because both other vertices neighbour the lowest one. A quadrilateral face follows its tensor order (boundary cycle 0–1–3–2): slot 0 holds the lowest vertex, slot 1 its lower neighbour, slot 2 its other neighbour and slot 3 the opposite vertex. The stored face is again a valid tensor-ordered quadrilateral.

**Worked example — canonical storage and orientation bits** (numbers are origin ids)

```text
Quadrilateral face, origin ids in the cell's local tensor slots:
  slot 0 (0,0) = 17   slot 1 (1,0) = 4   slot 2 (0,1) = 9   slot 3 (1,1) = 12
Boundary cycle (slots 0-1-3-2):    17 - 4 - 12 - 9
Lowest vertex:                     4
Its neighbours on the cycle:       17 and 12 -> lower neighbour 12, other neighbour 17
Opposite vertex:                   9
Stored face (slots 0..3):          [4, 12, 17, 9]
Check: stored cycle 4 - 12 - 9 - 17 uses the same four edges as 17 - 4 - 12 - 9.
Face bits (ndelement): rotation = 1 (vertex 4 sits at cycle position 1),
                       reflection = 0 (successor 12 < predecessor 17) -> 0b001

Triangle face with origin ids (9, 30, 4) -> stored [4, 9, 30]
Edge with origin ids (30, 4)             -> stored [4, 30], edge bit = 1
```

| Cell | Bits used | Layout (bit indices, least significant first) |
| --- | --- | --- |
| Interval | 0 | – |
| Triangle | 3 | edges 0–2 |
| Quadrilateral | 4 | edges 0–3 |
| Tetrahedron | 18 | edges 0–5, faces 6–17 (4 × 3) |
| Hexahedron | 30 | edges 0–11, faces 12–29 (6 × 3) |
| Prism | 24 | edges 0–8, faces 9–23 (5 × 3) |
| Pyramid | 23 | edges 0–7, faces 8–22 (5 × 3) |

For face f, the field starts at b = (number of edges) + 3f: bits b and b+1 hold the rotation count, and bit b+2 the reflection.

### A.7 Gmsh element types

| Cell | Complete Lagrange: type id (nodes) | Serendipity |
| --- | --- | --- |
| Point | 15 (1) | – |
| Interval | 1 (2), 8 (3), 26 (4), 27 (5), 28 (6), 62–66 (7–11) | – |
| Triangle | 2 (3), 9 (6), 21 (10), 23 (15), 25 (21), 42–46 (28–66) | – |
| Quadrilateral | 3 (4), 10 (9), 36 (16), 37 (25), 38 (36), 47–51 (49–121) | 16 (8) |
| Tetrahedron | 4 (4), 11 (10), 29 (20), 30 (35), 31 (56), 71–75 (84–286) | – |
| Hexahedron | 5 (8), 12 (27), 92 (64), 93 (125), 94–98 (216–1000) | 17 (20) |
| Prism | 6 (6), 13 (18), 90 (40), 91 (75), 106–110 (126–550) | 18 (15) |
| Pyramid | 7 (5), 14 (14), 118–124 (30, 55, 91, 140, 204, 285, 385) | 19 (13) |

Serendipity types are upgraded to complete Lagrange on read. All other incomplete types are rejected, including Gmsh's incomplete triangles and the incomplete pyramids 125–131. Sources: meshio `gmsh/common.py` and Gmsh's `GmshDefines.h`.

### A.8 VTK cell types

| Cell | Linear | Fixed quadratic (nodes) | Lagrange | gridcl writer |
| --- | --- | --- | --- | --- |
| Point | 1 | – | – | 1 |
| Interval | 3 | 21 (3) | 68 | 3 for p = 1, else 68 |
| Triangle | 5 | 22 (6) | 69 | 5 for p = 1, else 69 |
| Quadrilateral | 9 | 23 (8), 28 (9) | 70 | 9 for p = 1, else 70 |
| Tetrahedron | 10 | 24 (10) | 71 | 10 for p = 1, else 71 |
| Hexahedron | 12 | 25 (20), 29 (27) | 72 | 12 for p = 1, else 72 |
| Prism | 13 | 26 (15), 32 (18) | 73 | 13 for p = 1, else 73 |
| Pyramid | 14 | 27 (13) | none usable | 14 for p = 1; for p ≥ 2 vertices only (14) with a warning, or a subdivided output mesh |

VTK's enum defines 74 (`VTK_LAGRANGE_PYRAMID`) only as a placeholder, "Not Implemented yet", and DOLFINx's writer refuses pyramids with more than five nodes. High-order pyramids therefore go to Gmsh files at full degree and to VTK only as linear or subdivided cells. VTU files are written with `version="2.2"`, because VTK renumbers higher-order hexahedra in files below version 2.1.

### A.9 Node maps between files and gridcl (p ≤ 2)

Entry k is the internal (DefElement-order) node index of the file's k-th node: `internal[map[k]] = file[k]`. The quadrilateral, hexahedron, prism and pyramid rows are DOLFINx's published maps (`io/cells.cpp`), which coincide with the internal order for these cells. The triangle and tetrahedron rows are DOLFINx's maps composed with the index conversion of A.11.

**Warning:** DOLFINx's arrays follow the direction stated here, but its docstring states the inverse (`a_dolfin[i] = a_vtk[p[i]]`). DOLFINx's triangle and tetrahedron arrays also use Basix numbering. For example, DOLFINx's Gmsh tri6 array is `0 1 2 5 3 4`, but the internal row is `0 1 2 3 5 4`. The coordinate round-trip test is the arbiter.

| Cell (nodes) | Gmsh → internal | VTK Lagrange → internal |
| --- | --- | --- |
| Interval (any) | identity | identity |
| Triangle (3) | 0 1 2 | 0 1 2 |
| Triangle (6) | 0 1 2 3 5 4 | 0 1 2 3 5 4 |
| Quadrilateral (4) | 0 1 3 2 | 0 1 3 2 |
| Quadrilateral (9) | 0 1 3 2 4 6 7 5 8 | 0 1 3 2 4 6 7 5 8 |
| Tetrahedron (4) | 0 1 2 3 | 0 1 2 3 |
| Tetrahedron (10) | 0 1 2 3 4 7 5 6 9 8 | 0 1 2 3 4 7 5 6 8 9 |
| Hexahedron (8) | 0 1 3 2 4 5 7 6 | 0 1 3 2 4 5 7 6 |
| Hexahedron (27) | 0 1 3 2 4 5 7 6 8 9 10 11 12 13 15 14 16 17 18 19 20 21 22 23 24 25 26 | 0 1 3 2 4 5 7 6 8 11 13 9 16 18 19 17 10 12 15 14 22 23 21 24 20 25 26 |
| Prism (6) | identity | identity |
| Prism (18) | identity | 0 1 2 3 4 5 6 9 7 12 14 13 8 10 11 15 17 16 |
| Pyramid (5) | 0 1 3 2 4 | 0 1 3 2 4 |
| Pyramid (14) | 0 1 3 2 4 5 6 7 8 9 10 12 11 13 | not written (A.8) |

Higher degrees are generated from each format's rules ("High-order node orderings"). As acceptance tests, the generator must reproduce DOLFINx's Gmsh P3 tables: the 20-node tetrahedron after the A.11 conversion, and the 16-node quadrilateral, 64-node hexahedron, 40-node prism and 30-node pyramid unchanged. VTK prisms of degrees 3, 5 and 6 are checked against VTK-written fixtures.

### A.10 Sources and conformance setup

**Sources:**

- ndelement main at f8cbffe (nd-project/nd on Codeberg; GitHub mirror bempp/nd): `ndelement/src/reference_cell.rs` (A.1–A.3), `ndelement/src/orientation.rs` (A.6), `ndelement/src/ciarlet/lagrange.rs` (A.5).
- Scroggs, Brubeck, Dean, Dokken and Marsden, "DefElement: an encyclopedia of finite element definitions" ([arXiv:2506.20188](https://arxiv.org/html/2506.20188); published 2026): Definition 8, Conventions 1–2, Definition 9.
- Basix `cpp/basix/cell.cpp` (Basix numbering for A.11), plus `lattice.cpp` and `e-lagrange.cpp` (interior lattices).
- DOLFINx `cpp/dolfinx/io/cells.cpp` (A.8, A.9) and `cpp/dolfinx/mesh/permutationcomputation.cpp` (A.11).
- Symfem `symfem/references.py` at a pinned commit, whose simplex numbering equals ndelement's.
- meshio `gmsh/common.py`; Gmsh `GmshDefines.h`; VTK `vtkCellType.h` and `vtkXMLUnstructuredGridReader.cxx`.
- ndmesh's crate-level example (`ndmesh/src/lib.rs`) still lists second-order triangle points in the old (Basix) edge order. Test against ndelement's Lagrange element, not against that example.

**Conformance setup.** gridcl writes its own definitions and follows the DefElement convention; ndelement is used only to check them, in a crate of its own.

- **gridcl-refcell** holds gridcl's constant tables, Lagrange generator and orientation-bit function. It depends only on gridcl-dims.
- **gridcl-conformance** lives in `conformance/`, is excluded from the main workspace, has its own Cargo.lock and is never published. It depends on gridcl-refcell by path, on ndelement by git at the pinned commit, and on BLAS/LAPACK source crates, because ndelement's own BLAS/LAPACK crates are dev-dependencies only.
- **Fixtures:** the conformance crate exports JSON fixtures to `gridcl-refcell/tests/fixtures/`, together with the pinned commits of ndelement, Basix, DOLFINx and Symfem. They are committed, so ordinary `cargo test --workspace` needs neither BLAS nor a git dependency.
- **CI:** a separate job runs the conformance suite nightly and on every ndelement pin bump. A failure means the convention moved, and calls for an explicit decision: follow the change (update gridcl and re-export the fixtures) or keep the old pin.
- **Scope of comparison:**
    - sub-entity tables and orientation bits for all eight cells;
    - Lagrange point sets for intervals, triangles, quadrilaterals, tetrahedra and hexahedra, since ndelement has no prism or pyramid element;
    - prisms and pyramids against DOLFINx's Gmsh P3 maps and Gmsh fixtures.

    M1b lists the acceptance tests.

**Listing A.1 — conformance crate**

```text
gridcl/                            main workspace
  Cargo.toml                       [workspace] members = [...], exclude = ["conformance"]
  gridcl-refcell/
    tests/fixtures/*.json          exported by gridcl-conformance, committed
  ...
  conformance/                     gridcl-conformance: own Cargo.lock, never published
    Cargo.toml
    src/lib.rs                     extern crate lines for BLAS/LAPACK
    src/bin/export_fixtures.rs     writes gridcl-refcell/tests/fixtures/
    tests/tables.rs                sub-entity tables, all eight cells
    tests/orientation.rs           bits vs ndelement::orientation::compute_orientation
    tests/lagrange.rs              point sets node by node, five cells, p <= 4
```

```toml
# conformance/Cargo.toml
[package]
name = "gridcl-conformance"
edition = "2024"
publish = false

[dependencies]
gridcl-refcell = { path = "../gridcl-refcell" }
serde_json = "1"

[dependencies.ndelement]
git = "https://codeberg.org/nd-project/nd"
rev = "f8cbffea7796b968f4cb6327e016e440380a2222"   # main, 1 October 2026; never add `version =`

[target.'cfg(target_os = "linux")'.dependencies]
openblas-src = { version = "0.10", features = ["system"] }
lapack-src = { version = "0.13", features = ["openblas"] }
blas-src = { version = "0.14", features = ["openblas"] }

[target.'cfg(target_os = "macos")'.dependencies]
blas-src = { version = "0.14", features = ["accelerate"] }
lapack-src = { version = "0.13", features = ["accelerate"] }
```

```rust
// conformance/src/lib.rs
extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
```

### A.11 Conversion to Basix and DOLFINx

| Entities | gridcl / ndelement index i → Basix / DOLFINx index |
| --- | --- |
| Triangle edges | 2 − i |
| Tetrahedron edges | 5 − i |
| Tetrahedron faces | 3 − i |
| All other cells and entities; all vertices | identity |
| Symfem and DefElement | identical to gridcl (no conversion) |

- **Node maps:** reorder the edge and face blocks of triangles and tetrahedra with the index conversion. The order within each block is unchanged, because each sub-entity keeps its vertex list and Basix places points inside an entity by the same rule as A.5.
- **Orientation bits:** DOLFINx stores faces first, with the reflection in the low bit, and defines its rotation count relative to both the local and the global minimum. Recompute DOLFINx bits from the vertex ids with DOLFINx's rule, rather than translating bit fields.
- **DOF transformations:** the base transformations of Scroggs, Dokken, Richardson and Wells (ACM TOMS 48(2), 2022, doi 10.1145/3524456) depend only on the sub-entity type, so they are identical in both numberings. Only the block of the cell transformation each one lands in moves.

## Change log from the October 2026 draft

All 44 findings of the consolidated design review are addressed. The table gives each finding's fix and the section that now contains it.

| Finding | Fix | Where |
| --- | --- | --- |
| BLK-1 | Kernel crates may use cubecl without runtime features; runtime types only in gridcl-device | Workspace layout; Listing H1 |
| BLK-2 | `Comm`, `Serial` and `IndexMap` in gridcl-parallel with MPI optional, built in M2 | Workspace layout; M2 |
| BLK-3 | `CellType` and `TopoDim` in gridcl-dims; M1a type layer with a stub `Mesh`; `AnyMesh` and `DeviceGrid` in M3 | Listing B2; M1a, M3 |
| BLK-4 | Prism and pyramid layouts defined from DefElement; ndelement point-set comparison limited to five cells | Reference cells; A.5; A.10 |
| MAJ-1 | Inclusive scan minus 1; one sort across blocks; emission-index payload; owned-first renumbering | Topology construction |
| MAJ-2 | Owners chosen among owned-cell holders; candidates need all vertices shared; numbers flow forward | Topology construction |
| MAJ-3 | Origin ids define orientation | Reference cells; A.6; pipeline stage 8 |
| MAJ-4 | Locality ordering (stage 7) before numbering (stage 8) | Parallel creation pipeline |
| MAJ-5 | Reverse scatter by sort and segmented reduce | Listing S2 |
| MAJ-6 | `QuadRule` per cell type; `BlockFields` per block | Listing B4b |
| MAJ-7 | `Real` with `widen`/`narrow`; no clash with `cubecl::prelude::Scalar` | Listing B3 |
| MAJ-8 | `CommDyn` defined; generic `CommExt` | Listing S2 |
| MAJ-9 | Accessors return `Err` until created; `DeviceBuffer` not `Clone` (`share`, `deep_clone`); `AnyMeshRef` | Listings B6, B7, S1 |
| MAJ-10 | `Comm` without `Send + Sync`; `Mpi` in an `Rc`; CI on MPICH and Open MPI | Listing S2; CI matrix |
| MAJ-11 | `pseudo_inverse_jacobians`, `facet_normals`; parity-propagation orientation; manifold-only scope (Q5) | Listing B4; Geometry |
| MAJ-12 | `calculate_cube_count_elemwise`; checked arithmetic | Listing B5; Device limits |
| MAJ-13 | u32-word sort keys; reduce-then-scan; no 64-bit atomics | Device primitives |
| MAJ-14 | Precision policy; f32 tests on every backend | Device model; Testing |
| MAJ-15 | Allocation checks; chunked `QuadField`; buffer-count rule | Listings S1, B4b; Device limits |
| MAJ-16 | Backend table; `metal` feature; Vulkan and Metal CI jobs | Supported backends; CI matrix; M0 |
| MAJ-17 | Gmsh content, tag-matching and D_GEO rules | Import and export |
| MAJ-18 | coupe removed; block partitioner plus vendored METIS as CI default (Q3) | Partitioning; Risks |
| MAJ-19 | ptscotch-sys, parmetis-sys and `mpi::ffi` wrappers in scope | Partitioning; MPI bindings; M6, M7 |
| MAJ-20 | Every acceptance test automated and measurable | Implementation roadmap |
| MAJ-21 | BLAS/LAPACK source crates and `extern crate` lines in the conformance crate | Listing A.1 |
| MIN-1 | One host thread per rank | Streams and threading |
| MIN-2 | E0599 codes; `boundary().boundary()` compiles | Listing B10 |
| MIN-3 | Markers derive `PartialEq`, `Eq`, `Hash` | Listing B1 |
| MIN-4 | `Sealed`, `Codim0` and `TopoDim` impls; `WrongCellDim`; minimal struct bounds; `Cell1`–`Cell3` naming | Listings B1–B3 |
| MIN-5 | `$crate` paths in `with_mesh!`; `tdim()` and `gdim()` methods | Listings B4, B6 |
| MIN-6 | `GeomLayout` derives; u32 runtime scalars | Listing B5 |
| MIN-7 | `num_owned`, `num_ghosts`, `num_local` | Listing S2 |
| MIN-8 | Column-major layout comments; `read_node` documents its cost | Listings B4, B9 |
| MIN-9 | gridcl-mpi-tests crate; explicit exceptions in CLAUDE.md | Workspace layout; Listing H1 |
| MIN-10 | `CudaAllocator::Sync`; `&mut` collectives noted | Transports |
| MIN-11 | MSRV declared only in CubeCL's workspace | Risks |
| MIN-12 | About 24 monomorphisations; tabulation wording; own LU; prism quadrature | Design rationale; Geometry |
| MIN-13 | hdf5-metno with `mpio` | Import and export; Workspace layout |
| MIN-14 | Map-direction warning | A.9; High-order node orderings |
| MIN-15 | VTU version 2.2 | Import and export; A.8 |
| MIN-16 | Gmsh pyramid ids 118–124 | A.7 |
| MIN-17 | "Gmsh P3 tables" named; VTK P3 prisms via fixtures | A.9; High-order node orderings |
| MIN-18 | Symfem pin; DefElement citation; no `version =` on the ndelement git dependency | A.10; Listing A.1 |
| MIN-19 | Python bindings removed | Summary; Requirements; Workspace layout; Roadmap |

**Other changes made while consolidating:**

- The DOF-transformation reference is now Scroggs, Dokken, Richardson and Wells (ACM TOMS 2022). The draft's "arXiv:2607.08172" could not be verified.
- Draft milestones M1 and M1b are now M1a (type layer) and M1b (reference cells), because gridcl-refcell builds on gridcl-dims.
- `Real` is defined in gridcl-device and re-exported by gridcl.
- GLL points are offered on tensor cells only, matching ndelement.
- Coordinates are stored relative to a global origin; `measure` returns f64.
- Edge submeshes of 3D meshes are supported.
- The new gridcl-testing crate provides the test-device selection and the mesh generators.

**Revision 2 (17 October 2026): development environment.**

- New milestone E0 (development environment), before M0.
- Development and test targets: the M3 Max (CPU runtime and Metal on every change), Locust (CUDA for routine runs, reviews and benchmarks), Kathleen (CPU-only runs, AVX-512 CPU benchmarks, two-node MPI), Myriad D nodes (CPU fallback), Myriad V100 and A100 (multi-GPU, comparisons, fallback), and hosted CI for Linux CPU, Vulkan and MPI.
- CUDA toolkit and driver rules for V100 and for the driver's PTX JIT.
- The remote-test runner `scripts/hpc/remote-test.sh` and the probe `tools/env-check/`.
- Updated: requirements, fixed decisions, supported backends, testing targets, workspace layout, CI matrix, the M0, M7, M8 and M10 acceptance tests, and the risks.

**Revision 3 (17 October 2026): SIMD vectorization.**

- Requirement: CPU performance through SIMD where benchmarks show a gain; fixed decision 11 (SIMD only through CubeCL vectors).
- New section "Vectorization and CPU execution": CubeCL's vector model, how the CPU runtime runs kernels, `LaunchPlan` and `VectorPolicy` (Listing S2), padding and tiles, the candidate list, determinism across vector sizes and the benchmark protocol.
- `DeviceBuffer` padding and `vector_arg` (Listing S1); the tiled `QuadField` (Listing B4b); the vectorized Jacobian kernel (Listing B5).
- New milestone M3s (SIMD vectorization and benchmarks) and the gridcl-bench crate; E-vectors move from M9 to M3; updated M2, M3, M6, M9 and M10, the testing section, the CI matrix, the working agreement and the risks.
