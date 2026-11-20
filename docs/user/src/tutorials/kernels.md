# Writing kernels

tetraloma keeps a mesh on the device, and a library built on it, such as a finite element or
boundary element code, computes on the device too. It writes its own CubeCL kernels and gives
them the mesh's arrays directly, with no copy to the host. This tutorial shows the arrays a
kernel can read: the views of the trait `DeviceGrid`, the dofmap of a `GeometryBlock` and the
fields of a `BlockFields`. It then writes a kernel that integrates a function over a mesh,
wraps it in functions that work on every mesh type, and calls them on meshes whose dimensions
are known only at run time.

The program `custom_kernel` holds all of it, and checks its integrals against `measure` and
against exact values:

```sh
cargo run --release -p tetraloma-examples --example custom_kernel
```

Your crate writes kernels with CubeCL's `#[cube]` macro, so it depends on `cubecl` at the exact
version that tetraloma uses, as [Installation](../getting-started/installation.md) describes.
The scalar type `F` of a mesh implements [`Real`](../api/tetraloma/trait.Real.html), which
includes CubeCL's `Float`, so a kernel generic over `F: Float` accepts every mesh.

## DeviceGrid

The trait [`DeviceGrid`](../api/tetraloma/trait.DeviceGrid.html) is what a downstream crate
programs against. [`Mesh`](../api/tetraloma/struct.Mesh.html) implements it, and so does every
submesh, because a submesh is a `Mesh` too. Its associated types `F`, `TD` and `GD` are the
scalar and the two dimensions, and its methods return views of device arrays:

- `client()` and `comm()`: the device and the communicator of the grid;
- `cell_blocks()`: the cell blocks, with their cell type, index map and vertices;
- `geometry_block(b)`: the coordinate element and dofmap of block `b`;
- `nodes()` and `origin()`: the node coordinates, relative to the origin, and the origin;
- `node_index_map()`, `vertex_index_map()` and `vertex_origin_ids()`: the owned and ghost nodes
  and vertices, and the vertices' origin ids;
- `entity_blocks(dim)`, `connectivity(key)` and `permutation_info(b)`: the topology of
  [Iterating over a mesh](iterate.md).

The accessors build no topology. Entities and connectivity exist only after `create_entities`
and `create_connectivity` have run, and the accessors return an error before that. A function with
the bound `G: DeviceGrid` works on every mesh and submesh. The snippet below counts the owned
cells of the rectangle of [Several cell types](build-mesh.md#several-cell-types) and of its
boundary, a curve in the plane:

```rust
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::two_blocks;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

// Works on every grid, whatever its dimensions.
fn owned_cells<G: DeviceGrid>(grid: &G) -> usize {
    grid.cell_blocks().iter().map(|block| block.index_map.num_owned()).sum()
}

let client = cubecl::Device::cpu()?.client();
let mut mesh: PlanarMesh<f64> = two_blocks(&client)?;
assert_eq!(owned_cells(&mesh), 3);
// Six nodes of two coordinates, relative to the centre (1, 1/2) of the bounding box.
assert_eq!(mesh.nodes().len(), 2 * 6);
assert_eq!(mesh.origin(), [1.0, 0.5]);
assert_eq!(mesh.node_index_map().num_local(), 6);

// The boundary needs the edges first.
mesh.create_entities(1)?;
let (boundary, _) = mesh.boundary()?;
assert_eq!(owned_cells(&boundary), 6);
# Ok(())
# }
```

## GeometryBlock

A [`GeometryBlock`](../api/tetraloma/struct.GeometryBlock.html) describes the geometry of one
cell block. Its field `element` is the
[`CoordinateElement`](../api/tetraloma/struct.CoordinateElement.html): cell type, degree and
node family. Its field `dofmap` is a
[`DeviceStrided`](../api/tetraloma/struct.DeviceStrided.html) of `u32`: one row of
`nodes_per_cell()` node indices per cell, in the node order of
[Lagrange nodes](../elements/lagrange.md). With the basis functions $\phi_k$ of the element,
the map of cell $c$ is

$$
X_c(\xi) = o + \sum_k \phi_k(\xi) \, x_{d(c, k)},
$$

where $o$ is the origin, $d(c, k)$ the $k$-th entry of row $c$ of the dofmap, and $x_n$ the
coordinates of node $n$ in `nodes()`. A kernel needs the values $\phi_k(\xi_q)$ at the points of
a rule. `PointSet::tabulate(&element)` computes them once, keeps them on the device, and
returns them as a buffer in $[q][k]$ order. The snippet below evaluates $X_c(\xi_q)$ for the
two triangles of the rectangle on the host, as the kernel will on the device, and compares the
result with `physical_points`:

```rust
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::two_blocks;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = two_blocks(&client)?;
let rule = QuadRule::for_grid(&mesh, 2)?;

let b = 1; // the block of triangles
let geometry = mesh.geometry_block(b);
assert_eq!(geometry.element.cell, Cell2::Triangle);
assert_eq!((geometry.num_cells(), geometry.nodes_per_cell()), (2, 3));
assert_eq!(geometry.dofmap.stride, 3);

// For inspection: the arrays a kernel reads, copied to the host.
let points = rule.get(Cell2::Triangle)?;
let phi = points.tabulate(&geometry.element)?.to_host()?; // [q][k]
let dofmap = geometry.dofmap.data.to_host()?; // [cell][k]
let nodes = mesh.nodes().to_host()?; // [node][g], relative to the origin
let origin = mesh.origin();
let x = mesh.physical_points(&rule)?.blocks[b].to_host_cell_major()?;

let (npc, nq) = (geometry.nodes_per_cell(), points.len());
for c in 0..geometry.num_cells() {
    for q in 0..nq {
        for g in 0..2 {
            let mut y = origin[g];
            for k in 0..npc {
                let node = dofmap[c * npc + k] as usize;
                y += phi[q * npc + k] * nodes[node * 2 + g];
            }
            assert!((y - x[c * nq + q][g]).abs() < 1e-12);
        }
    }
}
# Ok(())
# }
```

The node coordinates are stored relative to the origin, the centre of the mesh's bounding box,
so that they keep their precision in `f32` far from zero. A kernel that needs absolute
coordinates adds the origin back, as the formula does.

## BlockFields

The geometric factors of [Geometric data](geometric-data.md) come as a
[`BlockFields`](../api/tetraloma/struct.BlockFields.html): its field `blocks` holds one
[`QuadField`](../api/tetraloma/struct.QuadField.html) per cell block. Inside a generic function,
[`detj_block`](../api/tetraloma/fn.detj_block.html) (`tetraloma::detj_block`) computes the
field of one block on any `DeviceGrid`. A field
stores its values in tiles of `tile()` consecutive cells, so that neighbouring cells sit next to
each other in memory. Component $k$ of point $q$ of cell $c$, counted from the first cell of
its chunk, is at

$$
\bigl( \lfloor c / w \rfloor \cdot n_q \cdot n_k + q \cdot n_k + k \bigr) \cdot w + (c \bmod w),
$$

with the tile width $w$, $n_q$ points and $n_k$ components. A scalar field such as
$\det J$ has $n_k = 1$. The tile width is 1, 2, 4, 8 or 16, set by the device's policy, so a
kernel takes it as a parameter. A large field is split into chunks, because one allocation may
not exceed the device's limit. Each [`QuadChunk`](../api/tetraloma/struct.QuadChunk.html) holds
`num_cells` cells from `first_cell` on, in a device buffer `data`. The snippet below finds the
values of $\det J$ of the triangles in the raw chunks:

```rust
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::two_blocks;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = two_blocks(&client)?;
let rule = QuadRule::for_grid(&mesh, 2)?;
let detj = mesh.detj(&rule)?;

let field = &detj.blocks[1]; // the triangles
let (tile, nq) = (field.tile(), field.num_points());
let cell_major = field.to_host_cell_major()?;
for chunk in field.chunks() {
    // For inspection: the tiled values of one chunk.
    let raw = chunk.data.to_host()?;
    for c in 0..chunk.num_cells {
        for q in 0..nq {
            let tiled = raw[((c / tile) * nq + q) * tile + c % tile];
            assert_eq!(tiled, cell_major[(chunk.first_cell + c) * nq + q]);
        }
    }
}
// Both triangles have area 1/2, twice the reference triangle's: det J = 1.
assert!(cell_major.iter().all(|d| (d - 1.0).abs() < 1e-12));
# Ok(())
# }
```

The quadrature weights are the last ingredient. A
[`PointSet`](../api/tetraloma/struct.PointSet.html) returns them on the host, in `f64`, with
`weights()`. A kernel needs them on the device, so your code narrows them to `F` and uploads
them once with `DeviceBuffer::from_host`.

## A kernel that integrates

The integral of a function $f$ over the owned cells of a mesh is

$$
\int_\Omega f \, dx \approx \sum_c \sum_q w_q \, \det J_c(\xi_q) \, f\bigl(X_c(\xi_q)\bigr).
$$

The kernel below computes it for $f = 1$ and $f = |x|^2$ at once. Each unit of the launch takes
whole cells. For each cell it sums over the points in a fixed order and writes the cell's two
integrals to its own two entries of `sums`. It binds six buffers: the nodes, the dofmap, the
basis values, one chunk of $\det J$, a small buffer of constants (the weights, then the origin)
and the output. The layout values `gdim`, `nodes_per_cell`, `num_points` and `tile` are
comptime parameters, so CubeCL compiles one kernel for each combination and knows the loop
bounds:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/custom_kernel.rs:kernel}}
```

The kernel sits in a module of its own, because the glob imports of CubeCL's prelude and of
tetraloma's prelude share some names.

### The launch shape

A launch needs a cube count and a cube dimension. tetraloma's own kernels take theirs from
launch plans that the `tetraloma` crate does not export, so a downstream kernel chooses its own
shape from the device's properties. The function below uses cubes of up to 64 units. On a GPU
each unit integrates one cell. The CPU runtime reports its core count and runs every unit of a
cube as a worker thread, so there each unit integrates a contiguous run of cells. One cell per
unit would make the threads share cache lines.

```rust,ignore
{{#include ../../../../tetraloma-examples/src/custom_kernel.rs:launch_shape}}
```

### The launch

The host function gets everything from the grid. It makes a rule with
`QuadRule::for_grid`, then, per cell block, tabulates the basis, computes $\det J$ with
`detj_block` and uploads the constants. It launches the kernel once per chunk of the
$\det J$ field, and reads the per-cell integrals back once per block:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/custom_kernel.rs:integrate}}
```

`launch_unchecked` and `DeviceBuffer::arg` are `unsafe`: the kernel does not check its indices,
so you must show that every read and write stays inside its buffer, as the `SAFETY` comment
does. Buffers are padded to a multiple of 16 values, and a field's chunk to whole tiles, so
reads up to the end of the last tile are allowed.

### Rules for downstream kernels

The kernel follows the rules that tetraloma's own kernels follow. They keep a kernel correct on
every backend and its results reproducible:

- **No float atomics.** They are missing on some backends, and their order of additions changes
  from run to run. Each unit writes only its own cells, and the sums run in a fixed order: the
  points of a cell in the kernel, then the cells in `f64` on the host. A large program would
  sum on the device instead, with a second kernel that adds fixed blocks of values, as
  `measure` does.
- **Any plane size.** The kernel uses no plane operations. If yours does, it must be correct
  for every plane size, including the size 1 of the CPU runtime.
- **No forward progress between cubes.** One cube must never wait for another, so a kernel
  cannot pass partial sums from cube to cube.
- **At most six buffers.** Some backends bind only a few buffers per kernel. Pack small inputs
  into one buffer, as the constants are here.
- **Owned cells only.** A distributed mesh holds ghost cells after its owned cells. Integrate
  the owned cells and sum over the ranks with `comm().sum`, so that each cell counts once.

The kernel computes at vector size 1. Since a field's tiles keep consecutive cells next to each
other, a kernel can also read several cells at once with CubeCL's `Vector<F, N>`. Each lane is
then one cell, and every lane must do the same operations.

The program checks the kernel on three meshes. On the rectangle $[0, 2] \times [0, 1]$, one
quadrilateral and two triangles, it finds $\int 1 = 2$ and $\int |x|^2 = 10/3$. On the box
$[0, 1] \times [0, 1] \times [0, 2]$, two hexahedra, it finds $2$ and $4$. On the quarter
annulus $1 \le r \le 2$ of degree 4 the exact values are $3\pi/4$ and $15\pi/8$; the cells only
approximate the circles, which leaves a relative error of about $2 \times 10^{-6}$. On every
mesh, $\int 1$ agrees with `measure` to rounding, because both use the same rule.

## Generic algorithms

`integrate` has the bound `G: DeviceGrid`, so it works on every mesh and submesh. The snippet
below applies it to the unit square of [From arrays](build-mesh.md#from-arrays) and to its
boundary, a closed curve of length 4:

```rust
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::unit_square;
use tetraloma_examples::custom_kernel::integrate;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let client = cubecl::Device::cpu()?.client();
let mut square: PlanarMesh<f64> = unit_square(&client)?;
// ∫ 1 = 1 and ∫ x² + y² = 2/3 over the square.
let [area, r2] = integrate(&square, 2)?;
assert!((area - 1.0).abs() < 1e-12 && (r2 - 2.0 / 3.0).abs() < 1e-12);

square.create_entities(1)?;
let (boundary, _) = square.boundary()?;
// ∫ 1 = 4 and ∫ x² + y² = 1/3 + 4/3 + 1/3 + 4/3 over the four sides.
let [length, r2] = integrate(&boundary, 2)?;
assert!((length - 4.0).abs() < 1e-12 && (r2 - 10.0 / 3.0).abs() < 1e-12);
# Ok(())
# }
```

Some functions need methods that only `Mesh` has, such as `measure`, `physical_points` or the
writers. They take a `&Mesh<F, TD, GD, C>` and state the bounds that the mesh's own methods
have: `F: Real`, `TD: TopoDim + EmbeddedIn<GD>`, `GD: Dim` and `C: Comm`. The program's
`integrals` is such a function. It calls `integrate`, compares $\int 1$ with `measure` and
prints both:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/custom_kernel.rs:generic}}
```

A function that needs a method of one codimension states it in its bounds, as
[Typed dimensions](../concepts/typed-dimensions.md#codimension) explains. `normals` exists only
on meshes of codimension 1, so a function that calls it requires `TD: Codim1<GD>`, and the
compiler rejects a call with a planar mesh:

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

// The number of normals at the points of a rule of degree 1: only for surfaces in 3D and
// curves in 2D.
fn count_normals<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> tetraloma::Result<usize>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let rule = QuadRule::for_grid(mesh, 1)?;
    let normals = mesh.normals(&rule)?;
    Ok(normals.blocks.iter().map(|f| f.num_cells() * f.num_points()).sum())
}

let client = cubecl::Device::cpu()?.client();
// One triangle in 3D; a rule of degree 1 has one point on it.
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let surface: SurfaceMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0], None)
    .cell_block(triangle, &[0, 1, 2], None)
    .build()?;
assert_eq!(count_normals(&surface)?, 1);
# Ok(())
# }
```

Prefer the bound `G: DeviceGrid` where the views suffice. Such a function also works on grid
types that implement the trait later, while a function over `Mesh` works only on meshes.

## with_mesh!

A file reader returns an [`AnyMesh`](../api/tetraloma/enum.AnyMesh.html), because the
dimensions of the mesh in a file are known only when it is read. The macro
[`with_mesh!`](../api/tetraloma/macro.with_mesh.html) turns it back into typed meshes: it
expands to a `match` with one arm per variant, and each arm runs the same expression with a
typed `&Mesh`. The expression must compile for all six mesh types, so it calls a generic
function such as `integrals`. The program keeps its three meshes as `AnyMesh` values and
integrates each one like this:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/custom_kernel.rs:with_mesh}}
```

The compiler builds `integrals` and `integrate` once for each of the six mesh types. CubeCL
compiles the kernel at run time, once for each combination of its comptime parameters that a
launch uses: here one for the quadrilaterals and one for the triangles of the rectangle, one
for the curved quadrilaterals of degree 4, and one for the hexahedra. If you expect one kind of
mesh only, convert with `TryFrom` instead, as
[Typed dimensions](../concepts/typed-dimensions.md#anymesh) shows. A mesh of other dimensions
then gives `GridError::DimensionMismatch`.
