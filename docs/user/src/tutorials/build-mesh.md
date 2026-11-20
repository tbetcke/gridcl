# Building a mesh

A mesh starts as arrays on the host: the coordinates of its nodes, and for each cell the nodes
it uses. The [`MeshBuilder`](../api/tetraloma/struct.MeshBuilder.html) checks these arrays,
numbers the vertices and nodes, and uploads everything to the device. This tutorial builds three
meshes: a unit square, a rectangle with two cell types, and a curved quarter annulus. The
program `build_mesh` holds all three, and later tutorials reuse them. You run it with

```sh
cargo run --release -p tetraloma-examples --example build_mesh
```

The program and the snippets below open the CPU runtime. [Compute devices](../devices.md)
shows how to choose a GPU instead.

## From arrays

You describe a mesh with three arrays and one choice. The coordinates are a flat `f64` array
with `gdim` values per node. The cells are a flat `u32` array with one row of node indices per
cell. The choice is the coordinate element: the cell type, the polynomial degree of the
geometry, and the family of its nodes. The function `unit_square` of the program builds the
unit square $[0, 1]^2$ from four quadrilaterals of degree 1:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/build_mesh.rs:unit_square}}
```

Each row of `cells` lists a cell's nodes in tetraloma's order. For a quadrilateral this is the
tensor order $(0, 0)$, $(1, 0)$, $(0, 1)$, $(1, 1)$ of the
[reference square](../elements/reference-cells.md#quadrilateral), not the counter-clockwise
order of Gmsh and VTK. A row in counter-clockwise order describes a twisted cell, and its
determinant changes sign inside the cell.

The type of the result fixes the dimensions. `PlanarMesh<F>` is a mesh of topological
dimension 2 in two-dimensional space; `SurfaceMesh<F>`, `VolumeMesh<F>` and `CurveMesh<F>` are
the other common choices. [Typed dimensions](../concepts/typed-dimensions.md#the-mesh-type)
explains the type parameters. The scalar `F` is `f32` or `f64`, and the builder takes its
coordinates in `f64` either way.

The second argument of `nodes` and the third of `cell_block` are optional origin ids. An origin
id is a number that names a node or a cell independently of how the mesh is stored or
distributed. Without them, a node's id is its index in your array, and a cell's id is its index
over all cells in call order. [Origin ids](../concepts/meshes.md#origin-ids) explains what
tetraloma uses them for.

`build` makes a serial mesh. `build_on` takes a communicator and makes a distributed one, as
[Distributed meshes](distributed.md) shows. The snippet below builds two triangles of the unit
square and reads back what the builder stored:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, Some(&[10, 11, 12, 13]))
    .cell_block(triangle, &[0, 1, 3, 0, 3, 2], None)
    .build()?;

assert_eq!(mesh.cell_blocks()[0].index_map.num_owned(), 2);
assert_eq!(mesh.vertex_index_map().num_local(), 4);
assert!((mesh.measure(1)? - 1.0).abs() < 1e-12);
// Coordinates are stored relative to the centre of the bounding box.
assert_eq!(mesh.origin(), [0.5, 0.5]);
# Ok(())
# }
```

The device stores coordinates relative to the centre of the bounding box, which `origin()`
returns in `f64`. This keeps the rounding error of `f32` meshes small far from the coordinate
origin. Every method that returns physical coordinates adds the origin back.

## Several cell types

A mesh may mix cell types of the same dimension: triangles with quadrilaterals, or tetrahedra
with hexahedra, prisms and pyramids. You add one `cell_block` per cell type. The function
`two_blocks` builds the rectangle $[0, 2] \times [0, 1]$ from one quadrilateral and two
triangles, with origin ids for its nodes and cells:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/build_mesh.rs:two_blocks}}
```

The mesh keeps one cell block per cell type, in the order of the `cell_block` calls. A second
call with a cell type that already has a block is an error. Every block may have its own
degree, so a mesh of curved triangles may hold straight quadrilaterals. The triangles above
list their vertices counter-clockwise, which gives them a positive determinant.

The snippet below checks the blocks of `two_blocks`. It calls the function from the crate
`tetraloma-examples`, where the program defines it:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::two_blocks;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = two_blocks(&client)?;
let blocks: Vec<(Cell2, usize)> = mesh
    .cell_blocks()
    .iter()
    .map(|b| (b.cell, b.index_map.num_owned()))
    .collect();
assert_eq!(blocks, [(Cell2::Quadrilateral, 1), (Cell2::Triangle, 2)]);
assert!((mesh.measure(1)? - 2.0).abs() < 1e-12);
# Ok(())
# }
```

A cell may also name its nodes by origin id instead of by index, with `cell_block_by_ids`.
This is the natural form when several processes each hold part of the input, because an index
into one process's array means nothing to another.

## Higher-degree geometry

A coordinate element of degree $p$ describes each cell by a polynomial map of degree $p$ from
the reference cell. A quadrilateral of degree $p$ has $(p + 1)^2$ nodes. Its row in the cells
array lists the vertices first, then the nodes inside each edge, then the interior nodes, as
[Lagrange nodes](../elements/lagrange.md#node-ordering) defines. You rarely write these rows by
hand. The function
[`lagrange_points`](../api/tetraloma_refcell/fn.lagrange_points.html) of the crate
`tetraloma-refcell` returns the reference coordinates of every node in this order, and you map
each one to its physical position.

The function `quarter_annulus` builds the quarter annulus $1 \le r \le 2$,
$0 \le \theta \le \pi/2$. It lays a grid of nodes over $(r, \theta)$ and places each node on
the exact polar map $(x, y) = (r \cos\theta, r \sin\theta)$. Each cell then finds its nodes in
the grid from their reference coordinates:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/build_mesh.rs:quarter_annulus}}
```

Between the nodes, each cell follows a polynomial, so the mesh only approximates the circular
arcs. The error shrinks quickly with the degree. The snippet below builds the annulus at
degree 2 from one cell in $r$ and two in $\theta$, and compares its area with the exact
$3\pi/4$:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::quarter_annulus;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = quarter_annulus(&client, 2, 1, 2)?;
// A grid of 3 × 5 nodes, each cell of degree 2 with 9 of them.
assert_eq!(mesh.node_index_map().num_local(), 15);
assert_eq!(mesh.geometry_block(0).element.degree, 2);

let exact = 3.0 * std::f64::consts::PI / 4.0;
let error = (mesh.measure(4)? - exact).abs() / exact;
assert!(error < 1e-3, "relative error {error}");
# Ok(())
# }
```

[Quadrature on a mesh](quadrature.md#high-order-geometry) follows this error up to degree 8.
Equispaced nodes exist on every cell type. Gauss–Lobatto–Legendre nodes
(`NodeFamily::Gll`) exist on intervals, quadrilaterals and hexahedra only, and suit degrees
above about 10; [GLL and equispaced nodes](../concepts/geometry.md#gll-and-equispaced-nodes)
compares the two.

## Errors

`build` checks its input before it uploads anything, and returns a
[`GridError`](../api/tetraloma/enum.GridError.html) that names the problem. The most common
ones are a coordinate array whose length is not a multiple of `gdim`, a node index beyond the
nodes, a repeated origin id, a second block of one cell type, and a node family that the cell
type does not offer. On a device without `f64` support, an `f64` mesh is
`UnsupportedPrecision`; use `f32` there. The snippet below provokes four of these errors:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];

// Five coordinates are not a whole number of 2D nodes.
let r = MeshBuilder::<f64, D2, D2>::new(&client)
    .nodes(&coords[..5], None)
    .cell_block(triangle, &[0, 1, 2], None)
    .build();
assert!(matches!(r, Err(GridError::LengthMismatch { expected: 6, found: 5 })));

// Node 3 does not exist.
let r = MeshBuilder::<f64, D2, D2>::new(&client)
    .nodes(&coords, None)
    .cell_block(triangle, &[0, 1, 3], None)
    .build();
assert!(matches!(r, Err(GridError::IndexOutOfRange { index: 3, len: 3 })));

// Two nodes with the origin id 7.
let r = MeshBuilder::<f64, D2, D2>::new(&client)
    .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], Some(&[7, 8, 9, 7]))
    .cell_block(triangle, &[0, 1, 2], None)
    .build();
assert!(matches!(r, Err(GridError::DuplicateOriginId(7))));

// GLL nodes exist on intervals, quadrilaterals and hexahedra only.
let gll = CoordinateElement::<D2> { family: NodeFamily::Gll, ..triangle };
let r = MeshBuilder::<f64, D2, D2>::new(&client)
    .nodes(&coords, None)
    .cell_block(gll, &[0, 1, 2], None)
    .build();
assert!(matches!(r, Err(GridError::UnsupportedNodeFamily { cell: "Triangle", .. })));
# Ok(())
# }
```

Every error implements `std::error::Error` and prints a sentence that names the problem, so the
`?` operator passes it on to your own error type.
