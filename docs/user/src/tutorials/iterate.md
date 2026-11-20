# Iterating over a mesh

A tetraloma mesh lives on the device, as flat arrays grouped into blocks. This tutorial walks
through those arrays: the cell blocks, the cells, the edges and faces, and the connectivity
between them. It reads every array back to the host and prints it. These reads are for
inspection and learning: each one copies data from the device and waits for it. A real program
leaves the arrays on the device and reads them in its own kernels, as
[Writing kernels](kernels.md) shows.

The program `iterate_cells` runs every step on the rectangle of
[Several cell types](build-mesh.md#several-cell-types): one quadrilateral and two triangles.

```sh
cargo run --release -p tetraloma-examples --example iterate_cells
```

## Blocks

The cells of a mesh are grouped into cell blocks, one per cell type. Within a block every cell
has the same number of vertices and nodes, so its arrays have a fixed stride and need no
offsets. `cell_blocks()` returns the blocks as
[`CellBlock`](../api/tetraloma/struct.CellBlock.html) values, and `geometry_block(b)` returns
the coordinate element and dofmap of block `b` as a
[`GeometryBlock`](../api/tetraloma/struct.GeometryBlock.html):

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:blocks}}
```

The block's [`IndexMap`](../api/tetraloma/struct.IndexMap.html) counts its cells. A
distributed mesh holds the cells it owns first and ghost cells, copies of its neighbours'
cells, after them. A serial mesh has no ghost cells, so `num_owned` and `num_local` agree.
[Ownership and ghosts](../concepts/meshes.md#ownership-and-ghosts) explains the two kinds.

The cell type of a block has the type `Cell2` in a mesh of dimension 2 and `Cell3` in one of
dimension 3. A `match` on it lists only the cell types that the dimension allows, so the
compiler checks that you handle each of them.

## Cells

A cell block stores one row per cell in each of its arrays. `vertices` holds the local vertex
indices in the order of the reference cell, and `origin_ids` holds each cell's origin id. The
geometry block's `dofmap` holds each cell's nodes, vertices first, as indices into the node
array `nodes()`. The node coordinates are relative to `origin()`, so the program adds it back:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:cells}}
```

The device stores an origin id as two 32-bit words, so that kernels compare ids without 64-bit
integers. The high word has its sign bit flipped, which keeps the order of negative ids. The
program decodes the pair with a small function:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:origin_id}}
```

For the rectangle the program prints the cells 100, 200 and 201 that the builder received.
The quadrilateral's vertices come out in tensor order: $(0, 0)$, $(1, 0)$, $(0, 1)$, $(1, 1)$.

## Entities

Edges and faces do not exist until you ask for them. `create_entities(d)` creates the entities
of dimension $d$ on the device, and the connectivity from cells to them. A face shared by two
cells of different types is found once. Entities of one dimension form [`EntityBlock`](../api/tetraloma/struct.EntityBlock.html)
values, one per entity type: a mesh of prisms, for example, has a triangle face block and a
quadrilateral face block. `entity_blocks(d)` returns them, and is an error until they exist:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:entities}}
```

Each entity stores its vertices in a canonical orientation: an edge runs from its vertex with
the smaller origin id to the other. Every process that holds the edge therefore agrees on its
direction, whatever the local numbering. [Orientation](../concepts/orientation.md) explains
the rule for faces. An entity's flat index is its block's `offset` plus its index in the
block; connectivity arrays use these flat indices. The rectangle has eight edges in one block
of intervals.

## Connectivity

Connectivity maps the entities of one dimension to those of another. A
[`ConnKey`](../api/tetraloma/struct.ConnKey.html) names the pair, and `connectivity(key)`
returns it as a [`Connectivity`](../api/tetraloma/enum.Connectivity.html). Downward
connectivity, from cells to their edges or faces, has a fixed number of entries per cell, so it
is one strided array per cell block. Upward connectivity, from entities to their cells, has
rows of different lengths, so it is one compressed sparse row array:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:connectivity}}
```

`create_entities(d)` makes the downward pair $(t, d)$, with $t$ the topological dimension, and
`create_connectivity(d, tdim)`
makes the upward one; both are collective on a distributed mesh. tetraloma stores only these
pairs. The vertices of a cell are in `CellBlock::vertices`, and those of an entity in
`EntityBlock::vertices`. In the rectangle six edges have one cell, the boundary, and two have
two cells.

## Reading data back to the host

Every device array has a method that copies it to the host.
[`DeviceBuffer::to_host`](../api/tetraloma/struct.DeviceBuffer.html) returns a buffer's values,
and `QuadField::to_host_cell_major` returns the values of a field at quadrature points in
$[\text{cell}][\text{point}]$ order. `read_node(i)` returns one node with the origin added back.
Each call waits for the device and copies the data, so use them for inspection, tests and
debugging, never inside a loop over the mesh:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/iterate_cells.rs:host}}
```

The snippet below reads the nodes of the unit square of
[From arrays](build-mesh.md#from-arrays) and checks their coordinates:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::unit_square;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = unit_square(&client)?;

// For inspection: all nodes at once, relative to the origin ...
let origin = mesh.origin();
let nodes = mesh.nodes().to_host()?;
assert_eq!(nodes.len(), 2 * 9);
for (i, x) in nodes.chunks_exact(2).enumerate() {
    // ... or one node at a time, with the origin added back.
    let y = mesh.read_node(i)?;
    assert!((x[0] + origin[0] - y[0]).abs() < 1e-12);
    assert!((x[1] + origin[1] - y[1]).abs() < 1e-12);
    // Every coordinate of the 2 × 2 grid is 0, 1/2 or 1.
    assert!(y.iter().all(|c| [0.0, 0.5, 1.0].contains(c)));
}
# Ok(())
# }
```

Writers and the measure read data back as part of their job; that is their output, not
inspection. Everything else that a program computes from the mesh, such as an assembled matrix
or an integral, is best computed by kernels on the device. [Compute devices](../devices.md)
says more about the cost of transfers.
