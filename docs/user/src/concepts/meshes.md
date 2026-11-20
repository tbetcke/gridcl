# Meshes and topology

A mesh splits a domain into simple pieces called cells and records how those pieces touch.
This chapter describes what tetraloma stores about a mesh, before any coordinates come in.
The geometry, that is where the cells lie in space, is the subject of the chapters that
follow, starting with [Reference cells and maps](reference-cells.md).

## Cells and entities

A cell is one piece of the mesh: an interval, a triangle, a quadrilateral, a tetrahedron, a
hexahedron, a prism or a pyramid. Its topological dimension $t$ is 1, 2 or 3. A mesh whose
cells have dimension $t$ is a mesh of topological dimension $t$, and every cell of it has the
same dimension. The geometric dimension $g$ is the number of coordinates of a point in space,
with $t \le g \le 3$. A triangulated sphere has $t = 2$ and $g = 3$.

The pieces of a cell's boundary are its entities, sorted by dimension:

- dimension 0: the vertices, the corners of the cells;
- dimension 1: the edges;
- dimension 2: the faces, which exist as sub-entities only in three-dimensional cells;
- dimension $t$: the cells themselves.

The entities of dimension $t - 1$ are called facets. The facets of a tetrahedron are its four
triangles, those of a triangle its three edges, and those of an interval its two vertices.

Vertices are different from nodes. A vertex is a corner of a cell, and it belongs to the
topology. A node is a point that carries a coordinate, and it belongs to the geometry. A
straight-sided cell has one node at each vertex. A curved cell of degree 2 or more has further
nodes on its edges, faces or interior, which are not vertices (see
[Lagrange geometry](geometry.md)).

A mesh holds its cells and vertices from the moment you build it. Edges and faces are created
only when you ask for them, with `create_entities(dim)` on
[`Mesh`](../api/tetraloma/struct.Mesh.html). Entity creation runs on the device: every cell
emits its sub-entities as sorted lists of vertex indices, one sort finds equal lists, and each
group of equal lists becomes one entity. The result does not depend on how many processes
share the mesh.

## Connectivity

Connectivity says which entities of one dimension touch which entities of another. tetraloma
stores two kinds, named by a pair of dimensions in a
[`ConnKey`](../api/tetraloma/struct.ConnKey.html):

- downward, $(t, d)$ with $d < t$: for every cell, its entities of dimension $d$ in the local
  order of the reference cell. Every cell of a block has the same number of them, so this is a
  table with a fixed row length.
- upward, $(d, t)$: for every entity of dimension $d$, the cells that contain it. Rows have
  different lengths (an interior facet has two cells, a boundary facet one), so they are stored
  in compressed sparse row form: an offsets array and an indices array.

`create_entities(d)` builds the downward connectivity $(t, d)$ as a by-product.
`create_connectivity(d, t)` builds the upward one, after creating the entities it needs. The
rows of the upward connectivity list their cells in ascending order, so the result is the same
on every run. The vertices of an entity and the vertices of a cell need no extra connectivity:
they are the `vertices` array of the entity block or of the cell block.

Entities of one dimension are numbered by one flat index across all their blocks: the first
index of the block plus the position inside it. Connectivity arrays hold these flat indices.
The tutorial [Iterating over a mesh](../tutorials/iterate.md#connectivity) shows how to read
them.

## Cell blocks

A mesh may mix cell types, for example hexahedra with prisms, or triangles with
quadrilaterals. tetraloma keeps the cells of one type together in a cell block
([`CellBlock`](../api/tetraloma/struct.CellBlock.html)), so that every array of a block has a
fixed row length and a device kernel can treat all its cells alike. The cell type is known per
block, at run time; the topological dimension is part of the mesh's type (see
[Typed dimensions](typed-dimensions.md)).

Each cell block holds the cell type, an index map of its cells, the vertices of each cell in
the vertex order of the reference cell, the origin id of each cell and one word of flags per
cell. Each cell block also has a geometry block, which holds the coordinate element of its
cells and the indices of their nodes in the mesh's node array.

Entities are grouped the same way. A mesh of prisms has two face blocks after
`create_entities(2)`: one of triangles and one of quadrilaterals. A face shared by a hexahedron
and a prism is still found once, because all cells emit their faces into one sort.

The example below builds the unit square from two triangles, creates its edges and reads the
results back to the host. The host reads (`to_host`) are there so that you can look at the
data; a program that computes on the mesh leaves it on the device.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// The unit square as two P1 triangles; the nodes get the origin ids 10 to 13.
let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mut mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, Some(&[10, 11, 12, 13]))
    .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    .build()?;

// One block of two triangles.
assert_eq!(mesh.cell_blocks().len(), 1);
assert_eq!(mesh.cell_blocks()[0].cell, Cell2::Triangle);
assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 2);

// Five edges: four on the boundary and the shared diagonal.
mesh.create_entities(1)?;
let edges = &mesh.entity_blocks(1)?[0];
assert_eq!(edges.cell, CellType::Interval);
assert_eq!(edges.index_map.num_local(), 5);

// Downward connectivity (2, 1): three edges per triangle, one of them shared.
let Connectivity::Downward(rows) = mesh.connectivity(ConnKey { from: 2, to: 1 })? else {
    unreachable!("cell -> edge is downward")
};
assert_eq!(rows[0].stride, 3);
let cell_edges = rows[0].data.to_host()?;
let shared: Vec<u32> = cell_edges[0..3]
    .iter()
    .copied()
    .filter(|e| cell_edges[3..6].contains(e))
    .collect();
assert_eq!(shared.len(), 1);
# Ok(()) }
```

## Ownership and ghosts

A distributed mesh is split over several processes (MPI ranks). Every cell belongs to exactly
one rank, its owner, so the owned cells of all ranks form the mesh without overlap. Besides its
owned cells, a rank may hold a layer of ghost cells: copies of cells owned by other ranks that
touch its own. You choose the layer with
[`GhostMode`](../api/tetraloma/enum.GhostMode.html): no ghosts, the cells that share a facet
with an owned cell, or a number of layers of cells that share a vertex.

Vertices, nodes, edges and faces have owners too. An entity on the boundary between ranks
exists on each of them, but only one rank owns it; on every other rank it is a ghost. The owner
is chosen among the ranks that hold the entity in an owned cell, by a hash of the entity's
origin ids. The choice therefore does not depend on how the ranks number their data locally.

Every set of distributed objects (each cell block, each entity block, the vertices and the
nodes) carries an index map ([`IndexMap`](../api/tetraloma/struct.IndexMap.html)). It numbers
the local objects as $[\text{owned} \mid \text{ghost}]$: the owned ones first, then the ghosts.
On the device, local indices are 32-bit unsigned integers. Across ranks, the owned objects of
rank $r$ take the global indices of a contiguous range, and each ghost knows the global index
and the rank of its owner. A halo exchange copies values from owners to ghosts, or adds the
ghosts' contributions back to the owners (see
[Halo exchange](../tutorials/halo-exchange.md#index-maps)).

A serial mesh uses the same structures with the one-rank communicator
[`Serial`](../api/tetraloma/struct.Serial.html). Its index maps have no ghosts, and every local
index is also the global index.

## Origin ids

Local indices change when you distribute a mesh differently, so they cannot identify a vertex
across runs. Every vertex therefore also has an origin id: a 64-bit signed integer that does not
depend on the partition. It is the node tag of a Gmsh file, the id you pass to the mesh
builder, or the node's position in the input if you pass none. Origin ids must be unique;
the builder and the readers return `GridError::DuplicateOriginId` otherwise.

Cells carry origin ids too: the Gmsh element tag, your id, or the cell's position in the
input. The cells of a submesh derive theirs from the parent's ids, for example $8c + f$ for
facet $f$ of the parent cell with id $c$.

tetraloma uses origin ids wherever a result must not depend on the partition:

- the orientation of edges and faces (see [Orientation](orientation.md#permutation-bits));
- the choice of owners for shared vertices and entities;
- the order of cells in output files.

On the device, an origin id $i$ is stored as two 32-bit words $(h \oplus 2^{31}, l)$, where $h$
and $l$ are the high and low words of $i$ and $\oplus$ is the bitwise exclusive or. With the
high bit flipped, comparing the word pairs as unsigned numbers gives the same order as
comparing the signed ids, so negative ids work too.

The example below decodes the vertex origin ids of the two-triangle mesh and checks that every
edge is stored from its lower origin id to its higher one.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the unit square as two triangles, node origin ids 10 to 13, as above
# let mut mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&coords, Some(&[10, 11, 12, 13]))
#     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
#     .build()?;
mesh.create_entities(1)?;

// Two words per vertex: the high word with its top bit flipped, then the low word.
let words = mesh.vertex_origin_ids().to_host()?;
let ids: Vec<i64> = words
    .chunks_exact(2)
    .map(|w| (i64::from((w[0] ^ 0x8000_0000) as i32) << 32) | i64::from(w[1]))
    .collect();
let mut sorted = ids.clone();
sorted.sort();
assert_eq!(sorted, [10, 11, 12, 13]);

// Every edge runs from its lower origin id to its higher one.
let edge_vertices = mesh.entity_blocks(1)?[0].vertices.data.to_host()?;
for e in edge_vertices.chunks_exact(2) {
    assert!(ids[e[0] as usize] < ids[e[1] as usize]);
}
# Ok(()) }
```
