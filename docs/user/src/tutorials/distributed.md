# Distributed meshes

A distributed mesh is split over the ranks of an MPI program. Each rank owns some of the cells,
every cell has exactly one owner, and each rank may also hold copies of neighbouring cells that
other ranks own, the ghost cells. The type says which communicator the mesh lives on: it is the
last parameter `C` of [`Mesh<F, TD, GD, C>`](../api/tetraloma/struct.Mesh.html). A serial mesh
uses [`Serial`](../api/tetraloma/struct.Serial.html), a communicator of one rank, and every
distributed method works on it too. With the feature `mpi`,
[`Mpi`](../api/tetraloma/struct.Mpi.html) wraps an MPI communicator.

This tutorial follows the program `distributed_mesh`. It builds the unit square as $8 \times 8$
quadrilaterals on every rank of `MPI_COMM_WORLD`, several times, with different partitioners,
ghost modes and cell orderings, and checks each result. Build and run it on four ranks with:

```sh
cargo build --release -p tetraloma-examples --features mpi --example distributed_mesh
mpirun -n 4 target/release/examples/distributed_mesh
```

[Cargo features](../features.md#mpi) lists what the feature `mpi` needs.

## Building on a communicator

The program's `main` initialises MPI, wraps the world communicator, opens one device per rank and
runs the program in f64 where the device supports it:

```rust,ignore
{{#include ../../../../tetraloma-examples/examples/distributed_mesh.rs:main}}
```

`check_world_size` belongs to the example crate, not to tetraloma. It compares the size of the
communicator with the variable `TETRALOMA_EXPECT_RANKS`, which the repository's script
`scripts/run-examples.sh` sets to the rank count it passes to `mpirun`, and returns an error if
they differ. Without the variable it accepts any size. A launcher that starts every process on
its own, as rank 0 of 1, then stops the program at once instead of running several one-rank
copies of it ([Troubleshooting](../troubleshooting.md#ranks-that-do-not-see-each-other)).

`local_gpu_index` gives each rank the index of a GPU on its node, as explained in
[Compute devices](../devices.md#one-gpu-per-mpi-rank). An `Mpi` communicator holds a reference
count and is not `Send`: each rank runs one host thread that owns its mesh.

[`MeshBuilder::build_on`](../api/tetraloma/struct.MeshBuilder.html) builds a mesh on a
communicator. It is collective: every rank calls it and passes its own share of the input. A
rank's share is a set of nodes, each with its coordinates and its origin id, and a set of cells.
The cells name their nodes by origin id with `cell_block_by_ids`, so a cell may use nodes that
another rank passed. Origin ids are the numbers of the input; they must be unique over all ranks.
Which rank passes what is up to you: an even share of a file or of a generator, or everything on
rank 0. The program's shares are even slices of the square's nodes and cells:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:chunk}}
```

Each rank then calls `build_on` with its share:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:build_on}}
```

`build_on` runs the same pipeline on every rank:

1. It reads the shares and checks them. Errors are agreed: the rank with an error returns it,
   and every other rank returns `GridError::RemoteFailure` with the lowest failing rank.
2. It builds the dual graph, whose vertices are the cells and whose edges join cells that share
   a facet, and partitions it: each cell gets a destination rank.
3. It sends each cell to its destination, adds the ghost cells and fetches the coordinates of
   every node they use.
4. It orders the owned cells, numbers cells, vertices and nodes, and uploads the mesh to the
   device.

`build` is `build_on(Serial)`, so the same steps build a serial mesh. Here they are on one rank,
with every option set:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
// The unit square as two quadrilaterals side by side; the cells name their nodes by origin id.
let coords = [0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0];
let quad = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 1, family: NodeFamily::Equispaced };
let mesh: PlanarMesh<f64, Serial> = MeshBuilder::new(&client)
    .nodes(&coords, Some(&[0, 1, 2, 3, 4, 5]))
    .cell_block_by_ids(quad, &[0, 1, 3, 4, 1, 2, 4, 5], Some(&[100, 101]))
    .partitioner(Block)
    .ghosts(GhostMode::SharedFacet)
    .ordering(CellOrdering::Hilbert)
    .build_on(Serial)?;

let cells = &mesh.cell_blocks()[0].index_map;
assert_eq!((cells.num_owned(), cells.num_ghosts()), (2, 0)); // one rank has no ghosts
assert_eq!((cells.owned_range(), cells.size_global()), ((0, 2), 2));
assert!((mesh.measure(1)? - 1.0).abs() < 1e-12);
# Ok(()) }
```

### Ownership

Every cell block, the vertices and the geometry nodes have an
[`IndexMap`](../api/tetraloma/struct.IndexMap.html): `block.index_map`, `vertex_index_map()` and
`node_index_map()`. Local indices list the owned entities first and the ghosts after them, so
`0..num_owned()` are the owned ones. Each rank's owned entities get a contiguous range of global
numbers, `owned_range()`, and a ghost knows the global number and the rank of its owner.

A vertex on the border between two ranks appears on both. One of the ranks that hold it in an
owned cell owns it, chosen by a hash of its origin id, and the others hold it as a ghost. So a
mesh without ghost cells still has ghost vertices and nodes. The choice depends only on the input
and the partition, not on the local numbering, and it is the same in every run. The program
prints the owned and ghost cells and vertices of each rank, and checks that the ranks own the 64
cells and 81 vertices once between them. [Meshes and topology](../concepts/meshes.md#ownership-and-ghosts)
explains ownership in more detail.

Quantities over the mesh count owned cells only, and are summed over the ranks: `measure`
returns the area 1 on every rank, whatever the ghost cells. Such methods are collective, like
`build_on`.

## Partitioners

The partitioner decides which rank owns each cell. It sees the dual graph, distributed over the
ranks as a [`DistGraph`](../api/tetraloma/struct.DistGraph.html) in the format of ParMETIS: each
rank holds a contiguous range of graph vertices, `vtxdist` gives the ranges, and `xadj` and
`adjncy` give each local vertex's neighbours as global vertex numbers. Vertex weights estimate
the work of each cell from its type and degree. On one rank, `build_on` calls no partitioner.

The built-in [`Block`](../api/tetraloma/struct.Block.html) partitioner is the default. It sends
graph vertex $i$ of $N$ to rank $\lfloor i p / N \rfloor$ on $p$ ranks. The graph numbers the
cells in input order, by rank and then by position, so `Block` keeps the input's order and
splits it into even pieces. It ignores the edges: it is fast, and it is good when the input is
already ordered so that neighbouring cells are close in the order. Graph partitioners keep the
number of cut edges small, which keeps the ghost layers and the halo exchanges small. Each is
behind a Cargo feature (see [Cargo features](../features.md#partitioners)):

- [`Metis`](../api/tetraloma/struct.Metis.html) (feature `metis`) and
  [`Scotch`](../api/tetraloma/struct.Scotch.html) (feature `scotch`) gather the graph on rank 0
  and partition it there, which suits meshes of up to about $10^7$ cells;
- [`PtScotch`](../api/tetraloma/struct.PtScotch.html) (feature `ptscotch`) and `ParMetis`
  (feature `parmetis`) partition in parallel on the MPI communicator.

The features `metis` and `parmetis` exclude each other. Each external partitioner takes a seed,
for example `PtScotch::default().seed(7)`, and gives the same partition for the same input, seed
and number of ranks.

The program builds the square with `Block` and with a partitioner of its own:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:partitioners}}
```

A partitioner is any type that implements the
[`Partitioner`](../api/tetraloma/trait.Partitioner.html) trait. `partition` is collective, and
returns the destination rank of each of this rank's graph vertices. The program's `RoundRobin`
sends vertex $i$ to rank $i \bmod p$:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:round_robin}}
```

Round-robin scatters neighbouring cells over the ranks, so on more than one rank it cuts more
edges than `Block` and gives more ghost cells. The program prints both counts, so you can compare them.

## Ghost modes

A ghost cell is a copy of a cell that another rank owns. Ghost cells let a rank compute on the
neighbourhood of its owned cells, for example to evaluate a quantity on a facet from both of its
cells, without asking other ranks. [`GhostMode`](../api/tetraloma/enum.GhostMode.html) chooses the
layer:

- `GhostMode::None`, the default, adds no ghost cells;
- `GhostMode::SharedFacet` adds the cells of other ranks that share a facet with an owned cell;
- `GhostMode::SharedVertex { layers }` adds the cells of other ranks that share a vertex with an
  owned cell, and repeats this `layers` times, each time from the cells added before.
  `layers: 0` is the same as `None`.

Ghost cells come after the owned cells in each cell block, ordered by the owner's rank and then
by origin id. The program builds the square once in each mode:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:ghosts}}
```

It checks that `None` gives no ghost cells, that each further mode gives at least as many as the
one before, and that ghosts across facets appear exactly when there is more than one rank. The
area stays 1 in every mode, because ghost cells are not counted twice. The vertices and nodes of
ghost cells become ghost vertices and nodes, and the [halo exchange](halo-exchange.md) keeps their
values up to date.

## Cell ordering

The order of a rank's owned cells is the order in which kernels visit them. Neighbouring cells
close together in memory share vertex and node data in the caches.
[`CellOrdering`](../api/tetraloma/enum.CellOrdering.html) chooses the order:

- `CellOrdering::Input`, the default, keeps the input order: by source rank, then by position;
- `CellOrdering::Hilbert` sorts the cells along a Hilbert curve through their vertex centroids,
  a curve that visits nearby points one after another;
- `CellOrdering::ReverseCuthillMcKee` numbers the cells by reverse Cuthill–McKee on the rank's
  facet graph, starting from a cell with the fewest neighbours. This keeps the indices of
  neighbouring cells close.

Ties are broken by origin id, so each ordering is deterministic. The ordering is applied after
partitioning and before numbering. It therefore changes the local order of the owned cells, and
with it the local order of vertices and nodes, which are numbered in order of the first cell that
uses them. It does not change which cells a rank owns. The program builds the square in each
order:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/distributed_mesh.rs:ordering}}
```

`owned_cell_ids` recovers each owned cell's row and column from its centre, which it reads back
to the host for inspection. The printed path is the distance from each cell to the next, summed
and measured in cell widths: the smaller it is, the closer consecutive cells lie. The program
checks that the input order is ascending, and that every ordering leaves each rank the same set
of owned cells.
