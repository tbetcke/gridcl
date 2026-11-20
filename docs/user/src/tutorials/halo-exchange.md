# Halo exchange

On a distributed mesh, an entity on the border between ranks exists on each of them: owned on one
and a ghost on the others, as [Distributed meshes](distributed.md#ownership) explains. Data on
these entities lives in arrays with one entry per local entity, owned entries first. The ghost
entries form the halo. A halo exchange keeps the halo consistent with the owners, in one of two
directions:

- **forward**, owner to ghosts: each ghost receives its owner's value, for example after the
  owners have computed new values;
- **reverse**, ghosts to owner: each owner combines the values of all its ghosts with its own, for
  example to add up contributions that several ranks computed for the same node.

This tutorial follows the program `halo_exchange`. It runs both exchanges on the geometry nodes of
a distributed surface mesh: the unit sphere as 320 curved triangles of degree 2, split over the
ranks by the `Block` partitioner, with one layer of ghost cells across edges. Build and run it on
two ranks with:

```sh
cargo build --release -p tetraloma-examples --features mpi --example halo_exchange
mpirun -n 2 target/release/examples/halo_exchange
```

Its `main` is that of the other MPI programs:

```rust,ignore
{{#include ../../../../tetraloma-examples/examples/halo_exchange.rs:main}}
```

## Index maps

An [`IndexMap`](../api/tetraloma/struct.IndexMap.html) describes how a set of entities is shared
between ranks. Each rank owns `num_owned()` of them, numbered locally from 0 and globally in the
contiguous range `owned_range()`. After them come `num_ghosts()` ghosts. For each ghost, the map
knows its global index, `ghosts()`, and its owner's rank, `ghost_owners()`. `num_local()` is the
sum of the two, and `size_global()` the number of owned entities over all ranks.

A mesh has index maps for its cell blocks, its vertices, its geometry nodes and, once created, its
entity blocks. The program takes the node index map of its surface mesh `scalp`:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/halo_exchange.rs:index_map}}
```

You can also make index maps for data of your own. `IndexMap::new(comm, n)` makes a map with `n`
owned indices and no ghosts. `IndexMap::with_ghosts(comm, n, ghosts, owners)` adds ghosts, given
by their global indices and owner ranks; the owners check that they own the indices asked for.
Both are collective, and global indices are numbered rank by rank in the order of the ranks:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
/// Each rank owns `n` indices; on more than one rank it also holds the first index of the next
/// rank, round the ring, as a ghost.
fn ring(comm: &dyn CommDyn, n: usize) -> tetraloma::Result<IndexMap> {
    let (rank, size) = (comm.rank(), comm.size());
    if size == 1 {
        return IndexMap::new(comm, n); // no other rank to take a ghost from
    }
    let next = (rank + 1) % size;
    // Rank r owns the global indices r n to (r + 1) n - 1.
    IndexMap::with_ghosts(comm, n, vec![(next * n) as i64], vec![next as i32])
}

let map = ring(&Serial, 4)?;
assert_eq!(map.owned_range(), (0, 4));
assert_eq!((map.num_owned(), map.num_ghosts(), map.num_local()), (4, 0, 4));
assert_eq!(map.size_global(), 4);
# Ok(()) }
```

With `Serial` the ring has no ghosts. Under `mpirun`, the same function called with an `Mpi`
communicator gives each rank one ghost.

## Scatterers

A [`Scatterer`](../api/tetraloma/struct.Scatterer.html) runs the exchanges of one index map on
device arrays. `index_map.scatterer(client, block_size)` makes one for arrays with `block_size`
values per index, so an array holds `block_size * num_local()` values: the owned blocks, then the
ghost blocks. Making a scatterer is not collective. The first scatterer of a map uploads the lists
of indices to send and receive to the device, and every scatterer keeps its buffers for reuse, so
make a scatterer once and use it for every exchange.

The exchange stays on the device except for the values that cross between ranks. A kernel packs
the values to send, they are copied to the host and sent by MPI to the neighbouring ranks only,
and the received values are uploaded and unpacked by another kernel. `transport()` reports this
as `Transport::HostStaged`, the transport every scatterer uses. The program makes a scatterer for one
vector of three values per node, in the forward step below.

On one rank there is nothing to exchange, and both directions leave an array unchanged. The calls
are the same as on many ranks:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
// Two unit squares side by side: six nodes.
let coords = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
let quad = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 1, family: NodeFamily::Equispaced };
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, None)
    .cell_block(quad, &[0, 1, 3, 4, 1, 2, 4, 5], None)
    .build()?;

let nodes = mesh.node_index_map();
let sc = nodes.scatterer(mesh.client(), 2)?; // block size 2: two values per node
assert_eq!(sc.transport(), Transport::HostStaged);

let mut u = DeviceBuffer::from_host(mesh.client(), &vec![1.0; 2 * nodes.num_local()])?;
sc.reverse(&mut u, ReverseOp::Add)?; // ghosts -> owners: add the contributions
sc.forward(&mut u)?; // owners -> ghosts: copy the sums back
assert_eq!(u.to_host()?, vec![1.0; 12]); // one rank: unchanged
# Ok(()) }
```

## Forward

`forward` gives every ghost block its owner's block and leaves the owned blocks unchanged. It moves
the values as 32-bit words, so it works for any element type whose size is a multiple of four
bytes, on every device. It is collective over the index map's communicator.

The program fills an array with the coordinates of the owned nodes and zeros on the ghosts, and
sends the owners' coordinates forward:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/halo_exchange.rs:forward}}
```

The coordinates come from the device with `to_host` only to set up and check the example. A
program that computes on the device would fill `u` with a kernel. After `forward`, the program
checks that every ghost node holds its owner's coordinates, which are its own coordinates too.

## Reverse

`reverse` combines, for every owned block that some rank holds as a ghost, the owner's value with
the value of every ghost copy. A [`ReverseOp`](../api/tetraloma/enum.ReverseOp.html) chooses how:

- `ReverseOp::Add` adds them, as in assembling a sum over cells that meet at a node;
- `ReverseOp::Min` and `ReverseOp::Max` take the smallest or largest value;
- `ReverseOp::Insert` replaces the owner's value by the contribution of the lowest neighbouring
  rank, and leaves it as it is if there is none.

The ghost values themselves stay unchanged, so a reverse exchange is often followed by a forward
one, which copies the combined values back to the ghosts. `reverse` works on f32, f64, i32 and u32
arrays, and returns `GridError::UnsupportedPrecision` for a type the device does not support, such
as f64 on Metal.

The program sets every value to 1 and adds the ghosts to their owners:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/halo_exchange.rs:reverse}}
```

Each owned value then equals 1 plus the number of ranks that hold its node as a ghost. The program
sums the owned values over all ranks and checks that the total is $3 (N + G)$, where $N$ is the number of
nodes and $G$ the number of ghost copies: each node has three components.

The reduction is deterministic. Each owned value is combined with its contributions in a fixed
order, by neighbouring rank and then by the ghost's position on that rank, without atomic
operations. A sum in floating point therefore gives the same bits on every run, for a given mesh
and number of ranks; [Precision](../concepts/precision.md#determinism) says more about
determinism.
