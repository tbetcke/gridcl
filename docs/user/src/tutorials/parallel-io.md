# Parallel I/O

On many ranks, reading and writing files is collective: every rank calls the same function, and
each rank reads or writes its own part. This tutorial follows the program `parallel_io`. It reads
the example file `tetraloma-examples/data/two_regions.msh`, the box $[0, 2] \times [0, 1]^2$ in 52
cells, on every rank. It then writes the mesh as a binary Gmsh file and reads that file back, and
writes it for visualisation as PVTU and as VTKHDF. Build and run it on four ranks with:

```sh
cargo build --release -p tetraloma-examples --features mpi,hdf5 --example parallel_io
mpirun -n 4 target/release/examples/parallel_io
```

It writes its files to `target/tetraloma-examples/parallel_io`, or to the directory you give as
its argument. Without the feature `hdf5` it writes no VTKHDF file. Its `main` opens the device and
the output directory:

```rust,ignore
{{#include ../../../../tetraloma-examples/examples/parallel_io.rs:main}}
```

The serial reading and writing functions are the subject of
[Reading and writing files](read-write.md). The same functions work on a distributed mesh; this
tutorial shows what changes.

## Reading Gmsh in parallel

[`gmsh::read`](../api/tetraloma/gmsh/fn.read.html) takes a communicator. Every rank calls it with
the same path and options, and each rank gets its part of the mesh:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/parallel_io.rs:read}}
```

How the file is read depends on its encoding:

- A binary file is read in pieces. The reader first scans the block headers, which give the
  position of every record, and then each rank reads its own byte range of the nodes and of the
  elements.
- An ASCII file is read on rank 0, which sends each rank the same piece that it would have read
  from a binary file. Files with serendipity elements are read this way too.

The pieces then go through the same pipeline as the input of `build_on`, described in
[Distributed meshes](distributed.md#building-on-a-communicator): partitioning, migration of the
cells, ghost cells, cell ordering and numbering.
[`GmshOptions`](../api/tetraloma/gmsh/struct.GmshOptions.html) holds the choices of that pipeline:

- `partitioner(p)`: the partitioner. Without one, a read on more than one rank uses PT-Scotch with
  the feature `ptscotch` and an MPI communicator, otherwise METIS on rank 0 with the feature
  `metis`, otherwise `Block`, and rank 0 logs a warning. A file that stores a partition into as
  many parts as there are ranks keeps that partition with `Block`.
- `ghosts(mode)`: the ghost cells, `GhostMode::None` by default.
- `ordering(o)`: the order of the owned cells, `CellOrdering::Input` by default.
- `gdim(d)`: the geometric dimension. Without it, the dimension is inferred from the coordinates
  of all nodes on all ranks.

Tags travel with their cells and entities. A ghost cell, facet or vertex gets the same tags as its
owner, so [regions and boundary groups](tags-submeshes.md#regions) work as on one rank. Errors are
the same as in a serial read, and every rank returns an error if any rank fails.

The program's file is ASCII. To show the binary path, it writes the mesh as binary Gmsh and reads
it back on every rank:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/parallel_io.rs:binary}}
```

The [`GmshWriter`](../api/tetraloma/gmsh/struct.GmshWriter.html) gathers the owned cells and
their nodes to rank 0, which writes one file; `.partitioned(true)` writes one file per rank
instead. After each read, the program checks that the ranks own the 52 cells once between them
and that the volume is 2.

On one rank, use `Serial` as the communicator; the partitioner then has nothing to do:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let client = cubecl::Device::cpu()?.client();
let options = GmshOptions::default()
    .partitioner(Block)
    .ghosts(GhostMode::SharedFacet);
// path: tetraloma-examples/data/two_regions.msh
let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, options)?;
let mesh: VolumeMesh<f64> = any.try_into()?;
let owned: usize = mesh.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
assert_eq!(owned, 52);
assert!((mesh.measure(1)? - 2.0).abs() < 1e-10);
# Ok(()) }
```

## PVTU output

A PVTU file is a small header that lists pieces, one VTU file per rank. ParaView and VTK read the
header and load the pieces together. `VtkWriter::pvtu(stem)` writes the header `<stem>.pvtu` on
rank 0 and the piece `<stem>_<rank>.vtu` on every rank. The write is collective, and every rank
returns an error if any rank fails.

Each piece holds all the rank's local cells, owned and ghost, and their nodes. Cells that appear
in two pieces would be drawn twice, so on more than one rank each piece carries the cell array
`vtkGhostType`: 0 for an owned cell and 1 for a ghost. The header's `GhostLevel` is 1 if any rank
has ghost cells, and ParaView then hides the ghosts.

Data you attach follows the local numbering: `cell_data` takes one value (or one block of
`components` values) per local cell, block by block with the owned cells first, and `point_data`
one per local geometry node. The program attaches the rank that owns each cell, which shows the
partition in ParaView. For the ghost cells it takes the owners from the cell index maps:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/parallel_io.rs:owner}}
```

It then writes the PVTU file, with the cell array `OriginId` that `origin_ids(true)` adds:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/parallel_io.rs:pvtu}}
```

On rank 0, it reads the header back and checks that it lists one piece per rank and the right
`GhostLevel`. On one rank, the same call writes a header and a single piece:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let client = cubecl::Device::cpu()?.client();
// path: tetraloma-examples/data/two_regions.msh
let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
let mesh: VolumeMesh<f64> = any.try_into()?;

let dir = std::env::temp_dir().join(format!("tetraloma-book-pvtu-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
VtkWriter::pvtu(dir.join("two_regions")).origin_ids(true).write(&mesh)?;
assert!(dir.join("two_regions.pvtu").is_file());
assert!(dir.join("two_regions_0.vtu").is_file());
# std::fs::remove_dir_all(&dir)?;
# Ok(()) }
```

`VtkWriter::vtu` also works on a distributed mesh: it gathers the owned cells to rank 0, which
writes one VTU file. This suits small meshes; for large ones, write PVTU or VTKHDF.

## VTKHDF output

VTKHDF is VTK's format in an HDF5 file. `VtkWriter::vtkhdf(stem)` writes one file
`<stem>.vtkhdf` that holds one partition per rank, in rank order. Each partition has the same
cells, points and arrays as the PVTU piece of that rank, `vtkGhostType` included, so ParaView
shows the same mesh. One file is easier to move and archive than a header and a piece per rank.

The writer exists only with the feature `hdf5`, which needs the HDF5 library (see
[Cargo features](../features.md#hdf5)). The program therefore puts it under
`#[cfg(feature = "hdf5")]`:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/parallel_io.rs:vtkhdf}}
```

With an `Mpi` communicator every rank writes its own partition into the shared file through
HDF5's MPI-IO driver. With any other communicator, such as `Serial`, the partitions are gathered
and rank 0 writes the file. VTKHDF output is not compressed: `.zlib(true)` returns
`GridError::NotYetImplemented`, and `.base64(true)` has no effect. On rank 0, the program checks
that the file starts with the HDF5 signature.
