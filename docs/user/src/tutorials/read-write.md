# Reading and writing files

Most meshes come from a mesh generator rather than from arrays. tetraloma reads Gmsh's MSH 4.1
format and writes Gmsh files for other mesh tools and VTK files for visualisation in ParaView.
This tutorial reads a file, fixes its dimensions, and writes it in each output format. The
program `read_gmsh` reads a file as this tutorial does:

```sh
cargo run --release -p tetraloma-examples --example read_gmsh [-- FILE.msh]
```

The examples use the file `data/two_regions.msh` of `tetraloma-examples`. It holds the box
$[0, 2] \times [0, 1]^2$ in two regions: eight hexahedra named `hex` in $[0, 1]^3$, and forty
tetrahedra with four pyramids named `tet` in the other half.

## Reading Gmsh

[`gmsh::read`](../api/tetraloma/gmsh/fn.read.html) reads a file into an
[`AnyMesh`](../api/tetraloma/enum.AnyMesh.html). The file, not your program, decides the
dimensions: the highest element dimension gives the topological dimension, and the extent of
the coordinates gives the geometric one. Elements of lower dimension become tags of the mesh's
facets, edges or vertices. The snippet below reads the example file:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
// `path` is the example file data/two_regions.msh of tetraloma-examples.
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
assert_eq!((any.tdim(), any.gdim()), (3, 3));
# Ok(())
# }
```

The second argument is the communicator. `Serial` reads on one process; under MPI every rank
passes its communicator, and [Parallel I/O](parallel-io.md) shows the distributed read.
[`GmshOptions`](../api/tetraloma/gmsh/struct.GmshOptions.html) can fix the geometric dimension
with `.gdim(3)`, for example to read a flat surface into three-dimensional space. Its other
options, the partitioner, the ghost cells and the cell order, matter on more than one rank.

The reader accepts ASCII and binary files of either byte order. It reads complete Lagrange
elements of the orders that Gmsh writes, and upgrades the serendipity elements of degree 2 to
complete ones. The physical groups and their names become the mesh's tags. Reading into an
`f32` mesh narrows the coordinates and logs one warning per file.

The program `read_gmsh` reads the file in the precision of the device and fixes the dimensions
in one step:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/read_gmsh.rs:read}}
```

It then looks up the two regions by name. Each region is a set of cells, and together they
cover the mesh:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/read_gmsh.rs:regions}}
```

[Tags and submeshes](tags-submeshes.md) shows what else you can do with tags.

## Fixing the dimensions

An `AnyMesh` has one variant for each of the six pairs of topological and geometric dimension.
Most programs expect one pair and convert once with `try_into`. The conversion fails with
`GridError::DimensionMismatch` if the file holds a different pair, so your program reports a
wrong file instead of computing nonsense. After the conversion the compiler knows the
dimensions, and methods such as `normals` exist only on the meshes where they make sense.

Code that works for every pair can stay with the `AnyMesh`. The macro
[`with_mesh!`](../api/tetraloma/macro.with_mesh.html) runs its body once with the concrete
mesh inside, whatever its type. A `match` on the variants handles the pairs one by one:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;

// The same code for every pair of dimensions.
let volume = with_mesh!(any.as_ref(), m => m.measure(1))?;
assert!((volume - 2.0).abs() < 1e-10);

// One arm per pair you handle.
match &any {
    AnyMesh::Volume(mesh) => println!("{} cell blocks", mesh.cell_blocks().len()),
    other => println!("dimensions ({}, {})", other.tdim(), other.gdim()),
}

// A conversion to the wrong pair is an error.
let planar: Result<PlanarMesh<f64>, GridError> = any.try_into();
assert!(matches!(
    planar,
    Err(GridError::DimensionMismatch { expected: (2, 2), found: (3, 3) })
));
# Ok(())
# }
```

[Typed dimensions](../concepts/typed-dimensions.md#anymesh) explains why tetraloma fixes the
dimensions in the type and keeps the cell types at run time.

## Writing Gmsh

[`GmshWriter`](../api/tetraloma/gmsh/struct.GmshWriter.html) writes a mesh as an MSH 4.1 file,
in ASCII unless you ask for binary. The node tags are the vertex origin ids, and the element
tags are the cell origin ids, so vertices and cells keep the numbers of the file you read. The
other nodes are numbered after the largest vertex id. Tagged facets,
edges and vertices are written as linear elements, with their physical groups and names. The
snippet below writes the example mesh and reads it back:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let mesh: VolumeMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?.try_into()?;

let out = std::env::temp_dir().join(format!("two_regions-{}.msh", std::process::id()));
gmsh::GmshWriter::new(&out).binary(true).write(&mesh)?;

let back: VolumeMesh<f64> = gmsh::read(&client, Serial, &out, GmshOptions::default())?.try_into()?;
std::fs::remove_file(&out)?;
assert!((back.measure(1)? - 2.0).abs() < 1e-10);
assert_eq!(back.tags().named(Codim::Vol, "hex")?.indices.len(), 8);
# Ok(())
# }
```

A tagged facet is written with its vertices in tetraloma's canonical orientation, so the
orientation of boundary elements in the original file is not kept. `write_any` writes an
`AnyMesh` without converting it first. On several ranks, the writer gathers the mesh into one
file, or writes one file per rank with `.partitioned(true)`.

## Writing VTU

[`VtkWriter`](../api/tetraloma/vtk/struct.VtkWriter.html) writes the VTK XML formats that
ParaView reads. `VtkWriter::vtu` writes one `.vtu` file. Cells of degree 2 and higher become VTK
Lagrange cells, so ParaView shows the curved geometry; `.lagrange(false)` writes straight cells
from the vertices instead. The data is appended raw unless you ask for `.base64(true)`, and
`.zlib(true)` compresses it.

You attach data that lives on the device. `.point_data(name, &buffer, components)` takes one
value or vector per geometry node, and `.cell_data(name, &buffer, components)` one per cell, in
the order of the cell blocks. `.origin_ids(true)` adds every cell's origin id as the cell data
`OriginId`. The snippet below writes the cell block of every cell:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let mesh: VolumeMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?.try_into()?;

// One value per cell, block after block, uploaded to the device.
let block_of: Vec<u32> = mesh
    .cell_blocks()
    .iter()
    .enumerate()
    .flat_map(|(b, block)| std::iter::repeat_n(b as u32, block.index_map.num_local()))
    .collect();
let block_of = DeviceBuffer::from_host(mesh.client(), &block_of)?;

let dir = std::env::temp_dir().join(format!("tetraloma-vtu-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
VtkWriter::vtu(dir.join("two_regions"))
    .cell_data("block", &block_of, 1)
    .origin_ids(true)
    .write(&mesh)?;

let bytes = std::fs::read(dir.join("two_regions.vtu"))?;
let header = String::from_utf8_lossy(&bytes);
assert!(header.contains("<VTKFile") && header.contains("Name=\"block\""));
std::fs::remove_dir_all(&dir)?;
# Ok(())
# }
```

The program `vtk_output` attaches point data: the coordinates of every geometry node. It reads
the nodes back to the host to compute them, which is fine for an example; a real program
computes its data on the device:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/vtk_output.rs:write}}
```

The points of the file are all geometry nodes, in `f64` for an `f64` mesh and in `f32` for an
`f32` mesh. High-order pyramids are written from their five vertices, with a warning, because
VTK has no Lagrange pyramid.

## Writing PVTU

`VtkWriter::pvtu` writes the parallel form of VTU: a small `.pvtu` header and one `.vtu` piece
per rank beside it. ParaView opens the header and loads the pieces. On more than one rank the
pieces carry the cell data `vtkGhostType`, which marks the ghost cells, so ParaView shows every
cell once. On one rank the result is a header with a single piece:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let mesh: VolumeMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?.try_into()?;

let dir = std::env::temp_dir().join(format!("tetraloma-pvtu-{}", std::process::id()));
std::fs::create_dir_all(&dir)?;
VtkWriter::pvtu(dir.join("two_regions")).write(&mesh)?;
assert!(dir.join("two_regions.pvtu").exists());
assert!(dir.join("two_regions_0.vtu").exists()); // the piece of rank 0
std::fs::remove_dir_all(&dir)?;
# Ok(())
# }
```

Every write is collective: on more than one rank, every rank calls it with the same writer
settings. [Parallel I/O](parallel-io.md#pvtu-output) shows PVTU output of a distributed mesh.

## Writing VTKHDF

VTKHDF stores the same pieces as PVTU in a single HDF5 file, which suits large runs on many
ranks better than thousands of small files. ParaView reads it as one data set. It needs the
HDF5 library and the Cargo feature `hdf5` of tetraloma, which [Cargo features](../features.md#hdf5)
describes. With that feature, the writer gains the constructor `vtkhdf`, used like `vtu`:

```text
VtkWriter::vtkhdf("out/two_regions").write(&mesh)?;   // writes out/two_regions.vtkhdf
```

The file holds one partition per rank, with the same cell types and the same `vtkGhostType`
array as the PVTU pieces. With an MPI communicator every rank writes its partition through
parallel HDF5; otherwise rank 0 writes the gathered pieces. VTKHDF output is never compressed,
and `.zlib(true)` is an error. [Parallel I/O](parallel-io.md#vtkhdf-output) runs it on several
ranks.
