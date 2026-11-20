# Cargo features

tetraloma's default build is serial and needs no system library: it builds meshes, computes their
geometry, reads and writes Gmsh files and writes VTU files on any device. Cargo features add MPI,
VTKHDF output and external partitioners, each with the system libraries it needs. The devices
themselves come from the runtime features of CubeCL, which your program turns on.

| Feature of `tetraloma` | Adds | Implies | Needs |
| --- | --- | --- | --- |
| `mpi` | the MPI communicator `Mpi`, distributed meshes and parallel I/O over MPI | – | MPI, libclang |
| `hdf5` | VTKHDF output | – | HDF5 |
| `metis` | the partitioner `Metis` (METIS on rank 0) | – | nothing: METIS is built from source |
| `scotch` | the partitioner `Scotch` (Scotch on rank 0) | – | Scotch |
| `ptscotch` | the partitioner `PtScotch` (PT-Scotch on all ranks) | `mpi` | PT-Scotch |
| `parmetis` | the partitioner `ParMetis` (ParMETIS on all ranks) | `mpi` | ParMETIS |
| `gpu-aware-cuda` | nothing yet (reserved) | – | – |

Turn features on in the dependency, for example
`tetraloma = { git = "https://github.com/tetracl/tetraloma", features = ["mpi", "ptscotch"] }`.
`metis` and `parmetis` cannot be on together.

## Runtime features

A runtime feature links one CubeCL backend into your program. They are features of the `cubecl`
dependency, not of tetraloma: tetraloma's library code works with whichever runtimes the final
program links, and you choose among them at run time.

```toml
[dependencies]
cubecl = { version = "=0.11.0", features = ["cpu", "cuda", "vulkan"] }
```

The features are `cpu`, `cuda`, `hip` (also called `rocm`), `vulkan`, `metal`, `metal-native` and
`wgpu`; `cuda-cpp` switches CUDA to CubeCL's NVRTC compiler. Each adds build time and, for the GPU
backends, a dependency on their drivers, so turn on only the ones you use.
[Compute devices](devices.md) describes each backend and the versions it needs.

The crate `tetraloma-examples` of the repository, which holds this book's programs, has the
features `cpu` (the default), `cuda`, `vulkan` and `metal`, which turn on the cubecl features of
the same names, and `hdf5`, which turns on tetraloma's. Its MPI programs need its feature `mpi`.

## MPI

The feature `mpi` adds [`Mpi`](api/tetraloma/struct.Mpi.html), a communicator over MPI, which every
type with a communicator parameter accepts: `Mesh<F, TD, GD, Mpi>` is a mesh distributed over the
ranks of an MPI program. Without the feature, the serial communicator `Serial` is the only one and
no MPI library is needed. [Distributed meshes](tutorials/distributed.md) shows how to build and
use distributed meshes.

tetraloma uses the [rsmpi](https://github.com/rsmpi/rsmpi) crate `mpi`, version 0.8.2. Its build
needs:

- an MPI implementation, found through pkg-config or through its compiler wrapper `mpicc` on the
  `PATH`. tetraloma is tested with MPICH and Open MPI, on 1 to 4 ranks, on every change.
- libclang, because rsmpi generates its MPI bindings with bindgen (Ubuntu: `libclang-dev`; macOS:
  the Xcode command-line tools).

Your program initialises MPI itself, so it depends on `mpi` as well, at the same version:

```toml
[dependencies]
tetraloma = { git = "https://github.com/tetracl/tetraloma", features = ["mpi"] }
mpi = { version = "=0.8.2", default-features = false }
```

Run the program with the MPI implementation's launcher, for example `mpirun -n 4 ./my-program`.

## HDF5

The feature `hdf5` adds VTKHDF output, `VtkWriter::vtkhdf`: one HDF5 file that holds the pieces of
all ranks ([Parallel I/O](tutorials/parallel-io.md)). It builds the crate `hdf5-metno` against a
system HDF5 of version 1.10.10 to 2.2.0. Set `HDF5_DIR` to the HDF5 installation, the directory
with `include/` and `lib/`:

```sh
export HDF5_DIR=/opt/homebrew/opt/hdf5-mpi    # Homebrew on macOS
```

With `mpi` on as well, every rank writes its own part of the file through HDF5's MPI-IO driver.
That needs an HDF5 built with MPI, against the same MPI implementation as your program: Homebrew's
`hdf5-mpi`, or Ubuntu's `libhdf5-openmpi-dev` or `libhdf5-mpich-dev`. Without `mpi`, any HDF5
works, and rank 0 writes the file.

Ubuntu keeps the headers and libraries of its MPI builds of HDF5 in separate trees. Make a
directory that joins them and point `HDF5_DIR` at it; the library directory is the one that holds
`libhdf5.so` (`dpkg -L libhdf5-openmpi-dev` lists it):

```sh
mkdir -p ~/hdf5-openmpi
ln -sfn /usr/include/hdf5/openmpi ~/hdf5-openmpi/include
ln -sfn /usr/lib/x86_64-linux-gnu/hdf5/openmpi ~/hdf5-openmpi/lib
export HDF5_DIR=~/hdf5-openmpi
```

## Partitioners

A distributed mesh is partitioned when it is built: its cells are assigned to ranks so that each
rank gets about the same number of cells and few cells have neighbours on other ranks. The
built-in [`Block`](api/tetraloma/struct.Block.html) partitioner needs no library: it splits the
cells into contiguous ranges in input order. The features below add graph partitioners, which cut
fewer cell faces between ranks. Each of them implements
[`Partitioner`](api/tetraloma/trait.Partitioner.html) and takes a `seed`; for a fixed seed and
number of ranks, the partition is reproducible.

| Feature | Partitioner | Runs | Library and how the build finds it |
| --- | --- | --- | --- |
| `metis` | [`Metis`](api/tetraloma/struct.Metis.html) | on rank 0 | METIS, built from source by the crate `metis`; no system library |
| `scotch` | [`Scotch`](api/tetraloma/struct.Scotch.html) | on rank 0 | Scotch (`libscotch`), in `/usr/include/scotch` and the system paths, or the directories `CPATH` and `LIBRARY_PATH` add |
| `ptscotch` | [`PtScotch`](api/tetraloma/struct.PtScotch.html) | on all ranks | PT-Scotch (`libptscotch`), in `SCOTCH_DIR` (with `include/` and `lib/`), else `/usr/include/scotch`, `/usr/include`, `/usr/local` and `/opt/homebrew` |
| `parmetis` | `ParMetis` | on all ranks | ParMETIS and METIS, in `PARMETIS_DIR` and `METIS_DIR`, else the system paths |

Some choices to know about:

- **Rank-0 partitioners** gather the whole graph on rank 0, so rank 0 needs memory for it. They
  suit meshes up to about $10^7$ cells; beyond that, use PT-Scotch or ParMETIS.
- **The default for distributed reads.** When you read a Gmsh file on several ranks without
  naming a partitioner, tetraloma uses PT-Scotch if `ptscotch` is on, else METIS if `metis` is on,
  else `Block` with a warning.
- **`metis` and `parmetis` exclude each other.** The METIS that the crate `metis` builds and the
  METIS that ParMETIS links clash at run time, so the build stops with an error if both are on.
- **Scotch versions.** tetraloma's results are reproducible with Scotch 7.0.4 and 7.0.14. Scotch
  7.0.18's threaded build gives different partitions from run to run, so avoid it; build 7.0.14
  from source if your package manager has a newer one.
- **ParMETIS** needs every rank to hold at least one cell before partitioning. Check its licence
  terms before you redistribute binaries that contain it.

Libraries in non-standard directories also need to be found at run time: add their `lib`
directories to `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS.

## Reserved features

The feature `gpu-aware-cuda` is reserved for a halo exchange that sends data directly from GPU
memory through a CUDA-aware MPI. You may turn it on, but it enables nothing yet: every halo exchange copies the values to the host, sends them with MPI and copies them back.
[`Transport`](api/tetraloma/enum.Transport.html) has the variant `GpuAware` for the same reason;
`HostStaged` is the only transport in use.

The crates of the workspace have further features for tetraloma's own tests and benchmarks
(`testing` and `bench`). They are not part of the public API, and a program should not turn them
on.
