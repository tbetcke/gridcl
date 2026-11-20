# Troubleshooting

This chapter collects the problems that users of tetraloma meet most often, sorted by where they
show up: when you build your program, when it opens a device, when it runs under MPI, and when its
numbers look wrong. Each entry names the symptom, its cause and the fix, and links to the chapter
that explains the background.

tetraloma reports warnings, for example about an `f32` read of a Gmsh file or a high-order pyramid
written to VTK, through the [`log`](https://docs.rs/log) crate. You see them only if your program
installs a logger, such as `env_logger`; without one they are dropped silently.

## Build problems

Most build failures come from a missing system library or from a feature that cannot find one.
The default build needs only the Rust toolchain, a C compiler and network access for its first
build; [Installation](getting-started/installation.md) lists the requirements and
[Cargo features](features.md) the libraries of each optional feature.

### The first build downloads LLVM

The CubeCL CPU runtime and the CUDA backend compile kernels with LLVM, and the build script of
CubeCL's `cubecl-llvm` crate downloads a prebuilt LLVM the first time it runs. A first build
without network access therefore fails in that build script. Build once with network access; the
download is kept in the directory `tracel` under your user data directory (on Linux the one that
`XDG_DATA_HOME` names, by default `~/.local/share`), and later builds reuse it. On Linux the
prebuilt LLVM needs glibc 2.34 or newer. On older systems, such as Red Hat Enterprise Linux 7,
build an LLVM of the same version yourself and place it where the build script looks for it
([Toolchain](getting-started/installation.md#toolchain)).

### Mismatched types from two CubeCL versions

A type mismatch between two types of the same name, such as `Client`, often with the note that
two different versions of the crate `cubecl` are being used, means that your program and tetraloma
use different versions of CubeCL. Their clients are then different types. Depend on exactly the version tetraloma uses,
`cubecl = { version = "=0.11.0", … }`, and check with `cargo tree -i cubecl` that only one version
is in the build. The same holds for the MPI crate `mpi`: use version 0.8.2, so that your program
and tetraloma share one MPI environment type.

### MPI or libclang not found

The feature `mpi` builds the rsmpi crate `mpi`, which finds the MPI implementation through
pkg-config or through the compiler wrapper `mpicc` on the `PATH`, and generates its bindings with
bindgen, which needs libclang. If the build reports that it cannot find MPI, install the MPI
development package (Ubuntu: `libopenmpi-dev` or `libmpich-dev`) or put `mpicc` on the `PATH`. If
it reports that it cannot find libclang, install it (Ubuntu: `libclang-dev`; macOS: the Xcode
command-line tools), or point the variable `LIBCLANG_PATH` at the directory that holds it.
[MPI](features.md#mpi) lists the requirements.

### HDF5 not found or the wrong HDF5

The feature `hdf5` builds against a system HDF5 of version 1.10.10 to 2.2.0. Set `HDF5_DIR` to
the installation, the directory with `include/` and `lib/`. On Ubuntu the MPI builds of HDF5 keep
their headers and libraries in separate trees, so make a directory that joins them, as
[HDF5](features.md#hdf5) shows. With `mpi` on as well, the HDF5 must be built with MPI, against
the same MPI implementation as your program: Homebrew's `hdf5-mpi`, or Ubuntu's
`libhdf5-openmpi-dev` or `libhdf5-mpich-dev`.

### Partitioner libraries not found

Each external partitioner looks for its library in its own way
([Partitioners](features.md#partitioners)):

- `scotch` searches `/usr/include/scotch` and the system paths; for a Scotch elsewhere, add its
  `include` and `lib` directories to `CPATH` and `LIBRARY_PATH`.
- `ptscotch` searches `SCOTCH_DIR` (with `include/` and `lib/`), then `/usr/include/scotch`,
  `/usr/include`, `/usr/local` and `/opt/homebrew`.
- `parmetis` searches `PARMETIS_DIR` for ParMETIS and `METIS_DIR` for METIS, then the system
  paths.
- `metis` needs no system library, because the crate `metis` builds METIS from source.

A library outside the standard directories must also be found when the program runs: add its
`lib` directory to `LD_LIBRARY_PATH` on Linux or `DYLD_LIBRARY_PATH` on macOS. On macOS with
Homebrew's `scotch` installed, the link may record Homebrew's library even when `CPATH` and
`LIBRARY_PATH` name another Scotch, because a build script puts `/opt/homebrew/lib` first on the
linker's search path. Set `DYLD_LIBRARY_PATH` to the `lib` directory of the Scotch you want, or
uninstall Homebrew's `scotch`.

### `metis` and `parmetis` together

The build stops with "the features `metis` and `parmetis` are mutually exclusive". The METIS that
the crate `metis` builds and the METIS that ParMETIS links define the same symbols and clash at
run time, so tetraloma refuses the combination. Turn on one of the two. Cargo unifies features
over the whole dependency graph, so check with `cargo tree -e features -i tetraloma` whether
another crate of your build turns on the other one.

### Slow programs in debug builds

A debug build compiles quickly, but the host code of tetraloma and CubeCL runs much more slowly
in it, so mesh creation, file reading and partitioning take far longer than in a release build.
The kernels themselves are compiled at run time and are equally fast in both profiles. Build with
`--release` for real work ([Performance settings](devices.md#performance-settings)).

## Device problems

A device problem shows up when your program opens a device or creates its first mesh. The chapter
[Compute devices](devices.md) describes every backend; the entries below cover the errors you are
most likely to see.

### `NotLinked` or `NoSuchDevice`

A `cubecl::Device` constructor returns `DeviceUnavailable::NotLinked` when your build does not
link that runtime: turn on the matching feature of the `cubecl` dependency, for example `cuda` for
`Device::cuda` ([Runtime features of CubeCL](getting-started/installation.md#runtime-features-of-cubecl)).
It returns `DeviceUnavailable::NoSuchDevice` when the runtime is linked but the machine has no
such device, or fewer than the index you asked for. Under MPI this happens when you start more
ranks per node than the node has GPUs ([One GPU per MPI rank](devices.md#one-gpu-per-mpi-rank)).
[Choosing a device at run time](devices.md#choosing-a-device-at-run-time) shows how to try
several devices in turn.

### `UnsupportedPrecision` on Metal and other GPUs

Creating an `f64` mesh fails with `GridError::UnsupportedPrecision` on a device without 64-bit
floating-point arithmetic. Metal never has it; Vulkan has it only if the adapter offers
`shaderFloat64`, and wgpu only with `SHADER_F64`. A reverse halo exchange of `f64` values fails
the same way. Use `f32` on these devices, and write your code generic over the scalar type so
that it chooses `f64` where the device supports it ([f64 support](devices.md#f64-support)).

### `AllocationTooLarge`

Every array of a mesh on one device must fit into a single allocation, whose largest size the
device sets. A larger mesh fails with `GridError::AllocationTooLarge`, which names the requested
size and the limit. The program `devices` prints the limit of your device. Spread such a mesh
over more ranks; fields at quadrature points are split into chunks automatically and are not
affected ([Choosing a device at run time](devices.md#choosing-a-device-at-run-time)).

### CUDA builds and the CUDA version

The CUDA backend needs the CUDA toolkit at build time, because the CUDA bindings read the CUDA
version from the toolkit's `nvcc`: set `CUDA_PATH` to the toolkit and put its `nvcc` on the
`PATH`. At run time only the NVIDIA driver is needed. CUDA 13 no longer supports Volta GPUs such
as the V100, so build for them with a CUDA 12 toolkit. With CubeCL's alternative compiler, the
cubecl feature `cuda-cpp`, the toolkit must not be newer than the CUDA version that `nvidia-smi`
reports for the driver; otherwise loading a kernel fails with
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` ([CUDA](devices.md#cuda)).

### The first call is slow

CubeCL compiles each kernel when it is first launched: with LLVM on the CPU runtime, to PTX
with LLVM on CUDA (which the driver then compiles for the GPU), and through the shader compiler
of the graphics API on wgpu backends. The first call of each tetraloma function, and the first
call with each new cell type, degree or quadrature rule, is therefore slower than later ones.
Time a computation after a first run, not including it.

### Errors from the compute server

`GridError::Server` passes on an error that the CubeCL runtime reported while it allocated
memory, launched a kernel or copied data. Its message is CubeCL's and names the cause. Try the
same program on the CPU runtime: if it runs there, the problem lies with the device or its
driver, not with your input.

## MPI problems

Distributed programs add the MPI library, the launcher and the placement of ranks to what can go
wrong. [Distributed meshes](tutorials/distributed.md) explains how a distributed mesh is built,
and [One GPU per MPI rank](devices.md#one-gpu-per-mpi-rank) how ranks share a node.

### Ranks that do not see each other

If every rank reports itself as rank 0 of 1, the launcher has started each process on its own. Most
often the program and the launcher come from different MPI implementations, for example a program
built against MPICH started with Open MPI's `mpirun`. Start the program with the `mpirun` or
`mpiexec` of the implementation it was built against. The same applies to HDF5: with `mpi` and
`hdf5` on, HDF5 must be built against the same MPI implementation as your program
([HDF5](features.md#hdf5)).

The MPICH package of Ubuntu 24.04 (MPICH 4.2.0) does the same, even with its own launcher. That
package is built for the PMIx process interface, which its launcher, hydra, does not provide, so
`mpirun -n 4` starts four separate one-rank programs (Launchpad bug 2072338). Use the MPICH of
Ubuntu 26.04 (4.3.2), an MPICH you build yourself, or Open MPI. The programs of tetraloma-examples
check their world size against the rank count that their run script requests, and stop at once if it
differs.

### `MPI_Init` hangs on a Mac

On Macs whose processors have three performance levels of cores, hwloc 2.15.0, which Homebrew's
Open MPI uses, never finishes discovering the processor's topology. Every Open MPI program, even
on one rank, then spins at 100% CPU inside `MPI_Init`. Set `HWLOC_COMPONENTS=-darwin` in the
environment of the run to switch off hwloc's macOS backend. hwloc then reports hardware threads
but no cores, so Open MPI's default placement by core fails: add `--map-by slot` to `mpirun`,
with `--oversubscribe` or with a host list such as `--host localhost:4`.

### Several ranks are slower than one

Each rank's CubeCL CPU runtime starts one worker thread per hardware thread that the rank may use.
Without binding, every rank may use every core of the node, so several ranks per node compete for
the same cores and run far more slowly than one. Bind each rank to its share of the cores: with
Open MPI, `mpirun --map-by node:PE=<cores per rank> --bind-to core`; with MPICH,
`mpiexec -bind-to core`. macOS cannot restrict a process to a set of cores, so on a Mac run
several ranks only with one GPU each, or one CPU rank per machine.

### A collective call hangs or fails on every rank

Collective functions, such as `build_on`, `measure`, `orient`, the halo exchanges, the readers and
the writers, must be called by every rank of the communicator, in the same order. A rank that
skips one leaves the others waiting for ever. When a collective fails on one rank, the others
return `GridError::RemoteFailure` with the lowest failing rank: look for that rank's own error,
which names the cause ([Building on a communicator](tutorials/distributed.md#building-on-a-communicator)).

### Partitioner errors

`GridError::PartitionerFailed` passes on a status code of a partitioner library; tetraloma's own
conditions use negative codes. The code −1 from ParMETIS means that a rank holds no cell before
partitioning, which ParMETIS does not allow: give every rank part of the input, or use PT-Scotch.
`GridError::NeedsMpiCommunicator` means that `PtScotch` or `ParMetis` was asked for more than
one part on a communicator that is not an `Mpi` communicator. Partitions that change from run to
run with the same seed can come from Scotch 7.0.18, whose threaded build is not reproducible; use
Scotch 7.0.14 ([Partitioners](features.md#partitioners)).

### `MessageTooLarge`

MPI counts and displacements are 32-bit signed integers. When one MPI message would hold more
than `i32::MAX` elements, `GridError::MessageTooLarge` reports it instead of letting MPI fail.
Spread the mesh over more ranks, so that each rank sends and receives less. A rank-0 partitioner
(`Metis`, `Scotch`) gathers the whole dual graph on rank 0 and can meet this limit on its own;
use `PtScotch` or `ParMetis` for very large meshes.

## Numerical problems

These entries cover results that look wrong and errors that the geometry and topology of a mesh
cause. [Precision](concepts/precision.md) explains which accuracy to expect in `f32` and `f64`,
and which results are reproducible bit for bit.

### Negative determinants or a negative measure

In codimension 0, $\det J$ is negative where a cell's map mirrors the reference cell, and
`measure` counts such cells negatively ([The determinant](concepts/geometric-factors.md#the-determinant)).
The usual cause is a vertex order that differs from tetraloma's. Triangles in the plane must list
their vertices counter-clockwise. Quadrilaterals and hexahedra list theirs in tensor order,
$(0, 0)$, $(1, 0)$, $(0, 1)$, $(1, 1)$, not counter-clockwise as Gmsh and VTK do: a row in
counter-clockwise order describes a twisted cell whose determinant changes sign inside it
([From arrays](tutorials/build-mesh.md#from-arrays)). The readers convert file orders for you;
the problem arises when you build a mesh from arrays.

### The pull-back does not converge

A point whose `last_step` is above the tolerance has not converged
([Convergence and failure](concepts/pull-back.md#convergence-and-failure)). If the point lies far
outside its cell, its reference point lies outside the box $[-0.1, 1.1]^t$ and the iteration
cannot reach it: the point belongs to another cell. If it lies inside a strongly curved cell, raise
`PullBackOptions::max_iterations` above its default of 16. In `f32`, a tolerance near the rounding
level, about $10^{-7}$, may never be met, because rounding alone changes $\xi$ by that much from
step to step; keep the default $10^{-5}$.

### Large coordinates in `f32`

tetraloma stores node coordinates relative to the centre of the mesh's bounding box, so a small
mesh far from the coordinate origin keeps nearly full `f32` accuracy
([Tolerances](concepts/precision.md#tolerances)). The origin removes a common offset, not a large
extent. A mesh that is $10^5$ units across with cells of size $10^{-3}$ still has coordinates near
$5 \times 10^4$ at its edges, where neighbouring `f32` numbers are about $0.004$ apart, so its
cells there lose most of their digits. Use `f64` for such meshes.

### Results that differ in the last digits

Results on different devices agree within the tolerances, $10^{-5}$ relative in `f32` and
$10^{-12}$ in `f64`, but not bit for bit, because compilers and hardware may contract or reorder
operations differently. Sums such as `measure` can also change in the last bits when you change
the vector size or the geometry strategy with the environment variables of
[Performance settings](devices.md#performance-settings). For a fixed setting on one device,
results are bitwise reproducible ([Determinism](concepts/precision.md#determinism)). Compare
results with a relative tolerance, not with `==`.

### Inaccurate geometry at high degree

Interpolation on equispaced nodes becomes badly conditioned as the degree grows, so meshes of
degree above about 10 lose accuracy. Use Gauss–Lobatto–Legendre nodes, `NodeFamily::Gll`, on
intervals, quadrilaterals and hexahedra; on other cell types they give
`GridError::UnsupportedNodeFamily` ([GLL and equispaced nodes](concepts/geometry.md#gll-and-equispaced-nodes)).
In `f32`, rounding ends the gain from higher degrees much earlier than in `f64`
([High-order geometry](tutorials/quadrature.md#high-order-geometry)).

### Orientation and submesh errors

`orient`, `boundary` and `facet_submesh` need a manifold, a mesh in which every facet belongs to
at most two cells; otherwise they return `GridError::NonManifold`. `orient` returns
`GridError::NonOrientable` for a surface without a consistent orientation, such as a Möbius strip,
and `GridError::NeedsReferencePoint` when `OrientationRule::Outward` meets a component with a
boundary, which encloses nothing: use `AwayFrom(p)` or `Towards(p)` for it.
`facet_submesh` takes boundary facets only and returns `GridError::InteriorFacet` for a tagged
facet with two cells ([Manifold orientation](concepts/orientation.md#manifold-orientation),
[Orientation](tutorials/tags-submeshes.md#orientation)). Normals that point inwards after reading
a file come from the vertex orders in the file: call `orient` with the rule you need.

### A file of unexpected dimensions

`SurfaceMesh::try_from` or `try_into` fails with `GridError::DimensionMismatch` when the file
holds another pair of dimensions. Without `GmshOptions::gdim`, the reader infers the geometric
dimension from the coordinates, so a surface whose nodes all have $z = 0$ reads as a planar mesh.
Read it with `GmshOptions::default().gdim(3)` to get a surface in three dimensions
([Reading Gmsh](tutorials/read-write.md#reading-gmsh)).
