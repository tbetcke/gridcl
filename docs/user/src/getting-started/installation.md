# Installation

You need a Rust toolchain, a dependency on tetraloma and on CubeCL in your project's `Cargo.toml`,
and at least one CubeCL runtime feature that decides which devices your program can use. Optional
features such as MPI or HDF5 output need system libraries as well. This chapter goes through these
steps in order.

## Toolchain

tetraloma needs stable Rust 1.99 or newer and uses the 2024 edition. Install Rust with
[rustup](https://rustup.rs), which also keeps it up to date:

```sh
rustup update stable
rustc --version
```

You also need a C compiler and linker: on macOS the Xcode command-line tools
(`xcode-select --install`), on Linux the distribution's build tools (for example the Ubuntu package
`build-essential`).

The CPU runtime and the CUDA backend of CubeCL compile kernels with LLVM. CubeCL does not use a
system LLVM: the build script of its `cubecl-llvm` crate downloads a prebuilt LLVM the first time it
builds, so this first build needs network access. The download is kept in the directory `tracel`
under your user data directory (on Linux the directory that `XDG_DATA_HOME` names, by default
`~/.local/share`), and later builds reuse it. On Linux this prebuilt LLVM needs glibc 2.34 or
newer; older systems, such as Red Hat Enterprise Linux 7, need an LLVM of the same version built
locally and placed where the build script looks for it.

## Adding tetraloma to a project

tetraloma is not on crates.io yet, so you depend on its Git repository. Your project also depends
on CubeCL itself, at exactly the version tetraloma uses, because you create the device and its
client with CubeCL's API:

```toml
[dependencies]
tetraloma = { git = "https://github.com/tetracl/tetraloma" }
cubecl = { version = "=0.11.0", features = ["cpu"] }
```

The exact version `=0.11.0` matters. CubeCL is young and announces breaking changes between minor
versions, and a client from another CubeCL version would be a different type from the one
tetraloma's functions take. The feature `cpu` turns on the CPU runtime; the next section lists the
others.

tetraloma's own optional features, such as `mpi` or `hdf5`, go on the tetraloma dependency:

```toml
[dependencies]
tetraloma = { git = "https://github.com/tetracl/tetraloma", features = ["mpi", "hdf5"] }
cubecl = { version = "=0.11.0", features = ["cpu", "cuda"] }
mpi = { version = "=0.8.2", default-features = false }
```

The chapter [Cargo features](../features.md) describes each of them. A program that uses MPI
depends on the [rsmpi](https://github.com/rsmpi/rsmpi) crate `mpi` as well, to initialise MPI.
Use the version tetraloma uses, 0.8.2, so that both share one MPI environment type.

In your code, the prelude brings the names most programs need into scope. The check below opens
the CPU runtime and confirms that your setup builds and runs:

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let client = cubecl::Device::cpu()?.client();
println!("tetraloma runs on {}", client.name());
assert_eq!(client.name(), "cpu");

// A device buffer: three numbers copied to the device and back.
let buffer = DeviceBuffer::from_host(&client, &[1.0f32, 2.0, 3.0])?;
assert_eq!(buffer.to_host()?, [1.0f32, 2.0, 3.0]);
# Ok(())
# }
```

Build your program with `--release` for real work. Debug builds check more, but their host code is
much slower.

## Runtime features of CubeCL

tetraloma's library crates enable no CubeCL runtime. Your program does, with features of the
`cubecl` dependency, and chooses one of the linked runtimes when it runs. Each runtime is
independent of the others, so you can link several and decide at run time which device to open.

| cubecl feature | Backend | Device | Needs |
| --- | --- | --- | --- |
| `cpu` | the CubeCL CPU runtime | `Device::cpu()` | LLVM, downloaded at build time |
| `cuda` | CUDA, NVIDIA GPUs | `Device::cuda(index)` | the CUDA toolkit at build time, the NVIDIA driver at run time |
| `hip` | HIP, AMD GPUs | `Device::rocm(index)` | ROCm |
| `vulkan` | wgpu with SPIR-V on Vulkan | `Device::vulkan(kind)` | a Vulkan driver |
| `metal` | wgpu with MSL on Apple GPUs | `Device::metal_msl(kind)` | macOS |
| `metal-native` | the native Metal runtime | `Device::metal(kind)` | macOS |
| `wgpu` | wgpu with WGSL (DirectX 12, WebGPU, …) | `Device::wgpu(kind)` | a graphics driver |

The CPU runtime needs no GPU and supports `f64`, so it is the natural start and the runtime this
book's examples use. Metal does not support `f64`. The chapter [Compute devices](../devices.md)
describes every backend, the CUDA toolkit and driver versions, and how to choose a device at run
time.

## System libraries

The default build needs nothing beyond the toolchain above. The GPU backends need their drivers,
and the optional features of tetraloma need these libraries:

| Feature | Libraries |
| --- | --- |
| `mpi` | an MPI implementation (MPICH or Open MPI) and libclang |
| `hdf5` | HDF5 1.10.10 to 2.2.0, built with MPI if `mpi` is on as well |
| `metis` | none: METIS is built from source |
| `scotch` | Scotch |
| `ptscotch` | Scotch with PT-Scotch, and MPI |
| `parmetis` | ParMETIS and its METIS, and MPI |

On Ubuntu, for example, these packages provide them with Open MPI:

```sh
sudo apt-get install openmpi-bin libopenmpi-dev libclang-dev libhdf5-openmpi-dev \
    libscotch-dev libptscotch-dev libparmetis-dev libmetis-dev
```

With MPICH, install `mpich libmpich-dev libhdf5-mpich-dev` instead of the three Open MPI
packages. On macOS, Homebrew provides `open-mpi` or `mpich`, and `hdf5-mpi`. Use a Scotch release
that tetraloma has verified, such as 7.0.14 built from source: Scotch 7.0.18 does not give
reproducible partitions. [Cargo features](../features.md) says where the build looks for each
library and which environment variables point it elsewhere.
