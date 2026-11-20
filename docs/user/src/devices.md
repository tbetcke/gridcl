# Compute devices

tetraloma runs on every device that CubeCL supports, and no tetraloma type names a backend. A
program picks a device at run time through `cubecl::Device`, gets its client, and passes the
client to the functions that create meshes and buffers. This chapter describes each backend, how
to choose one, which precisions it supports, how MPI ranks share GPUs, when data moves between
host and device, and which performance settings you can change.

The table summarises the backends. The cubecl feature is the one you turn on in your
`Cargo.toml` ([Installation](getting-started/installation.md#runtime-features-of-cubecl)).

| Backend | cubecl feature | Device | f64 | Tested |
| --- | --- | --- | --- | --- |
| CPU runtime | `cpu` | `Device::cpu()` | yes | on every change (macOS, Linux) |
| CUDA | `cuda` | `Device::cuda(index)` | yes | NVIDIA data-centre GPUs, after every kernel change |
| HIP | `hip` | `Device::rocm(index)` | yes | by hand |
| Vulkan | `vulkan` | `Device::vulkan(kind)` | if the adapter has `shaderFloat64` | continuous integration, software Vulkan |
| Metal | `metal` | `Device::metal_msl(kind)` | no | on every change (Apple silicon) |
| native Metal | `metal-native` | `Device::metal(kind)` | no | not tested |
| wgpu (WGSL) | `wgpu` | `Device::wgpu(kind)` | with `SHADER_F64` | best effort |

The program `devices` of the repository opens a device and prints what this chapter describes:

```sh
cargo run --release -p tetraloma-examples --example devices
```

## The CPU runtime

The CubeCL CPU runtime runs kernels on the host's processor, so it needs no GPU. It is the
reference backend: every test runs on it, in `f32` and `f64`. LLVM compiles each kernel for the
host CPU the first time it is launched, which makes the first call of each tetraloma function
slower than later ones.

A kernel launch runs on one worker thread per hardware thread. Each thread processes contiguous
blocks of cells or elements, and tetraloma's kernels use the CPU's SIMD registers through CubeCL's
vector types (NEON on Apple silicon and Arm servers, AVX2 or AVX-512 on x86-64). The CPU runtime
reports a plane size of 1 (a plane is a group of units that execute in lockstep, a warp on NVIDIA
GPUs), and every tetraloma algorithm is correct with it.

Results on the CPU runtime are bitwise reproducible. Kernels that compute each cell or point
independently, such as the geometric factors, also give the same bits at every vector width.
On macOS, some kernels run on the performance cores only, where that is faster.

## Metal

On Apple silicon, the supported path to the GPU is wgpu with Metal shading language, the cubecl
feature `metal`, opened with `Device::metal_msl(WgpuDeviceKind::DefaultDevice)`. Metal has no
64-bit floating-point arithmetic, so meshes on Metal use `f32`; an `f64` mesh is refused (see
[f64 support](#f64-support)). tetraloma is tested on Metal on every change.

CubeCL also has a native Metal runtime, the feature `metal-native` and `Device::metal`.
tetraloma does not test it; use the wgpu path unless you have a reason not to.

## CUDA

The cubecl feature `cuda` runs kernels on NVIDIA GPUs, and `Device::cuda(index)` opens the GPU
with that index. CUDA supports `f32` and `f64`. CubeCL compiles each kernel to PTX with its own
LLVM, and the NVIDIA driver compiles the PTX for the GPU when the kernel is first loaded.

The build needs the CUDA toolkit, because the CUDA bindings read the CUDA version from the
toolkit's `nvcc` at build time: set `CUDA_PATH` to the toolkit and put its `nvcc` on the `PATH`.
At run time only the driver is needed. Two rules about versions follow:

- **Volta GPUs (V100) need CUDA 12.** CUDA 13 dropped Volta (compute capability 7.0), so build
  for these GPUs with a CUDA 12.x toolkit. Ampere (A100) and Hopper (GH200) GPUs work with CUDA 12
  and 13.
- **The toolkit must not be newer than the driver with `cuda-cpp`.** CubeCL's alternative CUDA
  compiler, the cubecl feature `cuda-cpp`, compiles CUDA C++ with NVRTC from the toolkit. The
  driver must then understand the PTX that the toolkit emits, so the toolkit's CUDA version must
  not be newer than the one `nvidia-smi` reports for the driver. Otherwise loading a kernel
  fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`. The default `cuda` path has no such rule.

## Vulkan

The cubecl feature `vulkan` runs kernels through wgpu on Vulkan, with kernels compiled to SPIR-V.
`Device::vulkan(WgpuDeviceKind::DefaultDevice)` opens the default adapter;
`WgpuDeviceKind::DiscreteGpu(i)` and `IntegratedGpu(i)` choose one of several. Vulkan works on
Linux and Windows with the GPU vendor's driver, and on machines without a GPU with a software
driver such as Mesa's lavapipe, which tetraloma's continuous integration uses.

Whether `f64` works depends on the adapter: it needs the Vulkan feature `shaderFloat64`. Check
for it at run time as [f64 support](#f64-support) shows.

## HIP and wgpu

The cubecl feature `hip` runs kernels on AMD GPUs through ROCm, with `Device::rocm(index)`. It
supports `f32` and `f64`. tetraloma tests HIP by hand only, not after every change.

The cubecl feature `wgpu` lets wgpu choose among the graphics APIs the machine offers, such as
DirectX 12 on Windows, and compiles kernels to WGSL.
`Device::wgpu(WgpuDeviceKind::DefaultDevice)` opens it. `f64` needs the wgpu feature
`SHADER_F64`. This backend is supported on a best-effort basis. tetraloma's kernels use at most
six buffers each and no 64-bit or floating-point atomics, so they fit WebGPU-class devices.

## Choosing a device at run time

Each `Device` constructor exists in every build. When the build does not link a runtime, the
constructor returns `DeviceUnavailable::NotLinked`; when the machine has no such device, it
returns `DeviceUnavailable::NoSuchDevice`. A program can therefore try the devices it prefers in
turn, and the same source works with any set of features. In a build with only the CPU runtime,
this falls through to the CPU:

```rust
use cubecl::Device;
use cubecl::device::WgpuDeviceKind;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

// The first device that opens: a CUDA GPU, a Vulkan or Metal GPU, else the CPU.
let device = Device::cuda(0)
    .or_else(|_| Device::vulkan(WgpuDeviceKind::DefaultDevice))
    .or_else(|_| Device::metal_msl(WgpuDeviceKind::DefaultDevice))
    .or_else(|_| Device::cpu())?;
let client = device.client();
println!("running on {}", client.name());
# Ok(())
# }
```

The program `devices` goes one step further and reads the device's name from the environment
variable `TETRALOMA_TEST_DEVICE`, which tetraloma's own tests use for the same purpose. Its
function `open` maps a name to a device; an empty name tries the devices in turn:

```rust,ignore
{{#include ../../../tetraloma-examples/src/devices.rs:open}}
```

Its `main` reads the variable, opens the device and runs the rest of the program in `f64` or
`f32`:

```rust,ignore
{{#include ../../../tetraloma-examples/examples/devices.rs:main}}
```

On a Mac, this runs it on Metal:

```sh
TETRALOMA_TEST_DEVICE=metal cargo run --release -p tetraloma-examples --features metal --example devices
```

The program then prints the device's properties, read from the client:

```rust,ignore
{{#include ../../../tetraloma-examples/src/devices.rs:properties}}
```

Of these, the largest allocation limits the size of a mesh: every array of a mesh on one device
(on one MPI rank) must fit into a single allocation of this size, or mesh creation fails with
`GridError::AllocationTooLarge`. Spread such a mesh over more ranks. Fields at quadrature points
are split into chunks automatically and have no such limit.

## f64 support

Every mesh has a scalar type, `f32` or `f64`, and tetraloma checks it against the device whenever
a mesh is created. On a device without `f64`, creating an `f64` mesh fails with
[`GridError::UnsupportedPrecision`](api/tetraloma/enum.GridError.html) before any kernel runs.
The program `devices` shows it:

```rust,ignore
{{#include ../../../tetraloma-examples/src/devices.rs:mesh}}
```

To support every device, write the mesh code generic over the scalar and choose `F` from the
device's properties:

```rust
use cubecl::prelude::{Client, CubeElement};
use tetraloma::prelude::*;

/// The length of [0, 2] as a mesh of two intervals, computed in `F`.
fn length<F: Real>(client: &Client) -> tetraloma::Result<f64> {
    let p1 = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    let mesh: Mesh<F, D1, D1> = MeshBuilder::new(client)
        .nodes(&[0.0, 0.5, 2.0], None)
        .cell_block(p1, &[0, 1, 1, 2], None)
        .build()?;
    mesh.measure(1)
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
let has_f64 = client.properties().supports_type(<f64 as CubeElement>::cube_type());
// The CPU runtime supports f64; Metal does not.
assert!(has_f64);
let total = if has_f64 { length::<f64>(&client)? } else { length::<f32>(&client)? };
assert!((total - 2.0).abs() < 1e-12);
# Ok(())
# }
```

tetraloma's tests hold results to a relative tolerance of $10^{-5}$ in `f32` and $10^{-12}$ in
`f64`. Quadrature rules and basis tables are computed in `f64` on the host and rounded to
`F` once, and coordinates are stored relative to the mesh's centre, which keeps `f32` results
close to the `f64` ones. [Precision](concepts/precision.md) explains the rules.

## One GPU per MPI rank

An MPI program runs one process, a *rank*, per device. Ranks on the same node must open different
GPUs. [`local_gpu_index`](api/tetraloma/fn.local_gpu_index.html) returns the rank's index among the
ranks of its node, which is the GPU it should open:

```rust
use cubecl::Device;
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

// Under MPI, `comm` is `Mpi::world(&universe)`; one process alone is `Serial`.
let comm = Serial;
let gpu = local_gpu_index(&comm);
assert_eq!(gpu, 0);
// GPU `gpu` of this node, or the CPU runtime where there is no CUDA GPU.
let client = Device::cuda(gpu).or_else(|_| Device::cpu())?.client();
println!("rank {} runs on {}", comm.rank(), client.name());
# Ok(())
# }
```

Start as many ranks per node as the node has GPUs, for example `mpirun -n 4` on a node with four
GPUs. `local_gpu_index` does not know how many GPUs there are: with more ranks than GPUs,
`Device::cuda` fails on the extra ranks, and the code above falls back to the CPU there.

Each rank works on its mesh from one host thread: a [`Mesh`](api/tetraloma/struct.Mesh.html) is not
`Send`, so the compiler stops you from moving it to another thread. tetraloma does not initialise
MPI: your program does, with `mpi::initialize()` from the `mpi` crate, and wraps the world
communicator in tetraloma's `Mpi` ([Distributed meshes](tutorials/distributed.md)). Halo
exchanges copy the values to the host, send them with MPI and copy them back ([Halo
exchange](tutorials/halo-exchange.md)). Sending directly from GPU memory, with a GPU-aware
MPI, is planned but not yet available ([Reserved features](features.md#reserved-features)).

On CPU nodes, each rank's CPU runtime starts one thread per hardware thread that the rank may
use. If you run several ranks per node, bind each rank to its share of the cores (Open MPI:
`--map-by node:PE=<cores per rank> --bind-to core`), or the ranks' threads compete for the same
cores.

## Host–device transfers

The data of a mesh stays on the device, and tetraloma copies between host and device only where
you ask for it or where it cannot be avoided. Every copy blocks until it is done, so a copy inside
a loop over cells is slow; copy whole arrays once instead.

These calls copy:

- **Mesh creation** uploads the input arrays once. Reading a file reads it on the host and uploads
  the mesh.
- **Quadrature rules** are computed on the host and their tables are uploaded the first time a
  mesh uses them, then cached.
- **Explicit copies**: [`DeviceBuffer::from_host`](api/tetraloma/struct.DeviceBuffer.html),
  `write_host` and `to_host`, and `to_host_cell_major` of a
  [`QuadField`](api/tetraloma/struct.QuadField.html). Use them to give a kernel its input and to
  look at results.
- **`Mesh::read_node`** copies the whole node array to read one node: for tests and debugging only.
- **`measure`** sums each block on the device and reads one number per block, which it adds in
  `f64` on the host.
- **File output** and **halo exchange** copy to the host, by their nature.

Everything else, from geometric factors to connectivity, is computed on the device and returned
as device arrays. Distributed meshes add the exchanges between ranks, which go through the host.
Buffers are created, written and read like this:

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let client = cubecl::Device::cpu()?.client();
// Host to device.
let mut values = DeviceBuffer::from_host(&client, &[1.0f64, 2.0, 3.0])?;
// Overwrite on the device from the host; the length must match.
values.write_host(&[4.0, 5.0, 6.0])?;
// Device to host, blocking.
assert_eq!(values.to_host()?, [4.0f64, 5.0, 6.0]);
# Ok(())
# }
```

## Performance settings

Build with `--release`. The kernels are compiled at run time and are equally fast in either
profile, but the host code of tetraloma and CubeCL is much slower in a debug build.

tetraloma chooses vector widths, tile widths and kernel strategies from built-in policies, which
were measured on CPUs and GPUs of several vendors. The defaults are good on every platform tested,
so you need not change them. To experiment, three environment variables override the policies.
Each is read once, when tetraloma first needs it, and an invalid value stops the program with a
panic that names the valid ones.

| Variable | Values | Overrides |
| --- | --- | --- |
| `TETRALOMA_VECTOR_SIZE` | `policy`, `scalar`, `widest`, or a power of two from 1 to 16 | the number of SIMD lanes of every vectorised kernel |
| `TETRALOMA_QUAD_TILE` | 1, 2, 4, 8 or 16 | how many cells are interleaved in fields at quadrature points |
| `TETRALOMA_GEOM_STRATEGY` | `policy`, `direct`, `loops`, `sumfact`, `shared`, `cooperative`, or a list such as `Jacobian=sumfact,Points=loops` | the strategy of the geometric-factor kernels |

The vector size is capped by what the device and the data allow, so a large value is safe. A
strategy that does not apply to a cell type or rule falls back to `loops`. These settings are
meant to change speed; results agree within the rounding tolerances whatever you choose. On the
CPU runtime, kernels that treat cells or points independently give the same bits at every vector
size, while sums such as `measure` add their terms in an order that depends on the vector size,
and the strategies of the geometric factors round differently. For a fixed setting on one device,
results are bitwise reproducible.

```sh
TETRALOMA_VECTOR_SIZE=scalar cargo run --release -p tetraloma-examples --example first_program
```

Other choices are yours to make in the program: the polynomial degree of quadrature rules, and,
for distributed meshes, the partitioner and the cell ordering ([Distributed
meshes](tutorials/distributed.md)). Large meshes on one device are limited by the largest
allocation the device allows, which the program `devices` prints.
