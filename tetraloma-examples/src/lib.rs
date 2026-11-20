// Design: "API walkthrough", Listings B8, B9 and B13 (read_gmsh, surface_submesh,
// halo_exchange, vtk_output; ADR-0060, item 6); the book's programs: ADR-0067.
//! The example programs of tetraloma: complete programs that use the library as an application
//! does, and the programs of the [user book].
//!
//! This crate lives in the repository and is not published. It is for you if you want to run a
//! working program, change it and see what happens, or start your own program from one. The
//! book explains each program step by step and includes its code; every Rust block of the book
//! is compiled and run as a test of this crate.
//!
//! Each example under `examples/` is a thin `main` that picks its device ([`device`]) and calls
//! the `run` function of one module here. The walkthrough's `run` picks the precision of the
//! device (f64 where the device supports it, f32 otherwise); the book's programs take the scalar
//! as a type parameter, and their `main` picks it with [`has_f64`]. Each program checks its own
//! results with `assert!`, and the crate's tests run every `run` on the test device. The MPI
//! programs check their world size first (`check_world_size`, feature `mpi`).
//!
//! The walkthrough:
//!
//! - [`read_gmsh`]: read a Gmsh file and fix the dimensions once ([Reading and writing files]);
//! - [`surface_submesh`]: a boundary surface mesh, its normals, Jacobians and area
//!   ([Tags and submeshes]);
//! - [`geometric_factors`]: geometric factors at p = 1, 4 and 8 on a curved sphere
//!   ([Geometric data]);
//! - [`halo_exchange`]: `forward` and `reverse` on the node index map, under MPI ([Halo exchange]);
//! - [`vtk_output`]: VTU output ([Reading and writing files]).
//!
//! The book's programs:
//!
//! - [`first_program`]: a mesh, its geometric data and a VTU file ([A first program]);
//! - [`devices`]: the device the build opens and its properties ([Compute devices]);
//! - [`build_mesh`]: meshes from arrays ([Building a mesh]);
//! - [`iterate_cells`]: blocks, cells, entities and connectivity ([Iterating over a mesh]);
//! - [`pull_back`]: physical points to reference coordinates ([Pulling back points]);
//! - [`distributed_mesh`]: partitioners, ghost modes and cell ordering under MPI
//!   ([Distributed meshes]);
//! - [`parallel_io`]: Gmsh in, PVTU and VTKHDF out under MPI ([Parallel I/O]);
//! - [`custom_kernel`]: a downstream CubeCL kernel over a mesh ([Writing kernels]).
//!
//! [`figures`] generates the book's SVG figures of reference cells and node orderings (the
//! binary `figures` writes them), and a `#[cfg(doctest)]` module runs the Rust blocks of every
//! book chapter as doctests.
//!
//! The features `cpu` (the default), `cuda`, `vulkan` and `metal` choose the CubeCL runtimes the
//! programs can open; `mpi` builds the MPI programs (`halo_exchange`, `distributed_mesh`,
//! `parallel_io`) and `hdf5` the VTKHDF part of `parallel_io`. See [Cargo features]. To run a
//! program:
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example surface_submesh
//! cargo run --release -p tetraloma-examples --features metal --example geometric_factors
//! mpirun -n 2 target/release/examples/halo_exchange   # built with --features mpi
//! ```
//!
//! [user book]: https://tetracl.github.io/tetraloma/
//! [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html
//! [Tags and submeshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html
//! [Geometric data]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html
//! [Halo exchange]: https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html
//! [A first program]: https://tetracl.github.io/tetraloma/getting-started/first-program.html
//! [Compute devices]: https://tetracl.github.io/tetraloma/devices.html
//! [Building a mesh]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html
//! [Iterating over a mesh]: https://tetracl.github.io/tetraloma/tutorials/iterate.html
//! [Pulling back points]: https://tetracl.github.io/tetraloma/tutorials/pull-back.html
//! [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html
//! [Parallel I/O]: https://tetracl.github.io/tetraloma/tutorials/parallel-io.html
//! [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html
//! [Cargo features]: https://tetracl.github.io/tetraloma/features.html

use cubecl::Device;
use cubecl::device::DeviceUnavailable;
use cubecl::prelude::{Client, CubeElement};

#[cfg(doctest)]
mod book;
pub mod build_mesh;
pub mod custom_kernel;
pub mod devices;
pub mod distributed_mesh;
pub mod figures;
pub mod first_program;
pub mod geometric_factors;
pub mod halo_exchange;
pub mod iterate_cells;
pub mod parallel_io;
pub mod pull_back;
pub mod read_gmsh;
pub mod surface_submesh;
pub mod vtk_output;

/// The device of Listing B8 among the runtimes this build links: CUDA GPU `gpu` (feature
/// `cuda`), then Vulkan (`vulkan`), then Metal through wgpu (`metal`), then the CPU runtime
/// (`cpu`, the default); the first that opens.
///
/// `gpu` is the node-local GPU index, `local_gpu_index(&comm)` under MPI and 0 otherwise; only
/// CUDA uses it.
///
/// # Errors
///
/// The error of the last runtime tried if none opens, or `NotLinked` if the build links no
/// runtime.
///
/// # Example
///
/// ```
/// // The default build links the CPU runtime only.
/// let client = tetraloma_examples::device(0)?.client();
/// println!("running on {}", client.name());
/// # Ok::<(), cubecl::device::DeviceUnavailable>(())
/// ```
pub fn device(gpu: usize) -> Result<Device, DeviceUnavailable> {
    // One `cfg` per runtime feature: a runtime the build does not link is never tried.
    #[allow(unused_mut)]
    let mut device = Err(DeviceUnavailable::NotLinked(cubecl::device::RuntimeId::Cpu));
    #[cfg(feature = "cuda")]
    {
        device = device.or_else(|_| Device::cuda(gpu));
    }
    #[cfg(feature = "vulkan")]
    {
        use cubecl::device::WgpuDeviceKind;
        device = device.or_else(|_| Device::vulkan(WgpuDeviceKind::DefaultDevice));
    }
    #[cfg(feature = "metal")]
    {
        use cubecl::device::WgpuDeviceKind;
        device = device.or_else(|_| Device::metal_msl(WgpuDeviceKind::DefaultDevice));
    }
    #[cfg(feature = "cpu")]
    {
        device = device.or_else(|_| Device::cpu());
    }
    let _ = gpu;
    device
}

// The rank-count guard of ADR-0068.
/// The rank-count guard of the MPI examples: an error if the size of `comm` differs
/// from `TETRALOMA_EXPECT_RANKS`, which `scripts/run-examples.sh` exports to every rank; unset,
/// any size passes ([`tetraloma_testing::check_world_size`]).
///
/// # Errors
///
/// The guard's message, which names the world size and the requested count.
///
/// # Example
///
/// ```no_run
/// use tetraloma::Mpi;
///
/// let universe = mpi::initialize().ok_or("MPI already initialised")?;
/// let comm = Mpi::world(&universe);
/// tetraloma_examples::check_world_size(&comm)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[cfg(feature = "mpi")]
pub fn check_world_size(comm: &tetraloma::Mpi) -> Result<(), String> {
    use tetraloma::CommDyn;

    tetraloma_testing::check_world_size(comm.size())
}

/// Whether `client`'s device supports f64: the examples run in f64 there and in f32 otherwise
/// (Listing B8: an f64 mesh on a device without f64 is `UnsupportedPrecision`).
///
/// # Example
///
/// ```
/// let client = tetraloma_examples::device(0)?.client();
/// let scalar = if tetraloma_examples::has_f64(&client) { "f64" } else { "f32" };
/// println!("running in {scalar}");
/// # Ok::<(), cubecl::device::DeviceUnavailable>(())
/// ```
pub fn has_f64(client: &Client) -> bool {
    client
        .properties()
        .supports_type(<f64 as CubeElement>::cube_type())
}

/// The name of the scalar `F`, for the printed output.
pub(crate) fn scalar_name<F>() -> &'static str {
    if size_of::<F>() == 8 { "f64" } else { "f32" }
}

/// The relative tolerance of rounding in `F` (docs/design.md, "Testing and validation").
pub(crate) fn tol<F>() -> f64 {
    if size_of::<F>() == 8 { 1e-12 } else { 1e-5 }
}
