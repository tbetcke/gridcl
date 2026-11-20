//! The example programs of tetraloma: the API walkthrough (docs/design.md, Listings B8, B9 and
//! B13; ADR-0060 item 6) and the programs of the user book (docs/user; ADR-0067).
//!
//! Each example under `examples/` is a thin `main` that picks its device as Listing B8 does
//! ([`device`]) and calls the `run` function of one module here. The walkthrough's `run` picks
//! the precision of the device (f64 where the device supports it, f32 otherwise); the book's
//! programs take the scalar as a type parameter, and their `main` picks it with [`has_f64`].
//! `tests/examples.rs` runs every `run` on the test device, and `scripts/run-examples.sh` runs
//! the binaries.
//!
//! The walkthrough:
//!
//! - [`read_gmsh`]: read a Gmsh file and fix the dimensions once (Listing B8);
//! - [`surface_submesh`]: a boundary surface mesh, its normals, Jacobians and area (Listing B9);
//! - [`geometric_factors`]: geometric factors at p = 1, 4 and 8 on a curved sphere;
//! - [`halo_exchange`]: `forward` and `reverse` on the node index map (Listing B13);
//! - [`vtk_output`]: VTU output (Listing B13).
//!
//! The book's programs:
//!
//! - [`first_program`]: a mesh, its geometric data and a VTU file ("A first program");
//! - [`devices`]: the device the build opens and its properties ("Compute devices");
//! - [`build_mesh`]: meshes from arrays (tutorial "Building a mesh");
//! - [`iterate_cells`]: blocks, cells, entities and connectivity ("Iterating over a mesh");
//! - [`pull_back`]: physical points to reference coordinates ("Pulling back points");
//! - [`distributed_mesh`]: partitioners, ghost modes and cell ordering under MPI
//!   ("Distributed meshes");
//! - [`parallel_io`]: Gmsh in, PVTU and VTKHDF out under MPI ("Parallel I/O");
//! - [`custom_kernel`]: a downstream CubeCL kernel over a mesh ("Writing kernels").
//!
//! [`figures`] generates the book's SVG figures of reference cells and node orderings (the
//! binary `figures` writes them), and a `#[cfg(doctest)]` module runs the Rust blocks of every
//! book chapter as doctests.
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example surface_submesh
//! cargo run --release -p tetraloma-examples --features metal --example geometric_factors
//! mpirun -n 2 target/release/examples/halo_exchange   # built with --features mpi
//! ```

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
