//! The examples of tetraloma's API walkthrough (docs/design.md, Listings B8, B9 and B13;
//! ADR-0060 item 6).
//!
//! Each example under `examples/` is a thin `main` that picks its device as Listing B8 does
//! ([`device`]) and calls the `run` function of one module here; `run` picks the precision of
//! the device (f64 where the device supports it, f32 otherwise). `tests/examples.rs` runs every
//! `run` on the test device, and `scripts/run-examples.sh` runs the binaries.
//!
//! - [`read_gmsh`]: read a Gmsh file and fix the dimensions once (Listing B8);
//! - [`surface_submesh`]: a boundary surface mesh, its normals, Jacobians and area (Listing B9);
//! - [`geometric_factors`]: geometric factors at p = 1, 4 and 8 on a curved sphere;
//! - [`halo_exchange`]: `forward` and `reverse` on the node index map (Listing B13);
//! - [`vtk_output`]: VTU output (Listing B13).
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example surface_submesh
//! cargo run --release -p tetraloma-examples --features metal --example geometric_factors
//! mpirun -n 2 target/release/examples/halo_exchange   # built with --features mpi
//! ```

use cubecl::Device;
use cubecl::device::DeviceUnavailable;
use cubecl::prelude::{Client, CubeElement};

pub mod geometric_factors;
pub mod halo_exchange;
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
pub(crate) fn has_f64(client: &Client) -> bool {
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
