//! The device a build opens, and what the book's chapter "Compute devices" describes of it.
//!
//! [`open`] turns a device name (for example the value of [`DEVICE_VAR`]) into a CubeCL
//! device, and [`run`] prints the device's properties and builds a small mesh on it.

use std::error::Error;

use cubecl::Device;
use cubecl::device::{MetalDevice, WgpuDeviceKind};
use cubecl::prelude::{Client, CubeElement};
use tetraloma::prelude::*;

/// The environment variable that names the device of the `devices` program: `cpu`, `cuda`,
/// `hip`, `vulkan`, `metal`, `metal-native` or `wgpu`, or unset for the first device the build
/// can open. tetraloma's own tests read the same variable.
pub const DEVICE_VAR: &str = "TETRALOMA_TEST_DEVICE";

// ANCHOR: open
/// The device `name` selects; `gpu` is the GPU index of CUDA and HIP. An empty name tries CUDA,
/// HIP, Vulkan, Metal and the CPU runtime in turn and takes the first that opens.
///
/// # Errors
///
/// `DeviceUnavailable` if the build does not link the runtime (turn on its cubecl feature) or
/// the machine has no such device, and an error for an unknown name.
///
/// # Example
///
/// ```
/// let client = tetraloma_examples::devices::open("cpu", 0)?.client();
/// assert_eq!(client.name(), "cpu");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn open(name: &str, gpu: usize) -> Result<Device, Box<dyn Error>> {
    let device = match name {
        // Every constructor exists in every build: a runtime the build does not link is
        // `DeviceUnavailable::NotLinked`, and `or_else` moves on to the next.
        "" => Device::cuda(gpu)
            .or_else(|_| Device::rocm(gpu))
            .or_else(|_| Device::vulkan(WgpuDeviceKind::DefaultDevice))
            .or_else(|_| Device::metal_msl(WgpuDeviceKind::DefaultDevice))
            .or_else(|_| Device::cpu())?,
        "cpu" => Device::cpu()?,
        "cuda" => Device::cuda(gpu)?,
        "hip" => Device::rocm(gpu)?,
        "vulkan" => Device::vulkan(WgpuDeviceKind::DefaultDevice)?,
        "metal" => Device::metal_msl(WgpuDeviceKind::DefaultDevice)?,
        "metal-native" => Device::metal(MetalDevice::DefaultDevice)?,
        "wgpu" => Device::wgpu(WgpuDeviceKind::DefaultDevice)?,
        other => {
            return Err(format!(
                "unknown device '{other}': expected cpu, cuda, hip, vulkan, metal, \
                 metal-native or wgpu"
            )
            .into());
        }
    };
    Ok(device)
}
// ANCHOR_END: open

/// Prints the runtime, the name and the properties of `client`'s device (f64 support, the
/// limits tetraloma respects, the widest vector size for `F`), and builds a small mesh in `F`
/// on it.
///
/// The mesh is the unit square as two triangles; `run` checks that its area is 1. On a device
/// without f64 it also checks that an f64 mesh is refused with `UnsupportedPrecision`. It checks
/// that the device offers at least the six buffers per kernel that tetraloma's kernels use.
///
/// # Errors
///
/// The errors of mesh creation and of `measure`; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client) -> tetraloma::Result<()> {
    // ANCHOR: properties
    let properties = client.properties();
    let hardware = &properties.hardware;
    let f64_support = properties.supports_type(<f64 as CubeElement>::cube_type());
    println!("runtime:               {}", client.name());
    println!("device:                {}", properties.identity.name);
    println!(
        "f64:                   {}",
        if f64_support { "yes" } else { "no" }
    );
    println!(
        "largest allocation:    {} MiB",
        properties.memory.max_page_size >> 20
    );
    match properties.memory.max_memory() {
        Some(bytes) => println!("memory:                {} MiB", bytes >> 20),
        None => println!("memory:                not reported"),
    }
    println!("buffers per kernel:    {}", hardware.max_bindings);
    println!(
        "shared memory:         {} KiB per cube",
        hardware.max_shared_memory_size >> 10
    );
    println!("units per cube:        {}", hardware.max_units_per_cube);
    println!(
        "plane size:            {} to {}",
        hardware.plane_size_min, hardware.plane_size_max
    );
    match hardware.num_cpu_cores {
        Some(cores) => println!("CPU threads:           {cores}"),
        None => println!(
            "multiprocessors:       {}",
            hardware
                .num_streaming_multiprocessors
                .map_or_else(|| "not reported".to_string(), |n| n.to_string())
        ),
    }
    let widest = properties
        .io_optimized_vector_sizes(size_of::<F>())
        .next()
        .unwrap_or(1);
    println!(
        "widest vector:         {widest} lanes of {}",
        std::any::type_name::<F>()
    );
    // ANCHOR_END: properties
    assert!(
        hardware.max_bindings >= 6,
        "{} offers {} buffers per kernel; tetraloma's kernels use up to 6",
        properties.identity.name,
        hardware.max_bindings
    );
    assert!(properties.memory.max_page_size > 0 && widest.is_power_of_two());

    // ANCHOR: mesh
    // The unit square as two triangles of degree 1.
    let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let triangles = [0, 1, 2, 3, 2, 1];
    let p1 = CoordinateElement::<D2> {
        cell: Cell2::Triangle,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    let mesh: PlanarMesh<F> = MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(p1, &triangles, None)
        .build()?;
    let area = mesh.measure(1)?;
    println!("a mesh in {}: area {area}", std::any::type_name::<F>());

    if !f64_support {
        // Mesh creation checks the precision: an f64 mesh fails here, before any kernel runs.
        let refused: tetraloma::Result<PlanarMesh<f64>> = MeshBuilder::new(client)
            .nodes(&coords, None)
            .cell_block(p1, &triangles, None)
            .build();
        match refused {
            Err(error @ GridError::UnsupportedPrecision { .. }) => {
                println!("a mesh in f64: refused ({error})");
            }
            other => panic!("an f64 mesh without f64 support: {:?}", other.err()),
        }
    }
    // ANCHOR_END: mesh
    let tol = 10.0 * crate::tol::<F>();
    assert!((area - 1.0).abs() <= tol, "the area is {area}, not 1");
    Ok(())
}
