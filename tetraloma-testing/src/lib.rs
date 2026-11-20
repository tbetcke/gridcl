// Design: "Cargo features", "Dependency rules"; ADR-0059, ADR-0060.
//! Test support for tetraloma: the test device chosen by `TETRALOMA_TEST_DEVICE`, test meshes,
//! tolerance helpers, a threaded communicator and the Python oracles.
//!
//! This crate lives in the repository and is not published. It is for the developers of tetraloma:
//! the library's tests, its API examples and the benchmarks get their device and their test meshes
//! here. A program of your own opens its device through `cubecl::Device` and builds its meshes with
//! the mesh builder or a file reader, as the [user book] shows from [A first program] on. The API
//! examples name the test mesh each hidden line makes, so you can follow them without this crate.
//!
//! The variable `TETRALOMA_TEST_DEVICE` (for example `cpu`, `metal`, `vulkan` or `cuda`) picks the
//! device of the tests; [Compute devices] describes the same choice for programs. The crate's
//! features forward CubeCL's runtime features; besides this crate, only tetraloma-bench and
//! tetraloma-examples enable them. tetraloma-device's feature `gpu-aware-cuda` is empty and
//! reserved.
//!
//! [user book]: https://tetracl.github.io/tetraloma/
//! [A first program]: https://tetracl.github.io/tetraloma/getting-started/first-program.html
//! [Compute devices]: https://tetracl.github.io/tetraloma/devices.html#choosing-a-device-at-run-time

mod comm;
pub mod comm_checks;
pub mod io_parallel;
mod meshes;
pub mod oracle;
pub mod parallel;
pub mod partition;
pub mod reference;

pub use comm::ThreadComm;
pub use meshes::{
    HostBlock, HostChunk, HostChunkBlock, HostMesh, HostTags, ball, book, circle, cube_box,
    hemisphere, hex_prism_pyramid_box, interval_box, mixed_box, mixed_surface, mobius_strip,
    sphere, square_box, triangle_strip, y_junction,
};

use cubecl::Device;
use cubecl::device::{DeviceUnavailable, MetalDevice, WgpuDeviceKind};
use cubecl::prelude::{Client, CubeElement};

/// The environment variable that names the test device.
pub const TEST_DEVICE_VAR: &str = "TETRALOMA_TEST_DEVICE";

/// The client of the device named by the `TETRALOMA_TEST_DEVICE` environment variable.
///
/// | `TETRALOMA_TEST_DEVICE` | Device | Feature of tetraloma-testing |
/// | --- | --- | --- |
/// | unset, empty or `cpu` | `Device::cpu()` | `cpu` (default) |
/// | `cuda` | `Device::cuda(0)` | `cuda` |
/// | `hip` | `Device::rocm(0)` | `hip` |
/// | `vulkan` | `Device::vulkan(WgpuDeviceKind::DefaultDevice)` | `vulkan` |
/// | `metal` | `Device::metal_msl(WgpuDeviceKind::DefaultDevice)` | `metal` |
/// | `metal-native` | `Device::metal(MetalDevice::DefaultDevice)` | `metal-native` |
/// | `wgpu` | `Device::wgpu(WgpuDeviceKind::DefaultDevice)` | `wgpu` |
///
/// # Panics
///
/// If the value names no device above, if this build does not link the runtime (the message
/// names the feature to turn on), or if the machine has no such device. Every message contains
/// `DeviceUnavailable`.
///
/// # Example
///
/// ```
/// let client = tetraloma_testing::test_client();
/// let properties = client.properties();
/// assert!(properties.hardware.max_bindings >= 6);
/// ```
pub fn test_client() -> Client {
    client_of(|backend| backend.device())
}

/// The client of the device `open` gives for the backend `TETRALOMA_TEST_DEVICE` names, or the
/// panic of [`test_client`].
fn client_of(open: impl FnOnce(TestBackend) -> Result<Device, DeviceUnavailable>) -> Client {
    let value = std::env::var(TEST_DEVICE_VAR).ok();
    let backend = match TestBackend::parse(value.as_deref()) {
        Ok(backend) => backend,
        Err(message) => panic!("{message}"),
    };
    match open(backend) {
        Ok(device) => device.client(),
        Err(error) => panic!("{}", backend.unavailable(error)),
    }
}

/// The client of GPU `index` of the backend `TETRALOMA_TEST_DEVICE` names, for a rank that opens
/// the GPU of [`tetraloma::local_gpu_index`] (ADR-0060 item 1): `Device::cuda(index)` and
/// `Device::rocm(index)` on CUDA and HIP; the other backends ignore `index` and open the device
/// of [`test_client`].
///
/// # Panics
///
/// As [`test_client`], and if the machine has no GPU `index`.
///
/// # Example
///
/// ```
/// let client = tetraloma_testing::test_client_at(0);
/// assert!(client.properties().hardware.max_bindings >= 6);
/// ```
pub fn test_client_at(index: usize) -> Client {
    client_of(|backend| backend.device_at(index))
}

/// The client of the CPU runtime, whatever `TETRALOMA_TEST_DEVICE` says: the f64 side of f32-vs-f64
/// comparisons on devices without f64 (docs/plans/M3.md, question 15).
///
/// # Example
///
/// ```
/// let client = tetraloma_testing::cpu_client();
/// assert!(tetraloma_testing::supports::<f64>(&client));
/// ```
#[cfg(feature = "cpu")]
pub fn cpu_client() -> Client {
    match Device::cpu() {
        Ok(device) => device.client(),
        Err(error) => panic!("{}", TestBackend::Cpu.unavailable(error)),
    }
}

/// Relative tolerance of f32 results (docs/design.md, "Testing and validation").
pub const TOL_F32: f64 = 1e-5;

/// Relative tolerance of f64 results (docs/design.md, "Testing and validation").
pub const TOL_F64: f64 = 1e-12;

/// Whether results on the test device are compared bitwise: true on the CPU runtime, where
/// every lane computes exactly what the scalar kernel computes and nothing is contracted.
///
/// # Example
///
/// ```
/// // Unset TETRALOMA_TEST_DEVICE is the CPU runtime.
/// if std::env::var("TETRALOMA_TEST_DEVICE").unwrap_or_default().is_empty() {
///     assert!(tetraloma_testing::bitwise_device());
/// }
/// ```
pub fn bitwise_device() -> bool {
    let value = std::env::var(TEST_DEVICE_VAR).ok();
    TestBackend::parse(value.as_deref()) == Ok(TestBackend::Cpu)
}

/// Whether the device supports `T` in any way (`DeviceProperties::supports_type`).
///
/// # Example
///
/// ```
/// let client = tetraloma_testing::test_client();
/// assert!(tetraloma_testing::supports::<f32>(&client));
/// ```
pub fn supports<T: CubeElement>(client: &Client) -> bool {
    client.properties().supports_type(T::cube_type())
}

/// A backend `TETRALOMA_TEST_DEVICE` can name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TestBackend {
    Cpu,
    Cuda,
    Hip,
    Vulkan,
    Metal,
    MetalNative,
    Wgpu,
}

impl TestBackend {
    const ALL: [TestBackend; 7] = [
        TestBackend::Cpu,
        TestBackend::Cuda,
        TestBackend::Hip,
        TestBackend::Vulkan,
        TestBackend::Metal,
        TestBackend::MetalNative,
        TestBackend::Wgpu,
    ];

    /// The value of `TETRALOMA_TEST_DEVICE`, which is also the name of the cargo feature.
    fn name(self) -> &'static str {
        match self {
            TestBackend::Cpu => "cpu",
            TestBackend::Cuda => "cuda",
            TestBackend::Hip => "hip",
            TestBackend::Vulkan => "vulkan",
            TestBackend::Metal => "metal",
            TestBackend::MetalNative => "metal-native",
            TestBackend::Wgpu => "wgpu",
        }
    }

    /// The backend `value` names; unset or empty is the CPU runtime.
    fn parse(value: Option<&str>) -> Result<TestBackend, String> {
        let value = value.unwrap_or("");
        if value.is_empty() {
            return Ok(TestBackend::Cpu);
        }
        TestBackend::ALL
            .into_iter()
            .find(|backend| backend.name() == value)
            .ok_or_else(|| {
                let names: Vec<_> = TestBackend::ALL.iter().map(|b| b.name()).collect();
                format!(
                    "{TEST_DEVICE_VAR}={value}: DeviceUnavailable: unknown device, expected one of {}",
                    names.join(", ")
                )
            })
    }

    /// The first device of this backend.
    fn device(self) -> Result<Device, DeviceUnavailable> {
        match self {
            TestBackend::Cpu => Device::cpu(),
            TestBackend::Cuda => Device::cuda(0),
            TestBackend::Hip => Device::rocm(0),
            TestBackend::Vulkan => Device::vulkan(WgpuDeviceKind::DefaultDevice),
            TestBackend::Metal => Device::metal_msl(WgpuDeviceKind::DefaultDevice),
            TestBackend::MetalNative => Device::metal(MetalDevice::DefaultDevice),
            TestBackend::Wgpu => Device::wgpu(WgpuDeviceKind::DefaultDevice),
        }
    }

    /// GPU `index` of CUDA and HIP; the first device of the other backends.
    fn device_at(self, index: usize) -> Result<Device, DeviceUnavailable> {
        match self {
            TestBackend::Cuda => Device::cuda(index),
            TestBackend::Hip => Device::rocm(index),
            _ => self.device(),
        }
    }

    /// The panic message when the device cannot be opened.
    fn unavailable(self, error: DeviceUnavailable) -> String {
        let name = self.name();
        match error {
            DeviceUnavailable::NotLinked(_) => format!(
                "{TEST_DEVICE_VAR}={name} needs --features tetraloma-testing/{name} \
                 (DeviceUnavailable: {error})"
            ),
            DeviceUnavailable::NoSuchDevice { .. } => {
                format!("{TEST_DEVICE_VAR}={name}: DeviceUnavailable: {error}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::device::RuntimeId;

    #[test]
    fn unset_and_empty_are_the_cpu_runtime() {
        assert_eq!(TestBackend::parse(None), Ok(TestBackend::Cpu));
        assert_eq!(TestBackend::parse(Some("")), Ok(TestBackend::Cpu));
    }

    #[test]
    fn every_name_parses_to_its_backend() {
        for backend in TestBackend::ALL {
            assert_eq!(TestBackend::parse(Some(backend.name())), Ok(backend));
        }
        assert_eq!(
            TestBackend::parse(Some("metal-native")),
            Ok(TestBackend::MetalNative)
        );
    }

    #[test]
    fn unknown_names_are_device_unavailable() {
        for value in ["gpu", "CUDA", " cpu", "rocm", "metal_native"] {
            let message = TestBackend::parse(Some(value)).expect_err(value);
            assert!(message.contains("DeviceUnavailable"), "{message}");
            assert!(
                message.starts_with(&format!("TETRALOMA_TEST_DEVICE={value}:")),
                "{message}"
            );
            assert!(message.contains("metal-native"), "{message}");
        }
    }

    #[test]
    fn not_linked_names_the_feature() {
        let message = TestBackend::Cuda.unavailable(DeviceUnavailable::NotLinked(RuntimeId::Cuda));
        assert!(
            message
                .starts_with("TETRALOMA_TEST_DEVICE=cuda needs --features tetraloma-testing/cuda"),
            "{message}"
        );
        assert!(message.contains("DeviceUnavailable"), "{message}");
    }

    #[test]
    fn no_such_device_is_device_unavailable() {
        let error = DeviceUnavailable::NoSuchDevice {
            runtime: RuntimeId::Wgpu,
            available: 0,
        };
        let message = TestBackend::Vulkan.unavailable(error);
        assert!(
            message.starts_with("TETRALOMA_TEST_DEVICE=vulkan: DeviceUnavailable:"),
            "{message}"
        );
    }

    #[test]
    #[cfg(not(feature = "hip"))]
    fn unlinked_runtimes_are_not_linked() {
        assert!(matches!(
            TestBackend::Hip.device(),
            Err(DeviceUnavailable::NotLinked(_))
        ));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn the_cpu_device_opens() {
        assert!(TestBackend::Cpu.device().is_ok());
    }
}
