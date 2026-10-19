//! Test support for gridcl: the test device chosen by `GRIDCL_TEST_DEVICE`, test meshes and
//! tolerance helpers.
//!
//! This crate is not published. Its features forward CubeCL's runtime features; besides it,
//! only gridcl-bench and gridcl-device (`gpu-aware-cuda`) enable them (docs/design.md,
//! "Cargo features", "Dependency rules").

use cubecl::Device;
use cubecl::device::{DeviceUnavailable, MetalDevice, WgpuDeviceKind};
use cubecl::prelude::Client;

/// The environment variable that names the test device.
pub const TEST_DEVICE_VAR: &str = "GRIDCL_TEST_DEVICE";

/// The client of the device named by the `GRIDCL_TEST_DEVICE` environment variable.
///
/// | `GRIDCL_TEST_DEVICE` | Device | Feature of gridcl-testing |
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
/// let client = gridcl_testing::test_client();
/// let properties = client.properties();
/// assert!(properties.hardware.max_bindings >= 6);
/// ```
pub fn test_client() -> Client {
    let value = std::env::var(TEST_DEVICE_VAR).ok();
    let backend = match TestBackend::parse(value.as_deref()) {
        Ok(backend) => backend,
        Err(message) => panic!("{message}"),
    };
    match backend.device() {
        Ok(device) => device.client(),
        Err(error) => panic!("{}", backend.unavailable(error)),
    }
}

/// A backend `GRIDCL_TEST_DEVICE` can name.
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

    /// The value of `GRIDCL_TEST_DEVICE`, which is also the name of the cargo feature.
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

    /// The panic message when the device cannot be opened.
    fn unavailable(self, error: DeviceUnavailable) -> String {
        let name = self.name();
        match error {
            DeviceUnavailable::NotLinked(_) => format!(
                "{TEST_DEVICE_VAR}={name} needs --features gridcl-testing/{name} \
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
                message.starts_with(&format!("GRIDCL_TEST_DEVICE={value}:")),
                "{message}"
            );
            assert!(message.contains("metal-native"), "{message}");
        }
    }

    #[test]
    fn not_linked_names_the_feature() {
        let message = TestBackend::Cuda.unavailable(DeviceUnavailable::NotLinked(RuntimeId::Cuda));
        assert!(
            message.starts_with("GRIDCL_TEST_DEVICE=cuda needs --features gridcl-testing/cuda"),
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
            message.starts_with("GRIDCL_TEST_DEVICE=vulkan: DeviceUnavailable:"),
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
