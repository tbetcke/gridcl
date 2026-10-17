//! Opening devices and recording their properties and the host.

use cubecl::Device;
use cubecl::prelude::Client;

use crate::report::{DeviceReport, HostInfo};

/// A runtime env-check can be built with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RuntimeKind {
    Cpu,
    Metal,
    Cuda,
}

impl RuntimeKind {
    /// Every runtime, in report order.
    pub const ALL: [RuntimeKind; 3] = [RuntimeKind::Cpu, RuntimeKind::Metal, RuntimeKind::Cuda];

    /// The name used on the command line and in the report.
    pub fn name(self) -> &'static str {
        todo!()
    }

    /// Parses `cpu`, `metal` or `cuda`.
    pub fn parse(name: &str) -> Option<RuntimeKind> {
        let _ = name;
        todo!()
    }

    /// Whether this build links the runtime (its cargo feature is on).
    pub fn compiled_in(self) -> bool {
        todo!()
    }
}

/// A device that was opened, with its place among the runtime's devices.
#[derive(Clone, Debug)]
pub struct OpenedDevice {
    pub runtime: RuntimeKind,
    pub index: usize,
    /// Number of devices the runtime reports.
    pub visible: usize,
    pub device: Device,
}

/// Opens every device of `runtime`: one for `cpu` and `metal`, every visible GPU for `cuda`.
///
/// Fails if the runtime is not compiled in or reports no device.
pub fn open_devices(runtime: RuntimeKind) -> Result<Vec<OpenedDevice>, String> {
    let _ = runtime;
    todo!()
}

/// Records the properties of an opened device; `checks` is left empty.
pub fn device_report(opened: &OpenedDevice, client: &Client) -> DeviceReport {
    let _ = (opened, client);
    todo!()
}

/// Records the host: OS, CPU, SIMD flags and the Rust compiler.
pub fn host_info() -> HostInfo {
    todo!()
}

/// SIMD flags from the output of `sysctl hw.optional` (macOS): the names set to 1.
pub fn parse_macos_simd_flags(sysctl_output: &str) -> Vec<String> {
    let _ = sysctl_output;
    todo!()
}

/// SIMD flags from `/proc/cpuinfo` (Linux): the `flags` (x86-64) or `Features` (aarch64)
/// line of the first processor, restricted to SIMD extensions.
pub fn parse_cpuinfo_simd_flags(cpuinfo: &str) -> Vec<String> {
    let _ = cpuinfo;
    todo!()
}
