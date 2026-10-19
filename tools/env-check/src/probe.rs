//! Opening devices and recording their properties and the host.

use std::process::Command;

use cubecl::Device;
use cubecl::device::{DeviceUnavailable, WgpuDeviceKind};
use cubecl::ir::Type;
use cubecl::ir::features::AtomicUsage;
use cubecl::prelude::{Client, CubeElement};

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
        match self {
            RuntimeKind::Cpu => "cpu",
            RuntimeKind::Metal => "metal",
            RuntimeKind::Cuda => "cuda",
        }
    }

    /// Parses `cpu`, `metal` or `cuda`.
    pub fn parse(name: &str) -> Option<RuntimeKind> {
        RuntimeKind::ALL
            .into_iter()
            .find(|kind| kind.name() == name)
    }

    /// Whether this build links the runtime (its cargo feature is on).
    pub fn compiled_in(self) -> bool {
        match self {
            RuntimeKind::Cpu => cfg!(feature = "cpu"),
            RuntimeKind::Metal => cfg!(feature = "metal"),
            RuntimeKind::Cuda => cfg!(feature = "cuda"),
        }
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
    if !runtime.compiled_in() {
        return Err(format!(
            "{}: this build does not include the runtime (cargo feature `{}`)",
            runtime.name(),
            runtime.name()
        ));
    }
    let describe = |e: DeviceUnavailable| format!("{}: {e}", runtime.name());
    let single = |device: Device| OpenedDevice {
        runtime,
        index: 0,
        visible: 1,
        device,
    };
    match runtime {
        // The CPU runtime has one device, and env-check checks the default Metal adapter only.
        RuntimeKind::Cpu => Ok(vec![single(Device::cpu().map_err(describe)?)]),
        RuntimeKind::Metal => Ok(vec![single(
            Device::metal_msl(WgpuDeviceKind::DefaultDevice).map_err(describe)?,
        )]),
        RuntimeKind::Cuda => {
            let mut devices = Vec::new();
            loop {
                match Device::cuda(devices.len()) {
                    Ok(device) => devices.push(device),
                    Err(DeviceUnavailable::NoSuchDevice { .. }) if !devices.is_empty() => break,
                    Err(e) => return Err(describe(e)),
                }
            }
            let visible = devices.len();
            Ok(devices
                .into_iter()
                .enumerate()
                .map(|(index, device)| OpenedDevice {
                    runtime,
                    index,
                    visible,
                    device,
                })
                .collect())
        }
    }
}

/// Records the properties of an opened device; `checks` is left empty.
pub fn device_report(opened: &OpenedDevice, client: &Client) -> DeviceReport {
    let properties = client.properties();
    let hardware = &properties.hardware;
    let atomic_u32 = properties.atomic_type_usage(Type::atomic(u32::cube_type()));
    let (cx, cy, cz) = hardware.max_cube_count;
    let (dx, dy, dz) = hardware.max_cube_dim;
    let mut report = DeviceReport {
        runtime: opened.runtime.name().to_string(),
        index: opened.index,
        visible_devices: opened.visible,
        name: properties.identity.name.clone(),
        fingerprint: properties.identity.fingerprint.clone(),
        supports_f64: properties.supports_type(f64::cube_type()),
        supports_u64: properties.supports_type(u64::cube_type()),
        atomic_u32_add: atomic_u32.contains(AtomicUsage::Add),
        max_page_size: properties.memory.max_page_size,
        max_bindings: hardware.max_bindings,
        max_cube_count: [cx, cy, cz],
        plane_size_min: hardware.plane_size_min,
        plane_size_max: hardware.plane_size_max,
        load_width: hardware.load_width,
        vector_register_count: hardware.vector_register_count,
        max_units_per_cube: hardware.max_units_per_cube,
        max_cube_dim: [dx, dy, dz],
        io_vector_sizes_f32: client.io_optimized_vector_sizes(4).collect(),
        io_vector_sizes_f64: client.io_optimized_vector_sizes(8).collect(),
        checks: Vec::new(),
        passed: false,
    };
    report.finish();
    report
}

/// Records the host: OS, CPU, SIMD flags and the Rust compiler.
pub fn host_info() -> HostInfo {
    let os = std::env::consts::OS.to_string();
    let hardware_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let rustc_version = command_output(&rustc, &["--version"]).unwrap_or_else(unknown);
    let mut host = HostInfo {
        os,
        os_version: unknown(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: unknown(),
        hardware_threads,
        performance_cores: None,
        efficiency_cores: None,
        simd_flags: Vec::new(),
        rustc_version,
    };
    if cfg!(target_os = "macos") {
        let sysctl = |name: &str| command_output("sysctl", &["-n", name]);
        let product = command_output("sw_vers", &["-productVersion"]);
        let build = command_output("sw_vers", &["-buildVersion"]);
        host.os_version = match (product, build) {
            (Some(product), Some(build)) => format!("{product} ({build})"),
            (Some(product), None) => product,
            _ => unknown(),
        };
        host.cpu_model = sysctl("machdep.cpu.brand_string").unwrap_or_else(unknown);
        host.performance_cores = sysctl("hw.perflevel0.physicalcpu").and_then(|v| v.parse().ok());
        host.efficiency_cores = sysctl("hw.perflevel1.physicalcpu").and_then(|v| v.parse().ok());
        host.simd_flags = command_output("sysctl", &["hw.optional"])
            .map(|out| parse_macos_simd_flags(&out))
            .unwrap_or_default();
    } else if cfg!(target_os = "linux") {
        let distribution = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|text| parse_os_release(&text));
        let kernel = command_output("uname", &["-r"]);
        host.os_version = match (distribution, kernel) {
            (Some(d), Some(k)) => format!("{d}, kernel {k}"),
            (Some(d), None) => d,
            (None, Some(k)) => format!("kernel {k}"),
            (None, None) => unknown(),
        };
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        host.cpu_model = parse_cpuinfo_model(&cpuinfo)
            .or_else(|| command_output("lscpu", &[]).and_then(|out| parse_lscpu_model(&out)))
            .unwrap_or_else(unknown);
        host.simd_flags = parse_cpuinfo_simd_flags(&cpuinfo);
    }
    host
}

fn unknown() -> String {
    "unknown".to_string()
}

/// Trimmed standard output of a command that ran and succeeded.
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!text.is_empty()).then_some(text)
}

/// SIMD flags from the output of `sysctl hw.optional` (macOS): the names set to 1.
pub fn parse_macos_simd_flags(sysctl_output: &str) -> Vec<String> {
    sysctl_output
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            let name = name.trim().strip_prefix("hw.optional.")?;
            (value.trim() == "1").then(|| name.to_string())
        })
        .collect()
}

/// SIMD flags from `/proc/cpuinfo` (Linux): the `flags` (x86-64) or `Features` (aarch64)
/// line of the first processor, restricted to SIMD extensions.
pub fn parse_cpuinfo_simd_flags(cpuinfo: &str) -> Vec<String> {
    let Some(flags) = cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        matches!(key.trim(), "flags" | "Features").then_some(value)
    }) else {
        return Vec::new();
    };
    flags
        .split_whitespace()
        .filter(|flag| is_simd_flag(flag))
        .map(str::to_string)
        .collect()
}

/// Whether a `/proc/cpuinfo` flag names a SIMD extension (x86-64 or aarch64).
fn is_simd_flag(flag: &str) -> bool {
    const PREFIXES: [&str; 10] = [
        "mmx", "sse", "ssse", "avx", "amx", "fma", "asimd", "sve", "sme", "xop",
    ];
    const EXACT: [&str; 6] = ["f16c", "fp", "fphp", "i8mm", "bf16", "vpclmulqdq"];
    PREFIXES.iter().any(|p| flag.starts_with(p)) || EXACT.contains(&flag)
}

/// `PRETTY_NAME` from `/etc/os-release`.
fn parse_os_release(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let value = line.strip_prefix("PRETTY_NAME=")?;
        Some(value.trim().trim_matches('"').to_string())
    })
}

/// The `model name` of the first processor in `/proc/cpuinfo` (absent on most aarch64 hosts).
fn parse_cpuinfo_model(cpuinfo: &str) -> Option<String> {
    cpuinfo.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "model name").then(|| value.trim().to_string())
    })
}

/// The `Model name` row of `lscpu`.
fn parse_lscpu_model(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key.trim() == "Model name").then(|| value.trim().to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_names_round_trip() {
        for kind in RuntimeKind::ALL {
            assert_eq!(RuntimeKind::parse(kind.name()), Some(kind));
        }
        assert_eq!(RuntimeKind::parse("vulkan"), None);
        assert_eq!(RuntimeKind::parse("all"), None);
    }

    #[test]
    fn macos_flags_set_to_one() {
        let out = "hw.optional.arm.FEAT_FP16: 1\n\
                   hw.optional.arm.FEAT_SME: 0\n\
                   hw.optional.neon: 1\n\
                   hw.optional.AdvSIMD: 1\n\
                   hw.optional.armv8_crc32: 1\n\
                   hw.optional.ucnormal_mem: 0\n";
        assert_eq!(
            parse_macos_simd_flags(out),
            ["arm.FEAT_FP16", "neon", "AdvSIMD", "armv8_crc32"]
        );
        assert!(parse_macos_simd_flags("").is_empty());
    }

    #[test]
    fn cpuinfo_x86_flags_restricted_to_simd() {
        let cpuinfo = "processor\t: 0\n\
                       model name\t: Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz\n\
                       flags\t\t: fpu vme mmx fxsr sse sse2 ssse3 fma cx16 sse4_1 sse4_2 popcnt \
                       avx f16c avx2 avx512f avx512dq avx512_vnni\n\n\
                       processor\t: 1\n\
                       flags\t\t: fpu sse\n";
        assert_eq!(
            parse_cpuinfo_simd_flags(cpuinfo),
            [
                "mmx",
                "sse",
                "sse2",
                "ssse3",
                "fma",
                "sse4_1",
                "sse4_2",
                "avx",
                "f16c",
                "avx2",
                "avx512f",
                "avx512dq",
                "avx512_vnni"
            ]
        );
        assert_eq!(
            parse_cpuinfo_model(cpuinfo).as_deref(),
            Some("Intel(R) Xeon(R) Gold 6240 CPU @ 2.60GHz")
        );
    }

    #[test]
    fn cpuinfo_aarch64_features() {
        let cpuinfo = "processor\t: 0\n\
                       BogoMIPS\t: 2000.00\n\
                       Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp \
                       asimdhp asimddp sve asimdfhm sve2 i8mm bf16 svebf16\n\
                       CPU part\t: 0xd4f\n";
        assert_eq!(
            parse_cpuinfo_simd_flags(cpuinfo),
            [
                "fp", "asimd", "fphp", "asimdhp", "asimddp", "sve", "asimdfhm", "sve2", "i8mm",
                "bf16", "svebf16"
            ]
        );
        assert_eq!(parse_cpuinfo_model(cpuinfo), None);
        assert!(parse_cpuinfo_simd_flags("processor : 0\n").is_empty());
    }

    #[test]
    fn os_release_and_lscpu() {
        let text = "NAME=\"Red Hat Enterprise Linux\"\nPRETTY_NAME=\"Red Hat Enterprise Linux 9.4 (Plow)\"\n";
        assert_eq!(
            parse_os_release(text).as_deref(),
            Some("Red Hat Enterprise Linux 9.4 (Plow)")
        );
        let lscpu = "Architecture:  aarch64\nModel name:    Neoverse-V2\n";
        assert_eq!(parse_lscpu_model(lscpu).as_deref(), Some("Neoverse-V2"));
    }
}
