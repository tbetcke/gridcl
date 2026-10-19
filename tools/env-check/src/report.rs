//! The JSON report written by env-check.
//!
//! One report per process. Under MPI every rank writes its own report; the
//! acceptance test merges them.

use serde::{Deserialize, Serialize};

/// Version of the report layout; bump it when a field changes meaning.
pub const SCHEMA_VERSION: u32 = 1;

/// The outcome of one check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CheckStatus {
    /// The check ran and its result matched the host reference.
    Pass,
    /// The check ran and failed, or it could not run where it must.
    Fail,
    /// The device does not support what the check needs, and the design says so.
    Unsupported,
}

/// Whether an MPI library is CUDA-aware.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CudaAware {
    Yes,
    No,
    Unknown,
}

/// The whole report of one env-check process.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub schema_version: u32,
    /// env-check's own version.
    pub tool_version: String,
    pub host: HostInfo,
    /// One entry per device checked.
    pub devices: Vec<DeviceReport>,
    /// Present when env-check ran under MPI (`--mpi-only`).
    pub mpi: Option<MpiInfo>,
    /// Present when the CUDA runtime was checked.
    pub cuda: Option<CudaInfo>,
    /// Problems that are not tied to one device, such as a runtime that failed to open.
    pub errors: Vec<String>,
    /// True iff every device, the MPI check and the CUDA checks passed and `errors` is empty.
    pub passed: bool,
}

/// The host env-check ran on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInfo {
    /// `macos` or `linux` (`std::env::consts::OS`).
    pub os: String,
    /// Product version on macOS, distribution and kernel on Linux.
    pub os_version: String,
    /// `aarch64` or `x86_64`.
    pub arch: String,
    pub cpu_model: String,
    pub hardware_threads: usize,
    /// Performance cores (macOS `hw.perflevel0.physicalcpu`).
    pub performance_cores: Option<u32>,
    /// Efficiency cores (macOS `hw.perflevel1.physicalcpu`).
    pub efficiency_cores: Option<u32>,
    /// SIMD extensions: `hw.optional` flags set to 1 on macOS, `/proc/cpuinfo` flags on Linux.
    pub simd_flags: Vec<String>,
    /// Output of `rustc --version`.
    pub rustc_version: String,
}

/// Properties and checks of one device.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceReport {
    /// `cpu`, `metal` or `cuda`.
    pub runtime: String,
    /// Index of the device within its runtime.
    pub index: usize,
    /// Number of devices the runtime reports on this host.
    pub visible_devices: usize,
    /// `properties().identity.name`.
    pub name: String,
    /// `properties().identity.fingerprint` (on CUDA it carries `ptx_smNN`).
    pub fingerprint: String,
    pub supports_f64: bool,
    pub supports_u64: bool,
    /// Whether `atomic_type_usage(u32)` includes `Add`.
    pub atomic_u32_add: bool,
    pub max_page_size: u64,
    pub max_bindings: u32,
    pub max_cube_count: [u32; 3],
    pub plane_size_min: u32,
    pub plane_size_max: u32,
    pub load_width: u32,
    pub vector_register_count: Option<u32>,
    pub max_units_per_cube: u32,
    pub max_cube_dim: [u32; 3],
    /// `io_optimized_vector_sizes(4)`, widest first.
    pub io_vector_sizes_f32: Vec<usize>,
    /// `io_optimized_vector_sizes(8)`, widest first.
    pub io_vector_sizes_f64: Vec<usize>,
    pub checks: Vec<KernelCheck>,
    /// True iff no check failed.
    pub passed: bool,
}

/// One kernel run and its comparison with the host reference.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelCheck {
    /// `elementwise_add` or `atomic_histogram`.
    pub kernel: String,
    /// `f32`, `f64` or `u32`.
    pub precision: String,
    pub vector_size: usize,
    pub elements: usize,
    pub status: CheckStatus,
    /// Wall-clock time of the launch and read-back, when the kernel ran.
    pub millis: Option<f64>,
    /// The first mismatch, the error, or why the check is unsupported.
    pub detail: Option<String>,
}

/// The MPI check of one rank.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MpiInfo {
    pub rank: i32,
    pub size: i32,
    /// `MPI_Get_library_version`.
    pub library_version: String,
    /// `open-mpi`, `mpich`, `intel-mpi` or `unknown`.
    pub implementation: String,
    /// Sum of the rank ids, computed with allreduce.
    pub allreduce_result: i64,
    /// size * (size - 1) / 2.
    pub allreduce_expected: i64,
    pub allreduce_passed: bool,
    /// The device index this rank checked (`rank % visible devices`), per runtime.
    pub device_index: Option<usize>,
    pub cuda_aware: CudaAware,
    /// How `cuda_aware` was found, for example `ompi_info --parsable --all`.
    pub cuda_aware_method: String,
}

/// CUDA toolkit, driver and architecture.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaInfo {
    /// `llvm` (feature `cuda`) or `nvrtc` (feature `cuda-cpp`).
    pub compiler: String,
    /// Compute capability from `identity.fingerprint`, as `major.minor`, per device.
    pub compute_capability: Vec<String>,
    /// `nvidia-smi --query-gpu=compute_cap`, per device.
    pub nvidia_smi_compute_capability: Vec<String>,
    pub cuda_path: Option<String>,
    /// Toolkit version, from `$CUDA_PATH/version.json` or `nvcc --version`.
    pub toolkit_version: Option<String>,
    /// `version.json` or `nvcc`.
    pub toolkit_source: Option<String>,
    /// Driver version from the `nvidia-smi` header.
    pub driver_version: Option<String>,
    /// CUDA version from the `nvidia-smi` header.
    pub driver_cuda_version: Option<String>,
    /// Toolkit version no newer than the driver's CUDA version.
    pub toolkit_not_newer_than_driver: Option<bool>,
    /// `--expect-sm`, if given.
    pub expect_sm: Option<u32>,
    /// `--expect-cuda-major`, if given.
    pub expect_cuda_major: Option<u32>,
    /// Failed cross-checks and expectations.
    pub problems: Vec<String>,
    pub passed: bool,
}

impl Report {
    /// Sets `passed` from the devices, the MPI and CUDA records and `errors`.
    pub fn finish(&mut self) {
        let devices = self.devices.iter().all(|d| d.passed);
        let mpi = self.mpi.as_ref().is_none_or(|m| m.allreduce_passed);
        let cuda = self.cuda.as_ref().is_none_or(|c| c.passed);
        self.passed = devices && mpi && cuda && self.errors.is_empty();
    }

    /// The reasons the report fails apart from failed kernel checks: `errors`, the CUDA
    /// record's problems and a wrong MPI allreduce, one line each.
    pub fn failures(&self) -> Vec<String> {
        let mut lines = self.errors.clone();
        if let Some(cuda) = &self.cuda {
            lines.extend(cuda.problems.iter().map(|p| format!("cuda: {p}")));
        }
        if let Some(mpi) = self.mpi.as_ref().filter(|m| !m.allreduce_passed) {
            lines.push(format!(
                "MPI: allreduce of rank ids on rank {} of {} gave {}, expected {}",
                mpi.rank, mpi.size, mpi.allreduce_result, mpi.allreduce_expected
            ));
        }
        lines
    }

    /// Pretty-printed JSON.
    pub fn to_json(&self) -> String {
        // The report holds only strings, numbers, booleans, options and vectors, which
        // serde_json always serialises; a non-finite `millis` becomes `null`.
        serde_json::to_string_pretty(self).expect("a report always serialises to JSON")
    }

    /// Parses a report written by [`Report::to_json`].
    pub fn from_json(text: &str) -> Result<Report, serde_json::Error> {
        serde_json::from_str(text)
    }
}

impl DeviceReport {
    /// Sets `passed` from the checks.
    pub fn finish(&mut self) {
        self.passed = self.checks.iter().all(|c| c.status != CheckStatus::Fail);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(status: CheckStatus) -> KernelCheck {
        KernelCheck {
            kernel: "elementwise_add".to_string(),
            precision: "f32".to_string(),
            vector_size: 4,
            elements: 4096,
            status,
            millis: Some(1.234),
            detail: None,
        }
    }

    fn device(statuses: &[CheckStatus]) -> DeviceReport {
        let mut device = DeviceReport {
            runtime: "metal".to_string(),
            index: 0,
            visible_devices: 1,
            name: "Apple M3 Max".to_string(),
            fingerprint: "msl".to_string(),
            supports_f64: false,
            supports_u64: true,
            atomic_u32_add: true,
            max_page_size: 1 << 32,
            max_bindings: 28,
            max_cube_count: [65535; 3],
            plane_size_min: 32,
            plane_size_max: 32,
            load_width: 128,
            vector_register_count: None,
            max_units_per_cube: 1024,
            max_cube_dim: [1024; 3],
            io_vector_sizes_f32: vec![4, 2, 1],
            io_vector_sizes_f64: vec![2, 1],
            checks: statuses.iter().map(|&s| check(s)).collect(),
            passed: false,
        };
        device.finish();
        device
    }

    fn report(devices: Vec<DeviceReport>) -> Report {
        let mut report = Report {
            schema_version: SCHEMA_VERSION,
            tool_version: "0.1.0".to_string(),
            host: HostInfo {
                os: "macos".to_string(),
                os_version: "27.0".to_string(),
                arch: "aarch64".to_string(),
                cpu_model: "Apple M3 Max".to_string(),
                hardware_threads: 16,
                performance_cores: Some(12),
                efficiency_cores: Some(4),
                simd_flags: vec!["neon".to_string()],
                rustc_version: "rustc 1.95.0".to_string(),
            },
            devices,
            mpi: None,
            cuda: None,
            errors: Vec::new(),
            passed: false,
        };
        report.finish();
        report
    }

    #[test]
    fn device_passes_unless_a_check_fails() {
        assert!(device(&[CheckStatus::Pass, CheckStatus::Unsupported]).passed);
        assert!(device(&[]).passed);
        assert!(!device(&[CheckStatus::Pass, CheckStatus::Fail]).passed);
    }

    #[test]
    fn report_passes_iff_everything_passed() {
        assert!(report(vec![device(&[CheckStatus::Pass])]).passed);
        assert!(
            !report(vec![
                device(&[CheckStatus::Pass]),
                device(&[CheckStatus::Fail])
            ])
            .passed
        );

        let mut with_error = report(vec![device(&[CheckStatus::Pass])]);
        with_error.errors.push("cuda: no device".to_string());
        with_error.finish();
        assert!(!with_error.passed);

        let mut with_mpi = report(Vec::new());
        with_mpi.mpi = Some(MpiInfo {
            rank: 0,
            size: 2,
            library_version: "Open MPI v5.0.8".to_string(),
            implementation: "open-mpi".to_string(),
            allreduce_result: 0,
            allreduce_expected: 1,
            allreduce_passed: false,
            device_index: Some(0),
            cuda_aware: CudaAware::Unknown,
            cuda_aware_method: "ompi_info --parsable --all".to_string(),
        });
        with_mpi.finish();
        assert!(!with_mpi.passed);
    }

    #[test]
    fn failures_name_errors_cuda_problems_and_mpi() {
        assert!(
            report(vec![device(&[CheckStatus::Pass])])
                .failures()
                .is_empty()
        );

        let mut failing = report(Vec::new());
        failing.errors.push("metal: no device".to_string());
        failing.cuda = Some(CudaInfo {
            compiler: "llvm".to_string(),
            compute_capability: vec!["8.0".to_string()],
            nvidia_smi_compute_capability: vec!["8.0".to_string()],
            cuda_path: None,
            toolkit_version: None,
            toolkit_source: None,
            driver_version: None,
            driver_cuda_version: None,
            toolkit_not_newer_than_driver: None,
            expect_sm: Some(70),
            expect_cuda_major: None,
            problems: vec!["device 0: sm_80, expected sm_70".to_string()],
            passed: false,
        });
        failing.mpi = Some(MpiInfo {
            rank: 1,
            size: 2,
            library_version: "Open MPI v5.0.8".to_string(),
            implementation: "open-mpi".to_string(),
            allreduce_result: 0,
            allreduce_expected: 1,
            allreduce_passed: false,
            device_index: None,
            cuda_aware: CudaAware::Unknown,
            cuda_aware_method: "ompi_info --parsable --all".to_string(),
        });
        assert_eq!(
            failing.failures(),
            vec![
                "metal: no device".to_string(),
                "cuda: device 0: sm_80, expected sm_70".to_string(),
                "MPI: allreduce of rank ids on rank 1 of 2 gave 0, expected 1".to_string(),
            ]
        );
    }

    #[test]
    fn json_round_trip_and_lowercase_enums() {
        let mut original = report(vec![device(&[
            CheckStatus::Pass,
            CheckStatus::Unsupported,
            CheckStatus::Fail,
        ])]);
        original.cuda = Some(CudaInfo {
            compiler: "llvm".to_string(),
            compute_capability: vec!["7.0".to_string()],
            nvidia_smi_compute_capability: vec!["7.0".to_string()],
            cuda_path: None,
            toolkit_version: Some("12.2.140".to_string()),
            toolkit_source: Some("version.json".to_string()),
            driver_version: Some("550.54.15".to_string()),
            driver_cuda_version: Some("12.4".to_string()),
            toolkit_not_newer_than_driver: Some(true),
            expect_sm: Some(70),
            expect_cuda_major: None,
            problems: Vec::new(),
            passed: true,
        });
        original.finish();
        let text = original.to_json();
        assert_eq!(Report::from_json(&text).expect("parses"), original);

        let value: serde_json::Value = serde_json::from_str(&text).expect("JSON");
        let checks = &value["devices"][0]["checks"];
        assert_eq!(checks[0]["status"], "pass");
        assert_eq!(checks[1]["status"], "unsupported");
        assert_eq!(checks[2]["status"], "fail");
        assert_eq!(value["mpi"], serde_json::Value::Null);
        assert_eq!(value["passed"], false);
        assert_eq!(
            serde_json::to_string(&CudaAware::Yes).expect("serialises"),
            "\"yes\""
        );
        assert!(Report::from_json("{\"schema_version\": 1}").is_err());
    }
}
