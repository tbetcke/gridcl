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
        todo!()
    }

    /// Pretty-printed JSON.
    pub fn to_json(&self) -> String {
        todo!()
    }

    /// Parses a report written by [`Report::to_json`].
    pub fn from_json(text: &str) -> Result<Report, serde_json::Error> {
        let _ = text;
        todo!()
    }
}

impl DeviceReport {
    /// Sets `passed` from the checks.
    pub fn finish(&mut self) {
        todo!()
    }
}
