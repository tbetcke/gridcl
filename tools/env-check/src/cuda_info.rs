//! CUDA toolkit, driver and compute capability, by subprocess and file reads only
//! (no FFI; CLAUDE.md).

use crate::report::CudaInfo;

/// Which CubeCL backend compiled the CUDA kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudaCompiler {
    /// cubecl's default backend: LLVM to PTX (feature `cuda`).
    Llvm,
    /// The C++ backend compiled with NVRTC (feature `cuda-cpp`).
    Nvrtc,
}

impl CudaCompiler {
    /// The compiler of this build.
    pub fn of_build() -> CudaCompiler {
        todo!()
    }

    /// `llvm` or `nvrtc`.
    pub fn name(self) -> &'static str {
        todo!()
    }
}

/// Collects the CUDA record for devices with the given fingerprints and checks the expectations.
pub fn collect(
    compiler: CudaCompiler,
    fingerprints: &[String],
    expect_sm: Option<u32>,
    expect_cuda_major: Option<u32>,
) -> CudaInfo {
    let _ = (compiler, fingerprints, expect_sm, expect_cuda_major);
    todo!()
}

/// The `NN` of `ptx_smNN` in a CubeCL CUDA fingerprint, for example 70 or 90.
pub fn parse_fingerprint_sm(fingerprint: &str) -> Option<u32> {
    let _ = fingerprint;
    todo!()
}

/// `NN` as a compute capability `major.minor`, for example 70 → `7.0`.
pub fn sm_to_compute_capability(sm: u32) -> String {
    let _ = sm;
    todo!()
}

/// Driver version and CUDA version from the header of `nvidia-smi`.
pub fn parse_nvidia_smi_header(output: &str) -> Option<(String, String)> {
    let _ = output;
    todo!()
}

/// One compute capability per line of `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
pub fn parse_compute_caps(output: &str) -> Vec<String> {
    let _ = output;
    todo!()
}

/// The `cuda` version from `$CUDA_PATH/version.json`.
pub fn parse_version_json(text: &str) -> Option<String> {
    let _ = text;
    todo!()
}

/// The release version (`12.2`) from `nvcc --version`.
pub fn parse_nvcc_version(output: &str) -> Option<String> {
    let _ = output;
    todo!()
}

/// Whether toolkit version `toolkit` (`12.2.140` or `12.2`) is no newer than the driver's
/// CUDA version `driver` (`12.4`), comparing major and minor.
pub fn toolkit_not_newer(toolkit: &str, driver: &str) -> Option<bool> {
    let _ = (toolkit, driver);
    todo!()
}
