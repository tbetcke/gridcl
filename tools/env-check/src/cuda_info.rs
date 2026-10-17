//! CUDA toolkit, driver and compute capability, by subprocess and file reads only
//! (no FFI; CLAUDE.md).

use std::path::Path;
use std::process::Command;

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
    /// The compiler of this build: cubecl-cuda picks NVRTC when its `cpp` feature is on.
    pub fn of_build() -> CudaCompiler {
        if cfg!(feature = "cuda-cpp") {
            CudaCompiler::Nvrtc
        } else {
            CudaCompiler::Llvm
        }
    }

    /// `llvm` or `nvrtc`.
    pub fn name(self) -> &'static str {
        match self {
            CudaCompiler::Llvm => "llvm",
            CudaCompiler::Nvrtc => "nvrtc",
        }
    }
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Collects the CUDA record for devices with the given fingerprints and checks the expectations.
pub fn collect(
    compiler: CudaCompiler,
    fingerprints: &[String],
    expect_sm: Option<u32>,
    expect_cuda_major: Option<u32>,
) -> CudaInfo {
    let cuda_path = std::env::var("CUDA_PATH")
        .ok()
        .or_else(|| std::env::var("CUDA_HOME").ok())
        .filter(|p| !p.is_empty());
    let mut toolkit = cuda_path.as_deref().and_then(|path| {
        let text = std::fs::read_to_string(Path::new(path).join("version.json")).ok()?;
        parse_version_json(&text).map(|v| (v, "version.json"))
    });
    if toolkit.is_none() {
        let nvcc = match &cuda_path {
            Some(path) => Path::new(path)
                .join("bin/nvcc")
                .to_string_lossy()
                .into_owned(),
            None => "nvcc".to_string(),
        };
        toolkit = command_output(&nvcc, &["--version"])
            .or_else(|| command_output("nvcc", &["--version"]))
            .and_then(|out| parse_nvcc_version(&out))
            .map(|v| (v, "nvcc"));
    }
    let header = command_output("nvidia-smi", &[]).and_then(|out| parse_nvidia_smi_header(&out));
    let smi_caps = command_output(
        "nvidia-smi",
        &["--query-gpu=compute_cap", "--format=csv,noheader"],
    )
    .map(|out| parse_compute_caps(&out))
    .unwrap_or_default();
    evaluate(
        compiler,
        fingerprints,
        cuda_path,
        toolkit.map(|(v, s)| (v, s.to_string())),
        header,
        smi_caps,
        expect_sm,
        expect_cuda_major,
    )
}

/// Builds the record from what was found and checks it; separated from `collect` for tests.
#[allow(clippy::too_many_arguments)]
fn evaluate(
    compiler: CudaCompiler,
    fingerprints: &[String],
    cuda_path: Option<String>,
    toolkit: Option<(String, String)>,
    driver: Option<(String, String)>,
    nvidia_smi_compute_capability: Vec<String>,
    expect_sm: Option<u32>,
    expect_cuda_major: Option<u32>,
) -> CudaInfo {
    let mut problems = Vec::new();
    let mut compute_capability = Vec::new();
    for (index, fingerprint) in fingerprints.iter().enumerate() {
        let Some(sm) = parse_fingerprint_sm(fingerprint) else {
            problems.push(format!(
                "device {index}: no compute capability in fingerprint `{fingerprint}`"
            ));
            continue;
        };
        let capability = sm_to_compute_capability(sm);
        if !nvidia_smi_compute_capability.contains(&capability) {
            problems.push(format!(
                "device {index}: compute capability {capability} not among nvidia-smi's {:?}",
                nvidia_smi_compute_capability
            ));
        }
        if let Some(want) = expect_sm
            && sm != want
        {
            problems.push(format!("device {index}: sm_{sm}, expected sm_{want}"));
        }
        compute_capability.push(capability);
    }
    if fingerprints.is_empty() {
        problems.push("no CUDA device".to_string());
    }

    let (toolkit_version, toolkit_source) = match toolkit {
        Some((version, source)) => (Some(version), Some(source)),
        None => (None, None),
    };
    // The LLVM backend emits PTX itself; NVRTC and the toolkit rules need the toolkit.
    if toolkit_version.is_none() && (compiler == CudaCompiler::Nvrtc || expect_cuda_major.is_some())
    {
        problems.push("CUDA toolkit not found (CUDA_PATH/version.json or nvcc)".to_string());
    }
    if let (Some(want), Some(version)) = (expect_cuda_major, &toolkit_version) {
        let major = version
            .split('.')
            .next()
            .and_then(|m| m.parse::<u32>().ok());
        if major != Some(want) {
            problems.push(format!("CUDA toolkit {version}, expected CUDA {want}.x"));
        }
    }

    let (driver_version, driver_cuda_version) = match driver {
        Some((driver, cuda)) => (Some(driver), Some(cuda)),
        None => {
            problems.push("no driver version from nvidia-smi".to_string());
            (None, None)
        }
    };
    let toolkit_not_newer_than_driver = match (&toolkit_version, &driver_cuda_version) {
        (Some(toolkit), Some(driver)) => toolkit_not_newer(toolkit, driver),
        _ => None,
    };
    if toolkit_not_newer_than_driver == Some(false) {
        problems.push(format!(
            "CUDA toolkit {} is newer than the driver's CUDA {}",
            toolkit_version.as_deref().unwrap_or("?"),
            driver_cuda_version.as_deref().unwrap_or("?")
        ));
    }

    CudaInfo {
        compiler: compiler.name().to_string(),
        compute_capability,
        nvidia_smi_compute_capability,
        cuda_path,
        toolkit_version,
        toolkit_source,
        driver_version,
        driver_cuda_version,
        toolkit_not_newer_than_driver,
        expect_sm,
        expect_cuda_major,
        passed: problems.is_empty(),
        problems,
    }
}

/// The `NN` of `ptx_smNN` in a CubeCL CUDA fingerprint, for example 70 or 90.
pub fn parse_fingerprint_sm(fingerprint: &str) -> Option<u32> {
    let rest = &fingerprint[fingerprint.find("sm")? + 2..];
    let rest = rest.strip_prefix('_').unwrap_or(rest);
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok().filter(|&sm| sm >= 10)
}

/// `NN` as a compute capability `major.minor`, for example 70 → `7.0`.
pub fn sm_to_compute_capability(sm: u32) -> String {
    format!("{}.{}", sm / 10, sm % 10)
}

/// The value after `key` in `line`, up to the next run of two spaces or `|`.
fn header_value<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let rest = line[line.find(key)? + key.len()..].trim_start();
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '|')
        .unwrap_or(rest.len());
    Some(&rest[..end]).filter(|v| !v.is_empty())
}

/// Driver version and CUDA version from the header of `nvidia-smi`.
pub fn parse_nvidia_smi_header(output: &str) -> Option<(String, String)> {
    output.lines().find_map(|line| {
        let driver = header_value(line, "Driver Version:")?;
        let cuda = header_value(line, "CUDA Version:")?;
        Some((driver.to_string(), cuda.to_string()))
    })
}

/// One compute capability per line of `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.
pub fn parse_compute_caps(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim)
        .filter(|l| {
            !l.is_empty() && l.contains('.') && l.chars().all(|c| c.is_ascii_digit() || c == '.')
        })
        .map(str::to_string)
        .collect()
}

/// The `cuda` version from `$CUDA_PATH/version.json`.
pub fn parse_version_json(text: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    value
        .get("cuda")?
        .get("version")?
        .as_str()
        .map(str::to_string)
}

/// The release version (`12.2`) from `nvcc --version`.
pub fn parse_nvcc_version(output: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let rest = &line[line.find("release ")? + "release ".len()..];
        let version: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        Some(version).filter(|v| !v.is_empty())
    })
}

fn major_minor(version: &str) -> Option<(u32, u32)> {
    let mut parts = version.trim().split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().map_or(Some(0), |m| m.parse().ok())?;
    Some((major, minor))
}

/// Whether toolkit version `toolkit` (`12.2.140` or `12.2`) is no newer than the driver's
/// CUDA version `driver` (`12.4`), comparing major and minor.
pub fn toolkit_not_newer(toolkit: &str, driver: &str) -> Option<bool> {
    Some(major_minor(toolkit)? <= major_minor(driver)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    const NVIDIA_SMI: &str = "\
Fri Oct 17 17:00:00 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
|   0  Tesla V100-PCIE-32GB           On  |   00000000:3B:00.0 Off |                    0 |
+-----------------------------------------------------------------------------------------+
";

    const NVIDIA_SMI_OLD: &str = "\
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
";

    const VERSION_JSON: &str = r#"{
   "cuda" : {
      "name" : "CUDA SDK",
      "version" : "12.2.20230823"
   },
   "cuda_cudart" : {
      "name" : "CUDA Runtime (cudart)",
      "version" : "12.2.140"
   }
}"#;

    const NVCC: &str = "\
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0
";

    #[test]
    fn fingerprint_sm() {
        assert_eq!(parse_fingerprint_sm("ptx_sm70"), Some(70));
        assert_eq!(parse_fingerprint_sm("ptx_sm80"), Some(80));
        assert_eq!(parse_fingerprint_sm("ptx_sm90a"), Some(90));
        assert_eq!(parse_fingerprint_sm("ptx_sm_120"), Some(120));
        assert_eq!(parse_fingerprint_sm("spirv_0_0"), None);
        assert_eq!(parse_fingerprint_sm("ptx_sm"), None);
        assert_eq!(sm_to_compute_capability(70), "7.0");
        assert_eq!(sm_to_compute_capability(90), "9.0");
        assert_eq!(sm_to_compute_capability(120), "12.0");
    }

    #[test]
    fn nvidia_smi_header() {
        assert_eq!(
            parse_nvidia_smi_header(NVIDIA_SMI),
            Some(("550.54.15".to_string(), "12.4".to_string()))
        );
        assert_eq!(
            parse_nvidia_smi_header(NVIDIA_SMI_OLD),
            Some(("470.57.02".to_string(), "11.4".to_string()))
        );
        assert_eq!(parse_nvidia_smi_header("No devices were found"), None);
    }

    #[test]
    fn compute_caps() {
        assert_eq!(parse_compute_caps("7.0\n7.0\n"), vec!["7.0", "7.0"]);
        assert_eq!(parse_compute_caps(" 9.0 \n"), vec!["9.0"]);
        assert!(parse_compute_caps("Field \"compute_cap\" is not a valid field\n").is_empty());
    }

    #[test]
    fn toolkit_versions() {
        assert_eq!(
            parse_version_json(VERSION_JSON).as_deref(),
            Some("12.2.20230823")
        );
        assert_eq!(parse_version_json("{}"), None);
        assert_eq!(parse_version_json("not json"), None);
        assert_eq!(parse_nvcc_version(NVCC).as_deref(), Some("12.2"));
        assert_eq!(parse_nvcc_version("bash: nvcc: command not found"), None);
    }

    #[test]
    fn toolkit_against_driver() {
        assert_eq!(toolkit_not_newer("12.2.20230823", "12.4"), Some(true));
        assert_eq!(toolkit_not_newer("12.4", "12.4"), Some(true));
        assert_eq!(toolkit_not_newer("12.6.1", "12.4"), Some(false));
        assert_eq!(toolkit_not_newer("13.0", "12.9"), Some(false));
        assert_eq!(toolkit_not_newer("12", "12.0"), Some(true));
        assert_eq!(toolkit_not_newer("x", "12.0"), None);
    }

    fn v100(toolkit: Option<&str>, expect_sm: Option<u32>, major: Option<u32>) -> CudaInfo {
        evaluate(
            CudaCompiler::Llvm,
            &["ptx_sm70".to_string(), "ptx_sm70".to_string()],
            Some("/opt/cuda".to_string()),
            toolkit.map(|v| (v.to_string(), "version.json".to_string())),
            Some(("550.54.15".to_string(), "12.4".to_string())),
            vec!["7.0".to_string(), "7.0".to_string()],
            expect_sm,
            major,
        )
    }

    #[test]
    fn expectations() {
        let ok = v100(Some("12.2.20230823"), Some(70), Some(12));
        assert!(ok.passed, "{:?}", ok.problems);
        assert_eq!(ok.compute_capability, vec!["7.0", "7.0"]);
        assert_eq!(ok.toolkit_not_newer_than_driver, Some(true));
        assert_eq!(ok.compiler, "llvm");

        assert!(!v100(Some("12.2"), Some(80), None).passed, "wrong sm");
        assert!(
            !v100(Some("13.0"), None, Some(12)).passed,
            "CUDA 13 on V100"
        );
        assert!(
            !v100(Some("12.6"), None, None).passed,
            "toolkit newer than the driver"
        );
        assert!(
            !v100(None, None, Some(12)).passed,
            "expected major without a toolkit"
        );
        assert!(v100(None, None, None).passed, "LLVM path without a toolkit");
    }

    #[test]
    fn nvrtc_needs_the_toolkit_and_nvidia_smi_must_agree() {
        let info = evaluate(
            CudaCompiler::Nvrtc,
            &["ptx_sm80".to_string()],
            None,
            None,
            Some(("550.54.15".to_string(), "12.4".to_string())),
            vec!["7.0".to_string()],
            None,
            None,
        );
        assert!(!info.passed);
        assert_eq!(info.problems.len(), 2, "{:?}", info.problems);
    }
}
