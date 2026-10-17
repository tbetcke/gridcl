//! The MPI check: allreduce of rank ids, the library version and CUDA-awareness.
//!
//! Uses safe rsmpi only; CUDA-awareness is found with the implementation's own tools.

use std::process::Command;

use crate::report::CudaAware;
#[cfg(feature = "mpi")]
use crate::report::MpiInfo;

/// The MPI implementation, from `MPI_Get_library_version`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MpiImplementation {
    OpenMpi,
    Mpich,
    IntelMpi,
    Unknown,
}

impl MpiImplementation {
    /// Identifies the implementation from its library version string.
    pub fn from_library_version(version: &str) -> MpiImplementation {
        // Intel MPI is derived from MPICH and may name it, so it is tested first.
        if version.contains("Intel(R) MPI") || version.contains("Intel MPI") {
            MpiImplementation::IntelMpi
        } else if version.contains("Open MPI") {
            MpiImplementation::OpenMpi
        } else if version.contains("MPICH") {
            MpiImplementation::Mpich
        } else {
            MpiImplementation::Unknown
        }
    }

    /// `open-mpi`, `mpich`, `intel-mpi` or `unknown`.
    pub fn name(self) -> &'static str {
        match self {
            MpiImplementation::OpenMpi => "open-mpi",
            MpiImplementation::Mpich => "mpich",
            MpiImplementation::IntelMpi => "intel-mpi",
            MpiImplementation::Unknown => "unknown",
        }
    }
}

/// An initialised MPI environment and the world communicator.
#[cfg(feature = "mpi")]
pub struct MpiSession {
    universe: mpi::environment::Universe,
    library_version: String,
    cuda_aware: (CudaAware, String),
}

#[cfg(feature = "mpi")]
impl MpiSession {
    /// Initialises MPI with `Threading::Single`.
    ///
    /// The CUDA-awareness tools run first, so no process is forked after `MPI_Init`.
    pub fn init() -> Result<MpiSession, String> {
        // MPI_Get_library_version may be called before MPI_Init.
        let library_version = mpi::environment::library_version()
            .map_err(|e| format!("MPI library version is not UTF-8: {e}"))?
            // The C string's terminating NUL is part of the returned buffer.
            .trim_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_string();
        let implementation = MpiImplementation::from_library_version(&library_version);
        let cuda_aware = detect_cuda_aware(implementation);
        let (universe, _) = mpi::initialize_with_threading(mpi::Threading::Single)
            .ok_or_else(|| "MPI is already initialised".to_string())?;
        Ok(MpiSession {
            universe,
            library_version,
            cuda_aware,
        })
    }

    pub fn rank(&self) -> i32 {
        use mpi::topology::Communicator;
        self.universe.world().rank()
    }

    pub fn size(&self) -> i32 {
        use mpi::topology::Communicator;
        self.universe.world().size()
    }

    /// Runs the allreduce of rank ids and records the library and its CUDA-awareness.
    /// `device_index` is the device this rank checked.
    pub fn check(&self, device_index: Option<usize>) -> MpiInfo {
        use mpi::collective::{CommunicatorCollectives, SystemOperation};
        let world = self.universe.world();
        let (rank, size) = (self.rank(), self.size());
        let mut sum = 0i64;
        world.all_reduce_into(&i64::from(rank), &mut sum, SystemOperation::sum());
        let size64 = i64::from(size);
        let expected = size64 * (size64 - 1) / 2;
        MpiInfo {
            rank,
            size,
            library_version: self.library_version.clone(),
            implementation: MpiImplementation::from_library_version(&self.library_version)
                .name()
                .to_string(),
            allreduce_result: sum,
            allreduce_expected: expected,
            allreduce_passed: sum == expected,
            device_index,
            cuda_aware: self.cuda_aware.0,
            cuda_aware_method: self.cuda_aware.1.clone(),
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

/// CUDA-awareness of `implementation`, by its own tools: `ompi_info` for Open MPI,
/// the version tools and CVARs for MPICH and Intel MPI. Returns the value and the method.
pub fn detect_cuda_aware(implementation: MpiImplementation) -> (CudaAware, String) {
    match implementation {
        MpiImplementation::OpenMpi => {
            let method = "ompi_info --parsable --all (mpi_built_with_cuda_support)";
            match command_output("ompi_info", &["--parsable", "--all"]) {
                Some(out) => (parse_ompi_info_cuda(&out), method.to_string()),
                None => (CudaAware::Unknown, format!("{method}: ompi_info failed")),
            }
        }
        MpiImplementation::Mpich => {
            if std::env::var("MPIR_CVAR_ENABLE_GPU").as_deref() == Ok("0") {
                return (CudaAware::No, "MPIR_CVAR_ENABLE_GPU=0".to_string());
            }
            let method = "mpichversion (configure options and device)";
            match command_output("mpichversion", &[]) {
                Some(out) => (parse_mpich_version_cuda(&out), method.to_string()),
                None => (CudaAware::Unknown, format!("{method}: mpichversion failed")),
            }
        }
        MpiImplementation::IntelMpi => match std::env::var("I_MPI_OFFLOAD") {
            Ok(v) if v.trim() != "0" && !v.trim().is_empty() => {
                (CudaAware::Yes, format!("I_MPI_OFFLOAD={}", v.trim()))
            }
            Ok(v) => (CudaAware::No, format!("I_MPI_OFFLOAD={}", v.trim())),
            Err(_) => (
                CudaAware::Unknown,
                "I_MPI_OFFLOAD not set (Intel MPI GPU support is off by default)".to_string(),
            ),
        },
        MpiImplementation::Unknown => (
            CudaAware::Unknown,
            "unknown MPI implementation; no tool to ask".to_string(),
        ),
    }
}

/// `mpi_built_with_cuda_support` from `ompi_info --parsable --all`.
pub fn parse_ompi_info_cuda(output: &str) -> CudaAware {
    output
        .lines()
        .find_map(|line| {
            let value = line.split_once("mpi_built_with_cuda_support:value:")?.1;
            match value.trim() {
                "true" => Some(CudaAware::Yes),
                "false" => Some(CudaAware::No),
                _ => None,
            }
        })
        .unwrap_or(CudaAware::Unknown)
}

/// CUDA support from `mpichversion` (or `mpiexec --version` for MPICH), from the
/// device and configure options it lists.
pub fn parse_mpich_version_cuda(output: &str) -> CudaAware {
    let configure = output
        .lines()
        .find(|l| l.contains("configure:") || l.contains("Configure options:"));
    let Some(configure) = configure else {
        return CudaAware::Unknown;
    };
    let words: Vec<&str> = configure.split_whitespace().collect();
    if words
        .iter()
        .any(|w| *w == "--without-cuda" || *w == "--with-cuda=no" || *w == "--disable-gpu")
    {
        CudaAware::No
    } else if words.iter().any(|w| w.starts_with("--with-cuda")) {
        CudaAware::Yes
    } else {
        CudaAware::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implementations() {
        let cases = [
            (
                "Open MPI v5.0.10, package: Open MPI brew@... Distribution, ident: 5.0.10",
                MpiImplementation::OpenMpi,
            ),
            (
                "MPICH Version:      4.2.0\nMPICH Release date: Fri Feb  9 2024",
                MpiImplementation::Mpich,
            ),
            (
                "Intel(R) MPI Library 2021.10 for Linux* OS",
                MpiImplementation::IntelMpi,
            ),
            ("Something MPI 1.0", MpiImplementation::Unknown),
        ];
        for (version, want) in cases {
            assert_eq!(
                MpiImplementation::from_library_version(version),
                want,
                "{version}"
            );
        }
        assert_eq!(MpiImplementation::OpenMpi.name(), "open-mpi");
        assert_eq!(MpiImplementation::IntelMpi.name(), "intel-mpi");
    }

    #[test]
    fn ompi_info() {
        let yes = "mca:mpi:base:param:mpi_built_with_cuda_support:value:true\n\
                   mca:mpi:base:param:mpi_built_with_cuda_support:source:default\n";
        let no = "ompi:version:full:5.0.10\n\
                  mca:mpi:base:param:mpi_built_with_cuda_support:value:false\n";
        assert_eq!(parse_ompi_info_cuda(yes), CudaAware::Yes);
        assert_eq!(parse_ompi_info_cuda(no), CudaAware::No);
        assert_eq!(
            parse_ompi_info_cuda("ompi:version:full:5.0.10\n"),
            CudaAware::Unknown
        );
    }

    #[test]
    fn mpich_version() {
        let with = "MPICH Version:      4.2.0\n\
                    MPICH Device:       ch4:ofi\n\
                    MPICH configure:    --prefix=/opt/mpich --with-cuda=/opt/cuda --with-device=ch4:ofi\n";
        let without = "MPICH Version:      4.2.0\n\
                       MPICH configure:    --prefix=/opt/mpich --without-cuda\n";
        let plain = "MPICH Version:      4.2.0\n\
                     MPICH configure:    --prefix=/opt/mpich\n";
        assert_eq!(parse_mpich_version_cuda(with), CudaAware::Yes);
        assert_eq!(parse_mpich_version_cuda(without), CudaAware::No);
        assert_eq!(parse_mpich_version_cuda(plain), CudaAware::Unknown);
        assert_eq!(
            parse_mpich_version_cuda("MPICH Version: 4.2.0\n"),
            CudaAware::Unknown
        );
    }
}
