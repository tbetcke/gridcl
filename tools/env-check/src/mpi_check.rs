//! The MPI check: allreduce of rank ids, the library version and CUDA-awareness.
//!
//! Uses safe rsmpi only; CUDA-awareness is found with the implementation's own tools.

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
        let _ = version;
        todo!()
    }

    /// `open-mpi`, `mpich`, `intel-mpi` or `unknown`.
    pub fn name(self) -> &'static str {
        todo!()
    }
}

/// An initialised MPI environment and the world communicator.
#[cfg(feature = "mpi")]
pub struct MpiSession {
    universe: mpi::environment::Universe,
}

#[cfg(feature = "mpi")]
impl MpiSession {
    /// Initialises MPI with `Threading::Single`.
    pub fn init() -> Result<MpiSession, String> {
        todo!()
    }

    pub fn rank(&self) -> i32 {
        let _ = &self.universe;
        todo!()
    }

    pub fn size(&self) -> i32 {
        todo!()
    }

    /// Runs the allreduce of rank ids and records the library and its CUDA-awareness.
    /// `device_index` is the device this rank checked.
    pub fn check(&self, device_index: Option<usize>) -> MpiInfo {
        let _ = device_index;
        todo!()
    }
}

/// CUDA-awareness of `implementation`, by its own tools: `ompi_info` for Open MPI,
/// the version tools and CVARs for MPICH and Intel MPI. Returns the value and the method.
pub fn detect_cuda_aware(implementation: MpiImplementation) -> (CudaAware, String) {
    let _ = implementation;
    todo!()
}

/// `mpi_built_with_cuda_support` from `ompi_info --parsable --all`.
pub fn parse_ompi_info_cuda(output: &str) -> CudaAware {
    let _ = output;
    todo!()
}

/// CUDA support from `mpichversion` (or `mpiexec --version` for MPICH), from the
/// device and configure options it lists.
pub fn parse_mpich_version_cuda(output: &str) -> CudaAware {
    let _ = output;
    todo!()
}
