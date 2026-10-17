//! env-check: probe of a gridcl development target.
//!
//! For each runtime compiled in (`cpu`, `metal`, `cuda`) env-check opens the devices,
//! records their properties, runs an element-wise add in f32 and f64 at vector size 1
//! and at the widest IO-optimized size, and a u32 atomic histogram, and writes a JSON
//! report. With the `mpi` feature and `--mpi-only` every rank checks the device
//! `rank % devices` and an allreduce of rank ids. See docs/design.md, milestone E0.

pub mod cuda_info;
pub mod kernels;
pub mod mpi_check;
pub mod probe;
pub mod report;

use std::path::PathBuf;

use probe::RuntimeKind;
use report::Report;

/// Usage text of the command line.
pub const USAGE: &str = "usage: env-check [--runtime cpu|metal|cuda|all] [--elements N] \
[--out PATH] [--mpi-only] [--expect-sm NN] [--expect-cuda-major M]

  --runtime            runtime to check (default: all runtimes compiled in)
  --elements N         elements of the sample kernels (default: 10000000)
  --out PATH           write the JSON report to PATH instead of stdout;
                       {rank} in PATH is replaced by the MPI rank
  --mpi-only           run under MPI: allreduce of rank ids and the device rank % devices
  --expect-sm NN       fail unless every CUDA device has compute capability NN
  --expect-cuda-major M  fail unless the CUDA toolkit's major version is M

exit status 0 iff every check passes";

/// Which runtimes to check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSelection {
    /// Every runtime compiled in.
    All,
    One(RuntimeKind),
}

/// Parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub runtime: RuntimeSelection,
    pub elements: usize,
    pub out: Option<PathBuf>,
    pub mpi_only: bool,
    pub expect_sm: Option<u32>,
    pub expect_cuda_major: Option<u32>,
}

impl Options {
    /// Parses the arguments after the program name.
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Options, String> {
        let _ = args.into_iter();
        todo!()
    }
}

/// Runs every check the options select and returns the finished report.
pub fn run(options: &Options) -> Report {
    let _ = options;
    todo!()
}
