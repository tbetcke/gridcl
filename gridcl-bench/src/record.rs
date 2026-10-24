//! The JSON records of `docs/perf/data/`: one [`SweepRecord`] per measured configuration of the
//! vector-size sweep, and one [`ThroughputRecord`] per end-to-end throughput measurement.
//!
//! Files named `m3s-sweep-*.json` and `m3s-throughput-*.json` each hold a JSON array of records.

use std::path::Path;

use gridcl_device::{DeviceClass, KernelKind};

/// Scalar types of the benchmarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scalar {
    F32,
    F64,
    U32,
}

impl Scalar {
    /// The name in records and reports: `f32`, `f64` or `u32`.
    pub fn name(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::U32 => "u32",
        }
    }

    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            Scalar::F32 | Scalar::U32 => 4,
            Scalar::F64 => 8,
        }
    }
}

/// How the CPU `cube_dim` was chosen: the plan's (one unit per hardware thread) or the
/// performance-core count (M3 Max only; reported, not part of a configuration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeDimMode {
    Plan,
    PerformanceCores,
}

/// Core counts of the machine: performance and efficiency cores where the OS reports them
/// (`sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`), and hardware threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cores {
    pub performance: Option<u32>,
    pub efficiency: Option<u32>,
    pub threads: u32,
}

/// The runner's node-load snapshots before and after a run on Locust (ADR-0002, ADR-0005):
/// `idle`, `busy: <reasons>` or `not recorded`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeLoad {
    pub before: String,
    pub after: String,
}

/// One measured configuration of the sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepRecord {
    /// CubeCL runtime: `cpu`, `metal`, `cuda`, ...
    pub backend: String,
    /// Device name as the client reports it.
    pub device: String,
    pub device_class: DeviceClass,
    /// Machine: `m3max`, or a target of scripts/hpc/remote-test.sh (`locust`, `kathleen`,
    /// `myriad-cpu`, `myriad-v100`, `myriad-a100`).
    pub target: String,
    pub family: KernelKind,
    pub scalar: Scalar,
    /// Problem size of the protocol: 10^5, 10^6 or 10^7.
    pub nominal_n: usize,
    /// Items actually processed (cells of the nearest box for the mesh families).
    pub n: usize,
    /// Requested vector size (`with_kind_vector_size(Fixed(n))`).
    pub vector_size: usize,
    /// Vector size of the family's plans, from `record_plans`.
    pub vector_size_ran: usize,
    /// QuadField tile width (GeomFactors only).
    pub tile: Option<usize>,
    /// CPU block target in bytes (CPU classes only).
    pub block_bytes: Option<usize>,
    pub cube_dim: CubeDimMode,
    /// Units per cube of the family's plans.
    pub units: u32,
    /// Every timed run, in seconds.
    pub times_s: Vec<f64>,
    pub median_s: f64,
    /// Warm-up time minus the median.
    pub compile_s: f64,
    /// State of CubeCL's persistent kernel cache during the warm-up.
    pub kernel_cache: String,
    /// Effective bandwidth relative to `copy` at the same size on the same device, for the
    /// families with a byte count.
    pub bandwidth_ratio: Option<f64>,
    pub commit: String,
    /// `release` or `debug`.
    pub profile: String,
    pub cores: Cores,
    /// Why a Myriad GPU ran in place of Locust (ADR-0014), or a Myriad D node in place of
    /// Kathleen (ADR-0016); empty elsewhere.
    pub fallback_reason: Option<String>,
    /// Locust's node-load snapshots.
    pub node_load: Option<NodeLoad>,
}

/// What a throughput record measures (docs/plans/M3s.md, T13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Measure {
    /// Entities created by `create_entities(d)`, 0 < d < tdim, per second.
    EntitiesPerSecond,
    /// (cell, quadrature point) pairs with X, J, det J and K computed, per second.
    FactorsPerSecond,
}

/// One end-to-end throughput measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughputRecord {
    pub backend: String,
    pub device: String,
    pub device_class: DeviceClass,
    pub target: String,
    pub measure: Measure,
    /// The mesh, for example `P1 tetrahedron box`.
    pub mesh: String,
    pub cells: usize,
    pub scalar: Scalar,
    /// `policy` or `scalar`.
    pub vector_size_override: String,
    /// Entities created, or (cell, point) pairs.
    pub items: usize,
    pub times_s: Vec<f64>,
    pub median_s: f64,
    /// `items / median_s`.
    pub per_s: f64,
    pub commit: String,
    pub profile: String,
    pub cores: Cores,
    /// Why a Myriad D node ran in place of Kathleen (ADR-0016); empty elsewhere.
    pub fallback_reason: Option<String>,
    pub node_load: Option<NodeLoad>,
}

/// Every M3s record of a directory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerfData {
    pub sweeps: Vec<SweepRecord>,
    pub throughput: Vec<ThroughputRecord>,
}

/// The records of every `m3s-sweep-*.json` and `m3s-throughput-*.json` file in `dir`.
///
/// # Errors
///
/// A message naming the file, for unreadable files and malformed records.
pub fn load_dir(dir: &Path) -> Result<PerfData, String> {
    let _ = dir;
    todo!("M3s T8: load_dir")
}
