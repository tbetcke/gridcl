//! The candidate protocol of "Benchmarks" (ADR-0057 item 3; ADR-0060 item 2): the
//! configurations and strategies each policy entry measures in M10 and after a CubeCL pin bump.
//!
//! - **Baseline:** vector size 1 at the old CpuSimd128 tile, block and units (a Cpu entry), or
//!   at the Gpu tile (a Gpu entry); for strategies, `direct`, or `loops` where `direct` is not
//!   selectable.
//! - **History:** the choices of the unchanged M3s rule (`decide`) and M9h rule
//!   (`decide_degree`) on the committed records, for any old device class, as frozen in
//!   `gridcl-bench/history/{vector,degree}-policy-pre-m10.toml`.
//! - **Own:** the configuration the entry's current policy names.
//! - Pack and PullBack, never measured before M10, also measure the widest size and the M3s
//!   choices of Gather and GeomFactors.

use gridcl_device::{CpuUnits, DeviceClass, KernelKind};

use crate::record::Scalar;

/// One configuration of a vector-policy entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VectorConfig {
    /// Requested vector size.
    pub vector_size: usize,
    /// QuadField tile width (GeomFactors only).
    pub tile: Option<usize>,
    /// CPU block in bytes (Cpu only).
    pub block_bytes: Option<usize>,
    /// CPU units (Cpu only).
    pub units: Option<CpuUnits>,
}

/// The candidates of the vector-policy entry (`class`, `kind`, `scalar`'s size) on a device
/// whose widest vector size for `scalar` is `widest`, baseline first.
pub fn vector_candidates(
    _class: DeviceClass,
    _kind: KernelKind,
    _scalar: Scalar,
    _widest: usize,
) -> Vec<VectorConfig> {
    todo!("M10 T2")
}

/// The candidate strategies of the degree-policy entry (`class`, policy `family`, `cell`,
/// `bytes`) at `degree`, baseline first.
pub fn degree_candidates(
    _class: DeviceClass,
    _family: &str,
    _cell: &str,
    _bytes: usize,
    _degree: u32,
) -> Vec<&'static str> {
    todo!("M10 T2")
}
