//! The decision rule of "Benchmarks" as ADR-0015 reads it (items 5 and 6): per kernel family,
//! device class and element size, the vector size, CPU block and QuadField tile written to
//! `gridcl-device/vector-policy.toml`.

use gridcl_device::{DeviceClass, KernelKind};

use crate::record::PerfData;

/// The choice for one (device class, kernel kind, element size) entry of the policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub class: DeviceClass,
    pub kind: KernelKind,
    pub elem_bytes: usize,
    pub vector_size: usize,
    /// CPU block in units of 4 KiB; `None` on `Gpu`.
    pub cpu_block: Option<usize>,
    /// Why, in one line for the report.
    pub reason: String,
}

/// Every entry of the policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub choices: Vec<Choice>,
    /// (class, element size, tile) for every class and element size.
    pub quad_tiles: Vec<(DeviceClass, usize, usize)>,
}

impl Decision {
    /// The text of `gridcl-device/vector-policy.toml`.
    pub fn to_toml(&self) -> String {
        todo!("M3s T12: Decision::to_toml")
    }
}

/// The decision for `data`.
pub fn decide(data: &PerfData) -> Decision {
    let _ = data;
    todo!("M3s T12: decide")
}
