//! Histogram: per-cube histograms in shared memory, then a fixed-order sum per bin.

use crate::{DeviceBuffer, Result};

/// Counts of each value in `0..num_bins`. Values >= `num_bins` are not counted.
pub fn histogram(_values: &DeviceBuffer<u32>, _num_bins: u32) -> Result<DeviceBuffer<u32>> {
    todo!()
}
