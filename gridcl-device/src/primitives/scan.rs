//! Inclusive and exclusive scan, as reduce-then-scan with a vector-generic tile phase
//! (KernelKind::ScanTile).

use crate::{DeviceBuffer, DeviceScalar, Result};

/// Inclusive prefix sums: `out[i] = x[0] + … + x[i]`.
pub fn inclusive_scan<T: DeviceScalar>(_x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>> {
    todo!()
}

/// Exclusive prefix sums with **n + 1** elements, the last being the total, so the result can
/// serve as CSR offsets. This differs from the usual length-n exclusive scan.
pub fn exclusive_scan<T: DeviceScalar>(_x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>> {
    todo!()
}
