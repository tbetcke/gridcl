//! Segmented reduce (scalar).

use crate::{DeviceBuffer, DeviceScalar, ReduceOp, Result};

/// One value per CSR segment, combined in index order. Offsets are clamped to
/// `[0, values.len()]`; a decreasing pair gives an empty segment.
pub fn segmented_reduce<T: DeviceScalar>(
    _values: &DeviceBuffer<T>,
    _offsets: &DeviceBuffer<u32>,
    _op: ReduceOp,
) -> Result<DeviceBuffer<T>> {
    todo!()
}
