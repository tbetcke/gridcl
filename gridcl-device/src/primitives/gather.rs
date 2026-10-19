//! Gather (KernelKind::Gather, generic over the vector size) and scatter (scalar).

use crate::{DeviceBuffer, DeviceScalar, Result};

/// `out[i] = src[idx[i]]`. An index >= `src.len()` yields 0 through a mask (debug builds
/// panic); the read address is clamped and never out of bounds.
pub fn gather<T: DeviceScalar>(
    _src: &DeviceBuffer<T>,
    _idx: &DeviceBuffer<u32>,
) -> Result<DeviceBuffer<T>> {
    todo!()
}

/// `dst[idx[i]] = src[i]`. Out-of-range writes are dropped. Indices must be distinct:
/// duplicates are memory-safe but leave the result unspecified (debug builds panic).
pub fn scatter<T: DeviceScalar>(
    _src: &DeviceBuffer<T>,
    _idx: &DeviceBuffer<u32>,
    _dst: &mut DeviceBuffer<T>,
) -> Result<()> {
    todo!()
}
