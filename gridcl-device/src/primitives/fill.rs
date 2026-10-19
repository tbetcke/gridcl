//! Fill (KernelKind::Fill), generic over the vector size.

use crate::{DeviceBuffer, DeviceScalar, Result};

/// Writes `v` over the padded length.
pub fn fill<T: DeviceScalar>(_x: &mut DeviceBuffer<T>, _v: T) -> Result<()> {
    todo!()
}
