//! Reduce (KernelKind::Reduce), generic over the vector size.

use crate::{DeviceBuffer, DeviceScalar, ReduceOp, Result};

/// Reduces `x` with `op`. Length 1; an empty input gives the identity of `op`.
pub fn reduce<T: DeviceScalar>(_x: &DeviceBuffer<T>, _op: ReduceOp) -> Result<DeviceBuffer<T>> {
    todo!()
}
