//! Copy (KernelKind::Copy), generic over the vector size.

use cubecl::prelude::*;

use crate::{DeviceBuffer, Result};

/// Copies `src` into `dst`. Moves u32 words, so `size_of::<T>()` must be a multiple of 4;
/// unequal lengths give `LengthMismatch`.
pub fn copy<T: CubeElement>(_src: &DeviceBuffer<T>, _dst: &mut DeviceBuffer<T>) -> Result<()> {
    todo!()
}
