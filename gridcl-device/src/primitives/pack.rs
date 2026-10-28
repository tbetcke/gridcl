//! Pack and unpack (KernelKind::Pack), generic over the vector size: the device side of the
//! host-staged halo exchange (ADR-0026, item 13).

use cubecl::prelude::*;

use crate::{DeviceBuffer, Result};

/// `out[k·block .. (k+1)·block] = src[idx[k]·block .. (idx[k]+1)·block]`, moved as u32 words,
/// so `size_of::<T>()` must be a multiple of 4. An empty `idx` launches nothing.
pub fn pack<T: CubeElement>(
    src: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
    block: usize,
    out: &mut DeviceBuffer<T>,
) -> Result<()> {
    let _ = (src, idx, block, out);
    todo!("M6 T4: pack")
}

/// `dst[idx[k]·block .. (idx[k]+1)·block] = packed[k·block .. (k+1)·block]`, moved as u32
/// words; the indices must be distinct (checked in debug builds, as for `scatter`). An empty
/// `idx` launches nothing.
pub fn unpack<T: CubeElement>(
    packed: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
    block: usize,
    dst: &mut DeviceBuffer<T>,
) -> Result<()> {
    let _ = (packed, idx, block, dst);
    todo!("M6 T4: unpack")
}
