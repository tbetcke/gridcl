//! Stable LSD radix sort with 8-bit digits; SortDigits is generic over the vector size.

use crate::{DeviceBuffer, Result, SortKeys};

/// Stable LSD sort with 8-bit digits. Unequal lengths give `LengthMismatch`; `len >= 2^32`
/// gives `TooLarge`.
pub fn radix_sort(
    _keys: &SortKeys,
    _payload: &DeviceBuffer<u32>,
) -> Result<(SortKeys, DeviceBuffer<u32>)> {
    todo!()
}
