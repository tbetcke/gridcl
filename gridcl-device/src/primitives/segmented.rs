//! Segmented reduce (scalar): one unit per segment, combined sequentially in index order.

use cubecl::prelude::*;

use super::reduce::{Op, combine, identity};
use crate::plan::debug_assert_bindings;
use crate::{DeviceBuffer, DeviceScalar, GridError, KernelKind, LaunchPlan, ReduceOp, Result};

/// `out[s]` combines `values[lo..hi]` in index order, with `lo` and `hi` the offsets of
/// segment s clamped to `len`; an empty or decreasing pair gives the identity.
// Kernel code indexes slices; it cannot iterate over them.
#[allow(clippy::needless_range_loop)]
#[cube(launch_unchecked)]
fn segmented_kernel<T: Numeric>(
    values: &[T],
    offsets: &[u32],
    out: &mut [T],
    len: u32,
    num_segments: u32,
    items_per_unit: u32,
    #[comptime] op: Op,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let s = first + i;
        if s < num_segments as usize {
            let lo = min(offsets[s], len) as usize;
            let hi = min(offsets[s + 1], len) as usize;
            let mut acc = identity::<T>(op);
            for j in lo..hi {
                acc = combine::<T>(acc, values[j], op);
            }
            out[s] = acc;
        }
    }
}

/// One value per CSR segment, combined in index order. `offsets` has segments + 1 entries
/// (fewer than two give no segments). Offsets are clamped to `[0, values.len()]`; a decreasing
/// pair gives an empty segment, whose value is the identity of `op` (0, `T::max_value()`,
/// `T::min_value()`).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, ReduceOp, segmented_reduce};
///
/// let client = gridcl_testing::test_client();
/// let values = DeviceBuffer::from_host(&client, &[1u32, 2, 3, 4, 5]).unwrap();
/// let offsets = DeviceBuffer::from_host(&client, &[0u32, 2, 2, 5]).unwrap();
/// let sums = segmented_reduce(&values, &offsets, ReduceOp::Sum).unwrap();
/// assert_eq!(sums.to_host().unwrap(), vec![3, 0, 12]);
/// ```
pub fn segmented_reduce<T: DeviceScalar>(
    values: &DeviceBuffer<T>,
    offsets: &DeviceBuffer<u32>,
    op: ReduceOp,
) -> Result<DeviceBuffer<T>> {
    let client = values.client();
    let segments = offsets.len().saturating_sub(1);
    let out = DeviceBuffer::<T>::uninit(client, segments)?;
    if segments == 0 {
        return Ok(out);
    }
    let len = u32::try_from(values.len()).map_err(|_| GridError::TooLarge)?;
    let num_segments = u32::try_from(segments).map_err(|_| GridError::TooLarge)?;
    let plan = LaunchPlan::elementwise(client, KernelKind::Scalar, segments, size_of::<T>(), 1, 1)?;
    debug_assert_bindings(client, 3);
    // SAFETY: the kernel reads offsets[s] and offsets[s + 1] only for s < num_segments, so
    // below offsets.len(); values[j] only for j < min(offset, len) <= values.len(); and writes
    // out[s] only for s < num_segments = out.len().
    unsafe {
        segmented_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count,
            plan.cube_dim,
            values.arg(),
            offsets.arg(),
            out.arg(),
            len,
            num_segments,
            plan.items_per_unit,
            Op::from(op),
        );
    }
    Ok(out)
}
