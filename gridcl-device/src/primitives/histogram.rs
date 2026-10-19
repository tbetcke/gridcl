//! Histogram (scalar): per-cube histograms in shared memory, added to the output with u32
//! atomics.
//!
//! The bins are processed in chunks that fit in shared memory. Integer sums are exact, so the
//! order in which the cubes add their counts does not change the result.

use cubecl::prelude::*;

use crate::plan::debug_assert_bindings;
use crate::{DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, testing};

/// Bins of one chunk at most: 64 KiB of counts.
const MAX_CHUNK_BINS: usize = 1 << 14;

/// Counts the values in `chunk_start..chunk_start + chunk_len` of the cube's items in shared
/// memory, then adds the non-zero counts to `out`.
#[cube(launch_unchecked)]
fn histogram_kernel(
    values: &[u32],
    out: &[Atomic<u32>],
    len: u32,
    items_per_unit: u32,
    chunk_start: u32,
    chunk_len: u32,
    #[comptime] chunk_bins: u32,
    #[comptime] cube_dim: u32,
) {
    let counts = Shared::<[Atomic<u32>]>::new_slice(chunk_bins as usize);
    let mut b = UNIT_POS as usize;
    while b < chunk_bins as usize {
        counts[b].store(0u32);
        b += cube_dim as usize;
    }
    sync_cube();
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let j = first + i;
        if j < len as usize {
            let v = values[j];
            if v >= chunk_start && v - chunk_start < chunk_len {
                counts[(v - chunk_start) as usize].fetch_add(1u32);
            }
        }
    }
    sync_cube();
    let mut b = UNIT_POS as usize;
    while b < chunk_len as usize {
        let c = counts[b].load();
        if c > 0 {
            out[chunk_start as usize + b].fetch_add(c);
        }
        b += cube_dim as usize;
    }
}

/// Counts of each value in `0..num_bins`. Values >= `num_bins` are not counted.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, histogram};
///
/// let client = gridcl_testing::test_client();
/// let values = DeviceBuffer::from_host(&client, &[2u32, 0, 2, 9, 1, 2]).unwrap();
/// assert_eq!(histogram(&values, 4).unwrap().to_host().unwrap(), vec![1, 1, 3, 0]);
/// ```
pub fn histogram(values: &DeviceBuffer<u32>, num_bins: u32) -> Result<DeviceBuffer<u32>> {
    let client = values.client();
    let out = DeviceBuffer::<u32>::zeros(client, num_bins as usize)?;
    if values.len() == 0 || num_bins == 0 {
        return Ok(out);
    }
    let len = u32::try_from(values.len()).map_err(|_| GridError::TooLarge)?;
    let plan = LaunchPlan::elementwise(client, KernelKind::Reduce, values.len(), 4, 1, 1)?;
    let cube_dim = plan.cube_dim.num_elems();
    let chunk_bins = chunk_bins(
        testing::properties(client).hardware.max_shared_memory_size,
        num_bins,
    );
    debug_assert_bindings(client, 2);
    let mut chunk_start = 0u32;
    while chunk_start < num_bins {
        let chunk_len = chunk_bins.min(num_bins - chunk_start);
        // SAFETY: the kernel reads values[j] only for j < len = values.len(), and adds to
        // out[chunk_start + b] only for b < chunk_len, so below num_bins = out.len(). The
        // shared array has chunk_bins >= chunk_len entries, indexed below chunk_bins.
        unsafe {
            histogram_kernel::launch_unchecked(
                client,
                plan.cube_count.clone(),
                plan.cube_dim,
                values.arg(),
                out.arg(),
                len,
                plan.items_per_unit,
                chunk_start,
                chunk_len,
                chunk_bins,
                cube_dim,
            );
        }
        chunk_start += chunk_len;
    }
    Ok(out)
}

/// Bins per chunk: `num_bins` rounded up to a power of two (so few kernel variants compile),
/// at most `MAX_CHUNK_BINS` and the largest power of two of u32 counts that fits in
/// `shared_bytes`; at least 1.
fn chunk_bins(shared_bytes: usize, num_bins: u32) -> u32 {
    let fit = (shared_bytes / size_of::<u32>()).clamp(1, MAX_CHUNK_BINS);
    let cap = 1usize << fit.ilog2();
    (num_bins as usize).next_power_of_two().min(cap) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_bins_fit_shared_memory() {
        assert_eq!(chunk_bins(48 << 10, 70_000), 8192);
        assert_eq!(chunk_bins(1 << 20, 70_000), MAX_CHUNK_BINS as u32);
        assert_eq!(chunk_bins(48 << 10, 17), 32);
        assert_eq!(chunk_bins(48 << 10, 1), 1);
        assert_eq!(chunk_bins(0, 5), 1);
    }
}
