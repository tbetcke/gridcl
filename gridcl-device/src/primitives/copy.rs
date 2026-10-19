//! Copy (KernelKind::Copy), generic over the vector size.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::plan::debug_assert_bindings;
use crate::{DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, VECTOR_PAD};

/// `dst[v] = src[v]` for every vector `v < num_vectors`, by the loop of Listing B5.
#[cube(launch_unchecked)]
fn copy_kernel<N: Size>(
    src: &[Vector<u32, N>],
    dst: &mut [Vector<u32, N>],
    num_vectors: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            dst[v] = src[v];
        }
    }
}

/// Copies `src` into `dst`, padding included. Moves u32 words, so `size_of::<T>()` must be a
/// multiple of 4 (checked at compile time); unequal lengths give `LengthMismatch`.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, copy};
///
/// let client = gridcl_testing::test_client();
/// let src = DeviceBuffer::from_host(&client, &[1.0f64, 2.0, 3.0]).unwrap();
/// let mut dst = DeviceBuffer::<f64>::uninit(&client, 3).unwrap();
/// copy(&src, &mut dst).unwrap();
/// assert_eq!(dst.to_host().unwrap(), vec![1.0, 2.0, 3.0]);
/// ```
pub fn copy<T: CubeElement>(src: &DeviceBuffer<T>, dst: &mut DeviceBuffer<T>) -> Result<()> {
    const { assert!(size_of::<T>().is_multiple_of(4), "copy moves u32 words") };
    if src.len() != dst.len() {
        return Err(GridError::LengthMismatch {
            expected: src.len(),
            found: dst.len(),
        });
    }
    if src.len() == 0 {
        return Ok(());
    }
    debug_assert_eq!(src.padded_len(), dst.padded_len());
    let words = src
        .padded_len()
        .checked_mul(size_of::<T>() / 4)
        .ok_or(GridError::TooLarge)?;
    launch_copy(src.client(), src.handle(), dst.handle(), words)
}

/// Copies `words` u32 words from `src` to `dst`; `words` is a multiple of `VECTOR_PAD`.
fn launch_copy(client: &Client, src: &Handle, dst: &Handle, words: usize) -> Result<()> {
    debug_assert!(words.is_multiple_of(VECTOR_PAD));
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::Copy,
        words,
        size_of::<u32>(),
        VECTOR_PAD,
        1,
    )?;
    let n = plan.vector_size;
    debug_assert_bindings(client, 2);
    // SAFETY: both handles hold at least `words` u32 words (the padded length of buffers of
    // equal length), `words / n` whole vectors since `n` divides VECTOR_PAD, and the plan's
    // `num_vectors` is `words / n`. The kernel reads and writes only vectors below it.
    unsafe {
        copy_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            BufferArg::from_raw_parts(src.clone(), words / n),
            BufferArg::from_raw_parts(dst.clone(), words / n),
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    Ok(())
}
