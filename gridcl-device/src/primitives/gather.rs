//! Gather (KernelKind::Gather, generic over the vector size) and scatter (scalar).

use cubecl::prelude::*;

use crate::plan::debug_assert_bindings;
use crate::{DeviceBuffer, DeviceScalar, GridError, KernelKind, LaunchPlan, Result, VECTOR_PAD};

/// `out[v]` lane by lane from `src` at the indices of `idx[v]`, for every vector
/// `v < num_vectors` (Listing B5 loop). The read address is clamped to `src_len - 1`, and an
/// index >= `src_len` yields 0 through a select, so no lane branches on its index.
#[cube(launch_unchecked)]
fn gather_kernel<T: Numeric, N: Size>(
    src: &[T],
    idx: &[Vector<u32, N>],
    out: &mut [Vector<T, N>],
    src_len: u32,
    num_vectors: u32,
    items_per_unit: u32,
) {
    let last = src_len - 1;
    let zero = T::from_int(0);
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let index = idx[v];
            let mut value = Vector::<T, N>::broadcast(zero);
            #[unroll]
            for l in 0..N::value() {
                let k = index.extract(l);
                let x = src[min(k, last) as usize];
                value.insert(l, select(k <= last, x, zero));
            }
            out[v] = value;
        }
    }
}

/// `dst[idx[j]] = src[j]` for every `j < len` with `idx[j] < dst_len`.
#[cube(launch_unchecked)]
fn scatter_kernel<T: Numeric>(
    src: &[T],
    idx: &[u32],
    dst: &mut [T],
    len: u32,
    dst_len: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let j = first + i;
        if j < len as usize {
            let k = idx[j];
            if k < dst_len {
                dst[k as usize] = src[j];
            }
        }
    }
}

/// `out[i] = src[idx[i]]`. An index >= `src.len()` yields 0 through a mask (debug builds
/// panic); the read address is clamped and never out of bounds. An empty `src` gives zeros
/// without reading it.
///
/// # Panics
///
/// In debug builds, if an index is out of range: the check reads back `reduce(idx, Max)`.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, gather};
///
/// let client = gridcl_testing::test_client();
/// let src = DeviceBuffer::from_host(&client, &[10.0f32, 20.0, 30.0]).unwrap();
/// let idx = DeviceBuffer::from_host(&client, &[2u32, 0, 0, 1]).unwrap();
/// assert_eq!(gather(&src, &idx).unwrap().to_host().unwrap(), vec![30.0, 10.0, 10.0, 20.0]);
/// ```
pub fn gather<T: DeviceScalar>(
    src: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
) -> Result<DeviceBuffer<T>> {
    let client = src.client();
    if idx.len() == 0 {
        return DeviceBuffer::uninit(client, 0);
    }
    #[cfg(debug_assertions)]
    {
        let max = super::reduce(idx, crate::ReduceOp::Max)?.to_host()?[0];
        assert!(
            (max as usize) < src.len(),
            "gather: index out of range: {max} >= src.len() = {}",
            src.len()
        );
    }
    if src.len() == 0 {
        return DeviceBuffer::zeros(client, idx.len());
    }
    let src_len = u32::try_from(src.len()).map_err(|_| GridError::TooLarge)?;
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::Gather,
        idx.len(),
        size_of::<T>(),
        VECTOR_PAD,
        2,
    )?;
    let n = plan.vector_size;
    let out = DeviceBuffer::<T>::uninit(client, idx.len())?;
    debug_assert_bindings(client, 3);
    // SAFETY: `idx` and `out` hold pad_len(idx.len()) elements, and the plan's num_vectors is
    // pad_len(idx.len()) / n; the kernel touches only vectors below num_vectors. It reads `src`
    // at min(k, src_len - 1) with src_len = src.len() >= 1, so within `src`. Padding lanes of
    // `idx` hold arbitrary indices, which the clamp keeps in bounds too.
    unsafe {
        gather_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            src.arg(),
            idx.vector_arg(n),
            out.vector_arg(n),
            src_len,
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    Ok(out)
}

/// `dst[idx[i]] = src[i]`. Out-of-range writes are dropped. Indices must be distinct:
/// duplicates are memory-safe but leave the result unspecified (debug builds panic).
///
/// # Errors
///
/// `LengthMismatch` if `idx` and `src` differ in length; `TooLarge` if `dst` has 2^32 or more
/// elements.
///
/// # Panics
///
/// In debug builds, on a duplicate in-range index: the check reads back the maximum of
/// `histogram(idx, dst.len())`.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, scatter};
///
/// let client = gridcl_testing::test_client();
/// let src = DeviceBuffer::from_host(&client, &[1i32, 2, 3]).unwrap();
/// let idx = DeviceBuffer::from_host(&client, &[3u32, 0, 7]).unwrap();
/// let mut dst = DeviceBuffer::from_host(&client, &[0i32; 4]).unwrap();
/// scatter(&src, &idx, &mut dst).unwrap();
/// assert_eq!(dst.to_host().unwrap(), vec![2, 0, 0, 1]); // index 7 is dropped
/// ```
pub fn scatter<T: DeviceScalar>(
    src: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
    dst: &mut DeviceBuffer<T>,
) -> Result<()> {
    if idx.len() != src.len() {
        return Err(GridError::LengthMismatch {
            expected: src.len(),
            found: idx.len(),
        });
    }
    let dst_len = u32::try_from(dst.len()).map_err(|_| GridError::TooLarge)?;
    if src.len() == 0 || dst_len == 0 {
        return Ok(());
    }
    #[cfg(debug_assertions)]
    {
        let counts = super::histogram(idx, dst_len)?;
        let max = super::reduce(&counts, crate::ReduceOp::Max)?.to_host()?[0];
        assert!(
            max <= 1,
            "scatter: duplicate index ({max} writes to one element)"
        );
    }
    let client = src.client();
    let len = u32::try_from(src.len()).map_err(|_| GridError::TooLarge)?;
    let plan =
        LaunchPlan::elementwise(client, KernelKind::Gather, src.len(), size_of::<T>(), 1, 1)?;
    debug_assert_bindings(client, 3);
    // SAFETY: the kernel reads src[j] and idx[j] only for j < len = src.len() = idx.len(), and
    // writes dst[k] only for k < dst_len = dst.len().
    unsafe {
        scatter_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count,
            plan.cube_dim,
            src.arg(),
            idx.arg(),
            dst.arg(),
            len,
            dst_len,
            plan.items_per_unit,
        );
    }
    Ok(())
}
