//! Fill (KernelKind::Fill), generic over the vector size.

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::plan::debug_assert_bindings;
use crate::{DeviceBuffer, DeviceScalar, KernelKind, LaunchPlan, Result, VECTOR_PAD};

/// `x[v] = value` for every vector `v < num_vectors`, by the loop of Listing B5.
#[cube(launch_unchecked)]
fn fill_kernel<T: Numeric + LaunchArg<RuntimeArg = T>, N: Size>(
    x: &mut [Vector<T, N>],
    value: T,
    num_vectors: u32,
    items_per_unit: u32,
) {
    let fill = Vector::<T, N>::broadcast(value);
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            x[v] = fill;
        }
    }
}

/// Writes `v` over the padded length. An empty buffer is not touched.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, fill};
///
/// let client = gridcl_testing::test_client();
/// let mut x = DeviceBuffer::<i32>::uninit(&client, 5).unwrap();
/// fill(&mut x, -3).unwrap();
/// assert_eq!(x.to_host().unwrap(), vec![-3; 5]);
/// ```
pub fn fill<T: DeviceScalar>(x: &mut DeviceBuffer<T>, v: T) -> Result<()> {
    if x.len() == 0 {
        return Ok(());
    }
    launch_fill(x.client(), x.handle(), x.padded_len(), v)
}

/// Writes `value` to all `padded` elements of `T` behind `handle`. Vectors are at most as wide
/// as the largest power of two (up to `VECTOR_PAD`) that divides `padded`.
pub(crate) fn launch_fill<T: CubeElement + Numeric + LaunchArg<RuntimeArg = T>>(
    client: &Client,
    handle: &Handle,
    padded: usize,
    value: T,
) -> Result<()> {
    debug_assert!(padded > 0);
    let max_vector = VECTOR_PAD.min(1 << padded.trailing_zeros());
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::Fill,
        padded,
        size_of::<T>(),
        max_vector,
        1,
    )?;
    let n = plan.vector_size;
    debug_assert_bindings(client, 1);
    // SAFETY: the handle holds `padded` elements of `T`, `padded / n` whole vectors since `n`
    // divides `max_vector`, which divides `padded`; the plan's `num_vectors` is `padded / n`. The kernel writes only
    // vectors below `num_vectors`.
    unsafe {
        fill_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            BufferArg::from_raw_parts(handle.clone(), padded / n),
            value,
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    Ok(())
}
