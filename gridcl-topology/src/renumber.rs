//! Entity creation, step 7 (serial): owned entities ordered by the smallest flat cell index
//! that touches them, applied to entity → vertex and cell → entity.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, debug_assert_bindings,
};

/// `out[i] = i` for `i < len`.
#[cube(launch_unchecked)]
fn iota_kernel(out: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for k in 0..items_per_unit as usize {
        let i = first + k;
        if i < len as usize {
            out[i] = i as u32;
        }
    }
}

/// The buffer 0, 1, …, len − 1.
pub(crate) fn iota(client: &Client, len: usize) -> Result<DeviceBuffer<u32>> {
    let out = DeviceBuffer::<u32>::uninit(client, len)?;
    if len == 0 {
        return Ok(out);
    }
    let n = u32::try_from(len).map_err(|_| GridError::TooLarge)?;
    let plan = LaunchPlan::elementwise(client, KernelKind::Fill, len, size_of::<u32>(), 1, 1)?;
    debug_assert_bindings(client, 1);
    // SAFETY: `out` holds `len` words, and the kernel writes only indices below `len`.
    unsafe {
        iota_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            out.arg(),
            n,
            plan.items_per_unit,
        );
    }
    Ok(out)
}

/// The renumbering of the `runs` provisional entities of one type: `old2new[old]`. Provisional
/// numbering for now: the identity.
pub(crate) fn renumber(client: &Client, runs: usize) -> Result<DeviceBuffer<u32>> {
    iota(client, runs)
}
