//! Entity creation, step 7 (serial): owned entities ordered by the smallest flat cell index
//! that touches them, applied to entity → vertex and cell → entity.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, ReduceOp, Result, SortKeys,
    debug_assert_bindings, fill, radix_sort, run_length, scatter, segmented_reduce,
};

use crate::emission::EmissionPlan;

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

/// The flat cell index of the occurrence at each sorted position whose emission index lies in
/// one block's range [start, end) and belongs to a real cell (c < cells); other positions are
/// left untouched.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn occurrence_cells_kernel(
    payload: &[u32],
    values: &mut [u32],
    len: u32,
    start: u32,
    end: u32,
    padded: u32,
    cells: u32,
    cell_offset: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for k in 0..items_per_unit as usize {
        let p = first + k;
        if p < len as usize {
            let e = payload[p];
            if e >= start && e < end {
                let c = (e - start) % padded;
                if c < cells {
                    values[p] = cell_offset + c;
                }
            }
        }
    }
}

/// `dst[i] = src[i]` for `i < len`: a buffer into the plane of a one-word `SortKeys`.
#[cube(launch_unchecked)]
fn copy_words_kernel(src: &[u32], dst: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for k in 0..items_per_unit as usize {
        let i = first + k;
        if i < len as usize {
            dst[i] = src[i];
        }
    }
}

/// One-word sort keys holding `src`.
pub(crate) fn one_word_keys(src: &DeviceBuffer<u32>) -> Result<SortKeys> {
    let client = src.client();
    let len = src.len();
    let mut keys = SortKeys::uninit(client, len, 1)?;
    if len > 0 {
        let launch = LaunchPlan::elementwise(client, KernelKind::Copy, len, 4, 1, 1)?;
        debug_assert_bindings(client, 2);
        // SAFETY: `src` holds `len` words and the one plane of `keys` pad_len(len) ≥ len; the
        // kernel copies indices below `len`.
        unsafe {
            copy_words_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                src.arg(),
                keys.words_mut().arg(),
                fit(len)?,
                launch.items_per_unit,
            );
        }
    }
    Ok(keys)
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// Step 7 (serial, every entity owned): the renumbering `old2new` of the `runs` provisional
/// entities of one type, from the sorted keys and payloads of `plan`. Entities are ordered by
/// the smallest flat cell index that touches them (a segmented minimum over the sorted runs),
/// ties in provisional order (a stable sort). The sentinel group touches no cell and comes
/// last. `cell_offsets[b]` is the first flat cell index of cell block b.
pub(crate) fn renumber(
    client: &Client,
    sorted: &SortKeys,
    payload: &DeviceBuffer<u32>,
    plan: &EmissionPlan,
    cell_offsets: &[usize],
    runs: usize,
) -> Result<DeviceBuffer<u32>> {
    let len = payload.len();
    let mut values = DeviceBuffer::<u32>::uninit(client, len)?;
    fill(&mut values, u32::MAX)?;
    for range in plan.ranges.iter().filter(|r| r.cells > 0) {
        let launch = LaunchPlan::elementwise(client, KernelKind::Gather, len, 4, 1, 1)?;
        debug_assert_bindings(client, 2);
        // SAFETY: the kernel reads payload[p] and writes values[p] only for p < len, the
        // length of both.
        unsafe {
            occurrence_cells_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                payload.arg(),
                values.arg(),
                fit(len)?,
                fit(range.start)?,
                fit(range.end())?,
                fit(range.padded)?,
                fit(range.cells)?,
                fit(cell_offsets[range.block])?,
                launch.items_per_unit,
            );
        }
    }
    let offsets = run_length(sorted)?.offsets; // the count read of run-length
    let mins = segmented_reduce(&values, &offsets, ReduceOp::Min)?;
    let keys = one_word_keys(&mins)?;
    let ids = iota(client, runs)?;
    let (_, new2old) = radix_sort(&keys, &ids)?;
    let mut old2new = DeviceBuffer::<u32>::uninit(client, runs)?;
    scatter(&ids, &new2old, &mut old2new)?;
    Ok(old2new)
}
