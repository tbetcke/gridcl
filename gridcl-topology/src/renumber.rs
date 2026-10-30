//! Entity creation, step 7: owned entities ordered by the smallest flat cell index that touches
//! them, then the ghosts by (owner rank, global index), applied to entity → vertex and
//! cell → entity.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, ReduceOp, Result, SortKeys,
    debug_assert_bindings, fill, gather, radix_sort, run_length, scatter, segmented_reduce,
};

use crate::emission::EmissionPlan;
use crate::ownership::GHOST_CELL;

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
    let plan = LaunchPlan::elementwise(client, KernelKind::Scalar, len, size_of::<u32>(), 1, 1)?;
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
/// one block's range [start, end) and belongs to a real cell (c < cells), with the bit
/// `GHOST_CELL` (crate::ownership) set for a ghost cell (c ≥ owned); other positions are left untouched.
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
    owned: u32,
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
                    values[p] = (cell_offset + c) | (u32::cast_from(c >= owned) << 31);
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
        let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
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

/// The smallest touching-cell value of each of the `runs` provisional entities of one type,
/// from the sorted keys and payloads of `plan`: a segmented minimum over the sorted runs of the
/// flat cell index of each occurrence, with the bit `GHOST_CELL` set for a ghost cell, so an
/// entity lies in an owned cell exactly if its value is below `GHOST_CELL`. The sentinel group
/// touches no cell and gets u32::MAX. `cell_offsets[b]` is the first flat cell index of cell
/// block b and `owned[b]` its owned cell count; the flat cell count must be below `GHOST_CELL`.
pub(crate) fn touching_cells(
    client: &Client,
    sorted: &SortKeys,
    payload: &DeviceBuffer<u32>,
    plan: &EmissionPlan,
    cell_offsets: &[usize],
    owned: &[usize],
) -> Result<DeviceBuffer<u32>> {
    let len = payload.len();
    let mut values = DeviceBuffer::<u32>::uninit(client, len)?;
    fill(&mut values, u32::MAX)?;
    for range in plan.ranges.iter().filter(|r| r.cells > 0) {
        let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
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
                fit(owned[range.block])?,
                fit(cell_offsets[range.block])?,
                launch.items_per_unit,
            );
        }
    }
    let offsets = run_length(sorted)?.offsets; // the count read of run-length
    segmented_reduce(&values, &offsets, ReduceOp::Min)
}

/// Step 7, first pass: the renumbering `old2new` of the `runs` provisional entities of one
/// type that places the owned ones, by their smallest touching-cell value `mins`
/// ([`touching_cells`]): entities in an owned cell first, by their smallest owned cell, then
/// those only in ghost cells, by their smallest cell; ties in provisional order (a stable
/// sort). The provisional ids `ghosts` (ascending), owned by other ranks, follow every owned
/// entity, in provisional order, until [`place_ghosts`] orders them. An owned entity lies in
/// an owned cell, so with ghosts the owned entities come first; on one rank every cell is
/// owned, so entities are ordered by the smallest flat cell index that touches them. The
/// sentinel group comes last.
pub(crate) fn renumber(
    client: &Client,
    mut mins: DeviceBuffer<u32>,
    runs: usize,
    ghosts: &[u32],
) -> Result<DeviceBuffer<u32>> {
    if !ghosts.is_empty() {
        // The ghosts' owners and global numbers back (ADR-0026, item 14).
        let idx = DeviceBuffer::from_host(client, ghosts)?;
        let mut last = DeviceBuffer::<u32>::uninit(client, ghosts.len())?;
        fill(&mut last, GHOST_CELL)?;
        scatter(&last, &idx, &mut mins)?;
    }
    let keys = one_word_keys(&mins)?;
    let ids = iota(client, runs)?;
    let (_, new2old) = radix_sort(&keys, &ids)?;
    let mut old2new = DeviceBuffer::<u32>::uninit(client, runs)?;
    scatter(&ids, &new2old, &mut old2new)?;
    Ok(old2new)
}

/// Step 7, second pass: `old2new[ghosts[k]] = new[k]`, the ghosts' final local indices.
pub(crate) fn place_ghosts(
    old2new: &mut DeviceBuffer<u32>,
    ghosts: &[u32],
    new: &[u32],
) -> Result<()> {
    if ghosts.is_empty() {
        return Ok(());
    }
    // Global numbers back (ADR-0026, item 14).
    let client = old2new.client().clone();
    let idx = DeviceBuffer::from_host(&client, ghosts)?;
    let new = DeviceBuffer::from_host(&client, new)?;
    scatter(&new, &idx, old2new)
}

/// `old2new[g]` for the provisional ids `g` in `entities`, on the host: the local indices of
/// the owned sharing candidates, which their global numbers need (ADR-0026, item 14).
pub(crate) fn local_indices(old2new: &DeviceBuffer<u32>, entities: &[u32]) -> Result<Vec<u32>> {
    if entities.is_empty() {
        return Ok(Vec::new());
    }
    let idx = DeviceBuffer::from_host(old2new.client(), entities)?;
    gather(old2new, &idx)?.to_host()
}
