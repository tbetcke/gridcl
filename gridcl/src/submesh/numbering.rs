//! The index maps of child cells (ADR-0039, item 6): a child cell inherits the owner and ghost
//! status of its parent entity. Owned children are numbered by an exclusive scan of the owned
//! counts, in parent local order; a ghost child receives its number from its owner, which is
//! also the owner of the parent entity, by a forward scatter over the parent block's index map.
//! Ghost children keep the parent's ghost order, which is (owner rank, child global index)
//! because child numbers grow with parent numbers on every owner.

use cubecl::prelude::*;
use gridcl_device::{DeviceBuffer, GridError, Result, debug_assert_bindings};
use gridcl_parallel::{CommDyn, CommExt, IndexMap};

use super::{fit, plan};

/// `val[2q] = hi`, `val[2q + 1] = lo` of begin + i, with q = parents[i] − offset, for the
/// owned children i < len.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn owned_numbers_kernel(
    parents: &[u32],
    val: &mut [u32],
    len: u32,
    offset: u32,
    begin_hi: u32,
    begin_lo: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let q = (parents[i] - offset) as usize;
            let lo = begin_lo + i as u32;
            val[2 * q] = begin_hi + u32::cast_from(lo < begin_lo);
            val[2 * q + 1] = lo;
        }
    }
}

/// `out[3k..3k + 3] = [q, val[2q], val[2q + 1]]` with q = parents[num_owned + k] − offset, for
/// the ghost children k < len.
#[cube(launch_unchecked)]
fn ghost_numbers_kernel(
    parents: &[u32],
    val: &[u32],
    out: &mut [u32],
    len: u32,
    num_owned: u32,
    offset: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let k = first + it;
        if k < len as usize {
            let q = parents[num_owned as usize + k] - offset;
            out[3 * k] = q;
            out[3 * k + 1] = val[2 * q as usize];
            out[3 * k + 2] = val[2 * q as usize + 1];
        }
    }
}

/// The index map of one child cell block: `parents` are the flat parent entities of its cells
/// (owned first, `num_owned` of them), in the parent block with index map `parent_map` and
/// first flat index `offset`. One rank: `IndexMap::new`. More ranks: the ghost children's
/// numbers come from their owners by a forward scatter, and their (parent index, number) rows
/// go to the host for `IndexMap::with_ghosts` (CLAUDE.md, the child ghost lists). Collective.
///
/// # Errors
///
/// `TooLarge`; allocation, device and communication errors, and those of
/// `IndexMap::with_ghosts`.
pub(crate) fn cell_map(
    comm: &dyn CommDyn,
    parents: &DeviceBuffer<u32>,
    num_owned: usize,
    parent_map: &IndexMap,
    offset: usize,
) -> Result<IndexMap> {
    let n = parents.len();
    if comm.size() == 1 {
        return IndexMap::new(comm, n);
    }
    let client = parents.client();
    let begin = comm.exscan_sum(u64::try_from(num_owned).map_err(|_| GridError::TooLarge)?);
    let mut val = DeviceBuffer::<u32>::zeros(
        client,
        parent_map
            .num_local()
            .checked_mul(2)
            .ok_or(GridError::TooLarge)?,
    )?;
    if num_owned > 0 {
        fit(val.len())?;
        let launch = plan(client, num_owned)?;
        debug_assert_bindings(client, 2);
        // SAFETY: owned children i < num_owned ≤ parents.len() read parents[i], a flat index in
        // [offset, offset + num_local) of the parent block, and write val[2q] and val[2q + 1]
        // with q < num_local, within its 2 · num_local words.
        unsafe {
            owned_numbers_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                parents.arg(),
                val.arg(),
                fit(num_owned)?,
                fit(offset)?,
                (begin >> 32) as u32,
                begin as u32,
                launch.items_per_unit,
            );
        }
    }
    parent_map.scatterer(client, 2)?.forward(&mut val)?;
    let num_ghosts = n - num_owned;
    let mut ghosts = Vec::with_capacity(num_ghosts);
    let mut owners = Vec::with_capacity(num_ghosts);
    if num_ghosts > 0 {
        let out = DeviceBuffer::<u32>::uninit(
            client,
            num_ghosts.checked_mul(3).ok_or(GridError::TooLarge)?,
        )?;
        fit(out.len())?;
        let launch = plan(client, num_ghosts)?;
        debug_assert_bindings(client, 3);
        // SAFETY: ghost children k < num_ghosts read parents[num_owned + k] < n, a flat index in
        // the parent block, so q < num_local and val[2q + 1] < 2 · num_local; they write
        // out[3k + 2] < 3 · num_ghosts, its length.
        unsafe {
            ghost_numbers_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                parents.arg(),
                val.arg(),
                out.arg(),
                fit(num_ghosts)?,
                fit(num_owned)?,
                fit(offset)?,
                launch.items_per_unit,
            );
        }
        // The child ghost lists for IndexMap::with_ghosts (CLAUDE.md, ADR-0039 item 18).
        let parent_owners = parent_map.ghost_owners();
        let parent_owned = parent_map.num_owned();
        for row in out.to_host()?.as_chunks::<3>().0 {
            ghosts.push(((u64::from(row[1]) << 32) | u64::from(row[2])) as i64);
            owners.push(parent_owners[row[0] as usize - parent_owned]);
        }
    }
    IndexMap::with_ghosts(comm, num_owned, ghosts, owners)
}
