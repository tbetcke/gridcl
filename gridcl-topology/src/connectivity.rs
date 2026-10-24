//! Entity creation, step 9: entity → cells as a `DeviceCsr`, rows in cell order, by a stable
//! radix sort of the entity ids with the flat cell index as payload; offsets from `histogram`
//! and `exclusive_scan`. Deterministic without atomics.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceCsr, DeviceStrided, GridError, KernelKind, LaunchPlan, Result,
    debug_assert_bindings, exclusive_scan, histogram, radix_sort,
};

use crate::renumber::one_word_keys;

/// The (entity, flat cell) pairs of one cell block's downward rows, written at `base`:
/// `entities[base + i] = min(rows[i], last_entity)`, `cells[base + i] = cell_offset + i / k`.
/// The clamp matches the topology kernels, which treat a vertex index past the local vertices
/// as the last vertex; downward rows of created entities are always in range.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pairs_kernel(
    rows: &[u32],
    entities: &mut [u32],
    cells: &mut [u32],
    len: u32,
    last_entity: u32,
    k: u32,
    base: u32,
    cell_offset: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let o = base as usize + i;
            entities[o] = min(rows[i], last_entity);
            cells[o] = cell_offset + i as u32 / k;
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// entity → cells over `num_entities` entities from the downward rows of every cell block
/// (cell → entity with flat entity indices). Row e lists the flat indices of the cells that
/// contain entity e, ascending.
pub(crate) fn upward(
    client: &Client,
    downward: &[&DeviceStrided<u32>],
    num_entities: usize,
) -> Result<DeviceCsr> {
    let total = downward
        .iter()
        .try_fold(0usize, |acc, d| acc.checked_add(d.data.len()))
        .ok_or(GridError::TooLarge)?;
    let entities = DeviceBuffer::<u32>::uninit(client, total)?;
    let cells = DeviceBuffer::<u32>::uninit(client, total)?;
    let last_entity = fit(num_entities)?.saturating_sub(1);
    let (mut base, mut cell_offset) = (0usize, 0usize);
    for d in downward {
        let len = d.data.len();
        let k = d.stride as usize;
        if len > 0 {
            let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
            debug_assert_bindings(client, 3);
            // SAFETY: the kernel reads rows[i] for i < len, the rows' length, and writes
            // entities and cells at base + i < base + len ≤ total, their length.
            unsafe {
                pairs_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    d.data.arg(),
                    entities.arg(),
                    cells.arg(),
                    fit(len)?,
                    last_entity,
                    fit(k)?,
                    fit(base)?,
                    fit(cell_offset)?,
                    launch.items_per_unit,
                );
            }
        }
        base += len;
        cell_offset += len / k.max(1);
    }
    fit(cell_offset)?;
    let counts = histogram(&entities, fit(num_entities)?)?;
    let offsets = exclusive_scan(&counts)?;
    // The pairs are in flat cell order, so a stable sort by entity keeps each row in cell
    // order.
    let (_, indices) = radix_sort(&one_word_keys(&entities)?, &cells)?;
    Ok(DeviceCsr { offsets, indices })
}
