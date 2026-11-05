//! Selection of the child cells of a submesh (ADR-0039, items 6, 7, 10 and 11): a mask over
//! the parent entities, per parent block a flag kernel, an exclusive scan, a count read and a
//! compaction, then the child vertex rows and origin ids.
//!
//! Children keep the parent's local order within a block. The parent's entities are
//! [owned | ghost] with ghosts by (owner rank, global index), and a child's global index grows
//! with its parent's on every owner, so the restriction of that order is the child order of
//! ADR-0039 item 6 for cells.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, ReduceOp, Result, debug_assert_bindings,
    exclusive_scan, gather, reduce,
};
use gridcl_dims::{CellType, TopoDim};
use gridcl_parallel::{CommDyn, CommExt};
use gridcl_topology::{FacetCells, Topology};

use super::{fit, plan};

/// The child cells of one child block, before the vertex compaction.
pub(crate) struct Selected {
    /// The child cell type.
    pub(crate) cell: CellType,
    /// The flat parent entity of each child, in child order.
    pub(crate) parents: DeviceBuffer<u32>,
    /// The number of owned children; they come first.
    pub(crate) num_owned: usize,
    /// The child vertex rows, in parent local vertex indices.
    pub(crate) rows: DeviceStrided<u32>,
    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each child.
    pub(crate) origin_ids: DeviceBuffer<u32>,
    /// Facet children: the flat parent cell and the local facet index of each child.
    pub(crate) facet_of: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
}

/// The facet mask of `boundary` (`tagged` = 0) or `facet_submesh` (`tagged` = 1):
/// `mask[f] = 1` for a facet with one owned cell globally (and, with tags, a non-zero
/// `tags[f]`); `status[f]` = 2 for a facet with more than two cells, 1 for a tagged facet with
/// two, else 0. Scalar.
#[cube(launch_unchecked)]
fn facet_mask_kernel(
    counts: &[u32],
    tags: &[u32],
    mask: &mut [u32],
    status: &mut [u32],
    len: u32,
    items_per_unit: u32,
    #[comptime] tagged: bool,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = counts[i];
            let mut t = 1u32;
            if tagged {
                t = u32::cast_from(tags[i] > 0);
            }
            mask[i] = u32::cast_from(c == 1) * t;
            let mut s = 0u32;
            if c > 2 {
                s = 2u32;
            } else if c == 2 {
                s = t * u32::cast_from(tagged);
            }
            status[i] = s;
        }
    }
}

/// `flags[i] = mask[offset + i]` for i < len.
#[cube(launch_unchecked)]
fn block_flags_kernel(mask: &[u32], flags: &mut [u32], len: u32, offset: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            flags[i] = mask[offset as usize + i];
        }
    }
}

/// `out = [pos[num_owned], pos[len]]`: the owned and total counts of an exclusive scan of
/// length len + 1. Unit 0 only.
#[cube(launch_unchecked)]
fn counts_kernel(pos: &[u32], out: &mut [u32], num_owned: u32, len: u32) {
    if ABSOLUTE_POS == 0 {
        out[0] = pos[num_owned as usize];
        out[1] = pos[len as usize];
    }
}

/// `out[pos[i]] = offset + i` where `flags[i] = 1`, for i < len.
#[cube(launch_unchecked)]
fn compact_kernel(
    flags: &[u32],
    pos: &[u32],
    out: &mut [u32],
    len: u32,
    offset: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize && flags[i] == 1 {
            out[pos[i] as usize] = offset + i as u32;
        }
    }
}

/// The vertex rows of the facet children i < len whose parent cell lies in the parent block
/// `[cell_begin, cell_end)`: `out[i · k + j] = rows[(c − cell_begin) · stride + table[f · k
/// + j]]`, with c the parent cell, f the local facet and `table` the block's facet vertex
/// lists (`sub_entities`, k vertices each). Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn facet_rows_kernel(
    cells: &[u32],
    facets: &[u32],
    rows: &[u32],
    table: &[u32],
    out: &mut [u32],
    len: u32,
    k: u32,
    stride: u32,
    cell_begin: u32,
    cell_end: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = cells[i];
            if c >= cell_begin && c < cell_end {
                let base = (c - cell_begin) as usize * stride as usize;
                let f = facets[i] as usize;
                for j in 0..k as usize {
                    out[i * k as usize + j] = rows[base + table[f * k as usize + j] as usize];
                }
            }
        }
    }
}

/// The origin ids 8 · id(c) + f of the facet children i < len whose parent cell c lies in
/// `[cell_begin, cell_end)`, f being the local facet, with the parent block's interleaved ids
/// `ids`. On the biased high word w = hi + 2^31, 8 · id + f has the high word
/// 8w + 2^31 + (lo >> 29) (mod 2^32), and it fits i64 iff hi ∈ [−2^28, 2^28), that is
/// w ∈ [2^31 − 2^28, 2^31 + 2^28); otherwise `status[0]` is set to 1. Each child sets
/// `status[present]`, marking the parent block as present. Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::manual_range_contains)] // no ranges in kernels
fn facet_ids_kernel(
    cells: &[u32],
    facets: &[u32],
    ids: &[u32],
    out: &mut [u32],
    status: &[Atomic<u32>],
    len: u32,
    cell_begin: u32,
    cell_end: u32,
    present: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = cells[i];
            if c >= cell_begin && c < cell_end {
                let local = (c - cell_begin) as usize;
                let w = ids[2 * local];
                let lo = ids[2 * local + 1];
                if w < 1879048192u32 || w >= 2415919104u32 {
                    status[0].store(1u32);
                }
                status[present as usize].store(1u32);
                out[2 * i] = (w << 3u32) + 2147483648u32 + (lo >> 29u32);
                out[2 * i + 1] = (lo << 3u32) | facets[i];
            }
        }
    }
}

/// The facet mask of `facet_cells` and the agreed errors of ADR-0039 item 11: `NonManifold`
/// if any facet has more than two cells globally, else `InteriorFacet` if a tagged facet
/// (`tagged[f]` > 0) has two. `tagged = None` selects every boundary facet. One read of the
/// local status, combined by `comm.max`. Collective.
pub(crate) fn facet_mask(
    comm: &dyn CommDyn,
    facet_cells: &FacetCells,
    tagged: Option<&DeviceBuffer<u32>>,
) -> Result<DeviceBuffer<u32>> {
    let counts = &facet_cells.counts;
    let client = counts.client();
    let n = counts.len();
    let mask = DeviceBuffer::<u32>::zeros(client, n)?;
    let status = DeviceBuffer::<u32>::zeros(client, n)?;
    if n > 0 {
        let launch = plan(client, n)?;
        debug_assert_bindings(client, 4);
        // SAFETY: the kernel reads counts[i] and tags[i] (tagged[i] when given, else counts
        // again; both hold n words) and writes mask[i] and status[i] for i < n, their length.
        unsafe {
            facet_mask_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                counts.arg(),
                tagged.unwrap_or(counts).arg(),
                mask.arg(),
                status.arg(),
                fit(n)?,
                launch.items_per_unit,
                tagged.is_some(),
            );
        }
    }
    // The agreed NonManifold and InteriorFacet flags (CLAUDE.md, ADR-0039 item 18).
    let local = if n > 0 {
        reduce(&status, ReduceOp::Max)?.to_host()?[0]
    } else {
        0
    };
    match comm.max(local) {
        0 => Ok(mask),
        1 => Err(GridError::InteriorFacet),
        _ => Err(GridError::NonManifold),
    }
}

/// One parent block for [`select`]: its first flat index, owned and local counts.
pub(crate) struct ParentBlock {
    pub(crate) cell: CellType,
    pub(crate) offset: usize,
    pub(crate) num_owned: usize,
    pub(crate) num_local: usize,
}

/// The masked entities of one parent block.
pub(crate) struct Chosen {
    pub(crate) cell: CellType,
    /// The flat parent indices, in parent local order.
    pub(crate) parents: DeviceBuffer<u32>,
    /// The number of owned parents; they come first.
    pub(crate) num_owned: usize,
}

/// The masked entities of each parent block with a non-zero global count, the same list on
/// every rank (ADR-0039, item 10). One count read per block (owned and total); collective.
pub(crate) fn select(
    comm: &dyn CommDyn,
    mask: &DeviceBuffer<u32>,
    blocks: &[ParentBlock],
) -> Result<Vec<Chosen>> {
    let client = mask.client();
    let mut out = Vec::new();
    for block in blocks {
        let n = block.num_local;
        let (owned, total, parents) = if n == 0 {
            (0, 0, DeviceBuffer::<u32>::uninit(client, 0)?)
        } else {
            let flags = DeviceBuffer::<u32>::uninit(client, n)?;
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 2);
            // SAFETY: the kernel reads mask[offset + i] < offset + n ≤ mask.len() (the blocks
            // tile the flat entities) and writes flags[i] for i < n, its length.
            unsafe {
                block_flags_kernel::launch_unchecked(
                    client,
                    launch.cube_count.clone(),
                    launch.cube_dim,
                    mask.arg(),
                    flags.arg(),
                    fit(n)?,
                    fit(block.offset)?,
                    launch.items_per_unit,
                );
            }
            let pos = exclusive_scan(&flags)?;
            let counts = DeviceBuffer::<u32>::uninit(client, 2)?;
            let one = plan(client, 1)?;
            debug_assert_bindings(client, 2);
            // SAFETY: one unit reads pos[num_owned] and pos[n], within its n + 1 words, and
            // writes the two words of `counts`.
            unsafe {
                counts_kernel::launch_unchecked(
                    client,
                    one.cube_count,
                    one.cube_dim,
                    pos.arg(),
                    counts.arg(),
                    fit(block.num_owned)?,
                    fit(n)?,
                );
            }
            // The per-block count read of compaction (CLAUDE.md, ADR-0039 item 18).
            let c = counts.to_host()?;
            let (owned, total) = (c[0] as usize, c[1] as usize);
            let parents = DeviceBuffer::<u32>::uninit(client, total)?;
            if total > 0 {
                debug_assert_bindings(client, 3);
                // SAFETY: the kernel reads flags[i] and pos[i] for i < n and writes
                // parents[pos[i]] for flagged i, where pos[i] < pos[n] = total, its length.
                unsafe {
                    compact_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        flags.arg(),
                        pos.arg(),
                        parents.arg(),
                        fit(n)?,
                        fit(block.offset)?,
                        launch.items_per_unit,
                    );
                }
            }
            (owned, total, parents)
        };
        if comm.sum(total as u64) > 0 {
            out.push(Chosen {
                cell: block.cell,
                parents,
                num_owned: owned,
            });
        }
    }
    Ok(out)
}

/// One parent cell block for [`restrict_rows`].
pub(super) struct RowSource<'a> {
    /// The block's rows (vertex rows or dofmap).
    pub(super) rows: &'a DeviceStrided<u32>,
    /// The block's local cell count.
    pub(super) num_local: usize,
    /// k entries per facet of the block's cell type; `None` if no child has a parent cell in
    /// the block.
    pub(super) table: Option<&'a [u32]>,
}

/// The rows of the facet children (parent cells `cells`, local facets `facets`) restricted from
/// the parent blocks' rows: for the block b holding cell c, entry j of child i is
/// `rows_b[(c − begin_b) · stride_b + table_b[f · k + j]]`. Children whose block has no table
/// keep unspecified rows.
pub(super) fn restrict_rows(
    cells: &DeviceBuffer<u32>,
    facets: &DeviceBuffer<u32>,
    blocks: &[RowSource<'_>],
    k: usize,
) -> Result<DeviceBuffer<u32>> {
    let client = cells.client();
    let n = cells.len();
    let out = DeviceBuffer::<u32>::uninit(client, n.checked_mul(k).ok_or(GridError::TooLarge)?)?;
    fit(out.len())?;
    let mut cell_begin = 0usize;
    for b in blocks {
        let (rows, num_local) = (b.rows, b.num_local);
        let cell_end = cell_begin + num_local;
        if let Some(table) = b.table
            && n > 0
            && num_local > 0
        {
            let table = DeviceBuffer::from_host(client, table)?;
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 5);
            // SAFETY: children i < n read cells[i] and facets[i] (n words each); those with a
            // cell c in [cell_begin, cell_end) read the block's row c − cell_begin, at most
            // stride words from (c − cell_begin) · stride < rows.len(), at the positions
            // table[f · k + j] < stride (f < the block's facet count, the table holds k per
            // facet), and write out[i · k + j] < n · k, its length.
            unsafe {
                facet_rows_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    cells.arg(),
                    facets.arg(),
                    rows.data.arg(),
                    table.arg(),
                    out.arg(),
                    fit(n)?,
                    fit(k)?,
                    rows.stride,
                    fit(cell_begin)?,
                    fit(cell_end)?,
                    launch.items_per_unit,
                );
            }
        }
        cell_begin = cell_end;
    }
    Ok(out)
}

/// The facet children of the selected facets `parents` (flat facet indices, one child block of
/// type `cell`): their parent cell and local facet from `facet_cells`, their vertex rows in
/// the parent cell's local facet order (ADR-0039, item 8) and their origin ids
/// 8 · id(parent cell) + local facet (item 7). Also returns whether an origin id overflowed
/// i64, and which parent cell blocks hold a parent cell (for the element check of item 8),
/// read together with the overflow flag.
pub(crate) fn facet_children<TD: TopoDim>(
    topology: &Topology<TD>,
    facet_cells: &FacetCells,
    cell: CellType,
    parents: DeviceBuffer<u32>,
    num_owned: usize,
) -> Result<(Selected, bool, Vec<bool>)> {
    let client = topology.client();
    let n = parents.len();
    let cells = gather(&facet_cells.cell, &parents)?;
    let facets = gather(&facet_cells.local_facet, &parents)?;
    let k = gridcl_refcell::num_sub_entities(cell, 0);
    let tables: Vec<Vec<u32>> = topology
        .cell_blocks()
        .iter()
        .map(|b| facet_vertex_table(b.cell.into(), cell, k))
        .collect();
    let blocks: Vec<_> = topology
        .cell_blocks()
        .iter()
        .zip(&tables)
        .map(|(b, t)| RowSource {
            rows: &b.vertices,
            num_local: b.index_map.num_local(),
            table: Some(&t[..]),
        })
        .collect();
    let rows = restrict_rows(&cells, &facets, &blocks, k)?;
    let origin_ids =
        DeviceBuffer::<u32>::uninit(client, n.checked_mul(2).ok_or(GridError::TooLarge)?)?;
    fit(origin_ids.len())?;
    let num_blocks = topology.cell_blocks().len();
    // Word 0: overflow; word 1 + b: a child has its parent cell in block b.
    let status = DeviceBuffer::<u32>::zeros(client, 1 + num_blocks)?;
    let mut cell_begin = 0usize;
    for (b, block) in topology.cell_blocks().iter().enumerate() {
        let cell_end = cell_begin + block.index_map.num_local();
        if n > 0 && cell_end > cell_begin {
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 5);
            // SAFETY: children i < n read cells[i] and facets[i] (n words each); those with a
            // cell c in [cell_begin, cell_end) read words 2(c − cell_begin) and
            // 2(c − cell_begin) + 1 of the block's two words per cell, write out[2i] and
            // out[2i + 1] < 2n, its length, and status words 0 and 1 + b < 1 + num_blocks.
            unsafe {
                facet_ids_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    cells.arg(),
                    facets.arg(),
                    block.origin_ids.arg(),
                    origin_ids.arg(),
                    status.arg(),
                    fit(n)?,
                    fit(cell_begin)?,
                    fit(cell_end)?,
                    fit(1 + b)?,
                    launch.items_per_unit,
                );
            }
        }
        cell_begin = cell_end;
    }
    // The agreed TooLarge flag, with the parent blocks present for the agreed element check
    // (CLAUDE.md, ADR-0039 items 8 and 18); the caller agrees both.
    let (overflowed, present) = if n > 0 {
        let s = status.to_host()?;
        (s[0] != 0, s[1..].iter().map(|&w| w != 0).collect())
    } else {
        (false, vec![false; num_blocks])
    };
    Ok((
        Selected {
            cell,
            parents,
            num_owned,
            rows: DeviceStrided {
                data: rows,
                stride: fit(k)?,
            },
            origin_ids,
            facet_of: Some((cells, facets)),
        },
        overflowed,
        present,
    ))
}

/// The facet vertex lists of `parent` for facets of type `child`, `k` per facet; facets of
/// another type hold `k` zeros.
fn facet_vertex_table(parent: CellType, child: CellType, k: usize) -> Vec<u32> {
    let tdim = gridcl_refcell::tdim(parent);
    let mut table = Vec::new();
    for (f, &t) in gridcl_refcell::sub_entity_types(parent, tdim - 1)
        .iter()
        .enumerate()
    {
        if t == child {
            table.extend(
                gridcl_refcell::sub_entities(parent, tdim - 1)[f]
                    .iter()
                    .map(|&v| v as u32),
            );
        } else {
            table.extend(std::iter::repeat_n(0u32, k));
        }
    }
    table
}
