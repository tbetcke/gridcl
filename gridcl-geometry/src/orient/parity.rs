//! `orient`, steps 1 and 2 (docs/design.md, "Manifold orientation"; ADR-0039, item 12): the
//! manifold check, the direction each cell induces on its facets, the exchange of the two
//! sides of every process-boundary facet, and the neighbour table with the parity per
//! interior facet.
//!
//! A cell induces a direction on each of its facets: for a surface, whether its boundary
//! cycle (A.6: list order for triangles, 0–1–3–2 for quadrilaterals) runs along the local
//! edge or against it, combined with the edge's permutation bit (set where the local edge runs
//! against the canonical orientation, which comes from origin ids); for a curve, whether the
//! point is the cell's start or its end. The direction bit is therefore relative to the
//! canonical orientation, the same on every rank. Two cells are consistent across a facet if
//! their bits differ: the parity is 1 where they are equal.
//!
//! The facet's two sides (one record per local cell at the facet, in the order of the
//! (tdim − 1, tdim) row) live in a flat `[facet][2]` array. A facet with two cells globally
//! but one local cell lies on a process boundary (or at the edge of the ghost layer). The
//! facet owner holds an owned cell at it (entity ownership, "Ownership of vertices and
//! entities"), so when the other cell is owned elsewhere exactly one non-owner rank owns a
//! cell at it: [`SideExchange`] sends that rank's record to the owner (`reverse(Max)`, the
//! other ranks contribute 0) and both records back to every holder (`forward`). This works in
//! every `GhostMode`.

use cubecl::prelude::*;
use gridcl_device::{
    ConnKey, DeviceBuffer, DeviceCsr, DeviceStrided, GridError, KernelKind, LaunchPlan, ReduceOp,
    Result, debug_assert_bindings, fill, reduce,
};
use gridcl_dims::{CellType, TopoDim};
use gridcl_parallel::{CommDyn, CommExt, ReverseOp, Scatterer};
use gridcl_refcell::sub_entities;
use gridcl_topology::{CellBlock, Connectivity, EntityBlock};

/// No record: an empty side, or a facet without a neighbour.
pub(crate) const NONE: u32 = u32::MAX;

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// A scalar launch over `len` items.
fn plan(client: &Client, len: usize) -> Result<LaunchPlan> {
    LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)
}

/// `NonManifold` if any facet has more than two owned cells globally (step 1); one read of
/// the local maximum, combined by `comm.max` (the agreed error flag, CLAUDE.md).
pub(crate) fn manifold_check(comm: &dyn CommDyn, counts: &DeviceBuffer<u32>) -> Result<()> {
    let local = if counts.len() > 0 {
        reduce(counts, ReduceOp::Max)?.to_host()?[0]
    } else {
        0
    };
    if comm.max(local) > 2 {
        Err(GridError::NonManifold)
    } else {
        Ok(())
    }
}

/// The cycle bits of a cell type: bit e set where the boundary cycle runs against local
/// facet e (from its second vertex to its first). For intervals, bit e is the local facet
/// index (start 0, end 1).
pub(crate) fn cycle_bits(cell: CellType) -> u32 {
    let cycle: &[usize] = match cell {
        CellType::Interval => return 0b10,
        CellType::Triangle => &[0, 1, 2],
        CellType::Quadrilateral => &[0, 1, 3, 2],
        _ => unreachable!("orient needs cells of dimension 1 or 2"),
    };
    let m = cycle.len();
    let mut bits = 0;
    for (e, edge) in sub_entities(cell, 1).iter().enumerate() {
        let a = cycle
            .iter()
            .position(|&v| v == edge[0])
            .expect("edge vertex");
        if cycle[(a + 1) % m] != edge[1] {
            bits |= 1 << e;
        }
    }
    bits
}

/// The side record of each local (cell, local facet) entry i < len of one cell block, written
/// to `sides[2 f + slot]` with f the entry's facet and slot its cell's position in the facet's
/// (tdim − 1, tdim) row: `(flat cell << 2) | (owned << 1) | d`, with d the direction bit
/// `((cycle ^ perm[c]) >> e) & 1` (no permutation word for intervals).
///   rows     [cell][k] flat facets
///   perm     one orientation word per cell (unread for intervals)
///   offsets  up_cells  the (tdim − 1, tdim) rows: local cells per facet, ascending
///   sides    [facet][2] side records
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn sides_kernel(
    rows: &[u32],
    perm: &[u32],
    offsets: &[u32],
    up_cells: &[u32],
    sides: &mut [u32],
    len: u32,
    k: u32,
    cell_offset: u32,
    num_owned: u32,
    last_facet: u32,
    items_per_unit: u32,
    #[comptime] cycle: u32,
    #[comptime] has_perm: bool,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = i as u32 / k;
            let e = i as u32 - c * k;
            let f = min(rows[i], last_facet) as usize;
            let mut word = 0u32;
            if comptime!(has_perm) {
                word = perm[c as usize];
            }
            let d = ((word ^ cycle) >> e) & 1u32;
            let owned = u32::cast_from(c < num_owned);
            let cell = cell_offset + c;
            let slot = usize::cast_from(up_cells[offsets[f] as usize] != cell);
            sides[2 * f + slot] = (cell << 2u32) | (owned << 1u32) | d;
        }
    }
}

/// The contribution of each facet j < n of one facet block to the side exchange, `2 w` words
/// at `out[2 w j]`: where the facet has two cells globally and this rank owns a cell at it,
/// that cell's `w` words of `values` (indexed by the side, `[facet][2][w]`) go to the first
/// half if this rank owns the facet (j < num_owned), else to the second; everything else is
/// 0. Values must be non-zero, so that `reverse(Max)` keeps the single contribution.
///   counts  owned cells per facet, globally
///   sides   [facet][2] side records
///   values  [facet][2][w] words per side
///   out     [facet of the block][2][w]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn contribute_kernel(
    counts: &[u32],
    sides: &[u32],
    values: &[u32],
    out: &mut [u32],
    n: u32,
    offset: u32,
    num_owned: u32,
    items_per_unit: u32,
    #[comptime] w: usize,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let j = first + it;
        if j < n as usize {
            let f = offset as usize + j;
            #[unroll]
            for k in 0..2 * w {
                out[2 * w * j + k] = 0u32;
            }
            if counts[f] == 2u32 {
                let half = usize::cast_from(j >= num_owned as usize) * w;
                for slot in 0..2usize {
                    let s = sides[2 * f + slot];
                    if s != NONE && (s & 2u32) != 0u32 {
                        #[unroll]
                        for k in 0..w {
                            out[2 * w * j + half + k] = values[(2 * f + slot) * w + k];
                        }
                    }
                }
            }
        }
    }
}

/// The other side's words of each facet j < n of one facet block after the exchange: the
/// second half where this rank owns the facet, else the first, to `remote[(offset + j) w]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn remote_kernel(
    exchanged: &[u32],
    remote: &mut [u32],
    n: u32,
    offset: u32,
    num_owned: u32,
    items_per_unit: u32,
    #[comptime] w: usize,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let j = first + it;
        if j < n as usize {
            let half = usize::cast_from(j < num_owned as usize) * w;
            #[unroll]
            for k in 0..w {
                remote[(offset as usize + j) * w + k] = exchanged[2 * w * j + half + k];
            }
        }
    }
}

/// The neighbour record of each local (cell, local facet) entry i < len of one cell block:
/// `(neighbour << 1) | parity`, or [`NONE`] where the facet has not two cells globally.
/// The neighbour is the other local cell at the facet, or, for an owned cell whose neighbour
/// is not local, the remote slot `num_cells + f`; a ghost cell's remote facets get [`NONE`]
/// (ghost cells take their owner's state). The parity is 1 where both direction bits are
/// equal; the remote one is bit 0 of `remote[f]` (bit 1 marks it present).
///   rows    [cell][k] flat facets
///   counts  owned cells per facet, globally
///   sides   [facet][2] side records
///   remote  the other side's direction word per facet
///   table   [cell][k] neighbour records
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn table_kernel(
    rows: &[u32],
    counts: &[u32],
    sides: &[u32],
    remote: &[u32],
    table: &mut [u32],
    len: u32,
    k: u32,
    cell_offset: u32,
    num_owned: u32,
    num_cells: u32,
    last_facet: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = i as u32 / k;
            let f = min(rows[i], last_facet) as usize;
            let cell = cell_offset + c;
            let s0 = sides[2 * f];
            let s1 = sides[2 * f + 1];
            let first_is_mine = (s0 >> 2u32) == cell;
            let mine = select(first_is_mine, s0, s1);
            let other = select(first_is_mine, s1, s0);
            let r = remote[f];
            let local = ((other >> 2u32) << 1u32) | u32::cast_from((mine & 1u32) == (other & 1u32));
            let far =
                ((num_cells + f as u32) << 1u32) | u32::cast_from((mine & 1u32) == (r & 1u32));
            let interior = counts[f] == 2u32;
            let has_local = other != NONE;
            let has_far = c < num_owned && (r & 2u32) != 0u32;
            let mut record = 4294967295u32; // NONE
            if interior && has_local {
                record = local;
            }
            if interior && !has_local && has_far {
                record = far;
            }
            table[i] = record;
        }
    }
}

/// The two-sided exchange over the facet blocks' index maps (module docs), with `w` words per
/// side. The scatterers are made once and reused in every round of label propagation.
pub(crate) struct SideExchange<'a> {
    client: Client,
    blocks: &'a [EntityBlock],
    /// One scatterer of 2 w words per facet block; `None` on one rank, where no facet has a
    /// remote side.
    scatterers: Option<Vec<Scatterer>>,
    num_facets: usize,
    w: usize,
}

impl<'a> SideExchange<'a> {
    /// The exchange over `blocks` (all facet blocks, `num_facets` facets) with `w` words per
    /// side. Collective on more than one rank.
    pub(crate) fn new(
        client: &Client,
        comm: &dyn CommDyn,
        blocks: &'a [EntityBlock],
        w: usize,
    ) -> Result<Self> {
        let num_facets = blocks
            .iter()
            .try_fold(0usize, |acc, b| acc.checked_add(b.index_map.num_local()))
            .ok_or(GridError::TooLarge)?;
        let scatterers = if comm.size() > 1 {
            Some(
                blocks
                    .iter()
                    .map(|b| b.index_map.scatterer(client, 2 * w))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        };
        Ok(SideExchange {
            client: client.clone(),
            blocks,
            scatterers,
            num_facets,
            w,
        })
    }

    /// The other side's `w` words per facet, `[facet][w]`, from the words of each local side
    /// (`values`, `[facet][2][w]`, non-zero where present): defined where a facet has two
    /// cells globally and the other one is owned by another rank than an owned local cell at
    /// it; 0 where nobody contributed. All zeros on one rank. Collective.
    pub(crate) fn remote(
        &self,
        counts: &DeviceBuffer<u32>,
        sides: &DeviceBuffer<u32>,
        values: &DeviceBuffer<u32>,
    ) -> Result<DeviceBuffer<u32>> {
        let w = self.w;
        let len = self.num_facets.checked_mul(w).ok_or(GridError::TooLarge)?;
        let remote = DeviceBuffer::<u32>::zeros(&self.client, len)?;
        let Some(scatterers) = &self.scatterers else {
            return Ok(remote);
        };
        let client = &self.client;
        for (block, scatterer) in self.blocks.iter().zip(scatterers) {
            let n = block.index_map.num_local();
            let words = n.checked_mul(2 * w).ok_or(GridError::TooLarge)?;
            let mut exchanged = DeviceBuffer::<u32>::uninit(client, words)?;
            if n > 0 {
                let launch = plan(client, n)?;
                debug_assert_bindings(client, 4);
                // SAFETY: for j < n the kernel reads counts and sides at f = offset + j and
                // 2 f + 1, and values at (2 f + 1) w + w − 1, all within their num_facets,
                // 2 num_facets and 2 num_facets w words (the blocks tile the flat facets);
                // it writes out[2 w j + k] < 2 w n = words, exchanged's length.
                unsafe {
                    contribute_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        counts.arg(),
                        sides.arg(),
                        values.arg(),
                        exchanged.arg(),
                        fit(n)?,
                        block.offset,
                        fit(block.index_map.num_owned())?,
                        launch.items_per_unit,
                        w,
                    );
                }
            }
            // Collective over the block's map: every rank takes part, even with no facets.
            scatterer.reverse(&mut exchanged, ReverseOp::Max)?;
            scatterer.forward(&mut exchanged)?;
            if n > 0 {
                let launch = plan(client, n)?;
                debug_assert_bindings(client, 2);
                // SAFETY: for j < n the kernel reads exchanged[2 w j + k], k < 2 w, within
                // its 2 w n words, and writes remote[(offset + j) w + k] < num_facets · w,
                // remote's length.
                unsafe {
                    remote_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        exchanged.arg(),
                        remote.arg(),
                        fit(n)?,
                        block.offset,
                        fit(block.index_map.num_owned())?,
                        launch.items_per_unit,
                        w,
                    );
                }
            }
        }
        Ok(remote)
    }
}

/// The neighbour table of rule 12 and the facet sides it was built from.
#[allow(dead_code)] // `sides` and `num_cells` are read by label propagation (M9 T8)
pub(crate) struct Neighbours {
    /// Per cell block, per cell and local facet: `(neighbour << 1) | parity`, or [`NONE`].
    /// Neighbours below `num_cells` are flat local cells; `num_cells + f` is the remote side
    /// of facet f.
    pub(crate) table: Vec<DeviceStrided<u32>>,
    /// `[facet][2]` side records `(flat cell << 2) | (owned << 1) | d`, or [`NONE`].
    pub(crate) sides: DeviceBuffer<u32>,
    /// The local cells over all blocks.
    pub(crate) num_cells: usize,
}

/// The (tdim − 1, tdim) rows of a connectivity, which must be upward.
pub(crate) fn upward(c: &Connectivity) -> &DeviceCsr {
    match c {
        Connectivity::Upward(csr) => csr,
        Connectivity::Downward(_) => unreachable!("(tdim - 1, tdim) is upward"),
    }
}

/// The key (tdim − 1, tdim).
pub(crate) fn facet_to_cell(tdim: usize) -> ConnKey {
    ConnKey {
        from: (tdim - 1) as u8,
        to: tdim as u8,
    }
}

/// The neighbour table of a manifold (every count ≤ 2): the facet sides, their exchange over
/// `exchange` (w = 1: the direction bit, bit 1 marking it present) and the records.
/// `perms[b]` is block b's permutation info (`None` for intervals), `rows` the cells' facet
/// rows, `up` the (tdim − 1, tdim) rows. Collective.
pub(crate) fn neighbours<TD: TopoDim>(
    client: &Client,
    cell_blocks: &[CellBlock<TD>],
    perms: &[Option<&DeviceBuffer<u32>>],
    rows: &[&DeviceStrided<u32>],
    up: &DeviceCsr,
    counts: &DeviceBuffer<u32>,
    exchange: &SideExchange<'_>,
) -> Result<Neighbours> {
    let num_facets = counts.len();
    let num_cells = cell_blocks
        .iter()
        .try_fold(0usize, |acc, b| acc.checked_add(b.index_map.num_local()))
        .ok_or(GridError::TooLarge)?;
    // Side records hold cell << 2, neighbour records (num_cells + f) << 1.
    if num_cells >= 1 << 30 || num_cells + num_facets >= 1 << 31 {
        return Err(GridError::TooLarge);
    }
    let last_facet = fit(num_facets)?.saturating_sub(1);
    let mut sides = DeviceBuffer::<u32>::uninit(client, 2 * num_facets)?;
    fill(&mut sides, NONE)?;
    let mut cell_offset = 0usize;
    for ((block, r), perm) in cell_blocks.iter().zip(rows).zip(perms) {
        let len = r.data.len();
        if len > 0 && num_facets > 0 {
            let cell: CellType = block.cell.into();
            let launch = plan(client, len)?;
            debug_assert_bindings(client, 5);
            // SAFETY: for i < len = rows.len() the kernel reads rows[i], perm[i / k] (a word
            // per cell, only with a permutation word), and offsets[f] and up_cells at that
            // offset for f = min(rows[i], last_facet) < num_facets: the facet's row holds the
            // entry's own cell, so it is not empty. It writes sides[2 f + slot] < 2 num_facets,
            // sides' length.
            unsafe {
                sides_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    r.data.arg(),
                    perm.unwrap_or(&r.data).arg(),
                    up.offsets.arg(),
                    up.indices.arg(),
                    sides.arg(),
                    fit(len)?,
                    r.stride.max(1),
                    fit(cell_offset)?,
                    fit(block.index_map.num_owned())?,
                    last_facet,
                    launch.items_per_unit,
                    cycle_bits(cell),
                    perm.is_some(),
                );
            }
        }
        cell_offset += block.index_map.num_local();
    }

    // The direction bit of each side, with bit 1 set where the side exists.
    let values = DeviceBuffer::<u32>::uninit(client, 2 * num_facets)?;
    if num_facets > 0 {
        let launch = plan(client, 2 * num_facets)?;
        debug_assert_bindings(client, 2);
        // SAFETY: the kernel reads sides[i] and writes values[i] for i < 2 num_facets, both
        // buffers' length.
        unsafe {
            direction_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                sides.arg(),
                values.arg(),
                fit(2 * num_facets)?,
                launch.items_per_unit,
            );
        }
    }
    let remote = exchange.remote(counts, &sides, &values)?;

    let mut table = Vec::with_capacity(cell_blocks.len());
    let mut cell_offset = 0usize;
    for (block, r) in cell_blocks.iter().zip(rows) {
        let len = r.data.len();
        let data = DeviceBuffer::<u32>::uninit(client, len)?;
        if len > 0 && num_facets > 0 {
            let launch = plan(client, len)?;
            debug_assert_bindings(client, 5);
            // SAFETY: for i < len = rows.len() = table.len() the kernel reads rows[i], and
            // counts[f], sides[2 f + 1] and remote[f] for f = min(rows[i], last_facet) <
            // num_facets, within their num_facets, 2 num_facets and num_facets words; it
            // writes table[i].
            unsafe {
                table_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    r.data.arg(),
                    counts.arg(),
                    sides.arg(),
                    remote.arg(),
                    data.arg(),
                    fit(len)?,
                    r.stride.max(1),
                    fit(cell_offset)?,
                    fit(block.index_map.num_owned())?,
                    fit(num_cells)?,
                    last_facet,
                    launch.items_per_unit,
                );
            }
        }
        table.push(DeviceStrided {
            data,
            stride: r.stride,
        });
        cell_offset += block.index_map.num_local();
    }
    Ok(Neighbours {
        table,
        sides,
        num_cells,
    })
}

/// `values[i] = 2 | (sides[i] & 1)` for a side record, 0 for [`NONE`].
#[cube(launch_unchecked)]
fn direction_kernel(sides: &[u32], values: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let s = sides[i];
            let mut v = 0u32;
            if s != NONE {
                v = 2u32 | (s & 1u32);
            }
            values[i] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_bits_of_the_reference_cells() {
        // Triangle edges (1, 2), (0, 2), (0, 1) in the cycle 0–1–2: the second runs against it.
        assert_eq!(cycle_bits(CellType::Triangle), 0b010);
        // Quadrilateral edges (0, 1), (0, 2), (1, 3), (2, 3) in the cycle 0–1–3–2.
        assert_eq!(cycle_bits(CellType::Quadrilateral), 0b1010);
        assert_eq!(cycle_bits(CellType::Interval), 0b10);
    }
}
