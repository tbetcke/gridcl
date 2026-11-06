//! `orient`, step 3 (docs/design.md, "Manifold orientation"; ADR-0039, item 12): label
//! propagation with parity, pointer jumping, the exchange of process-boundary states and the
//! `NonOrientable` check.
//!
//! Every local cell c holds a state of four words: its label L (the smallest cell origin id
//! it has heard of, as interleaved biased words, so that unsigned order is i64 order), its sign
//! s (bit 0: whether c must be flipped relative to the cell whose id is L), a pointer P (a slot
//! of the state array) and r (bit 1 of the sign word: whether c must be flipped relative to
//! P). The state array holds the local cells, then one remote slot per facet, which holds the
//! state of the cell on the other side of a process-boundary facet (pointer to itself, r = 0).
//!
//! A round reads the old states and writes new ones (Jacobi, no atomics): c jumps to its
//! pointer's pointer (P ← P_P, r ← r ⊕ r_P) and takes L_P if smaller (s ← s_P ⊕ r); then every
//! neighbour n across a facet of parity p whose label is strictly smaller than the best so far
//! is adopted (L ← L_n, s ← s_n ⊕ p, P ← n, r ← p), in table order. The relations r stay true
//! relations of the facet constraints, so s is c's flip relative to the cell with id L. Labels
//! only decrease, and s changes only with L, so the rounds end; pointers form a forest whose
//! roots point to themselves (cells that have not adopted, remote slots and ghost cells), and
//! jumping halves the chains, so a strip with increasing ids needs O(log n) rounds.
//!
//! After a round the remote slots take the new states of the cells across process-boundary
//! facets (the side exchange of `parity.rs`, three words per side) and ghost cells take their
//! owner's (`forward` per cell block, then pointer to itself and r = 0). The round ends with
//! `comm.max` of 2 · changed + violated: a round in which no label or sign changed on any rank
//! has converged, and any facet whose cells violate s_a ⊕ s_b = parity then makes the mesh
//! non-orientable (the two cells of a facet carry one label after convergence).

use cubecl::prelude::*;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Result, debug_assert_bindings, fill};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, CommExt, Scatterer};
use gridcl_topology::CellBlock;

use super::parity::{NONE, Neighbours, SideExchange, fit, plan};

/// Words per state record: label hi, label lo, sign | relation << 1, pointer.
pub(crate) const STATE: usize = 4;

/// Words per side in the exchange of states: label hi, label lo, 2 | sign (non-zero).
const SIDE: usize = 3;

/// The converged states of label propagation (rule 12). Not part of the stable API
/// (ADR-0038; test: gridcl-geometry's `orient`).
#[doc(hidden)]
pub struct OrientationComponents {
    /// `[slot][4]` over the local cells (flat over the blocks), then the remote slots of the
    /// facets: label hi (biased like origin ids), label lo, sign in bit 0 (bit 1: relation to
    /// the pointer), pointer. Labels are the smallest cell origin id of each component; the
    /// sign is 1 where the cell must be flipped relative to that cell.
    pub state: DeviceBuffer<u32>,
    /// The number of rounds, the last one included (it found no change).
    pub rounds: usize,
}

/// (a_hi, a_lo) < (b_hi, b_lo), unsigned.
#[cube]
fn less(a_hi: u32, a_lo: u32, b_hi: u32, b_lo: u32) -> bool {
    a_hi < b_hi || (a_hi == b_hi && a_lo < b_lo)
}

/// The initial state of each local cell c < n of one cell block: its own id as label, sign 0,
/// pointer to itself.
///   ids    [cell][2] interleaved origin ids of the block
///   state  [slot][4]
#[cube(launch_unchecked)]
fn init_kernel(ids: &[u32], state: &mut [u32], n: u32, offset: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < n as usize {
            let slot = offset as usize + c;
            state[4 * slot] = ids[2 * c];
            state[4 * slot + 1] = ids[2 * c + 1];
            state[4 * slot + 2] = 0u32;
            state[4 * slot + 3] = slot as u32;
        }
    }
}

/// One round over the local cells i < n of one cell block (module docs), from `old` to `new`.
/// Sets `status[0]` where a label or sign changes and `status[1]` where a facet constraint is
/// violated by the old states.
///   old, new  [slot][4] states
///   table     [cell][k] neighbour records `(slot << 1) | parity`, or NONE
///   status    [changed, violated]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn round_kernel(
    old: &[u32],
    new: &mut [u32],
    table: &[u32],
    status: &[Atomic<u32>],
    n: u32,
    k: u32,
    offset: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < n as usize {
            let c = 4 * (offset as usize + i);
            let hi = old[c];
            let lo = old[c + 1];
            let s = old[c + 2] & 1u32;
            let r = (old[c + 2] >> 1u32) & 1u32;
            let q = 4 * old[c + 3] as usize;
            // Jump: the pointer's pointer, and its label if smaller.
            let mut ptr = old[q + 3];
            let mut rel = r ^ ((old[q + 2] >> 1u32) & 1u32);
            let mut l_hi = hi;
            let mut l_lo = lo;
            let mut sign = s;
            if less(old[q], old[q + 1], l_hi, l_lo) {
                l_hi = old[q];
                l_lo = old[q + 1];
                sign = (old[q + 2] & 1u32) ^ r;
            }
            let mut violated = false;
            for e in 0..k as usize {
                let record = table[i * k as usize + e];
                if record != NONE {
                    let m = record >> 1u32;
                    let p = record & 1u32;
                    let j = 4 * m as usize;
                    let sn = old[j + 2] & 1u32;
                    violated = violated || (s ^ sn) != p;
                    if less(old[j], old[j + 1], l_hi, l_lo) {
                        l_hi = old[j];
                        l_lo = old[j + 1];
                        sign = sn ^ p;
                        ptr = m;
                        rel = p;
                    }
                }
            }
            new[c] = l_hi;
            new[c + 1] = l_lo;
            new[c + 2] = sign | (rel << 1u32);
            new[c + 3] = ptr;
            if l_hi != hi || l_lo != lo || sign != s {
                status[0].store(1u32);
            }
            if violated {
                status[1].store(1u32);
            }
        }
    }
}

/// The exchange words of each side i < len: `(L hi, L lo, 2 | s)` of the side's cell, 0 for
/// an empty side.
///   sides   [facet][2] side records `(cell << 2) | (owned << 1) | d`, or NONE
///   state   [slot][4]
///   values  [facet][2][3]
#[cube(launch_unchecked)]
fn values_kernel(sides: &[u32], state: &[u32], values: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let side = sides[i];
            let mut v0 = 0u32;
            let mut v1 = 0u32;
            let mut v2 = 0u32;
            if side != NONE {
                let c = 4 * (side >> 2u32) as usize;
                v0 = state[c];
                v1 = state[c + 1];
                v2 = 2u32 | (state[c + 2] & 1u32);
            }
            values[3 * i] = v0;
            values[3 * i + 1] = v1;
            values[3 * i + 2] = v2;
        }
    }
}

/// The remote slot of each facet f < len from the exchanged words (`[facet][3]`): label and
/// sign of the other side, pointer to itself; the largest label where nothing arrived.
#[cube(launch_unchecked)]
fn remote_slots_kernel(
    remote: &[u32],
    state: &mut [u32],
    len: u32,
    num_cells: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let f = first + it;
        if f < len as usize {
            let present = remote[3 * f + 2] != 0u32;
            let slot = num_cells as usize + f;
            state[4 * slot] = select(present, remote[3 * f], 4294967295u32);
            state[4 * slot + 1] = select(present, remote[3 * f + 1], 4294967295u32);
            state[4 * slot + 2] = remote[3 * f + 2] & 1u32;
            state[4 * slot + 3] = slot as u32;
        }
    }
}

/// Label and sign of each local cell c < n of one cell block, `[cell][3]` (the third word is
/// the sign only), for the forward to ghost cells.
#[cube(launch_unchecked)]
fn pack_kernel(state: &[u32], words: &mut [u32], n: u32, offset: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < n as usize {
            let s = 4 * (offset as usize + c);
            words[3 * c] = state[s];
            words[3 * c + 1] = state[s + 1];
            words[3 * c + 2] = state[s + 2] & 1u32;
        }
    }
}

/// The owner's label and sign for each ghost cell num_owned + j, j < n, of one cell block,
/// with the pointer to itself and r = 0.
#[cube(launch_unchecked)]
fn unpack_kernel(
    words: &[u32],
    state: &mut [u32],
    n: u32,
    offset: u32,
    num_owned: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let j = first + it;
        if j < n as usize {
            let c = num_owned as usize + j;
            let slot = offset as usize + c;
            state[4 * slot] = words[3 * c];
            state[4 * slot + 1] = words[3 * c + 1];
            state[4 * slot + 2] = words[3 * c + 2];
            state[4 * slot + 3] = slot as u32;
        }
    }
}

/// The remote slots and ghost cells of a state array after a round (module docs).
struct Refresh<'a> {
    client: Client,
    exchange: SideExchange<'a>,
    /// A scatterer of three words per cell block; `None` on one rank.
    ghosts: Option<Vec<Scatterer>>,
}

impl Refresh<'_> {
    /// Writes the remote slots and ghost cells of `state` from its local owned cells.
    /// Collective on more than one rank.
    fn run<TD: TopoDim>(
        &self,
        cell_blocks: &[CellBlock<TD>],
        neighbours: &Neighbours,
        counts: &DeviceBuffer<u32>,
        state: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let client = &self.client;
        let num_facets = counts.len();
        let values = DeviceBuffer::<u32>::uninit(client, 2 * SIDE * num_facets)?;
        if num_facets > 0 {
            let launch = plan(client, 2 * num_facets)?;
            debug_assert_bindings(client, 3);
            // SAFETY: for i < 2 num_facets the kernel reads sides[i] and, for a side record,
            // the four words of its cell c < num_cells (records hold local cells), within
            // state's 4 (num_cells + num_facets) words; it writes values[3 i + 2] <
            // 6 num_facets, values' length.
            unsafe {
                values_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    neighbours.sides.arg(),
                    state.arg(),
                    values.arg(),
                    fit(2 * num_facets)?,
                    launch.items_per_unit,
                );
            }
        }
        // Collective, even without facets.
        let remote = self.exchange.remote(counts, &neighbours.sides, &values)?;
        if num_facets > 0 {
            let launch = plan(client, num_facets)?;
            debug_assert_bindings(client, 2);
            // SAFETY: for f < num_facets the kernel reads remote[3 f + 2] < 3 num_facets,
            // remote's length, and writes state[4 (num_cells + f) + 3] within its
            // 4 (num_cells + num_facets) words.
            unsafe {
                remote_slots_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    remote.arg(),
                    state.arg(),
                    fit(num_facets)?,
                    fit(neighbours.num_cells)?,
                    launch.items_per_unit,
                );
            }
        }

        let Some(scatterers) = &self.ghosts else {
            return Ok(());
        };
        let mut offset = 0usize;
        for (block, scatterer) in cell_blocks.iter().zip(scatterers) {
            let n = block.index_map.num_local();
            let owned = block.index_map.num_owned();
            let mut words = DeviceBuffer::<u32>::uninit(client, SIDE * n)?;
            if owned > 0 {
                let launch = plan(client, owned)?;
                debug_assert_bindings(client, 2);
                // SAFETY: for c < owned ≤ n the kernel reads state[4 (offset + c) + 2], a
                // local cell of the block, and writes words[3 c + 2] < 3 n, words' length.
                unsafe {
                    pack_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        state.arg(),
                        words.arg(),
                        fit(owned)?,
                        fit(offset)?,
                        launch.items_per_unit,
                    );
                }
            }
            // Collective over the block's map; fills the ghost words.
            scatterer.forward(&mut words)?;
            if n > owned {
                let launch = plan(client, n - owned)?;
                debug_assert_bindings(client, 2);
                // SAFETY: for j < n − owned the kernel reads words[3 (owned + j) + 2] < 3 n,
                // the forwarded ghost words, and writes state[4 (offset + owned + j) + 3], a
                // ghost cell of the block within the local cells.
                unsafe {
                    unpack_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        words.arg(),
                        state.arg(),
                        fit(n - owned)?,
                        fit(offset)?,
                        fit(owned)?,
                        launch.items_per_unit,
                    );
                }
            }
            offset += n;
        }
        Ok(())
    }
}

/// Label propagation over the neighbour table (module docs). `exchange` is the side exchange
/// the table was built with (its width is replaced), `counts` the owned cells per facet.
/// Collective.
///
/// # Errors
///
/// `NonOrientable` (agreed); allocation, device and communication errors.
pub(crate) fn propagate<TD: TopoDim>(
    client: &Client,
    comm: &dyn CommDyn,
    cell_blocks: &[CellBlock<TD>],
    neighbours: &Neighbours,
    counts: &DeviceBuffer<u32>,
    exchange: &SideExchange<'_>,
) -> Result<OrientationComponents> {
    let num_cells = neighbours.num_cells;
    let slots = num_cells
        .checked_add(counts.len())
        .ok_or(GridError::TooLarge)?;
    let len = slots.checked_mul(STATE).ok_or(GridError::TooLarge)?;
    fit(len)?;
    let refresh = Refresh {
        client: client.clone(),
        exchange: exchange.with_width(comm, SIDE)?,
        ghosts: if comm.size() > 1 {
            Some(
                cell_blocks
                    .iter()
                    .map(|b| b.index_map.scatterer(client, SIDE))
                    .collect::<Result<Vec<_>>>()?,
            )
        } else {
            None
        },
    };

    let mut old = DeviceBuffer::<u32>::uninit(client, len)?;
    let mut offset = 0usize;
    for block in cell_blocks {
        let n = block.index_map.num_local();
        if n > 0 {
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 2);
            // SAFETY: for c < n the kernel reads ids[2 c + 1] < 2 n, the block's interleaved
            // ids, and writes state[4 (offset + c) + 3] < 4 num_cells ≤ len.
            unsafe {
                init_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    block.origin_ids.arg(),
                    old.arg(),
                    fit(n)?,
                    fit(offset)?,
                    launch.items_per_unit,
                );
            }
        }
        offset += n;
    }
    refresh.run(cell_blocks, neighbours, counts, &mut old)?;

    let mut new = DeviceBuffer::<u32>::uninit(client, len)?;
    let mut status = DeviceBuffer::<u32>::uninit(client, 2)?;
    let mut rounds = 0usize;
    loop {
        fill(&mut status, 0)?;
        let mut offset = 0usize;
        for (block, table) in cell_blocks.iter().zip(&neighbours.table) {
            let n = block.index_map.num_local();
            if n > 0 {
                round(client, &old, &new, table, &status, n, offset)?;
            }
            offset += n;
        }
        refresh.run(cell_blocks, neighbours, counts, &mut new)?;
        let local = status.to_host()?;
        let global: u32 = comm.max(2 * local[0] + local[1]);
        rounds += 1;
        std::mem::swap(&mut old, &mut new);
        if global < 2 {
            if global == 1 {
                return Err(GridError::NonOrientable);
            }
            return Ok(OrientationComponents { state: old, rounds });
        }
    }
}

/// One launch of [`round_kernel`] over the n > 0 cells of a block at flat `offset`.
fn round(
    client: &Client,
    old: &DeviceBuffer<u32>,
    new: &DeviceBuffer<u32>,
    table: &DeviceStrided<u32>,
    status: &DeviceBuffer<u32>,
    n: usize,
    offset: usize,
) -> Result<()> {
    let k = table.stride.max(1);
    let launch = plan(client, n)?;
    debug_assert_bindings(client, 4);
    // SAFETY: for i < n the kernel reads table[i k + e] < n k, the block's table, and the
    // states of its cell offset + i < num_cells, of its pointer and of each record's slot:
    // pointers and records hold slots below num_cells + num_facets (written by this module and
    // by the table kernel), so every read lies within old's len words. It writes new[4 (offset +
    // i) + 3] < 4 num_cells, and status[0] or status[1] (two words).
    unsafe {
        round_kernel::launch_unchecked(
            client,
            launch.cube_count,
            launch.cube_dim,
            old.arg(),
            new.arg(),
            table.data.arg(),
            status.arg(),
            fit(n)?,
            k,
            fit(offset)?,
            launch.items_per_unit,
        );
    }
    Ok(())
}
