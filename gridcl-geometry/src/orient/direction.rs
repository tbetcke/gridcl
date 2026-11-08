//! `orient`, step 4 (docs/design.md, "Manifold orientation"; ADR-0039, item 12): the
//! direction of each component and the flags.
//!
//! Each owned cell c contributes (−1)^{s_c} ∫_c (x − p)·n dS, with s_c its sign from label
//! propagation and n dS = (J0 × J1) dξ for surfaces, (J0_y, −J0_x) dξ for curves (the cell's
//! geometric normal, flags ignored), by Gauss degree 3p (the largest p over blocks and ranks).
//! The device writes the parts A = ∫ x̃·n and B = ∫ n per cell, with x̃ = x − origin exact in F,
//! so that ∫ (x − p)·n = A + (origin − p)·B is finished in f64 on the host. The cells are sorted
//! by label (radix sort, stable) and summed per label in that order, with the open flag (any
//! boundary facet: a facet of an owned cell with no record in the neighbour table).
//!
//! The (label, sum, open) records of a rank go to rank `owner_hash(&[label]) mod P`, which adds
//! them in rank order and decides: `Outward` flips a component whose sum is negative (an open
//! component gives `NeedsReferencePoint`, agreed), `AwayFrom` likewise, `Towards` where it is
//! positive; a sum of exactly 0 flips nothing. The decisions travel back, and each owned cell
//! gets flag s_c ⊕ flip(label); ghost cells take their owner's by `forward`.
//!
//! For `Outward` the sum is A alone (p = origin): on a closed component ∮ n = 0, so p does not
//! matter, and the origin keeps the integrand small.

use std::collections::BTreeMap;

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, Real, ReduceOp, Result, SortKeys, debug_assert_bindings, gather,
    pad_len, radix_sort, segmented_reduce, unique_by_key_runs,
};
use gridcl_dims::{Codim1, Dim, TopoDim};
use gridcl_parallel::{CommDyn, CommExt, owner_hash};
use gridcl_topology::CellBlock;
use gridcl_topology::decode_origin_ids;

use super::OrientationRule;
use super::parity::{NONE, Neighbours, fit, plan};
use super::propagate::{OrientationComponents, STATE};
use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, FieldShape, JacShape, QuadField, QuadRule, VecShape};

/// The parts A and B of each owned cell c < num_cells of one range: F values
/// `[A, B_0, .., B_{gdim−1}]` at `out[(gdim + 1) · (slot_offset + first + c)]`, signed by the
/// cell's sign (module docs). The range starts at cell `first` of the block and at cells
/// `x_first` and `j_first` of the chunks of the points (relative to the origin) and of the
/// Jacobians, both tiled QuadFields (`QuadField`'s layout). Sums run over the points in order,
/// starting with q = 0.
///   x        points chunk, LEN = gdim
///   j        Jacobians chunk, LEN = tdim · gdim, column-major
///   weights  [q]
///   state    [slot][4] label propagation states
///   out      [slot][gdim + 1]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)] // kernel code indexes slices
fn parts_kernel<F: Float>(
    x: &[F],
    j: &[F],
    weights: &[F],
    state: &[u32],
    out: &mut [F],
    x_first: u32,
    j_first: u32,
    first: u32,
    num_cells: u32,
    slot_offset: u32,
    items_per_unit: u32,
    #[comptime] num_points: usize,
    #[comptime] tile: usize,
    #[comptime] gdim: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = begin + it;
        if c < num_cells as usize {
            let xc = x_first as usize + c;
            let jc = j_first as usize + c;
            let x_base = (xc / tile) * num_points * gdim * tile + xc % tile;
            let j_len = comptime!((gdim - 1) * gdim);
            let j_base = (jc / tile) * num_points * j_len * tile + jc % tile;
            let mut n = Array::<F>::new(comptime!(gdim));
            let mut a = F::from_int(0);
            let mut b = Array::<F>::new(comptime!(gdim));
            #[unroll]
            for g in 0..gdim {
                b[g] = F::from_int(0);
            }
            for q in 0..num_points {
                let jq = j_base + q * j_len * tile;
                if comptime!(gdim == 2) {
                    n[0] = j[jq + tile];
                    n[1] = -j[jq];
                } else {
                    let j0 = j[jq];
                    let j1 = j[jq + tile];
                    let j2 = j[jq + 2 * tile];
                    let j3 = j[jq + 3 * tile];
                    let j4 = j[jq + 4 * tile];
                    let j5 = j[jq + 5 * tile];
                    n[0] = j1 * j5 - j2 * j4;
                    n[1] = j2 * j3 - j0 * j5;
                    n[2] = j0 * j4 - j1 * j3;
                }
                let xq = x_base + q * gdim * tile;
                let w = weights[q];
                let mut xn = x[xq] * n[0];
                #[unroll]
                for g in 1..gdim {
                    xn += x[xq + g * tile] * n[g];
                }
                a += xn * w;
                #[unroll]
                for g in 0..gdim {
                    b[g] += n[g] * w;
                }
            }
            let slot = slot_offset as usize + first as usize + c;
            // (−1)^s as 1 − 2s, by arithmetic.
            let s = F::cast_from(state[STATE * slot + 2] & 1u32);
            let sign = F::from_int(1) - s - s;
            let o = (gdim + 1) * slot;
            out[o] = a * sign;
            #[unroll]
            for g in 0..gdim {
                out[o + 1 + g] = b[g] * sign;
            }
        }
    }
}

/// The sort key (label hi, label lo), the payload (the slot) and the open flag of each owned
/// cell c < n of one block: key c + key_offset, open flag at the slot (1 where a facet of the
/// cell has no record).
///   state  [slot][4]
///   table  [cell][k] neighbour records
///   words  planar sort keys, plane stride `pad`
///   slots  [key] payload
///   open   [slot]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn keys_kernel(
    state: &[u32],
    table: &[u32],
    words: &mut [u32],
    slots: &mut [u32],
    open: &mut [u32],
    n: u32,
    k: u32,
    slot_offset: u32,
    key_offset: u32,
    pad: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < n as usize {
            let slot = slot_offset as usize + c;
            let key = key_offset as usize + c;
            words[key] = state[STATE * slot];
            words[pad as usize + key] = state[STATE * slot + 1];
            slots[key] = slot as u32;
            let mut boundary = 0u32;
            for e in 0..k as usize {
                boundary = max(boundary, u32::cast_from(table[c * k as usize + e] == NONE));
            }
            open[slot] = boundary;
        }
    }
}

/// The sums of the parts per label r < num_labels, over the sorted positions
/// `offsets[r]..offsets[r + 1]` in order: `sums[(gdim + 1) r + i]` = Σ parts[(gdim + 1) ·
/// slots[j] + i].
///   parts    [slot][gdim + 1]
///   slots    [sorted position] the slot of each sorted key
///   offsets  [label + 1]
///   sums     [label][gdim + 1]
#[cube(launch_unchecked)]
fn label_sums_kernel<F: Float>(
    parts: &[F],
    slots: &[u32],
    offsets: &[u32],
    sums: &mut [F],
    num_labels: u32,
    items_per_unit: u32,
    #[comptime] width: usize,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let r = first + it;
        if r < num_labels as usize {
            let lo = offsets[r] as usize;
            let hi = offsets[r + 1] as usize;
            #[unroll]
            for i in 0..width {
                let mut acc = parts[width * slots[lo] as usize + i];
                for j in lo + 1..hi {
                    acc += parts[width * slots[j] as usize + i];
                }
                sums[width * r + i] = acc;
            }
        }
    }
}

/// The flag of each owned cell at sorted position i < n: s ⊕ flip of its label, at its slot.
///   slots  [sorted position]
///   group  [sorted position] label index
///   flips  [label]
///   state  [slot][4]
///   flags  [slot]
#[cube(launch_unchecked)]
fn flags_kernel(
    slots: &[u32],
    group: &[u32],
    flips: &[u32],
    state: &[u32],
    flags: &mut [u32],
    n: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < n as usize {
            let slot = slots[i] as usize;
            flags[slot] = (state[STATE * slot + 2] & 1u32) ^ flips[group[i] as usize];
        }
    }
}

/// `out[c] = flat[offset + c]` for c < n.
#[cube(launch_unchecked)]
fn split_kernel(flat: &[u32], out: &mut [u32], n: u32, offset: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < n as usize {
            out[c] = flat[offset as usize + c];
        }
    }
}

/// The per-label records of this rank, in label order: the label, its sums (as read, the
/// gdim + 1 parts; then the sum A + (origin − p)·B) and whether it has a boundary facet.
struct Records {
    labels: Vec<i64>,
    sums: Vec<f64>,
    open: Vec<bool>,
}

/// The owned cells sorted by label: the slot and the label index of each sorted position.
struct Sorted {
    slots: DeviceBuffer<u32>,
    group: DeviceBuffer<u32>,
}

/// The flags of every cell block of `g` under `rule` from the converged `components` and the
/// neighbour table (module docs). Collective.
///
/// # Errors
///
/// `NeedsReferencePoint` (agreed); allocation, device and communication errors.
pub(crate) fn flags<F, TD, GD, G>(
    g: &G,
    rule: OrientationRule,
    components: &OrientationComponents,
    neighbours: &Neighbours,
) -> Result<Vec<DeviceBuffer<u32>>>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let client = g.client();
    let comm = g.comm();
    let blocks = g.cell_blocks();
    let num_cells = neighbours.num_cells;
    let owned: usize = blocks.iter().map(|b| b.index_map.num_owned()).sum();
    let state = &components.state;

    // The point the sums are taken against, relative to the origin; zero for Outward.
    let origin = g.origin();
    let shift: Vec<f64> = match rule {
        OrientationRule::Outward => vec![0.0; GD::N],
        OrientationRule::AwayFrom(p) | OrientationRule::Towards(p) => {
            (0..GD::N).map(|i| origin.as_ref()[i] - p[i]).collect()
        }
    };

    // The Gauss degree, 3 · max p over the blocks of every rank: collective, so it is agreed
    // here, where ranks without owned cells take part too.
    let degree = (0..blocks.len())
        .map(|b| g.geometry_block(b).element.degree)
        .max()
        .unwrap_or(1);
    let degree = comm.max(3 * degree);

    let (records, sorted) = if owned > 0 {
        let parts = parts::<F, TD, GD, G>(g, state, num_cells, degree)?;
        let (mut records, sorted) = per_label(client, blocks, state, neighbours, &parts, owned)?;
        // A + (origin − p)·B, in f64.
        records.sums = records
            .sums
            .chunks_exact(GD::N + 1)
            .map(|s| s[0] + shift.iter().zip(&s[1..]).map(|(d, b)| d * b).sum::<f64>())
            .collect();
        (records, Some(sorted))
    } else {
        (
            Records {
                labels: Vec::new(),
                sums: Vec::new(),
                open: Vec::new(),
            },
            None,
        )
    };

    let flips = rendezvous(comm, rule, &records)?;

    let flat = DeviceBuffer::<u32>::zeros(client, num_cells)?;
    if let Some(sorted) = sorted {
        // The flip decisions back (CLAUDE.md, copy list).
        let flips = DeviceBuffer::from_host(client, &flips)?;
        let launch = plan(client, owned)?;
        debug_assert_bindings(client, 5);
        // SAFETY: for i < owned, the sorted positions, the kernel reads slots[i] and
        // group[i] (owned entries each), flips[group[i]] (group < labels, flips' length), and
        // the state of slot slots[i] < num_cells, within 4 (num_cells + facets) words; it
        // writes flags[slot] < num_cells, flat's length.
        unsafe {
            flags_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                sorted.slots.arg(),
                sorted.group.arg(),
                flips.arg(),
                state.arg(),
                flat.arg(),
                fit(owned)?,
                launch.items_per_unit,
            );
        }
    }

    let mut out = Vec::with_capacity(blocks.len());
    let mut offset = 0usize;
    for block in blocks {
        let n = block.index_map.num_local();
        let num_owned = block.index_map.num_owned();
        let mut flags = DeviceBuffer::<u32>::zeros(client, n)?;
        if num_owned > 0 {
            let launch = plan(client, num_owned)?;
            debug_assert_bindings(client, 2);
            // SAFETY: for c < num_owned ≤ n the kernel reads flat[offset + c] < num_cells and
            // writes flags[c] < n.
            unsafe {
                split_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    flat.arg(),
                    flags.arg(),
                    fit(num_owned)?,
                    fit(offset)?,
                    launch.items_per_unit,
                );
            }
        }
        if comm.size() > 1 {
            // Collective over the block's map: ghost cells take their owner's flags.
            block.index_map.scatterer(client, 1)?.forward(&mut flags)?;
        }
        out.push(flags);
        offset += n;
    }
    Ok(out)
}

/// The signed parts `[slot][gdim + 1]` of every owned cell (ghost slots unset), with the
/// Gauss rule of `degree`. Not collective.
fn parts<F, TD, GD, G>(
    g: &G,
    state: &DeviceBuffer<u32>,
    num_cells: usize,
    degree: u32,
) -> Result<DeviceBuffer<F>>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let client = g.client();
    let blocks = g.cell_blocks();
    let width = GD::N + 1;
    let rule = QuadRule::<F, TD>::for_grid(g, degree)?;
    let len = num_cells.checked_mul(width).ok_or(GridError::TooLarge)?;
    let out = DeviceBuffer::<F>::uninit(client, len)?;
    // Points relative to the origin: exact in F.
    let zero = DeviceBuffer::<F>::zeros(client, GD::N)?;
    let mut offset = 0usize;
    for (b, block) in blocks.iter().enumerate() {
        let n = block.index_map.num_local();
        let owned = block.index_map.num_owned();
        if owned > 0 {
            let pts = rule.get(block.cell)?;
            let x: QuadField<F, VecShape<GD>> =
                launch_factor(g, b, pts, Factor::Points, Some(&zero))?;
            let j: QuadField<F, JacShape<TD, GD>> =
                launch_factor(g, b, pts, Factor::Jacobian, None)?;
            let tile = x.tile();
            for r in ranges(&x, &j, owned) {
                let launch = plan(client, r.cells)?;
                debug_assert_bindings(client, 5);
                // SAFETY: the range's cells c < r.cells lie in one chunk of x and one of j
                // (`ranges`), at cells x_first + c and j_first + c below each chunk's
                // num_cells, whose tiled values (num_points · LEN per cell) lie within the
                // chunk; weights are read below num_points, the state of slots below
                // num_cells, and out is written at (gdim + 1) (offset + first + c) + gdim <
                // (gdim + 1) num_cells, its length.
                unsafe {
                    parts_kernel::launch_unchecked::<F>(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        r.x.arg(),
                        r.j.arg(),
                        pts.weights_device().arg(),
                        state.arg(),
                        out.arg(),
                        fit(r.x_first)?,
                        fit(r.j_first)?,
                        fit(r.first)?,
                        fit(r.cells)?,
                        fit(offset)?,
                        launch.items_per_unit,
                        pts.len(),
                        tile,
                        GD::N,
                    );
                }
            }
        }
        offset += n;
    }
    Ok(out)
}

/// A range of cells that lies in one chunk of each of two fields.
struct Range<'a, F: Real> {
    x: &'a DeviceBuffer<F>,
    x_first: usize,
    j: &'a DeviceBuffer<F>,
    j_first: usize,
    first: usize,
    cells: usize,
}

/// The ranges of the cells below `limit` that each lie in one chunk of `x` and one of `j`, in
/// cell order.
fn ranges<'a, F: Real, S: FieldShape, T: FieldShape>(
    x: &'a QuadField<F, S>,
    j: &'a QuadField<F, T>,
    limit: usize,
) -> Vec<Range<'a, F>> {
    let mut out = Vec::new();
    let (xs, js) = (x.chunks(), j.chunks());
    let (mut a, mut b) = (0, 0);
    while a < xs.len() && b < js.len() {
        let (xc, jc) = (&xs[a], &js[b]);
        let start = xc.first_cell.max(jc.first_cell);
        let x_end = xc.first_cell + xc.num_cells;
        let j_end = jc.first_cell + jc.num_cells;
        let end = x_end.min(j_end).min(limit);
        if start < end {
            out.push(Range {
                x: &xc.data,
                x_first: start - xc.first_cell,
                j: &jc.data,
                j_first: start - jc.first_cell,
                first: start,
                cells: end - start,
            });
        }
        if x_end <= j_end {
            a += 1;
        }
        if j_end <= x_end {
            b += 1;
        }
    }
    out
}

/// The owned cells sorted by label, the parts summed per label and the open flag of each
/// label, read back (the per-component records of the copy list).
fn per_label<F: Real, TD: TopoDim>(
    client: &Client,
    blocks: &[CellBlock<TD>],
    state: &DeviceBuffer<u32>,
    neighbours: &Neighbours,
    parts: &DeviceBuffer<F>,
    owned: usize,
) -> Result<(Records, Sorted)> {
    let width = parts.len() / neighbours.num_cells.max(1);
    let mut keys = SortKeys::uninit(client, owned, 2)?;
    let slots = DeviceBuffer::<u32>::uninit(client, owned)?;
    let open = DeviceBuffer::<u32>::uninit(client, neighbours.num_cells)?;
    let pad = pad_len(owned)?;
    let (mut slot_offset, mut key_offset) = (0usize, 0usize);
    for (block, table) in blocks.iter().zip(&neighbours.table) {
        let n = block.index_map.num_owned();
        if n > 0 {
            let k = table.stride.max(1);
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 5);
            // SAFETY: for c < n, the block's owned cells, the kernel reads the state of slot
            // slot_offset + c < num_cells and table[c k + e] < n k (the block's table has
            // k words per local cell); it writes key key_offset + c < owned in both planes
            // (pad ≥ owned), slots at the same key, and open at the slot < num_cells.
            unsafe {
                keys_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    state.arg(),
                    table.data.arg(),
                    keys.words_mut().arg(),
                    slots.arg(),
                    open.arg(),
                    fit(n)?,
                    k,
                    fit(slot_offset)?,
                    fit(key_offset)?,
                    fit(pad)?,
                    launch.items_per_unit,
                );
            }
        }
        slot_offset += block.index_map.num_local();
        key_offset += n;
    }
    let (sorted, slots) = radix_sort(&keys, &slots)?;
    drop(keys);
    // The count read of unique-by-key (CLAUDE.md).
    let (unique, offsets) = unique_by_key_runs(&sorted, &slots)?;
    let num_labels = unique.first.len();
    let sums = DeviceBuffer::<F>::uninit(client, num_labels * width)?;
    let launch = plan(client, num_labels)?;
    debug_assert_bindings(client, 4);
    // SAFETY: for r < num_labels the kernel reads offsets[r] and offsets[r + 1] (num_labels + 1
    // entries), slots at sorted positions below offsets[num_labels] = owned, the parts of their
    // slots < num_cells (width per slot), and writes sums below width · num_labels.
    unsafe {
        label_sums_kernel::launch_unchecked::<F>(
            client,
            launch.cube_count,
            launch.cube_dim,
            parts.arg(),
            slots.arg(),
            offsets.arg(),
            sums.arg(),
            fit(num_labels)?,
            launch.items_per_unit,
            width,
        );
    }
    let open = segmented_reduce(&gather(&open, &slots)?, &offsets, ReduceOp::Max)?;
    // The per-component (label, sum, open) records (CLAUDE.md, copy list).
    let records = Records {
        labels: decode_origin_ids(&unique.keys.to_host()?),
        sums: sums.to_host()?.into_iter().map(Real::widen).collect(),
        open: open.to_host()?.into_iter().map(|o| o != 0).collect(),
    };
    Ok((
        records,
        Sorted {
            slots,
            group: unique.group,
        },
    ))
}

/// The flip of each of this rank's labels: the records go to rank `owner_hash(&[label]) mod
/// P`, which adds the sums in rank order and decides; the decisions come back. Collective.
///
/// # Errors
///
/// `NeedsReferencePoint` for `Outward` on an open component (agreed); communication errors.
fn rendezvous(comm: &dyn CommDyn, rule: OrientationRule, records: &Records) -> Result<Vec<u32>> {
    let size = comm.size();
    let dest = |label: i64| (owner_hash(&[label]) % size as u64) as usize;
    // Records grouped by destination, in label order within each.
    let mut order: Vec<usize> = (0..records.labels.len()).collect();
    order.sort_by_key(|&i| dest(records.labels[i]));
    let mut counts = vec![0usize; size];
    let mut send = Vec::with_capacity(3 * order.len());
    for &i in &order {
        counts[dest(records.labels[i])] += 3;
        send.extend([
            records.labels[i],
            records.sums[i].to_bits() as i64,
            i64::from(records.open[i]),
        ]);
    }
    let (recv, recv_counts) = comm.all_to_all_v(&send, &counts)?;

    // Sources arrive in rank order, so the sums are added in rank order.
    let mut total: BTreeMap<i64, (f64, bool)> = BTreeMap::new();
    for r in recv.as_chunks::<3>().0 {
        let e = total.entry(r[0]).or_insert((0.0, false));
        e.0 += f64::from_bits(r[1] as u64);
        e.1 |= r[2] != 0;
    }
    let needs_point = matches!(rule, OrientationRule::Outward) && total.values().any(|e| e.1);
    if comm.max(u32::from(needs_point)) != 0 {
        return Err(GridError::NeedsReferencePoint);
    }
    let flip = |sum: f64| match rule {
        OrientationRule::Outward | OrientationRule::AwayFrom(_) => sum < 0.0,
        OrientationRule::Towards(_) => sum > 0.0,
    };
    let reply: Vec<u32> = recv
        .as_chunks::<3>()
        .0
        .iter()
        .map(|r| u32::from(flip(total[&r[0]].0)))
        .collect();
    let reply_counts: Vec<usize> = recv_counts.iter().map(|c| c / 3).collect();
    let (back, _) = comm.all_to_all_v(&reply, &reply_counts)?;

    let mut flips = vec![0u32; records.labels.len()];
    for (&i, f) in order.iter().zip(back) {
        flips[i] = f;
    }
    Ok(flips)
}
