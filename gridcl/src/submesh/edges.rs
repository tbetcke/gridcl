//! Edge submeshes (ADR-0039, items 3, 7 and 8): the tagged edges of a volume mesh as a curve
//! mesh in 3D.
//!
//! - Child cells take the edges' canonical vertex rows (A.6) and have flags 0.
//! - Origin ids: per edge, the minimum of v = 12 · id(c) + e over the cells c containing it, e
//!   being the edge's local index in c. Each rank takes the minimum over its local cells (the
//!   (1, 3) row and the (3, 1) rows), then the ranks agree by `Min` on the biased high word,
//!   `forward`, `Min` on the low word with non-matching high words set to `u32::MAX`, and
//!   `forward`. Every cell containing an edge is owned by a rank holding the edge, so the
//!   owner sees every cell.
//! - Interior nodes come from the first local cell of the edge's (1, 3) row, through the edge's
//!   closure map in the direction of the canonical rows (`closure::edge_table`).

use cubecl::prelude::*;
use gridcl_device::{
    ConnKey, DeviceBuffer, DeviceCsr, DeviceStrided, GridError, Real, Result,
    debug_assert_bindings, fill,
};
use gridcl_dims::{CellType, D3};
use gridcl_geometry::NodeFamily;
use gridcl_parallel::{Comm, CommDyn, CommExt, IndexMap, ReverseOp};
use gridcl_topology::{Connectivity, EntityBlock, Topology};

use super::select::{self, ParentBlock, RowSource, Selected};
use super::topology::{self, ChildTopology};
use super::{ChildBlock, closure, fit, gather_rows, geometry, plan};
use crate::{Mesh, MeshTags};

/// The running minimum of v = 12 · id(c) + e over the cells c of each edge's row that lie in
/// the cell block `[cell_begin, cell_end)`, for the edges i < len: `out[2i], out[2i + 1]` hold
/// the biased high word and the low word. v fits i64 iff id lies in [L − (e ≥ 4), U − (e ≥ 8)]
/// with U = −L = 0x0AAA_AAAA_AAAA_AAAA; otherwise `bad[i]` is set to 1. `ids` holds the block's
/// interleaved cell origin ids, `down` its (3, 1) rows (ne per cell). Scalar.
#[cube(launch_unchecked)]
#[allow(
    clippy::too_many_arguments,
    clippy::manual_range_contains,
    clippy::needless_range_loop
)] // no ranges or iterators in kernels
fn edge_ids_kernel(
    offsets: &[u32],
    indices: &[u32],
    down: &[u32],
    ids: &[u32],
    out: &mut [u32],
    bad: &mut [u32],
    len: u32,
    ne: u32,
    cell_begin: u32,
    cell_end: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            for r in offsets[i] as usize..offsets[i + 1] as usize {
                let c = indices[r];
                if c >= cell_begin && c < cell_end {
                    let lc = (c - cell_begin) as usize;
                    let mut e = 0u32;
                    for j in 0..ne as usize {
                        if down[lc * ne as usize + j] == i as u32 {
                            e = j as u32;
                        }
                    }
                    let w = ids[2 * lc];
                    let lo = ids[2 * lc + 1];
                    // The bounds of the module docs on the biased high word w.
                    let lo_min = 1431655766u32 - u32::cast_from(e >= 4);
                    let lo_max = 2863311530u32 - u32::cast_from(e >= 8);
                    let above = w > 1968526677u32 || (w == 1968526677u32 && lo >= lo_min);
                    let below = w < 2326440618u32 || (w == 2326440618u32 && lo <= lo_max);
                    if !(above && below) {
                        bad[i] = 1u32;
                    }
                    // 12 · id + e = (id << 3) + (id << 2) + e in two words, mod 2^64.
                    let hi = w ^ 2147483648u32;
                    let hi8 = (hi << 3u32) | (lo >> 29u32);
                    let lo8 = lo << 3u32;
                    let hi4 = (hi << 2u32) | (lo >> 30u32);
                    let lo4 = lo << 2u32;
                    let s_lo = lo8 + lo4;
                    let s_hi = hi8 + hi4 + u32::cast_from(s_lo < lo8);
                    let v_lo = s_lo + e;
                    let v_w = (s_hi + u32::cast_from(v_lo < s_lo)) ^ 2147483648u32;
                    let o_w = out[2 * i];
                    let o_lo = out[2 * i + 1];
                    if v_w < o_w || (v_w == o_w && v_lo < o_lo) {
                        out[2 * i] = v_w;
                        out[2 * i + 1] = v_lo;
                    }
                }
            }
        }
    }
}

/// The steps of the two-word minimum across ranks over the edges i < len: stage 0 copies the
/// high words, `hi[i] = pairs[2i]`; stage 1 keeps the low words of the agreed high words,
/// `lo[i] = pairs[2i + 1]` if `pairs[2i] = hi[i]`, else `u32::MAX`; stage 2 writes both back.
#[cube(launch_unchecked)]
fn pair_kernel(
    pairs: &mut [u32],
    hi: &mut [u32],
    lo: &mut [u32],
    len: u32,
    items_per_unit: u32,
    #[comptime] stage: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            if stage == 0 {
                hi[i] = pairs[2 * i];
            } else if stage == 1 {
                let mut l = pairs[2 * i + 1];
                if pairs[2 * i] != hi[i] {
                    l = u32::MAX;
                }
                lo[i] = l;
            } else {
                pairs[2 * i] = hi[i];
                pairs[2 * i + 1] = lo[i];
            }
        }
    }
}

/// For the children i < len whose edge's first local cell c (`indices[offsets[parents[i]]]`)
/// lies in the cell block `[cell_begin, cell_end)`: `first[i] = c` and `code[i]` = the edge's
/// local index in c, found in the block's (3, 1) rows `down` (ne per cell). Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn first_cell_kernel(
    parents: &[u32],
    offsets: &[u32],
    indices: &[u32],
    down: &[u32],
    first: &mut [u32],
    code: &mut [u32],
    len: u32,
    ne: u32,
    cell_begin: u32,
    cell_end: u32,
    items_per_unit: u32,
) {
    let start = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = start + it;
        if i < len as usize {
            let p = parents[i];
            let c = indices[offsets[p as usize] as usize];
            if c >= cell_begin && c < cell_end {
                let lc = (c - cell_begin) as usize;
                let mut e = 0u32;
                for j in 0..ne as usize {
                    if down[lc * ne as usize + j] == p {
                        e = j as u32;
                    }
                }
                first[i] = c;
                code[i] = e;
            }
        }
    }
}

/// For the children i < len whose first cell c = `first[i]` lies in the cell block
/// `[cell_begin, cell_end)`, with local edge e = `code[i]`: `code[i] = d · ne + e`, where d = 1
/// if the cell's local edge starts elsewhere than the child's first vertex `rows[2i]`
/// (`table` holds the block's two vertices per local edge, `vertices` its rows of nvc), and
/// `status[present] = 1`. Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn direction_kernel(
    first: &[u32],
    code: &mut [u32],
    vertices: &[u32],
    table: &[u32],
    rows: &[u32],
    status: &[Atomic<u32>],
    len: u32,
    nvc: u32,
    ne: u32,
    cell_begin: u32,
    cell_end: u32,
    present: u32,
    items_per_unit: u32,
) {
    let start = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = start + it;
        if i < len as usize {
            let c = first[i];
            if c >= cell_begin && c < cell_end {
                let lc = (c - cell_begin) as usize;
                let e = code[i];
                let a = vertices[lc * nvc as usize + table[2 * e as usize] as usize];
                code[i] = u32::cast_from(a != rows[2 * i]) * ne + e;
                status[present as usize].store(1u32);
            }
        }
    }
}

/// `status[0] = 1` if `bad[parents[i]]` is set for a child i < len.
#[cube(launch_unchecked)]
fn overflow_kernel(
    parents: &[u32],
    bad: &[u32],
    status: &[Atomic<u32>],
    len: u32,
    items_per_unit: u32,
) {
    let start = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = start + it;
        if i < len as usize && bad[parents[i] as usize] != 0 {
            status[0].store(1u32);
        }
    }
}

/// The (3, 1) rows of `parent`, one strided array per cell block, and its (1, 3) rows.
///
/// # Errors
///
/// `ConnectivityNotCreated` without (1, 3).
fn rows_of(parent: &Topology<D3>) -> Result<(&[DeviceStrided<u32>], &DeviceCsr)> {
    let Connectivity::Upward(up) = parent.connectivity(ConnKey { from: 1, to: 3 })? else {
        unreachable!("edge -> cell is upward");
    };
    let Connectivity::Downward(down) = parent.connectivity(ConnKey { from: 3, to: 1 })? else {
        unreachable!("cell -> edge is downward");
    };
    Ok((down, up))
}

/// Per local edge of `edges` (index map `map`), the agreed minimum origin id candidate as
/// interleaved (biased high, low) words, and whether a local cell of the edge has an id whose
/// candidate overflows i64 (module docs). Collective on more than one rank.
fn edge_ids(
    comm: &dyn CommDyn,
    parent: &Topology<D3>,
    map: &IndexMap,
    down: &[DeviceStrided<u32>],
    up: &DeviceCsr,
) -> Result<(DeviceBuffer<u32>, DeviceBuffer<u32>)> {
    let client = parent.client();
    let n = map.num_local();
    let mut ids =
        DeviceBuffer::<u32>::uninit(client, n.checked_mul(2).ok_or(GridError::TooLarge)?)?;
    fill(&mut ids, u32::MAX)?;
    let bad = DeviceBuffer::<u32>::zeros(client, n)?;
    if n > 0 {
        fit(ids.len())?;
        let launch = plan(client, n)?;
        let mut cell_begin = 0usize;
        for (block, rows) in parent.cell_blocks().iter().zip(down) {
            let cell_end = cell_begin + block.index_map.num_local();
            if cell_end > cell_begin {
                debug_assert_bindings(client, 6);
                // SAFETY: edges i < n read offsets[i] and offsets[i + 1] (n + 1 words) and the
                // row's flat cells indices[r] < indices.len(); cells c in [cell_begin,
                // cell_end) read their ne (3, 1) entries below (c − cell_begin + 1) · ne ≤
                // down.len() and their two id words below 2 · num_local = ids.len(); edges write
                // out[2i], out[2i + 1] < 2n and bad[i] < n, the lengths.
                unsafe {
                    edge_ids_kernel::launch_unchecked(
                        client,
                        launch.cube_count.clone(),
                        launch.cube_dim,
                        up.offsets.arg(),
                        up.indices.arg(),
                        rows.data.arg(),
                        block.origin_ids.arg(),
                        ids.arg(),
                        bad.arg(),
                        fit(n)?,
                        rows.stride,
                        fit(cell_begin)?,
                        fit(cell_end)?,
                        launch.items_per_unit,
                    );
                }
            }
            cell_begin = cell_end;
        }
    }
    if comm.size() > 1 {
        let scatterer = map.scatterer(client, 1)?;
        let mut hi = DeviceBuffer::<u32>::uninit(client, n)?;
        let mut lo = DeviceBuffer::<u32>::uninit(client, n)?;
        let stage = |stage: u32, hi: &DeviceBuffer<u32>, lo: &DeviceBuffer<u32>| -> Result<()> {
            if n == 0 {
                return Ok(());
            }
            let launch = plan(client, n)?;
            debug_assert_bindings(client, 3);
            // SAFETY: edges i < n access pairs[2i], pairs[2i + 1] < 2n, hi[i] and lo[i] < n,
            // within the lengths.
            unsafe {
                pair_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    ids.arg(),
                    hi.arg(),
                    lo.arg(),
                    fit(n)?,
                    launch.items_per_unit,
                    stage,
                );
            }
            Ok(())
        };
        stage(0, &hi, &lo)?;
        scatterer.reverse(&mut hi, ReverseOp::Min)?;
        scatterer.forward(&mut hi)?;
        stage(1, &hi, &lo)?;
        scatterer.reverse(&mut lo, ReverseOp::Min)?;
        scatterer.forward(&mut lo)?;
        stage(2, &hi, &lo)?;
    }
    Ok((ids, bad))
}

/// The edge children of one child block before the vertex compaction, with the first cell and
/// the code d · ne + e of each child (`direction_kernel`) and the parent cell blocks that
/// hold a first cell; also whether a child's origin id overflowed.
struct EdgeChildren {
    selected: Selected,
    first: DeviceBuffer<u32>,
    code: DeviceBuffer<u32>,
    present: Vec<bool>,
    overflowed: bool,
}

/// The edge children of `chosen` (the selected edges of the edge block `edges`): canonical
/// rows, origin ids from `ids`, first cells and codes. One read of the status (the agreed
/// `TooLarge` flag with the presence flags).
fn children(
    parent: &Topology<D3>,
    edges: &EntityBlock,
    (down, up): (&[DeviceStrided<u32>], &DeviceCsr),
    (ids, bad): (&DeviceBuffer<u32>, &DeviceBuffer<u32>),
    chosen: select::Chosen,
) -> Result<EdgeChildren> {
    let client = parent.client();
    let parents = chosen.parents;
    let n = parents.len();
    let rows = gather_rows(&edges.vertices.data, &parents, 2, edges.offset as usize)?;
    let origin_ids = gather_rows(ids, &parents, 2, 0)?;
    let first = DeviceBuffer::<u32>::uninit(client, n)?;
    let code = DeviceBuffer::<u32>::uninit(client, n)?;
    let num_blocks = parent.cell_blocks().len();
    // Word 0: overflow; word 1 + b: a child has its first cell in block b.
    let status = DeviceBuffer::<u32>::zeros(client, 1 + num_blocks)?;
    if n > 0 {
        let launch = plan(client, n)?;
        // Every child's first cell and local edge, before any direction is read.
        let mut cell_begin = 0usize;
        for (block, down) in parent.cell_blocks().iter().zip(down) {
            let cell_end = cell_begin + block.index_map.num_local();
            if cell_end > cell_begin {
                debug_assert_bindings(client, 6);
                // SAFETY: children i < n read parents[i], a local edge p, offsets[p] < n_edges
                // + 1 and indices[offsets[p]] (every local edge has a local cell); cells c in
                // [cell_begin, cell_end) read their ne (3, 1) entries below down.len(); children
                // write first[i] and code[i] < n, the lengths.
                unsafe {
                    first_cell_kernel::launch_unchecked(
                        client,
                        launch.cube_count.clone(),
                        launch.cube_dim,
                        parents.arg(),
                        up.offsets.arg(),
                        up.indices.arg(),
                        down.data.arg(),
                        first.arg(),
                        code.arg(),
                        fit(n)?,
                        down.stride,
                        fit(cell_begin)?,
                        fit(cell_end)?,
                        launch.items_per_unit,
                    );
                }
            }
            cell_begin = cell_end;
        }
        let mut cell_begin = 0usize;
        for (b, (block, down)) in parent.cell_blocks().iter().zip(down).enumerate() {
            let cell_end = cell_begin + block.index_map.num_local();
            if cell_end > cell_begin {
                let cell: CellType = block.cell.into();
                let table: Vec<u32> = gridcl_refcell::sub_entities(cell, 1)
                    .iter()
                    .flat_map(|e| [e[0] as u32, e[1] as u32])
                    .collect();
                let table = DeviceBuffer::from_host(client, &table)?;
                debug_assert_bindings(client, 6);
                // SAFETY: children i < n read first[i] and code[i] (n words, all written
                // above); those with c in [cell_begin, cell_end) have e = code[i] < ne, read
                // table[2e] < 2 · ne (two words per edge), the vertex (c − cell_begin) · nvc +
                // table[2e] < vertices.len() (table entries < nvc) and rows[2i] < 2n, and write
                // code[i] and status[1 + b] < 1 + num_blocks.
                unsafe {
                    direction_kernel::launch_unchecked(
                        client,
                        launch.cube_count.clone(),
                        launch.cube_dim,
                        first.arg(),
                        code.arg(),
                        block.vertices.data.arg(),
                        table.arg(),
                        rows.arg(),
                        status.arg(),
                        fit(n)?,
                        block.vertices.stride,
                        down.stride,
                        fit(cell_begin)?,
                        fit(cell_end)?,
                        fit(1 + b)?,
                        launch.items_per_unit,
                    );
                }
            }
            cell_begin = cell_end;
        }
        debug_assert_bindings(client, 3);
        // SAFETY: children i < n read parents[i], a local edge below bad.len(), and write only
        // status[0].
        unsafe {
            overflow_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                parents.arg(),
                bad.arg(),
                status.arg(),
                fit(n)?,
                launch.items_per_unit,
            );
        }
    }
    // The agreed TooLarge flag, with the parent blocks present for the agreed element check
    // (CLAUDE.md, ADR-0039 items 8 and 18); the caller agrees both.
    let (overflowed, present) = if n > 0 {
        let s = status.to_host()?;
        (s[0] != 0, s[1..].iter().map(|&w| w != 0).collect())
    } else {
        (false, vec![false; num_blocks])
    };
    Ok(EdgeChildren {
        selected: Selected {
            cell: chosen.cell,
            parents,
            num_owned: chosen.num_owned,
            parent_map: chosen.parent_map,
            parent_offset: chosen.parent_offset,
            rows: DeviceStrided {
                data: rows,
                stride: 2,
            },
            origin_ids,
            facet_of: None,
        },
        first,
        code,
        present,
        overflowed,
    })
}

/// The child topology of the edges of `mesh` tagged by `tags`, and the child blocks' geometry
/// (the agreed element of the first cells and their nodes along each edge). Collective.
///
/// # Errors
///
/// `DimensionMismatch` for tags of another dimension than 1; `EntitiesNotCreated(1)`;
/// `ConnectivityNotCreated` without (1, 3); agreed: `TooLarge` (origin ids beyond i64),
/// `UnsupportedDegree` or `UnsupportedNodeFamily` if the first cells of a child block differ in
/// degree or node family; allocation, device and communication errors.
pub(crate) fn edge_children<F: Real, C: Comm>(
    mesh: &Mesh<F, D3, D3, C>,
    tags: &MeshTags,
) -> Result<(ChildTopology<gridcl_dims::D1>, Vec<ChildBlock>)> {
    let comm: &dyn CommDyn = &mesh.comm;
    if tags.dim != 1 {
        return Err(GridError::DimensionMismatch {
            expected: (1, 3),
            found: (tags.dim, 3),
        });
    }
    let parent = &mesh.topology;
    let [edges] = parent.entity_blocks(1)? else {
        unreachable!("the edges of every cell type are intervals: one entity block");
    };
    let rows = rows_of(parent)?;
    let ids = edge_ids(comm, parent, &edges.index_map, rows.0, rows.1)?;
    let mask = select::tag_mask(tags, edges.index_map.num_local())?;
    let blocks = [ParentBlock {
        cell: edges.cell,
        offset: edges.offset as usize,
        index_map: &edges.index_map,
    }];
    let mut children_of = Vec::new();
    let mut overflow = false;
    for c in select::select(comm, &mask, &blocks)? {
        let e = children(parent, edges, rows, (&ids.0, &ids.1), c)?;
        overflow |= e.overflowed;
        children_of.push(e);
    }
    if comm.max(u32::from(overflow)) != 0 {
        return Err(GridError::TooLarge);
    }
    drop(ids);

    let elements: Vec<(u32, NodeFamily)> = mesh
        .geometry
        .blocks()
        .iter()
        .map(|b| (b.element.degree, b.element.family))
        .collect();
    let mut selected = Vec::with_capacity(children_of.len());
    let mut geometry_blocks = Vec::with_capacity(children_of.len());
    for e in children_of {
        let (degree, family) =
            geometry::agreed_element(comm, CellType::Interval, &elements, &e.present)?;
        let k = degree as usize + 1;
        let tables = parent
            .cell_blocks()
            .iter()
            .zip(&e.present)
            .map(|(b, &p)| {
                p.then(|| closure::edge_table(b.cell.into(), degree, family))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let sources: Vec<_> = mesh
            .geometry
            .blocks()
            .iter()
            .zip(parent.cell_blocks())
            .zip(&tables)
            .map(|((g, t), table)| RowSource {
                rows: &g.dofmap,
                num_local: t.index_map.num_local(),
                table: table.as_deref(),
            })
            .collect();
        let dofmap = select::restrict_rows(&e.first, &e.code, &sources, k)?;
        geometry_blocks.push((
            CellType::Interval,
            degree,
            family,
            DeviceStrided {
                data: dofmap,
                stride: fit(k)?,
            },
        ));
        selected.push(e.selected);
    }
    let child = topology::child_topology(comm, parent, selected, false)?;
    Ok((child, geometry_blocks))
}
