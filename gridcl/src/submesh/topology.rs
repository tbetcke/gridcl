//! The child topology of a submesh (ADR-0039, items 1, 6 and 7): vertex compaction, child cell
//! blocks with their origin ids, `Topology::from_device_ids` and the cell and vertex parts of
//! the `EntityMap`.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, ReduceOp, Result, SortKeys, debug_assert_bindings,
    exclusive_scan, gather, radix_sort, reduce,
};
use gridcl_dims::TopoDim;
use gridcl_parallel::CommDyn;
use gridcl_topology::{CellBlock, Topology};

use super::numbering::cell_map;
use super::select::Selected;
use super::sharers::child_vertices;
use super::{fit, iota, place, plan};

/// `used[rows[i]] = 1` for i < len, with u32 atomic stores (several rows share a vertex).
#[cube(launch_unchecked)]
fn mark_kernel(rows: &[u32], used: &[Atomic<u32>], len: u32, last: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            used[min(rows[i], last) as usize].store(1u32);
        }
    }
}

/// `out[pos[v]] = v` where `used[v] = 1`, for v < len.
#[cube(launch_unchecked)]
fn compact_kernel(used: &[u32], pos: &[u32], out: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize && used[v] == 1 {
            out[pos[v] as usize] = v as u32;
        }
    }
}

/// `out[2i + w] = ids[2 · map[i] + w]` for i < len and w ∈ {0, 1}: interleaved origin ids
/// through an index map.
#[cube(launch_unchecked)]
fn gather_pairs_kernel(ids: &[u32], map: &[u32], out: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let p = map[i] as usize;
            out[2 * i] = ids[2 * p];
            out[2 * i + 1] = ids[2 * p + 1];
        }
    }
}

/// `ids[2 · map[i] + w]` for every i and w ∈ {0, 1}: the interleaved origin ids of the
/// entities `map` indexes. Every index must be below `ids.len() / 2`.
pub(crate) fn gather_pairs(
    ids: &DeviceBuffer<u32>,
    map: &DeviceBuffer<u32>,
) -> Result<DeviceBuffer<u32>> {
    let client = ids.client();
    let n = map.len();
    let out = DeviceBuffer::<u32>::uninit(client, n.checked_mul(2).ok_or(GridError::TooLarge)?)?;
    if n > 0 {
        fit(out.len())?;
        let launch = plan(client, n)?;
        debug_assert_bindings(client, 3);
        // SAFETY: the kernel reads map[i] for i < n = map.len(), and ids at 2 · map[i] + w,
        // below ids.len() by the caller's contract; it writes out[2i + w] < 2n, its length.
        unsafe {
            gather_pairs_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                ids.arg(),
                map.arg(),
                out.arg(),
                fit(n)?,
                launch.items_per_unit,
            );
        }
    }
    Ok(out)
}

/// The cell and vertex parts of a child: its topology, the `EntityMap` arrays and, for facet
/// children, the parent cell and local facet of each child cell (flat child order).
pub(crate) struct ChildTopology<CTD: TopoDim> {
    pub(crate) topology: Topology<CTD>,
    pub(crate) child_to_parent: DeviceBuffer<u32>,
    pub(crate) parent_to_child: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    pub(crate) vertex_to_parent: DeviceBuffer<u32>,
    /// The child local vertex of each parent local vertex, meaningful for parent vertices of
    /// child cells only.
    #[allow(dead_code)] // for the child nodes (M9 T5b)
    pub(crate) parent_to_child_vertex: DeviceBuffer<u32>,
    pub(crate) facet_of: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
    /// Facet children: the parent cells and local facets of each child block (empty
    /// otherwise).
    pub(crate) facet_blocks: Vec<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
    /// Facet children: per child block, whether each parent cell block holds a parent cell on
    /// this rank (empty otherwise).
    pub(crate) present: Vec<Vec<bool>>,
}

/// The parent items (vertices or nodes, `nv` of them) that the rows `rows` use, compacted in
/// parent local order: the parent item of each child item, and the exclusive scan of the used
/// flags, whose entry at a used parent item is its child item. One count read.
pub(crate) fn compact(
    client: &Client,
    nv: usize,
    rows: &[&DeviceStrided<u32>],
) -> Result<(DeviceBuffer<u32>, DeviceBuffer<u32>)> {
    let used = DeviceBuffer::<u32>::zeros(client, nv)?;
    let last = fit(nv)?.saturating_sub(1);
    for r in rows {
        let len = r.data.len();
        if len > 0 && nv > 0 {
            let launch = plan(client, len)?;
            debug_assert_bindings(client, 2);
            // SAFETY: the kernel reads rows[i] for i < len, its length, and stores to
            // used[min(rows[i], last)] with last < nv = used.len().
            unsafe {
                mark_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    r.data.arg(),
                    used.arg(),
                    fit(len)?,
                    last,
                    launch.items_per_unit,
                );
            }
        }
    }
    let pos = exclusive_scan(&used)?;
    // The count read of the compaction (CLAUDE.md, ADR-0039 item 18).
    let count = if nv > 0 {
        reduce(&used, ReduceOp::Sum)?.to_host()?[0] as usize
    } else {
        0
    };
    let to_parent = DeviceBuffer::<u32>::uninit(client, count)?;
    if count > 0 {
        let launch = plan(client, nv)?;
        debug_assert_bindings(client, 3);
        // SAFETY: the kernel reads used[v] and pos[v] for v < nv (nv and nv + 1 words) and
        // writes out[pos[v]] for used v, where pos[v] < pos[nv] = count, its length.
        unsafe {
            compact_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                used.arg(),
                pos.arg(),
                to_parent.arg(),
                fit(nv)?,
                launch.items_per_unit,
            );
        }
    }
    Ok((to_parent, pos))
}

/// The child topology of `selected` (one entry per child block, in child block order): child
/// cells inherit their parent entities' owners, child vertices are numbered and shared as
/// ADR-0039 item 6 says (`numbering`, `sharers`); `facet_children` keeps the parent cells and
/// local facets. Collective.
pub(crate) fn child_topology<TD: TopoDim, CTD: TopoDim>(
    comm: &dyn CommDyn,
    parent: &Topology<TD>,
    selected: Vec<Selected>,
    facet_children: bool,
) -> Result<ChildTopology<CTD>> {
    let client = parent.client();
    let rows: Vec<(&DeviceStrided<u32>, usize)> =
        selected.iter().map(|s| (&s.rows, s.num_owned)).collect();
    let vertices = child_vertices(comm, parent, &rows)?;
    drop(rows);
    let num_cells = selected
        .iter()
        .try_fold(0usize, |acc, s| acc.checked_add(s.parents.len()))
        .ok_or(GridError::TooLarge)?;
    fit(num_cells)?;
    let child_to_parent = DeviceBuffer::<u32>::uninit(client, num_cells)?;
    let facet_of = if facet_children {
        Some((
            DeviceBuffer::<u32>::uninit(client, num_cells)?,
            DeviceBuffer::<u32>::uninit(client, num_cells)?,
        ))
    } else {
        None
    };
    let mut blocks = Vec::with_capacity(selected.len());
    let mut facet_blocks = Vec::new();
    let mut base = 0usize;
    for s in selected {
        let n = s.parents.len();
        place(&s.parents, &child_to_parent, base, n)?;
        if let Some((c, f)) = &s.facet_of {
            facet_blocks.push((c.share(), f.share()));
        }
        if let (Some((cells, facets)), Some((c, f))) = (&facet_of, &s.facet_of) {
            place(c, cells, base, n)?;
            place(f, facets, base, n)?;
        }
        let cell = CTD::Cell::try_from(s.cell).map_err(|e| GridError::WrongCellDim {
            found: crate::builder::cell_name(e.found),
            expected_tdim: e.expected_tdim,
        })?;
        let rows = DeviceStrided {
            data: gather(&vertices.to_child, &s.rows.data)?,
            stride: s.rows.stride,
        };
        let index_map = cell_map(
            comm,
            &s.parents,
            s.num_owned,
            &s.parent_map,
            s.parent_offset,
        )?;
        blocks.push(CellBlock::new(
            cell,
            index_map,
            rows,
            s.origin_ids,
            DeviceBuffer::zeros(client, n)?,
        ));
        base += n;
    }
    let ids = gather_pairs(parent.vertex_origin_ids(), &vertices.to_parent)?;
    let mut topology = Topology::from_device_ids(client, blocks, vertices.index_map, ids)?;
    if let Some(s) = vertices.sharers {
        topology.set_vertex_sharers(s)?;
    }
    let parent_to_child = sparse_inverse(&child_to_parent)?;
    Ok(ChildTopology {
        topology,
        child_to_parent,
        parent_to_child,
        vertex_to_parent: vertices.to_parent,
        parent_to_child_vertex: vertices.to_child,
        facet_of,
        facet_blocks,
        present: Vec::new(),
    })
}

/// The parent indices of `child_to_parent` in ascending order with the child of each: a stable
/// radix sort of (parent, child) pairs.
fn sparse_inverse(
    child_to_parent: &DeviceBuffer<u32>,
) -> Result<(DeviceBuffer<u32>, DeviceBuffer<u32>)> {
    let client = child_to_parent.client();
    let n = child_to_parent.len();
    let mut keys = SortKeys::uninit(client, n, 1)?;
    place(child_to_parent, keys.words_mut(), 0, n)?;
    let (sorted, children) = radix_sort(&keys, &iota(client, n)?)?;
    let parents = DeviceBuffer::<u32>::uninit(client, n)?;
    place(sorted.words(), &parents, 0, n)?;
    Ok((parents, children))
}
