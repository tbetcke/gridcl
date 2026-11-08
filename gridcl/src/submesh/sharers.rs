//! The vertices of a child (ADR-0039, item 6; ADR-0040): which parent vertices the child cells
//! use, their owners, local order, global numbers and `VertexSharers`.
//!
//! A child vertex whose parent vertex no other rank holds is owned here. For the shared ones
//! (on the device: a ghost of the parent's vertex map, or an owned vertex that a reverse
//! scatter finds ghosted elsewhere) one compaction sends the parent index, whether an owned
//! child cell holds it, its origin id and the number of unshared child vertices before it to
//! the host. There, one exchange over the parent's sharing neighbourhood restricts the
//! parent's sharer rows to the ranks that hold the vertex in a child cell, and the owner is
//! the owned-child-cell sharer at position `owner_hash(&[origin id]) mod n`, as at mesh
//! creation. Owned child vertices keep the parent local order; their global numbers follow
//! from an exclusive scan, and each owner sends those of its shared vertices to the other
//! sharers, as entity creation does. Ghosts are ordered by (owner rank, child global index),
//! and their places go back to the device.
//!
//! The device steps and the numbering are shared with the child nodes (`nodes`), which decide
//! their owners differently.

use std::collections::HashMap;

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, Result, debug_assert_bindings, exclusive_scan,
};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, CommExt, IndexMap, Neighbourhood, ReverseOp, owner_hash};
use gridcl_topology::{Topology, VertexSharers, decode_origin_ids};

use super::{fit, plan};

/// A child vertex used by child cells, some owned here.
const USED: u32 = 1;
/// A child vertex used by an owned child cell.
const USED_IN_OWNED: u32 = 3;

/// `code[rows[i]] = value` for i < len, clamped to `last`, with u32 atomic stores (several rows
/// share a vertex).
#[cube(launch_unchecked)]
fn mark_kernel(
    rows: &[u32],
    code: &[Atomic<u32>],
    len: u32,
    last: u32,
    value: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            code[min(rows[i], last) as usize].store(value);
        }
    }
}

/// `x[v] = 1` for the ghost vertices num_owned ≤ v < len, else 0.
#[cube(launch_unchecked)]
fn ghost_ones_kernel(x: &mut [u32], len: u32, num_owned: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize {
            x[v] = u32::cast_from(v >= num_owned as usize);
        }
    }
}

/// For v < len: `s[v] = 1` for a used shared vertex, `u[v] = 1` for a used unshared one (with
/// `has_shared` false, every vertex is unshared and `shared` is not read).
#[cube(launch_unchecked)]
fn classify_kernel(
    code: &[u32],
    shared: &[u32],
    s: &mut [u32],
    u: &mut [u32],
    len: u32,
    items_per_unit: u32,
    #[comptime] has_shared: bool,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize {
            let used = u32::cast_from(code[v] > 0);
            let mut sh = 0u32;
            if has_shared {
                sh = u32::cast_from(shared[v] > 0);
            }
            s[v] = used * sh;
            u[v] = used * (1 - sh);
        }
    }
}

/// `out = [pos_s[len], pos_u[len]]`. Unit 0 only.
#[cube(launch_unchecked)]
fn totals_kernel(pos_s: &[u32], pos_u: &[u32], out: &mut [u32], len: u32) {
    if ABSOLUTE_POS == 0 {
        out[0] = pos_s[len as usize];
        out[1] = pos_u[len as usize];
    }
}

/// Row `pos_s[v]` of `out` for every used shared item v < len: v, its code, its two words of
/// `extra` and `pos_u[v]`, the number of used unshared items before it.
#[cube(launch_unchecked)]
fn shared_rows_kernel(
    code: &[u32],
    shared: &[u32],
    pos_s: &[u32],
    pos_u: &[u32],
    extra: &[u32],
    out: &mut [u32],
    len: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize && code[v] > 0 && shared[v] > 0 {
            let row = pos_s[v] as usize * 5;
            out[row] = v as u32;
            out[row + 1] = code[v];
            out[row + 2] = extra[2 * v];
            out[row + 3] = extra[2 * v + 1];
            out[row + 4] = pos_u[v];
        }
    }
}

/// `ghost[rows[5i]] = place[i]` for i < len: the ghost place + 1 of each used shared vertex
/// (0 if owned here).
#[cube(launch_unchecked)]
fn ghost_places_kernel(
    rows: &[u32],
    place: &[u32],
    ghost: &mut [u32],
    len: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            ghost[rows[5 * i] as usize] = place[i];
        }
    }
}

/// `owned[v] = 1` for a used vertex owned here (no ghost place), for v < len.
#[cube(launch_unchecked)]
fn owned_kernel(code: &[u32], ghost: &[u32], owned: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize {
            owned[v] = u32::cast_from(code[v] > 0 && ghost[v] == 0);
        }
    }
}

/// For every used vertex v < len: its child vertex c (`pos_o[v]` if owned here, else
/// `ghost[v]` − 1), `to_child[v] = c` and `to_parent[c] = v`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn place_vertices_kernel(
    code: &[u32],
    ghost: &[u32],
    pos_o: &[u32],
    to_child: &mut [u32],
    to_parent: &mut [u32],
    len: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < len as usize && code[v] > 0 {
            let mut c = pos_o[v];
            if ghost[v] > 0 {
                c = ghost[v] - 1;
            }
            to_child[v] = c;
            to_parent[c as usize] = v as u32;
        }
    }
}

/// The items (vertices or nodes) of a child: compaction, owners and numbering.
pub(crate) struct ChildItems {
    /// The parent local item of each child item.
    pub(crate) to_parent: DeviceBuffer<u32>,
    /// The child item of each parent local item, meaningful for items of child cells.
    pub(crate) to_child: DeviceBuffer<u32>,
    pub(crate) index_map: IndexMap,
    /// The host decisions about the used shared items.
    pub(crate) decided: Decided,
}

/// The vertices of a child.
pub(crate) struct ChildVertices {
    /// The parent local vertex of each child vertex.
    pub(crate) to_parent: DeviceBuffer<u32>,
    /// The child vertex of each parent local vertex, meaningful for vertices of child cells.
    pub(crate) to_child: DeviceBuffer<u32>,
    pub(crate) index_map: IndexMap,
    /// The child's sharer rows; `None` if the parent has none.
    pub(crate) sharers: Option<VertexSharers>,
}

/// A used shared item on the host.
pub(crate) struct Shared {
    /// Its parent local index.
    pub(crate) parent: u32,
    /// Whether an owned child cell holds it here.
    pub(crate) in_owned: bool,
    /// Its two words of the caller's `extra` array (vertices: the interleaved origin id).
    pub(crate) extra: [u32; 2],
    /// The used unshared items before it.
    pub(crate) unshared_before: u32,
}

/// The host decisions about the used shared items.
pub(crate) struct Decided {
    pub(crate) num_owned: usize,
    /// The child sharer row of each shared item: (rank, holds it in an owned child cell),
    /// ascending.
    pub(crate) rows: Vec<Vec<(i32, bool)>>,
    /// The child item of each shared item.
    pub(crate) child: Vec<usize>,
    /// The ghost place + 1 of each shared item, 0 if owned here.
    pub(crate) places: Vec<u32>,
    /// (global index, owner) of each ghost, in ghost order.
    pub(crate) ghosts: Vec<(i64, i32)>,
}

impl Decided {
    /// One rank: every used item is owned.
    fn serial(num_unshared: usize) -> Self {
        Decided {
            num_owned: num_unshared,
            rows: Vec::new(),
            child: Vec::new(),
            places: Vec::new(),
            ghosts: Vec::new(),
        }
    }
}

/// The items of the parent index map `parent_map` that the child cell rows `rows` use (in
/// parent local items, each with its owned cells first, `num_owned` of them), with two words of
/// `extra` per parent item carried to the host for the shared ones. On more than one rank,
/// `decide` gets the used shared items (parent order) and the number of used unshared ones and
/// returns the owners' decisions; owned items keep the parent local order, ghosts take the
/// decided places. Collective.
///
/// # Errors
///
/// `TooLarge`; allocation, device and communication errors; those of `decide` and
/// `IndexMap::with_ghosts`.
pub(crate) fn child_items(
    comm: &dyn CommDyn,
    parent_map: &IndexMap,
    rows: &[(&DeviceStrided<u32>, usize)],
    extra: &DeviceBuffer<u32>,
    decide: impl FnOnce(&[Shared], usize) -> Result<Decided>,
) -> Result<ChildItems> {
    let client = extra.client();
    let nv = parent_map.num_local();
    let parallel = comm.size() > 1;

    // Codes: USED, then USED_IN_OWNED over the owned cells' rows.
    let code = DeviceBuffer::<u32>::zeros(client, nv)?;
    let last = fit(nv)?.saturating_sub(1);
    for (pass, value) in [(false, USED), (true, USED_IN_OWNED)] {
        for &(r, num_owned) in rows {
            let len = if pass {
                num_owned
                    .checked_mul(r.stride as usize)
                    .ok_or(GridError::TooLarge)?
            } else {
                r.data.len()
            };
            if len > 0 && nv > 0 {
                let launch = plan(client, len)?;
                debug_assert_bindings(client, 2);
                // SAFETY: the kernel reads rows[i] for i < len ≤ rows.len() (the owned cells
                // come first) and stores to code[min(rows[i], last)] with last < nv, its
                // length.
                unsafe {
                    mark_kernel::launch_unchecked(
                        client,
                        launch.cube_count,
                        launch.cube_dim,
                        r.data.arg(),
                        code.arg(),
                        fit(len)?,
                        last,
                        value,
                        launch.items_per_unit,
                    );
                }
            }
        }
    }

    // Shared parent items: ghosts, and owned items that another rank ghosts.
    let shared = if parallel {
        let mut x = DeviceBuffer::<u32>::uninit(client, nv)?;
        if nv > 0 {
            let launch = plan(client, nv)?;
            debug_assert_bindings(client, 1);
            // SAFETY: the kernel writes x[v] for v < nv, its length.
            unsafe {
                ghost_ones_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    x.arg(),
                    fit(nv)?,
                    fit(parent_map.num_owned())?,
                    launch.items_per_unit,
                );
            }
        }
        parent_map
            .scatterer(client, 1)?
            .reverse(&mut x, ReverseOp::Add)?;
        Some(x)
    } else {
        None
    };

    let s = DeviceBuffer::<u32>::uninit(client, nv)?;
    let u = DeviceBuffer::<u32>::uninit(client, nv)?;
    if nv > 0 {
        let launch = plan(client, nv)?;
        debug_assert_bindings(client, 4);
        // SAFETY: the kernel reads code[v] (and shared[v] when given; else code again, never
        // read) and writes s[v] and u[v] for v < nv, the length of all four.
        unsafe {
            classify_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                code.arg(),
                shared.as_ref().unwrap_or(&code).arg(),
                s.arg(),
                u.arg(),
                fit(nv)?,
                launch.items_per_unit,
                shared.is_some(),
            );
        }
    }
    let (pos_s, pos_u) = (exclusive_scan(&s)?, exclusive_scan(&u)?);
    drop((s, u));
    // The count read of the compaction (CLAUDE.md, ADR-0039 item 18).
    let (num_shared, num_unshared) = if nv > 0 {
        let out = DeviceBuffer::<u32>::uninit(client, 2)?;
        let one = plan(client, 1)?;
        debug_assert_bindings(client, 3);
        // SAFETY: one unit reads pos_s[nv] and pos_u[nv], within their nv + 1 words, and
        // writes the two words of `out`.
        unsafe {
            totals_kernel::launch_unchecked(
                client,
                one.cube_count,
                one.cube_dim,
                pos_s.arg(),
                pos_u.arg(),
                out.arg(),
                fit(nv)?,
            );
        }
        let t = out.to_host()?;
        (t[0] as usize, t[1] as usize)
    } else {
        (0, 0)
    };

    // The used shared items to the host.
    let shared_rows = DeviceBuffer::<u32>::uninit(
        client,
        num_shared.checked_mul(5).ok_or(GridError::TooLarge)?,
    )?;
    let mut list = Vec::with_capacity(num_shared);
    if let Some(sh) = &shared
        && num_shared > 0
    {
        fit(shared_rows.len())?;
        let launch = plan(client, nv)?;
        debug_assert_bindings(client, 6);
        // SAFETY: items v < nv read code, shared, pos_s and pos_u at v (nv or nv + 1 words
        // each) and extra at 2v and 2v + 1 (two words per item, the caller's contract); the
        // used shared ones write row pos_s[v] < pos_s[nv] = num_shared of `out`, 5 words each.
        unsafe {
            shared_rows_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                code.arg(),
                sh.arg(),
                pos_s.arg(),
                pos_u.arg(),
                extra.arg(),
                shared_rows.arg(),
                fit(nv)?,
                launch.items_per_unit,
            );
        }
        // The child sharer rows of shared vertices, or the node-owner candidates of shared
        // nodes, to the host (CLAUDE.md, ADR-0039 item 18).
        for row in shared_rows.to_host()?.as_chunks::<5>().0 {
            list.push(Shared {
                parent: row[0],
                in_owned: row[1] == USED_IN_OWNED,
                extra: [row[2], row[3]],
                unshared_before: row[4],
            });
        }
    }
    drop((pos_s, pos_u));

    let decided = if parallel {
        decide(&list, num_unshared)?
    } else {
        Decided::serial(num_unshared)
    };
    let num_owned = decided.num_owned;
    let num_local = num_owned + decided.ghosts.len();
    fit(num_local)?;

    // The ghost places back to the device (CLAUDE.md, ADR-0040).
    let ghost = DeviceBuffer::<u32>::zeros(client, nv)?;
    if num_shared > 0 {
        let place = DeviceBuffer::from_host(client, &decided.places)?;
        let launch = plan(client, num_shared)?;
        debug_assert_bindings(client, 3);
        // SAFETY: rows i < num_shared read rows[5i] < 5 · num_shared and place[i]
        // (num_shared words), and write ghost[rows[5i]], a parent local item below nv.
        unsafe {
            ghost_places_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                shared_rows.arg(),
                place.arg(),
                ghost.arg(),
                fit(num_shared)?,
                launch.items_per_unit,
            );
        }
    }
    let owned = DeviceBuffer::<u32>::uninit(client, nv)?;
    let to_child = DeviceBuffer::<u32>::uninit(client, nv)?;
    let to_parent = DeviceBuffer::<u32>::uninit(client, num_local)?;
    if nv > 0 {
        let launch = plan(client, nv)?;
        debug_assert_bindings(client, 3);
        // SAFETY: the kernel reads code[v] and ghost[v] and writes owned[v] for v < nv, their
        // length.
        unsafe {
            owned_kernel::launch_unchecked(
                client,
                launch.cube_count.clone(),
                launch.cube_dim,
                code.arg(),
                ghost.arg(),
                owned.arg(),
                fit(nv)?,
                launch.items_per_unit,
            );
        }
        let pos_o = exclusive_scan(&owned)?;
        debug_assert_bindings(client, 5);
        // SAFETY: used items v < nv read code, ghost and pos_o at v (nv or nv + 1 words) and
        // write to_child[v] < nv and to_parent[c], where c is pos_o[v] < num_owned for owned
        // items and a ghost place in [num_owned, num_local) otherwise, below num_local, its
        // length.
        unsafe {
            place_vertices_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                code.arg(),
                ghost.arg(),
                pos_o.arg(),
                to_child.arg(),
                to_parent.arg(),
                fit(nv)?,
                launch.items_per_unit,
            );
        }
    }

    let index_map = if parallel {
        let (ghosts, owners) = decided.ghosts.iter().copied().unzip();
        IndexMap::with_ghosts(comm, num_owned, ghosts, owners)?
    } else {
        IndexMap::new(comm, num_local)?
    };
    Ok(ChildItems {
        to_parent,
        to_child,
        index_map,
        decided,
    })
}

/// The vertices of the child cells whose rows (in parent local vertices) are `rows`, each with
/// its owned cells first (`num_owned` of them): ADR-0039 item 6. Collective.
///
/// # Errors
///
/// `TooLarge`; allocation, device and communication errors; those of `IndexMap::with_ghosts`.
///
/// # Panics
///
/// If a shared child vertex has no sharer with an owned child cell, or a ghost receives no
/// number: only with parent vertex sharers that are inconsistent across ranks.
pub(crate) fn child_vertices<TD: TopoDim>(
    comm: &dyn CommDyn,
    parent: &Topology<TD>,
    rows: &[(&DeviceStrided<u32>, usize)],
) -> Result<ChildVertices> {
    let items = child_items(
        comm,
        parent.vertex_index_map(),
        rows,
        parent.vertex_origin_ids(),
        |list, num_unshared| decide(comm, parent.vertex_sharers(), list, num_unshared),
    )?;
    let me = comm.rank() as i32;
    let num_local = items.index_map.num_local();
    let decided = &items.decided;
    let sharers = parent.vertex_sharers().map(|_| {
        let mut row_of = vec![usize::MAX; num_local];
        for (i, &c) in decided.child.iter().enumerate() {
            row_of[c] = i;
        }
        let mut s = VertexSharers {
            offsets: Vec::with_capacity(num_local + 1),
            ..VertexSharers::default()
        };
        s.offsets.push(0);
        for &i in &row_of {
            if i == usize::MAX {
                s.ranks.push(me);
                s.owned_cell.push(true);
            } else {
                for &(r, o) in &decided.rows[i] {
                    s.ranks.push(r);
                    s.owned_cell.push(o);
                }
            }
            s.offsets.push(s.ranks.len());
        }
        s
    });
    Ok(ChildVertices {
        to_parent: items.to_parent,
        to_child: items.to_child,
        index_map: items.index_map,
        sharers,
    })
}

/// The parent's sharing neighbourhood: the other ranks of every row of the parent's vertex
/// sharers, ascending. Every rank that holds a vertex, node or entity of a local cell also holds
/// one of its vertices, so the neighbourhood covers the exchanges of submesh creation.
pub(crate) struct Sharing {
    neighbours: Vec<usize>,
    slot: HashMap<usize, usize>,
    nb: Neighbourhood,
}

impl Sharing {
    /// The sharing neighbourhood of `sharers` (`None`: no neighbours). Collective.
    pub(crate) fn new(comm: &dyn CommDyn, sharers: Option<&VertexSharers>) -> Result<Self> {
        let me = comm.rank() as i32;
        let mut neighbours: Vec<usize> = sharers
            .map(|s| {
                s.ranks
                    .iter()
                    .filter(|&&r| r != me)
                    .map(|&r| r as usize)
                    .collect()
            })
            .unwrap_or_default();
        neighbours.sort_unstable();
        neighbours.dedup();
        let slot = neighbours
            .iter()
            .enumerate()
            .map(|(k, &r)| (r, k))
            .collect();
        let nb = Neighbourhood::new(comm, neighbours.clone(), neighbours.clone())?;
        Ok(Sharing {
            neighbours,
            slot,
            nb,
        })
    }

    /// Empty send lists, one per neighbour.
    pub(crate) fn sends(&self) -> Vec<Vec<i64>> {
        vec![Vec::new(); self.neighbours.len()]
    }

    /// The send list of rank `r`, a neighbour.
    pub(crate) fn to<'a>(&self, sends: &'a mut [Vec<i64>], r: i32) -> &'a mut Vec<i64> {
        &mut sends[self.slot[&(r as usize)]]
    }

    /// Sends `sends` (one list per neighbour) and returns what each neighbour sent, as
    /// (rank, words), by ascending rank. Collective over the neighbourhood.
    pub(crate) fn exchange(
        &self,
        comm: &dyn CommDyn,
        sends: &[Vec<i64>],
    ) -> Result<Vec<(i32, Vec<i64>)>> {
        let counts: Vec<usize> = sends.iter().map(Vec::len).collect();
        let (received, received_counts) =
            comm.neighbour_all_to_all_v(&self.nb, &sends.concat(), &counts)?;
        let mut at = 0;
        Ok(self
            .neighbours
            .iter()
            .zip(received_counts)
            .map(|(&r, n)| {
                let words = received[at..at + n].to_vec();
                at += n;
                (r as i32, words)
            })
            .collect())
    }
}

/// The child sharer rows, owners, global numbers and ghost places of the used shared vertices
/// `list` (parent order), with `num_unshared` used unshared vertices. Two exchanges over the
/// parent's sharing neighbourhood (the ranks of `sharers`). Collective.
fn decide(
    comm: &dyn CommDyn,
    sharers: Option<&VertexSharers>,
    list: &[Shared],
    num_unshared: usize,
) -> Result<Decided> {
    let me = comm.rank() as i32;
    let parent_row = |v: u32| -> &[i32] {
        sharers.map_or(&[], |s| {
            &s.ranks[s.offsets[v as usize]..s.offsets[v as usize + 1]]
        })
    };
    let sharing = Sharing::new(comm, sharers)?;
    let ids: Vec<i64> = list
        .iter()
        .map(|s| decode_origin_ids(&s.extra)[0])
        .collect();
    let lookup: HashMap<i64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Round 1: (origin id, holds it in an owned child cell) to the parent's other sharers.
    let mut sends = sharing.sends();
    for (s, &id) in list.iter().zip(&ids) {
        for &r in parent_row(s.parent).iter().filter(|&&r| r != me) {
            sharing
                .to(&mut sends, r)
                .extend([id, i64::from(s.in_owned)]);
        }
    }
    let mut rows: Vec<Vec<(i32, bool)>> = list.iter().map(|s| vec![(me, s.in_owned)]).collect();
    for (r, words) in sharing.exchange(comm, &sends)? {
        for pair in words.as_chunks::<2>().0 {
            if let Some(&i) = lookup.get(&pair[0]) {
                rows[i].push((r, pair[1] == 1));
            }
        }
    }
    let owners: Vec<i32> = rows
        .iter_mut()
        .zip(&ids)
        .map(|(row, &id)| {
            row.sort_unstable();
            hashed_owner(id, row, "vertex")
        })
        .collect();
    number(comm, &sharing, list, &ids, owners, rows, num_unshared)
}

/// The owned-cell sharer of `row` (ascending ranks) at position `owner_hash(&[key]) mod n`, as
/// at mesh creation.
///
/// # Panics
///
/// If no sharer holds the item in an owned cell: only with inconsistent parent sharers.
pub(crate) fn hashed_owner(key: i64, row: &[(i32, bool)], what: &str) -> i32 {
    let candidates: Vec<i32> = row.iter().filter(|e| e.1).map(|e| e.0).collect();
    assert!(
        !candidates.is_empty(),
        "child {what} {key} has no sharer with an owned child cell: inconsistent sharers"
    );
    candidates[(owner_hash(&[key]) % candidates.len() as u64) as usize]
}

/// The decisions about the used shared items `list` (parent order, unique keys `keys`) with
/// their `owners` and child sharer `rows`, and `num_unshared` used unshared items: owned items
/// in parent order, global numbers by an exclusive scan, one exchange of each owner's numbers
/// to the other child sharers, ghosts by (owner rank, global index). Collective.
pub(crate) fn number(
    comm: &dyn CommDyn,
    sharing: &Sharing,
    list: &[Shared],
    keys: &[i64],
    owners: Vec<i32>,
    rows: Vec<Vec<(i32, bool)>>,
    num_unshared: usize,
) -> Result<Decided> {
    let me = comm.rank() as i32;
    // Owned items in parent order: the unshared ones before each shared one, plus the shared
    // ones owned here.
    let mut child = vec![0usize; list.len()];
    let mut mine = 0usize;
    for (i, s) in list.iter().enumerate() {
        if owners[i] == me {
            child[i] = s.unshared_before as usize + mine;
            mine += 1;
        }
    }
    let num_owned = num_unshared + mine;
    let begin = comm.exscan_sum(u64::try_from(num_owned).map_err(|_| GridError::TooLarge)?);
    let begin = i64::try_from(begin).map_err(|_| GridError::TooLarge)?;

    // (key, global number) from each owner to the other child sharers.
    let mut sends = sharing.sends();
    for (i, &key) in keys.iter().enumerate().filter(|&(i, _)| owners[i] == me) {
        let global = begin + child[i] as i64;
        for &(r, _) in rows[i].iter().filter(|e| e.0 != me) {
            sharing.to(&mut sends, r).extend([key, global]);
        }
    }
    let lookup: HashMap<i64, usize> = keys.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let mut globals = vec![-1_i64; list.len()];
    for (_, words) in sharing.exchange(comm, &sends)? {
        for pair in words.as_chunks::<2>().0 {
            // Only the child sharers receive, so the key is a used shared item here.
            let &i = lookup
                .get(&pair[0])
                .expect("a received key is a shared child item");
            globals[i] = pair[1];
        }
    }

    // Ghosts by (owner rank, global index).
    let mut order: Vec<usize> = (0..list.len()).filter(|&i| owners[i] != me).collect();
    for &i in &order {
        assert!(
            globals[i] >= 0,
            "ghost child item {} received no global number: inconsistent sharers",
            keys[i]
        );
    }
    order.sort_unstable_by_key(|&i| (owners[i], globals[i]));
    let mut places = vec![0u32; list.len()];
    let mut ghosts = Vec::with_capacity(order.len());
    for (k, &i) in order.iter().enumerate() {
        child[i] = num_owned + k;
        places[i] = fit(num_owned + k + 1)?;
        ghosts.push((globals[i], owners[i]));
    }
    Ok(Decided {
        num_owned,
        rows,
        child,
        places,
        ghosts,
    })
}
