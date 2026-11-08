//! The geometry nodes of a child (ADR-0039, item 6; ADR-0040): which parent nodes the child
//! cells use, their owners, local order and global numbers.
//!
//! The device steps and the numbering are the vertices' (`sharers::child_items`, `number`). A
//! corner node takes the owner of its child vertex. Any other shared node is decided by its
//! parent node owner: every rank that uses it sends (parent global node index, holds it in an
//! owned child cell) to that owner, which returns the sorted child sharer row to every user;
//! the owner is then the owned-child-cell sharer at position `owner_hash(&[parent global node
//! index]) mod n` (node origin ids are not stored on a mesh). Both exchanges run over the
//! parent's sharing neighbourhood, which holds every other holder of a local node.

use std::collections::{BTreeMap, HashMap};

use cubecl::prelude::*;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Result, debug_assert_bindings};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, IndexMap};
use gridcl_topology::{Topology, VertexSharers};

use super::sharers::{ChildItems, Decided, Shared, Sharing, child_items, hashed_owner, number};
use super::{fit, plan};

/// `corner[2 · nodes[c · k + j]] = vertices[c · nvc + j] + 1` for the corners i = c · nvc + j <
/// len, with u32 atomic stores (cells share corners, all with the same value).
#[cube(launch_unchecked)]
fn corner_kernel(
    vertices: &[u32],
    nodes: &[u32],
    corner: &[Atomic<u32>],
    len: u32,
    nvc: u32,
    k: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = i / nvc as usize;
            let j = i % nvc as usize;
            let n = nodes[c * k as usize + j] as usize;
            corner[2 * n].store(vertices[i] + 1);
        }
    }
}

/// One child cell block for the nodes: its vertex rows (child local vertices), its dofmap rows
/// (parent local nodes, the corners first in vertex order) and its number of owned cells.
pub(crate) struct NodeRows<'a> {
    pub(crate) vertices: &'a DeviceStrided<u32>,
    pub(crate) dofmap: &'a DeviceStrided<u32>,
    pub(crate) num_owned: usize,
}

/// The nodes of the child cells `blocks` over the parent nodes with index map `node_map`, for
/// the child vertices with index map `vertex_map`; `parent` gives the sharing neighbourhood.
/// Collective.
///
/// # Errors
///
/// `TooLarge`; allocation, device and communication errors; those of `IndexMap::with_ghosts`.
///
/// # Panics
///
/// If a shared child node has no sharer with an owned child cell, or a ghost receives no
/// number: only with parent sharers that are inconsistent across ranks.
pub(crate) fn child_nodes<TD: TopoDim>(
    comm: &dyn CommDyn,
    parent: &Topology<TD>,
    node_map: &IndexMap,
    vertex_map: &IndexMap,
    blocks: &[NodeRows<'_>],
) -> Result<ChildItems> {
    let client = parent.client();
    let nn = node_map.num_local();
    // Two words per parent node: the child vertex + 1 of a corner node (0 otherwise), and 0.
    let corner = DeviceBuffer::<u32>::zeros(client, nn.checked_mul(2).ok_or(GridError::TooLarge)?)?;
    if comm.size() > 1 {
        for b in blocks {
            let len = b.vertices.data.len();
            if len == 0 {
                continue;
            }
            let launch = plan(client, len)?;
            debug_assert_bindings(client, 3);
            // SAFETY: corners i < len = vertices.len() read vertices[i] and, with c = i / nvc <
            // num_cells and j < nvc ≤ k, nodes[c · k + j] < num_cells · k = nodes.len(); that is
            // a parent local node n < nn, so corner[2n] < 2 · nn, its length.
            unsafe {
                corner_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    b.vertices.data.arg(),
                    b.dofmap.data.arg(),
                    corner.arg(),
                    fit(len)?,
                    b.vertices.stride,
                    b.dofmap.stride,
                    launch.items_per_unit,
                );
            }
        }
    }
    let rows: Vec<(&DeviceStrided<u32>, usize)> =
        blocks.iter().map(|b| (b.dofmap, b.num_owned)).collect();
    child_items(comm, node_map, &rows, &corner, |list, num_unshared| {
        decide(
            comm,
            parent.vertex_sharers(),
            node_map,
            vertex_map,
            list,
            num_unshared,
        )
    })
}

/// The owners, child sharer rows, global numbers and ghost places of the used shared nodes
/// `list` (parent order), with `num_unshared` used unshared nodes. Three exchanges over the
/// parent's sharing neighbourhood: candidates to the parent node owner, rows back, numbers from
/// the child owners. Collective.
fn decide(
    comm: &dyn CommDyn,
    sharers: Option<&VertexSharers>,
    node_map: &IndexMap,
    vertex_map: &IndexMap,
    list: &[Shared],
    num_unshared: usize,
) -> Result<Decided> {
    let me = comm.rank() as i32;
    let sharing = Sharing::new(comm, sharers)?;
    let (begin, owned) = (node_map.owned_range().0, node_map.num_owned());
    let key_and_owner = |v: u32| -> (i64, i32) {
        let v = v as usize;
        if v < owned {
            (begin + v as i64, me)
        } else {
            (
                node_map.ghosts()[v - owned],
                node_map.ghost_owners()[v - owned],
            )
        }
    };

    // Round 1: (parent global node index, holds it in an owned child cell) to the parent owner.
    let mut sends = sharing.sends();
    let mut gathered: BTreeMap<i64, Vec<(i32, bool)>> = BTreeMap::new();
    let mut keys = Vec::with_capacity(list.len());
    for s in list {
        let (key, owner) = key_and_owner(s.parent);
        keys.push(key);
        if owner == me {
            gathered.entry(key).or_default().push((me, s.in_owned));
        } else {
            sharing
                .to(&mut sends, owner)
                .extend([key, i64::from(s.in_owned)]);
        }
    }
    for (r, words) in sharing.exchange(comm, &sends)? {
        for pair in words.as_chunks::<2>().0 {
            gathered.entry(pair[0]).or_default().push((r, pair[1] == 1));
        }
    }

    // Round 2: the sorted child sharer row [key, n, (rank, owned cell) × n] to its other users.
    let mut sends = sharing.sends();
    for (&key, row) in &mut gathered {
        row.sort_unstable();
        for &(r, _) in row.iter().filter(|e| e.0 != me) {
            let out = sharing.to(&mut sends, r);
            out.extend([key, row.len() as i64]);
            out.extend(row.iter().flat_map(|&(r, o)| [i64::from(r), i64::from(o)]));
        }
    }
    let mut received: HashMap<i64, Vec<(i32, bool)>> = HashMap::new();
    for (_, words) in sharing.exchange(comm, &sends)? {
        let mut at = 0;
        while at < words.len() {
            let (key, n) = (words[at], words[at + 1] as usize);
            let row = words[at + 2..at + 2 + 2 * n]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|p| (p[0] as i32, p[1] == 1))
                .collect();
            received.insert(key, row);
            at += 2 + 2 * n;
        }
    }
    let rows: Vec<Vec<(i32, bool)>> = list
        .iter()
        .zip(&keys)
        .map(|(s, key)| {
            let row = if key_and_owner(s.parent).1 == me {
                gathered.remove(key)
            } else {
                received.remove(key)
            };
            row.unwrap_or_else(|| panic!("shared child node {key} has no sharer row"))
        })
        .collect();

    // Corner nodes take their vertex's owner, the others the hash of the parent global index.
    let (vertex_owned, vertex_owners) = (vertex_map.num_owned(), vertex_map.ghost_owners());
    let owners = list
        .iter()
        .zip(&keys)
        .zip(&rows)
        .map(|((s, &key), row)| match s.extra[0] {
            0 => hashed_owner(key, row, "node"),
            v if (v as usize - 1) < vertex_owned => me,
            v => vertex_owners[v as usize - 1 - vertex_owned],
        })
        .collect();
    number(comm, &sharing, list, &keys, owners, rows, num_unshared)
}
