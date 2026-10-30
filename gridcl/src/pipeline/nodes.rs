//! Stage 6: the coordinates of the nodes of every local cell, from a rendezvous directory, and
//! the agreed bounding box (docs/design.md, "Parallel creation pipeline"; ADR-0026, item 7).

use std::collections::HashMap;

use gridcl_device::{GridError, Result};
use gridcl_parallel::{CommDyn, CommExt};

use super::ghosts::LocalCells;
use super::input::NodeChunk;
use super::rendezvous::{Outbox, rendezvous_rank};
use super::{BlockSpec, agree};

/// The nodes of a rank's local cells (owned and ghost), by origin id, and the global bounding
/// box of the nodes of all cells.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Nodes {
    pub(crate) gdim: usize,
    /// The origin ids, ascending.
    pub(crate) ids: Vec<i64>,
    /// `[node][gdim]`, in the order of `ids`.
    pub(crate) coords: Vec<f64>,
    /// The lower corner of the bounding box; `+∞` on every axis for a mesh without cells.
    pub(crate) lo: Vec<f64>,
    /// The upper corner of the bounding box; `−∞` on every axis for a mesh without cells.
    pub(crate) hi: Vec<f64>,
}

impl Nodes {
    /// The coordinates of node `id`, which must be a node of a local cell.
    pub(crate) fn coords_of(&self, id: i64) -> &[f64] {
        let i = self.ids.binary_search(&id).expect("a node of a local cell");
        &self.coords[i * self.gdim..(i + 1) * self.gdim]
    }

    /// The global origin: the centre of the bounding box, or 0 for a mesh without cells. Equal
    /// on every rank, and bitwise equal to the centre of the box of the nodes in use on one
    /// rank.
    pub(crate) fn origin(&self) -> Vec<f64> {
        self.lo
            .iter()
            .zip(&self.hi)
            .map(|(&lo, &hi)| if lo > hi { 0.0 } else { 0.5 * (lo + hi) })
            .collect()
    }
}

/// Wire tags of the directory exchange.
const POST: i64 = 0;
const REQUEST: i64 = 1;

/// Role bits of a node request: the node is a vertex, or a non-vertex node, of an owned cell.
const VERTEX: i64 = 1;
const NON_VERTEX: i64 = 2;

/// Stage 6: the coordinates of every node of `local`'s cells. Collective.
///
/// Every rank posts its node chunk (origin id, global input position, coordinates) to the
/// rendezvous rank `owner_hash(&[id]) mod P`, and requests there the nodes of its local cells,
/// with the roles the nodes have in its owned cells. The directory checks the ids and roles and
/// answers in request order. The bounding box is agreed by `CommExt::min` and `CommExt::max`
/// over the nodes of the local cells, so it covers exactly the nodes that some cell uses.
///
/// # Errors
///
/// Agreed over the ranks, in this order:
/// - `DuplicateOriginId` for a node id posted twice (the first repeat on its directory rank);
/// - `IndexOutOfRange` for a cell node id that no rank posted: `index` is the id (`usize::MAX`
///   if negative), `len` the global node count;
/// - `NodeRoleMismatch` for a node that is a vertex of one cell and a non-vertex node of
///   another: the node's global input position (an exscan of the chunk sizes plus its index),
///   the lowest on its directory rank.
///
/// Errors of the exchanges.
pub(crate) fn fetch_nodes(
    comm: &dyn CommDyn,
    blocks: &[BlockSpec],
    gdim: usize,
    chunk: &NodeChunk,
    local: &LocalCells,
) -> Result<Nodes> {
    let p = comm.size();
    let num_chunk = chunk.ids.len();
    let first = comm.exscan_sum(num_chunk as u64) as i64;
    let total = comm.sum(num_chunk as u64);

    // The node ids of the local cells, ascending, with their roles in owned cells.
    let mut roles: HashMap<i64, i64> = HashMap::new();
    let cells = &local.cells;
    for i in 0..cells.len() {
        let nv = blocks[cells.block[i] as usize].num_vertices();
        for (j, &id) in cells.nodes_of(i).iter().enumerate() {
            let role = roles.entry(id).or_insert(0);
            if i < local.num_owned {
                *role |= if j < nv { VERTEX } else { NON_VERTEX };
            }
        }
    }
    let mut wanted: Vec<i64> = roles.keys().copied().collect();
    wanted.sort_unstable();

    let mut out = Outbox::new(p);
    for (i, &id) in chunk.ids.iter().enumerate() {
        let bits: Vec<i64> = chunk.coords[i * gdim..(i + 1) * gdim]
            .iter()
            .map(|c| c.to_bits() as i64)
            .collect();
        out.push(
            rendezvous_rank(&[id], p),
            &[&[POST, id, first + i as i64], &bits],
        );
    }
    for &id in &wanted {
        out.push(rendezvous_rank(&[id], p), &[&[REQUEST, id, roles[&id]]]);
    }
    let inbox = out.exchange(comm)?;

    // The directory: id → (global input position, coordinate words).
    let mut directory: HashMap<i64, (i64, &[i64])> = HashMap::new();
    let mut duplicate = None;
    for (_, r) in inbox.records().filter(|(_, r)| r[0] == POST) {
        if directory.insert(r[1], (r[2], &r[3..])).is_some() && duplicate.is_none() {
            duplicate = Some(r[1]);
        }
    }
    agree(
        comm,
        duplicate.map_or(Ok(()), |id| Err(GridError::DuplicateOriginId(id))),
    )?;

    let mut unknown = None;
    let mut node_roles: HashMap<i64, i64> = HashMap::new();
    let mut replies = Outbox::new(p);
    for (source, r) in inbox.records().filter(|(_, r)| r[0] == REQUEST) {
        let id = r[1];
        match directory.get(&id) {
            Some(&(_, words)) => {
                *node_roles.entry(id).or_insert(0) |= r[2];
                replies.push(source, &[&[id], words]);
            }
            None => {
                unknown.get_or_insert(id);
            }
        }
    }
    let mismatch = node_roles
        .iter()
        .filter(|&(_, &role)| role == VERTEX | NON_VERTEX)
        .map(|(id, _)| directory[id].0)
        .min();
    let local_error = match (unknown, mismatch) {
        (Some(id), _) => Err(GridError::IndexOutOfRange {
            index: usize::try_from(id).unwrap_or(usize::MAX),
            len: total as usize,
        }),
        (None, Some(position)) => Err(GridError::NodeRoleMismatch(position as usize)),
        (None, None) => Ok(()),
    };
    agree(comm, local_error)?;

    let inbox = replies.exchange(comm)?;
    let mut fetched: Vec<(i64, &[i64])> = inbox.records().map(|(_, r)| (r[0], &r[1..])).collect();
    fetched.sort_unstable_by_key(|&(id, _)| id);
    let ids: Vec<i64> = fetched.iter().map(|&(id, _)| id).collect();
    debug_assert_eq!(ids, wanted);
    let coords: Vec<f64> = fetched
        .iter()
        .flat_map(|&(_, words)| words.iter().map(|&w| f64::from_bits(w as u64)))
        .collect();

    let mut lo = vec![f64::INFINITY; gdim];
    let mut hi = vec![f64::NEG_INFINITY; gdim];
    for x in coords.chunks_exact(gdim) {
        for g in 0..gdim {
            lo[g] = lo[g].min(x[g]);
            hi[g] = hi[g].max(x[g]);
        }
    }
    let lo = lo.into_iter().map(|x| comm.min(x)).collect();
    let hi = hi.into_iter().map(|x| comm.max(x)).collect();
    Ok(Nodes {
        gdim,
        ids,
        coords,
        lo,
        hi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GhostMode;
    use crate::pipeline::input::NodeRefs;
    use crate::pipeline::testing::{OwnedInput, local_cells, meshes};
    use gridcl_testing::{HostChunk, ThreadComm};

    /// Every rank gets the coordinates of exactly the nodes of its local cells, and every rank
    /// agrees on the bounding box of the nodes in use, which equals the host computation, for
    /// even chunks and everything on rank 0, on 1–4 ranks.
    #[test]
    fn nodes_and_origin_equal_the_host_mesh() {
        for (name, h) in meshes() {
            let gd = h.gdim;
            let coords_of = |id: i64| {
                let n = (0..h.num_nodes()).find(|&n| h.node_id(n) == id).unwrap();
                h.nodes[n * gd..(n + 1) * gd].to_vec()
            };
            let used: Vec<usize> = h
                .blocks
                .iter()
                .flat_map(|b| &b.cells)
                .map(|&n| n as usize)
                .collect();
            let want_lo: Vec<f64> = (0..gd)
                .map(|g| {
                    used.iter()
                        .map(|&n| h.nodes[n * gd + g])
                        .fold(f64::INFINITY, f64::min)
                })
                .collect();
            let want_hi: Vec<f64> = (0..gd)
                .map(|g| {
                    used.iter()
                        .map(|&n| h.nodes[n * gd + g])
                        .fold(f64::NEG_INFINITY, f64::max)
                })
                .collect();
            for size in 1..=4 {
                for on_zero in [false, true] {
                    let out = ThreadComm::run(size, |comm| {
                        let r = comm.rank();
                        let chunk = match (on_zero, r) {
                            (false, _) => h.chunk(r, size),
                            (true, 0) => h.chunk(0, 1),
                            (true, _) => HostChunk::empty(gd),
                        };
                        let input = OwnedInput::of_chunk(&chunk);
                        let (read, local) =
                            local_cells(&comm, &input.input(), GhostMode::SharedFacet);
                        let nodes =
                            fetch_nodes(&comm, &read.blocks, gd, &read.nodes, &local).unwrap();
                        (local, nodes)
                    });
                    for (r, (local, nodes)) in out.iter().enumerate() {
                        let context = format!("{name}, rank {r} of {size}, on zero {on_zero}");
                        let mut want: Vec<i64> = local.cells.nodes.clone();
                        want.sort_unstable();
                        want.dedup();
                        assert_eq!(nodes.ids, want, "{context}");
                        for &id in &nodes.ids {
                            assert_eq!(nodes.coords_of(id), coords_of(id), "{context}");
                        }
                        assert_eq!(nodes.lo, want_lo, "{context}");
                        assert_eq!(nodes.hi, want_hi, "{context}");
                        let origin: Vec<f64> =
                            (0..gd).map(|g| 0.5 * (want_lo[g] + want_hi[g])).collect();
                        assert_eq!(nodes.origin(), origin, "{context}");
                    }
                }
            }
        }
    }

    /// A mesh without cells has the origin 0 on every rank.
    #[test]
    fn origin_without_cells_is_zero() {
        let out = ThreadComm::run(3, |comm| {
            let input = OwnedInput::of_chunk(&HostChunk::empty(2));
            let (read, local) = local_cells(&comm, &input.input(), GhostMode::None);
            fetch_nodes(&comm, &read.blocks, 2, &read.nodes, &local).unwrap()
        });
        for nodes in out {
            assert!(nodes.ids.is_empty());
            assert_eq!(nodes.origin(), [0.0, 0.0]);
        }
    }

    /// Run stages 1–6 on one rank of `size`, rank r holding nodes 2r and 2r + 1 (ids
    /// `node_ids(r)`) and the cells `cells(r)` of `block`.
    fn run_intervals(
        size: usize,
        node_ids: impl Fn(i64) -> Vec<i64> + Sync,
        block: impl Fn(i64) -> (BlockSpec, NodeRefs) + Sync,
    ) -> Vec<Result<Nodes>> {
        ThreadComm::run(size, |comm| {
            let r = comm.rank() as i64;
            let (spec, nodes) = block(r);
            let input = OwnedInput {
                tdim: 1,
                gdim: 1,
                coords: vec![2.0 * r as f64, 2.0 * r as f64 + 1.0],
                node_ids: Some(node_ids(r)),
                blocks: vec![(spec, nodes, None)],
                tags: vec![],
            };
            let (read, local) = local_cells(&comm, &input.input(), GhostMode::None);
            fetch_nodes(&comm, &read.blocks, 1, &read.nodes, &local)
        })
    }

    /// The errors are agreed: one rank reports its error, the others `RemoteFailure`.
    fn assert_agreed(out: &[Result<Nodes>], is_error: impl Fn(&GridError) -> bool) {
        let failing: Vec<usize> = (0..out.len())
            .filter(|&r| out[r].as_ref().is_err_and(&is_error))
            .collect();
        assert_eq!(failing.len(), 1, "{out:?}");
        for (r, res) in out.iter().enumerate().filter(|&(r, _)| r != failing[0]) {
            assert!(
                matches!(res, Err(GridError::RemoteFailure { rank }) if *rank == failing[0]),
                "rank {r}: {res:?}"
            );
        }
    }

    #[test]
    fn duplicate_unknown_and_mismatched_nodes_are_agreed_errors() {
        let p1 = || OwnedInput::interval_block(NodeRefs::Ids(vec![]), None).0;
        let intervals = |rows: Vec<i64>| (p1(), NodeRefs::Ids(rows));
        // Node id 3 on ranks 0 and 2, and an unknown id at the same time: the duplicate wins.
        let out = run_intervals(
            3,
            |r| {
                if r == 2 {
                    vec![3, 5]
                } else {
                    vec![2 * r, 2 * r + 1]
                }
            },
            |r| intervals(if r == 1 { vec![2, 99] } else { vec![] }),
        );
        assert_agreed(&out, |e| matches!(e, GridError::DuplicateOriginId(3)));

        // Rank 1 references node 99, which no rank posted; 6 nodes in all.
        let out = run_intervals(
            3,
            |r| vec![2 * r, 2 * r + 1],
            |r| {
                intervals(if r == 1 {
                    vec![2, 99]
                } else {
                    vec![2 * r, 2 * r + 1]
                })
            },
        );
        assert_agreed(&out, |e| {
            matches!(e, GridError::IndexOutOfRange { index: 99, len: 6 })
        });

        // P2 intervals: consistent roles, then node 1 (global position 1) as a vertex of rank
        // 0's interval and the midpoint of rank 1's.
        let p2 = BlockSpec { degree: 2, ..p1() };
        let out = run_intervals(
            2,
            |r| vec![2 * r, 2 * r + 1],
            |r| {
                if r == 0 {
                    (p2, NodeRefs::Ids(vec![0, 1, 3]))
                } else {
                    (p2, NodeRefs::Ids(vec![1, 2, 3]))
                }
            },
        );
        assert!(out.iter().all(|r| r.is_ok()), "{out:?}");
        let out = run_intervals(
            2,
            |r| vec![2 * r, 2 * r + 1],
            |r| {
                if r == 0 {
                    (p2, NodeRefs::Ids(vec![0, 1, 3]))
                } else {
                    (p2, NodeRefs::Ids(vec![0, 2, 1]))
                }
            },
        );
        assert_agreed(&out, |e| matches!(e, GridError::NodeRoleMismatch(1)));
    }
}
