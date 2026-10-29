//! Stage 8: sharer sets, owners and numbering of cells, vertices and nodes (docs/design.md,
//! "Parallel creation pipeline" and "Ownership of vertices and entities"; ADR-0026, items 8, 9
//! and 16).

use std::collections::{BTreeMap, HashMap, HashSet};

use gridcl_device::{GridError, Result};
use gridcl_parallel::{CommDyn, CommExt, owner_hash};
use gridcl_topology::VertexSharers;

use super::ghosts::LocalCells;
use super::nodes::Nodes;
use super::rendezvous::{Outbox, rendezvous_rank};
use super::{BlockSpec, Cells, agree};

/// The inputs of `IndexMap::with_ghosts` for `[owned | ghost]` local indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct MapInput {
    pub(crate) num_owned: usize,
    /// The global index of every ghost, in local order.
    pub(crate) ghosts: Vec<i64>,
    /// The owner rank of every ghost.
    pub(crate) ghost_owners: Vec<i32>,
}

/// One cell block after stage 8.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NumberedBlock {
    /// The block's cells in local order: the owned cells in stage-7 order, then the ghost
    /// cells by (owner rank, origin id).
    pub(crate) cells: Cells,
    pub(crate) map: MapInput,
    /// Local vertex indices, `[cell][vertices per cell]`.
    pub(crate) vertices: Vec<u32>,
    /// Local node indices, `[cell][nodes per cell]` in A.5 order.
    pub(crate) dofmap: Vec<u32>,
}

/// The output of stage 8.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Numbered {
    /// One entry per agreed block.
    pub(crate) blocks: Vec<NumberedBlock>,
    pub(crate) vertex_map: MapInput,
    /// The origin id of every local vertex.
    pub(crate) vertex_ids: Vec<i64>,
    /// The sharer sets of the local vertices.
    pub(crate) sharers: VertexSharers,
    pub(crate) node_map: MapInput,
    /// The origin id of every local node.
    pub(crate) node_ids: Vec<i64>,
    /// `[node][gdim]` in local order, not relative to the origin.
    pub(crate) coords: Vec<f64>,
}

/// The sharer set of a node, and its owner.
struct Row {
    ranks: Vec<i32>,
    owned_cell: Vec<bool>,
    owner: i32,
}

/// The owner of node `id` with sharer set `ranks` and `owned_cell`: the owned-cell sharer at
/// position `owner_hash(&[id]) mod n` of the n owned-cell sharers, in ascending rank order.
fn owner_of(id: i64, ranks: &[i32], owned_cell: &[bool]) -> i32 {
    let candidates: Vec<i32> = ranks
        .iter()
        .zip(owned_cell)
        .filter(|&(_, &o)| o)
        .map(|(&r, _)| r)
        .collect();
    let at = owner_hash(&[id]) % candidates.len() as u64;
    candidates[at as usize]
}

/// Wire kinds of the global-number requests.
const VERTEX: i64 = 0;
const NODE: i64 = 1;
const CELL: i64 = 2;

/// A u32 local index of `i`.
fn index_u32(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| GridError::TooLarge)
}

/// Stage 8: owners and local and global numbers of the cells, vertices and nodes of `local`,
/// whose node coordinates `nodes` holds. Collective.
///
/// - **Sharer sets.** Every rank posts the node ids of its local cells, each with whether an
///   owned cell holds it, to the rendezvous rank `owner_hash(&[id]) mod P`, which returns to
///   every holder the sharer set: the holding ranks, ascending, each with its owned-cell flag.
///   A vertex's sharer set is its node's.
/// - **Owners.** A vertex or node is owned by the owned-cell sharer at position
///   `owner_hash(&[id]) mod n` of its n owned-cell sharers; a cell by the rank that owns it
///   after stage 5.
/// - **Local numbers.** The cells of each block are its owned cells in stage-7 order, then its
///   ghost cells by (owner rank, origin id). Owned vertices and nodes are numbered in order of
///   the first local cell that touches them, blocks in agreed order and nodes in A.5 order
///   within a cell; ghost vertices and nodes follow, by (owner rank, global index).
/// - **Global numbers.** An exclusive scan of the owned counts (per block for cells) gives each
///   rank its owned range; ghosts ask their owners for their global numbers.
///
/// # Errors
///
/// `TooLarge`, agreed, if a local count does not fit u32. Errors of the exchanges.
pub(crate) fn number(
    comm: &dyn CommDyn,
    blocks: &[BlockSpec],
    local: &LocalCells,
    nodes: &Nodes,
) -> Result<Numbered> {
    let p = comm.size();
    let rank = comm.rank() as i32;
    let cells = &local.cells;
    let num_owned_cells = local.num_owned;

    // The local cells of each block, in local order (`cells` holds the owned cells first).
    let mut order: Vec<Vec<usize>> = vec![Vec::new(); blocks.len()];
    for i in 0..cells.len() {
        order[cells.block[i] as usize].push(i);
    }

    // Sharer sets from the rendezvous directory.
    let mut in_owned: HashSet<i64> = HashSet::new();
    let mut is_vertex: HashSet<i64> = HashSet::new();
    for i in 0..cells.len() {
        if i < num_owned_cells {
            in_owned.extend(cells.nodes_of(i));
        }
        is_vertex.extend(cells.vertices_of(i, blocks));
    }
    let mut out = Outbox::new(p);
    for &id in &nodes.ids {
        let owned = i64::from(in_owned.contains(&id));
        out.push(rendezvous_rank(&[id], p), &[&[id, owned]]);
    }
    let inbox = out.exchange(comm)?;
    // Records arrive by source rank, so each holder list is ascending.
    let mut directory: BTreeMap<i64, (Vec<i64>, Vec<i64>)> = BTreeMap::new();
    for (source, r) in inbox.records() {
        let (ranks, flags) = directory.entry(r[0]).or_default();
        ranks.push(source as i64);
        flags.push(r[1]);
    }
    let mut replies = Outbox::new(p);
    for (&id, (ranks, flags)) in &directory {
        for &r in ranks {
            replies.push(r as usize, &[&[id], ranks, flags]);
        }
    }
    let inbox = replies.exchange(comm)?;
    let mut rows: HashMap<i64, Row> = HashMap::with_capacity(nodes.ids.len());
    for (_, r) in inbox.records() {
        let n = (r.len() - 1) / 2;
        let ranks: Vec<i32> = r[1..=n].iter().map(|&x| x as i32).collect();
        let owned_cell: Vec<bool> = r[1 + n..].iter().map(|&x| x != 0).collect();
        let owner = owner_of(r[0], &ranks, &owned_cell);
        rows.insert(
            r[0],
            Row {
                ranks,
                owned_cell,
                owner,
            },
        );
    }

    // Owned vertices and nodes in order of the first touching local cell.
    let mut vertex_local: HashMap<i64, usize> = HashMap::new();
    let mut node_local: HashMap<i64, usize> = HashMap::new();
    let mut vertex_ids = Vec::new();
    let mut node_ids = Vec::new();
    for (b, cells_of_block) in order.iter().enumerate() {
        let nv = blocks[b].num_vertices();
        for &i in cells_of_block {
            for (j, &id) in cells.nodes_of(i).iter().enumerate() {
                if rows[&id].owner != rank {
                    continue;
                }
                node_local.entry(id).or_insert_with(|| {
                    node_ids.push(id);
                    node_ids.len() - 1
                });
                if j < nv {
                    vertex_local.entry(id).or_insert_with(|| {
                        vertex_ids.push(id);
                        vertex_ids.len() - 1
                    });
                }
            }
        }
    }
    let num_owned_vertices = vertex_ids.len();
    let num_owned_nodes = node_ids.len();
    let too_large = [is_vertex.len(), nodes.ids.len(), cells.len()]
        .into_iter()
        .any(|n| index_u32(n).is_err());
    agree(
        comm,
        if too_large {
            Err(GridError::TooLarge)
        } else {
            Ok(())
        },
    )?;

    // Owned ranges.
    let vertex_first = comm.exscan_sum(num_owned_vertices as u64) as i64;
    let node_first = comm.exscan_sum(num_owned_nodes as u64) as i64;
    let mut cell_global: HashMap<i64, i64> = HashMap::new();
    for cells_of_block in &order {
        let owned: Vec<usize> = cells_of_block
            .iter()
            .copied()
            .filter(|&i| i < num_owned_cells)
            .collect();
        let first = comm.exscan_sum(owned.len() as u64) as i64;
        for (k, &i) in owned.iter().enumerate() {
            cell_global.insert(cells.ids[i], first + k as i64);
        }
    }

    // Ghost global numbers from their owners.
    let mut requests = Outbox::new(p);
    for &id in &nodes.ids {
        let owner = rows[&id].owner;
        if owner != rank {
            requests.push(owner as usize, &[&[NODE, id]]);
            if is_vertex.contains(&id) {
                requests.push(owner as usize, &[&[VERTEX, id]]);
            }
        }
    }
    for i in num_owned_cells..cells.len() {
        requests.push(local.owner[i] as usize, &[&[CELL, cells.ids[i]]]);
    }
    let inbox = requests.exchange(comm)?;
    let mut answers = Outbox::new(p);
    for (source, r) in inbox.records() {
        let (kind, id) = (r[0], r[1]);
        let global = match kind {
            VERTEX => vertex_first + vertex_local[&id] as i64,
            NODE => node_first + node_local[&id] as i64,
            _ => cell_global[&id],
        };
        answers.push(source, &[&[kind, id, global]]);
    }
    let inbox = answers.exchange(comm)?;
    let ghost_global: HashMap<(i64, i64), i64> =
        inbox.records().map(|(_, r)| ((r[0], r[1]), r[2])).collect();

    // Ghost vertices and nodes by (owner rank, global index).
    let ghosts_of = |kind: i64, ids: &mut Vec<i64>, local: &mut HashMap<i64, usize>| {
        let mut ghosts: Vec<(i32, i64, i64)> = ghost_global
            .iter()
            .filter(|&(&(k, _), _)| k == kind)
            .map(|(&(_, id), &g)| (rows[&id].owner, g, id))
            .collect();
        ghosts.sort_unstable();
        for &(_, _, id) in &ghosts {
            local.insert(id, ids.len());
            ids.push(id);
        }
        MapInput {
            num_owned: ids.len() - ghosts.len(),
            ghosts: ghosts.iter().map(|&(_, g, _)| g).collect(),
            ghost_owners: ghosts.iter().map(|&(o, _, _)| o).collect(),
        }
    };
    let vertex_map = ghosts_of(VERTEX, &mut vertex_ids, &mut vertex_local);
    let node_map = ghosts_of(NODE, &mut node_ids, &mut node_local);
    debug_assert_eq!(vertex_map.num_owned, num_owned_vertices);
    debug_assert_eq!(node_map.num_owned, num_owned_nodes);

    let numbered_blocks = order
        .iter()
        .enumerate()
        .map(|(b, cells_of_block)| {
            let nv = blocks[b].num_vertices();
            let ghosts: Vec<usize> = cells_of_block
                .iter()
                .copied()
                .filter(|&i| i >= num_owned_cells)
                .collect();
            let mut vertices = Vec::with_capacity(cells_of_block.len() * nv);
            let mut dofmap = Vec::with_capacity(cells_of_block.len() * blocks[b].num_nodes());
            for &i in cells_of_block {
                let row = cells.nodes_of(i);
                vertices.extend(row[..nv].iter().map(|id| vertex_local[id] as u32));
                dofmap.extend(row.iter().map(|id| node_local[id] as u32));
            }
            NumberedBlock {
                cells: cells.select(cells_of_block),
                map: MapInput {
                    num_owned: cells_of_block.len() - ghosts.len(),
                    ghosts: ghosts
                        .iter()
                        .map(|&i| ghost_global[&(CELL, cells.ids[i])])
                        .collect(),
                    ghost_owners: ghosts.iter().map(|&i| local.owner[i]).collect(),
                },
                vertices,
                dofmap,
            }
        })
        .collect();

    let mut sharers = VertexSharers {
        offsets: vec![0],
        ranks: Vec::new(),
        owned_cell: Vec::new(),
    };
    for id in &vertex_ids {
        let row = &rows[id];
        sharers.ranks.extend_from_slice(&row.ranks);
        sharers.owned_cell.extend_from_slice(&row.owned_cell);
        sharers.offsets.push(sharers.ranks.len());
    }
    let coords = node_ids
        .iter()
        .flat_map(|&id| nodes.coords_of(id).iter().copied())
        .collect();
    Ok(Numbered {
        blocks: numbered_blocks,
        vertex_map,
        vertex_ids,
        sharers,
        node_map,
        node_ids,
        coords,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::nodes::fetch_nodes;
    use crate::pipeline::order::order_cells;
    use crate::pipeline::testing::{OwnedInput, local_cells_by, meshes};
    use crate::{CellOrdering, GhostMode};
    use gridcl_parallel::IndexMap;
    use gridcl_partition::Block;
    use gridcl_testing::{HostMesh, ThreadComm, parallel::RoundRobin};

    /// One rank's result of stages 1–8, with the first global index of its owned vertices,
    /// nodes and cells of each block, from the `IndexMap`s built of the stage-8 inputs.
    struct RankOut {
        local: LocalCells,
        numbered: Numbered,
        vertex_first: i64,
        node_first: i64,
        cell_first: Vec<i64>,
    }

    /// Stages 1–8 on `size` ranks from even chunks, with the round-robin partitioner if
    /// `round_robin`, else the block partitioner.
    fn run(
        h: &HostMesh,
        size: usize,
        mode: GhostMode,
        round_robin: bool,
        ordering: CellOrdering,
    ) -> Vec<RankOut> {
        ThreadComm::run(size, |comm| {
            let input = OwnedInput::of_chunk(&h.chunk(comm.rank(), size));
            let (read, local) = if round_robin {
                local_cells_by(&comm, &input.input(), mode, &RoundRobin)
            } else {
                local_cells_by(&comm, &input.input(), mode, &Block)
            };
            let nodes = fetch_nodes(&comm, &read.blocks, h.gdim, &read.nodes, &local).unwrap();
            let local = order_cells(&read.blocks, local, &nodes, ordering);
            let numbered = number(&comm, &read.blocks, &local, &nodes).unwrap();
            let first = |m: &MapInput| {
                let map = IndexMap::with_ghosts(
                    &comm,
                    m.num_owned,
                    m.ghosts.clone(),
                    m.ghost_owners.clone(),
                )
                .unwrap();
                assert_eq!(map.num_ghosts(), m.ghosts.len());
                map.owned_range().0
            };
            RankOut {
                vertex_first: first(&numbered.vertex_map),
                node_first: first(&numbered.node_map),
                cell_first: numbered.blocks.iter().map(|b| first(&b.map)).collect(),
                local,
                numbered,
            }
        })
    }

    /// The owner of a node by the rules of "Ownership of vertices and entities": the
    /// owned-cell sharer at position `owner_hash(&[id]) mod n`, in ascending rank order.
    fn reference_owner(id: i64, sharers: &[(i32, bool)]) -> i32 {
        let candidates: Vec<i32> = sharers.iter().filter(|s| s.1).map(|s| s.0).collect();
        candidates[(owner_hash(&[id]) % candidates.len() as u64) as usize]
    }

    /// The global index of every local index of `map`, owned from `first`.
    fn globals(map: &MapInput, first: i64) -> Vec<i64> {
        (0..map.num_owned as i64)
            .map(|k| first + k)
            .chain(map.ghosts.iter().copied())
            .collect()
    }

    /// Checks the numbering of every rank against host references of the ownership rules.
    fn check(h: &HostMesh, outs: &[RankOut], context: &str) {
        let coords_of = |id: i64| {
            let n = (0..h.num_nodes()).find(|&n| h.node_id(n) == id).unwrap();
            h.nodes[n * h.gdim..(n + 1) * h.gdim].to_vec()
        };
        // The sharer set of every node id: (rank, holds it in an owned cell), by rank.
        let mut sharers: BTreeMap<i64, Vec<(i32, bool)>> = BTreeMap::new();
        for (r, o) in outs.iter().enumerate() {
            let mut held: BTreeMap<i64, bool> = BTreeMap::new();
            for i in 0..o.local.cells.len() {
                for &id in o.local.cells.nodes_of(i) {
                    *held.entry(id).or_default() |= i < o.local.num_owned;
                }
            }
            for (id, owned) in held {
                sharers.entry(id).or_default().push((r as i32, owned));
            }
        }

        // Global index → (origin id, owner rank) of the owned vertices, nodes and cells.
        let mut owned_vertices: BTreeMap<i64, i64> = BTreeMap::new();
        let mut owned_nodes: BTreeMap<i64, i64> = BTreeMap::new();
        let mut owned_cells: Vec<BTreeMap<i64, i64>> = Vec::new();
        for (r, o) in outs.iter().enumerate() {
            let rank = r as i32;
            let context = format!("{context}, rank {r}");
            let nb = &o.numbered;
            let vm = &nb.vertex_map;
            let nm = &nb.node_map;
            let vglobal = globals(vm, o.vertex_first);
            let nglobal = globals(nm, o.node_first);
            let owner = |m: &MapInput, k: usize| {
                if k < m.num_owned {
                    rank
                } else {
                    m.ghost_owners[k - m.num_owned]
                }
            };

            // Vertices: sharer rows and owners by the rules; owned ones are unique globally.
            assert_eq!(
                nb.vertex_ids.len(),
                vm.num_owned + vm.ghosts.len(),
                "{context}"
            );
            assert_eq!(
                nb.sharers.offsets.len(),
                nb.vertex_ids.len() + 1,
                "{context}"
            );
            for (v, &id) in nb.vertex_ids.iter().enumerate() {
                let want = &sharers[&id];
                let row = nb.sharers.offsets[v]..nb.sharers.offsets[v + 1];
                let got: Vec<(i32, bool)> = nb.sharers.ranks[row.clone()]
                    .iter()
                    .copied()
                    .zip(nb.sharers.owned_cell[row].iter().copied())
                    .collect();
                assert_eq!(&got, want, "{context}: sharers of vertex {id}");
                assert_eq!(
                    owner(vm, v),
                    reference_owner(id, want),
                    "{context}: vertex {id}"
                );
                if v < vm.num_owned {
                    assert!(
                        owned_vertices.insert(vglobal[v], id).is_none(),
                        "{context}: vertex global {} twice",
                        vglobal[v]
                    );
                }
            }
            // Nodes: owners by the rules, coordinates.
            assert_eq!(
                nb.node_ids.len(),
                nm.num_owned + nm.ghosts.len(),
                "{context}"
            );
            for (n, &id) in nb.node_ids.iter().enumerate() {
                assert_eq!(
                    owner(nm, n),
                    reference_owner(id, &sharers[&id]),
                    "{context}: node {id}"
                );
                assert_eq!(
                    nb.coords[n * h.gdim..(n + 1) * h.gdim],
                    coords_of(id),
                    "{context}: node {id}"
                );
                if n < nm.num_owned {
                    assert!(owned_nodes.insert(nglobal[n], id).is_none(), "{context}");
                }
            }
            // Ghosts by (owner rank, global index).
            for (m, global) in [(vm, &vglobal), (nm, &nglobal)] {
                let keys: Vec<(i32, i64)> = (m.num_owned..global.len())
                    .map(|k| (owner(m, k), global[k]))
                    .collect();
                assert!(
                    keys.is_sorted() && keys.iter().all(|k| k.0 != rank),
                    "{context}"
                );
            }

            // Cell blocks: owned cells in local order, then the ghosts; rows map back to ids.
            owned_cells.resize(nb.blocks.len(), BTreeMap::new());
            let num_owned = o.local.num_owned;
            let mut owned_ids = Vec::new();
            let mut ghost_ids = Vec::new();
            for (b, block) in nb.blocks.iter().enumerate() {
                let cells = &block.cells;
                let m = &block.map;
                let n = cells.len();
                assert_eq!(n, m.num_owned + m.ghosts.len(), "{context}");
                let nv = block.vertices.len() / n.max(1);
                let k = block.dofmap.len() / n.max(1);
                for c in 0..n {
                    let vertices: Vec<i64> = block.vertices[c * nv..(c + 1) * nv]
                        .iter()
                        .map(|&v| nb.vertex_ids[v as usize])
                        .collect();
                    let row: Vec<i64> = block.dofmap[c * k..(c + 1) * k]
                        .iter()
                        .map(|&x| nb.node_ids[x as usize])
                        .collect();
                    assert_eq!(row, cells.nodes_of(c), "{context}");
                    assert_eq!(vertices, row[..nv], "{context}");
                }
                owned_ids.extend_from_slice(&cells.ids[..m.num_owned]);
                ghost_ids.extend_from_slice(&cells.ids[m.num_owned..]);
                for (c, &g) in globals(m, o.cell_first[b]).iter().enumerate() {
                    if c < m.num_owned {
                        owned_cells[b].insert(g, cells.ids[c]);
                    } else {
                        let i = num_owned
                            + o.local.cells.ids[num_owned..]
                                .iter()
                                .position(|&id| id == cells.ids[c])
                                .unwrap();
                        assert_eq!(
                            m.ghost_owners[c - m.num_owned],
                            o.local.owner[i],
                            "{context}"
                        );
                    }
                }
            }
            // The blocks hold the local cells, owned first, in local order within a block.
            let by_block = |range: std::ops::Range<usize>| -> Vec<i64> {
                let mut ids: Vec<(u32, usize)> =
                    range.map(|i| (o.local.cells.block[i], i)).collect();
                ids.sort_unstable();
                ids.iter().map(|&(_, i)| o.local.cells.ids[i]).collect()
            };
            assert_eq!(owned_ids, by_block(0..num_owned), "{context}");
            assert_eq!(
                ghost_ids,
                by_block(num_owned..o.local.cells.len()),
                "{context}"
            );
        }

        // Every vertex, node and cell has exactly one owner, and the globals are 0..N.
        let vertices: std::collections::BTreeSet<i64> = h
            .blocks
            .iter()
            .flat_map(|b| {
                let nv = gridcl_refcell::vertices(b.cell).len();
                b.cells
                    .chunks_exact(b.nodes_per_cell())
                    .flat_map(move |row| row[..nv].iter().map(|&n| h.node_id(n as usize)))
            })
            .collect();
        let nodes: std::collections::BTreeSet<i64> = h
            .blocks
            .iter()
            .flat_map(|b| b.cells.iter().map(|&n| h.node_id(n as usize)))
            .collect();
        let dense = |m: &BTreeMap<i64, i64>| m.keys().copied().eq(0..m.len() as i64);
        assert!(dense(&owned_vertices), "{context}");
        assert!(dense(&owned_nodes), "{context}");
        let ids = |m: &BTreeMap<i64, i64>| -> std::collections::BTreeSet<i64> {
            m.values().copied().collect()
        };
        assert_eq!(ids(&owned_vertices), vertices, "{context}: owned vertices");
        assert_eq!(ids(&owned_nodes), nodes, "{context}: owned nodes");
        assert_eq!(
            owned_vertices.len(),
            vertices.len(),
            "{context}: vertex owners"
        );
        assert_eq!(owned_nodes.len(), nodes.len(), "{context}: node owners");
        let num_cells: usize = owned_cells.iter().map(BTreeMap::len).sum();
        assert_eq!(
            num_cells,
            h.cell_origin_ids().iter().map(Vec::len).sum::<usize>()
        );
        assert!(owned_cells.iter().all(dense), "{context}");

        // Ghost global numbers are their owners'.
        let invert = |m: &BTreeMap<i64, i64>| -> HashMap<i64, i64> {
            m.iter().map(|(&g, &id)| (id, g)).collect()
        };
        let (vertex_global, node_global) = (invert(&owned_vertices), invert(&owned_nodes));
        let cell_global: Vec<HashMap<i64, i64>> = owned_cells.iter().map(invert).collect();
        for (r, o) in outs.iter().enumerate() {
            let nb = &o.numbered;
            for (m, ids, want) in [
                (&nb.vertex_map, &nb.vertex_ids, &vertex_global),
                (&nb.node_map, &nb.node_ids, &node_global),
            ] {
                for (k, &g) in m.ghosts.iter().enumerate() {
                    assert_eq!(g, want[&ids[m.num_owned + k]], "{context}, rank {r}");
                }
            }
            for (b, block) in nb.blocks.iter().enumerate() {
                let m = &block.map;
                for (k, &g) in m.ghosts.iter().enumerate() {
                    let id = block.cells.ids[m.num_owned + k];
                    assert_eq!(g, cell_global[b][&id], "{context}, rank {r}");
                }
            }
        }
    }

    /// Owners, sharer sets and global numbers follow the rules on 1–4 ranks, for every ghost
    /// mode, with the block and the round-robin partitioner.
    #[test]
    fn owners_and_numbers_follow_the_rules() {
        let modes = [
            GhostMode::None,
            GhostMode::SharedFacet,
            GhostMode::SharedVertex { layers: 2 },
        ];
        for (name, h) in meshes() {
            for size in 1..=4 {
                for mode in modes {
                    for round_robin in [false, true] {
                        let outs = run(&h, size, mode, round_robin, CellOrdering::Input);
                        let context =
                            format!("{name}, {size} ranks, {mode:?}, round robin {round_robin}");
                        check(&h, &outs, &context);
                    }
                }
            }
        }
    }

    /// Numbering follows the local cell order, so it holds for the locality orderings too.
    #[test]
    fn owners_and_numbers_follow_the_rules_in_locality_order() {
        let h = gridcl_testing::mixed_box(2, 1).with_shuffled_cells(5);
        for ordering in [CellOrdering::Hilbert, CellOrdering::ReverseCuthillMcKee] {
            for size in [1, 3] {
                let outs = run(&h, size, GhostMode::SharedFacet, false, ordering);
                check(&h, &outs, &format!("{ordering:?}, {size} ranks"));
            }
        }
    }

    /// On one rank with `Input` ordering, cells keep the input order within each block, and
    /// vertices and nodes are numbered in order of the first touching cell, blocks in call
    /// order: the numbering of the serial `MeshBuilder`.
    #[test]
    fn one_rank_numbers_by_first_touching_cell() {
        for (name, h) in meshes() {
            let out = run(&h, 1, GhostMode::None, false, CellOrdering::Input);
            let nb = &out[0].numbered;
            let (mut vertices, mut nodes) = (Vec::new(), Vec::new());
            for (b, ids) in h.blocks.iter().zip(h.cell_origin_ids()) {
                let nv = gridcl_refcell::vertices(b.cell).len();
                for row in b.cells.chunks_exact(b.nodes_per_cell()) {
                    for (j, &n) in row.iter().enumerate() {
                        let id = h.node_id(n as usize);
                        if !nodes.contains(&id) {
                            nodes.push(id);
                        }
                        if j < nv && !vertices.contains(&id) {
                            vertices.push(id);
                        }
                    }
                }
                let block = nb
                    .blocks
                    .iter()
                    .find(|x| x.cells.ids.first() == ids.first());
                assert_eq!(block.unwrap().cells.ids, ids, "{name}");
            }
            assert_eq!(nb.vertex_ids, vertices, "{name}");
            assert_eq!(nb.node_ids, nodes, "{name}");
            assert!(nb.vertex_map.ghosts.is_empty() && nb.node_map.ghosts.is_empty());
            assert!(nb.sharers.ranks.iter().all(|&r| r == 0));
            assert!(nb.sharers.owned_cell.iter().all(|&o| o));
        }
    }

    /// Ranks without cells number nothing, and the others are unaffected.
    #[test]
    fn empty_ranks() {
        let h = gridcl_testing::interval_box(2, 2);
        for mode in [GhostMode::None, GhostMode::SharedVertex { layers: 1 }] {
            let outs = run(&h, 4, mode, false, CellOrdering::Input);
            check(&h, &outs, &format!("{mode:?}"));
            let empty: Vec<usize> = (0..4).filter(|&r| outs[r].local.cells.len() == 0).collect();
            assert_eq!(empty, [1, 3], "{mode:?}");
            for &r in &empty {
                let nb = &outs[r].numbered;
                assert!(nb.vertex_ids.is_empty() && nb.node_ids.is_empty());
                assert_eq!(nb.sharers.offsets, [0]);
                assert!(
                    nb.blocks
                        .iter()
                        .all(|b| b.cells.len() == 0 && b.map.num_owned == 0)
                );
            }
        }
    }
}
