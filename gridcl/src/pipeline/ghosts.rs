//! Stage 5: ghost cells per [`GhostMode`] (docs/design.md, "Parallel creation pipeline";
//! ADR-0026, item 16).

use std::collections::{HashMap, HashSet};

use gridcl_device::Result;
use gridcl_parallel::CommDyn;

use super::rendezvous::{Outbox, rendezvous_rank};
use super::{BlockSpec, Cells, facet_keys};
use crate::GhostMode;

/// A rank's cells after stage 5: the owned cells, then the ghost cells ordered by (owner rank,
/// origin id).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocalCells {
    pub(crate) cells: Cells,
    pub(crate) num_owned: usize,
    /// The owner rank of every cell.
    pub(crate) owner: Vec<i32>,
}

/// Stage 5: adds to `owned` the ghost cells of `mode`, with their owner ranks and tags.
/// Collective.
///
/// - `SharedFacet`: the cells of other ranks that share a facet with an owned cell.
/// - `SharedVertex { layers: k }`: k rounds; round 1 adds the cells of other ranks that share a
///   vertex with an owned cell, and each further round those that share a vertex with the
///   cells the previous round added. `layers: 0` is `None`.
///
/// The keys (facet keys, or single vertex ids) of every owned cell go to a rendezvous
/// directory. In round 1 the directory sends each rank that owns a cell with a key the other
/// ranks' cells with that key; in later rounds ranks ask for the keys of their new ghosts.
/// Ranks fetch the cells they lack from the owners.
///
/// # Errors
///
/// Errors of the exchanges.
pub(crate) fn ghost_cells(
    comm: &dyn CommDyn,
    blocks: &[BlockSpec],
    owned: Cells,
    mode: GhostMode,
) -> Result<LocalCells> {
    let rank = comm.rank() as i32;
    let num_owned = owned.len();
    let (facets, rounds) = match mode {
        GhostMode::None | GhostMode::SharedVertex { layers: 0 } => {
            return Ok(LocalCells {
                cells: owned,
                num_owned,
                owner: vec![rank; num_owned],
            });
        }
        GhostMode::SharedFacet => (true, 1),
        GhostMode::SharedVertex { layers } => (false, layers),
    };
    let keys_of = |cells: &Cells, i: usize| -> Vec<Vec<i64>> {
        if facets {
            facet_keys(cells, i, blocks)
        } else {
            cells
                .vertices_of(i, blocks)
                .iter()
                .map(|&v| vec![v])
                .collect()
        }
    };
    let p = comm.size();

    // The directory: key → (owner, cell id) of every owned cell with that key.
    let mut out = Outbox::new(p);
    for i in 0..num_owned {
        for key in keys_of(&owned, i) {
            out.push(rendezvous_rank(&key, p), &[&[owned.ids[i]], &key]);
        }
    }
    let inbox = out.exchange(comm)?;
    let mut directory: HashMap<Vec<i64>, Vec<(i64, i64)>> = HashMap::new();
    for (source, r) in inbox.records() {
        directory
            .entry(r[1..].to_vec())
            .or_default()
            .push((source as i64, r[0]));
    }

    // Round 1: every owner of a key gets the other owners' cells with that key.
    let mut offers = Outbox::new(p);
    for entries in directory.values() {
        let mut owners: Vec<i64> = entries.iter().map(|&(r, _)| r).collect();
        owners.sort_unstable();
        owners.dedup();
        if owners.len() < 2 {
            continue;
        }
        for &s in &owners {
            for &(r, id) in entries.iter().filter(|&&(r, _)| r != s) {
                offers.push(s as usize, &[&[r, id]]);
            }
        }
    }
    let mut offered = offers.exchange(comm)?;

    let index_of: HashMap<i64, usize> = owned
        .ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let mut held: HashSet<i64> = owned.ids.iter().copied().collect();
    let mut ghosts = Cells::new();
    let mut ghost_owner = Vec::new();
    for round in 1..=rounds {
        // The offered cells this rank lacks, by (owner, id).
        let mut new: Vec<(i64, i64)> = offered
            .records()
            .map(|(_, r)| (r[0], r[1]))
            .filter(|(_, id)| !held.contains(id))
            .collect();
        new.sort_unstable();
        new.dedup();
        held.extend(new.iter().map(|&(_, id)| id));

        // Fetch them from their owners; they arrive by owner, in request order.
        let mut requests = Outbox::new(p);
        for &(r, id) in &new {
            requests.push(r as usize, &[&[id]]);
        }
        let inbox = requests.exchange(comm)?;
        let mut replies = Outbox::new(p);
        for (source, r) in inbox.records() {
            replies.push(source, &[&owned.record(index_of[&r[0]])]);
        }
        let inbox = replies.exchange(comm)?;
        let first_new = ghosts.len();
        for (source, r) in inbox.records() {
            ghosts.push_record(r);
            ghost_owner.push(source as i32);
        }
        if round == rounds {
            break;
        }

        // Ask the directory for the cells around the new ghosts.
        let mut asked = HashSet::new();
        let mut asks = Outbox::new(p);
        for i in first_new..ghosts.len() {
            for key in keys_of(&ghosts, i) {
                if asked.insert(key.clone()) {
                    asks.push(rendezvous_rank(&key, p), &[&key]);
                }
            }
        }
        let inbox = asks.exchange(comm)?;
        let mut answers = Outbox::new(p);
        for (source, key) in inbox.records() {
            for &(r, id) in directory.get(key).into_iter().flatten() {
                if r != source as i64 {
                    answers.push(source, &[&[r, id]]);
                }
            }
        }
        offered = answers.exchange(comm)?;
    }

    // Ghosts by (owner, origin id).
    let mut order: Vec<usize> = (0..ghosts.len()).collect();
    order.sort_unstable_by_key(|&i| (ghost_owner[i], ghosts.ids[i]));
    let mut cells = owned;
    cells.append(&ghosts.select(&order));
    let mut owner = vec![rank; num_owned];
    owner.extend(order.iter().map(|&i| ghost_owner[i]));
    Ok(LocalCells {
        cells,
        num_owned,
        owner,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::dual_graph::{dual_graph, partition};
    use crate::pipeline::input::read;
    use crate::pipeline::migrate::migrate;
    use crate::pipeline::testing::{
        OwnedInput, host_adjacency, host_cells, host_facets, host_vertices, meshes,
    };
    use gridcl_partition::Block;
    use gridcl_testing::{ThreadComm, parallel::RoundRobin};
    use std::collections::{BTreeMap, BTreeSet};

    /// The host reference: the ghost cells of `rank` under `mode`, given the owner of every
    /// cell.
    fn host_ghosts(
        owners: &BTreeMap<i64, i32>,
        facets: &BTreeMap<i64, Vec<i64>>,
        vertices: &BTreeMap<i64, Vec<i64>>,
        rank: i32,
        mode: GhostMode,
    ) -> BTreeSet<i64> {
        let (adjacency, rounds) = match mode {
            GhostMode::None | GhostMode::SharedVertex { layers: 0 } => return BTreeSet::new(),
            GhostMode::SharedFacet => (facets, 1),
            GhostMode::SharedVertex { layers } => (vertices, layers),
        };
        let mut held: BTreeSet<i64> = owners
            .iter()
            .filter(|&(_, &r)| r == rank)
            .map(|(&id, _)| id)
            .collect();
        let mut frontier = held.clone();
        let mut ghosts = BTreeSet::new();
        for _ in 0..rounds {
            let next: BTreeSet<i64> = frontier
                .iter()
                .flat_map(|id| &adjacency[id])
                .copied()
                .filter(|id| !held.contains(id))
                .collect();
            held.extend(&next);
            ghosts.extend(&next);
            frontier = next;
        }
        ghosts
    }

    /// Ghost sets, owners and ghost data equal the host computation, in every mode, for
    /// block and round-robin partitions on 1–4 ranks.
    #[test]
    fn ghost_sets_equal_the_host_computation() {
        let modes = [
            GhostMode::None,
            GhostMode::SharedFacet,
            GhostMode::SharedVertex { layers: 0 },
            GhostMode::SharedVertex { layers: 1 },
            GhostMode::SharedVertex { layers: 2 },
            GhostMode::SharedVertex { layers: 3 },
        ];
        for (name, h) in meshes() {
            let want_cells = host_cells(&h);
            let facets = host_adjacency(&host_facets(&h));
            let vertices = host_adjacency(&host_vertices(&h));
            for size in 1..=4 {
                for (m, &mode) in modes.iter().enumerate() {
                    let round_robin = (m + size) % 2 == 0;
                    let local = ThreadComm::run(size, |comm| {
                        let input = OwnedInput::of_chunk(&h.chunk(comm.rank(), size));
                        let read = read(&comm, &input.input()).unwrap();
                        let g = dual_graph(&comm, &read.blocks, &read.cells).unwrap();
                        let dest = if round_robin {
                            partition(&comm, &RoundRobin, &g)
                        } else {
                            partition(&comm, &Block, &g)
                        };
                        let owned = migrate(&comm, &read.cells, &dest.unwrap()).unwrap();
                        let local = ghost_cells(&comm, &read.blocks, owned, mode).unwrap();
                        (read.blocks, local)
                    });
                    let owners: BTreeMap<i64, i32> = local
                        .iter()
                        .enumerate()
                        .flat_map(|(r, (_, l))| {
                            l.cells.ids[..l.num_owned]
                                .iter()
                                .map(move |&id| (id, r as i32))
                        })
                        .collect();
                    assert_eq!(owners.len(), want_cells.len(), "{name}");
                    for (r, (blocks, l)) in local.iter().enumerate() {
                        let context = format!("{name}, {mode:?}, rank {r} of {size}");
                        let c = &l.cells;
                        let ghost_ids = &c.ids[l.num_owned..];
                        let want = host_ghosts(&owners, &facets, &vertices, r as i32, mode);
                        let got: BTreeSet<i64> = ghost_ids.iter().copied().collect();
                        assert_eq!(got.len(), ghost_ids.len(), "{context}: repeated ghosts");
                        assert_eq!(got, want, "{context}");
                        assert!(l.owner[..l.num_owned].iter().all(|&o| o == r as i32));
                        let keys: Vec<(i32, i64)> = (l.num_owned..c.len())
                            .map(|i| (l.owner[i], c.ids[i]))
                            .collect();
                        assert!(keys.is_sorted(), "{context}: ghost order");
                        for i in l.num_owned..c.len() {
                            assert_eq!(l.owner[i], owners[&c.ids[i]], "{context}");
                            let cell = blocks[c.block[i] as usize].cell;
                            let entry = (cell, c.nodes_of(i).to_vec(), c.tags_of(i).to_vec());
                            assert_eq!(entry, want_cells[&c.ids[i]], "{context}");
                        }
                    }
                }
            }
        }
    }
}
