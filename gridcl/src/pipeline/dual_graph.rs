//! Stages 2 and 3: the facet dual graph of the input cells, and its partition
//! (docs/design.md, "Parallel creation pipeline"; ADR-0026, item 2).

use std::collections::HashMap;

use gridcl_device::{GridError, Result};
use gridcl_parallel::{CommDyn, CommExt};
use gridcl_partition::{DistGraph, Partitioner};

use super::rendezvous::{Outbox, rendezvous_rank};
use super::{BlockSpec, Cells, agree, facet_keys};

/// Stage 2: the dual graph of `cells`, whose vertices are the cells in rank order and whose
/// edges join cells that share a facet. Facet keys (sorted vertex origin ids) meet at their
/// rendezvous rank, which sends every pair of cells with the same key to the first cell's
/// rank. `vwgt` is `None` (M7 adds weights). Collective.
///
/// # Errors
///
/// Errors of the exchanges.
pub(crate) fn dual_graph(
    comm: &dyn CommDyn,
    blocks: &[BlockSpec],
    cells: &Cells,
) -> Result<DistGraph> {
    let p = comm.size();
    let mut vtxdist = vec![0_i64];
    for n in comm.all_gather(cells.len() as i64) {
        vtxdist.push(vtxdist.last().expect("non-empty") + n);
    }
    let first = vtxdist[comm.rank()];

    let mut out = Outbox::new(p);
    for i in 0..cells.len() {
        for key in facet_keys(cells, i, blocks) {
            out.push(rendezvous_rank(&key, p), &[&[first + i as i64], &key]);
        }
    }
    let inbox = out.exchange(comm)?;

    // Every pair of cells with one key; the order of the pairs does not matter, as each row
    // is sorted below.
    let mut by_key: HashMap<&[i64], Vec<i64>> = HashMap::new();
    for (_, r) in inbox.records() {
        by_key.entry(&r[1..]).or_default().push(r[0]);
    }
    let rank_of = |g: i64| vtxdist.partition_point(|&x| x <= g) - 1;
    let mut pairs = Outbox::new(p);
    for group in by_key.values().filter(|g| g.len() > 1) {
        for &a in group {
            for &b in group.iter().filter(|&&b| b != a) {
                pairs.push(rank_of(a), &[&[a, b]]);
            }
        }
    }
    let inbox = pairs.exchange(comm)?;

    let mut rows: Vec<Vec<i64>> = vec![Vec::new(); cells.len()];
    for (_, r) in inbox.records() {
        rows[(r[0] - first) as usize].push(r[1]);
    }
    let mut xadj = vec![0_i64];
    let mut adjncy = Vec::new();
    for mut row in rows {
        row.sort_unstable();
        row.dedup();
        adjncy.extend_from_slice(&row);
        xadj.push(adjncy.len() as i64);
    }
    Ok(DistGraph {
        vtxdist,
        xadj,
        adjncy,
        vwgt: None,
    })
}

/// Stage 3: the destination rank of every local cell, from `partitioner` with one part per
/// rank. Collective.
///
/// # Errors
///
/// The partitioner's errors; `LengthMismatch` unless there is one destination per local graph
/// vertex, and `IndexOutOfRange` for a destination outside the ranks; agreed.
pub(crate) fn partition(
    comm: &dyn CommDyn,
    partitioner: &dyn Partitioner,
    graph: &DistGraph,
) -> Result<Vec<i32>> {
    let p = comm.size();
    let n = graph.xadj.len() - 1;
    let local = partitioner.partition(comm, graph, p).and_then(|dest| {
        if dest.len() != n {
            return Err(GridError::LengthMismatch {
                expected: n,
                found: dest.len(),
            });
        }
        let in_range = |d: i32| usize::try_from(d).is_ok_and(|d| d < p);
        if let Some(&bad) = dest.iter().find(|&&d| !in_range(d)) {
            return Err(GridError::IndexOutOfRange {
                index: usize::try_from(bad).unwrap_or(usize::MAX),
                len: p,
            });
        }
        Ok(dest)
    });
    agree(comm, local)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::input::read;
    use crate::pipeline::testing::{OwnedInput, host_adjacency, host_facets, meshes};
    use gridcl_partition::Block;
    use gridcl_testing::ThreadComm;

    /// The graph equals the serial facet graph of the whole mesh, in cell origin ids.
    #[test]
    fn graph_equals_the_facet_graph() {
        for (name, h) in meshes() {
            let want = host_adjacency(&host_facets(&h));
            for size in 1..=4 {
                let parts = ThreadComm::run(size, |comm| {
                    let input = OwnedInput::of_chunk(&h.chunk(comm.rank(), size));
                    let read = read(&comm, &input.input()).unwrap();
                    let g = dual_graph(&comm, &read.blocks, &read.cells).unwrap();
                    assert_eq!(g.vtxdist.len(), size + 1);
                    assert_eq!(g.xadj.len(), read.cells.len() + 1);
                    (read.cells.ids, g)
                });
                // Graph vertex → cell origin id.
                let ids: Vec<i64> = parts.iter().flat_map(|(ids, _)| ids.clone()).collect();
                let mut got = std::collections::BTreeMap::new();
                for (cells, g) in &parts {
                    for (i, &id) in cells.iter().enumerate() {
                        let row = &g.adjncy[g.xadj[i] as usize..g.xadj[i + 1] as usize];
                        let mut adj: Vec<i64> = row.iter().map(|&x| ids[x as usize]).collect();
                        adj.sort_unstable();
                        got.insert(id, adj);
                    }
                }
                assert_eq!(got, want, "{name} on {size} ranks");
            }
        }
    }

    /// A partitioner that returns too few destinations, or one out of range, gives an agreed
    /// error.
    #[test]
    fn bad_destinations_are_agreed_errors() {
        struct Bad(i32);
        impl Partitioner for Bad {
            fn name(&self) -> &str {
                "bad"
            }
            fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, _: usize) -> Result<Vec<i32>> {
                let n = g.xadj.len() - 1;
                Ok(match (comm.rank(), self.0) {
                    (1, -1) => vec![0; n + 1],
                    (1, d) => vec![d; n],
                    _ => vec![0; n],
                })
            }
        }
        for (bad, want_local) in [(-1, "length"), (3, "range"), (-4, "range")] {
            let out = ThreadComm::run(3, |comm| {
                let g = DistGraph {
                    vtxdist: vec![0, 1, 2, 3],
                    xadj: vec![0, 0],
                    adjncy: vec![],
                    vwgt: None,
                };
                partition(&comm, &Bad(bad), &g)
            });
            assert!(matches!(out[0], Err(GridError::RemoteFailure { rank: 1 })));
            assert!(matches!(out[2], Err(GridError::RemoteFailure { rank: 1 })));
            match (want_local, &out[1]) {
                (
                    "length",
                    Err(GridError::LengthMismatch {
                        expected: 1,
                        found: 2,
                    }),
                ) => {}
                ("range", Err(GridError::IndexOutOfRange { len: 3, .. })) => {}
                (_, other) => panic!("{bad}: {other:?}"),
            }
        }
        let ok = ThreadComm::run(2, |comm| {
            let g = DistGraph {
                vtxdist: vec![0, 2, 4],
                xadj: vec![0, 0, 0],
                adjncy: vec![],
                vwgt: None,
            };
            partition(&comm, &Block, &g).unwrap()
        });
        assert_eq!(ok, [vec![0, 0], vec![1, 1]]);
    }
}
