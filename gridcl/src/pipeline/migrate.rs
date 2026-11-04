//! Stage 4: the cells move to their destination ranks (docs/design.md, "Parallel creation
//! pipeline").

use gridcl_device::Result;
use gridcl_parallel::CommDyn;

use super::Cells;
use super::rendezvous::Outbox;

/// Stage 4: sends cell `i` of `cells`, with its block, origin id, input position, nodes and
/// tags, to rank `dest[i]`, and returns the cells this rank owns, by source rank and then in
/// input order. Cells whose destination is this rank stay in place, without a copy when no
/// lower rank sends any (the M8 memory bound, docs/plans/M8.md T5b). Collective.
///
/// # Errors
///
/// Errors of the exchange.
pub(crate) fn migrate(comm: &dyn CommDyn, mut cells: Cells, dest: &[i32]) -> Result<Cells> {
    let rank = comm.rank();
    let mut out = Outbox::new(comm.size());
    let mut stay = vec![false; cells.len()];
    for (i, &d) in dest.iter().enumerate() {
        if d as usize == rank {
            stay[i] = true;
        } else {
            out.push(d as usize, &[&cells.record(i)]);
        }
    }
    cells.retain(&stay);
    drop(stay);
    let inbox = out.exchange(comm)?;
    let lower = inbox.records().take_while(|&(s, _)| s < rank).count();
    // Exact room for the received cells (record layout of `Cells::record`).
    let (mut num, mut nodes, mut tags) = (0, 0, 0);
    for (_, r) in inbox.records() {
        let (n, e) = (r[3] as usize, r[4] as usize);
        (num, nodes, tags) = (num + 1, nodes + n, tags + r.len() - 5 - n - e);
    }
    let mut owned = if lower == 0 {
        cells.reserve(num, nodes, tags);
        cells
    } else {
        let mut owned = Cells::with_capacity(
            num + cells.len(),
            nodes + cells.nodes.len(),
            tags + cells.tags.len(),
            !cells.elementary.is_empty(),
        );
        for (_, r) in inbox.records().take(lower) {
            owned.push_record(r);
        }
        owned.append(&cells);
        owned
    };
    for (_, r) in inbox.records().skip(lower) {
        owned.push_record(r);
    }
    Ok(owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::dual_graph::{dual_graph, partition};
    use crate::pipeline::input::read;
    use crate::pipeline::testing::{OwnedInput, host_cells, meshes};
    use gridcl_partition::Block;
    use gridcl_testing::{ThreadComm, parallel::RoundRobin};

    /// After migration every cell is on the rank its partitioner chose, exactly once, with its
    /// nodes and tags, in input order.
    #[test]
    fn cells_reach_their_destinations() {
        for (name, h) in meshes() {
            let want = host_cells(&h);
            for size in 1..=4 {
                for round_robin in [false, true] {
                    let owned = ThreadComm::run(size, |comm| {
                        let input = OwnedInput::of_chunk(&h.chunk(comm.rank(), size));
                        let read = read(&comm, &input.input()).unwrap();
                        let g = dual_graph(&comm, &read.blocks, &read.cells).unwrap();
                        let dest = if round_robin {
                            partition(&comm, &RoundRobin, &g)
                        } else {
                            partition(&comm, &Block, &g)
                        };
                        let cells = migrate(&comm, read.cells, &dest.unwrap()).unwrap();
                        (read.blocks, cells)
                    });
                    let n = want.len() as i64;
                    let mut got = std::collections::BTreeMap::new();
                    for (rank, (blocks, cells)) in owned.iter().enumerate() {
                        assert!(cells.input.is_sorted(), "{name}");
                        for i in 0..cells.len() {
                            let g = cells.input[i];
                            let expected = if round_robin {
                                g % size as i64
                            } else {
                                g * size as i64 / n
                            };
                            assert_eq!(expected, rank as i64, "{name}: cell {g}");
                            let cell = blocks[cells.block[i] as usize].cell;
                            let entry =
                                (cell, cells.nodes_of(i).to_vec(), cells.tags_of(i).to_vec());
                            assert!(got.insert(cells.ids[i], entry).is_none());
                        }
                    }
                    assert_eq!(got, want, "{name} on {size} ranks");
                }
            }
        }
    }
}
