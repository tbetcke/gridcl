//! Stage 4: the cells move to their destination ranks (docs/design.md, "Parallel creation
//! pipeline").

use gridcl_device::Result;
use gridcl_parallel::CommDyn;

use super::Cells;
use super::rendezvous::Outbox;

/// Stage 4: sends cell `i` of `cells`, with its block, origin id, input position, nodes and
/// tags, to rank `dest[i]`, and returns the cells this rank owns, by source rank and then in
/// input order. Collective.
///
/// # Errors
///
/// Errors of the exchange.
pub(crate) fn migrate(comm: &dyn CommDyn, cells: &Cells, dest: &[i32]) -> Result<Cells> {
    let mut out = Outbox::new(comm.size());
    for (i, &d) in dest.iter().enumerate() {
        out.push(d as usize, &[&cells.record(i)]);
    }
    let inbox = out.exchange(comm)?;
    let mut owned = Cells::new();
    for (_, r) in inbox.records() {
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
                        let cells = migrate(&comm, &read.cells, &dest.unwrap()).unwrap();
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
