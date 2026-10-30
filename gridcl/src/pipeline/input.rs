//! Stage 1, input: the builder chunks of every rank become cells that reference nodes by
//! origin id (docs/design.md, "Parallel creation pipeline"; ADR-0026, items 1, 3 and 16).

use std::collections::{HashMap, HashSet};

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, CommExt};

use super::rendezvous::{Outbox, rendezvous_rank};
use super::{BlockSpec, Cells, agree};
use crate::builder::{cell_name, element_error, ids_or_index};

/// How the cells of a block reference their nodes.
pub(crate) enum NodeRefs {
    /// Indices into the rank's own node chunk (`MeshBuilder::cell_block`).
    Indices(Vec<u32>),
    /// Node origin ids (`MeshBuilder::cell_block_by_ids`).
    Ids(Vec<i64>),
}

impl NodeRefs {
    /// The number of entries.
    pub(crate) fn len(&self) -> usize {
        match self {
            NodeRefs::Indices(v) => v.len(),
            NodeRefs::Ids(v) => v.len(),
        }
    }
}

/// One cell block of the input: its element, cell → node rows and optional cell ids.
pub(crate) struct BlockInput<'a> {
    pub(crate) spec: BlockSpec,
    pub(crate) nodes: &'a NodeRefs,
    pub(crate) origin_ids: Option<&'a [i64]>,
}

/// One `MeshBuilder::physical_tags` call: `entity`'s vertex count of corner origin ids per
/// entry, and one value per entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TagInput {
    pub(crate) entity: CellType,
    pub(crate) corners: Vec<i64>,
    pub(crate) values: Vec<i32>,
}

/// One rank's input to the pipeline.
pub(crate) struct Input<'a> {
    /// The topological dimension of the cells.
    pub(crate) tdim: usize,
    /// The geometric dimension of the coordinates.
    pub(crate) gdim: usize,
    /// Node coordinates, `[node][gdim]`.
    pub(crate) coords: &'a [f64],
    /// Node origin ids; `None` for the default.
    pub(crate) node_ids: Option<&'a [i64]>,
    /// The cell blocks in call order.
    pub(crate) blocks: Vec<BlockInput<'a>>,
    /// The physical tags in call order.
    pub(crate) tags: &'a [TagInput],
}

/// A rank's node chunk with origin ids.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NodeChunk {
    pub(crate) ids: Vec<i64>,
    /// `[node][gdim]`.
    pub(crate) coords: Vec<f64>,
}

/// The tag entries of one `physical_tags` call of dimension below the cells': matched to
/// entities after they are created (M6 T15). `first` is the global input position of the
/// first entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LowerTags {
    pub(crate) tags: TagInput,
    pub(crate) first: i64,
}

/// The output of stage 1.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Read {
    /// The agreed cell blocks: the union over ranks, in order of first declaration by rank.
    pub(crate) blocks: Vec<BlockSpec>,
    /// This rank's nodes.
    pub(crate) nodes: NodeChunk,
    /// This rank's cells in input order, with their cell tags.
    pub(crate) cells: Cells,
    /// This rank's tags of lower dimension.
    pub(crate) lower_tags: Vec<LowerTags>,
}

/// Wire tags of the cell-tag rendezvous.
const CELL: i64 = 0;
const TAG: i64 = 1;

/// Stage 1. Collective.
///
/// - Agrees on the cell blocks.
/// - Gives default ids: an exscan of the node and cell counts plus the local index (cells
///   count over the blocks in call order).
/// - Translates `NodeRefs::Indices` to node origin ids.
/// - Checks cell ids for global duplicates by a rendezvous on `owner_hash(&[id])`.
/// - Matches cell tags to cells by a rendezvous on the sorted vertex key.
///
/// # Errors
///
/// Agreed over the ranks: `LengthMismatch` and `IndexOutOfRange` as `MeshBuilder::build`,
/// and for tag corners that are not `values.len()` rows; `DuplicateCellBlock` for one cell
/// type twice on a rank or in two elements over the ranks; `UnsupportedNodeFamily` and
/// `UnsupportedDegree`; `DuplicateOriginId` for a cell id given twice (the first repeat on
/// its rendezvous rank); `UnmatchedTagElement` for a tag of the cell dimension (or above)
/// that matches no cell, `tag` being the entry's global input position.
pub(crate) fn read(comm: &dyn CommDyn, input: &Input<'_>) -> Result<Read> {
    agree(comm, check(input))?;
    let (blocks, block_of) = agree(comm, union_blocks(comm, &input.blocks))?;

    let num_nodes = input.coords.len() / input.gdim;
    let num_cells: usize = input
        .blocks
        .iter()
        .map(|b| b.nodes.len() / b.spec.num_nodes())
        .sum();
    let num_tags: usize = input.tags.iter().map(|t| t.values.len()).sum();
    let node_first = comm.exscan_sum(num_nodes as u64) as usize;
    let cell_first = comm.exscan_sum(num_cells as u64) as usize;
    let tag_first = comm.exscan_sum(num_tags as u64) as i64;

    let (nodes, mut cells) = agree(comm, chunk(input, &block_of, node_first, cell_first))?;
    check_cell_ids(comm, &cells)?;
    let lower_tags = match_cell_tags(comm, input, &blocks, &mut cells, tag_first)?;
    Ok(Read {
        blocks,
        nodes,
        cells,
        lower_tags,
    })
}

/// The local checks of the input, in `MeshBuilder::build`'s order.
fn check(input: &Input<'_>) -> Result<()> {
    let gd = input.gdim;
    if !input.coords.len().is_multiple_of(gd) {
        return Err(GridError::LengthMismatch {
            expected: input.coords.len().next_multiple_of(gd),
            found: input.coords.len(),
        });
    }
    let num_nodes = input.coords.len() / gd;
    if let Some(ids) = input.node_ids
        && ids.len() != num_nodes
    {
        return Err(GridError::LengthMismatch {
            expected: num_nodes,
            found: ids.len(),
        });
    }
    let mut seen = HashSet::new();
    for b in &input.blocks {
        let spec = b.spec;
        if !seen.insert(spec.cell) {
            return Err(GridError::DuplicateCellBlock(cell_name(spec.cell)));
        }
        gridcl_refcell::lagrange_points(spec.cell, spec.degree, spec.family)
            .map_err(element_error)?;
        let k = spec.num_nodes();
        let len = b.nodes.len();
        if !len.is_multiple_of(k) {
            return Err(GridError::LengthMismatch {
                expected: len.next_multiple_of(k),
                found: len,
            });
        }
        if let NodeRefs::Indices(nodes) = b.nodes
            && let Some(&bad) = nodes.iter().find(|&&n| n as usize >= num_nodes)
        {
            return Err(GridError::IndexOutOfRange {
                index: bad as usize,
                len: num_nodes,
            });
        }
        if let Some(ids) = b.origin_ids
            && ids.len() != len / k
        {
            return Err(GridError::LengthMismatch {
                expected: len / k,
                found: ids.len(),
            });
        }
    }
    for t in input.tags {
        let nv = gridcl_refcell::vertices(t.entity).len();
        if t.corners.len() != t.values.len() * nv {
            return Err(GridError::LengthMismatch {
                expected: t.values.len() * nv,
                found: t.corners.len(),
            });
        }
    }
    Ok(())
}

/// The union of the blocks of every rank, in order of first declaration by rank, and the
/// position of each local block in it. Collective; the error is the same on every rank.
fn union_blocks(
    comm: &dyn CommDyn,
    blocks: &[BlockInput<'_>],
) -> Result<(Vec<BlockSpec>, Vec<u32>)> {
    let p = comm.size();
    let words: Vec<i64> = blocks.iter().flat_map(|b| b.spec.encode()).collect();
    let send: Vec<i64> = (0..p).flat_map(|_| words.iter().copied()).collect();
    let (all, _) = comm.all_to_all_v(&send, &vec![words.len(); p])?;
    let mut union: Vec<BlockSpec> = Vec::new();
    for w in all.as_chunks::<3>().0 {
        let spec = BlockSpec::decode(w);
        match union.iter().find(|u| u.cell == spec.cell) {
            Some(u) if *u != spec => {
                return Err(GridError::DuplicateCellBlock(cell_name(spec.cell)));
            }
            Some(_) => {}
            None => union.push(spec),
        }
    }
    let block_of = blocks
        .iter()
        .map(|b| {
            union
                .iter()
                .position(|u| *u == b.spec)
                .expect("in the union") as u32
        })
        .collect();
    Ok((union, block_of))
}

/// The node chunk and the cells of this rank, with default ids from `node_first` and
/// `cell_first`.
fn chunk(
    input: &Input<'_>,
    block_of: &[u32],
    node_first: usize,
    cell_first: usize,
) -> Result<(NodeChunk, Cells)> {
    let num_nodes = input.coords.len() / input.gdim;
    let node_ids = ids_or_index(input.node_ids, num_nodes, node_first)?;
    let mut cells = Cells::new();
    let mut row = Vec::new();
    for (b, &block) in input.blocks.iter().zip(block_of) {
        let k = b.spec.num_nodes();
        let n = b.nodes.len() / k;
        let first = cell_first + cells.len();
        let ids = ids_or_index(b.origin_ids, n, first)?;
        for (c, &id) in ids.iter().enumerate() {
            row.clear();
            match b.nodes {
                NodeRefs::Indices(v) => {
                    row.extend(v[c * k..(c + 1) * k].iter().map(|&x| node_ids[x as usize]));
                }
                NodeRefs::Ids(v) => row.extend_from_slice(&v[c * k..(c + 1) * k]),
            }
            let position = i64::try_from(first + c).map_err(|_| GridError::TooLarge)?;
            cells.push(block, id, position, &row, &[]);
        }
    }
    let nodes = NodeChunk {
        ids: node_ids,
        coords: input.coords.to_vec(),
    };
    Ok((nodes, cells))
}

/// `DuplicateOriginId` (agreed) for a cell id that two cells share: every id goes to its
/// rendezvous rank, which reports the first repeat in arrival order. Collective.
fn check_cell_ids(comm: &dyn CommDyn, cells: &Cells) -> Result<()> {
    let p = comm.size();
    let mut out = Outbox::new(p);
    for &id in &cells.ids {
        out.push(rendezvous_rank(&[id], p), &[&[id]]);
    }
    let inbox = out.exchange(comm)?;
    let mut seen = HashSet::new();
    let repeat = inbox
        .records()
        .map(|(_, r)| r[0])
        .find(|&id| !seen.insert(id));
    agree(
        comm,
        repeat.map_or(Ok(()), |id| Err(GridError::DuplicateOriginId(id))),
    )
}

/// Attaches the cell tags of `input` to `cells` and returns the tags of lower dimension.
/// Cells and tag entries meet at the rendezvous rank of their sorted vertex key, which sends
/// each match to the cell's rank. Collective.
fn match_cell_tags(
    comm: &dyn CommDyn,
    input: &Input<'_>,
    blocks: &[BlockSpec],
    cells: &mut Cells,
    tag_first: i64,
) -> Result<Vec<LowerTags>> {
    let p = comm.size();
    let mut out = Outbox::new(p);
    for i in 0..cells.len() {
        let mut key = cells.vertices_of(i, blocks).to_vec();
        key.sort_unstable();
        out.push(rendezvous_rank(&key, p), &[&[CELL, i as i64], &key]);
    }
    let mut lower = Vec::new();
    let mut unmatched: Option<(i64, usize)> = None;
    let mut position = tag_first;
    for t in input.tags {
        let nv = gridcl_refcell::vertices(t.entity).len();
        let dim = gridcl_refcell::tdim(t.entity);
        if dim == input.tdim {
            for (j, (corners, &value)) in t.corners.chunks_exact(nv).zip(&t.values).enumerate() {
                let mut key = corners.to_vec();
                key.sort_unstable();
                let at = position + j as i64;
                out.push(
                    rendezvous_rank(&key, p),
                    &[&[TAG, i64::from(value), at], &key],
                );
            }
        } else if dim < input.tdim {
            lower.push(LowerTags {
                tags: t.clone(),
                first: position,
            });
        } else if !t.values.is_empty() && unmatched.is_none() {
            unmatched = Some((position, dim));
        }
        position += t.values.len() as i64;
    }
    let inbox = out.exchange(comm)?;

    // Match in arrival order: the first unmatched entry has the lowest position here.
    let mut holders: HashMap<&[i64], Vec<(usize, i64)>> = HashMap::new();
    for (source, r) in inbox.records().filter(|(_, r)| r[0] == CELL) {
        holders.entry(&r[2..]).or_default().push((source, r[1]));
    }
    let mut replies = Outbox::new(p);
    for (_, r) in inbox.records().filter(|(_, r)| r[0] == TAG) {
        match holders.get(&r[3..]) {
            Some(cells) => {
                for &(rank, cell) in cells {
                    replies.push(rank, &[&[cell, r[1]]]);
                }
            }
            None => {
                if unmatched.is_none_or(|(at, _)| r[2] < at) {
                    unmatched = Some((r[2], input.tdim));
                }
            }
        }
    }
    agree(
        comm,
        unmatched.map_or(Ok(()), |(tag, dim)| {
            Err(GridError::UnmatchedTagElement { dim, tag })
        }),
    )?;

    let inbox = replies.exchange(comm)?;
    let mut values: Vec<Vec<i32>> = vec![Vec::new(); cells.len()];
    for (_, r) in inbox.records() {
        values[r[0] as usize].push(r[1] as i32);
    }
    cells.tags.clear();
    cells.tag_offsets.truncate(1);
    for mut v in values {
        v.sort_unstable();
        v.dedup();
        cells.tags.extend_from_slice(&v);
        cells.tag_offsets.push(cells.tags.len());
    }
    Ok(lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::testing::{OwnedInput, host_cells, meshes};
    use gridcl_testing::{HostChunk, ThreadComm};

    /// The cells of every rank together equal the whole mesh, with every cell's tags, for
    /// even chunks and for everything on rank 0.
    #[test]
    fn cells_and_tags_equal_the_host_mesh() {
        for (name, h) in meshes() {
            let want = host_cells(&h);
            for size in 1..=4 {
                for on_zero in [false, true] {
                    let reads = ThreadComm::run(size, |comm| {
                        let r = comm.rank();
                        let chunk = match (on_zero, r) {
                            (false, _) => h.chunk(r, size),
                            (true, 0) => h.chunk(0, 1),
                            (true, _) => HostChunk::empty(h.gdim),
                        };
                        read(&comm, &OwnedInput::of_chunk(&chunk).input()).unwrap()
                    });
                    let mut got = std::collections::BTreeMap::new();
                    let mut inputs = Vec::new();
                    for read in &reads {
                        assert_eq!(read.blocks, reads[0].blocks, "{name}");
                        let c = &read.cells;
                        for i in 0..c.len() {
                            let cell = read.blocks[c.block[i] as usize].cell;
                            let entry = (cell, c.nodes_of(i).to_vec(), c.tags_of(i).to_vec());
                            assert!(got.insert(c.ids[i], entry).is_none());
                            inputs.push(c.input[i]);
                        }
                    }
                    assert_eq!(got, want, "{name} on {size} ranks");
                    assert_eq!(inputs, (0..inputs.len() as i64).collect::<Vec<_>>());
                    let lower: usize = reads
                        .iter()
                        .flat_map(|r| &r.lower_tags)
                        .map(|t| t.tags.values.len())
                        .sum();
                    let tdim = gridcl_refcell::tdim(h.blocks[0].cell);
                    let facet_tags: usize = h
                        .physical_tags()
                        .iter()
                        .filter(|t| gridcl_refcell::tdim(t.entity) < tdim)
                        .map(|t| t.values.len())
                        .sum();
                    assert_eq!(lower, facet_tags, "{name}");
                }
            }
        }
    }

    /// Default ids are global construction indices: one rank's input split over three ranks
    /// gives the same ids.
    #[test]
    fn default_ids_count_over_ranks() {
        let reads = ThreadComm::run(3, |comm| {
            let r = comm.rank() as i64;
            // Rank r holds nodes 2r, 2r + 1 and one interval between them.
            let input = OwnedInput {
                tdim: 1,
                gdim: 1,
                coords: vec![2.0 * r as f64, 2.0 * r as f64 + 1.0],
                node_ids: None,
                blocks: vec![OwnedInput::interval_block(
                    NodeRefs::Indices(vec![0, 1]),
                    None,
                )],
                tags: vec![],
            };
            read(&comm, &input.input()).unwrap()
        });
        for (r, read) in reads.iter().enumerate() {
            let r = r as i64;
            assert_eq!(read.nodes.ids, [2 * r, 2 * r + 1]);
            assert_eq!(read.cells.ids, [r]);
            assert_eq!(read.cells.nodes, [2 * r, 2 * r + 1]);
        }
    }

    /// The input errors are agreed: the failing rank reports its error, the others
    /// `RemoteFailure`.
    #[test]
    fn errors_are_agreed() {
        let h = gridcl_testing::interval_box(6, 1);
        // Cell id 4 on ranks 1 and 2.
        let out = ThreadComm::run(3, |comm| {
            let mut input = OwnedInput::of_chunk(&h.chunk(comm.rank(), 3));
            if comm.rank() == 1 {
                input.blocks[0].2 = Some(vec![2, 4]);
            }
            read(&comm, &input.input()).map(|_| ())
        });
        let dup = out
            .iter()
            .position(|r| matches!(r, Err(GridError::DuplicateOriginId(4))));
        let dup = dup.expect("one rank reports the duplicate");
        for (r, res) in out.iter().enumerate().filter(|&(r, _)| r != dup) {
            assert!(
                matches!(res, Err(GridError::RemoteFailure { rank }) if *rank == dup),
                "rank {r}: {res:?}"
            );
        }

        // Rank 1 declares intervals of degree 2.
        let out = ThreadComm::run(2, |comm| {
            let mut input = OwnedInput::of_chunk(&h.chunk(comm.rank(), 2));
            if comm.rank() == 1 {
                input.blocks[0] = (
                    BlockSpec {
                        degree: 2,
                        ..input.blocks[0].0
                    },
                    NodeRefs::Ids(vec![0, 1, 2]),
                    None,
                );
            }
            read(&comm, &input.input()).map(|_| ())
        });
        assert!(
            out.iter()
                .all(|r| matches!(r, Err(GridError::DuplicateCellBlock("Interval")))),
            "{out:?}"
        );

        // A local length error on rank 1 only.
        let out = ThreadComm::run(3, |comm| {
            let mut input = OwnedInput::of_chunk(&h.chunk(comm.rank(), 3));
            if comm.rank() == 1 {
                input.blocks[0].1 = NodeRefs::Ids(vec![0, 1, 2]);
            }
            read(&comm, &input.input()).map(|_| ())
        });
        assert!(matches!(out[0], Err(GridError::RemoteFailure { rank: 1 })));
        assert!(matches!(
            out[1],
            Err(GridError::LengthMismatch {
                expected: 4,
                found: 3
            })
        ));
        assert!(matches!(out[2], Err(GridError::RemoteFailure { rank: 1 })));

        // An unmatched cell tag: 6 cells give 8 entries (cells 0 and 4 twice), and the entry
        // at global position 5 matches no cell.
        let out = ThreadComm::run(2, |comm| {
            let mut input = OwnedInput::of_chunk(&h.chunk(comm.rank(), 2));
            input.tags.truncate(1);
            if comm.rank() == 1 {
                // Rank 1 holds positions 4–7; its second entry gets the corners (0, 6).
                input.tags[0].corners[2..4].copy_from_slice(&[0, 6]);
            }
            read(&comm, &input.input()).map(|_| ())
        });
        let tag = |r: &Result<()>| match r {
            Err(GridError::UnmatchedTagElement { dim: 1, tag }) => Some(*tag),
            _ => None,
        };
        let failing: Vec<usize> = (0..2).filter(|&r| tag(&out[r]).is_some()).collect();
        assert_eq!(failing.len(), 1, "{out:?}");
        assert_eq!(tag(&out[failing[0]]), Some(5));
    }
}
