//! The parallel creation pipeline (docs/design.md, "Parallel creation pipeline"; ADR-0026).
//!
//! Each stage is a host function over `&dyn CommDyn` from documented inputs to documented
//! outputs, and ends with an agreed status (ADR-0026, item 15): every rank returns `Err` when
//! any rank fails. Stages exchange variable-length i64 records ([`rendezvous`]).
//!
//! 1. [`input::read`]: the agreed cell blocks, default ids, the global duplicate check of cell
//!    ids, and cell tags matched to their cells.
//! 2. [`dual_graph::dual_graph`]: the facet dual graph of the input cells.
//! 3. [`dual_graph::partition`]: a destination rank per input cell.
//! 4. [`migrate::migrate`]: the cells move to their destinations.
//! 5. [`ghosts::ghost_cells`]: the ghost cells of the [`GhostMode`](crate::GhostMode).
//! 6. [`nodes::fetch_nodes`]: the coordinates of the local cells' nodes, and the agreed
//!    bounding box and origin.
//! 7. [`order::order_cells`]: the owned cells in the [`CellOrdering`](crate::CellOrdering).
//! 8. [`number::number`]: sharer sets, owners, and local and global numbers of the cells,
//!    vertices and nodes.

// Stages 1–8 are wired into `MeshBuilder::build_on` by M6 T12; until then only the tests call
// them.
#![allow(dead_code)]

pub(crate) mod dual_graph;
pub(crate) mod ghosts;
pub(crate) mod input;
pub(crate) mod migrate;
pub(crate) mod nodes;
pub(crate) mod number;
pub(crate) mod order;
pub(crate) mod rendezvous;

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_geometry::NodeFamily;
use gridcl_parallel::{CommDyn, CommExt};

/// The agreed result of a collective step (ADR-0026, item 15): a failing rank returns its own
/// error, every other rank `RemoteFailure` with the lowest failing rank. Collective.
pub(crate) fn agree<T>(comm: &dyn CommDyn, local: Result<T>) -> Result<T> {
    let lowest = comm.min(if local.is_err() {
        comm.rank() as u64
    } else {
        u64::MAX
    });
    match local {
        Ok(_) if lowest != u64::MAX => Err(GridError::RemoteFailure {
            rank: lowest as usize,
        }),
        other => other,
    }
}

/// The cell types in the order of their wire code.
const CELL_TYPES: [CellType; 8] = [
    CellType::Point,
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

/// The element of a cell block, without its dimension type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BlockSpec {
    pub(crate) cell: CellType,
    pub(crate) degree: u32,
    pub(crate) family: NodeFamily,
}

impl BlockSpec {
    /// Nodes per cell.
    pub(crate) fn num_nodes(&self) -> usize {
        gridcl_refcell::num_lagrange_nodes(self.cell, self.degree)
    }

    /// Vertices per cell: the first nodes of each row.
    pub(crate) fn num_vertices(&self) -> usize {
        gridcl_refcell::vertices(self.cell).len()
    }

    /// The three wire words (cell code, degree, family code).
    fn encode(&self) -> [i64; 3] {
        let cell = CELL_TYPES.iter().position(|&c| c == self.cell);
        let family = match self.family {
            NodeFamily::Equispaced => 0,
            NodeFamily::Gll => 1,
        };
        [
            cell.expect("every cell type has a code") as i64,
            i64::from(self.degree),
            family,
        ]
    }

    /// The spec of three wire words from [`encode`](BlockSpec::encode).
    fn decode(words: &[i64]) -> BlockSpec {
        BlockSpec {
            cell: CELL_TYPES[words[0] as usize],
            degree: words[1] as u32,
            family: if words[2] == 0 {
                NodeFamily::Equispaced
            } else {
                NodeFamily::Gll
            },
        }
    }
}

/// Cells on one rank as host arrays: per cell its block (an index into the agreed block list),
/// origin id, global input position (source rank, then position on it), node origin ids in
/// A.5 order and physical tag values (ascending, distinct).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Cells {
    pub(crate) block: Vec<u32>,
    pub(crate) ids: Vec<i64>,
    pub(crate) input: Vec<i64>,
    /// Row offsets of `nodes`, `len() + 1` entries from 0.
    pub(crate) node_offsets: Vec<usize>,
    pub(crate) nodes: Vec<i64>,
    /// Row offsets of `tags`, `len() + 1` entries from 0.
    pub(crate) tag_offsets: Vec<usize>,
    pub(crate) tags: Vec<i32>,
}

impl Cells {
    /// No cells.
    pub(crate) fn new() -> Cells {
        Cells {
            block: Vec::new(),
            ids: Vec::new(),
            input: Vec::new(),
            node_offsets: vec![0],
            nodes: Vec::new(),
            tag_offsets: vec![0],
            tags: Vec::new(),
        }
    }

    /// The number of cells.
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The node origin ids of cell `i`.
    pub(crate) fn nodes_of(&self, i: usize) -> &[i64] {
        &self.nodes[self.node_offsets[i]..self.node_offsets[i + 1]]
    }

    /// The vertex origin ids of cell `i`: the first nodes of its row.
    pub(crate) fn vertices_of(&self, i: usize, blocks: &[BlockSpec]) -> &[i64] {
        &self.nodes_of(i)[..blocks[self.block[i] as usize].num_vertices()]
    }

    /// The tag values of cell `i`.
    pub(crate) fn tags_of(&self, i: usize) -> &[i32] {
        &self.tags[self.tag_offsets[i]..self.tag_offsets[i + 1]]
    }

    /// Appends a cell.
    pub(crate) fn push(&mut self, block: u32, id: i64, input: i64, nodes: &[i64], tags: &[i32]) {
        self.block.push(block);
        self.ids.push(id);
        self.input.push(input);
        self.nodes.extend_from_slice(nodes);
        self.node_offsets.push(self.nodes.len());
        self.tags.extend_from_slice(tags);
        self.tag_offsets.push(self.tags.len());
    }

    /// Cell `i` as one wire record: [block, id, input, node count, nodes…, tags…].
    pub(crate) fn record(&self, i: usize) -> Vec<i64> {
        let nodes = self.nodes_of(i);
        let mut r = vec![
            i64::from(self.block[i]),
            self.ids[i],
            self.input[i],
            nodes.len() as i64,
        ];
        r.extend_from_slice(nodes);
        r.extend(self.tags_of(i).iter().map(|&t| i64::from(t)));
        r
    }

    /// Appends the cell of a wire record from [`record`](Cells::record).
    pub(crate) fn push_record(&mut self, r: &[i64]) {
        let n = r[3] as usize;
        let tags: Vec<i32> = r[4 + n..].iter().map(|&t| t as i32).collect();
        self.push(r[0] as u32, r[1], r[2], &r[4..4 + n], &tags);
    }

    /// The cells at `order`, in that order.
    pub(crate) fn select(&self, order: &[usize]) -> Cells {
        let mut out = Cells::new();
        for &i in order {
            out.push(
                self.block[i],
                self.ids[i],
                self.input[i],
                self.nodes_of(i),
                self.tags_of(i),
            );
        }
        out
    }

    /// Appends every cell of `other`.
    pub(crate) fn append(&mut self, other: &Cells) {
        for i in 0..other.len() {
            self.push(
                other.block[i],
                other.ids[i],
                other.input[i],
                other.nodes_of(i),
                other.tags_of(i),
            );
        }
    }
}

/// The sorted vertex origin ids of each facet of cell `i` (none for cells of dimension 0).
pub(crate) fn facet_keys(cells: &Cells, i: usize, blocks: &[BlockSpec]) -> Vec<Vec<i64>> {
    let cell = blocks[cells.block[i] as usize].cell;
    let tdim = gridcl_refcell::tdim(cell);
    if tdim == 0 {
        return Vec::new();
    }
    let vertices = cells.vertices_of(i, blocks);
    gridcl_refcell::sub_entities(cell, tdim - 1)
        .iter()
        .map(|local| {
            let mut key: Vec<i64> = local.iter().map(|&v| vertices[v]).collect();
            key.sort_unstable();
            key
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod testing;

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_testing::ThreadComm;

    #[test]
    fn agree_names_the_lowest_failing_rank() {
        let out = ThreadComm::run(4, |comm| {
            let local = if comm.rank() % 2 == 1 {
                Err(GridError::TooLarge)
            } else {
                Ok(comm.rank())
            };
            agree(&comm, local)
        });
        assert!(matches!(out[0], Err(GridError::RemoteFailure { rank: 1 })));
        assert!(matches!(out[1], Err(GridError::TooLarge)));
        assert!(matches!(out[2], Err(GridError::RemoteFailure { rank: 1 })));
        assert!(matches!(out[3], Err(GridError::TooLarge)));
        let ok = ThreadComm::run(3, |comm| agree(&comm, Ok(comm.rank())).unwrap());
        assert_eq!(ok, [0, 1, 2]);
    }

    #[test]
    fn block_specs_and_cells_round_trip() {
        for &cell in &CELL_TYPES {
            for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
                let spec = BlockSpec {
                    cell,
                    degree: 3,
                    family,
                };
                assert_eq!(BlockSpec::decode(&spec.encode()), spec);
            }
        }
        let mut cells = Cells::new();
        cells.push(1, -5, 7, &[3, 4, 5], &[2, 9]);
        cells.push(0, 8, 2, &[1, 2], &[]);
        let mut back = Cells::new();
        back.push_record(&cells.record(1));
        back.push_record(&cells.record(0));
        assert_eq!(back, cells.select(&[1, 0]));
        assert_eq!(back.tags_of(1), [2, 9]);
    }
}
