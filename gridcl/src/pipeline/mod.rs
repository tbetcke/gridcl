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
//! 9. [`upload::upload`]: index maps, and the device arrays of topology and geometry.
//!
//! [`tags::registry`] then matches the physical tags, creating the entities of each tagged
//! dimension (stage 10).
//!
//! [`MeshBuilder::build_on`](crate::MeshBuilder::build_on) runs the stages in order.

pub(crate) mod dual_graph;
pub(crate) mod ghosts;
pub(crate) mod input;
pub(crate) mod migrate;
pub(crate) mod nodes;
pub(crate) mod number;
pub(crate) mod order;
pub(crate) mod rendezvous;
pub(crate) mod tags;
pub(crate) mod upload;

use cubecl::MemoryScope;
use cubecl::prelude::Client;
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

/// The log target of [`log_memory`].
pub(crate) const MEMORY_LOG: &str = "gridcl::memory";

/// Logs the device memory pools after `stage` at debug level, target [`MEMORY_LOG`] (the M8
/// memory profile, docs/plans/M8.md T5a). The pools, a blocking device query, are read only
/// when that target is enabled.
pub(crate) fn log_memory(client: &Client, stage: &str) {
    if log::log_enabled!(target: MEMORY_LOG, log::Level::Debug) {
        let usage = client.memory_report(MemoryScope::Device).usage();
        log::debug!(
            target: MEMORY_LOG,
            "{stage}: pools in use {} B, reserved {} B",
            usage.bytes_in_use,
            usage.bytes_reserved
        );
    }
}

/// Gives the empty pages of the device pools back after `stage` (`Client::memory_cleanup`), so
/// the scratch of one step does not stay reserved beside the next (the M8 memory bound), and
/// logs the pools ([`log_memory`]). Best effort: CubeCL refuses only while a stream records a
/// graph, which mesh creation never does, and a refusal changes no result, so it is not an
/// error that one rank could see and the others not.
pub(crate) fn release_pools(client: &Client, stage: &str) {
    let _ = client.memory_cleanup();
    log_memory(client, stage);
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
/// A.5 order, physical tag values (ascending, distinct) and elementary tag value.
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
    /// The elementary tag of every cell, or empty when no cell has one (the Gmsh reader gives
    /// every cell one, `MeshBuilder` none; ADR-0035, item 8).
    pub(crate) elementary: Vec<i32>,
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
            elementary: Vec::new(),
        }
    }

    /// No cells, with room for `cells` cells of `nodes` node ids and `tags` tag values in all,
    /// with elementary tags if `elementary`.
    pub(crate) fn with_capacity(
        cells: usize,
        nodes: usize,
        tags: usize,
        elementary: bool,
    ) -> Cells {
        let mut out = Cells::new();
        out.block.reserve_exact(cells);
        out.ids.reserve_exact(cells);
        out.input.reserve_exact(cells);
        out.node_offsets.reserve_exact(cells);
        out.nodes.reserve_exact(nodes);
        out.tag_offsets.reserve_exact(cells);
        out.tags.reserve_exact(tags);
        if elementary {
            out.elementary.reserve_exact(cells);
        }
        out
    }

    /// Room for `cells` more cells of `nodes` node ids and `tags` tag values in all.
    pub(crate) fn reserve(&mut self, cells: usize, nodes: usize, tags: usize) {
        self.block.reserve_exact(cells);
        self.ids.reserve_exact(cells);
        self.input.reserve_exact(cells);
        self.node_offsets.reserve_exact(cells);
        self.nodes.reserve_exact(nodes);
        self.tag_offsets.reserve_exact(cells);
        self.tags.reserve_exact(tags);
        if !self.elementary.is_empty() {
            self.elementary.reserve_exact(cells);
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

    /// The elementary tag of cell `i`, if the cells have them.
    pub(crate) fn elementary_of(&self, i: usize) -> Option<i32> {
        self.elementary.get(i).copied()
    }

    /// Appends a cell. Either every cell has an elementary tag or none has.
    pub(crate) fn push(
        &mut self,
        block: u32,
        id: i64,
        input: i64,
        nodes: &[i64],
        tags: &[i32],
        elementary: Option<i32>,
    ) {
        match elementary {
            Some(e) => {
                debug_assert_eq!(
                    self.elementary.len(),
                    self.len(),
                    "elementary tags on all cells"
                );
                self.elementary.push(e);
            }
            None => debug_assert!(self.elementary.is_empty(), "elementary tags on all cells"),
        }
        self.block.push(block);
        self.ids.push(id);
        self.input.push(input);
        self.nodes.extend_from_slice(nodes);
        self.node_offsets.push(self.nodes.len());
        self.tags.extend_from_slice(tags);
        self.tag_offsets.push(self.tags.len());
    }

    /// Cell `i` as one wire record: [block, id, input, node count, elementary count (0 or 1),
    /// nodes…, elementary…, tags…].
    pub(crate) fn record(&self, i: usize) -> Vec<i64> {
        let nodes = self.nodes_of(i);
        let elementary = self.elementary_of(i);
        let mut r = vec![
            i64::from(self.block[i]),
            self.ids[i],
            self.input[i],
            nodes.len() as i64,
            i64::from(elementary.is_some()),
        ];
        r.extend_from_slice(nodes);
        r.extend(elementary.map(i64::from));
        r.extend(self.tags_of(i).iter().map(|&t| i64::from(t)));
        r
    }

    /// Appends the cell of a wire record from [`record`](Cells::record).
    pub(crate) fn push_record(&mut self, r: &[i64]) {
        let n = r[3] as usize;
        let e = r[4] as usize;
        let elementary = r[5 + n..5 + n + e].first().map(|&v| v as i32);
        let tags: Vec<i32> = r[5 + n + e..].iter().map(|&t| t as i32).collect();
        self.push(r[0] as u32, r[1], r[2], &r[5..5 + n], &tags, elementary);
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
                self.elementary_of(i),
            );
        }
        out
    }

    /// Keeps the cells `i` with `keep[i]`, in order, in place.
    pub(crate) fn retain(&mut self, keep: &[bool]) {
        debug_assert_eq!(keep.len(), self.len());
        let has_elementary = !self.elementary.is_empty();
        let (mut n, mut node_start, mut tag_start) = (0, 0, 0);
        for (i, &k) in keep.iter().enumerate() {
            let (node_end, tag_end) = (self.node_offsets[i + 1], self.tag_offsets[i + 1]);
            if k {
                let at = self.node_offsets[n];
                self.nodes.copy_within(node_start..node_end, at);
                self.node_offsets[n + 1] = at + node_end - node_start;
                let at = self.tag_offsets[n];
                self.tags.copy_within(tag_start..tag_end, at);
                self.tag_offsets[n + 1] = at + tag_end - tag_start;
                self.block[n] = self.block[i];
                self.ids[n] = self.ids[i];
                self.input[n] = self.input[i];
                if has_elementary {
                    self.elementary[n] = self.elementary[i];
                }
                n += 1;
            }
            (node_start, tag_start) = (node_end, tag_end);
        }
        self.block.truncate(n);
        self.ids.truncate(n);
        self.input.truncate(n);
        self.nodes.truncate(self.node_offsets[n]);
        self.node_offsets.truncate(n + 1);
        self.tags.truncate(self.tag_offsets[n]);
        self.tag_offsets.truncate(n + 1);
        if has_elementary {
            self.elementary.truncate(n);
        }
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
                other.elementary_of(i),
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
        cells.push(1, -5, 7, &[3, 4, 5], &[2, 9], None);
        cells.push(0, 8, 2, &[1, 2], &[], None);
        let mut back = Cells::new();
        back.push_record(&cells.record(1));
        back.push_record(&cells.record(0));
        assert_eq!(back, cells.select(&[1, 0]));
        assert_eq!(back.tags_of(1), [2, 9]);
        assert_eq!(back.elementary_of(1), None);

        // With elementary tags.
        let mut cells = Cells::new();
        cells.push(1, -5, 7, &[3, 4, 5], &[2, 9], Some(-3));
        cells.push(0, 8, 2, &[1, 2], &[], Some(4));
        let mut back = Cells::new();
        back.push_record(&cells.record(1));
        back.push_record(&cells.record(0));
        assert_eq!(back, cells.select(&[1, 0]));
        assert_eq!(back.tags_of(1), [2, 9]);
        assert_eq!(back.elementary, [4, -3]);
    }

    #[test]
    fn retain_equals_select() {
        for elementary in [false, true] {
            let mut cells = Cells::new();
            for i in 0..7_i64 {
                let nodes: Vec<i64> = (0..(i % 3 + 1)).map(|k| 10 * i + k).collect();
                let tags: Vec<i32> = (0..(i % 2) as i32).collect();
                cells.push(
                    i as u32,
                    -i,
                    2 * i,
                    &nodes,
                    &tags,
                    elementary.then_some(i as i32),
                );
            }
            for keep in [
                vec![true; 7],
                vec![false; 7],
                vec![false, true, true, false, false, true, false],
                vec![true, false, false, true, true, false, true],
            ] {
                let order: Vec<usize> = (0..7).filter(|&i| keep[i]).collect();
                let mut kept = cells.clone();
                kept.retain(&keep);
                assert_eq!(kept, cells.select(&order), "{keep:?}");
            }
        }
    }
}
