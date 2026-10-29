//! Test inputs and host references of the pipeline stages.

use std::collections::BTreeMap;

use gridcl_dims::{Cell2, Cell3, CellType};
use gridcl_geometry::NodeFamily;
use gridcl_testing::{HostChunk, HostMesh};

use super::BlockSpec;
use super::input::{BlockInput, Input, NodeRefs, TagInput};

/// An owned [`Input`].
pub(crate) struct OwnedInput {
    pub(crate) tdim: usize,
    pub(crate) gdim: usize,
    pub(crate) coords: Vec<f64>,
    pub(crate) node_ids: Option<Vec<i64>>,
    pub(crate) blocks: Vec<(BlockSpec, NodeRefs, Option<Vec<i64>>)>,
    pub(crate) tags: Vec<TagInput>,
}

impl OwnedInput {
    /// The input of a chunk: cells by node id, with the chunk's tags. The cell dimension comes
    /// from the first block (1 for a chunk without blocks).
    pub(crate) fn of_chunk(c: &HostChunk) -> OwnedInput {
        OwnedInput {
            tdim: c.blocks.first().map_or(1, |b| gridcl_refcell::tdim(b.cell)),
            gdim: c.gdim,
            coords: c.nodes.clone(),
            node_ids: Some(c.node_ids.clone()),
            blocks: c
                .blocks
                .iter()
                .map(|b| {
                    let spec = BlockSpec {
                        cell: b.cell,
                        degree: b.degree,
                        family: NodeFamily::Equispaced,
                    };
                    (
                        spec,
                        NodeRefs::Ids(b.node_ids.clone()),
                        Some(b.cell_ids.clone()),
                    )
                })
                .collect(),
            tags: c
                .tags
                .iter()
                .map(|t| TagInput {
                    entity: t.entity,
                    corners: t.corners.clone(),
                    values: t.values.clone(),
                })
                .collect(),
        }
    }

    /// A block of P1 intervals.
    pub(crate) fn interval_block(
        nodes: NodeRefs,
        ids: Option<Vec<i64>>,
    ) -> (BlockSpec, NodeRefs, Option<Vec<i64>>) {
        let spec = BlockSpec {
            cell: CellType::Interval,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        (spec, nodes, ids)
    }

    /// The borrowed input.
    pub(crate) fn input(&self) -> Input<'_> {
        Input {
            tdim: self.tdim,
            gdim: self.gdim,
            coords: &self.coords,
            node_ids: self.node_ids.as_deref(),
            blocks: self
                .blocks
                .iter()
                .map(|(spec, nodes, ids)| BlockInput {
                    spec: *spec,
                    nodes,
                    origin_ids: ids.as_deref(),
                })
                .collect(),
            tags: &self.tags,
        }
    }
}

/// The test meshes of the stage tests: several cell types, dimensions and degrees, with
/// shuffled ids in one.
pub(crate) fn meshes() -> Vec<(&'static str, HostMesh)> {
    vec![
        ("interval", gridcl_testing::interval_box(7, 1)),
        (
            "triangles p2",
            gridcl_testing::square_box(Cell2::Triangle, 3, 2),
        ),
        (
            "quadrilaterals",
            gridcl_testing::square_box(Cell2::Quadrilateral, 3, 1),
        ),
        ("mixed", gridcl_testing::mixed_box(2, 1)),
        (
            "tetrahedra, shuffled ids",
            gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1).with_shuffled_ids(11),
        ),
        ("sphere", gridcl_testing::sphere(1, 1)),
    ]
}

/// Every cell of `h` by origin id: its type, node origin ids and sorted distinct cell tag
/// values.
pub(crate) fn host_cells(h: &HostMesh) -> BTreeMap<i64, (CellType, Vec<i64>, Vec<i32>)> {
    let mut tags: BTreeMap<Vec<i64>, Vec<i32>> = BTreeMap::new();
    for t in h.physical_tags() {
        let nv = gridcl_refcell::vertices(t.entity).len();
        for (corners, &v) in t.corners.chunks_exact(nv).zip(&t.values) {
            let mut key = corners.to_vec();
            key.sort_unstable();
            tags.entry(key).or_default().push(v);
        }
    }
    let mut out = BTreeMap::new();
    for (block, ids) in h.blocks.iter().zip(h.cell_origin_ids()) {
        let nv = gridcl_refcell::vertices(block.cell).len();
        for (row, id) in block.cells.chunks_exact(block.nodes_per_cell()).zip(ids) {
            let nodes: Vec<i64> = row.iter().map(|&n| h.node_id(n as usize)).collect();
            let mut key = nodes[..nv].to_vec();
            key.sort_unstable();
            let mut values = tags.get(&key).cloned().unwrap_or_default();
            values.sort_unstable();
            values.dedup();
            out.insert(id, (block.cell, nodes, values));
        }
    }
    out
}

/// The sorted vertex keys of the facets of every cell of `h`, by cell origin id.
pub(crate) fn host_facets(h: &HostMesh) -> BTreeMap<i64, Vec<Vec<i64>>> {
    host_keys(h, true)
}

/// The vertex origin ids of every cell of `h`, by cell origin id, each as a key of one.
pub(crate) fn host_vertices(h: &HostMesh) -> BTreeMap<i64, Vec<Vec<i64>>> {
    host_keys(h, false)
}

/// Facet keys or single-vertex keys of every cell of `h`.
fn host_keys(h: &HostMesh, facets: bool) -> BTreeMap<i64, Vec<Vec<i64>>> {
    let mut out = BTreeMap::new();
    for (block, ids) in h.blocks.iter().zip(h.cell_origin_ids()) {
        let nv = gridcl_refcell::vertices(block.cell).len();
        let tdim = gridcl_refcell::tdim(block.cell);
        for (row, id) in block.cells.chunks_exact(block.nodes_per_cell()).zip(ids) {
            let v: Vec<i64> = row[..nv].iter().map(|&n| h.node_id(n as usize)).collect();
            let keys = if facets {
                gridcl_refcell::sub_entities(block.cell, tdim - 1)
                    .iter()
                    .map(|local| {
                        let mut key: Vec<i64> = local.iter().map(|&j| v[j]).collect();
                        key.sort_unstable();
                        key
                    })
                    .collect()
            } else {
                v.iter().map(|&x| vec![x]).collect()
            };
            out.insert(id, keys);
        }
    }
    out
}

/// The cells adjacent to each cell of `keys` through a shared key, by origin id.
pub(crate) fn host_adjacency(keys: &BTreeMap<i64, Vec<Vec<i64>>>) -> BTreeMap<i64, Vec<i64>> {
    let mut by_key: BTreeMap<&[i64], Vec<i64>> = BTreeMap::new();
    for (&id, ks) in keys {
        for k in ks {
            by_key.entry(k).or_default().push(id);
        }
    }
    let mut out: BTreeMap<i64, Vec<i64>> = keys.keys().map(|&id| (id, Vec::new())).collect();
    for ids in by_key.values() {
        for &a in ids {
            out.get_mut(&a)
                .expect("a cell")
                .extend(ids.iter().filter(|&&b| b != a));
        }
    }
    for v in out.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    out
}
