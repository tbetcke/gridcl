//! Distributed input for the M6 tests: generated physical tags, chunks of a [`HostMesh`] in the
//! form `MeshBuilder::build_on` takes (cells reference nodes by origin id), and shuffled cells.

use std::collections::HashMap;

use cubecl::prelude::Client;
use gridcl::{CoordinateElement, Dim, EmbeddedIn, MeshBuilder, NodeFamily, Real, Result, TopoDim};
use gridcl_dims::CellType;

use super::lattice::SplitMix64;
use super::{HostMesh, check_dims};

/// Physical tags of one entity type in the form `MeshBuilder::physical_tags` takes: one entry
/// per tagged entity, keyed by the origin ids of its corner vertices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostTags {
    /// The entity type.
    pub entity: CellType,
    /// `entity`'s vertex count of corner origin ids per entry.
    pub corners: Vec<i64>,
    /// The tag value of each entry.
    pub values: Vec<i32>,
}

/// One rank's share of a [`HostMesh`] (see [`HostMesh::chunk`]).
#[derive(Clone, Debug, PartialEq)]
pub struct HostChunk {
    /// The geometric dimension.
    pub gdim: usize,
    /// Node coordinates, `[node][gdim]`.
    pub nodes: Vec<f64>,
    /// Node origin ids.
    pub node_ids: Vec<i64>,
    /// The cell blocks with at least one cell in this chunk.
    pub blocks: Vec<HostChunkBlock>,
    /// The tag entries of this chunk, one list per entity type with entries here.
    pub tags: Vec<HostTags>,
    /// The tag name table (name, dimension, value).
    pub tag_names: Vec<(String, usize, i32)>,
}

/// The cells of one cell block in a [`HostChunk`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostChunkBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// Cell → node origin ids in A.5 order, one row per cell.
    pub node_ids: Vec<i64>,
    /// Cell origin ids.
    pub cell_ids: Vec<i64>,
}

/// The cell tag values of [`HostMesh::physical_tags`]: every cell has one of `CELL_VALUES`, and
/// every fourth cell also `CELL_EXTRA`.
const CELL_VALUES: [i32; 3] = [1, 2, 3];
const CELL_EXTRA: i32 = 7;
/// The boundary facet tag values, alternating in order of first appearance.
const FACET_VALUES: [i32; 2] = [100, 101];

/// The range `[rank·n/size, (rank+1)·n/size)`.
fn share(n: usize, rank: usize, size: usize) -> core::ops::Range<usize> {
    rank * n / size..(rank + 1) * n / size
}

impl HostMesh {
    /// The origin id of every cell over all blocks, in block order: `cell_ids`, or the cell's
    /// index over all blocks (the default of `MeshBuilder`).
    pub fn cell_origin_ids(&self) -> Vec<Vec<i64>> {
        let mut first = 0;
        self.blocks
            .iter()
            .map(|block| {
                let n = block.num_cells();
                let ids = match &block.cell_ids {
                    Some(ids) => ids.clone(),
                    None => (first..first + n).map(|i| i as i64).collect(),
                };
                first += n;
                ids
            })
            .collect()
    }

    /// Generated physical tags: every cell gets 1, 2 or 3 by its index over all blocks, every
    /// fourth cell also 7, and the boundary facets (those of exactly one cell) 100 and 101
    /// alternately, in order of first appearance. One list per entity type, cells first.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, CellType};
    ///
    /// let tags = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).physical_tags();
    /// assert_eq!(tags[0].entity, CellType::Quadrilateral);
    /// // Four cells, cells 0 and 4k twice.
    /// assert_eq!(tags[0].values, [1, 7, 2, 3, 1]);
    /// // Eight boundary edges.
    /// assert_eq!((tags[1].entity, tags[1].values.len()), (CellType::Interval, 8));
    /// ```
    pub fn physical_tags(&self) -> Vec<HostTags> {
        let mut lists: Vec<HostTags> = Vec::new();
        let list_of = |lists: &mut Vec<HostTags>, entity: CellType| match lists
            .iter()
            .position(|t| t.entity == entity)
        {
            Some(i) => i,
            None => {
                lists.push(HostTags {
                    entity,
                    corners: Vec::new(),
                    values: Vec::new(),
                });
                lists.len() - 1
            }
        };

        // Cells.
        let mut g = 0;
        for block in &self.blocks {
            let nv = gridcl_refcell::vertices(block.cell).len();
            for row in block.cells.chunks_exact(block.nodes_per_cell()) {
                let corners: Vec<i64> = row[..nv]
                    .iter()
                    .map(|&n| self.node_id(n as usize))
                    .collect();
                let mut values = vec![CELL_VALUES[g % CELL_VALUES.len()]];
                if g % 4 == 0 {
                    values.push(CELL_EXTRA);
                }
                let i = list_of(&mut lists, block.cell);
                for v in values {
                    lists[i].corners.extend_from_slice(&corners);
                    lists[i].values.push(v);
                }
                g += 1;
            }
        }

        // Boundary facets: count every facet by its sorted corner ids.
        let mut facets: Vec<(CellType, Vec<i64>)> = Vec::new();
        let mut count: HashMap<Vec<i64>, (usize, usize)> = HashMap::new();
        for block in &self.blocks {
            let tdim = gridcl_refcell::tdim(block.cell);
            if tdim == 0 {
                continue;
            }
            let subs = gridcl_refcell::sub_entities(block.cell, tdim - 1);
            let types = gridcl_refcell::sub_entity_types(block.cell, tdim - 1);
            for row in block.cells.chunks_exact(block.nodes_per_cell()) {
                for (local, &ty) in subs.iter().zip(types) {
                    let corners: Vec<i64> = local
                        .iter()
                        .map(|&v| self.node_id(row[v] as usize))
                        .collect();
                    let mut key = corners.clone();
                    key.sort_unstable();
                    let entry = count.entry(key).or_insert((facets.len(), 0));
                    if entry.1 == 0 {
                        facets.push((ty, corners));
                    }
                    entry.1 += 1;
                }
            }
        }
        let mut boundary: Vec<bool> = vec![false; facets.len()];
        for &(first, n) in count.values() {
            boundary[first] = n == 1;
        }
        let mut k = 0;
        for ((ty, corners), on_boundary) in facets.into_iter().zip(boundary) {
            if on_boundary {
                let i = list_of(&mut lists, ty);
                lists[i].corners.extend_from_slice(&corners);
                lists[i].values.push(FACET_VALUES[k % FACET_VALUES.len()]);
                k += 1;
            }
        }
        lists
    }

    /// The name table of [`physical_tags`](HostMesh::physical_tags): the cell values in the
    /// cells' dimension and the facet values one below (none for a mesh without cells).
    pub fn tag_names(&self) -> Vec<(String, usize, i32)> {
        let Some(block) = self.blocks.first() else {
            return Vec::new();
        };
        let tdim = gridcl_refcell::tdim(block.cell);
        let mut names: Vec<(String, usize, i32)> = CELL_VALUES
            .iter()
            .chain([&CELL_EXTRA])
            .map(|&v| (format!("cells-{v}"), tdim, v))
            .collect();
        if tdim > 0 {
            names.extend(
                FACET_VALUES
                    .iter()
                    .map(|&v| (format!("boundary-{v}"), tdim - 1, v)),
            );
        }
        names
    }

    /// Rank `rank`'s chunk of `size` even chunks: the cells `[rank·N/size, (rank+1)·N/size)` over
    /// all blocks in order, the nodes `[rank·M/size, (rank+1)·M/size)` and the same share of
    /// each list of [`physical_tags`](HostMesh::physical_tags). Cells reference nodes by origin
    /// id, so they may use nodes of other chunks; cell ids are materialised
    /// ([`cell_origin_ids`](HostMesh::cell_origin_ids)). Every chunk has the full name table.
    ///
    /// # Panics
    ///
    /// If `rank >= size`.
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::interval_box(4, 1);
    /// let c = h.chunk(1, 2);
    /// // Cells [2, 4) of 4 and nodes [⌊5/2⌋, 5) = [2, 5) of 5.
    /// assert_eq!(c.blocks[0].cell_ids, [2, 3]);
    /// assert_eq!(c.blocks[0].node_ids, [2, 3, 3, 4]);
    /// assert_eq!(c.node_ids, [2, 3, 4]);
    /// ```
    pub fn chunk(&self, rank: usize, size: usize) -> HostChunk {
        assert!(rank < size, "rank {rank} of {size}");
        let gd = self.gdim;
        let ids = self.cell_origin_ids();
        let cells = share(self.num_cells(), rank, size);
        let mut blocks = Vec::new();
        let mut first = 0;
        for (block, ids) in self.blocks.iter().zip(&ids) {
            let n = block.num_cells();
            let k = block.nodes_per_cell();
            let mine = cells.start.max(first).min(first + n) - first
                ..cells.end.max(first).min(first + n) - first;
            first += n;
            if mine.is_empty() {
                continue;
            }
            blocks.push(HostChunkBlock {
                cell: block.cell,
                degree: block.degree,
                node_ids: block.cells[mine.start * k..mine.end * k]
                    .iter()
                    .map(|&n| self.node_id(n as usize))
                    .collect(),
                cell_ids: ids[mine].to_vec(),
            });
        }
        let nodes = share(self.num_nodes(), rank, size);
        let tags = self
            .physical_tags()
            .into_iter()
            .filter_map(|t| {
                let nc = gridcl_refcell::vertices(t.entity).len();
                let mine = share(t.values.len(), rank, size);
                (!mine.is_empty()).then(|| HostTags {
                    entity: t.entity,
                    corners: t.corners[mine.start * nc..mine.end * nc].to_vec(),
                    values: t.values[mine].to_vec(),
                })
            })
            .collect();
        HostChunk {
            gdim: gd,
            nodes: self.nodes[nodes.start * gd..nodes.end * gd].to_vec(),
            node_ids: nodes.map(|i| self.node_id(i)).collect(),
            blocks,
            tags,
            tag_names: self.tag_names(),
        }
    }

    /// The mesh with the cells of every block in a random order seeded by `seed`; every cell
    /// keeps its nodes and origin id (cell ids are materialised first).
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::interval_box(8, 1);
    /// let s = h.with_shuffled_cells(5);
    /// assert_ne!(s.blocks[0].cells, h.blocks[0].cells);
    /// // Cell i still spans nodes i and i + 1.
    /// let ids = s.blocks[0].cell_ids.as_ref().unwrap();
    /// for (row, &id) in s.blocks[0].cells.chunks(2).zip(ids) {
    ///     assert_eq!(row, [id as u32, id as u32 + 1]);
    /// }
    /// ```
    pub fn with_shuffled_cells(&self, seed: u64) -> HostMesh {
        let mut rng = SplitMix64::new(seed);
        let ids = self.cell_origin_ids();
        let mut out = self.clone();
        for (block, ids) in out.blocks.iter_mut().zip(ids) {
            let n = block.num_cells();
            let k = block.nodes_per_cell();
            // perm[new] = old (Fisher–Yates).
            let mut perm: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.below(i as u64 + 1) as usize;
                perm.swap(i, j);
            }
            let cells = block.cells.clone();
            block.cells = perm
                .iter()
                .flat_map(|&old| cells[old * k..(old + 1) * k].iter().copied())
                .collect();
            block.cell_ids = Some(perm.iter().map(|&old| ids[old]).collect());
        }
        out
    }
}

impl HostChunk {
    /// A chunk without nodes, cells, tags or names.
    pub fn empty(gdim: usize) -> HostChunk {
        HostChunk {
            gdim,
            nodes: Vec::new(),
            node_ids: Vec::new(),
            blocks: Vec::new(),
            tags: Vec::new(),
            tag_names: Vec::new(),
        }
    }

    /// A `MeshBuilder` with this chunk's nodes, cells (by node origin id), tags and names, with
    /// equispaced coordinate elements; the caller sets the options and calls `build_on`.
    ///
    /// # Errors
    ///
    /// As `HostMesh::build`'s dimension checks.
    pub fn builder<F, TD, GD>(&self, client: &Client) -> Result<MeshBuilder<F, TD, GD>>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        check_dims::<TD, GD>(self.gdim)?;
        let mut builder =
            MeshBuilder::<F, TD, GD>::new(client).nodes(&self.nodes, Some(&self.node_ids));
        for block in &self.blocks {
            let element = CoordinateElement {
                cell: super::td_cell::<TD>(block.cell)?,
                degree: block.degree,
                family: NodeFamily::Equispaced,
            };
            builder = builder.cell_block_by_ids(element, &block.node_ids, Some(&block.cell_ids));
        }
        for t in &self.tags {
            builder = builder.physical_tags(t.entity, &t.corners, &t.values);
        }
        let names: Vec<(&str, usize, i32)> = self
            .tag_names
            .iter()
            .map(|(name, dim, value)| (name.as_str(), *dim, *value))
            .collect();
        Ok(builder.tag_names(&names))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_dims::{Cell2, Cell3};

    /// The chunks of every rank count hold each cell, node and tag entry exactly once.
    #[test]
    fn chunks_cover_the_mesh() {
        for h in [
            crate::mixed_box(2, 2),
            crate::square_box(Cell2::Triangle, 3, 1),
            crate::cube_box(Cell3::Prism, 1, 1).with_shuffled_ids(3),
        ] {
            let ids = h.cell_origin_ids();
            let tags = h.physical_tags();
            for size in 1..=7 {
                let chunks: Vec<HostChunk> = (0..size).map(|r| h.chunk(r, size)).collect();
                let node_ids: Vec<i64> = chunks.iter().flat_map(|c| c.node_ids.clone()).collect();
                assert_eq!(
                    node_ids,
                    (0..h.num_nodes()).map(|i| h.node_id(i)).collect::<Vec<_>>()
                );
                for (b, block) in h.blocks.iter().enumerate() {
                    let rows: Vec<i64> = chunks
                        .iter()
                        .flat_map(|c| c.blocks.iter().filter(|x| x.cell == block.cell))
                        .flat_map(|x| x.node_ids.clone())
                        .collect();
                    let want: Vec<i64> =
                        block.cells.iter().map(|&n| h.node_id(n as usize)).collect();
                    assert_eq!(rows, want, "block {b}, {size} chunks");
                    let cells: Vec<i64> = chunks
                        .iter()
                        .flat_map(|c| c.blocks.iter().filter(|x| x.cell == block.cell))
                        .flat_map(|x| x.cell_ids.clone())
                        .collect();
                    assert_eq!(cells, ids[b]);
                }
                for t in &tags {
                    let values: Vec<i32> = chunks
                        .iter()
                        .flat_map(|c| c.tags.iter().filter(|x| x.entity == t.entity))
                        .flat_map(|x| x.values.clone())
                        .collect();
                    assert_eq!(values, t.values);
                }
                assert!(chunks.iter().all(|c| c.tag_names == h.tag_names()));
            }
        }
    }

    /// Boundary facets of a 2 × 2 × 2 hexahedron box: 24 quadrilaterals.
    #[test]
    fn boundary_facets_of_a_box() {
        let tags = crate::cube_box(Cell3::Hexahedron, 2, 1).physical_tags();
        assert_eq!(tags.len(), 2);
        assert_eq!(
            (tags[1].entity, tags[1].values.len()),
            (CellType::Quadrilateral, 24)
        );
        assert_eq!(tags[1].corners.len(), 4 * 24);
        // 8 cells, cells 0 and 4 twice.
        assert_eq!(tags[0].values.len(), 10);
        assert!(
            crate::sphere(1, 1).physical_tags().len() == 1,
            "a closed surface has no boundary"
        );
    }
}
