//! The steps of the conversion that the parallel read paths share and that
//! [`to_gmsh_mesh`](super::to_gmsh_mesh) is built from (ADR-0035, rules 1, 2, 4 and 5):
//!
//! - [`Layout`]: D_TOP, the cell blocks and the entities, from the small sections and the
//!   element block headers, which every rank has after a scan;
//! - [`Layout::cells`] and [`Layout::tag_groups`]: a chunk of D_TOP elements as rows of node
//!   tags in A.5 order with their element, elementary, physical and partition tags, and a
//!   chunk of lower-dimensional elements as tag groups;
//! - [`complete`]: the serendipity upgrade, which needs the whole file;
//! - [`node_range`] and [`element_range`]: the chunks of a file in memory, equal to the reads
//!   of [`MshIndex`](super::MshIndex);
//! - [`geometric_dim`]: D_GEO from the node chunks of all ranks.
//!
//! Only `geometric_dim` communicates. Duplicate tags, missing nodes and unmatched tag elements
//! are found by gridcl's builder (rule 3); on one file, by `to_gmsh_mesh`.
//!
//! These items are `#[doc(hidden)]`: helpers of gridcl's reader outside the stable API.

use std::collections::HashMap;
use std::ops::Range;

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, CommExt};

use super::convert::{GmshTagGroup, check_len, content_error, index_u32};
use super::element_type::cell_name;
use super::{
    ElementBlock, ElementChunk, MshFile, NodeBlock, NodeChunk, element_type, lagrange_element_type,
};

/// The header of one element block: what a scan knows of it.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockHeader {
    pub dim: usize,
    pub entity_tag: i32,
    pub element_type: i32,
    /// The number of elements.
    pub count: usize,
}

impl BlockHeader {
    /// The header of `block`.
    pub fn of(block: &ElementBlock) -> Self {
        BlockHeader {
            dim: block.dim,
            entity_tag: block.entity_tag,
            element_type: block.element_type,
            count: block.tags.len(),
        }
    }
}

/// What the elements of one entity carry: the elementary tag, the physical tags (sorted,
/// distinct), whether they belong to the serial mesh, and their partition (1-based; 0 for
/// none or not exactly one in 1–`numPartitions`).
#[derive(Clone, Debug)]
struct EntityInfo {
    elementary: i32,
    physical: Vec<i32>,
    keep: bool,
    partition: i32,
}

/// The sorted, distinct values of `tags`.
fn sorted(tags: &[i32]) -> Vec<i32> {
    let mut v = tags.to_vec();
    v.sort_unstable();
    v.dedup();
    v
}

/// The entities of `file` by (dimension, tag). A partitioned entity reports its parent's tag
/// as the elementary tag; one whose parent has a higher dimension lies on a partition
/// interface, and its elements are not part of the serial mesh.
fn entity_info(file: &MshFile) -> HashMap<(usize, i32), EntityInfo> {
    let mut info = HashMap::new();
    if let Some(e) = &file.entities {
        for (dim, list) in [&e.points, &e.curves, &e.surfaces, &e.volumes]
            .into_iter()
            .enumerate()
        {
            for x in list {
                info.insert(
                    (dim, x.tag),
                    EntityInfo {
                        elementary: x.tag,
                        physical: sorted(&x.physical_tags),
                        keep: true,
                        partition: 0,
                    },
                );
            }
        }
    }
    if let Some(p) = &file.partitioned_entities {
        let valid = 1..=i32::try_from(p.num_partitions).unwrap_or(i32::MAX);
        for (dim, list) in [&p.points, &p.curves, &p.surfaces, &p.volumes]
            .into_iter()
            .enumerate()
        {
            for x in list {
                let partition = match x.partitions[..] {
                    [q] if valid.contains(&q) => q,
                    _ => 0,
                };
                info.insert(
                    (dim, x.entity.tag),
                    EntityInfo {
                        elementary: x.parent_tag,
                        physical: sorted(&x.entity.physical_tags),
                        keep: x.parent_dim == dim,
                        partition,
                    },
                );
            }
        }
    }
    info
}

/// The layout of a file's mesh, from its small sections and element block headers:
/// D_TOP, the cell blocks in `CellType` order (empty blocks included) and the entities.
///
/// The cell blocks are those of the kept D_TOP element blocks, one degree per cell type, with
/// serendipity types as degree 2.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Layout {
    /// D_TOP: the highest dimension of a kept, non-empty element block.
    pub tdim: usize,
    /// The cell blocks, (cell type, complete degree), in `CellType` order.
    pub blocks: Vec<(CellType, u32)>,
    /// Whether the file has serendipity elements (types 16–19) of any dimension.
    pub serendipity: bool,
    /// `numPartitions` of `$PartitionedEntities`; 0 without that section.
    pub num_partitions: usize,
    info: HashMap<(usize, i32), EntityInfo>,
}

impl Layout {
    /// The layout of a file with the small sections of `sections` and the element blocks of
    /// `headers`, in file order.
    ///
    /// # Errors
    ///
    /// `UnsupportedElementType` for a type outside A.7; `Parse` for a type of another dimension
    /// than its block or a file without kept elements of dimension ≥ 1; `DuplicateCellBlock`
    /// for one cell type in two degrees.
    pub fn new(sections: &MshFile, headers: &[BlockHeader]) -> Result<Layout> {
        let info = entity_info(sections);
        let num_partitions = sections
            .partitioned_entities
            .as_ref()
            .map_or(0, |p| p.num_partitions);
        let mut serendipity = false;
        let mut kept = Vec::new();
        for h in headers {
            let ty = element_type(h.element_type)?;
            if gridcl_refcell::tdim(ty.cell) != h.dim {
                return Err(content_error(format!(
                    "element type {} in an entity of dimension {}",
                    h.element_type, h.dim
                )));
            }
            serendipity |= ty.serendipity;
            if info.get(&(h.dim, h.entity_tag)).is_none_or(|e| e.keep) {
                kept.push((h, ty));
            }
        }
        let tdim = kept
            .iter()
            .filter(|(h, _)| h.count > 0)
            .map(|(h, _)| h.dim)
            .max()
            .unwrap_or(0);
        if tdim == 0 {
            return Err(content_error("no elements of dimension 1 or more"));
        }
        let mut blocks: Vec<(CellType, u32)> = Vec::new();
        for (_, ty) in kept.iter().filter(|(h, _)| h.dim == tdim) {
            match blocks.iter().find(|(c, _)| *c == ty.cell) {
                Some(&(_, degree)) if degree != ty.degree => {
                    return Err(GridError::DuplicateCellBlock(cell_name(ty.cell)));
                }
                Some(_) => {}
                None => blocks.push((ty.cell, ty.degree)),
            }
        }
        blocks.sort_unstable();
        Ok(Layout {
            tdim,
            blocks,
            serendipity,
            num_partitions,
            info,
        })
    }

    /// The elementary tag, sorted physical tags, kept flag and partition of the elements of
    /// `block`. Entities missing from `$Entities` have their own tag, no physical tags and no
    /// partition.
    fn entity(&self, block: &ElementBlock) -> (i32, &[i32], bool, i32) {
        match self.info.get(&(block.dim, block.entity_tag)) {
            Some(e) => (e.elementary, &e.physical, e.keep, e.partition),
            None => (block.entity_tag, &[], true, 0),
        }
    }

    /// Whether the elements of `block` belong to the serial mesh (not on a partition
    /// interface).
    pub(super) fn keeps(&self, block: &ElementBlock) -> bool {
        self.entity(block).2
    }

    /// The cells of the D_TOP elements of `blocks` (other dimensions are skipped): one
    /// [`CellChunk`] per layout block, in layout order, cells in the order of `blocks`.
    ///
    /// # Errors
    ///
    /// `Parse` for a serendipity element (run [`complete`] first) or a cell type the layout
    /// lacks; `DuplicateCellBlock` for another degree than the layout's;
    /// `UnsupportedElementType`; `LengthMismatch` for a block whose node list is not its type's
    /// node count per element; `TooLarge`.
    pub fn cells(&self, blocks: &[ElementBlock]) -> Result<Vec<CellChunk>> {
        let mut out: Vec<CellChunk> = self
            .blocks
            .iter()
            .map(|&(cell, degree)| CellChunk {
                cell,
                degree,
                nodes: Vec::new(),
                tags: Vec::new(),
                elementary: Vec::new(),
                physical: Vec::new(),
                partitions: Vec::new(),
            })
            .collect();
        for b in blocks.iter().filter(|b| b.dim == self.tdim) {
            let (elementary, physical, keep, partition) = self.entity(b);
            if !keep {
                continue;
            }
            let ty = element_type(b.element_type)?;
            check_len(b.nodes.len(), b.tags.len(), ty.num_nodes)?;
            if ty.serendipity {
                return Err(content_error(format!(
                    "element type {}: serendipity cells must be completed first",
                    ty.id
                )));
            }
            let Some(slot) = out.iter_mut().find(|c| c.cell == ty.cell) else {
                return Err(content_error(format!(
                    "element type {}: no cell block of its type",
                    ty.id
                )));
            };
            if slot.degree != ty.degree {
                return Err(GridError::DuplicateCellBlock(cell_name(ty.cell)));
            }
            let map =
                gridcl_refcell::gmsh_to_internal(ty.cell, ty.degree).expect("Gmsh order exists");
            let k = ty.num_nodes;
            slot.nodes.reserve(b.nodes.len());
            for (row, &element) in b.nodes.chunks_exact(k).zip(&b.tags) {
                let c = index_u32(slot.tags.len())?;
                let start = slot.nodes.len();
                slot.nodes.resize(start + k, 0);
                for (j, &t) in row.iter().enumerate() {
                    slot.nodes[start + map[j]] = t;
                }
                slot.tags.push(element);
                slot.elementary.push(elementary);
                slot.physical.extend(physical.iter().map(|&p| (c, p)));
                if self.num_partitions > 0 {
                    slot.partitions.push(partition);
                }
            }
        }
        Ok(out)
    }

    /// The tag groups of the kept elements of dimension below D_TOP in `blocks`, in
    /// `CellType` order, elements in the order of `blocks`, corners in internal vertex order.
    ///
    /// # Errors
    ///
    /// `UnsupportedElementType`; `LengthMismatch` as [`cells`](Layout::cells); `TooLarge`.
    pub fn tag_groups(&self, blocks: &[ElementBlock]) -> Result<Vec<GmshTagGroup>> {
        let mut groups: Vec<GmshTagGroup> = Vec::new();
        for b in blocks.iter().filter(|b| b.dim < self.tdim) {
            let (elementary, physical, keep, _) = self.entity(b);
            if !keep {
                continue;
            }
            let ty = element_type(b.element_type)?;
            check_len(b.nodes.len(), b.tags.len(), ty.num_nodes)?;
            let nv = gridcl_refcell::vertices(ty.cell).len();
            let vertex_map = match ty.cell {
                CellType::Point => vec![0],
                cell => gridcl_refcell::gmsh_to_internal(cell, 1).expect("Gmsh order exists"),
            };
            let g = match groups.iter().position(|g| g.cell == ty.cell) {
                Some(g) => g,
                None => {
                    groups.push(GmshTagGroup {
                        dim: b.dim,
                        cell: ty.cell,
                        corners: Vec::new(),
                        tags: Vec::new(),
                        elementary: Vec::new(),
                        physical: Vec::new(),
                    });
                    groups.len() - 1
                }
            };
            let group = &mut groups[g];
            let k = ty.num_nodes;
            for (row, &element) in b.nodes.chunks_exact(k).zip(&b.tags) {
                let e = index_u32(group.tags.len())?;
                group.corners.extend(to_internal(&vertex_map, &row[..nv]));
                group.tags.push(element);
                group.elementary.push(elementary);
                group.physical.extend(physical.iter().map(|&p| (e, p)));
            }
        }
        groups.sort_by_key(|g| g.cell);
        for g in &mut groups {
            g.physical.sort_unstable();
        }
        Ok(groups)
    }
}

/// The cells of one layout block in a chunk of D_TOP elements.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellChunk {
    pub cell: CellType,
    /// The complete degree.
    pub degree: u32,
    /// Node tags, `[cell][nodes per cell]` in A.5 order.
    pub nodes: Vec<i64>,
    /// The element tag of each cell (its origin id).
    pub tags: Vec<i64>,
    /// The elementary tag of each cell.
    pub elementary: Vec<i32>,
    /// (cell index in this chunk's block, physical tag), sorted.
    pub physical: Vec<(u32, i32)>,
    /// The file's partition of each cell, 1-based, 0 for none; empty for a file without
    /// `$PartitionedEntities`.
    pub partitions: Vec<i32>,
}

/// The internal order of `file` (one element in Gmsh order): `out[map[k]] = file[k]`.
fn to_internal<T: Copy + Default>(map: &[usize], file: &[T]) -> Vec<T> {
    let mut out = vec![T::default(); file.len()];
    for (k, &i) in map.iter().enumerate() {
        out[i] = file[k];
    }
    out
}

/// The sorted corner tags of quadrilateral face `f` of a cell whose internal-order node tags
/// are `tags`.
fn face_key(cell: CellType, f: usize, tags: &[i64]) -> Vec<i64> {
    let mut key: Vec<i64> = gridcl_refcell::sub_entities(cell, 2)[f]
        .iter()
        .map(|&v| tags[v])
        .collect();
    key.sort_unstable();
    key
}

/// The sub-entity (dimension, index) whose interior holds internal node `node` of a P2 `cell`.
fn entity_of_node(cell: CellType, node: usize) -> (usize, usize) {
    for dim in 0..=gridcl_refcell::tdim(cell) {
        for index in 0..gridcl_refcell::num_sub_entities(cell, dim) {
            if gridcl_refcell::lagrange_entity_nodes(cell, 2, dim, index).contains(&node) {
                return (dim, index);
            }
        }
    }
    unreachable!("every node of {cell:?} lies in a sub-entity")
}

/// The nodes of a file, then those the serendipity upgrade adds.
struct Nodes {
    coords: Vec<[f64; 3]>,
    tags: Vec<i64>,
    index: HashMap<i64, u32>,
    /// The tag of the next added node.
    next_tag: i64,
}

impl Nodes {
    /// The nodes of `file`; added nodes get tags above the largest.
    fn of(file: &MshFile) -> Result<Nodes> {
        let n: usize = file.node_blocks.iter().map(|b| b.tags.len()).sum();
        let mut nodes = Nodes {
            coords: Vec::with_capacity(n),
            tags: Vec::with_capacity(n),
            index: HashMap::with_capacity(n),
            next_tag: 1,
        };
        for b in &file.node_blocks {
            for (&t, &x) in b.tags.iter().zip(&b.coords) {
                nodes.index.insert(t, index_u32(nodes.coords.len())?);
                nodes.coords.push(x);
                nodes.tags.push(t);
                nodes.next_tag = nodes
                    .next_tag
                    .max(t.checked_add(1).ok_or(GridError::TooLarge)?);
            }
        }
        Ok(nodes)
    }

    /// The index of node tag `t`, or `Parse` for element `element`.
    fn get(&self, t: i64, element: i64) -> Result<u32> {
        self.index
            .get(&t)
            .copied()
            .ok_or_else(|| missing_node(t, element))
    }

    /// Adds a node at `x` with a fresh tag; returns its tag.
    fn add(&mut self, x: [f64; 3]) -> Result<i64> {
        let t = self.next_tag;
        self.next_tag = t.checked_add(1).ok_or(GridError::TooLarge)?;
        self.index.insert(t, index_u32(self.coords.len())?);
        self.coords.push(x);
        self.tags.push(t);
        Ok(t)
    }
}

/// `Parse` for node `t` of element `element` that `$Nodes` lacks.
pub(super) fn missing_node(t: i64, element: i64) -> GridError {
    content_error(format!("element {element}: node {t} is not in $Nodes"))
}

/// The complete-P2 node tags of a serendipity cell in internal order. Missing face-centre
/// nodes are shared through `faces` (sorted face corner tags → node tag), which complete-P2
/// neighbours filled first; other missing nodes are added.
fn upgrade(
    cell: CellType,
    file_tags: &[i64],
    element: i64,
    nodes: &mut Nodes,
    faces: &mut HashMap<Vec<i64>, i64>,
) -> Result<Vec<i64>> {
    let up = gridcl_refcell::serendipity_upgrade(cell).expect("a serendipity cell");
    let map = gridcl_refcell::gmsh_to_internal(cell, 2).expect("P2 exists");
    let nf = up.num_file_nodes;
    let mut internal = vec![0i64; up.num_nodes];
    for (k, &t) in file_tags.iter().enumerate() {
        internal[map[k]] = t;
    }
    let coords: Vec<[f64; 3]> = file_tags
        .iter()
        .map(|&t| Ok(nodes.coords[nodes.get(t, element)? as usize]))
        .collect::<Result<_>>()?;
    for (m, weights) in up.weights.chunks_exact(nf).enumerate() {
        let node = map[nf + m];
        let (dim, index) = entity_of_node(cell, node);
        let key =
            (dim == 2 && gridcl_refcell::tdim(cell) == 3).then(|| face_key(cell, index, &internal));
        if let Some(&t) = key.as_ref().and_then(|k| faces.get(k)) {
            internal[node] = t;
            continue;
        }
        let mut x = [0.0; 3];
        for (w, c) in weights.iter().zip(&coords) {
            for g in 0..3 {
                x[g] += w * c[g];
            }
        }
        let t = nodes.add(x)?;
        internal[node] = t;
        if let Some(k) = key {
            faces.insert(k, t);
        }
    }
    Ok(internal)
}

/// `file` with its serendipity cells upgraded to complete P2 (ADR-0022, item 11), or `None`
/// if it has none: the cells of each serendipity D_TOP block are rewritten in place as the
/// complete type, and the added nodes follow the file's nodes in one more node block.
///
/// Cells are visited by cell type in layout order, in file order within a type. An added
/// face-centre node is shared per face, keyed by its sorted corner tags, reuses the node of a
/// complete-P2 neighbour, and otherwise takes a fresh tag above the file's largest node tag,
/// in order of first use. Lower-dimensional serendipity elements are left as they are (tag
/// groups use their corners).
///
/// # Errors
///
/// `Parse` for a node of a serendipity cell that `$Nodes` lacks; `LengthMismatch`,
/// `UnsupportedElementType` as [`Layout::cells`]; `TooLarge`.
#[doc(hidden)]
pub fn complete(file: &MshFile, layout: &Layout) -> Result<Option<MshFile>> {
    let top = |b: &ElementBlock| b.dim == layout.tdim && layout.keeps(b);
    let mut any = false;
    for b in file.element_blocks.iter().filter(|b| top(b)) {
        let ty = element_type(b.element_type)?;
        check_len(b.nodes.len(), b.tags.len(), ty.num_nodes)?;
        any |= ty.serendipity;
    }
    if !any {
        return Ok(None);
    }
    let mut nodes = Nodes::of(file)?;
    let num_file_nodes = nodes.tags.len();

    // Face-centre nodes of complete-P2 cells, which serendipity neighbours reuse.
    let mut faces: HashMap<Vec<i64>, i64> = HashMap::new();
    for &(cell, degree) in &layout.blocks {
        if degree != 2 || gridcl_refcell::tdim(cell) != 3 {
            continue;
        }
        let map = gridcl_refcell::gmsh_to_internal(cell, 2).expect("P2 exists");
        for b in file.element_blocks.iter().filter(|b| top(b)) {
            let ty = element_type(b.element_type)?;
            if ty.cell != cell || ty.serendipity {
                continue;
            }
            for row in b.nodes.chunks_exact(ty.num_nodes) {
                let tags = to_internal(&map, row);
                for f in 0..gridcl_refcell::num_sub_entities(cell, 2) {
                    let interior = gridcl_refcell::lagrange_entity_nodes(cell, 2, 2, f);
                    if !interior.is_empty() {
                        faces
                            .entry(face_key(cell, f, &tags))
                            .or_insert(tags[interior.start]);
                    }
                }
            }
        }
    }

    let mut out = file.clone();
    for &(cell, _) in &layout.blocks {
        let complete = lagrange_element_type(cell, 2)?;
        let map = gridcl_refcell::gmsh_to_internal(cell, 2).expect("P2 exists");
        for (b, block) in file.element_blocks.iter().enumerate() {
            let ty = element_type(block.element_type)?;
            if !top(block) || ty.cell != cell || !ty.serendipity {
                continue;
            }
            let mut rewritten = Vec::with_capacity(block.tags.len() * complete.num_nodes);
            for (row, &element) in block.nodes.chunks_exact(ty.num_nodes).zip(&block.tags) {
                let internal = upgrade(cell, row, element, &mut nodes, &mut faces)?;
                rewritten.extend(map.iter().map(|&i| internal[i]));
            }
            out.element_blocks[b].element_type = complete.id;
            out.element_blocks[b].nodes = rewritten;
        }
    }
    if nodes.tags.len() > num_file_nodes {
        let entity_tag = file
            .element_blocks
            .iter()
            .find(|b| top(b))
            .map_or(0, |b| b.entity_tag);
        out.node_blocks.push(NodeBlock {
            dim: layout.tdim,
            entity_tag,
            tags: nodes.tags.split_off(num_file_nodes),
            coords: nodes.coords.split_off(num_file_nodes),
        });
    }
    Ok(Some(out))
}

/// `IndexOutOfRange` unless `range` is a forward range within `0..len`.
fn check_range(range: &Range<usize>, len: usize) -> Result<()> {
    if range.start > range.end || range.end > len {
        return Err(GridError::IndexOutOfRange {
            index: range.end.max(range.start),
            len,
        });
    }
    Ok(())
}

/// The nodes `range` of `file` in file order over all node blocks: what
/// [`MshIndex::read_nodes`](super::MshIndex::read_nodes) reads from the file.
///
/// # Errors
///
/// `IndexOutOfRange` if `range` ends beyond the nodes or is reversed.
#[doc(hidden)]
pub fn node_range(file: &MshFile, range: Range<usize>) -> Result<NodeChunk> {
    let n: usize = file.node_blocks.iter().map(|b| b.tags.len()).sum();
    check_range(&range, n)?;
    let mut out = NodeChunk {
        tags: Vec::with_capacity(range.len()),
        coords: Vec::with_capacity(range.len()),
    };
    let mut first = 0;
    for b in &file.node_blocks {
        let lo = range.start.max(first);
        let hi = range.end.min(first + b.tags.len());
        if lo < hi {
            out.tags.extend_from_slice(&b.tags[lo - first..hi - first]);
            out.coords
                .extend_from_slice(&b.coords[lo - first..hi - first]);
        }
        first += b.tags.len();
    }
    Ok(out)
}

/// The elements `range` of dimension `dim` of `file`, in file order within the dimension, as
/// the blocks that hold them cut to the range: what
/// [`MshIndex::read_elements`](super::MshIndex::read_elements) reads from the file.
///
/// # Errors
///
/// `IndexOutOfRange` if `range` ends beyond the elements of `dim` or is reversed;
/// `LengthMismatch` for a block whose node list does not divide into its elements.
#[doc(hidden)]
pub fn element_range(file: &MshFile, dim: usize, range: Range<usize>) -> Result<ElementChunk> {
    let n: usize = file
        .element_blocks
        .iter()
        .filter(|b| b.dim == dim)
        .map(|b| b.tags.len())
        .sum();
    check_range(&range, n)?;
    let mut out = ElementChunk::default();
    let mut first = 0;
    for b in file.element_blocks.iter().filter(|b| b.dim == dim) {
        let count = b.tags.len();
        let lo = range.start.max(first);
        let hi = range.end.min(first + count);
        if lo < hi {
            let k = b.nodes.len() / count;
            check_len(b.nodes.len(), count, k)?;
            let (lo, hi) = (lo - first, hi - first);
            out.blocks.push(ElementBlock {
                dim,
                entity_tag: b.entity_tag,
                element_type: b.element_type,
                tags: b.tags[lo..hi].to_vec(),
                nodes: b.nodes[lo * k..hi * k].to_vec(),
            });
        }
        first += count;
    }
    Ok(out)
}

/// D_GEO of the nodes over `comm`: 3 if any |z| > 10^−12 L, else 2 if any |y| > 10^−12 L,
/// else 1, with L the diameter of the bounding box (docs/design.md, "Geometric dimension").
fn infer_gdim(coords: &[[f64; 3]], comm: &dyn CommDyn) -> usize {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut abs = [0.0f64; 3];
    for x in coords {
        for g in 0..3 {
            lo[g] = lo[g].min(x[g]);
            hi[g] = hi[g].max(x[g]);
            abs[g] = abs[g].max(x[g].abs());
        }
    }
    let mut diameter2 = 0.0;
    for g in 0..3 {
        let (l, h) = (-comm.max(-lo[g]), comm.max(hi[g]));
        if h >= l {
            diameter2 += (h - l) * (h - l);
        }
        abs[g] = comm.max(abs[g]);
    }
    let threshold = 1e-12 * diameter2.sqrt();
    if abs[2] > threshold {
        3
    } else if abs[1] > threshold {
        2
    } else {
        1
    }
}

/// D_GEO of a mesh of dimension `tdim` whose ranks hold the nodes `coords`: `requested`, or
/// else inferred over all ranks and at least `tdim`. Collective.
///
/// # Errors
///
/// `DimensionMismatch` for a `requested` outside 1–3 or below the inferred D_GEO (which is at
/// least `tdim`). Every rank gets the same result.
#[doc(hidden)]
pub fn geometric_dim(
    tdim: usize,
    coords: &[[f64; 3]],
    comm: &dyn CommDyn,
    requested: Option<usize>,
) -> Result<usize> {
    let inferred = infer_gdim(coords, comm).max(tdim);
    match requested {
        None => Ok(inferred),
        Some(g) if (1..=3).contains(&g) && g >= inferred => Ok(g),
        Some(g) => Err(GridError::DimensionMismatch {
            expected: (tdim, inferred),
            found: (tdim, g),
        }),
    }
}
