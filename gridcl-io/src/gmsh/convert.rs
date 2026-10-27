//! The conversion between a file and the host mesh description (ADR-0022).

use std::collections::{HashMap, HashSet};

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, CommExt};

use super::element_type::cell_name;
use super::{ElementBlock, FORMAT, MshFile, PhysicalName, element_type, lagrange_element_type};

/// The cells of one cell type and degree.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::{parse, to_gmsh_mesh};
/// use gridcl_parallel::Serial;
///
/// // Two triangles and their shared edge as a tag element of entity 5.
/// let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
///     $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 5 1 1\n3 1 3\n$EndElements\n";
/// let mesh = to_gmsh_mesh(&parse(text)?, &Serial, None)?;
/// let block = &mesh.blocks[0];
/// assert_eq!((block.cell, block.degree), (CellType::Triangle, 1));
/// assert_eq!((&block.tags[..], &block.elementary[..]), (&[1, 2][..], &[1, 1][..]));
/// assert_eq!(block.nodes, [0, 1, 2, 0, 2, 3]);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GmshCellBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree of the complete Lagrange element (serendipity cells are upgraded to 2).
    pub degree: u32,
    /// Indices into [`GmshMesh::nodes`], `[cell][nodes per cell]` in A.5 order.
    pub nodes: Vec<u32>,
    /// The element tag of each cell (its origin id).
    pub tags: Vec<i64>,
    /// The elementary (entity) tag of each cell.
    pub elementary: Vec<i32>,
    /// (cell index in the block, physical tag), sorted.
    pub physical: Vec<(u32, i32)>,
}

/// The lower-dimensional elements of one cell type, which become tags.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::{parse, to_gmsh_mesh};
/// use gridcl_parallel::Serial;
///
/// // Two triangles and their shared edge as a tag element of entity 5.
/// let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
///     $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 5 1 1\n3 1 3\n$EndElements\n";
/// let mesh = to_gmsh_mesh(&parse(text)?, &Serial, None)?;
/// let group = &mesh.tag_groups[0];
/// assert_eq!((group.dim, group.cell), (1, CellType::Interval));
/// assert_eq!((&group.corners[..], &group.tags[..]), (&[1, 3][..], &[3][..]));
/// assert_eq!(group.elementary, [5]);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GmshTagGroup {
    /// The dimension of the elements.
    pub dim: usize,
    /// The element's cell type.
    pub cell: CellType,
    /// The corner node tags of each element in internal vertex order, `[element][vertices]`.
    pub corners: Vec<i64>,
    /// The element tag of each element.
    pub tags: Vec<i64>,
    /// The elementary (entity) tag of each element.
    pub elementary: Vec<i32>,
    /// (element index in the group, physical tag), sorted.
    pub physical: Vec<(u32, i32)>,
}

/// The mesh a file describes: D_TOP and D_GEO, nodes with tags, cell blocks of the D_TOP
/// elements in `CellType` order, the lower-dimensional elements as tag groups, and the
/// physical names.
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::{parse, to_gmsh_mesh};
/// use gridcl_parallel::Serial;
///
/// // Two triangles and their shared edge as a tag element of entity 5.
/// let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
///     $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 5 1 1\n3 1 3\n$EndElements\n";
/// let mesh = to_gmsh_mesh(&parse(text)?, &Serial, None)?;
/// assert_eq!((mesh.tdim, mesh.gdim), (2, 2));
/// assert_eq!(mesh.node_tags, [1, 2, 3, 4]);
/// assert_eq!(&mesh.nodes[4..6], [1.0, 1.0]);
/// assert_eq!((mesh.blocks.len(), mesh.tag_groups.len()), (1, 1));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct GmshMesh {
    /// D_TOP: the highest element dimension.
    pub tdim: usize,
    /// D_GEO: inferred, or fixed by the caller.
    pub gdim: usize,
    /// The node coordinates, `[node][gdim]`.
    pub nodes: Vec<f64>,
    /// The node tags (origin ids).
    pub node_tags: Vec<i64>,
    /// The cell blocks, in `CellType` order.
    pub blocks: Vec<GmshCellBlock>,
    /// The tag groups, in `CellType` order.
    pub tag_groups: Vec<GmshTagGroup>,
    /// The physical names.
    pub names: Vec<PhysicalName>,
}

/// `Parse` for a file whose content, not its syntax, is wrong; there is no byte offset.
fn content_error(message: impl Into<String>) -> GridError {
    GridError::Parse {
        format: FORMAT,
        offset: 0,
        message: format!("{} (no byte offset: the file's content)", message.into()),
    }
}

/// u32 of `i`, or `TooLarge`.
fn index_u32(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| GridError::TooLarge)
}

/// Fails with `LengthMismatch` unless `found` is `count · per`; `TooLarge` if that overflows.
fn check_len(found: usize, count: usize, per: usize) -> Result<()> {
    let expected = count.checked_mul(per).ok_or(GridError::TooLarge)?;
    if found == expected {
        Ok(())
    } else {
        Err(GridError::LengthMismatch { expected, found })
    }
}

/// Fails with `DuplicateOriginId` at the first repeated tag.
fn check_unique(tags: impl IntoIterator<Item = i64>) -> Result<()> {
    let mut seen = HashSet::new();
    for t in tags {
        if !seen.insert(t) {
            return Err(GridError::DuplicateOriginId(t));
        }
    }
    Ok(())
}

/// What the elements of one entity carry: the elementary tag, the physical tags, and whether
/// they belong to the serial mesh.
struct EntityInfo {
    elementary: i32,
    physical: Vec<i32>,
    keep: bool,
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
                        physical: x.physical_tags.clone(),
                        keep: true,
                    },
                );
            }
        }
    }
    if let Some(p) = &file.partitioned_entities {
        for (dim, list) in [&p.points, &p.curves, &p.surfaces, &p.volumes]
            .into_iter()
            .enumerate()
        {
            for x in list {
                info.insert(
                    (dim, x.entity.tag),
                    EntityInfo {
                        elementary: x.parent_tag,
                        physical: x.entity.physical_tags.clone(),
                        keep: x.parent_dim == dim,
                    },
                );
            }
        }
    }
    info
}

/// The elementary tag and sorted, distinct physical tags of the elements of `block`, and
/// whether they belong to the serial mesh. Entities missing from `$Entities` have their own
/// tag and no physical tags.
fn block_tags(
    info: &HashMap<(usize, i32), EntityInfo>,
    block: &ElementBlock,
) -> (i32, Vec<i32>, bool) {
    match info.get(&(block.dim, block.entity_tag)) {
        Some(e) => {
            let mut physical = e.physical.clone();
            physical.sort_unstable();
            physical.dedup();
            (e.elementary, physical, e.keep)
        }
        None => (block.entity_tag, Vec::new(), true),
    }
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

/// The nodes of the mesh being built: the file's, then those the serendipity upgrade adds.
struct Nodes {
    coords: Vec<[f64; 3]>,
    tags: Vec<i64>,
    index: HashMap<i64, u32>,
    /// The tag of the next added node.
    next_tag: i64,
}

impl Nodes {
    /// The index of node tag `t`, or `Parse` for element `element`.
    fn get(&self, t: i64, element: i64) -> Result<u32> {
        self.index
            .get(&t)
            .copied()
            .ok_or_else(|| content_error(format!("element {element}: node {t} is not in $Nodes")))
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

/// One D_TOP element: its block in the file and its index in the block.
#[derive(Clone, Copy)]
struct CellRef<'a> {
    block: &'a ElementBlock,
    i: usize,
    elementary: i32,
    physical: &'a [i32],
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

/// The mesh `file` describes. `gdim` fixes D_GEO; otherwise it is inferred over all nodes of
/// the file, reduced over `comm`.
///
/// - D_TOP is the highest element dimension; its elements form cell blocks in `CellType`
///   order, cells in file order within a block. Serendipity cells are upgraded to complete P2
///   (ADR-0022, item 11): an added face-centre node is shared per face, keyed by its sorted
///   corner tags, reuses the node of a complete-P2 neighbour, and otherwise takes a fresh tag
///   above the file's largest node tag, in order of first use.
/// - Lower-dimensional elements become tag groups with their corner tags in internal vertex
///   order.
/// - Elementary tags are the entity tags; in a partitioned file, the parent entity's.
///   Elements on partition interfaces (whose partitioned entity has a parent of higher
///   dimension) are dropped, so the file reads as its unpartitioned mesh.
/// - Nodes are all the file's nodes in file order, then the added ones.
///
/// # Errors
///
/// - `DuplicateOriginId` for repeated node or element tags, checked over the whole file first;
/// - `Parse` for a file without elements of dimension ≥ 1 or with an element node that is not
///   in `$Nodes`;
/// - `DuplicateCellBlock` for one cell type in two degrees (serendipity counts as degree 2);
/// - `DimensionMismatch` for a `gdim` outside 1–3, below D_TOP or below the inferred D_GEO;
/// - `UnmatchedTagElement` for a lower-dimensional element with a corner that is not a vertex
///   of a cell;
/// - `UnsupportedElementType` for a type outside A.7; `Parse` for an element type of another
///   dimension than its block, and `LengthMismatch` for an element block whose node list does
///   not hold its type's node count per element or a node block with more or fewer coordinates
///   than tags (both only in a hand-built `MshFile`);
/// - `TooLarge` if a count or tag overflows.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::{parse, to_gmsh_mesh};
/// use gridcl_parallel::Serial;
///
/// // Two triangles in the plane z = 0 and their shared edge as a tag element.
/// let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
///     $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 5 1 1\n3 1 3\n$EndElements\n";
/// let mesh = to_gmsh_mesh(&parse(text)?, &Serial, None)?;
/// assert_eq!((mesh.tdim, mesh.gdim), (2, 2));
/// assert_eq!(mesh.blocks[0].cell, CellType::Triangle);
/// assert_eq!(mesh.blocks[0].nodes, [0, 1, 2, 0, 2, 3]);
/// assert_eq!(mesh.tag_groups[0].corners, [1, 3]);
/// assert_eq!(mesh.tag_groups[0].elementary, [5]);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn to_gmsh_mesh(file: &MshFile, comm: &dyn CommDyn, gdim: Option<usize>) -> Result<GmshMesh> {
    check_unique(file.node_blocks.iter().flat_map(|b| b.tags.iter().copied()))?;
    check_unique(
        file.element_blocks
            .iter()
            .flat_map(|b| b.tags.iter().copied()),
    )?;
    for b in &file.element_blocks {
        let ty = element_type(b.element_type)?;
        if gridcl_refcell::tdim(ty.cell) != b.dim {
            return Err(content_error(format!(
                "element type {} in an entity of dimension {}",
                b.element_type, b.dim
            )));
        }
        check_len(b.nodes.len(), b.tags.len(), ty.num_nodes)?;
    }
    for b in &file.node_blocks {
        check_len(b.coords.len(), b.tags.len(), 1)?;
    }
    let info = entity_info(file);
    let kept: Vec<(&ElementBlock, i32, Vec<i32>)> = file
        .element_blocks
        .iter()
        .filter_map(|b| {
            let (elementary, physical, keep) = block_tags(&info, b);
            keep.then_some((b, elementary, physical))
        })
        .collect();
    let tdim = kept
        .iter()
        .filter(|(b, _, _)| !b.tags.is_empty())
        .map(|(b, _, _)| b.dim)
        .max()
        .unwrap_or(0);
    if tdim == 0 {
        return Err(content_error("no elements of dimension 1 or more"));
    }

    // Nodes and D_GEO.
    let num_nodes: usize = file.node_blocks.iter().map(|b| b.tags.len()).sum();
    let mut nodes = Nodes {
        coords: Vec::with_capacity(num_nodes),
        tags: Vec::with_capacity(num_nodes),
        index: HashMap::with_capacity(num_nodes),
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
    let inferred = infer_gdim(&nodes.coords, comm).max(tdim);
    let gdim = match gdim {
        None => inferred,
        Some(g) if (1..=3).contains(&g) && g >= inferred => g,
        Some(g) => {
            return Err(GridError::DimensionMismatch {
                expected: (tdim, inferred),
                found: (tdim, g),
            });
        }
    };

    // The D_TOP elements by cell type, with one degree per type.
    let mut cells: HashMap<CellType, (u32, Vec<CellRef<'_>>)> = HashMap::new();
    for (b, elementary, physical) in kept.iter().filter(|(b, _, _)| b.dim == tdim) {
        let ty = element_type(b.element_type)?;
        let entry = cells.entry(ty.cell).or_insert((ty.degree, Vec::new()));
        if entry.0 != ty.degree {
            return Err(GridError::DuplicateCellBlock(cell_name(ty.cell)));
        }
        entry.1.extend((0..b.tags.len()).map(|i| CellRef {
            block: b,
            i,
            elementary: *elementary,
            physical,
        }));
    }
    let mut types: Vec<CellType> = cells.keys().copied().collect();
    types.sort_unstable();

    // Face-centre nodes of complete-P2 cells, which serendipity neighbours reuse.
    let mut faces: HashMap<Vec<i64>, i64> = HashMap::new();
    for &cell in &types {
        let (degree, refs) = &cells[&cell];
        if *degree != 2 || gridcl_refcell::tdim(cell) != 3 {
            continue;
        }
        let map = gridcl_refcell::gmsh_to_internal(cell, 2).expect("P2 exists");
        for r in refs {
            let k = element_type(r.block.element_type)?.num_nodes;
            if k != map.len() {
                continue; // serendipity
            }
            let tags = to_internal(&map, &r.block.nodes[r.i * k..(r.i + 1) * k]);
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

    // Cell blocks.
    let mut blocks = Vec::with_capacity(types.len());
    let mut vertices: HashSet<i64> = HashSet::new();
    for &cell in &types {
        let (degree, refs) = &cells[&cell];
        let complete = lagrange_element_type(cell, *degree)?;
        let map = gridcl_refcell::gmsh_to_internal(cell, *degree).expect("Gmsh order exists");
        let nv = gridcl_refcell::vertices(cell).len();
        let mut block = GmshCellBlock {
            cell,
            degree: *degree,
            nodes: Vec::with_capacity(refs.len() * complete.num_nodes),
            tags: Vec::with_capacity(refs.len()),
            elementary: Vec::with_capacity(refs.len()),
            physical: Vec::new(),
        };
        for (c, r) in refs.iter().enumerate() {
            let ty = element_type(r.block.element_type)?;
            let k = ty.num_nodes;
            let file_tags = &r.block.nodes[r.i * k..(r.i + 1) * k];
            let element = r.block.tags[r.i];
            let tags = if ty.serendipity {
                upgrade(cell, file_tags, element, &mut nodes, &mut faces)?
            } else {
                to_internal(&map, file_tags)
            };
            vertices.extend(&tags[..nv]);
            for &t in &tags {
                block.nodes.push(nodes.get(t, element)?);
            }
            block.tags.push(element);
            block.elementary.push(r.elementary);
            let c = index_u32(c)?;
            block.physical.extend(r.physical.iter().map(|&p| (c, p)));
        }
        blocks.push(block);
    }

    // Tag groups: the lower-dimensional elements by cell type.
    let mut groups: Vec<GmshTagGroup> = Vec::new();
    for (b, elementary, physical) in kept.iter().filter(|(b, _, _)| b.dim < tdim) {
        let ty = element_type(b.element_type)?;
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
        for (i, &element) in b.tags.iter().enumerate() {
            let corners = to_internal(&vertex_map, &b.nodes[i * k..i * k + nv]);
            for &t in &corners {
                nodes.get(t, element)?;
                if !vertices.contains(&t) {
                    return Err(GridError::UnmatchedTagElement {
                        dim: b.dim,
                        tag: element,
                    });
                }
            }
            let e = index_u32(group.tags.len())?;
            group.corners.extend(corners);
            group.tags.push(element);
            group.elementary.push(*elementary);
            group.physical.extend(physical.iter().map(|&p| (e, p)));
        }
    }
    groups.sort_by_key(|g| g.cell);
    for g in &mut groups {
        g.physical.sort_unstable();
    }

    let mut coords = Vec::with_capacity(nodes.coords.len() * gdim);
    for x in &nodes.coords {
        coords.extend_from_slice(&x[..gdim]);
    }
    Ok(GmshMesh {
        tdim,
        gdim,
        nodes: coords,
        node_tags: nodes.tags,
        blocks,
        tag_groups: groups,
        names: file.physical_names.clone(),
    })
}

/// One entity of the written file: dimension, elementary tag and physical tags.
type EntityKey = (usize, i32, Vec<i32>);

/// The elements of one written entity: (element type, element tags, node tags in Gmsh order).
type EntityElements = Vec<(i32, Vec<i64>, Vec<i64>)>;

/// The physical tags of each of `n` elements from sorted (index, tag) pairs.
fn physical_sets(n: usize, pairs: &[(u32, i32)]) -> Vec<Vec<i32>> {
    let mut sets = vec![Vec::new(); n];
    for &(i, p) in pairs {
        if let Some(set) = sets.get_mut(i as usize) {
            set.push(p);
        }
    }
    for set in &mut sets {
        set.sort_unstable();
        set.dedup();
    }
    sets
}

/// `x + shift`, or `TooLarge`.
fn shifted(x: i64, shift: i64) -> Result<i64> {
    x.checked_add(shift).ok_or(GridError::TooLarge)
}

/// The constant that makes the smallest of `tags` 1, or 0 if none is below 1.
fn shift_to_one(tags: impl Iterator<Item = i64>) -> Result<i64> {
    match tags.min() {
        Some(min) if min < 1 => 1i64.checked_sub(min).ok_or(GridError::TooLarge),
        _ => Ok(0),
    }
}

/// The file of `mesh` (ADR-0022, item 10):
///
/// - **Node tags:** the tags of vertex nodes (the first nodes of each cell) are kept; the other
///   nodes, whose `node_tags` are ignored, are numbered after the largest vertex tag in node
///   order. If any tag is then below 1, all node tags are shifted by one constant so that the
///   smallest is 1. Tag-group corners are shifted with them.
/// - **Element tags:** cells keep theirs; tag elements, whose `tags` are ignored, are numbered
///   after the largest cell tag. If any is below 1, all element tags are shifted alike.
/// - **Elements:** cells at their degree in Gmsh order; tag elements linear, from their
///   corners.
/// - **`$Entities`:** one entity per (dimension, elementary tag, set of physical tags), with the
///   elementary tag as its tag (a fresh tag above the dimension's largest when two physical
///   sets share an elementary tag), a bounding box from its nodes, the first point's
///   coordinates for dimension 0, and empty boundary lists.
/// - One node block, in the first D_TOP entity; three coordinates per node.
///
/// # Errors
///
/// `LengthMismatch` if the node arrays disagree, or a block's or group's arrays disagree with
/// its element count; `IndexOutOfRange` for a cell node beyond the nodes; `UnsupportedDegree`
/// for a cell Gmsh has no type for; `TooLarge` if a tag overflows.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::{GmshCellBlock, GmshMesh, from_gmsh_mesh};
///
/// // Two triangles whose origin ids start at 0, as MeshBuilder's defaults do.
/// let mesh = GmshMesh {
///     tdim: 2,
///     gdim: 2,
///     nodes: vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
///     node_tags: vec![0, 1, 2, 3],
///     blocks: vec![GmshCellBlock {
///         cell: CellType::Triangle,
///         degree: 1,
///         nodes: vec![0, 1, 2, 0, 2, 3],
///         tags: vec![0, 1],
///         elementary: vec![1, 1],
///         physical: vec![],
///     }],
///     tag_groups: vec![],
///     names: vec![],
/// };
/// let file = from_gmsh_mesh(&mesh)?;
/// // Node and element tags are shifted by one so that they start at 1.
/// assert_eq!(file.node_blocks[0].tags, [1, 2, 3, 4]);
/// assert_eq!(file.node_blocks[0].coords[2], [1.0, 1.0, 0.0]);
/// assert_eq!(file.element_blocks[0].tags, [1, 2]);
/// assert_eq!(file.entities.unwrap().surfaces[0].max, [1.0, 1.0, 0.0]);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn from_gmsh_mesh(mesh: &GmshMesh) -> Result<MshFile> {
    let n = mesh.node_tags.len();
    let gd = mesh.gdim;
    let expected = n.checked_mul(gd).ok_or(GridError::TooLarge)?;
    if mesh.nodes.len() != expected || !(1..=3).contains(&gd) {
        return Err(GridError::LengthMismatch {
            expected,
            found: mesh.nodes.len(),
        });
    }
    let coords: Vec<[f64; 3]> = mesh
        .nodes
        .chunks_exact(gd)
        .map(|x| {
            let mut c = [0.0; 3];
            c[..gd].copy_from_slice(x);
            c
        })
        .collect();

    // Shapes first, so that no index below goes out of range.
    for b in &mesh.blocks {
        let ty = lagrange_element_type(b.cell, b.degree)?;
        check_len(b.nodes.len(), b.tags.len(), ty.num_nodes)?;
        check_len(b.elementary.len(), b.tags.len(), 1)?;
        if let Some(&i) = b.nodes.iter().find(|&&i| i as usize >= n) {
            return Err(GridError::IndexOutOfRange {
                index: i as usize,
                len: n,
            });
        }
    }
    for g in &mesh.tag_groups {
        let nv = gridcl_refcell::vertices(g.cell).len();
        check_len(g.corners.len(), g.tags.len(), nv)?;
        check_len(g.elementary.len(), g.tags.len(), 1)?;
    }

    // Node tags: vertices keep theirs, the others follow the largest vertex tag.
    let mut is_vertex = vec![false; n];
    for b in &mesh.blocks {
        let k = gridcl_refcell::num_lagrange_nodes(b.cell, b.degree);
        let nv = gridcl_refcell::vertices(b.cell).len();
        for row in b.nodes.chunks_exact(k) {
            for &i in &row[..nv] {
                is_vertex[i as usize] = true;
            }
        }
    }
    let max_vertex = (0..n)
        .filter(|&i| is_vertex[i])
        .map(|i| mesh.node_tags[i])
        .max()
        .unwrap_or(0);
    let mut next = max_vertex;
    let mut node_tags = Vec::with_capacity(n);
    for (&vertex, &tag) in is_vertex.iter().zip(&mesh.node_tags) {
        node_tags.push(if vertex {
            tag
        } else {
            next = next.checked_add(1).ok_or(GridError::TooLarge)?;
            next
        });
    }
    let node_shift = shift_to_one(node_tags.iter().copied())?;
    let node_tags: Vec<i64> = node_tags
        .iter()
        .map(|&t| shifted(t, node_shift))
        .collect::<Result<_>>()?;
    let index_of: HashMap<i64, usize> = (0..n)
        .filter(|&i| is_vertex[i])
        .map(|i| (mesh.node_tags[i], i))
        .collect();

    // Element tags: cells keep theirs, tag elements follow the largest cell tag.
    let max_cell = mesh
        .blocks
        .iter()
        .flat_map(|b| b.tags.iter().copied())
        .max()
        .unwrap_or(0);
    let num_tag_elements: usize = mesh.tag_groups.iter().map(|g| g.tags.len()).sum();
    let last = i64::try_from(num_tag_elements)
        .ok()
        .and_then(|k| max_cell.checked_add(k))
        .ok_or(GridError::TooLarge)?;
    // Tag elements follow the cells, so the smallest cell tag decides the shift.
    let element_shift = shift_to_one(mesh.blocks.iter().flat_map(|b| b.tags.iter().copied()))?;
    shifted(last, element_shift)?;

    // Elements grouped by entity, in order of first appearance per dimension.
    let mut entities: Vec<EntityKey> = Vec::new();
    let mut elements: Vec<EntityElements> = Vec::new();
    let mut boxes: Vec<([f64; 3], [f64; 3])> = Vec::new();
    let mut first_point: Vec<Option<[f64; 3]>> = Vec::new();
    let mut add = |key: EntityKey, ty: i32, tag: i64, nodes: Vec<i64>, xs: Vec<[f64; 3]>| {
        let e = match entities.iter().position(|k| *k == key) {
            Some(e) => e,
            None => {
                entities.push(key);
                elements.push(Vec::new());
                boxes.push(([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]));
                first_point.push(None);
                entities.len() - 1
            }
        };
        let list = &mut elements[e];
        let b = match list.iter().position(|(t, _, _)| *t == ty) {
            Some(b) => b,
            None => {
                list.push((ty, Vec::new(), Vec::new()));
                list.len() - 1
            }
        };
        list[b].1.push(tag);
        list[b].2.extend(nodes);
        let (lo, hi) = &mut boxes[e];
        for x in &xs {
            for ((l, h), &c) in lo.iter_mut().zip(hi.iter_mut()).zip(x) {
                *l = l.min(c);
                *h = h.max(c);
            }
        }
        first_point[e].get_or_insert(xs[0]);
    };

    let mut groups: Vec<&GmshTagGroup> = mesh.tag_groups.iter().collect();
    groups.sort_by_key(|g| g.dim);
    let mut next_element = max_cell;
    for g in groups {
        let ty = lagrange_element_type(g.cell, 1)?;
        let nv = gridcl_refcell::vertices(g.cell).len();
        let map = match g.cell {
            CellType::Point => vec![0],
            cell => gridcl_refcell::gmsh_to_internal(cell, 1).expect("Gmsh order exists"),
        };
        let physical = physical_sets(g.tags.len(), &g.physical);
        for (e, corners) in g.corners.chunks_exact(nv).enumerate() {
            next_element += 1;
            let file_nodes: Vec<i64> = map.iter().map(|&i| corners[i]).collect();
            let xs = file_nodes
                .iter()
                .map(|t| index_of.get(t).map(|&i| coords[i]))
                .collect::<Option<Vec<_>>>()
                .ok_or(GridError::UnmatchedTagElement {
                    dim: g.dim,
                    tag: g.tags[e],
                })?;
            let nodes = file_nodes
                .iter()
                .map(|&t| shifted(t, node_shift))
                .collect::<Result<_>>()?;
            add(
                (g.dim, g.elementary[e], physical[e].clone()),
                ty.id,
                shifted(next_element, element_shift)?,
                nodes,
                xs,
            );
        }
    }
    for b in &mesh.blocks {
        let ty = lagrange_element_type(b.cell, b.degree)?;
        let map = gridcl_refcell::gmsh_to_internal(b.cell, b.degree).expect("Gmsh order exists");
        let physical = physical_sets(b.tags.len(), &b.physical);
        for (c, row) in b.nodes.chunks_exact(ty.num_nodes).enumerate() {
            let nodes = map.iter().map(|&i| node_tags[row[i] as usize]).collect();
            let xs = row.iter().map(|&i| coords[i as usize]).collect();
            add(
                (mesh.tdim, b.elementary[c], physical[c].clone()),
                ty.id,
                shifted(b.tags[c], element_shift)?,
                nodes,
                xs,
            );
        }
    }

    // Entity tags: the elementary tag, or a fresh one when the tag is taken in its dimension.
    let mut tags = vec![0i32; entities.len()];
    let mut taken: HashSet<(usize, i32)> = HashSet::new();
    let mut fresh = Vec::new();
    for (e, (dim, elementary, _)) in entities.iter().enumerate() {
        if taken.insert((*dim, *elementary)) {
            tags[e] = *elementary;
        } else {
            fresh.push(e);
        }
    }
    for e in fresh {
        let dim = entities[e].0;
        let max = taken
            .iter()
            .filter(|(d, _)| *d == dim)
            .map(|&(_, t)| t)
            .max()
            .unwrap_or(0);
        let t = max.checked_add(1).ok_or(GridError::TooLarge)?;
        taken.insert((dim, t));
        tags[e] = t;
    }

    let mut order: Vec<usize> = (0..entities.len()).collect();
    order.sort_by_key(|&e| (entities[e].0, tags[e]));
    let mut model = super::Entities::default();
    let mut element_blocks = Vec::new();
    for &e in &order {
        let (dim, _, physical) = &entities[e];
        let (min, max) = if *dim == 0 {
            let x = first_point[e].expect("an entity has elements");
            (x, x)
        } else {
            boxes[e]
        };
        let entity = super::Entity {
            tag: tags[e],
            min,
            max,
            physical_tags: physical.clone(),
            boundary: Vec::new(),
        };
        match dim {
            0 => model.points.push(entity),
            1 => model.curves.push(entity),
            2 => model.surfaces.push(entity),
            _ => model.volumes.push(entity),
        }
        let mut list = elements[e].clone();
        list.sort_by_key(|(ty, _, _)| element_type(*ty).map(|t| t.cell).ok());
        for (ty, tags_, nodes) in list {
            element_blocks.push(ElementBlock {
                dim: *dim,
                entity_tag: tags[e],
                element_type: ty,
                tags: tags_,
                nodes,
            });
        }
    }
    let node_entity = order
        .iter()
        .find(|&&e| entities[e].0 == mesh.tdim)
        .map_or(1, |&e| tags[e]);
    Ok(MshFile {
        physical_names: mesh.names.clone(),
        entities: Some(model),
        partitioned_entities: None,
        node_blocks: vec![super::NodeBlock {
            dim: mesh.tdim,
            entity_tag: node_entity,
            tags: node_tags,
            coords,
        }],
        element_blocks,
        periodic: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use super::*;
    use crate::gmsh::{NodeBlock, read_file};
    use gridcl_parallel::Serial;
    use gridcl_refcell::{NodeFamily, lagrange_points};

    fn fixtures() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gmsh")
    }

    fn convert(name: &str) -> GmshMesh {
        let file = read_file(&fixtures().join(name)).unwrap();
        to_gmsh_mesh(&file, &Serial, None).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    /// The fixtures that read into a mesh (all but the duplicate-tag files).
    fn mesh_fixtures() -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(fixtures())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".msh") && !n.starts_with("duplicate_"))
            .collect();
        names.sort();
        names
    }

    /// A file of one node block and the element blocks `(dim, type, tags, node tags)`.
    fn file(nodes: &[(i64, [f64; 3])], blocks: &[(usize, i32, &[i64], &[i64])]) -> MshFile {
        MshFile {
            node_blocks: vec![NodeBlock {
                dim: 3,
                entity_tag: 1,
                tags: nodes.iter().map(|n| n.0).collect(),
                coords: nodes.iter().map(|n| n.1).collect(),
            }],
            element_blocks: blocks
                .iter()
                .map(|&(dim, ty, tags, nodes)| ElementBlock {
                    dim,
                    entity_tag: 1,
                    element_type: ty,
                    tags: tags.to_vec(),
                    nodes: nodes.to_vec(),
                })
                .collect(),
            ..MshFile::default()
        }
    }

    /// The coordinates `[node][gdim]` of node `i` of `mesh`.
    fn node(mesh: &GmshMesh, i: u32) -> &[f64] {
        &mesh.nodes[i as usize * mesh.gdim..(i as usize + 1) * mesh.gdim]
    }

    /// Design "High-order node orderings / Checks" and the pyramid risk row: on the affine
    /// fixtures, every cell's nodes in internal order are the affine image of
    /// `lagrange_points`, for every type, pyramids up to degree 3 included. To 10^−10 relative:
    /// Gmsh's transfinite vertices are off by up to a few 10^−12, and a wrong node order is
    /// off by a fraction of the cell size.
    #[test]
    fn node_maps_agree_with_gmsh_fixtures() {
        let mut seen = HashSet::new();
        for name in mesh_fixtures() {
            let mesh = convert(&name);
            let scale = mesh.nodes.iter().fold(1.0f64, |a, x| a.max(x.abs()));
            for b in &mesh.blocks {
                seen.insert((b.cell, b.degree));
                let td = gridcl_refcell::tdim(b.cell);
                let refv = gridcl_refcell::vertices(b.cell);
                // The vertices at the unit vectors span the affine map from vertex 0.
                let axes: Vec<usize> = (0..td)
                    .map(|a| {
                        refv.iter()
                            .position(|v| (0..td).all(|j| v[j] == if j == a { 1.0 } else { 0.0 }))
                            .expect("a vertex on each axis")
                    })
                    .collect();
                let points = lagrange_points(b.cell, b.degree, NodeFamily::Equispaced).unwrap();
                let k = points.len() / td;
                for (c, row) in b.nodes.chunks_exact(k).enumerate() {
                    let x0 = node(&mesh, row[0]);
                    for (n, xi) in points.chunks_exact(td).enumerate() {
                        for (g, &origin) in x0.iter().enumerate() {
                            let want = origin
                                + (0..td)
                                    .map(|a| xi[a] * (node(&mesh, row[axes[a]])[g] - origin))
                                    .sum::<f64>();
                            let got = node(&mesh, row[n])[g];
                            assert!(
                                (got - want).abs() <= 1e-10 * scale,
                                "{name}: {:?} p{} cell {c} node {n}: {got} vs {want}",
                                b.cell,
                                b.degree
                            );
                        }
                    }
                }
            }
        }
        use CellType::*;
        for cell in [
            Interval,
            Triangle,
            Quadrilateral,
            Tetrahedron,
            Hexahedron,
            Prism,
        ] {
            for p in 1..=4 {
                assert!(seen.contains(&(cell, p)), "{cell:?} p{p} not checked");
            }
        }
        for p in 1..=3 {
            assert!(seen.contains(&(Pyramid, p)), "Pyramid p{p} not checked");
        }
    }

    fn segment(y: f64, z: f64) -> MshFile {
        file(
            &[(1, [0.0, 0.0, z]), (2, [1.0, y, z])],
            &[(1, 1, &[1], &[1, 2])],
        )
    }

    #[test]
    fn convert_gdim_thresholds() {
        let gdim = |f: &MshFile| to_gmsh_mesh(f, &Serial, None).unwrap().gdim;
        assert_eq!(gdim(&segment(0.0, 0.0)), 1);
        assert_eq!(gdim(&segment(0.5e-12, 0.0)), 1);
        assert_eq!(gdim(&segment(2e-12, 0.0)), 2);
        assert_eq!(gdim(&segment(0.0, 2e-12)), 3);
        // A plane at z = c != 0 is 3D (the design's absolute rule).
        assert_eq!(gdim(&segment(0.0, 1.0)), 3);
        // Nodes of no element count too.
        let mut f = segment(0.0, 0.0);
        f.node_blocks[0].tags.push(3);
        f.node_blocks[0].coords.push([0.0, 1.0, 0.0]);
        assert_eq!(gdim(&f), 2);
        // Never below D_TOP.
        let tri = file(
            &[(1, [0.0; 3]), (2, [1.0, 0.0, 0.0]), (3, [2.0, 0.0, 0.0])],
            &[(2, 2, &[1], &[1, 2, 3])],
        );
        assert_eq!(gdim(&tri), 2);
    }

    #[test]
    fn convert_gdim_option() {
        let planar = file(
            &[(1, [0.0; 3]), (2, [1.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0])],
            &[(2, 2, &[1], &[1, 2, 3])],
        );
        let m = to_gmsh_mesh(&planar, &Serial, Some(3)).unwrap();
        assert_eq!((m.gdim, m.nodes.len()), (3, 9));
        assert_eq!(m.nodes[3..6], [1.0, 0.0, 0.0]);
        for g in [0, 1, 4] {
            assert!(
                matches!(
                    to_gmsh_mesh(&planar, &Serial, Some(g)),
                    Err(GridError::DimensionMismatch {
                        expected: (2, 2),
                        found
                    }) if found == (2, g)
                ),
                "gdim {g}"
            );
        }
        assert!(matches!(
            to_gmsh_mesh(&segment(1.0, 0.0), &Serial, Some(1)),
            Err(GridError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn convert_duplicate_tags() {
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(fixtures().join("manifest.json")).unwrap(),
        )
        .unwrap();
        let mut checked = 0;
        for f in manifest["files"].as_array().unwrap() {
            let Some(id) = f["duplicate_id"].as_i64() else {
                continue;
            };
            let name = f["file"].as_str().unwrap();
            let file = read_file(&fixtures().join(name)).unwrap();
            assert!(
                matches!(to_gmsh_mesh(&file, &Serial, None), Err(GridError::DuplicateOriginId(t)) if t == id),
                "{name}"
            );
            checked += 1;
        }
        assert_eq!(checked, 2);
        let dup = file(
            &[(1, [0.0; 3]), (2, [1.0, 0.0, 0.0]), (2, [2.0, 0.0, 0.0])],
            &[(1, 1, &[1], &[1, 2])],
        );
        assert!(matches!(
            to_gmsh_mesh(&dup, &Serial, None),
            Err(GridError::DuplicateOriginId(2))
        ));
    }

    #[test]
    fn convert_rejects_bad_content() {
        let nodes = [(1, [0.0; 3]), (2, [1.0, 0.0, 0.0]), (3, [0.0, 1.0, 0.0])];
        // Only points.
        let points = file(&nodes, &[(0, 15, &[1, 2], &[1, 2])]);
        assert!(matches!(
            to_gmsh_mesh(&points, &Serial, None),
            Err(GridError::Parse { .. })
        ));
        // A node that is not in $Nodes.
        let missing = file(&nodes, &[(2, 2, &[1], &[1, 2, 4])]);
        assert!(matches!(
            to_gmsh_mesh(&missing, &Serial, None),
            Err(GridError::Parse { .. })
        ));
        // Triangles of degrees 1 and 2.
        let mut six = nodes.to_vec();
        six.extend([
            (4, [0.5, 0.0, 0.0]),
            (5, [0.5, 0.5, 0.0]),
            (6, [0.0, 0.5, 0.0]),
        ]);
        let mixed = file(
            &six,
            &[(2, 2, &[1], &[1, 2, 3]), (2, 9, &[2], &[1, 2, 3, 4, 5, 6])],
        );
        assert!(matches!(
            to_gmsh_mesh(&mixed, &Serial, None),
            Err(GridError::DuplicateCellBlock("Triangle"))
        ));
        // A tag element whose corner is an edge node of the P2 triangle.
        let unmatched = file(
            &six,
            &[(2, 9, &[1], &[1, 2, 3, 4, 5, 6]), (1, 1, &[7], &[1, 4])],
        );
        assert!(matches!(
            to_gmsh_mesh(&unmatched, &Serial, None),
            Err(GridError::UnmatchedTagElement { dim: 1, tag: 7 })
        ));
        // Hand-built files: a node list one short, and a triangle in a curve.
        let mut short = file(&nodes, &[(2, 2, &[1], &[1, 2, 3])]);
        short.element_blocks[0].nodes.pop();
        assert!(matches!(
            to_gmsh_mesh(&short, &Serial, None),
            Err(GridError::LengthMismatch {
                expected: 3,
                found: 2
            })
        ));
        let mut no_coords = file(&nodes, &[(2, 2, &[1], &[1, 2, 3])]);
        no_coords.node_blocks[0].coords.pop();
        assert!(matches!(
            to_gmsh_mesh(&no_coords, &Serial, None),
            Err(GridError::LengthMismatch { .. })
        ));
        let mut wrong_dim = file(&nodes, &[(2, 2, &[1], &[1, 2, 3])]);
        wrong_dim.element_blocks[0].dim = 1;
        assert!(matches!(
            to_gmsh_mesh(&wrong_dim, &Serial, None),
            Err(GridError::Parse { .. })
        ));
    }

    /// Upgraded serendipity boxes equal their complete-P2 twins to 10^−14: the same cells (by
    /// element tag) with the same node coordinates in internal order, and the same number of
    /// nodes in use. Gmsh numbers the nodes of the two files differently, so node tags are not
    /// compared.
    #[test]
    fn convert_serendipity_equals_complete_twin() {
        for stem in ["quadrilateral", "hexahedron", "prism", "hybrid"] {
            for suffix in ["", "-binary"] {
                let s = convert(&format!("{stem}_s2{suffix}.msh"));
                let p = convert(&format!("{stem}_p2{suffix}.msh"));
                let used = |m: &GmshMesh| {
                    m.blocks
                        .iter()
                        .flat_map(|b| b.nodes.iter().copied())
                        .collect::<HashSet<_>>()
                        .len()
                };
                assert_eq!(used(&s), used(&p), "{stem}{suffix}: nodes in use");
                let scale = p.nodes.iter().fold(1.0f64, |a, x| a.max(x.abs()));
                assert_eq!(s.blocks.len(), p.blocks.len());
                for (a, b) in s.blocks.iter().zip(&p.blocks) {
                    assert_eq!((a.cell, a.degree, &a.tags), (b.cell, b.degree, &b.tags));
                    let k = a.nodes.len() / a.tags.len();
                    for (ra, rb) in a.nodes.chunks_exact(k).zip(b.nodes.chunks_exact(k)) {
                        for (&i, &j) in ra.iter().zip(rb) {
                            for (x, y) in node(&s, i).iter().zip(node(&p, j)) {
                                assert!(
                                    (x - y).abs() <= 1e-14 * scale,
                                    "{stem}{suffix}: {x} vs {y}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// The Gmsh-order P2 nodes of the hexahedron [x0, x0 + 1] × [0, 1]², with tags from the
    /// lattice position (shared between neighbours).
    fn hex_p2(x0: f64) -> Vec<(i64, [f64; 3])> {
        let points = lagrange_points(CellType::Hexahedron, 2, NodeFamily::Equispaced).unwrap();
        let map = gridcl_refcell::gmsh_to_internal(CellType::Hexahedron, 2).unwrap();
        map.iter()
            .map(|&i| {
                let x = [x0 + points[3 * i], points[3 * i + 1], points[3 * i + 2]];
                let lattice = x.map(|c| (2.0 * c).round() as i64);
                (1 + lattice[0] + 5 * lattice[1] + 15 * lattice[2], x)
            })
            .collect()
    }

    /// Two hexahedra sharing the face x = 1; `serendipity` says which are 20-node.
    fn two_hexes(serendipity: [bool; 2]) -> (MshFile, Vec<Vec<i64>>) {
        let cells: Vec<Vec<(i64, [f64; 3])>> = [0.0, 1.0]
            .iter()
            .zip(serendipity)
            .map(|(&x0, s)| {
                let mut nodes = hex_p2(x0);
                if s {
                    nodes.truncate(20);
                }
                nodes
            })
            .collect();
        let mut all: Vec<(i64, [f64; 3])> = cells.iter().flatten().copied().collect();
        all.sort_by_key(|n| n.0);
        all.dedup_by_key(|n| n.0);
        let tags: Vec<Vec<i64>> = cells
            .iter()
            .map(|c| c.iter().map(|n| n.0).collect())
            .collect();
        let blocks: Vec<(usize, i32, &[i64], &[i64])> = vec![
            (3, if serendipity[0] { 17 } else { 12 }, &[1], &tags[0]),
            (3, if serendipity[1] { 17 } else { 12 }, &[2], &tags[1]),
        ];
        (file(&all, &blocks), tags)
    }

    /// The node index of the centre of the face x = 1, from each cell.
    fn shared_centres(m: &GmshMesh) -> Vec<u32> {
        m.blocks[0]
            .nodes
            .as_chunks::<27>()
            .0
            .iter()
            .map(|row| {
                *row.iter()
                    .find(|&&i| node(m, i) == [1.0, 0.5, 0.5])
                    .expect("a node at the face centre")
            })
            .collect()
    }

    #[test]
    fn convert_serendipity_neighbours_share_face_centres() {
        // A 20-node hexahedron beside a 27-node one reuses its neighbour's face centre.
        let (f, _) = two_hexes([false, true]);
        let max_tag = f.node_blocks[0].tags.iter().max().copied().unwrap();
        let m = to_gmsh_mesh(&f, &Serial, None).unwrap();
        let c = shared_centres(&m);
        assert_eq!(c[0], c[1]);
        // Five new face centres and the volume centre, tagged above the file's tags in order.
        let added = &m.node_tags[f.node_blocks[0].tags.len()..];
        assert_eq!(added, (max_tag + 1..=max_tag + 6).collect::<Vec<_>>());
        // Two 20-node hexahedra create the shared face centre once.
        let (f, _) = two_hexes([true, true]);
        let m = to_gmsh_mesh(&f, &Serial, None).unwrap();
        let c = shared_centres(&m);
        assert_eq!(c[0], c[1]);
        assert_eq!(m.node_tags.len(), f.node_blocks[0].tags.len() + 2 * 7 - 1);
        // In the same order, whichever cell comes first in the file.
        let (g, _) = two_hexes([true, false]);
        let m = to_gmsh_mesh(&g, &Serial, None).unwrap();
        let c = shared_centres(&m);
        assert_eq!(c[0], c[1]);
        assert_eq!(m.node_tags.len(), g.node_blocks[0].tags.len() + 6);
    }

    /// A partitioned file reads as its unpartitioned mesh: the same cells and tag elements with
    /// the parents' elementary tags and the same physical tags; partition-interface elements
    /// are dropped.
    #[test]
    fn convert_partitioned_reads_as_unpartitioned() {
        type Key = (CellType, Vec<i64>, i32, Vec<i32>);
        fn keys(m: &GmshMesh) -> Vec<Key> {
            let mut out = Vec::new();
            for b in &m.blocks {
                let k = b.nodes.len() / b.tags.len();
                let nv = gridcl_refcell::vertices(b.cell).len();
                for (c, row) in b.nodes.chunks_exact(k).enumerate() {
                    let mut corners: Vec<i64> =
                        row[..nv].iter().map(|&i| m.node_tags[i as usize]).collect();
                    corners.sort_unstable();
                    let physical = b.physical.iter().filter(|p| p.0 as usize == c);
                    out.push((
                        b.cell,
                        corners,
                        b.elementary[c],
                        physical.map(|p| p.1).collect(),
                    ));
                }
            }
            for g in &m.tag_groups {
                let nv = gridcl_refcell::vertices(g.cell).len();
                for (e, row) in g.corners.chunks_exact(nv).enumerate() {
                    let mut corners = row.to_vec();
                    corners.sort_unstable();
                    let physical = g.physical.iter().filter(|p| p.0 as usize == e);
                    out.push((
                        g.cell,
                        corners,
                        g.elementary[e],
                        physical.map(|p| p.1).collect(),
                    ));
                }
            }
            out.sort();
            out
        }
        let p = convert("partitioned_p1.msh");
        let s = convert("two_regions_p1.msh");
        assert_eq!((p.tdim, p.gdim), (s.tdim, s.gdim));
        assert_eq!(keys(&p), keys(&s));
    }

    /// Tag groups: corners in internal vertex order (a Gmsh quadrilateral 0 1 2 3 is internal
    /// 0 1 3 2), physical pairs sorted, one group per cell type in `CellType` order.
    #[test]
    fn convert_tag_groups() {
        let m = convert("two_regions_p1.msh");
        let cells: Vec<CellType> = m.tag_groups.iter().map(|g| g.cell).collect();
        assert_eq!(cells, [CellType::Point, CellType::Interval]);
        for g in &m.tag_groups {
            assert!(g.physical.windows(2).all(|w| w[0] < w[1]));
            assert_eq!(
                g.corners.len(),
                g.tags.len() * gridcl_refcell::vertices(g.cell).len()
            );
        }
        let quads = &m.blocks[1];
        assert_eq!(quads.cell, CellType::Quadrilateral);
        let file = read_file(&fixtures().join("two_regions_p1.msh")).unwrap();
        let fb = file
            .element_blocks
            .iter()
            .find(|b| b.element_type == 3)
            .unwrap();
        let tags: Vec<i64> = quads.nodes[..4]
            .iter()
            .map(|&i| m.node_tags[i as usize])
            .collect();
        assert_eq!(tags, [fb.nodes[0], fb.nodes[1], fb.nodes[3], fb.nodes[2]]);
    }

    /// The cells of `m` by element tag: (cell, degree, coordinate bits in internal order,
    /// vertex tags, elementary tag, physical tags).
    type CellKey = (CellType, u32, Vec<u64>, Vec<i64>, i32, Vec<i32>);

    fn cells_by_tag(m: &GmshMesh) -> HashMap<i64, CellKey> {
        let mut out = HashMap::new();
        for b in &m.blocks {
            let k = b.nodes.len() / b.tags.len().max(1);
            let nv = gridcl_refcell::vertices(b.cell).len();
            let physical = physical_sets(b.tags.len(), &b.physical);
            for (c, row) in b.nodes.chunks_exact(k).enumerate() {
                let coords = row
                    .iter()
                    .flat_map(|&i| node(m, i).iter().map(|x| x.to_bits()))
                    .collect();
                let vertices = row[..nv].iter().map(|&i| m.node_tags[i as usize]).collect();
                let key = (
                    b.cell,
                    b.degree,
                    coords,
                    vertices,
                    b.elementary[c],
                    physical[c].clone(),
                );
                assert!(out.insert(b.tags[c], key).is_none());
            }
        }
        out
    }

    /// The tag elements of `m`: (cell, corner tags, elementary tag, physical tags), sorted.
    fn tag_elements(m: &GmshMesh) -> Vec<(CellType, Vec<i64>, i32, Vec<i32>)> {
        let mut out = Vec::new();
        for g in &m.tag_groups {
            let nv = gridcl_refcell::vertices(g.cell).len();
            let physical = physical_sets(g.tags.len(), &g.physical);
            for (e, corners) in g.corners.chunks_exact(nv).enumerate() {
                out.push((
                    g.cell,
                    corners.to_vec(),
                    g.elementary[e],
                    physical[e].clone(),
                ));
            }
        }
        out.sort();
        out
    }

    /// `to_gmsh_mesh(from_gmsh_mesh(m))` is `m` up to the documented renumbering, for every
    /// corpus file: the same cells by element tag (cell, degree, exact coordinates, vertex
    /// tags, elementary and physical tags), tag elements, names and dimensions. Written in both
    /// encodings, the file parses back unchanged and mshio reads it (except with 18-node
    /// prisms, ADR-0023).
    #[test]
    fn convert_from_gmsh_mesh_round_trips_the_corpus() {
        for name in mesh_fixtures() {
            let m = convert(&name);
            let file = from_gmsh_mesh(&m).unwrap_or_else(|e| panic!("{name}: {e}"));
            let back = to_gmsh_mesh(&file, &Serial, Some(m.gdim)).unwrap();
            assert_eq!((back.tdim, back.gdim), (m.tdim, m.gdim), "{name}");
            assert_eq!(cells_by_tag(&back), cells_by_tag(&m), "{name}: cells");
            assert_eq!(
                tag_elements(&back),
                tag_elements(&m),
                "{name}: tag elements"
            );
            assert_eq!(back.names, m.names, "{name}: names");
            // Cells keep their tags; tag elements follow them.
            let max_cell = m.blocks.iter().flat_map(|b| b.tags.iter()).max().unwrap();
            for g in &back.tag_groups {
                assert!(
                    g.tags.iter().all(|t| t > max_cell),
                    "{name}: tag element tags"
                );
            }
            let prisms_p2 = m
                .blocks
                .iter()
                .any(|b| b.cell == CellType::Prism && b.degree == 2);
            for encoding in [crate::gmsh::Encoding::Ascii, crate::gmsh::Encoding::Binary] {
                let mut bytes = Vec::new();
                file.write_to(&mut bytes, encoding).unwrap();
                assert_eq!(
                    crate::gmsh::parse(&bytes).unwrap(),
                    file,
                    "{name} {encoding:?}"
                );
                if !prisms_p2 {
                    mshio::parse_msh_bytes(&bytes)
                        .unwrap_or_else(|e| panic!("{name} {encoding:?}: mshio: {e:?}"));
                }
            }
        }
    }

    /// Origin ids below 1 (MeshBuilder's defaults start at 0) are shifted so that node and
    /// element tags start at 1; non-vertex nodes follow the largest vertex tag, tag elements
    /// the largest cell tag.
    #[test]
    fn convert_from_gmsh_mesh_shifts_tags() {
        // One P2 triangle with vertex ids -2, 0, 5 and an edge as a tag element.
        let mesh = GmshMesh {
            tdim: 2,
            gdim: 2,
            nodes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5],
            node_tags: vec![-2, 0, 5, 99, 99, 99],
            blocks: vec![GmshCellBlock {
                cell: CellType::Triangle,
                degree: 2,
                nodes: vec![0, 1, 2, 3, 5, 4],
                tags: vec![-1],
                elementary: vec![3],
                physical: vec![(0, 7)],
            }],
            tag_groups: vec![GmshTagGroup {
                dim: 1,
                cell: CellType::Interval,
                corners: vec![-2, 0],
                tags: vec![1000],
                elementary: vec![4],
                physical: vec![],
            }],
            names: vec![],
        };
        let file = from_gmsh_mesh(&mesh).unwrap();
        // Vertices -2, 0, 5 and the edge nodes 6, 7, 8, all shifted by 3.
        assert_eq!(file.node_blocks[0].tags, [1, 3, 8, 9, 10, 11]);
        // Cell -1 and tag element 0 (after the cell), shifted by 2.
        let line = &file.element_blocks[0];
        assert_eq!(
            (line.element_type, &line.tags[..], &line.nodes[..]),
            (1, &[2][..], &[1, 3][..])
        );
        let tri = &file.element_blocks[1];
        assert_eq!((tri.element_type, &tri.tags[..]), (9, &[1][..]));
        // Gmsh order: vertices, then the edges (0,1), (1,2), (2,0); internal 3, 5, 4.
        assert_eq!(tri.nodes, [1, 3, 8, 9, 10, 11]);
        let e = file.entities.as_ref().unwrap();
        assert_eq!((e.curves[0].tag, e.surfaces[0].tag), (4, 3));
        assert_eq!(e.surfaces[0].physical_tags, [7]);
        assert_eq!(file.node_blocks[0].entity_tag, 3);
    }

    /// Malformed meshes give errors, not panics: a cell node beyond the nodes, and block or
    /// group arrays that disagree with their element counts.
    #[test]
    fn convert_from_gmsh_mesh_rejects_malformed_meshes() {
        let good = GmshMesh {
            tdim: 2,
            gdim: 2,
            nodes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            node_tags: vec![1, 2, 3],
            blocks: vec![GmshCellBlock {
                cell: CellType::Triangle,
                degree: 1,
                nodes: vec![0, 1, 2],
                tags: vec![1],
                elementary: vec![1],
                physical: vec![],
            }],
            tag_groups: vec![GmshTagGroup {
                dim: 1,
                cell: CellType::Interval,
                corners: vec![1, 2],
                tags: vec![1],
                elementary: vec![2],
                physical: vec![],
            }],
            names: vec![],
        };
        assert!(from_gmsh_mesh(&good).is_ok());
        type Edit = (&'static str, fn(&mut GmshMesh));
        let edits: [Edit; 5] = [
            ("cell node", |m| m.blocks[0].nodes[2] = 3),
            ("cell nodes", |m| m.blocks[0].nodes.push(0)),
            ("cell elementary", |m| m.blocks[0].elementary.clear()),
            ("group corners", |m| {
                m.tag_groups[0].corners.pop();
            }),
            ("group elementary", |m| m.tag_groups[0].elementary.push(2)),
        ];
        for (what, edit) in edits {
            let mut bad = good.clone();
            edit(&mut bad);
            match from_gmsh_mesh(&bad) {
                Err(GridError::IndexOutOfRange { .. } | GridError::LengthMismatch { .. }) => {}
                r => panic!("{what}: {:?}", r.err()),
            }
        }
    }

    /// One entity per (dimension, elementary tag, physical set), with bounding boxes from its
    /// nodes, the point's coordinates for dimension 0 and empty boundaries; a second physical
    /// set on one elementary tag gets a fresh entity tag; an untagged mesh gets one entity 1.
    #[test]
    fn convert_from_gmsh_mesh_entities() {
        let mesh = GmshMesh {
            tdim: 1,
            gdim: 3,
            nodes: vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 2.0, -1.0],
            node_tags: vec![1, 2, 3],
            blocks: vec![GmshCellBlock {
                cell: CellType::Interval,
                degree: 1,
                nodes: vec![0, 1, 1, 2],
                tags: vec![1, 2],
                elementary: vec![5, 5],
                physical: vec![(0, 1), (1, 2)],
            }],
            tag_groups: vec![GmshTagGroup {
                dim: 0,
                cell: CellType::Point,
                corners: vec![3],
                tags: vec![1],
                elementary: vec![9],
                physical: vec![(0, 4)],
            }],
            names: vec![],
        };
        let file = from_gmsh_mesh(&mesh).unwrap();
        let e = file.entities.unwrap();
        let p = &e.points[0];
        assert_eq!(
            (e.points.len(), p.tag, p.min, p.max),
            (1, 9, [3.0, 2.0, -1.0], [3.0, 2.0, -1.0])
        );
        type Curve = (i32, [f64; 3], [f64; 3], Vec<i32>);
        let curves: Vec<Curve> = e
            .curves
            .iter()
            .map(|c| (c.tag, c.min, c.max, c.physical_tags.clone()))
            .collect();
        assert_eq!(
            curves,
            [
                (5, [0.0, 0.0, 0.0], [1.0, 2.0, 0.0], vec![1]),
                (6, [1.0, 2.0, -1.0], [3.0, 2.0, 0.0], vec![2]),
            ]
        );
        assert!(e.curves.iter().all(|c| c.boundary.is_empty()));
        assert_eq!(file.node_blocks[0].entity_tag, 5);
        let mut plain = mesh.clone();
        plain.blocks[0].elementary = vec![1, 1];
        plain.blocks[0].physical.clear();
        plain.tag_groups.clear();
        let e = from_gmsh_mesh(&plain).unwrap().entities.unwrap();
        assert_eq!((e.curves.len(), e.curves[0].tag), (1, 1));
    }
}
