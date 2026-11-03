//! Host downloads of a mesh for the writers (ADR-0035, items 11 and 12): one rank's piece (all
//! local cells with their ghost flags, all local nodes), the owned part gathered to rank 0
//! (cells by (rank, owned order), nodes in global node order) with data selected alike, and
//! the global Gmsh numbering of nodes and tag elements.

use std::collections::BTreeSet;

use gridcl_device::{GridError, Real, Result};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::DeviceGrid;
use gridcl_io::gmsh::GmshTagGroup;
use gridcl_parallel::{Comm, CommDyn, CommExt, CommScalar, IndexMap};
use gridcl_topology::decode_origin_ids;

use crate::Mesh;
use crate::gmsh::{elementary_of, values_by_index};
use crate::pipeline::agree;

/// The cells of one block on the host.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DownloadBlock {
    pub(crate) cell: CellType,
    pub(crate) degree: u32,
    /// Indices into [`Download::nodes`], `[cell][nodes per cell]` in A.5 order.
    pub(crate) nodes: Vec<u32>,
    /// The origin id of every cell.
    pub(crate) origin_ids: Vec<i64>,
    /// Cells `[0, num_owned)` are owned, the others ghosts.
    pub(crate) num_owned: usize,
}

/// Cells and nodes of a mesh on the host.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Download {
    pub(crate) gdim: usize,
    /// `[node][gdim]`, as the writers write them (`Mesh::equispaced_host_nodes`).
    pub(crate) nodes: Vec<f64>,
    /// One entry per cell block of the mesh, empty blocks included.
    pub(crate) blocks: Vec<DownloadBlock>,
}

/// The global index of every local node of `map`: the owned range, then the ghosts.
fn global_indices(map: &IndexMap) -> Vec<i64> {
    let start = map.owned_range().0;
    (start..start + map.num_owned() as i64)
        .chain(map.ghosts().iter().copied())
        .collect()
}

/// `rows` of every rank, concatenated in rank order on rank 0; empty elsewhere. Collective.
fn gather_rows<T: CommScalar>(comm: &dyn CommDyn, rows: &[T]) -> Result<Vec<T>> {
    let mut counts = vec![0; comm.size()];
    counts[0] = rows.len();
    Ok(comm.all_to_all_v(rows, &counts)?.0)
}

/// `rows · components` values, or `LengthMismatch`.
fn checked_len(len: usize, rows: usize, components: usize) -> Result<usize> {
    let expected = rows.checked_mul(components).ok_or(GridError::TooLarge)?;
    if len != expected {
        return Err(GridError::LengthMismatch {
            expected,
            found: len,
        });
    }
    Ok(expected)
}

/// The piece of this rank: every local cell, owned before ghost in each block, and every local
/// node, in local order.
///
/// # Errors
///
/// `TooLarge` if a count overflows; device errors.
pub(crate) fn piece<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<Download>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut blocks = Vec::with_capacity(mesh.cell_blocks().len());
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let geometry = mesh.geometry_block(b);
        let cells = block.index_map.num_local();
        let len = cells
            .checked_mul(geometry.dofmap.stride as usize)
            .ok_or(GridError::TooLarge)?;
        let mut nodes = geometry.dofmap.data.to_host()?;
        nodes.truncate(len);
        let mut origin_ids = decode_origin_ids(&block.origin_ids.to_host()?);
        origin_ids.truncate(cells);
        blocks.push(DownloadBlock {
            cell: block.cell.into(),
            degree: geometry.element.degree,
            nodes,
            origin_ids,
            num_owned: block.index_map.num_owned(),
        });
    }
    Ok(Download {
        gdim: GD::N,
        nodes: mesh.equispaced_host_nodes()?,
        blocks,
    })
}

/// The owned cells and nodes of every rank on rank 0: the cells of each block by (rank, owned
/// order), the nodes in global node order (the owned nodes of every rank, in rank order).
/// Other ranks get the blocks without cells and no nodes. Collective, with an agreed status.
///
/// # Errors
///
/// `TooLarge` if a global node index does not fit u32; device errors; agreed.
pub(crate) fn gather<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<Download>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let comm = mesh.comm();
    let local = piece(mesh).and_then(|mut p| {
        let global = global_indices(mesh.node_index_map());
        for block in &mut p.blocks {
            let k = gridcl_refcell::num_lagrange_nodes(block.cell, block.degree);
            block.nodes.truncate(block.num_owned * k);
            block.origin_ids.truncate(block.num_owned);
            for n in &mut block.nodes {
                *n = u32::try_from(global[*n as usize]).map_err(|_| GridError::TooLarge)?;
            }
        }
        p.nodes.truncate(mesh.node_index_map().num_owned() * GD::N);
        Ok(p)
    });
    let local = agree(comm, local)?;
    let mut blocks = Vec::with_capacity(local.blocks.len());
    for block in local.blocks {
        let nodes = gather_rows(comm, &block.nodes)?;
        let origin_ids = gather_rows(comm, &block.origin_ids)?;
        blocks.push(DownloadBlock {
            num_owned: origin_ids.len(),
            nodes,
            origin_ids,
            ..block
        });
    }
    Ok(Download {
        gdim: GD::N,
        nodes: gather_rows(comm, &local.nodes)?,
        blocks,
    })
}

/// Cell data in flat local cell order, `components` values per cell, selected and gathered
/// as [`gather`] gathers the cells. Collective, with an agreed status.
///
/// # Errors
///
/// `LengthMismatch` unless `data` has `components` values per local cell; agreed.
pub(crate) fn gather_cell_data<F, TD, GD, C, T>(
    mesh: &Mesh<F, TD, GD, C>,
    data: &[T],
    components: usize,
) -> Result<Vec<T>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
    T: CommScalar,
{
    let comm = mesh.comm();
    let local = (|| {
        let blocks = mesh.cell_blocks();
        let cells = blocks
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.index_map.num_local()))
            .ok_or(GridError::TooLarge)?;
        checked_len(data.len(), cells, components)?;
        let mut first = 0;
        Ok(blocks
            .iter()
            .map(|b| {
                let owned = first..first + b.index_map.num_owned() * components;
                first += b.index_map.num_local() * components;
                owned
            })
            .collect::<Vec<_>>())
    })();
    let mut out = Vec::new();
    for owned in agree(comm, local)? {
        out.extend(gather_rows(comm, &data[owned])?);
    }
    Ok(out)
}

/// Point data in local node order, `components` values per node, selected and gathered as
/// [`gather`] gathers the nodes. Collective, with an agreed status.
///
/// # Errors
///
/// `LengthMismatch` unless `data` has `components` values per local node; agreed.
pub(crate) fn gather_point_data<F, TD, GD, C, T>(
    mesh: &Mesh<F, TD, GD, C>,
    data: &[T],
    components: usize,
) -> Result<Vec<T>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
    T: CommScalar,
{
    let comm = mesh.comm();
    let map = mesh.node_index_map();
    let owned =
        checked_len(data.len(), map.num_local(), components).map(|_| map.num_owned() * components);
    let owned = agree(comm, owned)?;
    gather_rows(comm, &data[..owned])
}

/// The global Gmsh numbering of a mesh (ADR-0035, item 11), before the shift of tags below 1.
/// On one rank it is ADR-0022 item 7's numbering.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GmshNumbering {
    /// The tag of every local node: a vertex node's vertex origin id; any other node's
    /// M + 1 + k, with M the largest vertex origin id over all ranks (0 without vertices) and k
    /// its position among the non-vertex nodes in global node order.
    pub(crate) node_tags: Vec<i64>,
    /// The smallest node tag over all ranks; `None` without nodes.
    pub(crate) node_min: Option<i64>,
    /// The smallest cell origin id over all ranks; `None` without cells.
    pub(crate) cell_min: Option<i64>,
    /// This rank's owned tagged entities below D_TOP, one group per tagged entity block, in
    /// (dimension, block, flat index) order: corners as vertex origin ids, `tags` the element
    /// tags, which follow the largest cell origin id over all ranks by an exscan over the owned
    /// tagged entities.
    pub(crate) tag_groups: Vec<GmshTagGroup>,
}

/// The local part of [`gmsh_numbering`], before any collective step.
struct LocalNumbering {
    /// The vertex origin id of every vertex node, `None` for the other nodes.
    vertex_tags: Vec<Option<i64>>,
    max_vertex: Option<i64>,
    max_cell: Option<i64>,
    min_cell: Option<i64>,
    /// The tag groups with element tags counted from 0 on this rank.
    tag_groups: Vec<GmshTagGroup>,
}

/// The owned tagged entities below D_TOP of `mesh`, with local element positions as tags.
fn owned_tag_groups<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    vertex_ids: &[i64],
) -> Result<Vec<GmshTagGroup>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let tags = mesh.tags();
    let mut groups = Vec::new();
    let mut next = 0i64;
    for dim in 0..TD::N {
        let elementary = values_by_index(tags.elementary(dim))?;
        let physical = values_by_index(tags.physical(dim))?;
        if elementary.is_empty() && physical.is_empty() {
            continue;
        }
        for block in mesh.entity_blocks(dim)? {
            let owned =
                u32::try_from(block.index_map.num_owned()).map_err(|_| GridError::TooLarge)?;
            let range = block.offset..block.offset + owned;
            let tagged: Vec<u32> = elementary
                .keys()
                .chain(physical.keys())
                .copied()
                .filter(|i| range.contains(i))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if tagged.is_empty() {
                continue;
            }
            let rows = block.vertices.data.to_host()?;
            let nv = block.vertices.stride as usize;
            let mut group = GmshTagGroup {
                dim,
                cell: block.cell,
                corners: Vec::with_capacity(tagged.len() * nv),
                tags: Vec::with_capacity(tagged.len()),
                elementary: Vec::with_capacity(tagged.len()),
                physical: Vec::new(),
            };
            for (e, &i) in tagged.iter().enumerate() {
                let local = (i - block.offset) as usize;
                group.corners.extend(
                    rows[local * nv..(local + 1) * nv]
                        .iter()
                        .map(|&v| vertex_ids[v as usize]),
                );
                group.tags.push(next);
                next += 1;
                group.elementary.push(elementary_of(&elementary, i));
                if let Some(values) = physical.get(&i) {
                    group.physical.extend(values.iter().map(|&v| (e as u32, v)));
                }
            }
            groups.push(group);
        }
    }
    Ok(groups)
}

/// The local, fallible part of [`gmsh_numbering`].
fn local_numbering<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<LocalNumbering>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let vertex_ids = decode_origin_ids(&mesh.vertex_origin_ids().to_host()?);
    let mut vertex_tags = vec![None; mesh.node_index_map().num_local()];
    let (mut max_vertex, mut max_cell, mut min_cell) = (None, None, None);
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let geometry = mesh.geometry_block(b);
        let dofmap = geometry.dofmap.data.to_host()?;
        let vertices = block.vertices.data.to_host()?;
        let k = geometry.dofmap.stride as usize;
        let nv = block.vertices.stride as usize;
        let cells = block.index_map.num_local();
        for c in 0..cells {
            for j in 0..nv {
                let id = vertex_ids[vertices[c * nv + j] as usize];
                vertex_tags[dofmap[c * k + j] as usize] = Some(id);
                max_vertex = max_vertex.max(Some(id));
            }
        }
        for &id in decode_origin_ids(&block.origin_ids.to_host()?)
            .iter()
            .take(cells)
        {
            max_cell = max_cell.max(Some(id));
            min_cell = Some(min_cell.map_or(id, |m: i64| m.min(id)));
        }
    }
    Ok(LocalNumbering {
        vertex_tags,
        max_vertex,
        max_cell,
        min_cell,
        tag_groups: owned_tag_groups(mesh, &vertex_ids)?,
    })
}

/// The largest `x` over the ranks that have one; `None` if none has. Collective.
fn max_over(comm: &dyn CommDyn, x: Option<i64>) -> Option<i64> {
    let any = comm.max(u32::from(x.is_some())) == 1;
    any.then(|| comm.max(x.unwrap_or(i64::MIN)))
}

/// The smallest `x` over the ranks that have one; `None` if none has. Collective.
fn min_over(comm: &dyn CommDyn, x: Option<i64>) -> Option<i64> {
    let any = comm.max(u32::from(x.is_some())) == 1;
    any.then(|| comm.min(x.unwrap_or(i64::MAX)))
}

/// `start + count`, or `TooLarge`.
fn checked_end(start: i64, count: u64) -> Result<i64> {
    i64::try_from(count)
        .ok()
        .and_then(|c| start.checked_add(c))
        .ok_or(GridError::TooLarge)
}

/// The global Gmsh numbering of `mesh` (ADR-0035, item 11). Collective, with an agreed
/// status: the owners number their non-vertex nodes by an exscan and send the tags to the
/// ranks that hold them as ghosts.
///
/// # Errors
///
/// `EntitiesNotCreated` for tags of a dimension without entities; `TooLarge` if a tag
/// overflows; device errors; agreed.
pub(crate) fn gmsh_numbering<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<GmshNumbering>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let comm = mesh.comm();
    let local = agree(comm, local_numbering(mesh))?;
    let map = mesh.node_index_map();
    let owned = map.num_owned();

    // Owned nodes: vertex origin ids, the others M + 1 + k by an exscan.
    let m = max_over(comm, local.max_vertex).unwrap_or(0);
    let others = local.vertex_tags[..owned]
        .iter()
        .filter(|t| t.is_none())
        .count() as u64;
    let first = comm.exscan_sum(others);
    let total = comm.sum(others);
    // The same on every rank, so agreed.
    checked_end(m, total)?;
    let mut next = m + first as i64;
    let mut node_tags: Vec<i64> = local.vertex_tags[..owned]
        .iter()
        .map(|t| {
            t.unwrap_or_else(|| {
                next += 1;
                next
            })
        })
        .collect();

    // Ghost nodes: their tags from the owners, asked by global index.
    let ghosts = map.ghosts();
    let owners = map.ghost_owners();
    let mut order: Vec<usize> = (0..ghosts.len()).collect();
    order.sort_by_key(|&g| owners[g]);
    let mut counts = vec![0usize; comm.size()];
    for &o in owners {
        counts[o as usize] += 1;
    }
    let asked: Vec<i64> = order.iter().map(|&g| ghosts[g]).collect();
    let (requests, from) = comm.all_to_all_v(&asked, &counts)?;
    let start = map.owned_range().0;
    let replies: Vec<i64> = requests
        .iter()
        .map(|&g| node_tags[(g - start) as usize])
        .collect();
    let (answers, _) = comm.all_to_all_v(&replies, &from)?;
    let mut ghost_tags = vec![0i64; ghosts.len()];
    for (&g, &t) in order.iter().zip(&answers) {
        ghost_tags[g] = t;
    }
    node_tags.extend(ghost_tags);

    // Tag elements: after the largest cell origin id, by an exscan over the owned ones.
    let max_cell = max_over(comm, local.max_cell).unwrap_or(0);
    let count: u64 = local.tag_groups.iter().map(|g| g.tags.len() as u64).sum();
    let offset = comm.exscan_sum(count);
    let total = comm.sum(count);
    checked_end(max_cell, total)?;
    let mut tag_groups = local.tag_groups;
    for t in tag_groups.iter_mut().flat_map(|g| g.tags.iter_mut()) {
        // offset + t < total, so no overflow.
        *t = max_cell + (offset as i64 + *t + 1);
    }

    let node_min = min_over(comm, node_tags.iter().copied().min());
    Ok(GmshNumbering {
        node_tags,
        node_min,
        cell_min: min_over(comm, local.min_cell),
        tag_groups,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};
    use std::path::{Path, PathBuf};

    use gridcl_testing::ThreadComm;

    use super::*;
    use crate::{AnyMesh, Block, GhostMode, GmshOptions, Serial};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gridcl-io/tests/fixtures/gmsh")
            .join(name)
    }

    /// Files of every cell dimension, mixed blocks, degree 2 and tags in every dimension.
    const FILES: [&str; 4] = [
        "triangle_p2.msh",
        "two_regions_p1.msh",
        "hybrid_p2.msh",
        "tetrahedron_p2-binary.msh",
    ];

    /// Cell origin id → (cell type, degree, node coordinate bits in cell order).
    type CellMap = BTreeMap<i64, (CellType, u32, Vec<u64>)>;

    fn cell_map(d: &Download) -> CellMap {
        let mut out = CellMap::new();
        for b in &d.blocks {
            let k = gridcl_refcell::num_lagrange_nodes(b.cell, b.degree);
            for (c, &id) in b.origin_ids.iter().enumerate() {
                let bits = b.nodes[c * k..(c + 1) * k]
                    .iter()
                    .flat_map(|&n| &d.nodes[n as usize * d.gdim..(n as usize + 1) * d.gdim])
                    .map(|x| x.to_bits())
                    .collect();
                out.insert(id, (b.cell, b.degree, bits));
            }
        }
        out
    }

    fn read<C: Comm>(name: &str, comm: C, mode: GhostMode) -> AnyMesh<f32, C> {
        let client = gridcl_testing::test_client();
        let options = GmshOptions::default().partitioner(Block).ghosts(mode);
        crate::gmsh::read::<f32, C>(&client, comm, fixture(name), options).unwrap()
    }

    /// What one rank returns from the gather tests.
    #[derive(Debug)]
    struct Gathered {
        piece: CellMap,
        /// Each block's (local cells, owned cells, index map's local and owned counts).
        counts: Vec<(usize, usize, usize, usize)>,
        gathered: Download,
        /// Cell origin ids attached as cell data and node global indices as point data,
        /// gathered.
        cell_data: Vec<i64>,
        point_data: Vec<i64>,
    }

    fn gathered<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Gathered
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let p = piece(mesh).unwrap();
        let counts = p
            .blocks
            .iter()
            .zip(mesh.cell_blocks())
            .map(|(b, m)| {
                (
                    b.origin_ids.len(),
                    b.num_owned,
                    m.index_map.num_local(),
                    m.index_map.num_owned(),
                )
            })
            .collect();
        let ids: Vec<i64> = p.blocks.iter().flat_map(|b| b.origin_ids.clone()).collect();
        let globals = global_indices(mesh.node_index_map());
        let pairs: Vec<i64> = globals.iter().flat_map(|&g| [g, -g]).collect();
        Gathered {
            piece: cell_map(&p),
            counts,
            gathered: gather(mesh).unwrap(),
            cell_data: gather_cell_data(mesh, &ids, 1).unwrap(),
            point_data: gather_point_data(mesh, &pairs, 2).unwrap(),
        }
    }

    /// On 1–4 threads, with and without ghosts: the gathered cells by origin id equal the
    /// one-rank piece (types, degrees, coordinates bitwise); nodes come in global order; cell
    /// and point data follow the cells and nodes; the piece holds every local cell with the
    /// index map's owned count, and its ghosts are owned elsewhere.
    #[test]
    fn gathered_cells_equal_the_serial_download() {
        for name in FILES {
            let serial =
                crate::with_mesh!(read(name, Serial, GhostMode::None).as_ref(), m => gathered(m));
            assert_eq!(serial.piece, cell_map(&serial.gathered), "{name}");
            for p in 1..=4 {
                for mode in [GhostMode::None, GhostMode::SharedFacet] {
                    let at = format!("{name}, {p} threads, {mode:?}");
                    let out = ThreadComm::run(
                        p,
                        |comm| crate::with_mesh!(read(name, comm, mode).as_ref(), m => gathered(m)),
                    );
                    let root = &out[0];
                    assert_eq!(cell_map(&root.gathered), serial.piece, "{at}");
                    let ids: Vec<i64> = root
                        .gathered
                        .blocks
                        .iter()
                        .flat_map(|b| b.origin_ids.clone())
                        .collect();
                    assert_eq!(root.cell_data, ids, "{at}");
                    let n = root.gathered.nodes.len() / root.gathered.gdim;
                    let want: Vec<i64> = (0..n as i64).flat_map(|g| [g, -g]).collect();
                    assert_eq!(root.point_data, want, "{at}");
                    assert!(
                        root.gathered
                            .blocks
                            .iter()
                            .all(|b| b.num_owned == b.origin_ids.len())
                    );
                    let mut owners: HashMap<i64, usize> = HashMap::new();
                    for (r, o) in out.iter().enumerate() {
                        for &(local, owned, map_local, map_owned) in &o.counts {
                            assert_eq!((local, owned), (map_local, map_owned), "{at}");
                        }
                        if r > 0 {
                            assert!(
                                o.gathered.nodes.is_empty() && o.cell_data.is_empty(),
                                "{at}"
                            );
                            assert!(
                                o.gathered.blocks.iter().all(|b| b.origin_ids.is_empty()),
                                "{at}"
                            );
                        }
                    }
                    for (r, o) in out.iter().enumerate() {
                        for (id, cell) in &o.piece {
                            assert_eq!(serial.piece.get(id), Some(cell), "{at}: cell {id}");
                            owners.entry(*id).or_insert(r);
                        }
                    }
                    assert_eq!(owners.len(), serial.piece.len(), "{at}");
                }
            }
        }
    }

    /// The ghost flags of a piece: per block, the cells from `num_owned` on are the ghosts,
    /// each owned by exactly one other rank.
    #[test]
    fn piece_ghost_flags_follow_the_index_maps() {
        for name in ["triangle_p2.msh", "hybrid_p2.msh"] {
            for p in 2..=4 {
                let at = format!("{name}, {p} threads");
                let out = ThreadComm::run(p, |comm| {
                    let mesh = read(name, comm, GhostMode::SharedFacet);
                    let p = crate::with_mesh!(mesh.as_ref(), m => piece(m)).unwrap();
                    let mut owned = Vec::new();
                    let mut ghosts = Vec::new();
                    for b in &p.blocks {
                        owned.extend_from_slice(&b.origin_ids[..b.num_owned]);
                        ghosts.extend_from_slice(&b.origin_ids[b.num_owned..]);
                    }
                    (owned, ghosts)
                });
                let mut owner: HashMap<i64, usize> = HashMap::new();
                for (r, (owned, _)) in out.iter().enumerate() {
                    for &id in owned {
                        assert!(owner.insert(id, r).is_none(), "{at}: cell {id} owned twice");
                    }
                }
                assert!(out.iter().any(|(_, g)| !g.is_empty()), "{at}: ghosts exist");
                for (r, (_, ghosts)) in out.iter().enumerate() {
                    for id in ghosts {
                        assert!(
                            owner.get(id).is_some_and(|&o| o != r),
                            "{at}: ghost {id} of rank {r}"
                        );
                    }
                }
            }
        }
    }

    /// The M4 file of a one-rank mesh: node tags and (element tag, sorted corner node tags) of
    /// its tag elements.
    fn m4_numbering<F, TD, GD, C>(
        mesh: &Mesh<F, TD, GD, C>,
    ) -> (Vec<i64>, BTreeSet<(i64, Vec<i64>)>)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let file = gridcl_io::gmsh::from_gmsh_mesh(&crate::gmsh::download(mesh).unwrap()).unwrap();
        let tags = file
            .node_blocks
            .iter()
            .flat_map(|b| b.tags.clone())
            .collect();
        let mut elements = BTreeSet::new();
        for b in file.element_blocks.iter().filter(|b| b.dim < TD::N) {
            let k = b.nodes.len() / b.tags.len();
            for (e, &t) in b.tags.iter().enumerate() {
                let mut corners = b.nodes[e * k..(e + 1) * k].to_vec();
                corners.sort_unstable();
                elements.insert((t, corners));
            }
        }
        (tags, elements)
    }

    /// `numbering` shifted as `from_gmsh_mesh` shifts: node tags and the tag elements.
    fn shifted(n: &GmshNumbering) -> (Vec<i64>, BTreeSet<(i64, Vec<i64>)>) {
        let shift = |min: Option<i64>| min.filter(|&m| m < 1).map_or(0, |m| 1 - m);
        let (ns, es) = (shift(n.node_min), shift(n.cell_min));
        let mut elements = BTreeSet::new();
        for g in &n.tag_groups {
            let nv = g.corners.len() / g.tags.len();
            for (e, &t) in g.tags.iter().enumerate() {
                let mut corners: Vec<i64> = g.corners[e * nv..(e + 1) * nv]
                    .iter()
                    .map(|c| c + ns)
                    .collect();
                corners.sort_unstable();
                elements.insert((t + es, corners));
            }
        }
        (n.node_tags.iter().map(|t| t + ns).collect(), elements)
    }

    /// On one rank the numbering is M4's: every node tag and every tag element of the file
    /// `from_gmsh_mesh` writes, also for a builder mesh with origin ids from 0 (shifted).
    #[test]
    fn one_rank_numbering_equals_m4() {
        for name in FILES {
            let mesh = read(name, Serial, GhostMode::None);
            let (m4, numbering) = crate::with_mesh!(mesh.as_ref(), m => (m4_numbering(m), gmsh_numbering(m).unwrap()));
            assert_eq!(shifted(&numbering), m4, "{name}");
        }
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::cube_box(crate::Cell3::Tetrahedron, 2, 2);
        let element = crate::CoordinateElement::<crate::D3> {
            cell: crate::Cell3::Tetrahedron,
            degree: 2,
            family: crate::NodeFamily::Equispaced,
        };
        let mesh: crate::VolumeMesh<f32> = crate::MeshBuilder::new(&client)
            .nodes(&h.nodes, None)
            .cell_block(element, &h.blocks[0].cells, None)
            .build()
            .unwrap();
        let numbering = gmsh_numbering(&mesh).unwrap();
        assert_eq!((numbering.node_min, numbering.cell_min), (Some(0), Some(0)));
        assert_eq!(shifted(&numbering), m4_numbering(&mesh));
    }

    /// The largest vertex origin id of `mesh` over all ranks.
    fn max_vertex_id<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> i64
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let ids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
        mesh.comm().max(ids.into_iter().max().unwrap_or(i64::MIN))
    }

    /// Tagged entity (sorted corner ids) → (dimension, elementary tag) of the owned groups.
    fn entities(n: &GmshNumbering) -> BTreeMap<Vec<i64>, (usize, i32)> {
        let mut out = BTreeMap::new();
        for g in &n.tag_groups {
            let nv = g.corners.len() / g.tags.len();
            for e in 0..g.tags.len() {
                let mut corners = g.corners[e * nv..(e + 1) * nv].to_vec();
                corners.sort_unstable();
                assert!(out.insert(corners, (g.dim, g.elementary[e])).is_none());
            }
        }
        out
    }

    /// On 1–4 threads, with and without ghosts: every rank holding a node gives it the same
    /// tag; the vertex tags are the one-rank ones and the others are M + 1, …, M + K in global
    /// node order; the owned tag entities of all ranks are the one-rank ones, each owned once,
    /// and their tags are the one-rank tags; the minima are the one-rank minima.
    #[test]
    fn parallel_numbering_is_global() {
        for name in FILES {
            let serial = read(name, Serial, GhostMode::None);
            let (serial, m) = crate::with_mesh!(serial.as_ref(), s => (gmsh_numbering(s).unwrap(), max_vertex_id(s)));
            let mut serial_tags = serial.node_tags.clone();
            serial_tags.sort_unstable();
            let mut serial_numbers: Vec<i64> = serial
                .tag_groups
                .iter()
                .flat_map(|g| g.tags.clone())
                .collect();
            serial_numbers.sort_unstable();
            for p in 1..=4 {
                for mode in [GhostMode::None, GhostMode::SharedFacet] {
                    let at = format!("{name}, {p} threads, {mode:?}");
                    let out = ThreadComm::run(p, |comm| {
                        let mesh = read(name, comm, mode);
                        crate::with_mesh!(mesh.as_ref(), x => (gmsh_numbering(x).unwrap(), global_indices(x.node_index_map())))
                    });
                    let mut by_global: BTreeMap<i64, i64> = BTreeMap::new();
                    let mut all = BTreeMap::new();
                    let mut numbers = Vec::new();
                    for (n, globals) in &out {
                        assert_eq!(n.node_tags.len(), globals.len(), "{at}");
                        for (&g, &t) in globals.iter().zip(&n.node_tags) {
                            assert_eq!(*by_global.entry(g).or_insert(t), t, "{at}: node {g}");
                        }
                        for (corners, v) in entities(n) {
                            assert!(all.insert(corners, v).is_none(), "{at}: owned once");
                        }
                        numbers.extend(n.tag_groups.iter().flat_map(|g| g.tags.clone()));
                        assert_eq!(
                            (n.node_min, n.cell_min),
                            (serial.node_min, serial.cell_min),
                            "{at}"
                        );
                    }
                    assert_eq!(all, entities(&serial), "{at}");
                    numbers.sort_unstable();
                    assert_eq!(numbers, serial_numbers, "{at}: tag-element tags");
                    let tags: Vec<i64> = by_global.values().copied().collect();
                    let mut sorted = tags.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, serial_tags, "{at}: the same tags");
                    let others: Vec<i64> = tags.into_iter().filter(|&t| t > m).collect();
                    let want: Vec<i64> = (m + 1..m + 1 + others.len() as i64).collect();
                    assert_eq!(others, want, "{at}: non-vertex tags in global node order");
                }
            }
        }
    }
}
