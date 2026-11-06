//! Submeshes: boundary, facet, edge and cell submeshes with their `EntityMap`
//! (docs/design.md, "Tags, regions and submeshes"; ADR-0039, items 1 and 3-11).
//!
//! A submesh is built in stages: a mask over the parent entities and its compaction per parent
//! block (`select`), the child topology with the cell and vertex maps (`topology`), then the
//! child geometry, flags and tags. Kernels plan as `KernelKind::Scalar`.

mod closure;
mod flags;
mod geometry;
mod nodes;
mod numbering;
mod select;
mod sharers;
mod tags;
mod topology;

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, KernelKind, LaunchPlan, Real, Result,
    debug_assert_bindings, histogram,
};
use gridcl_dims::{CellType, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{Geometry, NodeFamily};
use gridcl_parallel::{Comm, CommDyn, CommExt};
use gridcl_topology::facet_cell_counts;

use crate::entity_map::EntityMapParts;
use crate::{CurveMesh, EntityMap, Mesh, MeshTags};
use select::ParentBlock;
use topology::ChildTopology;

/// u32 of `x`, or `TooLarge`.
pub(super) fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// A scalar launch over `len` u32 items.
pub(super) fn plan(client: &Client, len: usize) -> Result<LaunchPlan> {
    LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)
}

/// `dst[base + i] = src[i]` for i < len.
#[cube(launch_unchecked)]
fn place_kernel(src: &[u32], dst: &mut [u32], len: u32, base: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            dst[base as usize + i] = src[i];
        }
    }
}

/// `out[i] = i` for i < len.
#[cube(launch_unchecked)]
fn iota_kernel(out: &mut [u32], len: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            out[i] = i as u32;
        }
    }
}

/// Copies the first `len` words of `src` to `dst[base..base + len]`.
///
/// # Panics
///
/// If `src` holds fewer than `len` words or `dst` fewer than `base + len`.
pub(super) fn place(
    src: &DeviceBuffer<u32>,
    dst: &DeviceBuffer<u32>,
    base: usize,
    len: usize,
) -> Result<()> {
    assert!(
        len <= src.len() && base + len <= dst.len(),
        "place: out of range"
    );
    if len == 0 {
        return Ok(());
    }
    let client = src.client();
    let launch = plan(client, len)?;
    debug_assert_bindings(client, 2);
    // SAFETY: the kernel reads src[i] and writes dst[base + i] for i < len, within both
    // buffers by the assertion above.
    unsafe {
        place_kernel::launch_unchecked(
            client,
            launch.cube_count,
            launch.cube_dim,
            src.arg(),
            dst.arg(),
            fit(len)?,
            fit(base)?,
            launch.items_per_unit,
        );
    }
    Ok(())
}

/// The buffer 0, 1, …, len − 1.
pub(super) fn iota(client: &Client, len: usize) -> Result<DeviceBuffer<u32>> {
    let out = DeviceBuffer::<u32>::uninit(client, len)?;
    if len > 0 {
        let launch = plan(client, len)?;
        debug_assert_bindings(client, 1);
        // SAFETY: the kernel writes out[i] for i < len, its length.
        unsafe {
            iota_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                out.arg(),
                fit(len)?,
                launch.items_per_unit,
            );
        }
    }
    Ok(out)
}

/// The child topology of a facet submesh of `mesh`: every boundary facet (`tags = None`) or
/// the tagged facets, which must be boundary facets (ADR-0039, items 5-8 and 11). Collective.
///
/// # Errors
///
/// `DimensionMismatch` for tags of another dimension than tdim − 1; `EntitiesNotCreated(tdim
/// − 1)`; agreed: `NonManifold`, `InteriorFacet`, `TooLarge` (origin ids beyond i64);
/// allocation, device and communication errors.
pub(crate) fn facet_topology<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    tags: Option<&MeshTags>,
) -> Result<ChildTopology<TD::Lower>>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim,
{
    let comm: &dyn CommDyn = &mesh.comm;
    let fdim = TD::N - 1;
    if let Some(t) = tags
        && t.dim != fdim
    {
        return Err(GridError::DimensionMismatch {
            expected: (fdim, GD::N),
            found: (t.dim, GD::N),
        });
    }
    let parent = &mesh.topology;
    let facet_blocks = parent.entity_blocks(fdim)?;
    let facet_cells = facet_cell_counts(comm, parent)?;
    let tagged = match tags {
        Some(t) => Some(histogram(&t.indices, fit(facet_cells.counts.len())?)?),
        None => None,
    };
    let mask = select::facet_mask(comm, &facet_cells, tagged.as_ref())?;
    let blocks: Vec<ParentBlock> = facet_blocks
        .iter()
        .map(|b| ParentBlock {
            cell: b.cell,
            offset: b.offset as usize,
            index_map: &b.index_map,
        })
        .collect();
    let mut selected = Vec::new();
    let mut present = Vec::new();
    let mut overflow = false;
    for c in select::select(comm, &mask, &blocks)? {
        let (s, o, p) = select::facet_children(parent, &facet_cells, c)?;
        overflow |= o;
        selected.push(s);
        present.push(p);
    }
    if comm.max(u32::from(overflow)) != 0 {
        return Err(GridError::TooLarge);
    }
    let mut child = topology::child_topology(comm, parent, selected, true)?;
    child.present = present;
    Ok(child)
}

/// A child geometry and its `node_to_parent`.
pub(crate) type ChildGeometry<F, CTD, GD> = (Geometry<F, CTD, GD>, DeviceBuffer<u32>);

/// The child geometry of the facet children `child` of `mesh`: each child block takes
/// the agreed element of its parent cells, and its dofmap through the closure node maps
/// (ADR-0039, items 8 and 9). Also returns `node_to_parent`. Collective.
///
/// # Errors
///
/// Agreed: `UnsupportedDegree` or `UnsupportedNodeFamily` if the parent cells of a child block
/// differ in degree or node family; allocation, device and communication errors.
pub(crate) fn facet_geometry<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    child: &ChildTopology<TD::Lower>,
) -> Result<ChildGeometry<F, TD::Lower, GD>>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    let comm: &dyn CommDyn = &mesh.comm;
    let parent_blocks = mesh.geometry.blocks();
    let elements: Vec<(u32, NodeFamily)> = parent_blocks
        .iter()
        .map(|b| (b.element.degree, b.element.family))
        .collect();
    let mut blocks = Vec::new();
    for ((block, (cells, facets)), present) in child
        .topology
        .cell_blocks()
        .iter()
        .zip(&child.facet_blocks)
        .zip(&child.present)
    {
        let cell: CellType = block.cell.into();
        let (degree, family) = geometry::agreed_element(comm, cell, &elements, present)?;
        let k = gridcl_refcell::num_lagrange_nodes(cell, degree);
        let tables = parent_blocks
            .iter()
            .zip(present)
            .map(|(b, &p)| {
                p.then(|| closure::closure_table(b.element.cell.into(), cell, degree, family))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let rows: Vec<_> = parent_blocks
            .iter()
            .zip(mesh.topology.cell_blocks())
            .zip(&tables)
            .map(|((g, t), table)| select::RowSource {
                rows: &g.dofmap,
                num_local: t.index_map.num_local(),
                table: table.as_deref(),
            })
            .collect();
        let dofmap = select::restrict_rows(cells, facets, &rows, k)?;
        blocks.push((
            cell,
            degree,
            family,
            DeviceStrided {
                data: dofmap,
                stride: fit(k)?,
            },
        ));
    }
    let node_rows: Vec<nodes::NodeRows<'_>> = blocks
        .iter()
        .zip(child.topology.cell_blocks())
        .map(|(b, t)| nodes::NodeRows {
            vertices: &t.vertices,
            dofmap: &b.3,
            num_owned: t.index_map.num_owned(),
        })
        .collect();
    let items = nodes::child_nodes(
        comm,
        &mesh.topology,
        mesh.geometry.node_index_map(),
        child.topology.vertex_index_map(),
        &node_rows,
    )?;
    drop(node_rows);
    geometry::child_geometry(mesh.geometry.nodes(), items, mesh.geometry.origin(), blocks)
}

/// The facet submesh of `mesh` on every boundary facet (`tags = None`) or on the tagged
/// facets: child topology, geometry, flags, `EntityMap` and tags. Collective.
#[allow(clippy::type_complexity)] // Listing B4's signature
fn facet_mesh<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    tags: Option<&MeshTags>,
) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    let fdim = TD::N - 1;
    let child = facet_topology(mesh, tags)?;
    let (geometry, node_to_parent) = facet_geometry(mesh, &child)?;
    let flags = flags::facet_flags(mesh, &child)?;
    let ChildTopology {
        mut topology,
        child_to_parent,
        parent_to_child,
        vertex_to_parent,
        facet_of,
        ..
    } = child;
    for (b, f) in flags.into_iter().enumerate() {
        topology.set_cell_flags(b, f)?;
    }
    let map = EntityMap::from_parts(EntityMapParts {
        parent_dim: fdim,
        child_to_parent,
        parent_to_child,
        vertex_to_parent,
        node_to_parent,
        facet_of,
    });
    let num_facets = mesh
        .topology
        .entity_blocks(fdim)?
        .iter()
        .map(|b| b.index_map.num_local())
        .sum();
    let tags = tags::child_tags(&mesh.tags, &map, num_facets, fdim, &mesh.client)?;
    let mut child = Mesh::from_parts(mesh.client.clone(), mesh.comm.clone(), topology, geometry);
    child.set_tags(tags);
    Ok((child, map))
}

/// `Mesh::boundary`.
#[allow(clippy::type_complexity)] // Listing B4's signature
pub(crate) fn boundary<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    facet_mesh(mesh, None)
}

/// `Mesh::facet_submesh`.
#[allow(clippy::type_complexity)] // Listing B4's signature
pub(crate) fn facet_submesh<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    facets: &MeshTags,
) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    facet_mesh(mesh, Some(facets))
}

/// `Mesh::edge_submesh`.
pub(crate) fn edge_submesh<F: Real, C: Comm>(
    _mesh: &Mesh<F, D3, D3, C>,
    _edges: &MeshTags,
) -> Result<(CurveMesh<F, C>, EntityMap)> {
    todo!("edge_submesh: M9 T6")
}

/// `Mesh::cell_submesh`.
pub(crate) fn cell_submesh<F, TD, GD, C>(
    _mesh: &Mesh<F, TD, GD, C>,
    _cells: &MeshTags,
) -> Result<(Mesh<F, TD, GD, C>, EntityMap)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    todo!("cell_submesh: M9 T6")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, HashMap};

    use cubecl::prelude::Client;
    use gridcl_dims::{CellType, D2, D3, Dim, EmbeddedIn, TopoDim};
    use gridcl_testing::HostMesh;
    use gridcl_topology::{Topology, decode_origin_ids};

    use super::{facet_geometry, facet_topology};
    use crate::prelude::*;

    /// `h` through the crate's own builder (gridcl-testing's `build` gives the dependency's
    /// `Mesh`), with the entities of every dimension below the cells.
    fn build<TD, GD>(client: &Client, h: &HostMesh) -> Mesh<f32, TD, GD>
    where
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        build_family(client, h, NodeFamily::Equispaced)
    }

    /// [`build`] with coordinate elements of node family `family`.
    fn build_family<TD, GD>(client: &Client, h: &HostMesh, family: NodeFamily) -> Mesh<f32, TD, GD>
    where
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let mut builder =
            MeshBuilder::<f32, TD, GD>::new(client).nodes(&h.nodes, h.node_ids.as_deref());
        for block in &h.blocks {
            let element = CoordinateElement::<TD> {
                cell: TD::Cell::try_from(block.cell).expect("cell dimension"),
                degree: block.degree,
                family,
            };
            builder = builder.cell_block(element, &block.cells, block.cell_ids.as_deref());
        }
        let mut mesh = builder.build().expect("build");
        for d in 0..TD::N {
            mesh.create_entities(d).expect("entities");
        }
        mesh
    }

    /// SplitMix64 steps from `state`.
    fn next(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Device tags of dimension `dim` with value 1 on `indices`.
    fn tags(client: &Client, dim: usize, indices: &[u32]) -> MeshTags {
        MeshTags {
            dim,
            indices: DeviceBuffer::from_host(client, indices).expect("indices"),
            values: DeviceBuffer::from_host(client, &vec![1_i32; indices.len()]).expect("values"),
        }
    }

    /// The facets of `h` with one cell, as sorted vertex origin ids: the host oracle.
    fn host_boundary(h: &HostMesh) -> BTreeSet<Vec<i64>> {
        let mut count: HashMap<Vec<i64>, usize> = HashMap::new();
        for b in &h.blocks {
            let k = b.nodes_per_cell();
            let tdim = gridcl_refcell::tdim(b.cell);
            for row in b.cells.chunks_exact(k) {
                for f in gridcl_refcell::sub_entities(b.cell, tdim - 1) {
                    let mut key: Vec<i64> = f.iter().map(|&v| h.node_id(row[v] as usize)).collect();
                    key.sort_unstable();
                    *count.entry(key).or_default() += 1;
                }
            }
        }
        count
            .into_iter()
            .filter(|(_, c)| *c == 1)
            .map(|(k, _)| k)
            .collect()
    }

    /// The vertex origin ids of every row of `data` (stride `stride`), by vertex ids `ids`.
    fn id_rows(data: &DeviceBuffer<u32>, stride: u32, ids: &[i64]) -> Vec<Vec<i64>> {
        data.to_host()
            .expect("rows")
            .chunks_exact(stride as usize)
            .map(|r| r.iter().map(|&v| ids[v as usize]).collect())
            .collect()
    }

    /// The flat cell rows of `t` as vertex origin ids, in reference order.
    fn cell_rows<TD: TopoDim>(t: &Topology<TD>) -> Vec<Vec<i64>> {
        let ids = decode_origin_ids(&t.vertex_origin_ids().to_host().expect("ids"));
        t.cell_blocks()
            .iter()
            .flat_map(|b| id_rows(&b.vertices.data, b.vertices.stride, &ids))
            .collect()
    }

    /// Checks the child of the facet selection `chosen` (flat facets, ascending; `None`: the
    /// boundary) of `mesh`: the parents are the chosen facets, the maps invert each other, child
    /// rows are the parent cells' local facet rows, vertex ids are the parents', and child
    /// origin ids are 8 · id(parent cell) + local facet. Returns the child topology.
    fn check<TD>(mesh: &Mesh<f32, TD, TD>, chosen: Option<&[u32]>) -> Topology<TD::Lower>
    where
        TD: TopoDim + EmbeddedIn<TD>,
        TD::Lower: TopoDim,
    {
        let fdim = TD::N - 1;
        let t = chosen.map(|c| tags(mesh.client(), fdim, c));
        let child = facet_topology(mesh, t.as_ref()).expect("facet_topology");
        let c2p = child.child_to_parent.to_host().expect("c2p");
        let (parents, children) = (
            child.parent_to_child.0.to_host().expect("parents"),
            child.parent_to_child.1.to_host().expect("children"),
        );
        if let Some(c) = chosen {
            assert_eq!(parents, c, "parents are the chosen facets");
        }
        assert!(parents.windows(2).all(|w| w[0] < w[1]));
        for (&p, &c) in parents.iter().zip(&children) {
            assert_eq!(c2p[c as usize], p);
        }
        let parent_ids =
            decode_origin_ids(&mesh.topology.vertex_origin_ids().to_host().expect("ids"));
        let child_ids =
            decode_origin_ids(&child.topology.vertex_origin_ids().to_host().expect("ids"));
        let v2p = child.vertex_to_parent.to_host().expect("v2p");
        assert_eq!(v2p.len(), child_ids.len());
        for (v, &p) in v2p.iter().enumerate() {
            assert_eq!(child_ids[v], parent_ids[p as usize], "child vertex {v}");
        }
        // Facet rows of the parent: canonical entity rows (sorted) and cell-local rows.
        let mut facet_rows = Vec::new();
        for b in mesh.topology.entity_blocks(fdim).expect("facets") {
            facet_rows.extend(id_rows(&b.vertices.data, b.vertices.stride, &parent_ids));
        }
        let parent_cells = cell_rows(&mesh.topology);
        let mut cell_ids = Vec::new();
        let mut cell_types = Vec::new();
        for b in mesh.topology.cell_blocks() {
            cell_ids.extend(decode_origin_ids(
                &b.origin_ids.to_host().expect("cell ids"),
            ));
            let cell: CellType = b.cell.into();
            cell_types.extend(std::iter::repeat_n(cell, b.index_map.num_local()));
        }
        let (cells, facets) = child.facet_of.as_ref().expect("facet children");
        let (cells, facets) = (
            cells.to_host().expect("cells"),
            facets.to_host().expect("facets"),
        );
        let mut child_cell_ids = Vec::new();
        for b in child.topology.cell_blocks() {
            child_cell_ids.extend(decode_origin_ids(&b.origin_ids.to_host().expect("ids")));
        }
        for (c, row) in cell_rows(&child.topology).iter().enumerate() {
            let mut sorted = row.clone();
            sorted.sort_unstable();
            let mut want = facet_rows[c2p[c] as usize].clone();
            want.sort_unstable();
            assert_eq!(
                sorted, want,
                "child cell {c}: vertices of parent facet {}",
                c2p[c]
            );
            let (pc, f) = (cells[c] as usize, facets[c] as usize);
            let local: Vec<i64> = gridcl_refcell::sub_entities(cell_types[pc], TD::N - 1)[f]
                .iter()
                .map(|&v| parent_cells[pc][v])
                .collect();
            assert_eq!(row, &local, "child cell {c}: local facet order");
            assert_eq!(
                child_cell_ids[c],
                8 * cell_ids[pc] + f as i64,
                "child origin id {c}"
            );
        }
        child.topology
    }

    #[test]
    fn boundary_topology_of_boxes() {
        let client = gridcl_testing::test_client();
        for n in 1..=3 {
            for cell in [
                Cell3::Tetrahedron,
                Cell3::Hexahedron,
                Cell3::Prism,
                Cell3::Pyramid,
            ] {
                let h = gridcl_testing::cube_box(cell, n, 1);
                boundary_of::<D3>(&client, &h, &format!("{cell:?} box n={n}"));
            }
            let h = gridcl_testing::mixed_box(n, 1);
            boundary_of::<D3>(&client, &h, &format!("mixed box n={n}"));
            for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
                let h = gridcl_testing::square_box(cell, n, 1);
                boundary_of::<D2>(&client, &h, &format!("{cell:?} square n={n}"));
            }
        }
        // Closed forms: 12n² triangles of a tetrahedral box, 6n² quadrilaterals of a hexahedral
        // one, with 6n² + 2 vertices each.
        for n in 1..=3 {
            for (cell, per_square) in [(Cell3::Tetrahedron, 2), (Cell3::Hexahedron, 1)] {
                let mesh = build::<D3, D3>(&client, &gridcl_testing::cube_box(cell, n, 1));
                let child = check(&mesh, None);
                let cells: usize = child
                    .cell_blocks()
                    .iter()
                    .map(|b| b.index_map.num_local())
                    .sum();
                assert_eq!(cells, 6 * per_square * n * n, "{cell:?} n={n}");
                assert_eq!(
                    child.vertex_index_map().num_local(),
                    6 * n * n + 2,
                    "{cell:?} n={n}"
                );
            }
        }
    }

    /// The boundary of `h` equals the host oracle, and its Euler characteristic is 2 (a sphere)
    /// for volume meshes and 0 (a circle) for planar ones.
    fn boundary_of<TD>(client: &Client, h: &HostMesh, what: &str)
    where
        TD: TopoDim + EmbeddedIn<TD>,
        TD::Lower: TopoDim,
    {
        let mesh = build::<TD, TD>(client, h);
        let mut child = check(&mesh, None);
        let mut got: Vec<Vec<i64>> = cell_rows(&child)
            .into_iter()
            .map(|mut r| {
                r.sort_unstable();
                r
            })
            .collect();
        got.sort_unstable();
        let want: Vec<Vec<i64>> = host_boundary(h).into_iter().collect();
        assert_eq!(got, want, "{what}: boundary facets");
        let v = child.vertex_index_map().num_local() as i64;
        let c = got.len() as i64;
        let euler = if TD::N == 3 {
            child.create_entities(&Serial, 1).expect("edges");
            let e: usize = child
                .entity_blocks(1)
                .expect("edges")
                .iter()
                .map(|b| b.index_map.num_local())
                .sum();
            v - e as i64 + c
        } else {
            v - c
        };
        assert_eq!(
            euler,
            if TD::N == 3 { 2 } else { 0 },
            "{what}: Euler characteristic"
        );
    }

    #[test]
    fn random_facet_subsets() {
        let client = gridcl_testing::test_client();
        let mut rng = 7_u64;
        for h in [
            gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1).with_shuffled_ids(1),
            gridcl_testing::mixed_box(2, 1).with_shuffled_ids(2),
            gridcl_testing::hex_prism_pyramid_box(2, 1).with_shuffled_ids(3),
        ] {
            let mesh = build::<D3, D3>(&client, &h);
            check(&mesh, None);
            let child = facet_topology(&mesh, None).expect("boundary");
            let boundary = child.child_to_parent.to_host().expect("c2p");
            let mut sorted = boundary.clone();
            sorted.sort_unstable();
            for _ in 0..4 {
                let chosen: Vec<u32> = sorted
                    .iter()
                    .copied()
                    .filter(|_| next(&mut rng) & 1 == 1)
                    .collect();
                check(&mesh, Some(&chosen));
            }
            check(&mesh, Some(&[]));
        }
    }

    #[test]
    fn origin_id_overflow_is_too_large() {
        let client = gridcl_testing::test_client();
        let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let tri = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        let mesh_with = |id: i64| {
            let mut m: Mesh<f32, D2, D2> = MeshBuilder::new(&client)
                .nodes(&coords, None)
                .cell_block(tri, &[0, 1, 2], Some(&[id]))
                .build()
                .expect("one triangle");
            m.create_entities(1).expect("edges");
            m
        };
        // 8 · id + 2 fits i64 for id = 2^60 − 1 and −2^60, not for 2^60 or −2^60 − 1.
        for id in [(1_i64 << 60) - 1, -(1_i64 << 60), 0, -1] {
            let child = check(&mesh_with(id), None);
            assert_eq!(child.cell_blocks()[0].index_map.num_local(), 3);
        }
        for id in [1_i64 << 60, -(1_i64 << 60) - 1, i64::MAX, i64::MIN] {
            let r = facet_topology(&mesh_with(id), None);
            assert!(
                matches!(r, Err(GridError::TooLarge)),
                "id {id}: {:?}",
                r.err().map(|e| e.to_string())
            );
        }
    }

    #[test]
    fn errors() {
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1);
        let mut builder = MeshBuilder::<f32, D2, D2>::new(&client).nodes(&h.nodes, None);
        let quad = CoordinateElement::<D2> {
            cell: Cell2::Quadrilateral,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        builder = builder.cell_block(quad, &h.blocks[0].cells, None);
        let mesh = builder.build().expect("build");
        assert!(matches!(
            facet_topology(&mesh, None),
            Err(GridError::EntitiesNotCreated(1))
        ));
        let mesh = build::<D2, D2>(&client, &h);
        let r = facet_topology(&mesh, Some(&tags(&client, 2, &[0])));
        assert!(matches!(r, Err(GridError::DimensionMismatch { .. })));
        // Every edge: the four interior ones have two cells.
        let all: Vec<u32> = (0..12).collect();
        let r = facet_topology(&mesh, Some(&tags(&client, 1, &all)));
        assert!(matches!(r, Err(GridError::InteriorFacet)));
    }

    /// The boundary geometry of `mesh` is the restriction of the parent's: child nodes are
    /// bitwise parent nodes with the parent's origin, and at interior points of every child
    /// cell the child map equals the parent map at the facet's points (host tabulation in f64
    /// of the same node values).
    fn geometry_restricts<TD>(mesh: &Mesh<f32, TD, TD>, what: &str)
    where
        TD: TopoDim + EmbeddedIn<TD>,
        TD::Lower: TopoDim + EmbeddedIn<TD>,
    {
        let gd = TD::N;
        let child = facet_topology(mesh, None).expect("facet_topology");
        let (geometry, n2p) = facet_geometry(mesh, &child).expect("facet_geometry");
        let pn = mesh.geometry.nodes().to_host().expect("parent nodes");
        let cn = geometry.nodes().to_host().expect("child nodes");
        let n2p = n2p.to_host().expect("node_to_parent");
        assert_eq!(cn.len(), n2p.len() * gd);
        for (i, &p) in n2p.iter().enumerate() {
            for g in 0..gd {
                assert_eq!(
                    cn[i * gd + g].to_bits(),
                    pn[p as usize * gd + g].to_bits(),
                    "{what}: node {i}"
                );
            }
        }
        let bits = |o: TD::Array<f64>| o.as_ref().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(geometry.origin()),
            bits(mesh.geometry.origin()),
            "{what}: origin"
        );

        // Parent cells by flat index: (cell type, element, dofmap row).
        let mut parent = Vec::new();
        for (t, g) in mesh
            .topology
            .cell_blocks()
            .iter()
            .zip(mesh.geometry.blocks())
        {
            let cell: CellType = t.cell.into();
            let e = gridcl_refcell::LagrangeElement::new(cell, g.element.degree, g.element.family)
                .expect("element");
            let rows = g.dofmap.data.to_host().expect("dofmap");
            for row in rows.chunks_exact(g.dofmap.stride as usize) {
                parent.push((cell, e.clone(), row.to_vec()));
            }
        }
        let (cells, facets) = child.facet_of.as_ref().expect("facet children");
        let (cells, facets) = (
            cells.to_host().expect("cells"),
            facets.to_host().expect("facets"),
        );
        let mut c = 0;
        for g in geometry.blocks() {
            let cell: CellType = g.element.cell.into();
            let e = gridcl_refcell::LagrangeElement::new(cell, g.element.degree, g.element.family)
                .expect("element");
            let t: Vec<f64> = match cell {
                CellType::Interval => vec![0.3, 0.85],
                CellType::Triangle => vec![0.2, 0.3, 0.6, 0.1],
                _ => vec![0.3, 0.7, 0.9, 0.2],
            };
            let nq = t.len() / (TD::N - 1);
            let tab = e.tabulate(&t, 0).expect("tabulate");
            let eval = |tab: &gridcl_refcell::Tabulation,
                        nodes: &[f32],
                        row: &[u32],
                        q: usize|
             -> Vec<f64> {
                (0..gd)
                    .map(|d| {
                        row.iter()
                            .enumerate()
                            .map(|(j, &n)| tab.get(0, q, j) * nodes[n as usize * gd + d] as f64)
                            .sum()
                    })
                    .collect()
            };
            for row in g
                .dofmap
                .data
                .to_host()
                .expect("dofmap")
                .chunks_exact(g.dofmap.stride as usize)
            {
                let (pcell, pe, prow) = &parent[cells[c] as usize];
                let tp = gridcl_refcell::facet_points_to_cell(*pcell, facets[c] as usize, &t);
                let ptab = pe.tabulate(&tp, 0).expect("tabulate");
                for q in 0..nq {
                    let x = eval(&tab, &cn, row, q);
                    let y = eval(&ptab, &pn, prow, q);
                    let err = x
                        .iter()
                        .zip(&y)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max);
                    assert!(
                        err < 1e-10,
                        "{what}: child cell {c} point {q}: {x:?} vs {y:?}"
                    );
                }
                c += 1;
            }
        }
        assert_eq!(c, cells.len());
    }

    #[test]
    fn child_geometry_restricts_the_parent() {
        let client = gridcl_testing::test_client();
        for p in 2..=3 {
            let mesh = build::<D3, D3>(&client, &gridcl_testing::ball(1, p));
            geometry_restricts(&mesh, &format!("ball p={p}"));
            let mesh = build::<D3, D3>(&client, &gridcl_testing::hex_prism_pyramid_box(2, p));
            geometry_restricts(&mesh, &format!("hex/prism/pyramid box p={p}"));
            let mesh = build::<D3, D3>(&client, &gridcl_testing::cube_box(Cell3::Pyramid, 1, p));
            geometry_restricts(&mesh, &format!("pyramid box p={p}"));
        }
        let mesh = build::<D3, D3>(&client, &gridcl_testing::mixed_box(2, 3).perturbed(0.05));
        geometry_restricts(&mesh, "perturbed mixed box p=3");
        for cell in [Cell3::Tetrahedron, Cell3::Prism] {
            let mesh = build::<D3, D3>(
                &client,
                &gridcl_testing::cube_box(cell, 1, 4).perturbed(0.05),
            );
            geometry_restricts(&mesh, &format!("perturbed {cell:?} box p=4"));
        }
        for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
            let h = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 4).perturbed(0.05);
            geometry_restricts(
                &build_family::<D3, D3>(&client, &h, family),
                &format!("hexahedra p=4 {family:?}"),
            );
            let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 4).perturbed(0.05);
            geometry_restricts(
                &build_family::<D2, D2>(&client, &h, family),
                &format!("quadrilaterals p=4 {family:?}"),
            );
        }
        let h = gridcl_testing::square_box(Cell2::Triangle, 2, 4).perturbed(0.05);
        geometry_restricts(&build::<D2, D2>(&client, &h), "perturbed triangles p=4");
    }

    /// A hexahedron of degree 1 and a disjoint prism of degree 2: their quadrilateral boundary
    /// facets differ in degree.
    #[test]
    fn mixed_degree_facet_type_is_unsupported() {
        let client = gridcl_testing::test_client();
        let mut nodes: Vec<f64> = gridcl_refcell::vertices(CellType::Hexahedron).concat();
        let prism = gridcl_refcell::lagrange_points(CellType::Prism, 2, NodeFamily::Equispaced)
            .expect("prism points");
        nodes.extend(
            prism
                .as_chunks::<3>()
                .0
                .iter()
                .flat_map(|x| [x[0] + 2.0, x[1], x[2]]),
        );
        let element = |cell, degree| CoordinateElement::<D3> {
            cell,
            degree,
            family: NodeFamily::Equispaced,
        };
        let mut mesh: Mesh<f32, D3, D3> = MeshBuilder::new(&client)
            .nodes(&nodes, None)
            .cell_block(
                element(Cell3::Hexahedron, 1),
                &(0..8).collect::<Vec<u32>>(),
                None,
            )
            .cell_block(
                element(Cell3::Prism, 2),
                &(8..26).collect::<Vec<u32>>(),
                None,
            )
            .build()
            .expect("two cells");
        mesh.create_entities(2).expect("faces");
        let child = facet_topology(&mesh, None).expect("facet_topology");
        let r = facet_geometry(&mesh, &child);
        assert!(
            matches!(
                r,
                Err(GridError::UnsupportedDegree {
                    cell: "Quadrilateral",
                    degree: 2
                })
            ),
            "{:?}",
            r.err().map(|e| e.to_string())
        );
    }
}
