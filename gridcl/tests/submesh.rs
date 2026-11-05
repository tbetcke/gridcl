//! M9 acceptance on one rank (docs/plans/M9.md, acceptance coverage): normals on the boundary
//! of a centred ball point outward, ∮ x·n = gdim · V on closed boundaries, and tagged interior
//! facets give `InteriorFacet`; with the `EntityMap` property test (T3) and the edge and cell
//! submeshes (T6).

use std::collections::{BTreeSet, HashMap};

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::parallel::{cell_centroids, check_entity_map, entity_rows, normals_and_flux};
use gridcl_testing::{HostMesh, TOL_F32};
use proptest::prelude::*;

/// `h` with its generated tags (cells 1–3 and 7, boundary facets 100 and 101) on one rank, with
/// the entities of every dimension below the cells.
fn build_tagged<F, TD, GD>(h: &HostMesh, client: &Client) -> Mesh<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let mut mesh = h
        .chunk(0, 1)
        .builder::<F, TD, GD>(client)
        .and_then(MeshBuilder::build)
        .unwrap_or_else(|e| panic!("build {}: {e}", name::<F>()));
    for d in 0..TD::N {
        mesh.create_entities(d).expect("entities");
    }
    mesh
}

/// Device tags of dimension `dim` with value 1 on the local entities `indices` (ascending).
fn tags(client: &Client, dim: usize, indices: &[u32]) -> MeshTags {
    MeshTags {
        dim,
        indices: DeviceBuffer::from_host(client, indices).expect("indices"),
        values: DeviceBuffer::from_host(client, &vec![1_i32; indices.len()]).expect("values"),
    }
}

/// The flat indices of the facets tagged 100 or 101: the boundary facets of the host mesh.
fn boundary_facets<F, TD, GD>(mesh: &Mesh<F, TD, GD>) -> Vec<u32>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let t = mesh.tags().physical(TD::N - 1).expect("facet tags");
    let set: BTreeSet<u32> = t.indices.to_host().expect("indices").into_iter().collect();
    set.into_iter().collect()
}

/// The relative tolerance of the divergence identity: 10^−10 in f64, 10^−5 in f32.
fn flux_tol<F: Real>() -> f64 {
    if size_of::<F>() == 8 { 1e-10 } else { TOL_F32 }
}

// ---------------------------------------------------------------------------------------------
// Acceptance: outward normals on the boundary of a centred ball.

fn ball_normals<F: Real>(client: &Client) {
    for p in 1..=3 {
        let mesh = build_tagged::<F, D3, D3>(&gridcl_testing::ball(1, p), client);
        let (child, _) = mesh.boundary().expect("boundary");
        assert_eq!(child.cell_blocks()[0].index_map.size_global(), 96);
        let (min, _) = normals_and_flux(&child, &[0.0; 3], 2 * p).expect("normals");
        assert!(
            min > 0.0,
            "ball p={p} ({}): n·x = {min:e} at some quadrature point",
            name::<F>()
        );
    }
}

#[test]
fn ball_boundary_normals_point_outward() {
    for_each_precision!(client(), ball_normals);
}

// ---------------------------------------------------------------------------------------------
// Acceptance: ∮ x·n = gdim · V.

/// Checks ∮ x·n = gdim · V on the boundary of the codimension-0 mesh `h`, with Gauss degree 3p
/// on both sides.
fn flux_identity<F, TD>(client: &Client, what: &str, h: &HostMesh, p: u32)
where
    F: Real,
    TD: TopoDim + Codim0<TD>,
    TD::Lower: TopoDim + Codim1<TD>,
{
    let mesh = build_tagged::<F, TD, TD>(h, client);
    let (child, _) = mesh.boundary().expect("boundary");
    let (_, flux) = normals_and_flux(&child, &[0.0; 3], 3 * p).expect("flux");
    let want = TD::N as f64 * mesh.measure(3 * p).expect("measure");
    let err = (flux - want).abs() / want.abs();
    assert!(
        err <= flux_tol::<F>(),
        "{what} p={p} ({}): ∮ x·n = {flux} vs gdim · V = {want} (relative error {err:e})",
        name::<F>()
    );
}

fn divergence<F: Real>(client: &Client) {
    // An affine map with a translation, a shear and a stretch.
    let affine = |x: &[f64]| vec![x[0] + 0.3 * x[1] + 1.5, x[1] - 0.25, 1.2 * x[2] + 0.7];
    for p in 1..=2 {
        for kind in [
            BoxKind::Tetrahedron,
            BoxKind::Hexahedron,
            BoxKind::Prism,
            BoxKind::Mixed,
        ] {
            let h = kind.host(2, p).perturbed(0.05);
            flux_identity::<F, D3>(client, &format!("perturbed {kind:?} box"), &h, p);
        }
        let h = BoxKind::Pyramid.host(2, p).mapped(3, affine);
        flux_identity::<F, D3>(client, "translated pyramid box", &h, p);
        let h = gridcl_testing::hex_prism_pyramid_box(2, p).mapped(3, affine);
        flux_identity::<F, D3>(client, "translated hex/prism/pyramid box", &h, p);
    }
    for p in 1..=3 {
        flux_identity::<F, D3>(client, "ball", &gridcl_testing::ball(1, p), p);
        for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
            let h = gridcl_testing::square_box(cell, 3, p).perturbed(0.05);
            flux_identity::<F, D2>(client, &format!("perturbed {cell:?} square"), &h, p);
        }
    }
}

#[test]
fn divergence_identity_on_closed_boundaries() {
    for_each_precision!(client(), divergence);
}

// ---------------------------------------------------------------------------------------------
// Acceptance: interior facets in facet_submesh.

/// Every facet, and the boundary facets plus one interior facet, give `InteriorFacet`; the
/// boundary facets alone give no error; tags of another dimension give `DimensionMismatch`.
fn rejects_interior<TD>(client: &Client, what: &str, h: &HostMesh)
where
    TD: TopoDim + Codim0<TD>,
    TD::Lower: TopoDim + EmbeddedIn<TD>,
{
    let mesh = build_tagged::<f32, TD, TD>(h, client);
    let fdim = TD::N - 1;
    let num_facets: u32 = mesh
        .entity_blocks(fdim)
        .expect("facets")
        .iter()
        .map(|b| b.index_map.num_local() as u32)
        .sum();
    let boundary = boundary_facets(&mesh);
    let interior = (0..num_facets)
        .find(|f| boundary.binary_search(f).is_err())
        .expect("an interior facet");
    let mut one_more = boundary.clone();
    one_more.push(interior);
    one_more.sort_unstable();
    let all: Vec<u32> = (0..num_facets).collect();
    for (case, indices) in [("every facet", &all), ("one interior facet", &one_more)] {
        let r = mesh.facet_submesh(&tags(client, fdim, indices));
        assert!(
            matches!(r, Err(GridError::InteriorFacet)),
            "{what}, {case}: {:?}",
            r.err()
        );
    }
    let (child, _) = mesh
        .facet_submesh(&tags(client, fdim, &boundary))
        .expect("boundary facets");
    let cells: usize = child
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.num_local())
        .sum();
    assert_eq!(cells, boundary.len(), "{what}: boundary facets");
    let r = mesh.facet_submesh(&tags(client, TD::N, &[0]));
    assert!(
        matches!(r, Err(GridError::DimensionMismatch { .. })),
        "{what}: cell tags: {:?}",
        r.err()
    );
}

#[test]
fn interior_facet_is_rejected() {
    let client = client();
    rejects_interior::<D3>(
        &client,
        "tetrahedra",
        &gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1),
    );
    rejects_interior::<D3>(&client, "mixed box", &gridcl_testing::mixed_box(2, 2));
    rejects_interior::<D2>(
        &client,
        "quadrilaterals",
        &gridcl_testing::square_box(Cell2::Quadrilateral, 3, 1),
    );
}

// ---------------------------------------------------------------------------------------------
// T4b: empty boundaries and child tags.

/// The boundary of a closed boundary is a valid empty mesh.
#[test]
fn boundary_of_a_boundary_is_empty() {
    let client = client();
    for h in [
        gridcl_testing::mixed_box(2, 2),
        gridcl_testing::ball(1, 2),
        gridcl_testing::hex_prism_pyramid_box(1, 1),
    ] {
        let mesh = build_tagged::<f32, D3, D3>(&h, &client);
        let (mut surface, _) = mesh.boundary().expect("boundary");
        surface.create_entities(1).expect("edges");
        let (curve, map) = surface.boundary().expect("boundary of the boundary");
        assert!(curve.cell_blocks().is_empty());
        assert_eq!(curve.vertex_index_map().num_local(), 0);
        assert_eq!(curve.measure(1).expect("measure"), 0.0);
        assert_eq!(map.child_to_parent().len(), 0);
    }
}

/// The physical tags of `m` of dimension `dim`, keyed by the sorted vertex origin ids of each
/// tagged entity.
fn tags_by_key<F, TD, GD>(m: &Mesh<F, TD, GD>, dim: usize) -> HashMap<Vec<i64>, Vec<i32>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let rows = entity_rows(m, dim).expect("rows");
    let mut out: HashMap<Vec<i64>, Vec<i32>> = HashMap::new();
    if let Some(t) = m.tags().physical(dim) {
        let indices = t.indices.to_host().expect("indices");
        let values = t.values.to_host().expect("values");
        assert!(
            indices
                .iter()
                .zip(&values)
                .collect::<Vec<_>>()
                .windows(2)
                .all(|w| w[0] < w[1]),
            "tags sorted by (index, value)"
        );
        for (i, v) in indices.into_iter().zip(values) {
            out.entry(rows[i as usize].clone()).or_default().push(v);
        }
    }
    out
}

/// The child of `boundary` and of `facet_submesh` carries the parent's facet tags of its
/// cells as cell tags, keyed by origin ids, with the facet rows of the name table and their
/// group sizes; the parent's cell tags and names are not carried.
#[test]
fn child_tags_equal_parent_facet_tags() {
    let client = client();
    for (what, h) in [
        ("mixed box", gridcl_testing::mixed_box(2, 1)),
        ("prism box", BoxKind::Prism.host(2, 2)),
    ] {
        let mesh = build_tagged::<f32, D3, D3>(&h, &client);
        let parent = tags_by_key(&mesh, 2);
        assert!(!parent.is_empty(), "{what}: parent facet tags");
        let (child, _) = mesh.boundary().expect("boundary");
        assert_eq!(tags_by_key(&child, 2), parent, "{what}: boundary tags");
        assert_eq!(child.tags().lookup(2, "boundary-100"), Some(100));
        assert_eq!(child.tags().lookup(3, "cells-1"), None);
        assert!(child.tags().physical(3).is_none());
        let group = mesh.tags().group(2, 101).expect("group 101");
        assert_eq!(child.tags().group_size(2, 101), Some(group.indices.len()));
        let (sub, _) = mesh.facet_submesh(&group).expect("facet_submesh");
        let want: HashMap<Vec<i64>, Vec<i32>> = parent
            .iter()
            .filter(|(_, v)| v.contains(&101))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        assert_eq!(tags_by_key(&sub, 2), want, "{what}: facet_submesh tags");
        assert_eq!(sub.tags().group_size(2, 100), Some(0));
    }
}

// ---------------------------------------------------------------------------------------------
// T3: the EntityMap property test.

/// `facet_submesh` of a random half of the boundary facets of `h`: the `EntityMap` checks of
/// `check_entity_map` (child → parent → child is the identity, child vertex lists are the
/// parent facets' as origin ids), and the parents are exactly the tagged facets.
fn round_trip<TD>(client: &Client, h: &HostMesh, seed: u64)
where
    TD: TopoDim + Codim0<TD>,
    TD::Lower: TopoDim + EmbeddedIn<TD>,
{
    let mesh = build_tagged::<f32, TD, TD>(h, client);
    let mut rng = SplitMix::new(seed);
    let chosen: Vec<u32> = boundary_facets(&mesh)
        .into_iter()
        .filter(|_| rng.next_u64() & 1 == 1)
        .collect();
    let (child, map) = mesh
        .facet_submesh(&tags(client, TD::N - 1, &chosen))
        .expect("facet_submesh");
    assert_eq!(map.parent_dim(), TD::N - 1);
    check_entity_map(&mesh, &child, &map, "round trip").expect("EntityMap");
    let (parents, _) = map.parent_to_child();
    assert_eq!(parents.to_host().expect("parents"), chosen);
}

/// The boxes of the property test, by index.
fn round_trip_case(client: &Client, kind: usize, n: usize, seed: u64) {
    const KINDS: [BoxKind; 7] = [
        BoxKind::Tetrahedron,
        BoxKind::Hexahedron,
        BoxKind::Prism,
        BoxKind::Pyramid,
        BoxKind::Mixed,
        BoxKind::Triangle,
        BoxKind::Quadrilateral,
    ];
    let kind = KINDS[kind];
    let h = kind.host(n, 1).with_shuffled_ids(seed);
    match kind.tdim() {
        3 => round_trip::<D3>(client, &h, seed),
        _ => round_trip::<D2>(client, &h, seed),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn entity_map_round_trip(kind in 0..7_usize, n in 1..=4_usize, seed in any::<u64>()) {
        round_trip_case(&client(), kind, n, seed);
    }
}

/// The large case of the property test: about 10^5 facets.
#[test]
fn entity_map_round_trip_large() {
    let h = gridcl_testing::cube_box(Cell3::Tetrahedron, 20, 1);
    round_trip::<D3>(&client(), &h, 3);
}

// ---------------------------------------------------------------------------------------------
// T6: edge and cell submeshes.

/// The length of every edge of the hexahedra of `h`, keyed by sorted vertex origin ids: Gauss
/// rules of `degree` on the cells' maps along their reference edges.
fn host_edge_lengths(h: &HostMesh, degree: u32) -> HashMap<Vec<i64>, f64> {
    let cell = CellType::Hexahedron;
    let rule = gridcl_refcell::gauss_rule(CellType::Interval, degree);
    let refs = gridcl_refcell::vertices(cell);
    let k = h.blocks[0].nodes_per_cell();
    let mut out = HashMap::new();
    for edge in gridcl_refcell::sub_entities(cell, 1) {
        let (a, b) = (refs[edge[0]], refs[edge[1]]);
        let dir: Vec<f64> = (0..3).map(|d| b[d] - a[d]).collect();
        let points: Vec<f64> = rule
            .points
            .iter()
            .flat_map(|&t| (0..3).map(move |d| a[d] + t * (b[d] - a[d])))
            .collect();
        let jac = host_jacobians(h, 0, &points);
        let nq = rule.weights.len();
        for (c, row) in h.blocks[0].cells.chunks_exact(k).enumerate() {
            let mut key = vec![
                h.node_id(row[edge[0]] as usize),
                h.node_id(row[edge[1]] as usize),
            ];
            key.sort_unstable();
            let length: f64 = (0..nq)
                .map(|q| {
                    let j = &jac[(c * nq + q) * 9..(c * nq + q + 1) * 9];
                    let t: Vec<f64> = (0..3)
                        .map(|g| (0..3).map(|d| j[d * 3 + g] * dir[d]).sum())
                        .collect();
                    rule.weights[q] * t.iter().map(|x| x * x).sum::<f64>().sqrt()
                })
                .sum();
            out.entry(key).or_insert(length);
        }
    }
    out
}

fn edges_of_ball<F: Real>(client: &Client) {
    let (p, degree) = (2, 6);
    let h = gridcl_testing::ball(1, p);
    let mut mesh = build_tagged::<F, D3, D3>(&h, client);
    let rows = entity_rows(&mesh, 1).expect("edge rows");
    let even: Vec<u32> = (0..rows.len() as u32)
        .filter(|&e| rows[e as usize][0] % 2 == 0)
        .collect();
    let even_tags = tags(client, 1, &even);
    let r = mesh.edge_submesh(&even_tags);
    assert!(
        matches!(r, Err(GridError::ConnectivityNotCreated(_))),
        "without (1, 3): {:?}",
        r.err()
    );
    mesh.create_connectivity(1, 3).expect("(1, 3)");
    let r = mesh.edge_submesh(&tags(client, 2, &[0]));
    assert!(
        matches!(r, Err(GridError::DimensionMismatch { .. })),
        "face tags: {:?}",
        r.err()
    );
    let (child, map) = mesh.edge_submesh(&even_tags).expect("edge_submesh");
    check_entity_map(&mesh, &child, &map, "edges of the ball").expect("EntityMap");
    let lengths = host_edge_lengths(&h, degree);
    let want: f64 = even.iter().map(|&e| lengths[&rows[e as usize]]).sum();
    let got = child.measure(degree).expect("measure");
    assert!(
        (got - want).abs() <= 10.0 * tol::<F>() * want,
        "edges of the ball ({}): measure {got} vs summed lengths {want}",
        name::<F>()
    );
    for b in child.cell_blocks() {
        assert!(b.flags.to_host().expect("flags").iter().all(|&f| f == 0));
    }
}

#[test]
fn edge_submesh_of_ball() {
    for_each_precision!(client(), edges_of_ball);
}

fn half_box<F: Real>(client: &Client) {
    let mesh =
        build_tagged::<F, D3, D3>(&gridcl_testing::cube_box(Cell3::Hexahedron, 4, 2), client);
    let left: Vec<u32> = cell_centroids(&mesh)
        .expect("centroids")
        .iter()
        .enumerate()
        .filter(|(_, x)| x[0] < 0.5)
        .map(|(c, _)| c as u32)
        .collect();
    assert_eq!(left.len(), 32);
    let (child, map) = mesh
        .cell_submesh(&tags(client, 3, &left))
        .expect("cell_submesh");
    assert_eq!(map.parent_dim(), 3);
    check_entity_map(&mesh, &child, &map, "half box").expect("EntityMap");
    let volume = child.measure(4).expect("measure");
    assert!(
        (volume - 0.5).abs() <= tol::<F>(),
        "half box ({}): volume {volume}",
        name::<F>()
    );
    let r = mesh.cell_submesh(&tags(client, 2, &[0]));
    assert!(
        matches!(r, Err(GridError::DimensionMismatch { .. })),
        "face tags: {:?}",
        r.err()
    );
}

#[test]
fn cell_submesh_of_half_box() {
    for_each_precision!(client(), half_box);
}
