//! Facet classification (M9 T2, ADR-0039 item 11): `facet_cell_counts` gives every local facet
//! its global number of cells, in every `GhostMode`, and the single cell and local facet of
//! each boundary facet.

use std::collections::BTreeMap;

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_dims::{Cell2, Cell3, CellType};
use gridcl_refcell::sub_entities;
use gridcl_testing::parallel::{self, Case, Distribution, Family, GHOST_MODES, MeshCheck};
use gridcl_testing::{HostMesh, ThreadComm};
use gridcl_topology::{decode_origin_ids, facet_cell_counts};

/// The number of cells of every facet of `h`, keyed by its sorted vertex origin ids.
fn host_counts(h: &HostMesh) -> BTreeMap<Vec<i64>, u32> {
    let mut out = BTreeMap::new();
    for block in &h.blocks {
        let k = block.nodes_per_cell();
        let dim = gridcl_refcell::tdim(block.cell) - 1;
        for c in 0..block.num_cells() {
            for sub in sub_entities(block.cell, dim) {
                let mut key: Vec<i64> = sub
                    .iter()
                    .map(|&v| h.node_id(block.cells[c * k + v] as usize))
                    .collect();
                key.sort_unstable();
                *out.entry(key).or_insert(0) += 1;
            }
        }
    }
    out
}

/// What one rank's mesh gives: (counts, boundary facets) with every local facet's sorted
/// vertex origin ids, after checking the single cell of each boundary facet.
fn device_counts<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Vec<(Vec<i64>, u32)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let dim = TD::N - 1;
    let fc = facet_cell_counts(m.comm(), m.topology()).expect("facet_cell_counts");
    let keys = parallel::entity_rows(m, dim).unwrap();
    let counts = fc.counts.to_host().unwrap();
    let cell = fc.cell.to_host().unwrap();
    let local_facet = fc.local_facet.to_host().unwrap();
    assert_eq!(
        (counts.len(), cell.len(), local_facet.len()),
        (keys.len(), keys.len(), keys.len())
    );

    // The vertex origin ids of every local cell, flat, in reference vertex order.
    let ids = decode_origin_ids(&m.vertex_origin_ids().to_host().unwrap());
    let mut cells: Vec<(CellType, Vec<i64>)> = Vec::new();
    for b in m.cell_blocks() {
        let rows = b.vertices.data.to_host().unwrap();
        for r in rows.chunks_exact(b.vertices.stride as usize) {
            cells.push((b.cell.into(), r.iter().map(|&v| ids[v as usize]).collect()));
        }
    }
    for f in 0..keys.len() {
        if counts[f] == 1 {
            let (ty, vs) = &cells[cell[f] as usize];
            let mut key: Vec<i64> = sub_entities(*ty, dim)[local_facet[f] as usize]
                .iter()
                .map(|&v| vs[v])
                .collect();
            key.sort_unstable();
            assert_eq!(
                key, keys[f],
                "facet {f}: cell {} local facet {}",
                cell[f], local_facet[f]
            );
        } else {
            assert_eq!((cell[f], local_facet[f]), (u32::MAX, u32::MAX), "facet {f}");
        }
    }
    keys.into_iter().zip(counts).collect()
}

/// Every local facet's count on `comm` equals the host count of its key.
struct Check<'a> {
    comm: &'a ThreadComm,
    client: &'a Client,
}

impl MeshCheck for Check<'_> {
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let reference = host_counts(host);
        let mesh = parallel::build::<F, TD, GD, _>(
            self.comm.clone(),
            self.client,
            host,
            case,
            CellOrdering::Input,
        )
        .unwrap_or_else(|e| panic!("{case}: {e}"));
        for (key, count) in device_counts(&mesh) {
            assert_eq!(
                Some(&count),
                reference.get(&key),
                "{case}, rank {}: facet {key:?}",
                self.comm.rank()
            );
        }
    }
}

const FAMILIES: [Family; 11] = [
    Family::Cube(Cell3::Tetrahedron),
    Family::Cube(Cell3::Hexahedron),
    Family::Cube(Cell3::Prism),
    Family::Cube(Cell3::Pyramid),
    Family::Mixed,
    Family::Square(Cell2::Triangle),
    Family::Square(Cell2::Quadrilateral),
    Family::Interval,
    Family::Sphere,
    Family::Book,
    Family::YJunction,
];

/// The boundary facets of `h` on one rank.
fn boundary_count<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(h: &HostMesh) -> usize {
    let client = gridcl_testing::test_client();
    let mut mesh = h.build::<f32, TD, GD>(&client).expect("build");
    mesh.create_entities(TD::N - 1).unwrap();
    let counts = device_counts(&mesh);
    let host = host_counts(h);
    assert_eq!(counts.len(), host.len());
    for (key, count) in &counts {
        assert_eq!(Some(count), host.get(key), "{key:?}");
    }
    counts.iter().filter(|(_, c)| *c == 1).count()
}

#[test]
fn boundary_facets_of_boxes_in_closed_form() {
    let n = 3;
    assert_eq!(
        boundary_count::<D1, D1>(&gridcl_testing::interval_box(n, 2)),
        2
    );
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        assert_eq!(
            boundary_count::<D2, D2>(&gridcl_testing::square_box(cell, n, 1)),
            4 * n
        );
    }
    // Six faces of n × n squares, each square one quadrilateral or two triangles.
    let squares = 6 * n * n;
    assert_eq!(
        boundary_count::<D3, D3>(&gridcl_testing::cube_box(Cell3::Hexahedron, n, 1)),
        squares
    );
    assert_eq!(
        boundary_count::<D3, D3>(&gridcl_testing::cube_box(Cell3::Tetrahedron, n, 1)),
        2 * squares
    );
    for (h, n) in [
        (gridcl_testing::cube_box(Cell3::Prism, n, 2), n),
        (gridcl_testing::cube_box(Cell3::Pyramid, n, 1), n),
        (gridcl_testing::mixed_box(n, 1), n),
        (gridcl_testing::hex_prism_pyramid_box(2, 1), 2),
    ] {
        let expected = host_counts(&h).values().filter(|&&c| c == 1).count();
        assert!(expected >= 6 * n * n);
        assert_eq!(boundary_count::<D3, D3>(&h), expected);
    }
    // Closed surfaces and curves have none.
    assert_eq!(boundary_count::<D2, D3>(&gridcl_testing::sphere(1, 1)), 0);
    assert_eq!(boundary_count::<D1, D2>(&gridcl_testing::circle(12, 1)), 0);
}

#[test]
fn three_cells_on_an_edge() {
    let client = gridcl_testing::test_client();
    let mut mesh = gridcl_testing::book(3)
        .build::<f32, D2, D3>(&client)
        .unwrap();
    mesh.create_entities(1).unwrap();
    let counts = device_counts(&mesh);
    // The spine has three pages; the other 2 · 3 edges of each page are boundary.
    assert_eq!(counts.iter().filter(|(_, c)| *c == 3).count(), 1);
    assert_eq!(counts.iter().filter(|(_, c)| *c == 1).count(), 6);
}

#[test]
fn needs_facet_entities() {
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    assert!(matches!(
        facet_cell_counts(mesh.comm(), mesh.topology()),
        Err(GridError::EntitiesNotCreated(1))
    ));
}

/// Counts on 1–4 threads in every ghost mode equal the global counts, also for ranks that
/// hold no boundary facets (the sphere).
#[test]
fn counts_match_one_rank_in_every_ghost_mode() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for (i, family) in FAMILIES.into_iter().enumerate() {
                for (g, ghosts) in GHOST_MODES.into_iter().enumerate() {
                    let case = Case {
                        family,
                        degree: 1,
                        ghosts,
                        distribution: [Distribution::Even, Distribution::RoundRobin][(i + g) % 2],
                        double: false,
                    };
                    parallel::dispatch(
                        &client,
                        &case,
                        &mut Check {
                            comm: &comm,
                            client: &client,
                        },
                    );
                }
            }
        });
    }
}
