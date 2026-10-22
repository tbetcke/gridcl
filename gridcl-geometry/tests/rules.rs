//! `PointSet`, `QuadRule` and `FacetQuadRule` (M3 T11): reference volumes, mixed meshes, errors,
//! the tabulation cache and layouts, and agreement of the facet rules with gridcl-refcell.

use cubecl::prelude::Client;
use gridcl::DeviceGrid;
use gridcl_device::{DeviceScalar, GridError, Real};
use gridcl_dims::{Cell1, Cell2, Cell3, CellType, D1, D2, D3, TopoDim};
use gridcl_geometry::{CoordinateElement, FacetQuadRule, NodeFamily, QuadRule};
use gridcl_refcell::{
    LagrangeElement, facet_points_to_cell, gauss_rule, reference_facet_normals, sub_entity_types,
};
use gridcl_testing::{TOL_F32, TOL_F64, supports, test_client};

/// The relative tolerance of `F`.
fn tol<F: Real>() -> f64 {
    if size_of::<F>() == 4 {
        TOL_F32
    } else {
        TOL_F64
    }
}

/// Runs `check::<f32>` and, where the device supports it, `check::<f64>`.
macro_rules! both_precisions {
    ($check:ident) => {{
        let client = test_client();
        $check::<f32>(&client);
        if supports::<f64>(&client) {
            $check::<f64>(&client);
        }
    }};
}

/// The volume of a reference cell.
fn reference_volume(cell: CellType) -> f64 {
    match cell {
        CellType::Point => 1.0,
        CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron => 1.0,
        CellType::Triangle | CellType::Prism => 0.5,
        CellType::Tetrahedron => 1.0 / 6.0,
        CellType::Pyramid => 1.0 / 3.0,
    }
}

const CELLS1: [Cell1; 1] = [Cell1::Interval];
const CELLS2: [Cell2; 2] = [Cell2::Triangle, Cell2::Quadrilateral];
const CELLS3: [Cell3; 4] = [
    Cell3::Tetrahedron,
    Cell3::Hexahedron,
    Cell3::Prism,
    Cell3::Pyramid,
];

fn volumes_of<F: Real, TD: TopoDim>(client: &Client, cells: &[TD::Cell]) {
    for degree in 0..=8 {
        let rule = QuadRule::<F, TD>::gauss(client, cells, degree).unwrap();
        for &cell in cells {
            let ps = rule.get(cell).unwrap();
            assert_eq!(ps.cell(), cell);
            assert_eq!(ps.points().len(), ps.len() * TD::N);
            assert!(!ps.is_empty());
            let sum: f64 = ps.weights().iter().sum();
            let volume = reference_volume(cell.into());
            assert!(
                (sum - volume).abs() < 1e-14,
                "{cell:?}, degree {degree}: weights sum to {sum}, not {volume}"
            );
        }
    }
}

fn volumes<F: Real>(client: &Client) {
    volumes_of::<F, D1>(client, &CELLS1);
    volumes_of::<F, D2>(client, &CELLS2);
    volumes_of::<F, D3>(client, &CELLS3);
}

#[test]
fn weights_sum_to_the_reference_volume() {
    both_precisions!(volumes);
}

#[test]
fn repeated_cells_get_one_point_set_in_order_of_first_appearance() {
    let client = test_client();
    let cells = [Cell3::Prism, Cell3::Hexahedron, Cell3::Prism];
    let rule = QuadRule::<f32, D3>::gauss(&client, &cells, 2).unwrap();
    assert_eq!(rule.get(Cell3::Prism).unwrap().cell(), Cell3::Prism);
    assert_eq!(
        rule.get(Cell3::Hexahedron).unwrap().cell(),
        Cell3::Hexahedron
    );
    let facet = FacetQuadRule::<f32, D3>::gauss(&client, &cells, 2).unwrap();
    assert_eq!(facet.get(Cell3::Prism).unwrap().facet_offsets().len(), 6);
}

#[test]
fn one_rule_serves_a_mixed_mesh() {
    let client = test_client();
    let mesh = gridcl_testing::mixed_box(2, 1)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 3).unwrap();
    let facet_rule = FacetQuadRule::for_grid(&mesh, 3).unwrap();
    assert!(mesh.cell_blocks().len() > 1);
    for block in mesh.cell_blocks() {
        let ps = rule.get(block.cell).unwrap();
        assert_eq!(ps.cell(), block.cell);
        assert_eq!(ps.client().name(), client.name());
        assert_eq!(
            facet_rule.get(block.cell).unwrap().points().cell(),
            block.cell
        );
    }
}

#[test]
fn a_missing_cell_type_is_cell_not_in_rule() {
    let client = test_client();
    let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 2).unwrap();
    assert!(matches!(
        rule.get(Cell3::Tetrahedron),
        Err(GridError::CellNotInRule("Tetrahedron"))
    ));
    let facet = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 2).unwrap();
    assert!(matches!(
        facet.get(Cell3::Tetrahedron),
        Err(GridError::CellNotInRule("Tetrahedron"))
    ));
}

#[test]
fn a_gll_triangle_is_an_unsupported_node_family() {
    let client = test_client();
    let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 2).unwrap();
    let ps = rule.get(Cell2::Triangle).unwrap();
    let e = CoordinateElement::<D2> {
        cell: Cell2::Triangle,
        degree: 2,
        family: NodeFamily::Gll,
    };
    assert!(matches!(
        ps.tabulate(&e),
        Err(GridError::UnsupportedNodeFamily {
            cell: "Triangle",
            family: "Gll"
        })
    ));
    assert!(matches!(
        ps.tabulate_derivatives(&e),
        Err(GridError::UnsupportedNodeFamily {
            cell: "Triangle",
            family: "Gll"
        })
    ));
}

#[test]
fn degree_zero_is_an_unsupported_degree() {
    let client = test_client();
    let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Prism], 2).unwrap();
    let e = CoordinateElement::<D3> {
        cell: Cell3::Prism,
        degree: 0,
        family: NodeFamily::Equispaced,
    };
    assert!(matches!(
        rule.get(Cell3::Prism).unwrap().tabulate(&e),
        Err(GridError::UnsupportedDegree {
            cell: "Prism",
            degree: 0
        })
    ));
}

/// The coordinate elements of `cells` checked against gridcl-refcell.
fn elements<TD: TopoDim>(cells: &[TD::Cell]) -> Vec<CoordinateElement<TD>> {
    let mut out = Vec::new();
    for &cell in cells {
        for degree in 1..=3 {
            out.push(CoordinateElement {
                cell,
                degree,
                family: NodeFamily::Equispaced,
            });
            let c: CellType = cell.into();
            if matches!(
                c,
                CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron
            ) {
                out.push(CoordinateElement {
                    cell,
                    degree,
                    family: NodeFamily::Gll,
                });
            }
        }
    }
    out
}

/// `|a - b| <= tol * max(1, |b|)`.
fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

fn tabulations_of<F: Real + DeviceScalar, TD: TopoDim>(client: &Client, cells: &[TD::Cell]) {
    let rule = QuadRule::<F, TD>::gauss(client, cells, 4).unwrap();
    let td = TD::N;
    for e in elements::<TD>(cells) {
        let ps = rule.get(e.cell).unwrap();
        let reference = LagrangeElement::new(e.cell.into(), e.degree, e.family)
            .unwrap()
            .tabulate(ps.points(), 1)
            .unwrap();
        let (nq, nk) = (ps.len(), reference.num_basis);

        let values: Vec<f64> = ps
            .tabulate(&e)
            .unwrap()
            .to_host()
            .unwrap()
            .into_iter()
            .map(Real::widen)
            .collect();
        assert_eq!(values.len(), nq * nk);
        for q in 0..nq {
            for k in 0..nk {
                let (got, want) = (values[q * nk + k], reference.get(0, q, k));
                assert!(
                    close(got, want, tol::<F>()),
                    "{e:?} phi[{q}][{k}]: {got} vs {want}"
                );
            }
        }

        let derivatives: Vec<f64> = ps
            .tabulate_derivatives(&e)
            .unwrap()
            .to_host()
            .unwrap()
            .into_iter()
            .map(Real::widen)
            .collect();
        assert_eq!(derivatives.len(), nq * nk * td);
        for q in 0..nq {
            for k in 0..nk {
                for d in 0..td {
                    let got = derivatives[(q * nk + k) * td + d];
                    let want = reference.get(1 + d, q, k);
                    assert!(
                        close(got, want, tol::<F>()),
                        "{e:?} dphi[{q}][{k}][{d}]: {got} vs {want}"
                    );
                }
            }
        }

        // The cache: overwrite the shared allocation, then read it back through a new call.
        let seven = F::narrow(7.0);
        let mut a = ps.tabulate(&e).unwrap();
        gridcl_device::fill(&mut a, seven).unwrap();
        let again = ps.tabulate(&e).unwrap().to_host().unwrap();
        assert!(
            again.iter().all(|&x| x == seven),
            "{e:?}: values not cached"
        );
        let mut b = ps.tabulate_derivatives(&e).unwrap();
        gridcl_device::fill(&mut b, seven).unwrap();
        let again = ps.tabulate_derivatives(&e).unwrap().to_host().unwrap();
        assert!(
            again.iter().all(|&x| x == seven),
            "{e:?}: derivatives not cached"
        );
    }
}

fn tabulations<F: Real + DeviceScalar>(client: &Client) {
    tabulations_of::<F, D1>(client, &CELLS1);
    tabulations_of::<F, D2>(client, &CELLS2);
    tabulations_of::<F, D3>(client, &CELLS3);
}

#[test]
fn tabulations_match_gridcl_refcell_and_are_uploaded_once() {
    both_precisions!(tabulations);
}

fn facets_of<F: Real, TD: TopoDim>(client: &Client, cells: &[TD::Cell]) {
    let td = TD::N;
    for degree in 0..=5 {
        let rule = FacetQuadRule::<F, TD>::gauss(client, cells, degree).unwrap();
        for &cell in cells {
            let c: CellType = cell.into();
            let fps = rule.get(cell).unwrap();
            let ps = fps.points();
            assert_eq!(ps.cell(), cell);
            let facet_types = sub_entity_types(c, td - 1);
            let normals = reference_facet_normals(c);
            let offsets = fps.facet_offsets();
            assert_eq!(offsets.len(), facet_types.len() + 1);
            assert_eq!(offsets[0], 0);
            assert_eq!(offsets[facet_types.len()], ps.len());
            assert_eq!(ps.points().len(), ps.len() * td);
            for (f, &facet_type) in facet_types.iter().enumerate() {
                let facet_rule = gauss_rule(facet_type, degree);
                let (start, end) = (offsets[f], offsets[f + 1]);
                assert_eq!(end - start, facet_rule.weights.len(), "{c:?} facet {f}");
                assert_eq!(&ps.weights()[start..end], &facet_rule.weights[..]);
                assert_eq!(
                    &ps.points()[start * td..end * td],
                    &facet_points_to_cell(c, f, &facet_rule.points)[..],
                    "{c:?} facet {f}"
                );
                let volume: f64 = ps.weights()[start..end].iter().sum();
                assert!((volume - reference_volume(facet_type)).abs() < 1e-14);
                for q in start..end {
                    assert_eq!(fps.reference_normal(q), &normals[f * td..(f + 1) * td]);
                }
            }
        }
    }
}

fn facets<F: Real>(client: &Client) {
    facets_of::<F, D1>(client, &CELLS1);
    facets_of::<F, D2>(client, &CELLS2);
    facets_of::<F, D3>(client, &CELLS3);
}

#[test]
fn facet_points_and_normals_agree_with_gridcl_refcell() {
    both_precisions!(facets);
}

#[test]
fn facet_point_sets_tabulate_like_any_point_set() {
    let client = test_client();
    let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Pyramid], 2).unwrap();
    let ps = rule.get(Cell3::Pyramid).unwrap().points();
    let e = CoordinateElement::<D3> {
        cell: Cell3::Pyramid,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    let phi = ps.tabulate(&e).unwrap().to_host().unwrap();
    assert_eq!(phi.len(), ps.len() * 5);
    for row in phi.chunks(5) {
        let sum: f32 = row.iter().sum();
        assert!((f64::from(sum) - 1.0).abs() < TOL_F32);
    }
}
