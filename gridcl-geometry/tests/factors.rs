//! Geometric factors (M3 T14–T15): K·J = I for all six (TD, GD) pairs, det J of affine cells
//! is constant and matches the host, and the facet normals of affine cells are their
//! transformed reference normals.

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{FieldShape, QuadField};
use gridcl_testing::{HostMesh, TOL_F32, TOL_F64};

fn tol<F: Real>() -> f64 {
    if size_of::<F>() == 4 {
        TOL_F32
    } else {
        TOL_F64
    }
}

/// A field read back as flat f64 values, [cell][point][component].
fn flat<F: Real, S: FieldShape>(field: &QuadField<F, S>) -> Vec<f64> {
    let host = field.to_host_cell_major().unwrap();
    assert_eq!(size_of::<S::Host<F>>(), S::LEN * size_of::<F>());
    // SAFETY: the host value of every gridcl shape is an array of exactly S::LEN values of F
    // (checked by size), so the vector holds host.len() · LEN initialised values of F.
    let values =
        unsafe { core::slice::from_raw_parts(host.as_ptr().cast::<F>(), host.len() * S::LEN) };
    values.iter().map(|x| x.widen()).collect()
}

/// `h` (gdim 1) lifted onto a smooth curve in `gdim` dimensions.
fn lift(h: &HostMesh, gdim: usize) -> HostMesh {
    let pi = std::f64::consts::PI;
    let nodes = h
        .nodes
        .iter()
        .flat_map(|&x| {
            let mut p = vec![x, 0.2 * (pi * x).sin()];
            if gdim == 3 {
                p.push(0.1 * (pi * x).cos());
            }
            p
        })
        .collect();
    HostMesh {
        gdim,
        nodes,
        ..h.clone()
    }
}

/// K·J = I (TD × TD) at every point of every block, and the det J field equals the host
/// pseudo-determinant of the J field.
fn check_kj<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    client: &Client,
    h: &HostMesh,
    what: &str,
) {
    let mesh = h.build::<F, TD, GD>(client).unwrap();
    let rule = QuadRule::for_grid(&mesh, 4).unwrap();
    let (td, gd) = (TD::N, GD::N);
    let j = mesh.jacobians(&rule).unwrap();
    let k = mesh.pseudo_inverse_jacobians(&rule).unwrap();
    let d = mesh.detj(&rule).unwrap();
    for b in 0..mesh.cell_blocks().len() {
        let (jv, kv, dv) = (flat(&j.blocks[b]), flat(&k.blocks[b]), flat(&d.blocks[b]));
        let n = td * gd;
        assert_eq!(jv.len(), kv.len());
        for (p, (jp, kp)) in jv.chunks_exact(n).zip(kv.chunks_exact(n)).enumerate() {
            for t in 0..td {
                for s in 0..td {
                    // (K J)(t, s) = Σ_g K(t, g) J(g, s); K(t, g) at g·TD + t, J(g, s) at s·GD + g.
                    let x: f64 = (0..gd).map(|g| kp[g * td + t] * jp[s * gd + g]).sum();
                    let want = if t == s { 1.0 } else { 0.0 };
                    assert!(
                        (x - want).abs() <= 100.0 * tol::<F>(),
                        "{what}: (KJ)({t},{s}) = {x} at point {p}"
                    );
                }
            }
            let host = host_detj(jp, td, gd);
            assert!(
                (dv[p] - host).abs() <= 10.0 * tol::<F>() * host.abs(),
                "{what}: det J {} vs {host} at point {p}",
                dv[p]
            );
        }
    }
}

/// det J, or the pseudo-determinant, of a column-major gdim × tdim matrix in f64.
fn host_detj(j: &[f64], td: usize, gd: usize) -> f64 {
    let col = |d: usize| &j[d * gd..(d + 1) * gd];
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    match (td, gd) {
        (1, 1) => j[0],
        (2, 2) => j[0] * j[3] - j[2] * j[1],
        (3, 3) => {
            let m = |g: usize, d: usize| j[d * 3 + g];
            m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
        }
        (1, _) => dot(col(0), col(0)).sqrt(),
        _ => {
            let (a, b, c) = (
                dot(col(0), col(0)),
                dot(col(0), col(1)),
                dot(col(1), col(1)),
            );
            (a * c - b * b).sqrt()
        }
    }
}

fn kj_all<F: Real>(client: &Client) {
    use gridcl_testing::*;
    let curve = interval_box(5, 3).perturbed(0.05);
    check_kj::<F, D1, D1>(client, &curve, "interval");
    check_kj::<F, D1, D2>(client, &lift(&curve, 2), "curve in 2D");
    check_kj::<F, D1, D3>(client, &lift(&curve, 3), "curve in 3D");
    check_kj::<F, D2, D2>(
        client,
        &square_box(Cell2::Quadrilateral, 3, 2).perturbed(0.05),
        "quadrilaterals",
    );
    check_kj::<F, D2, D2>(
        client,
        &square_box(Cell2::Triangle, 3, 3).perturbed(0.05),
        "triangles",
    );
    check_kj::<F, D2, D3>(client, &mixed_surface(1, 2), "mixed surface");
    check_kj::<F, D3, D3>(client, &mixed_box(2, 2).perturbed(0.05), "mixed box");
    check_kj::<F, D3, D3>(
        client,
        &cube_box(Cell3::Prism, 2, 2).perturbed(0.05),
        "prisms",
    );
}

#[test]
fn k_times_j_is_the_identity_for_all_six_pairs() {
    let client = gridcl_testing::test_client();
    kj_all::<f32>(&client);
    if gridcl_testing::supports::<f64>(&client) {
        kj_all::<f64>(&client);
    }
}

/// On affine (P1, unperturbed) cells, det J is the same at every point of a cell, bitwise:
/// the derivative tabulation of P1 elements does not depend on the point.
#[test]
fn detj_of_affine_cells_is_constant() {
    use gridcl_testing::*;
    let client = gridcl_testing::test_client();
    let check = |values: Vec<f64>, q: usize, what: &str| {
        for (c, cell) in values.chunks_exact(q).enumerate() {
            assert!(
                cell.iter().all(|&x| x == cell[0] && x > 0.0),
                "{what}: cell {c}: {cell:?}"
            );
        }
    };
    let mesh = mixed_box(2, 1).build::<f32, D3, D3>(&client).unwrap();
    let rule = QuadRule::for_grid(&mesh, 3).unwrap();
    let d = mesh.detj(&rule).unwrap();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        if block.cell == Cell3::Pyramid {
            continue; // the P1 pyramid basis is rational; its affine cells still have constant det J
        }
        check(
            flat(&d.blocks[b]),
            rule.get(block.cell).unwrap().len(),
            "mixed box",
        );
    }
    let mesh = square_box(Cell2::Triangle, 3, 1)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 4).unwrap();
    check(
        flat(&mesh.detj(&rule).unwrap().blocks[0]),
        rule.get(Cell2::Triangle).unwrap().len(),
        "triangles",
    );
}

/// Facet normals of affine cells are their transformed reference normals: Kᵀ n_ref / |Kᵀ n_ref|
/// with K constant on the cell. On axis-aligned boxes they are the reference normals; on the
/// mixed box they equal Kᵀ n_ref / |Kᵀ n_ref| with K from the pseudo-inverse field.
#[test]
fn facet_normals_of_affine_cells_are_transformed_reference_normals() {
    use gridcl_testing::*;
    let client = gridcl_testing::test_client();
    // Axis-aligned hexahedra and quadrilaterals: n = n_ref.
    let mesh = cube_box(Cell3::Hexahedron, 2, 1)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    let rule = FacetQuadRule::for_grid(&mesh, 2).unwrap();
    let fps = rule.get(Cell3::Hexahedron).unwrap();
    let n = flat(&mesh.facet_normals(&rule).unwrap().blocks[0]);
    let q = fps.points().len();
    for (c, cell) in n.chunks_exact(q * 3).enumerate() {
        for p in 0..q {
            for (g, &want) in fps.reference_normal(p).iter().enumerate() {
                assert!(
                    (cell[p * 3 + g] - want).abs() < 1e-6,
                    "hexahedron {c}, point {p}"
                );
            }
        }
    }
    // The mixed box: K from the pseudo-inverse field at the facet points.
    let h = mixed_box(2, 1);
    let mesh = h
        .build::<f64, D3, D3>(&gridcl_testing::cpu_client())
        .unwrap();
    let rule = FacetQuadRule::for_grid(&mesh, 2).unwrap();
    let normals = mesh.facet_normals(&rule).unwrap();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let fps = rule.get(block.cell).unwrap();
        let q = fps.points().len();
        let k = flat(&gridcl::pseudo_inverse_block(&mesh, b, fps.points()).unwrap());
        let n = flat(&normals.blocks[b]);
        for (c, (nc, kc)) in n.chunks_exact(q * 3).zip(k.chunks_exact(q * 9)).enumerate() {
            for p in 0..q {
                let r = fps.reference_normal(p);
                let m: Vec<f64> = (0..3)
                    .map(|g| (0..3).map(|t| kc[p * 9 + g * 3 + t] * r[t]).sum())
                    .collect();
                let len = m.iter().map(|x| x * x).sum::<f64>().sqrt();
                for g in 0..3 {
                    assert!(
                        (nc[p * 3 + g] - m[g] / len).abs() < 1e-12,
                        "{:?} {c}, point {p}",
                        block.cell
                    );
                }
                let unit: f64 = (0..3).map(|g| nc[p * 3 + g] * nc[p * 3 + g]).sum();
                assert!((unit - 1.0).abs() < 1e-12);
            }
        }
    }
}
