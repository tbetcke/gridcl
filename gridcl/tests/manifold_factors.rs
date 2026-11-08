//! The manifold tests of normals, pseudo-inverse and facet normals (ADR-0012 item 10; ADR-0039
//! item 19), on curved boundary submeshes and the sphere: normals have unit length, are
//! orthogonal to J's columns and are flipped by flag bit 0; K J = I, J K J = J and J K n = 0;
//! the parent's facet normals at boundary facets equal the boundary submesh's normals.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{facet_normals_block, physical_points_block};
use gridcl_testing::HostMesh;

const DEGREE: u32 = 4;

/// The unit square of degree-2 quadrilaterals bent by a smooth map, so its boundary curves.
fn curved_square() -> HostMesh {
    gridcl_testing::square_box(Cell2::Quadrilateral, 3, 2).mapped(2, |x| {
        vec![
            x[0] + 0.1 * (std::f64::consts::PI * x[1]).sin(),
            x[1] + 0.1 * x[0] * x[0],
        ]
    })
}

/// `h` on one rank with its facet entities, and its boundary submesh.
#[allow(clippy::type_complexity)]
fn boundary_of<F, TD>(client: &Client, h: &HostMesh) -> (Mesh<F, TD, TD>, Mesh<F, TD::Lower, TD>)
where
    F: Real,
    TD: TopoDim + Codim0<TD>,
    TD::Lower: TopoDim + EmbeddedIn<TD>,
{
    let mut mesh = build::<F, TD, TD>(h, client);
    mesh.create_entities(TD::N - 1).expect("facets");
    let (child, _) = mesh.boundary().expect("boundary");
    (mesh, child)
}

/// Per point of every block of the codimension-1 mesh `m`: J (column-major, GD × TD), K
/// (column-major, TD × GD) and n, in f64.
#[allow(clippy::type_complexity)]
fn factors<F, TD, GD>(m: &Mesh<F, TD, GD>) -> Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let rule = QuadRule::for_grid(m, DEGREE).expect("rule");
    let j = flat_f64(&m.jacobians(&rule).expect("J"));
    let k = flat_f64(&m.pseudo_inverse_jacobians(&rule).expect("K"));
    let n = flat_f64(&m.normals(&rule).expect("n"));
    let (td, gd) = (TD::N, GD::N);
    let mut out = Vec::new();
    for b in 0..j.len() {
        for ((j, k), n) in j[b]
            .chunks_exact(td * gd)
            .zip(k[b].chunks_exact(td * gd))
            .zip(n[b].chunks_exact(gd))
        {
            out.push((j.to_vec(), k.to_vec(), n.to_vec()));
        }
    }
    out
}

/// Unit normals orthogonal to every column of J.
fn unit_and_orthogonal<F, TD, GD>(what: &str, m: &Mesh<F, TD, GD>)
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let (td, gd) = (TD::N, GD::N);
    let points = factors(m);
    assert!(!points.is_empty(), "{what}: no points");
    for (i, (j, _, n)) in points.iter().enumerate() {
        let len: f64 = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(
            (len - 1.0).abs() <= tol::<F>(),
            "{what}: |n| = {len} at {i}"
        );
        for d in 0..td {
            let col = &j[d * gd..(d + 1) * gd];
            let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            let dot: f64 = col.iter().zip(n).map(|(a, b)| a * b).sum();
            assert!(
                dot.abs() <= tol::<F>() * norm,
                "{what}: n·J_{d} = {dot:e} at {i}"
            );
        }
    }
}

fn normals_check<F: Real>(client: &Client) {
    let (_, ball) = boundary_of::<F, D3>(client, &gridcl_testing::ball(1, 2));
    unit_and_orthogonal("ball boundary", &ball);
    let (_, square) = boundary_of::<F, D2>(client, &curved_square());
    unit_and_orthogonal("curved square boundary", &square);
    let sphere = build::<F, D2, D3>(&gridcl_testing::sphere(1, 2), client);
    unit_and_orthogonal("sphere", &sphere);
}

#[test]
fn normals_unit_and_orthogonal() {
    for_each_precision!(client(), normals_check);
}

/// On the boundary of the ball, every normal is the geometric normal J0 × J1 / |J0 × J1| times
/// −1 where the cell's flag bit 0 is set; on the sphere, `orient(Towards(0))` sets bit 0 on
/// every cell and negates every normal exactly.
fn flips<F: Real>(client: &Client) {
    let (_, ball) = boundary_of::<F, D3>(client, &gridcl_testing::ball(1, 2));
    let rule = QuadRule::for_grid(&ball, DEGREE).expect("rule");
    let j = flat_f64(&ball.jacobians(&rule).expect("J"));
    let n = flat_f64(&ball.normals(&rule).expect("n"));
    for (b, block) in ball.cell_blocks().iter().enumerate() {
        let flags = block.flags.to_host().expect("flags");
        let nq = rule.get(block.cell).expect("point set").len();
        for (i, (j, n)) in j[b]
            .as_chunks::<6>()
            .0
            .iter()
            .zip(n[b].as_chunks::<3>().0)
            .enumerate()
        {
            let v = [
                j[1] * j[5] - j[2] * j[4],
                j[2] * j[3] - j[0] * j[5],
                j[0] * j[4] - j[1] * j[3],
            ];
            let s = if flags[i / nq] & 1 == 1 { -1.0 } else { 1.0 };
            let dot: f64 = v.iter().zip(n).map(|(a, b)| a * b).sum();
            assert!(
                s * dot > 0.0,
                "ball boundary: normal {i} against flag {}",
                flags[i / nq]
            );
        }
    }

    let mut sphere = build::<F, D2, D3>(&gridcl_testing::sphere(1, 2), client);
    let rule = QuadRule::for_grid(&sphere, DEGREE).expect("rule");
    let before: Vec<Vec<F>> = sphere
        .normals(&rule)
        .expect("n")
        .blocks
        .iter()
        .map(flat)
        .collect();
    sphere
        .orient(OrientationRule::Towards([0.0; 3]))
        .expect("orient");
    for block in sphere.cell_blocks() {
        assert!(
            block
                .flags
                .to_host()
                .expect("flags")
                .iter()
                .all(|f| f & 1 == 1)
        );
    }
    let after = sphere.normals(&rule).expect("n");
    for (a, b) in after.blocks.iter().zip(&before) {
        let negated: Vec<F> = b.iter().map(|&x| -x).collect();
        assert_bits(&flat(a), &negated, "sphere normals after Towards");
    }
}

#[test]
fn flag_flips_normal() {
    for_each_precision!(client(), flips);
}

/// K J = I, J K J = J and J K n = 0 at every point of the codimension-1 mesh `m`.
fn projects<F, TD, GD>(what: &str, m: &Mesh<F, TD, GD>)
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let (td, gd) = (TD::N, GD::N);
    let jv = |j: &[f64], g: usize, d: usize| j[d * gd + g];
    let kv = |k: &[f64], t: usize, g: usize| k[g * td + t];
    for (i, (j, k, n)) in factors(m).iter().enumerate() {
        let scale: f64 = j.iter().map(|x| x.abs()).fold(0.0, f64::max);
        // K J (TD × TD).
        let kj: Vec<f64> = (0..td * td)
            .map(|e| (0..gd).map(|g| kv(k, e / td, g) * jv(j, g, e % td)).sum())
            .collect();
        for (e, x) in kj.iter().enumerate() {
            let want = if e / td == e % td { 1.0 } else { 0.0 };
            assert!(
                (x - want).abs() <= tol::<F>(),
                "{what}: (K J)_{e} = {x} at {i}"
            );
        }
        // J K J = J.
        for g in 0..gd {
            for d in 0..td {
                let jkj: f64 = (0..td).map(|t| jv(j, g, t) * kj[t * td + d]).sum();
                assert!(
                    (jkj - jv(j, g, d)).abs() <= tol::<F>() * scale,
                    "{what}: (J K J)_{g}{d} at {i}"
                );
            }
        }
        // J K n = 0.
        for g in 0..gd {
            let jkn: f64 = (0..td)
                .map(|t| jv(j, g, t) * (0..gd).map(|h| kv(k, t, h) * n[h]).sum::<f64>())
                .sum();
            assert!(
                jkn.abs() <= tol::<F>() * scale,
                "{what}: (J K n)_{g} = {jkn:e} at {i}"
            );
        }
    }
}

fn pinv_check<F: Real>(client: &Client) {
    let (_, ball) = boundary_of::<F, D3>(client, &gridcl_testing::ball(1, 2));
    projects("ball boundary", &ball);
    let (_, square) = boundary_of::<F, D2>(client, &curved_square());
    projects("curved square boundary", &square);
}

#[test]
fn pinv_projects_onto_tangent_space() {
    for_each_precision!(client(), pinv_check);
}

/// The parent's facet normals equal the boundary submesh's normals at the same physical points:
/// every Gauss point of the child matches one facet point of the parent.
fn facet_vs_boundary<F, TD>(client: &Client, what: &str, h: &HostMesh)
where
    F: Real,
    TD: TopoDim + Codim0<TD>,
    TD::Lower: TopoDim + Codim1<TD>,
{
    let what = format!("{what} ({})", name::<F>());
    let gd = TD::N;
    let (parent, child) = boundary_of::<F, TD>(client, h);
    let facet_rule = FacetQuadRule::for_grid(&parent, DEGREE).expect("facet rule");
    let mut facet_points: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    for (b, block) in parent.cell_blocks().iter().enumerate() {
        let fps = facet_rule.get(block.cell).expect("facet point set");
        let x = flat(&physical_points_block(&parent, b, fps.points()).expect("X"));
        let n = flat(&facet_normals_block(&parent, b, fps).expect("facet normals"));
        for (x, n) in x.chunks_exact(gd).zip(n.chunks_exact(gd)) {
            facet_points.push((
                x.iter().map(|v| v.widen()).collect(),
                n.iter().map(|v| v.widen()).collect(),
            ));
        }
    }
    let rule = QuadRule::for_grid(&child, DEGREE).expect("rule");
    let x = flat_f64(&child.physical_points(&rule).expect("X"));
    let n = flat_f64(&child.normals(&rule).expect("n"));
    let dist = |a: &[f64], b: &[f64]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (p - q) * (p - q))
            .sum::<f64>()
            .sqrt()
    };
    let mut matched = 0;
    for (x, n) in x.iter().zip(&n) {
        for (x, n) in x.chunks_exact(gd).zip(n.chunks_exact(gd)) {
            let (d, (_, want)) = facet_points
                .iter()
                .map(|p| (dist(&p.0, x), p))
                .min_by(|a, b| a.0.total_cmp(&b.0))
                .expect("facet points");
            assert!(
                d <= 10.0 * tol::<F>(),
                "{what}: no facet point at {x:?} ({d:e})"
            );
            let err = dist(n, want);
            assert!(
                err <= 10.0 * tol::<F>(),
                "{what}: normals at {x:?} differ by {err:e}"
            );
            matched += 1;
        }
    }
    assert!(matched > 0, "{what}: no boundary points");
}

fn facet_normals_check<F: Real>(client: &Client) {
    facet_vs_boundary::<F, D3>(client, "ball", &gridcl_testing::ball(1, 2));
    facet_vs_boundary::<F, D3>(
        client,
        "perturbed mixed box",
        &gridcl_testing::mixed_box(2, 2).perturbed(0.05),
    );
    facet_vs_boundary::<F, D2>(client, "curved square", &curved_square());
}

#[test]
fn facet_normals_equal_boundary_normals() {
    for_each_precision!(client(), facet_normals_check);
}
