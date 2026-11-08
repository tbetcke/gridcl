//! M9 acceptance of `orient` on one rank (docs/plans/M9.md, acceptance coverage): `Outward`
//! recovers randomly flipped closed surfaces and curves, and the errors `NeedsReferencePoint`,
//! `NonManifold` and `NonOrientable`.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::HostMesh;
use gridcl_testing::parallel::normals_and_flux;

/// `h` with half of its cells flipped, after checking that some are.
fn flipped(h: &HostMesh, seed: u64) -> HostMesh {
    let f = h.with_flipped_cells(seed, 0.5);
    assert_ne!(&f, h, "no cell flipped");
    f
}

/// Orients `h` with `rule` and returns the smallest n·x at the Gauss points of degree 2p and
/// ∮ x·n.
fn orient_and_measure<F, TD, GD>(
    client: &Client,
    h: &HostMesh,
    rule: OrientationRule,
    p: u32,
) -> (f64, f64)
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let mut mesh = build::<F, TD, GD>(h, client);
    mesh.orient(rule).expect("orient");
    normals_and_flux(&mesh, &[0.0; 3], 2 * p).expect("normals")
}

fn outward<F: Real>(client: &Client) {
    for p in 1..=3 {
        let surfaces = [
            ("sphere", gridcl_testing::sphere(1, p)),
            ("mixed surface", gridcl_testing::mixed_surface(1, p)),
        ];
        for (what, h) in surfaces {
            let (min, flux) = orient_and_measure::<F, D2, D3>(
                client,
                &flipped(&h, 5 + u64::from(p)),
                OrientationRule::Outward,
                p,
            );
            assert!(
                min > 0.0 && flux > 0.0,
                "{what} p={p} ({}): n·x ≥ {min:e}",
                name::<F>()
            );
        }
        let h = flipped(&gridcl_testing::circle(12, p), 7);
        let (min, flux) = orient_and_measure::<F, D1, D2>(client, &h, OrientationRule::Outward, p);
        assert!(
            min > 0.0 && flux > 0.0,
            "circle p={p} ({}): n·x ≥ {min:e}",
            name::<F>()
        );
    }
}

#[test]
fn outward_recovers_flipped_sphere() {
    for_each_precision!(client(), outward);
}

/// The error of `orient(rule)` on `h`.
fn orient_error<F, TD, GD>(client: &Client, h: &HostMesh, rule: OrientationRule) -> GridError
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let mut mesh = build::<F, TD, GD>(h, client);
    mesh.orient(rule).expect_err("orient must fail")
}

fn open_surface<F: Real>(client: &Client) {
    let strip =
        gridcl_testing::triangle_strip(12).mapped(3, |x| vec![x[0], x[1], 0.1 * x[0] * x[1]]);
    for (what, h) in [
        ("hemisphere p=1", gridcl_testing::hemisphere(1, 1)),
        ("hemisphere p=2", gridcl_testing::hemisphere(1, 2)),
        ("triangle strip in 3D", strip),
    ] {
        let e = orient_error::<F, D2, D3>(client, &flipped(&h, 3), OrientationRule::Outward);
        assert!(
            matches!(e, GridError::NeedsReferencePoint),
            "{what} ({}): {e}",
            name::<F>()
        );
    }
}

#[test]
fn outward_on_open_surface_needs_reference_point() {
    for_each_precision!(client(), open_surface);
}

/// `AwayFrom` and `Towards` the centre of the sphere orient its open cap (T9).
fn reference_point<F: Real>(client: &Client) {
    for p in 1..=2 {
        let h = flipped(&gridcl_testing::hemisphere(1, p), 11);
        let (min, _) =
            orient_and_measure::<F, D2, D3>(client, &h, OrientationRule::AwayFrom([0.0; 3]), p);
        assert!(
            min > 0.0,
            "AwayFrom, p={p} ({}): n·x ≥ {min:e}",
            name::<F>()
        );
        let mut mesh = build::<F, D2, D3>(&h, client);
        mesh.orient(OrientationRule::Towards([0.0; 3]))
            .expect("orient");
        let rule = QuadRule::for_grid(&mesh, 2 * p).expect("rule");
        let (x, n) = (
            mesh.physical_points(&rule).unwrap(),
            mesh.normals(&rule).unwrap(),
        );
        for (x, n) in flat_f64(&x).iter().zip(flat_f64(&n)) {
            for (x, n) in x.as_chunks::<3>().0.iter().zip(n.as_chunks::<3>().0) {
                let xn: f64 = x.iter().zip(n).map(|(a, b)| a * b).sum();
                assert!(xn < 0.0, "Towards, p={p} ({}): n·x = {xn:e}", name::<F>());
            }
        }
    }
}

#[test]
fn away_from_and_towards_on_hemisphere() {
    for_each_precision!(client(), reference_point);
}

fn non_manifold<F: Real>(client: &Client) {
    for pages in [3, 4] {
        let e = orient_error::<F, D2, D3>(
            client,
            &gridcl_testing::book(pages),
            OrientationRule::Outward,
        );
        assert!(
            matches!(e, GridError::NonManifold),
            "book({pages}) ({}): {e}",
            name::<F>()
        );
    }
    for p in 1..=2 {
        let e = orient_error::<F, D1, D2>(
            client,
            &gridcl_testing::y_junction(p),
            OrientationRule::AwayFrom([0.0; 3]),
        );
        assert!(
            matches!(e, GridError::NonManifold),
            "y_junction({p}) ({}): {e}",
            name::<F>()
        );
    }
}

#[test]
fn three_cells_on_an_edge_is_non_manifold() {
    for_each_precision!(client(), non_manifold);
}

fn mobius<F: Real>(client: &Client) {
    let strip = gridcl_testing::mobius_strip(12);
    for (what, h) in [
        ("as built", strip.clone()),
        ("flipped", flipped(&strip, 13)),
    ] {
        for rule in [
            OrientationRule::Outward,
            OrientationRule::AwayFrom([0.0; 3]),
        ] {
            let e = orient_error::<F, D2, D3>(client, &h, rule);
            assert!(
                matches!(e, GridError::NonOrientable),
                "{what}, {rule:?} ({}): {e}",
                name::<F>()
            );
        }
    }
}

#[test]
fn mobius_strip_is_non_orientable() {
    for_each_precision!(client(), mobius);
}
