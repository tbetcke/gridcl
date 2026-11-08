//! M9h T0a: the f64 host references of `gridcl_testing::reference` agree with M9's device
//! factors and pull-back at p = 2 (and at p = 3 on GLL hexahedra), on every (TD, GD) pair,
//! within the tolerance of the precision, relative per quantity over each block.

use cubecl::prelude::Client;
use gridcl::{
    Cell2, Cell3, Codim0, Codim1, D1, D2, D3, DeviceGrid, Dim, EmbeddedIn, FacetQuadRule, Mesh,
    NodeFamily, PullBackOptions, QuadRule, Real, TopoDim, detj_block, facet_normals_block,
    jacobians_block, normals_block, physical_points_block, pseudo_inverse_block,
};
use gridcl_testing::reference::{self, field_f64, rel_err, tolerance};
use gridcl_testing::{HostMesh, supports, test_client};

const P: u32 = 2;

fn name<F: Real>() -> &'static str {
    if size_of::<F>() == 4 { "f32" } else { "f64" }
}

fn assert_close<F: Real>(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let err = rel_err(got, want);
    let tol = tolerance::<F>(1.0);
    assert!(
        err <= tol,
        "{what} ({}): relative error {err:e} > {tol:e}",
        name::<F>()
    );
}

fn build<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    h: &HostMesh,
    client: &Client,
) -> Mesh<F, TD, GD> {
    h.build(client).expect("mesh")
}

/// X, J, det J, K and the pull-back of every block.
fn common<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    client: &Client,
    what: &str,
    h: &HostMesh,
) {
    let mesh = build::<F, TD, GD>(h, client);
    let degree = h.blocks.iter().map(|b| b.degree).max().unwrap_or(1);
    let rule = QuadRule::for_grid(&mesh, 2 * degree).expect("rule");
    let td = TD::N;
    for b in 0..mesh.cell_blocks().len() {
        let pts = rule.get(mesh.cell_blocks()[b].cell).expect("point set");
        let p = pts.points();
        let what = format!("{what}, block {b}");
        let x = field_f64(&physical_points_block(&mesh, b, pts).expect("X")).unwrap();
        assert_close::<F>(
            &x,
            &reference::points(&mesh, b, p).unwrap(),
            &format!("X of {what}"),
        );
        let j = field_f64(&jacobians_block(&mesh, b, pts).expect("J")).unwrap();
        assert_close::<F>(
            &j,
            &reference::jacobians(&mesh, b, p).unwrap(),
            &format!("J of {what}"),
        );
        let d = field_f64(&detj_block(&mesh, b, pts).expect("det J")).unwrap();
        assert_close::<F>(
            &d,
            &reference::detj(&mesh, b, p).unwrap(),
            &format!("det J of {what}"),
        );
        let k = field_f64(&pseudo_inverse_block(&mesh, b, pts).expect("K")).unwrap();
        assert_close::<F>(
            &k,
            &reference::pseudo_inverses(&mesh, b, p).unwrap(),
            &format!("K of {what}"),
        );
        // The host pull-back of the device's X recovers the reference points to f64 accuracy,
        // and the device pull-back agrees with it.
        let nq = pts.len();
        let xi = reference::pull_back(&mesh, b, &x, nq).unwrap();
        let want: Vec<f64> = (0..xi.len() / td)
            .flat_map(|i| p[(i % nq) * td..(i % nq + 1) * td].to_vec())
            .collect();
        let scale = reference::pull_back_scale(&mesh, b, p).unwrap();
        assert_eq!(scale, 1.0, "{what}: pull-back scale at p ≤ 4");
        assert_close::<F>(&xi, &want, &format!("host ξ of {what}"));
        let xf = mesh.physical_points(&rule).expect("X");
        let pb = mesh
            .pull_back(&xf, &PullBackOptions::default())
            .expect("pull-back");
        let dev = field_f64(&pb.reference.blocks[b]).unwrap();
        assert_close::<F>(&dev, &xi, &format!("device ξ of {what}"));
    }
}

fn facets<F: Real, TD: TopoDim + Codim0<TD>>(client: &Client, what: &str, h: &HostMesh) {
    let mesh = build::<F, TD, TD>(h, client);
    let rule = FacetQuadRule::for_grid(&mesh, 2 * P).expect("facet rule");
    for b in 0..mesh.cell_blocks().len() {
        let fps = rule.get(mesh.cell_blocks()[b].cell).expect("facet points");
        let n = field_f64(&facet_normals_block(&mesh, b, fps).expect("n")).unwrap();
        let want = reference::facet_normals(&mesh, b, fps).unwrap();
        assert_close::<F>(&n, &want, &format!("facet normals of {what}, block {b}"));
    }
}

fn normals<F: Real, TD: TopoDim + Codim1<GD>, GD: Dim>(client: &Client, what: &str, h: &HostMesh) {
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, 2 * P).expect("rule");
    for b in 0..mesh.cell_blocks().len() {
        let pts = rule.get(mesh.cell_blocks()[b].cell).expect("point set");
        let n = field_f64(&normals_block(&mesh, b, pts).expect("n")).unwrap();
        let want = reference::normals(&mesh, b, pts.points()).unwrap();
        assert_close::<F>(&n, &want, &format!("normals of {what}, block {b}"));
    }
}

fn check<F: Real>(client: &Client) {
    let lift = |h: &HostMesh, gd: usize| {
        h.mapped(gd, move |x| {
            let mut y = vec![x[0], 0.2 * (std::f64::consts::PI * x[0]).sin()];
            if gd == 3 {
                y.push(0.1 * (std::f64::consts::PI * x[0]).cos());
            }
            y
        })
    };
    let interval = gridcl_testing::interval_box(5, P).perturbed(0.05);
    common::<F, D1, D1>(client, "intervals", &interval);
    facets::<F, D1>(client, "intervals", &interval);
    let curve = lift(&interval, 2);
    common::<F, D1, D2>(client, "curve in 2D", &curve);
    normals::<F, D1, D2>(client, "curve in 2D", &curve);
    common::<F, D1, D3>(client, "curve in 3D", &lift(&interval, 3));
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        let h = gridcl_testing::square_box(cell, 3, P).perturbed(0.05);
        common::<F, D2, D2>(client, &format!("{cell:?} box"), &h);
        facets::<F, D2>(client, &format!("{cell:?} box"), &h);
    }
    for (what, h) in [
        ("sphere", gridcl_testing::sphere(1, P)),
        ("mixed surface", gridcl_testing::mixed_surface(1, P)),
    ] {
        common::<F, D2, D3>(client, what, &h);
        normals::<F, D2, D3>(client, what, &h);
    }
    for cell in [
        Cell3::Tetrahedron,
        Cell3::Hexahedron,
        Cell3::Prism,
        Cell3::Pyramid,
    ] {
        let h = gridcl_testing::cube_box(cell, 2, P).perturbed(0.05);
        common::<F, D3, D3>(client, &format!("{cell:?} box"), &h);
        facets::<F, D3>(client, &format!("{cell:?} box"), &h);
    }
    common::<F, D3, D3>(client, "ball", &gridcl_testing::ball(0, P));
    let gll = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
        .perturbed(0.05)
        .with_node_family(NodeFamily::Gll);
    common::<F, D3, D3>(client, "GLL hexahedron box p = 3", &gll);
}

#[test]
fn reference_agrees_with_m9_device_factors() {
    let client = test_client();
    check::<f32>(&client);
    if supports::<f64>(&client) {
        check::<f64>(&client);
    }
}
