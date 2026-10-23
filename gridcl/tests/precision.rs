//! M3 acceptance: f32 results agree with f64 to 1e-5 relative: measures, and the X, J, det J and
//! K fields as relative max norms per field. Where the test device lacks f64, the f64 run uses
//! the CPU runtime (docs/plans/M3.md, question 15).

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::{HostMesh, TOL_F32};

/// Named fields, widened to f64, per block.
type Fields = Vec<(&'static str, Vec<Vec<f64>>)>;

/// Measure and the X, J, det J and K fields of `h`, widened to f64, per block.
fn results<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    h: &HostMesh,
    client: &Client,
    degree: u32,
) -> (f64, Fields) {
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
    let fields = vec![
        ("X", flat_f64(&mesh.physical_points(&rule).expect("X"))),
        ("J", flat_f64(&mesh.jacobians(&rule).expect("J"))),
        ("det J", flat_f64(&mesh.detj(&rule).expect("det J"))),
        (
            "K",
            flat_f64(&mesh.pseudo_inverse_jacobians(&rule).expect("K")),
        ),
    ];
    (mesh.measure(degree).expect("measure"), fields)
}

fn compare<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(what: &str, h: &HostMesh, degree: u32) {
    let client = client();
    let f64_client = if gridcl_testing::supports::<f64>(&client) {
        client.clone()
    } else {
        gridcl_testing::cpu_client()
    };
    let (m32, f32s) = results::<f32, TD, GD>(h, &client, degree);
    let (m64, f64s) = results::<f64, TD, GD>(h, &f64_client, degree);
    let err = (m32 - m64).abs() / m64.abs();
    assert!(
        err <= TOL_F32,
        "{what}: measure {m32} (f32) vs {m64} (f64), error {err:e}"
    );
    for ((field, a), (_, b)) in f32s.iter().zip(&f64s) {
        for (blk, (x, y)) in a.iter().zip(b).enumerate() {
            let err = rel_max_err(x, y);
            assert!(
                err <= TOL_F32,
                "{what}: {field}, block {blk}: relative max error {err:e}"
            );
        }
    }
}

#[test]
fn f32_agrees_with_f64() {
    compare::<D2, D3>("sphere, p = 2", &gridcl_testing::sphere(1, 2), 6);
    compare::<D2, D3>(
        "mixed surface, p = 3",
        &gridcl_testing::mixed_surface(1, 3),
        8,
    );
    compare::<D3, D3>(
        "perturbed mixed box, p = 2",
        &BoxKind::Mixed.host(2, 2).perturbed(0.05),
        6,
    );
    compare::<D3, D3>(
        "perturbed hexahedra, p = 3",
        &BoxKind::Hexahedron.host(2, 3).perturbed(0.05),
        9,
    );
    compare::<D2, D2>(
        "perturbed triangles, p = 2",
        &BoxKind::Triangle.host(3, 2).perturbed(0.05),
        4,
    );
}
