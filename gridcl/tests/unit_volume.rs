//! M3 acceptance: unit volume for p = 1–4, to 1e-12 in f64 and 1e-5 in f32.
//!
//! Every box with its interior perturbed, except the nodes of pyramid cells, which stay affine
//! because the pyramid's det J is rational. Quadrature of degree p − 1 in 1D, 2p in 2D and 3p in
//! 3D integrates det J of these maps exactly (docs/plans/M3.md, question 24).

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;

const N: usize = 2;
const AMPLITUDE: f64 = 0.05;

fn degree(tdim: usize, p: u32) -> u32 {
    match tdim {
        1 => p - 1,
        2 => 2 * p,
        _ => 3 * p,
    }
}

fn check<F: Real>(client: &Client) {
    for kind in BoxKind::ALL {
        for p in 1..=4 {
            let h = kind.host(N, p).perturbed(AMPLITUDE);
            let q = degree(kind.tdim(), p);
            let volume = match kind.tdim() {
                1 => build::<F, D1, D1>(&h, client).measure(q),
                2 => build::<F, D2, D2>(&h, client).measure(q),
                _ => build::<F, D3, D3>(&h, client).measure(q),
            }
            .unwrap_or_else(|e| panic!("{kind:?} box, p = {p}: {e}"));
            let err = (volume - 1.0).abs();
            assert!(
                err <= tol::<F>(),
                "{kind:?} box, p = {p}, {}: volume {volume}, error {err:e}",
                name::<F>()
            );
        }
    }
}

#[test]
fn unit_volume_f64() {
    let client = client();
    if !gridcl_testing::supports::<f64>(&client) {
        eprintln!("unit_volume_f64: f64 is not supported on this device; skipped");
        return;
    }
    check::<f64>(&client);
}

#[test]
fn unit_volume_f32() {
    check::<f32>(&client());
}
