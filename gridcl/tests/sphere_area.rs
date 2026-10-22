//! M3 acceptance: sphere-area convergence rate ≥ p + 1 − 0.2 for p = 1–4 over four refinements
//! (f64): the least-squares slope of log relative area error against log h (docs/plans/M3.md,
//! question 18), at quadrature degree 4p + 2.

mod common;

use common::*;
use gridcl::prelude::*;

/// The four refinements of the icosahedron; h halves with each.
const REFINEMENTS: [u32; 4] = [0, 1, 2, 3];

/// The least-squares slope of y against x.
fn slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
    sxy / sxx
}

#[test]
fn sphere_area_converges() {
    let client = client();
    if !gridcl_testing::supports::<f64>(&client) {
        eprintln!("sphere_area_converges: f64 is not supported on this device; skipped");
        return;
    }
    let exact = 4.0 * std::f64::consts::PI;
    for p in 1..=4u32 {
        let mut log_h = Vec::new();
        let mut log_err = Vec::new();
        for r in REFINEMENTS {
            let mesh = build::<f64, D2, D3>(&gridcl_testing::sphere(r, p), &client);
            let area = mesh.measure(4 * p + 2).expect("measure");
            let err = (area - exact).abs() / exact;
            log_h.push(-(r as f64) * 2f64.ln());
            log_err.push(err.ln());
        }
        let rate = slope(&log_h, &log_err);
        assert!(
            rate >= f64::from(p) + 1.0 - 0.2,
            "p = {p}: rate {rate:.3} below {}; log errors {log_err:?}",
            f64::from(p) + 0.8
        );
    }
}
