//! Listing B10: the boundary of a curve would be a point mesh, and D0 is not a TopoDim (E0599).
#![allow(dead_code, unused)]

use gridcl::prelude::*;

fn check(curve_in_2d: &Mesh<f64, D1, D2>, curve: &CurveMesh<f64>) {
    let _ = curve_in_2d.boundary();
    let _ = curve.boundary();
}

fn main() {
    let _ = check as fn(&_, &_);
}
