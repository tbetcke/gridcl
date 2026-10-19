//! Normals exist in codimension 1: on surface meshes and on curves in the plane.
#![allow(dead_code, unused)]

#[path = "../stub/mesh.rs"]
mod stub;

use gridcl_dims::*;
use stub::*;

fn check(
    surface: &SurfaceMesh<f64>,
    surface_rule: &QuadRule<f64, D2>,
    curve_in_2d: &Mesh<f64, D1, D2>,
    curve_rule: &QuadRule<f64, D1>,
) {
    let _ = surface.normals(surface_rule);
    let _ = curve_in_2d.normals(curve_rule);
}

fn main() {
    let _ = check as fn(&_, &_, &_, &_);
}
