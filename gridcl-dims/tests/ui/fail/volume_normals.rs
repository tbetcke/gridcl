//! Listing B10: normals exist only in codimension 1, so not on a volume mesh (E0599).
#![allow(dead_code, unused)]

#[path = "../stub/mesh.rs"]
mod stub;

use gridcl_dims::*;
use stub::*;

fn check(volume: &VolumeMesh<f64>, rule: &QuadRule<f64, D3>) {
    let _ = volume.normals(rule);
}

fn main() {
    let _ = check as fn(&_, &_);
}
