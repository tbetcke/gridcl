//! volume.boundary().boundary() compiles: the boundary of a surface is a curve.
#![allow(dead_code, unused)]

#[path = "../stub/mesh.rs"]
mod stub;

use gridcl_dims::*;
use stub::*;

fn check(volume: &VolumeMesh<f64>) {
    let (surface, _) = volume.boundary().unwrap();
    let _: Result<(CurveMesh<f64>, EntityMap), StubError> = surface.boundary();
}

fn main() {
    let _ = check as fn(&_);
}
