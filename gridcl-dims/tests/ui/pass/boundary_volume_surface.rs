//! The boundary of a volume mesh is a surface mesh, and that of a surface mesh a curve mesh.
#![allow(dead_code, unused)]

#[path = "../stub/mesh.rs"]
mod stub;

use gridcl_dims::*;
use stub::*;

fn check(volume: &VolumeMesh<f64>, surface: &SurfaceMesh<f64>) {
    let _: Result<(SurfaceMesh<f64>, EntityMap), StubError> = volume.boundary();
    let _: Result<(CurveMesh<f64>, EntityMap), StubError> = surface.boundary();
}

fn main() {
    let _ = check as fn(&_, &_);
}
