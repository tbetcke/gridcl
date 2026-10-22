//! volume.boundary().boundary() compiles: the boundary of a surface is a curve.
#![allow(dead_code, unused)]

use gridcl::prelude::*;

fn check(volume: &VolumeMesh<f64>) -> Result<()> {
    let (surface, _) = volume.boundary()?;
    let _: Result<(CurveMesh<f64>, EntityMap)> = surface.boundary();
    Ok(())
}

fn main() {
    let _ = check as fn(&_) -> _;
}
