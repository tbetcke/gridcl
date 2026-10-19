//! Listing B10: a CellBlock<D2> holds only Cell2 values, so a Cell3 is rejected (E0308).
#![allow(dead_code, unused)]

#[path = "../stub/mesh.rs"]
mod stub;

use gridcl_dims::*;
use stub::*;

fn check() {
    let _ = CellBlock::<D2>::new(Cell3::Tetrahedron);
}

fn main() {
    let _ = check as fn();
}
