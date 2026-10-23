//! Listing B10: a CellBlock<D2> holds only Cell2 values, so a Cell3 is rejected (E0308).
#![allow(dead_code, unused)]

use gridcl::prelude::*;
use gridcl::DeviceStrided;

fn check(
    index_map: IndexMap,
    vertices: DeviceStrided<u32>,
    origin_ids: DeviceBuffer<u32>,
    flags: DeviceBuffer<u32>,
) {
    let _ = CellBlock::<D2>::new(Cell3::Tetrahedron, index_map, vertices, origin_ids, flags);
}

fn main() {
    let _ = check as fn(_, _, _, _);
}
