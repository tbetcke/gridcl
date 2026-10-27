//! GLL → equispaced conversion of the geometry nodes (docs/design.md, "Reference cells and
//! conventions", node families; ADR-0024, item 7).

use gridcl_device::{DeviceBuffer, Result};

use crate::DeviceGrid;

/// The L-vector of `grid` with the numbering, length and origin of
/// [`nodes`](DeviceGrid::nodes), where every node of a GLL block holds the equispaced
/// interpolant of its cell's geometry. Equispaced blocks are copied unchanged.
///
/// A node shared by several cells takes its value from its first occurrence in flat
/// (block, cell, local node) order, so the result is bitwise deterministic.
///
/// # Errors
///
/// Allocation and device errors.
pub fn equispaced_nodes<G: DeviceGrid>(grid: &G) -> Result<DeviceBuffer<G::F>> {
    let _ = grid;
    todo!()
}
