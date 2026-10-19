//! Orientation and permutation bits, A.6 (docs/design.md, Appendix A).

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

use gridcl_dims::CellType;

/// The permutation info of a cell (A.6): one bit per edge, then three bits per face, with
/// ndelement's layout (`compute_orientation`).
///
/// `vertex_ids` are the origin ids of the cell's vertices in local order; never local or
/// post-partition global ids. Cells of dimension below 2 have no bits.
///
/// # Panics
///
/// If `vertex_ids.len()` is not the cell's vertex count.
pub fn orientation_bits(cell: CellType, vertex_ids: &[i64]) -> u32 {
    todo!()
}
