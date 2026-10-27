//! A host mesh description to the file model.

use gridcl_device::Result;

use super::{VtkMesh, VtuFile};

/// The file of `mesh`: every node as a point (in f32 if `f32_points`), and every cell with its
/// type from [`vtk_cell_type`](super::vtk_cell_type) and its nodes in VTK order (the inverse
/// of `gridcl_refcell::vtk_to_internal`). Linear cells use their corner nodes. Also returns
/// the number of pyramids of degree ≥ 2 written as linear cells with `lagrange`.
///
/// # Errors
///
/// `LengthMismatch` if `nodes` or a block's node list has the wrong length; `IndexOutOfRange`
/// for a node index past the last node; `TooLarge` if a count overflows.
pub fn to_vtu(mesh: &VtkMesh, lagrange: bool, f32_points: bool) -> Result<(VtuFile, usize)> {
    let _ = (mesh, lagrange, f32_points);
    todo!()
}
