//! Node maps between file formats and the internal Lagrange order, A.9 (docs/design.md,
//! "High-order node orderings").
//!
//! Entry k of a map is the internal index of the file's k-th node:
//! `internal[map[k]] = file[k]`.

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

mod gmsh;
mod vtk;

use gridcl_dims::CellType;

use crate::RefCellError;

/// The map from Gmsh's node order to the internal order for `cell` of degree `degree`.
///
/// # Errors
///
/// [`RefCellError::UnsupportedDegree`] if `degree == 0`.
pub fn gmsh_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError> {
    todo!()
}

/// The map from VTK's Lagrange node order to the internal order for `cell` of degree `degree`.
///
/// # Errors
///
/// - [`RefCellError::UnsupportedDegree`] if `degree == 0`;
/// - [`RefCellError::NotInFormat`] for pyramids of degree 2 or more (A.8).
pub fn vtk_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError> {
    todo!()
}
