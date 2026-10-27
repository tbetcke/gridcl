//! The VTK cell types of A.8.

use gridcl_dims::CellType;

/// The VTK cell type gridcl writes for `cell` of degree `degree` (A.8): the linear type for
/// p = 1 or `lagrange == false`, else the Lagrange type (68–73). Pyramids of degree p ≥ 2 give
/// 14: they are written from their vertices (ADR-0024).
pub fn vtk_cell_type(cell: CellType, degree: u32, lagrange: bool) -> u8 {
    let _ = (cell, degree, lagrange);
    todo!()
}
