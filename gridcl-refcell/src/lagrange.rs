//! Lagrange node counts and layouts, A.4 and A.5 (docs/design.md, Appendix A).

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

use core::ops::Range;

use gridcl_dims::CellType;

use crate::RefCellError;

/// The 1D point family of a Lagrange layout ("Reference cells and conventions", node families).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeFamily {
    /// Equispaced points, on every cell.
    Equispaced,
    /// Gauss–Lobatto–Legendre points, on intervals, quadrilaterals and hexahedra only.
    Gll,
}

/// The number of Lagrange nodes N(p) of degree `degree` on `cell` (A.4).
///
/// The point cell has one node. Degree 0 gives one node on every cell.
///
/// # Panics
///
/// If the count overflows `usize`; the message names the cell and the degree.
pub fn num_lagrange_nodes(cell: CellType, degree: u32) -> usize {
    todo!()
}

/// The number of Lagrange nodes in the interior of `cell` (A.4); 0 below the first degree with
/// interior nodes, and 0 for the point cell.
///
/// # Panics
///
/// If the count overflows `usize`; the message names the cell and the degree.
pub fn num_interior_lagrange_nodes(cell: CellType, degree: u32) -> usize {
    todo!()
}

/// The Lagrange nodes of degree `degree` on `cell` in internal order (A.5), flat and
/// node-major with [`tdim`](crate::tdim) coordinates per node.
///
/// # Errors
///
/// - [`RefCellError::UnsupportedDegree`] if `degree == 0`;
/// - [`RefCellError::UnsupportedNodeFamily`] for GLL on a cell other than an interval,
///   quadrilateral or hexahedron.
pub fn lagrange_points(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<f64>, RefCellError> {
    todo!()
}

/// The node indices (a range of the internal order) of the Lagrange nodes inside sub-entity
/// `index` of dimension `dim`, excluding its boundary (A.5 block layout).
///
/// # Panics
///
/// If `dim > tdim(cell)` or `index` is not a sub-entity of that dimension.
pub fn lagrange_entity_nodes(
    cell: CellType,
    degree: u32,
    dim: usize,
    index: usize,
) -> Range<usize> {
    todo!()
}
