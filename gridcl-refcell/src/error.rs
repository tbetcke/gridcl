//! Errors of gridcl-refcell.

use core::fmt;

use gridcl_dims::CellType;

use crate::NodeFamily;

/// An error from a reference-cell layout, element, tabulation or node map.
///
/// gridcl-refcell depends only on gridcl-dims, so it cannot return `GridError`; gridcl-geometry
/// maps these errors onto `GridError` (docs/plans/M1b.md, question 3).
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::{NodeFamily, RefCellError};
///
/// let err = RefCellError::UnsupportedNodeFamily { cell: CellType::Triangle, family: NodeFamily::Gll };
/// assert_eq!(err.to_string(), "node family Gll is not supported on Triangle cells");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefCellError {
    /// The node family is not offered on this cell: GLL exists on intervals, quadrilaterals and
    /// hexahedra only.
    UnsupportedNodeFamily {
        /// The cell type.
        cell: CellType,
        /// The requested node family.
        family: NodeFamily,
    },
    /// The degree is not supported: layouts, elements and node maps need p ≥ 1.
    UnsupportedDegree {
        /// The cell type.
        cell: CellType,
        /// The requested degree.
        degree: u32,
    },
    /// Tabulation supports derivative orders 0 and 1 only.
    UnsupportedDerivativeOrder(u32),
    /// A flat point array whose length is not a multiple of the cell's topological dimension.
    PointsShape {
        /// The length of the point array.
        len: usize,
        /// The topological dimension of the cell.
        tdim: usize,
    },
    /// The file format has no node ordering for this cell and degree.
    NotInFormat {
        /// The file format, `"Gmsh"` or `"VTK"`.
        format: &'static str,
        /// The cell type.
        cell: CellType,
        /// The degree.
        degree: u32,
    },
    /// The generalised Vandermonde matrix is singular: the nodes are not unisolvent.
    Singular,
}

impl fmt::Display for RefCellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedNodeFamily { cell, family } => {
                write!(
                    f,
                    "node family {family:?} is not supported on {cell:?} cells"
                )
            }
            Self::UnsupportedDegree { cell, degree } => {
                write!(
                    f,
                    "degree {degree} is not supported on {cell:?} cells (p >= 1 required)"
                )
            }
            Self::UnsupportedDerivativeOrder(n) => {
                write!(f, "derivative order {n} is not supported (0 or 1)")
            }
            Self::PointsShape { len, tdim } => write!(
                f,
                "a point array of length {len} is not a whole number of points of dimension {tdim}"
            ),
            Self::NotInFormat {
                format,
                cell,
                degree,
            } => {
                write!(
                    f,
                    "{format} has no node ordering for {cell:?} cells of degree {degree}"
                )
            }
            Self::Singular => write!(
                f,
                "the Vandermonde matrix is singular (nodes not unisolvent)"
            ),
        }
    }
}

impl std::error::Error for RefCellError {}
