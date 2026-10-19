//! Quadrature rules on the reference cells (docs/design.md, "Tabulation and quadrature").

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

use gridcl_dims::CellType;

/// A quadrature rule on a reference cell, in f64.
///
/// gridcl-geometry's `QuadRule<F, TD>` builds on it.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadratureRule {
    /// The cell type.
    pub cell: CellType,
    /// The degree of exactness.
    pub degree: u32,
    /// The points, flat and node-major with [`tdim`](crate::tdim) coordinates per point.
    pub points: Vec<f64>,
    /// One weight per point; they sum to the volume of the reference cell.
    pub weights: Vec<f64>,
}

/// A Gauss rule on `cell` that is exact for polynomials of degree `degree`.
///
/// Collapsed Gauss–Jacobi on triangles, tetrahedra and pyramids; tensor Gauss on
/// quadrilaterals and hexahedra; triangle times interval on prisms. On pyramids the rule is
/// exact for the rationomials of degree `degree` of A.5's pyramid space
/// (docs/plans/M1b.md, question 11). The point cell gets one point of weight 1.
pub fn gauss_rule(cell: CellType, degree: u32) -> QuadratureRule {
    todo!()
}
