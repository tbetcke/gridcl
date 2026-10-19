//! Tabulation of Lagrange bases (docs/design.md, "Tabulation and quadrature").

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

use gridcl_dims::CellType;

use crate::{NodeFamily, RefCellError};

/// A Lagrange element on a reference cell: its nodes and the coefficients of its basis in an
/// orthonormal polynomial set, found with gridcl-refcell's own dense LU.
#[derive(Clone, Debug)]
pub struct LagrangeElement {
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    points: Vec<f64>,
    coefficients: Vec<f64>,
}

impl LagrangeElement {
    /// The Lagrange element of degree `degree` on `cell` with nodes of `family`.
    ///
    /// # Errors
    ///
    /// The errors of [`lagrange_points`](crate::lagrange_points), and
    /// [`RefCellError::Singular`] if the Vandermonde matrix is singular.
    pub fn new(cell: CellType, degree: u32, family: NodeFamily) -> Result<Self, RefCellError> {
        todo!()
    }

    /// The cell type.
    pub fn cell(&self) -> CellType {
        self.cell
    }

    /// The degree.
    pub fn degree(&self) -> u32 {
        self.degree
    }

    /// The node family.
    pub fn family(&self) -> NodeFamily {
        self.family
    }

    /// The number of basis functions, [`num_lagrange_nodes`](crate::num_lagrange_nodes).
    pub fn dim(&self) -> usize {
        todo!()
    }

    /// The nodes, as [`lagrange_points`](crate::lagrange_points) returns them.
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// The basis functions at `points` (flat, node-major): values for `nderivs == 0`, values
    /// and first derivatives for `nderivs == 1`.
    ///
    /// # Errors
    ///
    /// - [`RefCellError::UnsupportedDerivativeOrder`] if `nderivs > 1`;
    /// - [`RefCellError::PointsShape`] if `points.len()` is not a multiple of the cell's
    ///   topological dimension.
    pub fn tabulate(&self, points: &[f64], nderivs: u32) -> Result<Tabulation, RefCellError> {
        let _ = &self.coefficients;
        todo!()
    }
}

/// Basis values and derivatives at a set of points.
///
/// `data[(deriv * num_points + point) * num_basis + basis]`, where `deriv` 0 is the value and
/// `1 + d` the derivative along reference coordinate `d`. gridcl-geometry reorders this to
/// Listing B5's `dphi[q][k][d]` when it narrows and uploads.
#[derive(Clone, Debug, PartialEq)]
pub struct Tabulation {
    /// The number of derivative blocks: 1 for values only, 1 + tdim with first derivatives.
    pub num_derivs: usize,
    /// The number of points.
    pub num_points: usize,
    /// The number of basis functions.
    pub num_basis: usize,
    /// The table, laid out as above.
    pub data: Vec<f64>,
}

impl Tabulation {
    /// The entry for derivative block `deriv`, point `point` and basis function `basis`.
    ///
    /// # Panics
    ///
    /// If an index is out of range.
    pub fn get(&self, deriv: usize, point: usize, basis: usize) -> f64 {
        todo!()
    }
}
