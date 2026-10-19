//! Tabulation of Lagrange bases (docs/design.md, "Tabulation and quadrature").
//!
//! The Lagrange basis ψ_j is expanded in the cell's orthonormal set φ_k (`polynomials.rs`):
//! ψ_j = Σ_k C_kj φ_k. The Kronecker property ψ_j(x_i) = δ_ij at the nodes x_i gives V C = I
//! for the generalised Vandermonde matrix V_ik = φ_k(x_i), solved with the crate's own LU.

use gridcl_dims::CellType;

use crate::lu::Lu;
use crate::{NodeFamily, RefCellError, lagrange_points, num_lagrange_nodes, polynomials, tdim};

/// A Lagrange element on a reference cell: its nodes and the coefficients of its basis in an
/// orthonormal polynomial set, found with gridcl-refcell's own dense LU.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::{LagrangeElement, NodeFamily};
///
/// let element = LagrangeElement::new(CellType::Triangle, 2, NodeFamily::Equispaced).unwrap();
/// assert_eq!(element.dim(), 6);
///
/// // Values and first derivatives at the centroid.
/// let table = element.tabulate(&[1.0 / 3.0, 1.0 / 3.0], 1).unwrap();
/// assert_eq!((table.num_derivs, table.num_points, table.num_basis), (3, 1, 6));
/// let sum: f64 = (0..6).map(|k| table.get(0, 0, k)).sum();
/// assert!((sum - 1.0).abs() < 1e-14);
/// // The derivatives of a partition of unity sum to zero.
/// let dsum: f64 = (0..6).map(|k| table.get(1, 0, k)).sum();
/// assert!(dsum.abs() < 1e-13);
/// ```
#[derive(Clone, Debug)]
pub struct LagrangeElement {
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    points: Vec<f64>,
    /// C, row-major N × N: ψ_j = Σ_k C[k * N + j] φ_k.
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
        let points = lagrange_points(cell, degree, family)?;
        let n = num_lagrange_nodes(cell, degree);
        let vandermonde = polynomials::tabulate(cell, degree as usize, &points, 0);
        let mut coefficients = vec![0.0; n * n];
        for i in 0..n {
            coefficients[i * n + i] = 1.0;
        }
        Lu::factor(vandermonde, n)?.solve(&mut coefficients);
        Ok(Self {
            cell,
            degree,
            family,
            points,
            coefficients,
        })
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
        num_lagrange_nodes(self.cell, self.degree)
    }

    /// The nodes, as [`lagrange_points`](crate::lagrange_points) returns them.
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// The basis functions at `points` (flat, node-major): values for `nderivs == 0`, values
    /// and first derivatives for `nderivs == 1`. For the point cell, `points` must be empty and
    /// stands for its one point.
    ///
    /// # Errors
    ///
    /// - [`RefCellError::UnsupportedDerivativeOrder`] if `nderivs > 1`;
    /// - [`RefCellError::PointsShape`] if `points.len()` is not a multiple of the cell's
    ///   topological dimension.
    pub fn tabulate(&self, points: &[f64], nderivs: u32) -> Result<Tabulation, RefCellError> {
        if nderivs > 1 {
            return Err(RefCellError::UnsupportedDerivativeOrder(nderivs));
        }
        let td = tdim(self.cell);
        let shape_ok = if td == 0 {
            points.is_empty()
        } else {
            points.len().is_multiple_of(td)
        };
        if !shape_ok {
            return Err(RefCellError::PointsShape {
                len: points.len(),
                tdim: td,
            });
        }
        let num_points = points.len().checked_div(td).unwrap_or(1);
        let num_derivs = if nderivs == 0 { 1 } else { 1 + td };
        let n = self.dim();
        let phi = polynomials::tabulate(self.cell, self.degree as usize, points, nderivs as usize);
        // data[r][j] = Σ_k phi[r][k] C[k][j] for every row r = (deriv, point).
        let mut data = vec![0.0; num_derivs * num_points * n];
        for (row, out) in phi.chunks_exact(n).zip(data.chunks_exact_mut(n)) {
            for (k, &f) in row.iter().enumerate() {
                if f != 0.0 {
                    for (o, c) in out.iter_mut().zip(&self.coefficients[k * n..(k + 1) * n]) {
                        *o += f * c;
                    }
                }
            }
        }
        Ok(Tabulation {
            num_derivs,
            num_points,
            num_basis: n,
            data,
        })
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
        assert!(
            deriv < self.num_derivs && point < self.num_points && basis < self.num_basis,
            "Tabulation::get: ({deriv}, {point}, {basis}) is outside ({}, {}, {})",
            self.num_derivs,
            self.num_points,
            self.num_basis
        );
        self.data[(deriv * self.num_points + point) * self.num_basis + basis]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gauss_rule;
    use CellType::*;

    fn elements() -> Vec<(CellType, NodeFamily)> {
        let mut out: Vec<_> = [
            Interval,
            Triangle,
            Quadrilateral,
            Tetrahedron,
            Hexahedron,
            Prism,
            Pyramid,
        ]
        .into_iter()
        .map(|c| (c, NodeFamily::Equispaced))
        .collect();
        out.extend([Interval, Quadrilateral, Hexahedron].map(|c| (c, NodeFamily::Gll)));
        out
    }

    #[test]
    fn kronecker_at_the_nodes() {
        for (cell, family) in elements() {
            for p in 1..=5 {
                let e = LagrangeElement::new(cell, p, family).unwrap();
                let t = e.tabulate(e.points(), 0).unwrap();
                for i in 0..e.dim() {
                    for j in 0..e.dim() {
                        let want = if i == j { 1.0 } else { 0.0 };
                        assert!(
                            (t.get(0, i, j) - want).abs() < 1e-12,
                            "{cell:?} {family:?} p = {p}: ({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    /// x^a y^b z^c / (1 − z)^r and its gradient; r = min(a, b) on pyramids, else 0.
    fn space_function(e: [i32; 3], r: i32, x: &[f64]) -> (f64, [f64; 3]) {
        let mut xs = [0.0; 3];
        xs[..x.len()].copy_from_slice(x);
        let pw = |v: f64, k: i32| if k < 0 { 0.0 } else { v.powi(k) };
        let s = 1.0 - xs[2];
        let base = pw(xs[0], e[0]) * pw(xs[1], e[1]) * pw(xs[2], e[2]);
        // At the apex x = y = 0 and the value tends to 0 for r ≥ 1 (the gradient is only
        // needed at quadrature points, inside the cell).
        let v = if r > 0 && s == 0.0 {
            0.0
        } else {
            base / s.powi(r)
        };
        let dx =
            f64::from(e[0]) * pw(xs[0], e[0] - 1) * pw(xs[1], e[1]) * pw(xs[2], e[2]) / s.powi(r);
        let dy =
            f64::from(e[1]) * pw(xs[0], e[0]) * pw(xs[1], e[1] - 1) * pw(xs[2], e[2]) / s.powi(r);
        let dz = f64::from(e[2]) * pw(xs[0], e[0]) * pw(xs[1], e[1]) * pw(xs[2], e[2] - 1)
            / s.powi(r)
            + f64::from(r) * base / s.powi(r + 1);
        (v, [dx, dy, dz])
    }

    fn own_space(cell: CellType, p: i32) -> Vec<([i32; 3], i32)> {
        let mut out = vec![];
        for c in 0..=p {
            for b in 0..=p {
                for a in 0..=p {
                    let keep = match cell {
                        Interval => b == 0 && c == 0,
                        Triangle => c == 0 && a + b <= p,
                        Quadrilateral => c == 0,
                        Tetrahedron => a + b + c <= p,
                        Hexahedron => true,
                        Prism => a + b <= p,
                        Pyramid => a + c <= p && b + c <= p,
                        Point => unreachable!(),
                    };
                    if keep {
                        out.push(([a, b, c], if cell == Pyramid { a.min(b) } else { 0 }));
                    }
                }
            }
        }
        out
    }

    /// Interpolation reproduces first derivatives of the element's own space at the points of
    /// `gauss_rule(cell, 2p)`: 1e-10 relative to the largest derivative there (not an
    /// acceptance tolerance).
    #[test]
    fn reproduces_first_derivatives() {
        for (cell, family) in elements() {
            let td = tdim(cell);
            for p in 1..=8 {
                let e = LagrangeElement::new(cell, p, family).unwrap();
                let points = gauss_rule(cell, 2 * p).points;
                let t = e.tabulate(&points, 1).unwrap();
                // The constant has a zero gradient: nothing to scale by.
                for (exps, r) in own_space(cell, p as i32).into_iter().skip(1) {
                    let at_nodes: Vec<f64> = e
                        .points()
                        .chunks_exact(td)
                        .map(|x| space_function(exps, r, x).0)
                        .collect();
                    let exact: Vec<[f64; 3]> = points
                        .chunks_exact(td)
                        .map(|x| space_function(exps, r, x).1)
                        .collect();
                    let scale = exact
                        .iter()
                        .flat_map(|g| g.iter())
                        .fold(1e-300f64, |m, v| m.max(v.abs()));
                    for (q, grad) in exact.iter().enumerate() {
                        for (d, want) in grad.iter().enumerate().take(td) {
                            let got: f64 = at_nodes
                                .iter()
                                .enumerate()
                                .map(|(k, c)| c * t.get(1 + d, q, k))
                                .sum();
                            let err = (got - want).abs() / scale;
                            assert!(
                                err < 1e-10,
                                "{cell:?} {family:?} p = {p}, {exps:?}/{r}, d = {d}: {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn errors() {
        let e = LagrangeElement::new(Tetrahedron, 2, NodeFamily::Equispaced).unwrap();
        assert_eq!(
            e.tabulate(&[0.1, 0.2], 0),
            Err(RefCellError::PointsShape { len: 2, tdim: 3 })
        );
        assert_eq!(
            e.tabulate(&[0.1, 0.2, 0.3], 2),
            Err(RefCellError::UnsupportedDerivativeOrder(2))
        );
        assert_eq!(
            LagrangeElement::new(Triangle, 3, NodeFamily::Gll).unwrap_err(),
            RefCellError::UnsupportedNodeFamily {
                cell: Triangle,
                family: NodeFamily::Gll
            }
        );
        assert_eq!(
            LagrangeElement::new(Quadrilateral, 0, NodeFamily::Equispaced).unwrap_err(),
            RefCellError::UnsupportedDegree {
                cell: Quadrilateral,
                degree: 0
            }
        );
    }

    #[test]
    fn accessors_and_layout() {
        let e = LagrangeElement::new(Prism, 2, NodeFamily::Equispaced).unwrap();
        assert_eq!(
            (e.cell(), e.degree(), e.family(), e.dim()),
            (Prism, 2, NodeFamily::Equispaced, 18)
        );
        assert_eq!(
            e.points(),
            lagrange_points(Prism, 2, NodeFamily::Equispaced).unwrap()
        );
        let t = e.tabulate(&[0.2, 0.3, 0.4, 0.1, 0.1, 0.9], 1).unwrap();
        assert_eq!(
            (t.num_derivs, t.num_points, t.num_basis, t.data.len()),
            (4, 2, 18, 144)
        );
        assert_eq!(t.get(3, 1, 17), t.data[(3 * 2 + 1) * 18 + 17]);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn get_out_of_range() {
        let e = LagrangeElement::new(Interval, 1, NodeFamily::Equispaced).unwrap();
        e.tabulate(&[0.5], 0).unwrap().get(1, 0, 0);
    }
}
