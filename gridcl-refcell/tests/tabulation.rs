//! M1b acceptance: partition of unity and reproduction of each element's own space to 1e-12
//! for p ≤ 8 (f64).
//!
//! Tolerances (docs/plans/M1b.md, question 10): |Σφ − 1| ≤ 1e-12 absolute; reproduction to
//! 1e-12 relative to the max norm of the function on the cell, here bounded below by its
//! maximum over the test points (nodes and quadrature points), which makes the check stricter.

use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, NodeFamily, gauss_rule, tdim};

const TOL: f64 = 1e-12;
const MAX_DEGREE: u32 = 8;

/// Every cell with D_TOP ≥ 1 with equispaced nodes, and the tensor cells with GLL nodes.
fn elements() -> Vec<(CellType, NodeFamily)> {
    use CellType::*;
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

/// The test points: the points of `gauss_rule(cell, 2p)` followed by the nodes.
fn test_points(element: &LagrangeElement) -> Vec<f64> {
    let mut points = gauss_rule(element.cell(), 2 * element.degree()).points;
    points.extend_from_slice(element.points());
    points
}

/// A function of the element's own space (A.5): x^a y^b z^c / (1 − z)^r, with r = 0 except on
/// pyramids (Definition 9: r = min(a, b)).
#[derive(Clone, Copy, Debug)]
struct SpaceFunction {
    e: [u32; 3],
    r: u32,
}

impl SpaceFunction {
    fn eval(self, x: &[f64]) -> f64 {
        let mut v = 1.0;
        for (xi, ei) in x.iter().zip(self.e) {
            v *= xi.powi(ei as i32);
        }
        if self.r == 0 {
            return v;
        }
        let s = 1.0 - x[2];
        if s == 0.0 {
            // At the apex x = y = 0; the function tends to 0, since r = min(a, b) ≥ 1 means
            // a + b − r = max(a, b) ≥ 1 powers of (1 − z) remain.
            return 0.0;
        }
        v / s.powi(self.r as i32)
    }
}

/// A basis of the element's own space (A.5).
fn own_space(cell: CellType, p: u32) -> Vec<SpaceFunction> {
    let mut out = Vec::new();
    let r0 = |a, b, c| SpaceFunction { e: [a, b, c], r: 0 };
    for c in 0..=p {
        for b in 0..=p {
            for a in 0..=p {
                let f = match cell {
                    CellType::Interval => (b == 0 && c == 0).then(|| r0(a, 0, 0)),
                    CellType::Triangle => (c == 0 && a + b <= p).then(|| r0(a, b, 0)),
                    CellType::Quadrilateral => (c == 0).then(|| r0(a, b, 0)),
                    CellType::Tetrahedron => (a + b + c <= p).then(|| r0(a, b, c)),
                    CellType::Hexahedron => Some(r0(a, b, c)),
                    CellType::Prism => (a + b <= p).then(|| r0(a, b, c)),
                    CellType::Pyramid => (a + c <= p && b + c <= p).then(|| SpaceFunction {
                        e: [a, b, c],
                        r: a.min(b),
                    }),
                    CellType::Point => unreachable!("no element on the point cell"),
                };
                out.extend(f);
            }
        }
    }
    out
}

#[test]
fn partition_of_unity() {
    for (cell, family) in elements() {
        for p in 1..=MAX_DEGREE {
            let element = LagrangeElement::new(cell, p, family).unwrap();
            let points = test_points(&element);
            let table = element.tabulate(&points, 0).unwrap();
            assert_eq!(table.num_basis, element.dim());
            for q in 0..table.num_points {
                let sum: f64 = (0..table.num_basis).map(|k| table.get(0, q, k)).sum();
                assert!(
                    (sum - 1.0).abs() <= TOL,
                    "{cell:?} {family:?} p = {p}, point {q}: sum {sum} (error {:e})",
                    (sum - 1.0).abs()
                );
            }
        }
    }
}

#[test]
fn reproduces_own_space() {
    for (cell, family) in elements() {
        let td = tdim(cell);
        for p in 1..=MAX_DEGREE {
            let element = LagrangeElement::new(cell, p, family).unwrap();
            let points = test_points(&element);
            let table = element.tabulate(&points, 0).unwrap();
            let space = own_space(cell, p);
            assert_eq!(
                space.len(),
                element.dim(),
                "{cell:?} p = {p}: dimension of the space"
            );
            for f in space {
                let at_nodes: Vec<f64> = element
                    .points()
                    .chunks_exact(td)
                    .map(|x| f.eval(x))
                    .collect();
                let exact: Vec<f64> = points.chunks_exact(td).map(|x| f.eval(x)).collect();
                let scale = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                for (q, want) in exact.iter().enumerate() {
                    let got: f64 = at_nodes
                        .iter()
                        .enumerate()
                        .map(|(k, c)| c * table.get(0, q, k))
                        .sum();
                    let err = (got - want).abs() / scale;
                    assert!(
                        err <= TOL,
                        "{cell:?} {family:?} p = {p}, {f:?} at point {q}: {got} vs {want} (relative error {err:e})"
                    );
                }
            }
        }
    }
}
