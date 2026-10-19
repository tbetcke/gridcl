//! Quadrature rules on the reference cells (docs/design.md, "Tabulation and quadrature").

use gridcl_dims::CellType;

use crate::jacobi::gauss_jacobi;

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
///
/// Exactness by cell: x^a y^b z^c with every exponent at most `degree` on intervals,
/// quadrilaterals and hexahedra; total degree at most `degree` on triangles and tetrahedra;
/// a + b ≤ `degree` and c ≤ `degree` on prisms. Every rule uses m = ⌊degree/2⌋ + 1 points
/// per direction, so the number of points is m^tdim. All points lie strictly inside the cell
/// and all weights are positive. Points are ordered with the first coordinate running fastest.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::gauss_rule;
///
/// // A rule of degree 4 on the reference triangle integrates x² y² exactly: 2! 2! / 6! = 1/180.
/// let rule = gauss_rule(CellType::Triangle, 4);
/// assert_eq!(rule.points.len(), 2 * rule.weights.len());
/// let integral: f64 = rule
///     .points
///     .chunks_exact(2)
///     .zip(&rule.weights)
///     .map(|(p, w)| w * p[0] * p[0] * p[1] * p[1])
///     .sum();
/// assert!((integral - 1.0 / 180.0).abs() < 1e-15);
///
/// // The weights sum to the area 1/2.
/// assert!((rule.weights.iter().sum::<f64>() - 0.5).abs() < 1e-15);
/// ```
pub fn gauss_rule(cell: CellType, degree: u32) -> QuadratureRule {
    // m points are exact to degree 2m − 1 ≥ degree in every collapsed direction.
    let m = degree as usize / 2 + 1;
    let (points, weights) = match cell {
        CellType::Point => (Vec::new(), vec![1.0]),
        CellType::Interval => gauss_jacobi(m, 0.0),
        CellType::Quadrilateral => {
            let g = gauss_jacobi(m, 0.0);
            tensor(&[&g, &g], |p| p.to_vec())
        }
        CellType::Hexahedron => {
            let g = gauss_jacobi(m, 0.0);
            tensor(&[&g, &g, &g], |p| p.to_vec())
        }
        CellType::Triangle => {
            // x = ξ (1 − η), y = η; Jacobian (1 − η) in the Gauss–Jacobi(1, 0) weight.
            let (g, j1) = (gauss_jacobi(m, 0.0), gauss_jacobi(m, 1.0));
            tensor(&[&g, &j1], |p| vec![p[0] * (1.0 - p[1]), p[1]])
        }
        CellType::Tetrahedron => {
            // x = ξ (1 − η)(1 − ζ), y = η (1 − ζ), z = ζ; Jacobian (1 − η)(1 − ζ)².
            let (g, j1, j2) = (
                gauss_jacobi(m, 0.0),
                gauss_jacobi(m, 1.0),
                gauss_jacobi(m, 2.0),
            );
            tensor(&[&g, &j1, &j2], |p| {
                let s = 1.0 - p[2];
                vec![p[0] * (1.0 - p[1]) * s, p[1] * s, p[2]]
            })
        }
        CellType::Prism => {
            // The triangle rule times a Gauss rule in z.
            let (g, j1) = (gauss_jacobi(m, 0.0), gauss_jacobi(m, 1.0));
            tensor(&[&g, &j1, &g], |p| vec![p[0] * (1.0 - p[1]), p[1], p[2]])
        }
        CellType::Pyramid => {
            // x = ξ (1 − z), y = η (1 − z); Jacobian (1 − z)² in the Gauss–Jacobi(2, 0) weight.
            let (g, j2) = (gauss_jacobi(m, 0.0), gauss_jacobi(m, 2.0));
            tensor(&[&g, &g, &j2], |p| {
                let s = 1.0 - p[2];
                vec![p[0] * s, p[1] * s, p[2]]
            })
        }
    };
    QuadratureRule {
        cell,
        degree,
        points,
        weights,
    }
}

/// The tensor product of one-dimensional rules `(points, weights)`, the first running fastest,
/// with each tensor point mapped to the cell by `map`. Returns flat points and weights.
fn tensor(
    rules: &[&(Vec<f64>, Vec<f64>)],
    map: impl Fn(&[f64]) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let total: usize = rules.iter().map(|r| r.1.len()).product();
    let mut points = Vec::with_capacity(total * rules.len());
    let mut weights = Vec::with_capacity(total);
    let mut index = vec![0usize; rules.len()];
    let mut p = vec![0.0; rules.len()];
    for _ in 0..total {
        let mut w = 1.0;
        for (d, (rule, &i)) in rules.iter().zip(&index).enumerate() {
            p[d] = rule.0[i];
            w *= rule.1[i];
        }
        points.extend(map(&p));
        weights.push(w);
        // Advance the multi-index, first direction fastest.
        for (i, rule) in index.iter_mut().zip(rules) {
            *i += 1;
            if *i < rule.1.len() {
                break;
            }
            *i = 0;
        }
    }
    (points, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tdim;

    const CELLS: [CellType; 8] = [
        CellType::Point,
        CellType::Interval,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Hexahedron,
        CellType::Prism,
        CellType::Pyramid,
    ];

    fn volume(cell: CellType) -> f64 {
        match cell {
            CellType::Point
            | CellType::Interval
            | CellType::Quadrilateral
            | CellType::Hexahedron => 1.0,
            CellType::Triangle | CellType::Prism => 0.5,
            CellType::Tetrahedron => 1.0 / 6.0,
            CellType::Pyramid => 1.0 / 3.0,
        }
    }

    /// Whether `p` lies strictly inside the reference cell (A.1).
    fn strictly_inside(cell: CellType, p: &[f64]) -> bool {
        let unit = |t: f64| 0.0 < t && t < 1.0;
        match cell {
            CellType::Point => p.is_empty(),
            CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron => {
                p.iter().all(|&t| unit(t))
            }
            CellType::Triangle => p[0] > 0.0 && p[1] > 0.0 && p[0] + p[1] < 1.0,
            CellType::Tetrahedron => p.iter().all(|&t| t > 0.0) && p[0] + p[1] + p[2] < 1.0,
            CellType::Prism => p[0] > 0.0 && p[1] > 0.0 && p[0] + p[1] < 1.0 && unit(p[2]),
            CellType::Pyramid => {
                p[0] > 0.0 && p[1] > 0.0 && p[2] > 0.0 && p[0].max(p[1]) + p[2] < 1.0
            }
        }
    }

    #[test]
    fn weights_sum_to_the_cell_volume() {
        for cell in CELLS {
            for n in 0..=20 {
                let rule = gauss_rule(cell, n);
                let sum: f64 = rule.weights.iter().sum();
                let v = volume(cell);
                assert!(
                    ((sum - v) / v).abs() <= 1e-14,
                    "{cell:?}, degree {n}: {sum}"
                );
                assert!(rule.weights.iter().all(|&w| w > 0.0), "{cell:?}: positive");
            }
        }
    }

    #[test]
    fn points_lie_strictly_inside_the_cell() {
        for cell in CELLS {
            let td = tdim(cell);
            for n in 0..=20 {
                let rule = gauss_rule(cell, n);
                assert_eq!(rule.cell, cell);
                assert_eq!(rule.degree, n);
                assert_eq!(rule.points.len(), rule.weights.len() * td);
                if td > 0 {
                    for p in rule.points.chunks_exact(td) {
                        assert!(strictly_inside(cell, p), "{cell:?}, degree {n}: {p:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn point_count_is_m_to_the_tdim() {
        for cell in CELLS {
            for n in 0..=20u32 {
                let m = n as usize / 2 + 1;
                let expected = m.pow(tdim(cell) as u32);
                assert_eq!(gauss_rule(cell, n).weights.len(), expected, "{cell:?}, {n}");
            }
        }
    }

    #[test]
    fn point_rule_is_one_point_of_weight_one() {
        for n in [0, 1, 7] {
            let rule = gauss_rule(CellType::Point, n);
            assert_eq!(
                rule,
                QuadratureRule {
                    cell: CellType::Point,
                    degree: n,
                    points: Vec::new(),
                    weights: vec![1.0],
                }
            );
        }
    }

    #[test]
    fn tensor_order_runs_the_first_coordinate_fastest() {
        let rule = gauss_rule(CellType::Quadrilateral, 2);
        let (g, _) = gauss_jacobi(2, 0.0);
        let expected: Vec<f64> = [(0, 0), (1, 0), (0, 1), (1, 1)]
            .iter()
            .flat_map(|&(i, j)| [g[i], g[j]])
            .collect();
        assert_eq!(rule.points, expected);
    }
}
