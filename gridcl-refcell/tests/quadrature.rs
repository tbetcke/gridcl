//! M1b acceptance: quadrature integrates monomials (rationomials on pyramids) up to its degree
//! to 1e-13, relative to the exact integral (docs/plans/M1b.md, questions 10 and 11).

use gridcl_dims::CellType;
use gridcl_refcell::{gauss_rule, tdim};

const CELLS: [CellType; 7] = [
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

const TOL: f64 = 1e-13;

fn factorial(n: u32) -> f64 {
    (1..=n).map(f64::from).product()
}

/// ∫ x^a y^b over the reference triangle: a! b! / (a + b + 2)!.
fn triangle_integral(a: u32, b: u32) -> f64 {
    factorial(a) * factorial(b) / factorial(a + b + 2)
}

/// A function of a point in reference coordinates.
type PointFn = Box<dyn Fn(&[f64]) -> f64>;

/// One test function on a cell: its values at a point and its exact integral.
struct Integrand {
    label: String,
    f: PointFn,
    exact: f64,
}

fn monomial(label: String, e: [u32; 3], exact: f64) -> Integrand {
    let f = move |x: &[f64]| x.iter().zip(e).map(|(xi, ei)| xi.powi(ei as i32)).product();
    Integrand {
        label,
        f: Box::new(f),
        exact,
    }
}

/// The integrands a rule of degree n must integrate exactly on `cell`:
/// - interval, quadrilateral, hexahedron: x^a y^b z^c with every exponent ≤ n (tensor rules);
/// - triangle, tetrahedron: total degree ≤ n;
/// - prism: a + b ≤ n and c ≤ n (triangle rule times interval rule);
/// - pyramid: the rationomials ξ^a η^b (1−z)^max(a,b) z^c with max(a,b) + c ≤ n, where
///   ξ = x/(1−z) and η = y/(1−z), i.e. x^a y^b z^c (1−z)^(max(a,b) − a − b) (question 11).
fn integrands(cell: CellType, n: u32) -> Vec<Integrand> {
    let mut out = Vec::new();
    match cell {
        CellType::Interval => {
            for a in 0..=n {
                out.push(monomial(
                    format!("x^{a}"),
                    [a, 0, 0],
                    1.0 / f64::from(a + 1),
                ));
            }
        }
        CellType::Quadrilateral => {
            for b in 0..=n {
                for a in 0..=n {
                    let exact = 1.0 / f64::from((a + 1) * (b + 1));
                    out.push(monomial(format!("x^{a} y^{b}"), [a, b, 0], exact));
                }
            }
        }
        CellType::Hexahedron => {
            for c in 0..=n {
                for b in 0..=n {
                    for a in 0..=n {
                        let exact = 1.0 / f64::from((a + 1) * (b + 1) * (c + 1));
                        out.push(monomial(format!("x^{a} y^{b} z^{c}"), [a, b, c], exact));
                    }
                }
            }
        }
        CellType::Triangle => {
            for b in 0..=n {
                for a in 0..=n - b {
                    out.push(monomial(
                        format!("x^{a} y^{b}"),
                        [a, b, 0],
                        triangle_integral(a, b),
                    ));
                }
            }
        }
        CellType::Tetrahedron => {
            for c in 0..=n {
                for b in 0..=n - c {
                    for a in 0..=n - c - b {
                        // a! b! c! / (a + b + c + 3)!
                        let exact =
                            factorial(a) * factorial(b) * factorial(c) / factorial(a + b + c + 3);
                        out.push(monomial(format!("x^{a} y^{b} z^{c}"), [a, b, c], exact));
                    }
                }
            }
        }
        CellType::Prism => {
            for c in 0..=n {
                for b in 0..=n {
                    for a in 0..=n - b {
                        let exact = triangle_integral(a, b) / f64::from(c + 1);
                        out.push(monomial(format!("x^{a} y^{b} z^{c}"), [a, b, c], exact));
                    }
                }
            }
        }
        CellType::Pyramid => {
            for c in 0..=n {
                for b in 0..=n - c {
                    for a in 0..=n - c {
                        let m = a.max(b);
                        if m + c > n {
                            continue;
                        }
                        // ∫∫∫ ξ^a η^b (1−z)^m z^c (1−z)^2 dξ dη dz
                        //   = 1/((a+1)(b+1)) · c! (m+2)! / (c+m+3)!
                        let exact = factorial(c) * factorial(m + 2)
                            / factorial(c + m + 3)
                            / f64::from((a + 1) * (b + 1));
                        let f = move |x: &[f64]| {
                            let s = 1.0 - x[2];
                            let (xi, eta) = (x[0] / s, x[1] / s);
                            xi.powi(a as i32)
                                * eta.powi(b as i32)
                                * s.powi(m as i32)
                                * x[2].powi(c as i32)
                        };
                        out.push(Integrand {
                            label: format!("xi^{a} eta^{b} (1-z)^{m} z^{c}"),
                            f: Box::new(f),
                            exact,
                        });
                    }
                }
            }
        }
        CellType::Point => unreachable!("the point cell has no monomials"),
    }
    out
}

#[test]
fn integrates_monomials() {
    for cell in CELLS {
        let td = tdim(cell);
        for n in 0..=20 {
            let rule = gauss_rule(cell, n);
            assert_eq!(rule.cell, cell);
            assert_eq!(
                rule.points.len(),
                rule.weights.len() * td,
                "{cell:?}, degree {n}"
            );
            for g in integrands(cell, n) {
                let approx: f64 = rule
                    .points
                    .chunks_exact(td)
                    .zip(&rule.weights)
                    .map(|(x, w)| w * (g.f)(x))
                    .sum();
                let err = (approx - g.exact).abs() / g.exact;
                assert!(
                    err <= TOL,
                    "{cell:?}, degree {n}, {}: {approx} vs {} (relative error {err:e})",
                    g.label,
                    g.exact
                );
            }
        }
    }
}
