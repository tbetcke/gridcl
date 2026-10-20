//! Orthonormal polynomial sets with first derivatives (docs/design.md, "Tabulation and
//! quadrature").
//!
//! - intervals: Legendre polynomials on [0, 1];
//! - quadrilaterals and hexahedra: tensor products of them;
//! - triangles and tetrahedra: Dubiner's collapsed-coordinate bases;
//! - prisms: Dubiner on the triangle times Legendre in z;
//! - pyramids: the orthonormal rationomials of `pyramid.rs`, spanning A.5's pyramid space.
//!
//! Each set is orthonormal in L²(reference cell) and spans the cell's Lagrange space of degree
//! p (A.5), so its size is [`num_lagrange_nodes`](crate::num_lagrange_nodes). The collapsed
//! factors are evaluated as scaled Jacobi polynomials s^n P_n^(a, b)(x/s), which are
//! polynomials in (x, s); nothing is divided by a collapsed coordinate, so values and
//! derivatives are exact up to rounding at the collapsed vertices too.

mod pyramid;

use gridcl_dims::CellType;

use crate::{num_lagrange_nodes, tdim};

/// A scaled Jacobi polynomial Q_n(x, s) = s^n P_n^(a, b)(x / s) and its partial derivatives.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Scaled {
    /// Q_n(x, s).
    pub(crate) v: f64,
    /// ∂Q_n/∂x.
    pub(crate) dx: f64,
    /// ∂Q_n/∂s.
    pub(crate) ds: f64,
}

/// Q_0, …, Q_n for the Jacobi parameters (a, b) at (x, s), by the homogeneous form of the
/// three-term recurrence (a polynomial in x and s; s = 1 gives P_n^(a, b)(x)).
pub(crate) fn scaled_jacobi(n: usize, a: f64, b: f64, x: f64, s: f64) -> Vec<Scaled> {
    let mut q = Vec::with_capacity(n + 1);
    q.push(Scaled {
        v: 1.0,
        dx: 0.0,
        ds: 0.0,
    });
    if n == 0 {
        return q;
    }
    q.push(Scaled {
        v: 0.5 * ((a + b + 2.0) * x + (a - b) * s),
        dx: 0.5 * (a + b + 2.0),
        ds: 0.5 * (a - b),
    });
    for m in 1..n {
        let m = m as f64;
        let c0 = 2.0 * (m + 1.0) * (m + a + b + 1.0) * (2.0 * m + a + b);
        let c1 = 2.0 * m + a + b + 1.0;
        let c2 = (2.0 * m + a + b + 2.0) * (2.0 * m + a + b);
        let c3 = a * a - b * b;
        let c4 = 2.0 * (m + a) * (m + b) * (2.0 * m + a + b + 2.0);
        let (p1, p0) = (q[q.len() - 1], q[q.len() - 2]);
        let lin = c2 * x + c3 * s;
        q.push(Scaled {
            v: (c1 * lin * p1.v - c4 * s * s * p0.v) / c0,
            dx: (c1 * (c2 * p1.v + lin * p1.dx) - c4 * s * s * p0.dx) / c0,
            ds: (c1 * (c3 * p1.v + lin * p1.ds) - c4 * (2.0 * s * p0.v + s * s * p0.ds)) / c0,
        });
    }
    q
}

/// The orthonormal Legendre polynomials √(2i + 1) P_i(2t − 1) on [0, 1], i = 0..=p, as
/// (value, derivative).
fn legendre(p: usize, t: f64) -> Vec<(f64, f64)> {
    scaled_jacobi(p, 0.0, 0.0, 2.0 * t - 1.0, 1.0)
        .into_iter()
        .enumerate()
        .map(|(i, q)| {
            let c = ((2 * i + 1) as f64).sqrt();
            (c * q.v, 2.0 * c * q.dx)
        })
        .collect()
}

/// The orthonormal Dubiner basis on the reference triangle at (x, y), with i + j ≤ p, as
/// (value, ∂/∂x, ∂/∂y), i outer and j inner.
fn dubiner_triangle(p: usize, x: f64, y: f64) -> Vec<[f64; 3]> {
    let s = 1.0 - y;
    let a = scaled_jacobi(p, 0.0, 0.0, 2.0 * x - s, s);
    let mut out = Vec::with_capacity((p + 1) * (p + 2) / 2);
    for (i, ai) in a.iter().enumerate() {
        let b = scaled_jacobi(p - i, (2 * i + 1) as f64, 0.0, 2.0 * y - 1.0, 1.0);
        for (j, bj) in b.iter().enumerate() {
            let c = (((2 * i + 1) * (2 * i + 2 * j + 2)) as f64).sqrt();
            // ∂(2x − s)/∂y = 1 and ∂s/∂y = −1.
            let ay = ai.dx - ai.ds;
            out.push([
                c * ai.v * bj.v,
                c * 2.0 * ai.dx * bj.v,
                c * (ay * bj.v + ai.v * 2.0 * bj.dx),
            ]);
        }
    }
    out
}

/// The orthonormal Dubiner basis on the reference tetrahedron at (x, y, z), with
/// i + j + k ≤ p, as (value, ∂/∂x, ∂/∂y, ∂/∂z).
fn dubiner_tetrahedron(p: usize, x: f64, y: f64, z: f64) -> Vec<[f64; 4]> {
    let s1 = 1.0 - y - z;
    let s2 = 1.0 - z;
    let a = scaled_jacobi(p, 0.0, 0.0, 2.0 * x - s1, s1);
    let mut out = Vec::with_capacity((p + 1) * (p + 2) * (p + 3) / 6);
    for (i, ai) in a.iter().enumerate() {
        let b = scaled_jacobi(p - i, (2 * i + 1) as f64, 0.0, 2.0 * y - s2, s2);
        // ∂/∂y and ∂/∂z of Q_i(2x − s1, s1): both move s1 by −1.
        let ayz = ai.dx - ai.ds;
        for (j, bj) in b.iter().enumerate() {
            let c = scaled_jacobi(
                p - i - j,
                (2 * i + 2 * j + 2) as f64,
                0.0,
                2.0 * z - 1.0,
                1.0,
            );
            let bz = bj.dx - bj.ds;
            for (k, ck) in c.iter().enumerate() {
                let n = (((2 * i + 1) * (2 * i + 2 * j + 2) * (2 * i + 2 * j + 2 * k + 3)) as f64)
                    .sqrt();
                out.push([
                    n * ai.v * bj.v * ck.v,
                    n * 2.0 * ai.dx * bj.v * ck.v,
                    n * (ayz * bj.v + ai.v * 2.0 * bj.dx) * ck.v,
                    n * (ayz * bj.v * ck.v + ai.v * bz * ck.v + ai.v * bj.v * 2.0 * ck.dx),
                ]);
            }
        }
    }
    out
}

/// The orthonormal basis of degree `p` on `cell` at one point `x` (tdim coordinates), as
/// [value, ∂/∂x_0, …, ∂/∂x_{tdim−1}] per basis function (unused entries are 0).
pub(crate) fn orthonormal_at(cell: CellType, p: usize, x: &[f64]) -> Vec<[f64; 4]> {
    match cell {
        CellType::Point => vec![[1.0, 0.0, 0.0, 0.0]],
        CellType::Interval => legendre(p, x[0])
            .into_iter()
            .map(|(v, d)| [v, d, 0.0, 0.0])
            .collect(),
        CellType::Quadrilateral => {
            let (lx, ly) = (legendre(p, x[0]), legendre(p, x[1]));
            let mut out = Vec::with_capacity((p + 1) * (p + 1));
            for &(vy, dy) in &ly {
                for &(vx, dx) in &lx {
                    out.push([vx * vy, dx * vy, vx * dy, 0.0]);
                }
            }
            out
        }
        CellType::Hexahedron => {
            let (lx, ly, lz) = (legendre(p, x[0]), legendre(p, x[1]), legendre(p, x[2]));
            let mut out = Vec::with_capacity((p + 1).pow(3));
            for &(vz, dz) in &lz {
                for &(vy, dy) in &ly {
                    for &(vx, dx) in &lx {
                        out.push([vx * vy * vz, dx * vy * vz, vx * dy * vz, vx * vy * dz]);
                    }
                }
            }
            out
        }
        CellType::Triangle => dubiner_triangle(p, x[0], x[1])
            .into_iter()
            .map(|[v, dx, dy]| [v, dx, dy, 0.0])
            .collect(),
        CellType::Tetrahedron => dubiner_tetrahedron(p, x[0], x[1], x[2]),
        CellType::Prism => {
            let (t, lz) = (dubiner_triangle(p, x[0], x[1]), legendre(p, x[2]));
            let mut out = Vec::with_capacity(t.len() * lz.len());
            for &(vz, dz) in &lz {
                for &[v, dx, dy] in &t {
                    out.push([v * vz, dx * vz, dy * vz, v * dz]);
                }
            }
            out
        }
        CellType::Pyramid => pyramid::orthonormal_at(p, x),
    }
}

/// The orthonormal set of degree `p` on `cell` at `points` (flat, node-major):
/// `data[(deriv * num_points + point) * num_basis + basis]`, deriv 0 the value and 1 + d the
/// derivative along x_d; `nderivs` is 0 or 1.
pub(crate) fn tabulate(cell: CellType, p: usize, points: &[f64], nderivs: usize) -> Vec<f64> {
    let td = tdim(cell);
    let nb = num_lagrange_nodes(cell, p as u32);
    // The point cell has no coordinates: one point.
    let npts = points.len().checked_div(td).unwrap_or(1);
    let nd = if nderivs == 0 { 1 } else { 1 + td };
    let mut data = vec![0.0; nd * npts * nb];
    for q in 0..npts {
        let x = &points[q * td..(q + 1) * td];
        let values = orthonormal_at(cell, p, x);
        debug_assert_eq!(values.len(), nb);
        for (k, v) in values.iter().enumerate() {
            for d in 0..nd {
                data[(d * npts + q) * nb + k] = v[d];
            }
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeFamily, gauss_rule, lagrange_points};
    use CellType::*;

    /// The cells with polynomial sets in T5 (the pyramid follows in T6).
    const CELLS: [CellType; 6] = [
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
    ];

    /// ∫ φ_k φ_l with `gauss_rule(cell, 2p)`, which is exact for these products.
    fn gram(cell: CellType, p: usize) -> (usize, Vec<f64>) {
        let rule = gauss_rule(cell, 2 * p as u32);
        let t = tabulate(cell, p, &rule.points, 0);
        let nb = num_lagrange_nodes(cell, p as u32);
        let mut g = vec![0.0; nb * nb];
        for (q, w) in rule.weights.iter().enumerate() {
            let row = &t[q * nb..(q + 1) * nb];
            for k in 0..nb {
                for l in 0..nb {
                    g[k * nb + l] += w * row[k] * row[l];
                }
            }
        }
        (nb, g)
    }

    #[test]
    fn scaled_jacobi_matches_jacobi() {
        for (a, b) in [(0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (2.0, 1.0)] {
            for &x in &[-1.0, -0.3, 0.2, 0.9] {
                let q = scaled_jacobi(6, a, b, x, 1.0);
                for (n, qn) in q.iter().enumerate() {
                    let (v, d) = crate::jacobi::jacobi(n, a, b, x);
                    assert!((qn.v - v).abs() < 1e-12 * v.abs().max(1.0), "n = {n}");
                    assert!((qn.dx - d).abs() < 1e-11 * d.abs().max(1.0), "n = {n}");
                }
                // Homogeneity: Q_n(s x, s) = s^n P_n(x).
                let s = 0.37;
                for (n, qn) in scaled_jacobi(6, a, b, s * x, s).iter().enumerate() {
                    let want = s.powi(n as i32) * q[n].v;
                    assert!((qn.v - want).abs() < 1e-13, "n = {n}");
                }
            }
        }
    }

    #[test]
    fn orthonormal() {
        for cell in CELLS {
            for p in 0..=8 {
                let (nb, g) = gram(cell, p);
                for k in 0..nb {
                    for l in 0..nb {
                        let want = if k == l { 1.0 } else { 0.0 };
                        let err = (g[k * nb + l] - want).abs();
                        assert!(
                            err < 1e-12,
                            "{cell:?} p = {p}: <{k}, {l}> = {}",
                            g[k * nb + l]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn derivatives_match_central_differences() {
        let h = 1e-6;
        for cell in CELLS {
            let td = tdim(cell);
            for p in [1, 4, 8] {
                // Nodes of a degree-7 lattice away from the boundary, so that the central
                // differences stay inside the cell (the sum test is conservative for tensor
                // cells).
                let points = lagrange_points(cell, 7, NodeFamily::Equispaced).unwrap();
                for x in points.chunks_exact(td) {
                    if x.iter().any(|&c| c < h || c > 1.0 - h) || x.iter().sum::<f64>() > 1.0 - h {
                        continue;
                    }
                    let at = orthonormal_at(cell, p, x);
                    for d in 0..td {
                        let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
                        xp[d] += h;
                        xm[d] -= h;
                        let (fp, fm) = (orthonormal_at(cell, p, &xp), orthonormal_at(cell, p, &xm));
                        for k in 0..at.len() {
                            let fd = (fp[k][0] - fm[k][0]) / (2.0 * h);
                            let scale = at[k][1 + d].abs().max(1.0);
                            assert!(
                                (fd - at[k][1 + d]).abs() < 1e-6 * scale * (p * p) as f64,
                                "{cell:?} p = {p}, basis {k}, d = {d} at {x:?}: {fd} vs {}",
                                at[k][1 + d]
                            );
                        }
                    }
                }
            }
        }
    }

    /// The monomials of the cell's Lagrange space of degree p (A.5).
    fn monomials(cell: CellType, p: usize) -> Vec<[i32; 3]> {
        let p = p as i32;
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
                        _ => unreachable!(),
                    };
                    if keep {
                        out.push([a, b, c]);
                    }
                }
            }
        }
        out
    }

    /// The set spans the cell's space: it has the space's dimension, and every monomial of the
    /// space equals its orthogonal projection onto the set (Parseval: ‖m‖² = Σ (m, φ_k)²), so
    /// the monomial Vandermonde has the set's full rank in the span.
    #[test]
    fn spans_the_cell_space() {
        for cell in CELLS {
            let td = tdim(cell);
            for p in 0..=6 {
                let ms = monomials(cell, p);
                assert_eq!(
                    ms.len(),
                    num_lagrange_nodes(cell, p as u32),
                    "{cell:?} p = {p}"
                );
                let rule = gauss_rule(cell, 2 * p as u32);
                let t = tabulate(cell, p, &rule.points, 0);
                let nb = ms.len();
                for e in ms {
                    let m =
                        |x: &[f64]| x.iter().zip(e).map(|(xi, ei)| xi.powi(ei)).product::<f64>();
                    let mut norm2 = 0.0;
                    let mut coef = vec![0.0; nb];
                    for (q, (x, w)) in rule.points.chunks_exact(td).zip(&rule.weights).enumerate() {
                        let mx = m(x);
                        norm2 += w * mx * mx;
                        for k in 0..nb {
                            coef[k] += w * mx * t[q * nb + k];
                        }
                    }
                    let proj2: f64 = coef.iter().map(|c| c * c).sum();
                    assert!(
                        (norm2 - proj2).abs() < 1e-12 * norm2,
                        "{cell:?} p = {p}, {e:?}"
                    );
                }
            }
        }
    }
}
