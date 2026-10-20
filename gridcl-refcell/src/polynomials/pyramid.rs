//! An orthonormal rationomial basis of A.5's pyramid space (DefElement Definition 9).
//!
//! With s = 1 − z, ξ = x/s and η = y/s, the functions
//!
//! P_i(2ξ − 1) P_j(2η − 1) s^max(i,j) P_k^(2 max(i,j) + 2, 0)(2z − 1),  max(i, j) + k ≤ p,
//!
//! normalised by √((2i + 1)(2j + 1)(2k + 2 max(i, j) + 3)), are orthonormal on the reference
//! pyramid (dx dy dz = s² dξ dη dz) and span exactly the space x^a y^b z^c / s^min(a,b) with
//! a + c ≤ p and b + c ≤ p, whose dimension is A.4's (p + 1)(p + 2)(2p + 3)/6.
//!
//! s^i P_i(2ξ − 1) = Q_i(2x − s, s) is a polynomial (a scaled Legendre polynomial), so each
//! function is Q_i Q_j / s^min(i,j) times a Jacobi polynomial in z: rational only when
//! min(i, j) ≥ 1. At the apex (s = 0) those functions are 0 and so is their limit along the
//! pyramid's axis x = y = s/2 for every first derivative; the apex returns these limits. (Their
//! derivatives have no limit over all directions, as for any non-polynomial function in the
//! pyramid space.)

use super::scaled_jacobi;

/// The orthonormal pyramid basis of degree `p` at `x`, as (value, ∂/∂x, ∂/∂y, ∂/∂z).
pub(super) fn orthonormal_at(p: usize, x: &[f64]) -> Vec<[f64; 4]> {
    let s = 1.0 - x[2];
    let a = scaled_jacobi(p, 0.0, 0.0, 2.0 * x[0] - s, s);
    let b = scaled_jacobi(p, 0.0, 0.0, 2.0 * x[1] - s, s);
    let mut out = Vec::with_capacity((p + 1) * (p + 2) * (2 * p + 3) / 6);
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let m = i.max(j);
            if m > p {
                continue;
            }
            let n = i.min(j) as i32;
            // F = Q_i Q_j / s^n and its derivatives (∂(2x − s)/∂z = 1, ∂s/∂z = −1).
            let f = if n > 0 && s == 0.0 {
                [0.0; 4]
            } else {
                let r = s.powi(-n);
                let az = ai.dx - ai.ds;
                let bz = bj.dx - bj.ds;
                // ∂(s^−n)/∂z = n s^−n−1, present only for n ≥ 1 (then s > 0 here).
                let rz = if n > 0 { f64::from(n) * r / s } else { 0.0 };
                [
                    ai.v * bj.v * r,
                    2.0 * ai.dx * bj.v * r,
                    2.0 * ai.v * bj.dx * r,
                    (az * bj.v + ai.v * bz) * r + ai.v * bj.v * rz,
                ]
            };
            let c = scaled_jacobi(p - m, (2 * m + 2) as f64, 0.0, 2.0 * x[2] - 1.0, 1.0);
            for (k, ck) in c.iter().enumerate() {
                let norm = (((2 * i + 1) * (2 * j + 1) * (2 * k + 2 * m + 3)) as f64).sqrt();
                out.push([
                    norm * f[0] * ck.v,
                    norm * f[1] * ck.v,
                    norm * f[2] * ck.v,
                    norm * (f[3] * ck.v + f[0] * 2.0 * ck.dx),
                ]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::orthonormal_at;
    use crate::{NodeFamily, gauss_rule, lagrange_points, num_lagrange_nodes};
    use gridcl_dims::CellType::Pyramid;

    #[test]
    fn dimension_is_a4() {
        for p in 0..=10 {
            let n = orthonormal_at(p, &[0.2, 0.3, 0.1]).len();
            assert_eq!(n, num_lagrange_nodes(Pyramid, p as u32), "p = {p}");
        }
    }

    #[test]
    fn orthonormal() {
        for p in 0..=8 {
            let rule = gauss_rule(Pyramid, 2 * p as u32);
            let rows: Vec<Vec<f64>> = rule
                .points
                .as_chunks::<3>()
                .0
                .iter()
                .map(|x| orthonormal_at(p, x).iter().map(|v| v[0]).collect())
                .collect();
            let nb = rows[0].len();
            for k in 0..nb {
                for l in k..nb {
                    let g: f64 = rows
                        .iter()
                        .zip(&rule.weights)
                        .map(|(r, w)| w * r[k] * r[l])
                        .sum();
                    let want = if k == l { 1.0 } else { 0.0 };
                    assert!((g - want).abs() < 1e-12, "p = {p}: <{k}, {l}> = {g}");
                }
            }
        }
    }

    /// The span is A.5's set: every rationomial ξ^a η^b s^max(a,b) z^c with a + c ≤ p and
    /// b + c ≤ p equals its projection onto the basis (Parseval), and the counts agree.
    #[test]
    fn spans_the_pyramid_space() {
        for p in 0..=6 {
            let rule = gauss_rule(Pyramid, 2 * p as u32);
            let rows: Vec<Vec<f64>> = rule
                .points
                .as_chunks::<3>()
                .0
                .iter()
                .map(|x| orthonormal_at(p, x).iter().map(|v| v[0]).collect())
                .collect();
            let mut count = 0;
            for c in 0..=p as i32 {
                for b in 0..=p as i32 - c {
                    for a in 0..=p as i32 - c {
                        count += 1;
                        let f = |x: &[f64]| {
                            let s = 1.0 - x[2];
                            (x[0] / s).powi(a)
                                * (x[1] / s).powi(b)
                                * s.powi(a.max(b))
                                * x[2].powi(c)
                        };
                        let mut norm2 = 0.0;
                        let mut proj = vec![0.0; rows[0].len()];
                        for ((x, w), r) in rule
                            .points
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .zip(&rule.weights)
                            .zip(&rows)
                        {
                            let fx = f(x);
                            norm2 += w * fx * fx;
                            for (pk, rk) in proj.iter_mut().zip(r) {
                                *pk += w * fx * rk;
                            }
                        }
                        let proj2: f64 = proj.iter().map(|v| v * v).sum();
                        assert!(
                            (norm2 - proj2).abs() < 1e-12 * norm2,
                            "p = {p}, ({a}, {b}, {c})"
                        );
                    }
                }
            }
            assert_eq!(count, num_lagrange_nodes(Pyramid, p as u32));
        }
    }

    #[test]
    fn derivatives_match_central_differences_away_from_the_apex() {
        let h = 1e-6;
        for p in [1, 4, 8] {
            let points = lagrange_points(Pyramid, 7, NodeFamily::Equispaced).unwrap();
            for x in points.as_chunks::<3>().0.iter() {
                // Strictly inside, with room for the differences.
                if x.iter().any(|&c| c < h) || x[0] + x[2] > 1.0 - h || x[1] + x[2] > 1.0 - h {
                    continue;
                }
                let at = orthonormal_at(p, x);
                for d in 0..3 {
                    let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
                    xp[d] += h;
                    xm[d] -= h;
                    let (fp, fm) = (orthonormal_at(p, &xp), orthonormal_at(p, &xm));
                    for k in 0..at.len() {
                        let fd = (fp[k][0] - fm[k][0]) / (2.0 * h);
                        let scale = at[k][1 + d].abs().max(1.0) * (p * p) as f64;
                        assert!(
                            (fd - at[k][1 + d]).abs() < 1e-6 * scale,
                            "p = {p}, basis {k}, d = {d} at {x:?}: {fd} vs {}",
                            at[k][1 + d]
                        );
                    }
                }
            }
        }
    }

    /// The apex returns the limit along the axis x = y = s/2.
    #[test]
    fn apex_is_the_axis_limit() {
        for p in 1..=6 {
            let apex = orthonormal_at(p, &[0.0, 0.0, 1.0]);
            // A power of two, so 1 − eps and eps/2 are exact and the point is on the axis.
            let eps = 2f64.powi(-40);
            let near = orthonormal_at(p, &[eps / 2.0, eps / 2.0, 1.0 - eps]);
            for (k, (a, n)) in apex.iter().zip(&near).enumerate() {
                for d in 0..4 {
                    assert!(
                        (a[d] - n[d]).abs() < 1e-6 * n[d].abs().max(1.0),
                        "p = {p}, {k}, {d}: {a:?} vs {n:?}"
                    );
                }
            }
        }
    }
}
