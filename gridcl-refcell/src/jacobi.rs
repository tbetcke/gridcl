//! Jacobi polynomials, Gauss–Jacobi rules and Gauss–Lobatto–Legendre points.
//!
//! Host code in f64 (docs/design.md, "Tabulation and quadrature"). The polynomials are
//! evaluated by their three-term recurrence; the nodes are found by Newton iteration to
//! machine precision. Rules are returned on [0, 1], the reference interval (A.1).

/// The maximum number of Newton steps per node. Convergence normally takes fewer than ten.
const MAX_NEWTON_STEPS: usize = 100;

/// A Newton update at or below this many ulps of the node ends the iteration.
const NEWTON_ULPS: f64 = 4.0;

/// The Jacobi polynomial P_n^(α, β) and its derivative at `x` in [−1, 1].
///
/// Orthogonal on [−1, 1] for the weight (1 − x)^α (1 + x)^β, with α, β > −1 and the standard
/// normalisation P_n^(α, β)(1) = binom(n + α, n). Evaluated with the three-term recurrence and
/// its derivative, never a closed form. Returns `(value, derivative)`.
pub(crate) fn jacobi(n: usize, alpha: f64, beta: f64, x: f64) -> (f64, f64) {
    debug_assert!(
        alpha > -1.0 && beta > -1.0,
        "Jacobi parameters must exceed −1"
    );
    if n == 0 {
        return (1.0, 0.0);
    }
    let ab = alpha + beta;
    // P_1 = (α − β)/2 + (α + β + 2)/2 x.
    let (mut p_prev, mut dp_prev) = (1.0, 0.0);
    let mut p = 0.5 * (alpha - beta) + 0.5 * (ab + 2.0) * x;
    let mut dp = 0.5 * (ab + 2.0);
    for k in 2..=n {
        let k = k as f64;
        let c = 2.0 * k + ab;
        // 2k (k + α + β)(2k + α + β − 2) P_k
        //   = (2k + α + β − 1)[(2k + α + β)(2k + α + β − 2) x + α² − β²] P_{k−1}
        //     − 2 (k + α − 1)(k + β − 1)(2k + α + β) P_{k−2}
        let denom = 2.0 * k * (k + ab) * (c - 2.0);
        let a1 = (c - 1.0) * c * (c - 2.0);
        let a0 = (c - 1.0) * (alpha * alpha - beta * beta);
        let b = 2.0 * (k + alpha - 1.0) * (k + beta - 1.0) * c;
        let p_next = ((a1 * x + a0) * p - b * p_prev) / denom;
        let dp_next = ((a1 * x + a0) * dp + a1 * p - b * dp_prev) / denom;
        (p_prev, dp_prev, p, dp) = (p, dp, p_next, dp_next);
    }
    (p, dp)
}

/// Whether the Newton update `delta` at the node `x` is small enough to stop.
fn converged(delta: f64, x: f64) -> bool {
    delta.abs() <= NEWTON_ULPS * f64::EPSILON * x.abs().max(1.0)
}

/// The m-point Gauss–Jacobi rule on [0, 1] for the weight (1 − x)^α.
///
/// α = 0 gives Gauss–Legendre. The points ascend and lie strictly inside (0, 1); the weights
/// are positive and sum to ∫₀¹ (1 − x)^α dx = 1/(α + 1). The rule is exact for polynomials of
/// degree 2m − 1 against the weight. Returns `(points, weights)`.
///
/// The nodes are the zeros of P_m^(α, 0) on [−1, 1], mapped by t = (1 + x)/2, found by Newton
/// iteration with deflation by the nodes already found. The weights are 1/((1 − x²) P_m'(x)²)
/// at the converged nodes, scaled to the exact total.
///
/// # Panics
///
/// If `m` is 0 or α ≤ −1.
pub(crate) fn gauss_jacobi(m: usize, alpha: f64) -> (Vec<f64>, Vec<f64>) {
    assert!(m >= 1, "a Gauss–Jacobi rule needs at least one point");
    assert!(alpha > -1.0, "the Gauss–Jacobi parameter must exceed −1");
    let mut nodes: Vec<f64> = Vec::with_capacity(m);
    for k in 0..m {
        // Chebyshev–Gauss initial guess, averaged with the previous node to stay to its right.
        let mut x = -(std::f64::consts::PI * (2 * k + 1) as f64 / (2 * m) as f64).cos();
        if let Some(&prev) = nodes.last() {
            x = 0.5 * (x + prev);
        }
        for _ in 0..MAX_NEWTON_STEPS {
            let (p, dp) = jacobi(m, alpha, 0.0, x);
            let deflation: f64 = nodes.iter().map(|&r| 1.0 / (x - r)).sum();
            let delta = p / (dp - deflation * p);
            x -= delta;
            if converged(delta, x) {
                break;
            }
        }
        nodes.push(x);
    }
    let raw: Vec<f64> = nodes
        .iter()
        .map(|&x| {
            let (_, dp) = jacobi(m, alpha, 0.0, x);
            1.0 / ((1.0 - x * x) * dp * dp)
        })
        .collect();
    let scale = 1.0 / ((alpha + 1.0) * raw.iter().sum::<f64>());
    let points = nodes.iter().map(|&x| 0.5 * (1.0 + x)).collect();
    let weights = raw.iter().map(|&w| w * scale).collect();
    (points, weights)
}

/// The m Gauss–Lobatto–Legendre points on [0, 1], ascending.
///
/// The endpoints are exactly 0.0 and 1.0; the interior points are the zeros of P'_{m−1}
/// (equivalently of P_{m−2}^(1, 1)), found by Newton iteration from the Chebyshev–Gauss–Lobatto
/// points. Only the lower half is computed; the upper half is its mirror image t ↦ 1 − t, with
/// the lower points rounded so that the mirror is exact both ways, and the midpoint of an odd
/// m is exactly 0.5, so the set is exactly symmetric.
///
/// # Panics
///
/// If `m` < 2.
// Used by the GLL Lagrange layouts (M1b T4).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn gll_points(m: usize) -> Vec<f64> {
    assert!(m >= 2, "GLL points need at least the two endpoints");
    let n = m - 1;
    let mut points = vec![0.0; m];
    points[n] = 1.0;
    // Interior lower half: indices 1..m/2 (the midpoint of odd m is set below).
    for i in 1..m / 2 {
        let mut x = -(std::f64::consts::PI * i as f64 / n as f64).cos();
        for _ in 0..MAX_NEWTON_STEPS {
            let (p, dp) = jacobi(n - 1, 1.0, 1.0, x);
            let delta = p / dp;
            x -= delta;
            if converged(delta, x) {
                break;
            }
        }
        // Round t to 1 − u with u = fl(1 − t) ∈ [1/2, 1]: both subtractions are then exact
        // (Sterbenz), so points[n − i] = 1 − points[i] and points[i] = 1 − points[n − i].
        let u = 1.0 - 0.5 * (1.0 + x);
        points[i] = 1.0 - u;
        points[n - i] = u;
    }
    if m % 2 == 1 {
        points[m / 2] = 0.5;
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b} (difference {:e})", a - b);
    }

    fn factorial(n: u32) -> f64 {
        (1..=n).map(f64::from).product()
    }

    #[test]
    fn jacobi_matches_closed_forms() {
        for i in 0..=20 {
            let x = -1.0 + 0.1 * f64::from(i);
            // Legendre P_2 and P_3.
            let (p, dp) = jacobi(2, 0.0, 0.0, x);
            assert_close(p, 0.5 * (3.0 * x * x - 1.0), 1e-15);
            assert_close(dp, 3.0 * x, 1e-15);
            let (p, dp) = jacobi(3, 0.0, 0.0, x);
            assert_close(p, 0.5 * (5.0 * x * x * x - 3.0 * x), 1e-15);
            assert_close(dp, 0.5 * (15.0 * x * x - 3.0), 1e-14);
            // P_1^(α, β) = (α − β)/2 + (α + β + 2)/2 x with α = 2, β = 1.
            let (p, dp) = jacobi(1, 2.0, 1.0, x);
            assert_close(p, 0.5 + 2.5 * x, 1e-15);
            assert_close(dp, 2.5, 1e-15);
            // P_2^(1, 1) = (15 x² − 3)/4, a multiple of P_3'.
            let (p, dp) = jacobi(2, 1.0, 1.0, x);
            assert_close(p, 0.25 * (15.0 * x * x - 3.0), 1e-14);
            assert_close(dp, 7.5 * x, 1e-14);
        }
        // P_n^(α, β)(1) = binom(n + α, n).
        let (p, _) = jacobi(5, 2.0, 0.0, 1.0);
        assert_close(p, 21.0, 1e-13);
    }

    #[test]
    fn gauss_legendre_matches_closed_forms() {
        let (x, w) = gauss_jacobi(1, 0.0);
        assert_eq!((x, w), (vec![0.5], vec![1.0]));

        let (x, w) = gauss_jacobi(2, 0.0);
        let d = 0.5 / 3.0f64.sqrt();
        assert_close(x[0], 0.5 - d, 1e-15);
        assert_close(x[1], 0.5 + d, 1e-15);
        assert_close(w[0], 0.5, 1e-15);
        assert_close(w[1], 0.5, 1e-15);

        let (x, w) = gauss_jacobi(3, 0.0);
        let d = 0.5 * 0.6f64.sqrt();
        for (xi, ei) in x.iter().zip([0.5 - d, 0.5, 0.5 + d]) {
            assert_close(*xi, ei, 1e-15);
        }
        for (wi, ei) in w.iter().zip([5.0 / 18.0, 8.0 / 18.0, 5.0 / 18.0]) {
            assert_close(*wi, ei, 1e-15);
        }
    }

    #[test]
    fn gauss_jacobi_matches_closed_forms() {
        // One point: the mean of the weight, ∫ x (1 − x)^α / ∫ (1 − x)^α = 1/(α + 2).
        let (x, w) = gauss_jacobi(1, 1.0);
        assert_close(x[0], 1.0 / 3.0, 1e-15);
        assert_close(w[0], 0.5, 1e-15);
        let (x, w) = gauss_jacobi(1, 2.0);
        assert_close(x[0], 0.25, 1e-15);
        assert_close(w[0], 1.0 / 3.0, 1e-15);
        // Two points for (1 − x): the zeros of x² − 4x/5 + 1/10, i.e. 2/5 ∓ √6/10.
        let (x, w) = gauss_jacobi(2, 1.0);
        let d = 6.0f64.sqrt() / 10.0;
        assert_close(x[0], 0.4 - d, 1e-15);
        assert_close(x[1], 0.4 + d, 1e-15);
        // The weights follow from exactness for 1 and x: w0 + w1 = 1/2, w0 x0 + w1 x1 = 1/6.
        let w1 = (1.0 / 6.0 - 0.5 * x[0]) / (x[1] - x[0]);
        assert_close(w[1], w1, 1e-15);
        assert_close(w[0], 0.5 - w1, 1e-15);
    }

    #[test]
    fn gauss_jacobi_is_exact_to_degree_2m_minus_1() {
        for alpha in [0u32, 1, 2] {
            for m in 1..=15 {
                let (x, w) = gauss_jacobi(m, f64::from(alpha));
                assert!(x.windows(2).all(|p| p[0] < p[1]), "ascending");
                assert!(x[0] > 0.0 && x[m - 1] < 1.0, "interior");
                assert!(w.iter().all(|&wi| wi > 0.0), "positive weights");
                for k in 0..2 * m as u32 {
                    // ∫₀¹ x^k (1 − x)^α dx = k! α! / (k + α + 1)!
                    let exact = factorial(k) * factorial(alpha) / factorial(k + alpha + 1);
                    let approx: f64 = x
                        .iter()
                        .zip(&w)
                        .map(|(xi, wi)| wi * xi.powi(k as i32))
                        .sum();
                    assert!(
                        ((approx - exact) / exact).abs() <= 1e-13,
                        "α = {alpha}, m = {m}, x^{k}: {approx} vs {exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn gll_points_match_closed_forms() {
        assert_eq!(gll_points(2), [0.0, 1.0]);
        assert_eq!(gll_points(3), [0.0, 0.5, 1.0]);
        // Interior points on [−1, 1]: ±1/√5 (m = 4), 0 and ±√(3/7) (m = 5),
        // ±√(1/3 ∓ 2√7/21) (m = 6).
        let to01 = |x: f64| 0.5 * (1.0 + x);
        let s5 = 0.2f64.sqrt();
        let s7 = (3.0f64 / 7.0).sqrt();
        let r7 = 2.0 * 7.0f64.sqrt() / 21.0;
        let (a6, b6) = ((1.0 / 3.0 + r7).sqrt(), (1.0 / 3.0 - r7).sqrt());
        let expected: [&[f64]; 3] = [
            &[0.0, to01(-s5), to01(s5), 1.0],
            &[0.0, to01(-s7), 0.5, to01(s7), 1.0],
            &[0.0, to01(-a6), to01(-b6), to01(b6), to01(a6), 1.0],
        ];
        for (m, e) in (4..=6).zip(expected) {
            let p = gll_points(m);
            assert_eq!(p.len(), m);
            for (pi, ei) in p.iter().zip(e) {
                assert_close(*pi, *ei, 1e-15);
            }
        }
    }

    #[test]
    fn gll_points_are_symmetric_zeros_of_the_derivative() {
        for m in 2..=40 {
            let p = gll_points(m);
            assert_eq!(p.len(), m);
            assert_eq!(p[0], 0.0);
            assert_eq!(p[m - 1], 1.0);
            assert!(p.windows(2).all(|q| q[0] < q[1]), "m = {m}: ascending");
            for i in 0..m {
                assert_eq!(p[i], 1.0 - p[m - 1 - i], "m = {m}: symmetric");
            }
            for &t in &p[1..m - 1] {
                // Zero of P'_{m−1}, scaled by the derivative's size at the endpoint.
                let (_, dp) = jacobi(m - 1, 0.0, 0.0, 2.0 * t - 1.0);
                let scale = jacobi(m - 1, 0.0, 0.0, 1.0).1;
                assert!(dp.abs() <= 1e-13 * scale, "m = {m}: P' = {dp}");
            }
        }
    }

    #[test]
    fn gll_rule_is_exact_to_degree_2m_minus_3() {
        // Weights 2 / (n (n + 1) P_n(x)²) on [−1, 1], n = m − 1, halved for [0, 1].
        for m in 2..=20 {
            let n = m - 1;
            let p = gll_points(m);
            let w: Vec<f64> = p
                .iter()
                .map(|&t| {
                    let (pn, _) = jacobi(n, 0.0, 0.0, 2.0 * t - 1.0);
                    1.0 / ((n * (n + 1)) as f64 * pn * pn)
                })
                .collect();
            for k in 0..=(2 * m - 3) as i32 {
                let approx: f64 = p.iter().zip(&w).map(|(t, wi)| wi * t.powi(k)).sum();
                let exact = 1.0 / f64::from(k + 1);
                assert!(((approx - exact) / exact).abs() <= 1e-13, "m = {m}, x^{k}");
            }
        }
    }
}
