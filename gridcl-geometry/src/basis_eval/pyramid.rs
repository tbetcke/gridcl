//! The pyramid's rationomial set (gridcl-refcell's `polynomials/pyramid.rs`), unnormalised:
//! Q_i(2x − s, s) Q_j(2y − s, s) / s^min(i,j) · P_k^(2 max(i,j) + 2, 0)(2z − 1), s = 1 − z,
//! with i outer, then j, then k ≤ p − max(i, j).
//!
//! ζ is clamped to at most 1 − 2^−10 by an arithmetic `min` before evaluation (ADR-0039,
//! item 15), so s ≥ 2^−10 and the division by s^min(i,j) needs no branch. Points above the
//! clamp get the values and derivatives at the clamped point.

use cubecl::prelude::*;

use super::{cst, int, scaled_jacobi};

/// The index of the first triple with this (i, j).
fn pyramid_row(p: usize, i: usize, j: usize) -> usize {
    (0..=p)
        .flat_map(|a| (0..=p).map(move |b| (a, b)))
        .take_while(|&(a, b)| (a, b) < (i, j))
        .map(|(a, b)| p - a.max(b) + 1)
        .sum()
}

/// The pyramid set: (value, ∂x, ∂y, ∂z) at 4k.
#[cube]
pub(crate) fn pyramid<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    phi: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
) {
    let two = cst::<F, N>(2i64);
    // 1 − 2^−10 (the f32 nearest to the literal), exact in f32 and f64.
    let z = min(xi[2], Vector::<F, N>::broadcast(F::new(0.999_023_44_f32)));
    let s = one - z;
    let inv = one / s;
    // inv^0, …, inv^(p + 1).
    let mut pw = Array::<Vector<F, N>>::new(comptime!(p + 2));
    pw[0] = one;
    #[unroll]
    for t in 1..p + 2 {
        pw[t] = pw[t - 1] * inv;
    }
    let a = scaled_jacobi::<F, N>(two * xi[0] - s, s, one, zero, p, 0i64, 0i64, true);
    let b = scaled_jacobi::<F, N>(two * xi[1] - s, s, one, zero, p, 0i64, 0i64, true);
    #[unroll]
    for i in 0..p + 1 {
        let ai = a[3 * i];
        let aix = a[3 * i + 1];
        // ∂(2x − s)/∂z = 1 and ∂s/∂z = −1.
        let az = aix - a[3 * i + 2];
        #[unroll]
        for j in 0..p + 1 {
            let m = comptime!(i.max(j));
            let n = comptime!(i.min(j));
            let bj = b[3 * j];
            let bjx = b[3 * j + 1];
            let bz = bjx - b[3 * j + 2];
            // F = Q_i Q_j / s^n and its derivatives; ∂(s^−n)/∂z = n s^−n−1.
            let r = pw[n];
            let f0 = ai * bj * r;
            let f1 = two * aix * bj * r;
            let f2 = two * ai * bjx * r;
            let mut f3 = (az * bj + ai * bz) * r;
            if comptime!(n > 0) {
                f3 +=
                    ai * bj * (int::<F, N>(comptime!(n as i64), one, zero) * pw[comptime!(n + 1)]);
            }
            let c = scaled_jacobi::<F, N>(
                two * z - one,
                one,
                one,
                zero,
                comptime!(p - m),
                comptime!(2 * m as i64 + 2),
                0i64,
                false,
            );
            #[unroll]
            for k in 0..p - m + 1 {
                let o = comptime!(4 * (pyramid_row(p, i, j) + k));
                let ck = c[3 * k];
                let ckx = c[3 * k + 1];
                phi[o] = f0 * ck;
                phi[o + 1] = f1 * ck;
                phi[o + 2] = f2 * ck;
                phi[o + 3] = f3 * ck + f0 * (two * ckx);
            }
        }
    }
}
