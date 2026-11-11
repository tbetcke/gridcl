//! The basis evaluator of the `loops` strategy (docs/plans/M9h.md, T6b; ADR-0041): the
//! orthogonal sets of [`super::expansion`] with the degree a runtime argument. Loops over basis
//! functions are runtime loops, recurrence coefficients come from runtime integers, and 1D
//! values live in small local arrays of a comptime capacity ([`loops_capacity`]), so the
//! operation count of a kernel does not grow with the degree (C18, C20).
//!
//! Every recurrence runs as a rolling pair (Q_m, Q_{m+1}) advanced in the loop over its index,
//! and every basis function is added to the sums as it is produced, in the order of
//! [`super::expansion`], with the same products. Nothing uses a float constant 0 or 1, and every
//! recurrence constant has a runtime operand (C16): the coefficients of a = 0 that are 0 or 1
//! are the runtime `zero` and `one`, as in [`super::expansion`], and the others are products of
//! runtime loop indices. So the results are the same at every vector size on the CPU runtime.
//! The coefficients a_k are read from a slice of vectors, a(k, c) at
//! (k g + c) · `a_stride` + `a_at`.

use cubecl::prelude::*;
use gridcl_dims::CellType;

use super::cst;

/// The comptime capacity of the 1D arrays of [`expansion_loops`] for `degree`: at least
/// degree + 2 (the pyramid's powers of 1/s), a power of two and at least 4, so few kernels
/// serve every degree.
pub(crate) fn loops_capacity(degree: u32) -> usize {
    (degree as usize + 2).next_power_of_two().max(4)
}

/// Lane-uniform coefficients for [`expansion_loops`]: each value repeated `VECTOR_PAD` times, so
/// the vector at index i · (`VECTOR_PAD` / N) of the buffer is value i broadcast at every vector
/// size N (`a_stride` = `VECTOR_PAD` / N).
#[cfg(any(test, feature = "bench"))]
pub(crate) fn broadcast_rows(values: &[f64]) -> Vec<f64> {
    values
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, gridcl_device::VECTOR_PAD))
        .collect()
}

/// The rolling state (Q_0, Q_1) of gridcl-refcell's `scaled_jacobi` with (a, b) = (0, 0) at
/// (x, s): (value, ∂x, ∂s) of Q_m at 0..3 and of Q_{m+1} at 3..6, with the operations of
/// [`super::scaled_jacobi`] (a − b and ∂s Q_1 the runtime zero, ∂x Q_1 the runtime one).
#[cube]
fn start_a0<F: Float, N: Size>(
    x: Vector<F, N>,
    s: Vector<F, N>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
) -> Array<Vector<F, N>> {
    let half = Vector::<F, N>::broadcast(F::new(0.5f32));
    let mut q = Array::<Vector<F, N>>::new(6usize);
    q[0] = one;
    q[1] = zero;
    q[2] = zero;
    q[3] = half * (cst::<F, N>(2i64) * x + zero * s);
    q[4] = one;
    q[5] = zero;
    q
}

/// The rolling state (Q_0, Q_1) with b = 0 and a runtime a ≥ 1.
#[cube]
fn start<F: Float, N: Size>(
    x: Vector<F, N>,
    s: Vector<F, N>,
    a: u32,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] with_ds: bool,
) -> Array<Vector<F, N>> {
    let half = Vector::<F, N>::broadcast(F::new(0.5f32));
    let fa = Vector::<F, N>::broadcast(F::cast_from(a));
    let ab2 = Vector::<F, N>::broadcast(F::cast_from(a + 2));
    let mut q = Array::<Vector<F, N>>::new(6usize);
    q[0] = one;
    q[1] = zero;
    q[2] = zero;
    q[3] = half * (ab2 * x + fa * s);
    q[4] = half * ab2;
    q[5] = zero;
    if comptime!(with_ds) {
        q[5] = half * fa;
    }
    q
}

/// Advances the state from (Q_{m−1}, Q_m) to (Q_m, Q_{m+1}), m ≥ 1, by the recurrence of
/// [`super::scaled_jacobi`] with b = 0; `asq` is c3 = a² as a runtime value (the runtime zero
/// where a = 0). ∂s is `zero` unless `with_ds`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn step<F: Float, N: Size>(
    q: &mut Array<Vector<F, N>>,
    m: u32,
    a: u32,
    asq: Vector<F, N>,
    x: Vector<F, N>,
    s: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] with_ds: bool,
) {
    let c0 = Vector::<F, N>::broadcast(F::cast_from(2 * (m + 1) * (m + a + 1) * (2 * m + a)));
    let c1 = Vector::<F, N>::broadcast(F::cast_from(2 * m + a + 1));
    let c2 = Vector::<F, N>::broadcast(F::cast_from((2 * m + a + 2) * (2 * m + a)));
    let c4 = Vector::<F, N>::broadcast(F::cast_from(2 * (m + a) * m * (2 * m + a + 2)));
    let p1 = q[3];
    let p0 = q[0];
    let p1x = q[4];
    let p0x = q[1];
    let lin = c2 * x + asq * s;
    let ss = s * s;
    let v = (c1 * lin * p1 - c4 * ss * p0) / c0;
    let dx = (c1 * (c2 * p1 + lin * p1x) - c4 * ss * p0x) / c0;
    let mut ds = zero;
    if comptime!(with_ds) {
        let p1s = q[5];
        let p0s = q[2];
        let two = cst::<F, N>(2i64);
        ds = (c1 * (asq * p1 + lin * p1s) - c4 * (two * s * p0 + ss * p0s)) / c0;
    }
    q[0] = q[3];
    q[1] = q[4];
    q[2] = q[5];
    q[3] = v;
    q[4] = dx;
    q[5] = ds;
}

/// a² of a runtime a ≥ 1 as a vector.
#[cube]
fn square<F: Float, N: Size>(a: u32) -> Vector<F, N> {
    Vector::<F, N>::broadcast(F::cast_from(a * a))
}

/// [`super::emit`] with the coefficients in a slice: a(k, c) at (k g + c) · `a_stride` + `a_at`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn emit_at<F: Float, N: Size>(
    acc: &mut Array<Vector<F, N>>,
    a: &[Vector<F, N>],
    a_at: usize,
    a_stride: usize,
    k: usize,
    v0: Vector<F, N>,
    v1: Vector<F, N>,
    v2: Vector<F, N>,
    v3: Vector<F, N>,
    #[comptime] td: usize,
    #[comptime] g: usize,
) {
    #[unroll]
    for c in 0..g {
        let w = a[(k * g + c) * a_stride + a_at];
        let o = c * (td + 1);
        acc[o] += w * v0;
        acc[o + 1] += w * v1;
        if comptime!(td >= 2) {
            acc[o + 2] += w * v2;
        }
        if comptime!(td >= 3) {
            acc[o + 3] += w * v3;
        }
    }
}

/// P_i(2t − 1) and its derivative along t at 2i and 2i + 1 of `out`, i ≤ p
/// ([`super::legendre`]).
#[cube]
fn legendre_into<F: Float, N: Size>(
    out: &mut Array<Vector<F, N>>,
    t: Vector<F, N>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    p: usize,
) {
    let two = cst::<F, N>(2i64);
    let x = two * t - one;
    let mut q = start_a0::<F, N>(x, one, one, zero);
    for i in 0..p + 1 {
        out[2 * i] = q[0];
        out[2 * i + 1] = two * q[1];
        step::<F, N>(&mut q, (i + 1) as u32, 0u32, zero, x, one, zero, false);
    }
}

/// The triangle set at (x, y) in the triangle order (i outer, j ≤ p − i inner), added to the
/// sums from basis function `k0` on: on triangles as is, on prisms (`prism`) times the Legendre
/// value `vz` and derivative `dz` in z.
#[cube]
#[allow(clippy::too_many_arguments)]
fn triangle_set<F: Float, N: Size>(
    x: Vector<F, N>,
    y: Vector<F, N>,
    a: &[Vector<F, N>],
    a_at: usize,
    a_stride: usize,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    p: usize,
    k0: usize,
    vz: Vector<F, N>,
    dz: Vector<F, N>,
    #[comptime] prism: bool,
    #[comptime] g: usize,
) {
    let two = cst::<F, N>(2i64);
    let s = one - y;
    let ax = two * x - s;
    let by = two * y - one;
    let mut qa = start_a0::<F, N>(ax, s, one, zero);
    let mut k = k0;
    for i in 0..p + 1 {
        let ai = qa[0];
        let aix = qa[1];
        // ∂(2x − s)/∂y = 1 and ∂s/∂y = −1.
        let aiy = aix - qa[2];
        let alpha = (2 * i + 1) as u32;
        let asq = square::<F, N>(alpha);
        let mut qb = start::<F, N>(by, one, alpha, one, zero, false);
        for j in 0..p - i + 1 {
            let bj = qb[0];
            let bjx = qb[1];
            let t0 = ai * bj;
            let t1 = two * aix * bj;
            let t2 = aiy * bj + ai * (two * bjx);
            if comptime!(prism) {
                emit_at::<F, N>(
                    acc,
                    a,
                    a_at,
                    a_stride,
                    k,
                    t0 * vz,
                    t1 * vz,
                    t2 * vz,
                    t0 * dz,
                    3usize,
                    g,
                );
            } else {
                emit_at::<F, N>(acc, a, a_at, a_stride, k, t0, t1, t2, zero, 2usize, g);
            }
            k += 1;
            step::<F, N>(&mut qb, (j + 1) as u32, alpha, asq, by, one, zero, false);
        }
        step::<F, N>(&mut qa, (i + 1) as u32, 0u32, zero, ax, s, zero, true);
    }
}

/// [`super::expansion`] with the degree at run time: Σ_k a_k φ̃_k(ξ) and its first derivatives
/// for `g` columns, entry c (1 + tdim) the sum of column c and c (1 + tdim) + 1 + d its
/// derivative along ξ_d. `cap` is [`loops_capacity`] of a degree at least `degree`; the
/// pyramid clamps ζ to at most 1 − 2^−10 first (ADR-0039, item 15).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expansion_loops<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &[Vector<F, N>],
    a_at: usize,
    a_stride: usize,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    degree: usize,
    #[comptime] cell: CellType,
    #[comptime] g: usize,
    #[comptime] cap: usize,
) -> Array<Vector<F, N>> {
    let td = comptime!(gridcl_refcell::tdim(cell));
    let p = degree;
    let two = cst::<F, N>(2i64);
    let mut acc = Array::<Vector<F, N>>::new(comptime!(g * (td + 1)));
    #[unroll]
    for c in 0..g * (td + 1) {
        acc[c] = zero;
    }
    if comptime!(cell == CellType::Interval) {
        let mut lx = Array::<Vector<F, N>>::new(comptime!(2 * cap));
        legendre_into::<F, N>(&mut lx, xi[0], one, zero, p);
        for i in 0..p + 1 {
            emit_at::<F, N>(
                &mut acc,
                a,
                a_at,
                a_stride,
                i,
                lx[2 * i],
                lx[2 * i + 1],
                zero,
                zero,
                1usize,
                g,
            );
        }
    } else if comptime!(cell == CellType::Quadrilateral || cell == CellType::Hexahedron) {
        // x from a local array (innermost), y and z as rolling recurrences.
        let mut lx = Array::<Vector<F, N>>::new(comptime!(2 * cap));
        legendre_into::<F, N>(&mut lx, xi[0], one, zero, p);
        let ty = two * xi[1] - one;
        let mut k = 0usize;
        if comptime!(cell == CellType::Quadrilateral) {
            let mut qy = start_a0::<F, N>(ty, one, one, zero);
            for iy in 0..p + 1 {
                let vy = qy[0];
                let dy = two * qy[1];
                for ix in 0..p + 1 {
                    let vx = lx[2 * ix];
                    let dx = lx[2 * ix + 1];
                    emit_at::<F, N>(
                        &mut acc,
                        a,
                        a_at,
                        a_stride,
                        k,
                        vx * vy,
                        dx * vy,
                        vx * dy,
                        zero,
                        2usize,
                        g,
                    );
                    k += 1;
                }
                step::<F, N>(&mut qy, (iy + 1) as u32, 0u32, zero, ty, one, zero, false);
            }
        } else {
            let tz = two * xi[2] - one;
            let mut qz = start_a0::<F, N>(tz, one, one, zero);
            for iz in 0..p + 1 {
                let vz = qz[0];
                let dz = two * qz[1];
                let mut qy = start_a0::<F, N>(ty, one, one, zero);
                for iy in 0..p + 1 {
                    let vy = qy[0];
                    let dy = two * qy[1];
                    let vyz = vy * vz;
                    let dyz = dy * vz;
                    let ydz = vy * dz;
                    for ix in 0..p + 1 {
                        let vx = lx[2 * ix];
                        let dx = lx[2 * ix + 1];
                        emit_at::<F, N>(
                            &mut acc,
                            a,
                            a_at,
                            a_stride,
                            k,
                            vx * vyz,
                            dx * vyz,
                            vx * dyz,
                            vx * ydz,
                            3usize,
                            g,
                        );
                        k += 1;
                    }
                    step::<F, N>(&mut qy, (iy + 1) as u32, 0u32, zero, ty, one, zero, false);
                }
                step::<F, N>(&mut qz, (iz + 1) as u32, 0u32, zero, tz, one, zero, false);
            }
        }
    } else if comptime!(cell == CellType::Triangle) {
        triangle_set::<F, N>(
            xi[0], xi[1], a, a_at, a_stride, &mut acc, one, zero, p, 0usize, zero, zero, false, g,
        );
    } else if comptime!(cell == CellType::Prism) {
        // Triangle times Legendre in z, z outer: the triangle set is recomputed per z layer.
        let nt = (p + 1) * (p + 2) / 2;
        let tz = two * xi[2] - one;
        let mut qz = start_a0::<F, N>(tz, one, one, zero);
        for iz in 0..p + 1 {
            let vz = qz[0];
            let dz = two * qz[1];
            triangle_set::<F, N>(
                xi[0],
                xi[1],
                a,
                a_at,
                a_stride,
                &mut acc,
                one,
                zero,
                p,
                iz * nt,
                vz,
                dz,
                true,
                g,
            );
            step::<F, N>(&mut qz, (iz + 1) as u32, 0u32, zero, tz, one, zero, false);
        }
    } else if comptime!(cell == CellType::Tetrahedron) {
        let s1 = one - xi[1] - xi[2];
        let s2 = one - xi[2];
        let ax = two * xi[0] - s1;
        let by = two * xi[1] - s2;
        let cz = two * xi[2] - one;
        let mut qa = start_a0::<F, N>(ax, s1, one, zero);
        let mut o = 0usize;
        for i in 0..p + 1 {
            let ai = qa[0];
            let aix = qa[1];
            // ∂/∂y and ∂/∂z of Q_i(2x − s1, s1): both move s1 by −1.
            let ayz = aix - qa[2];
            let beta = (2 * i + 1) as u32;
            let bsq = square::<F, N>(beta);
            let mut qb = start::<F, N>(by, s2, beta, one, zero, true);
            for j in 0..p - i + 1 {
                let bj = qb[0];
                let bjx = qb[1];
                // ∂/∂z of Q_j(2y − s2, s2): s2 moves by −1.
                let bz = bjx - qb[2];
                let ab = ai * bj;
                let dxab = two * aix * bj;
                let dyab = ayz * bj + ai * (two * bjx);
                let dzab = ayz * bj + ai * bz;
                let gamma = (2 * (i + j) + 2) as u32;
                let gsq = square::<F, N>(gamma);
                let mut qc = start::<F, N>(cz, one, gamma, one, zero, false);
                for k in 0..p - i - j + 1 {
                    let ck = qc[0];
                    let ckx = qc[1];
                    emit_at::<F, N>(
                        &mut acc,
                        a,
                        a_at,
                        a_stride,
                        o,
                        ab * ck,
                        dxab * ck,
                        dyab * ck,
                        dzab * ck + ab * (two * ckx),
                        3usize,
                        g,
                    );
                    o += 1;
                    step::<F, N>(&mut qc, (k + 1) as u32, gamma, gsq, cz, one, zero, false);
                }
                step::<F, N>(&mut qb, (j + 1) as u32, beta, bsq, by, s2, zero, true);
            }
            step::<F, N>(&mut qa, (i + 1) as u32, 0u32, zero, ax, s1, zero, true);
        }
    } else if comptime!(cell == CellType::Pyramid) {
        // 1 − 2^−10 (the f32 nearest to the literal), exact in f32 and f64 (`pyramid.rs`).
        let z = min(xi[2], Vector::<F, N>::broadcast(F::new(0.999_023_44_f32)));
        let s = one - z;
        let inv = one / s;
        // inv^0, …, inv^(p + 1).
        let mut pw = Array::<Vector<F, N>>::new(cap);
        pw[0] = one;
        for t in 1..p + 2 {
            pw[t] = pw[t - 1] * inv;
        }
        let ax = two * xi[0] - s;
        let by = two * xi[1] - s;
        let cz = two * z - one;
        let mut qa = start_a0::<F, N>(ax, s, one, zero);
        let mut o = 0usize;
        for i in 0..p + 1 {
            let ai = qa[0];
            let aix = qa[1];
            // ∂(2x − s)/∂z = 1 and ∂s/∂z = −1.
            let az = aix - qa[2];
            let mut qb = start_a0::<F, N>(by, s, one, zero);
            for j in 0..p + 1 {
                let m = max(i, j);
                let n = min(i, j);
                let bj = qb[0];
                let bjx = qb[1];
                let bz = bjx - qb[2];
                // F = Q_i Q_j / s^n and its derivatives; ∂(s^−n)/∂z = n s^−n−1.
                let r = pw[n];
                let f0 = ai * bj * r;
                let f1 = two * aix * bj * r;
                let f2 = two * ai * bjx * r;
                let mut f3 = (az * bj + ai * bz) * r;
                if n > 0 {
                    let fnn = Vector::<F, N>::broadcast(F::cast_from(n as u32));
                    f3 += ai * bj * (fnn * pw[n + 1]);
                }
                let gamma = (2 * m + 2) as u32;
                let gsq = square::<F, N>(gamma);
                let mut qc = start::<F, N>(cz, one, gamma, one, zero, false);
                for k in 0..p - m + 1 {
                    let ck = qc[0];
                    let ckx = qc[1];
                    emit_at::<F, N>(
                        &mut acc,
                        a,
                        a_at,
                        a_stride,
                        o,
                        f0 * ck,
                        f1 * ck,
                        f2 * ck,
                        f3 * ck + f0 * (two * ckx),
                        3usize,
                        g,
                    );
                    o += 1;
                    step::<F, N>(&mut qc, (k + 1) as u32, gamma, gsq, cz, one, zero, false);
                }
                step::<F, N>(&mut qb, (j + 1) as u32, 0u32, zero, by, s, zero, true);
            }
            step::<F, N>(&mut qa, (i + 1) as u32, 0u32, zero, ax, s, zero, true);
        }
    } else {
        // The point cell: one basis function, the constant 1.
        #[unroll]
        for c in 0..g {
            acc[c] = a[c * a_stride + a_at];
        }
    }
    acc
}
