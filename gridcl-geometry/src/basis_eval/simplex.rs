//! Dubiner's collapsed-coordinate sets on triangles and tetrahedra, and Dubiner times Legendre
//! on prisms (gridcl-refcell's `dubiner_triangle`, `dubiner_tetrahedron` and prism set),
//! unnormalised. Nothing is divided by a collapsed coordinate.

use cubecl::prelude::*;

use super::{cst, emit, legendre, scaled_jacobi};

/// The index of the first (i, j) pair of row i in the triangle order (i outer, j ≤ p − i
/// inner).
fn triangle_row(p: usize, i: usize) -> usize {
    (0..i).map(|r| p - r + 1).sum()
}

/// The index of the first (i, j, k) triple with this (i, j) in the tetrahedron order.
fn tetrahedron_row(p: usize, i: usize, j: usize) -> usize {
    let planes: usize = (0..i).map(|r| (p - r + 1) * (p - r + 2) / 2).sum();
    planes + (0..j).map(|r| p - i - r + 1).sum::<usize>()
}

/// The triangle set at (x, y): (value, ∂x, ∂y) at 3k, k in the triangle order. Small: at most
/// 45 entries for p = 4.
#[cube]
fn dubiner<F: Float, N: Size>(
    x: Vector<F, N>,
    y: Vector<F, N>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
) -> Array<Vector<F, N>> {
    let two = cst::<F, N>(2i64);
    let mut out = Array::<Vector<F, N>>::new(comptime!(3 * (p + 1) * (p + 2) / 2));
    let s = one - y;
    let a = scaled_jacobi::<F, N>(two * x - s, s, one, zero, p, 0i64, 0i64, true);
    #[unroll]
    for i in 0..p + 1 {
        let b = scaled_jacobi::<F, N>(
            two * y - one,
            one,
            one,
            zero,
            comptime!(p - i),
            comptime!(2 * i as i64 + 1),
            0i64,
            false,
        );
        let ai = a[3 * i];
        let aix = a[3 * i + 1];
        // ∂(2x − s)/∂y = 1 and ∂s/∂y = −1.
        let aiy = aix - a[3 * i + 2];
        #[unroll]
        for j in 0..p - i + 1 {
            let o = comptime!(3 * (triangle_row(p, i) + j));
            let bj = b[3 * j];
            let bjx = b[3 * j + 1];
            out[o] = ai * bj;
            out[o + 1] = two * aix * bj;
            out[o + 2] = aiy * bj + ai * (two * bjx);
        }
    }
    out
}

/// The triangle set, k in the triangle order.
#[cube]
pub(crate) fn triangle<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
) {
    let t = dubiner::<F, N>(xi[0], xi[1], one, zero, p);
    #[unroll]
    for k in 0..(p + 1) * (p + 2) / 2 {
        emit::<F, N>(
            acc,
            a,
            k,
            t[3 * k],
            t[3 * k + 1],
            t[3 * k + 2],
            zero,
            2usize,
            g,
        );
    }
}

/// The tetrahedron set, k with i outer, then j, then k.
#[cube]
pub(crate) fn tetrahedron<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
) {
    let two = cst::<F, N>(2i64);
    let s1 = one - xi[1] - xi[2];
    let s2 = one - xi[2];
    let qa = scaled_jacobi::<F, N>(two * xi[0] - s1, s1, one, zero, p, 0i64, 0i64, true);
    #[unroll]
    for i in 0..p + 1 {
        let qb = scaled_jacobi::<F, N>(
            two * xi[1] - s2,
            s2,
            one,
            zero,
            comptime!(p - i),
            comptime!(2 * i as i64 + 1),
            0i64,
            true,
        );
        let ai = qa[3 * i];
        let aix = qa[3 * i + 1];
        // ∂/∂y and ∂/∂z of Q_i(2x − s1, s1): both move s1 by −1.
        let ayz = aix - qa[3 * i + 2];
        #[unroll]
        for j in 0..p - i + 1 {
            let qc = scaled_jacobi::<F, N>(
                two * xi[2] - one,
                one,
                one,
                zero,
                comptime!(p - i - j),
                comptime!(2 * (i + j) as i64 + 2),
                0i64,
                false,
            );
            let bj = qb[3 * j];
            let bjx = qb[3 * j + 1];
            // ∂/∂z of Q_j(2y − s2, s2): s2 moves by −1.
            let bz = bjx - qb[3 * j + 2];
            let ab = ai * bj;
            let dxab = two * aix * bj;
            let dyab = ayz * bj + ai * (two * bjx);
            let dzab = ayz * bj + ai * bz;
            #[unroll]
            for k in 0..p - i - j + 1 {
                let ck = qc[3 * k];
                let ckx = qc[3 * k + 1];
                let o = comptime!(tetrahedron_row(p, i, j) + k);
                emit::<F, N>(
                    acc,
                    a,
                    o,
                    ab * ck,
                    dxab * ck,
                    dyab * ck,
                    dzab * ck + ab * (two * ckx),
                    3usize,
                    g,
                );
            }
        }
    }
}

/// The prism set, triangle times Legendre in z with z outer.
#[cube]
pub(crate) fn prism<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
) {
    let t = dubiner::<F, N>(xi[0], xi[1], one, zero, p);
    let lz = legendre::<F, N>(xi[2], one, zero, p);
    let nt = comptime!((p + 1) * (p + 2) / 2);
    #[unroll]
    for iz in 0..p + 1 {
        let vz = lz[2 * iz];
        let dz = lz[2 * iz + 1];
        #[unroll]
        for k in 0..nt {
            let v = t[3 * k];
            emit::<F, N>(
                acc,
                a,
                comptime!(iz * nt + k),
                v * vz,
                t[3 * k + 1] * vz,
                t[3 * k + 2] * vz,
                v * dz,
                3usize,
                g,
            );
        }
    }
}
