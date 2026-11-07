//! Legendre products on intervals, quadrilaterals and hexahedra (gridcl-refcell's `legendre`
//! and its tensor products), unnormalised; x fastest.

use cubecl::prelude::*;

use super::legendre;

/// φ̃_i = P_i(2x − 1): entry 2i the value, 2i + 1 the derivative.
#[cube]
pub(crate) fn interval<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    phi: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
) {
    let l = legendre::<F, N>(xi[0], one, zero, p);
    #[unroll]
    for i in 0..2 * (p + 1) {
        phi[i] = l[i];
    }
}

/// φ̃_k = P_ix P_iy with k = iy (p + 1) + ix: (value, ∂x, ∂y) at 3k.
#[cube]
pub(crate) fn quadrilateral<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    phi: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
) {
    let lx = legendre::<F, N>(xi[0], one, zero, p);
    let ly = legendre::<F, N>(xi[1], one, zero, p);
    #[unroll]
    for iy in 0..p + 1 {
        #[unroll]
        for ix in 0..p + 1 {
            let o = 3 * (iy * (p + 1) + ix);
            let vx = lx[2 * ix];
            let dx = lx[2 * ix + 1];
            let vy = ly[2 * iy];
            let dy = ly[2 * iy + 1];
            phi[o] = vx * vy;
            phi[o + 1] = dx * vy;
            phi[o + 2] = vx * dy;
        }
    }
}

/// φ̃_k = P_ix P_iy P_iz with k = (iz (p + 1) + iy) (p + 1) + ix: (value, ∂x, ∂y, ∂z) at 4k.
#[cube]
pub(crate) fn hexahedron<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    phi: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
) {
    let lx = legendre::<F, N>(xi[0], one, zero, p);
    let ly = legendre::<F, N>(xi[1], one, zero, p);
    let lz = legendre::<F, N>(xi[2], one, zero, p);
    #[unroll]
    for iz in 0..p + 1 {
        #[unroll]
        for iy in 0..p + 1 {
            let vy = ly[2 * iy];
            let dy = ly[2 * iy + 1];
            let vz = lz[2 * iz];
            let dz = lz[2 * iz + 1];
            let vyz = vy * vz;
            #[unroll]
            for ix in 0..p + 1 {
                let o = 4 * ((iz * (p + 1) + iy) * (p + 1) + ix);
                let vx = lx[2 * ix];
                let dx = lx[2 * ix + 1];
                phi[o] = vx * vyz;
                phi[o + 1] = dx * vyz;
                phi[o + 2] = vx * dy * vz;
                phi[o + 3] = vx * vy * dz;
            }
        }
    }
}
