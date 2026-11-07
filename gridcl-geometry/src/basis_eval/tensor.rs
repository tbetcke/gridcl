//! Legendre products on intervals, quadrilaterals and hexahedra (gridcl-refcell's `legendre`
//! and its tensor products), unnormalised; x fastest.

use cubecl::prelude::*;

use super::{emit, legendre};

/// φ̃_i = P_i(2x − 1), k = i.
#[cube]
pub(crate) fn interval<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
) {
    let l = legendre::<F, N>(xi[0], one, zero, p);
    #[unroll]
    for i in 0..p + 1 {
        emit::<F, N>(acc, a, i, l[2 * i], l[2 * i + 1], zero, zero, 1usize, g);
    }
}

/// φ̃_k = P_ix P_iy with k = iy (p + 1) + ix.
#[cube]
pub(crate) fn quadrilateral<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
) {
    let lx = legendre::<F, N>(xi[0], one, zero, p);
    let ly = legendre::<F, N>(xi[1], one, zero, p);
    #[unroll]
    for iy in 0..p + 1 {
        let vy = ly[2 * iy];
        let dy = ly[2 * iy + 1];
        #[unroll]
        for ix in 0..p + 1 {
            let vx = lx[2 * ix];
            let dx = lx[2 * ix + 1];
            let k = comptime!(iy * (p + 1) + ix);
            emit::<F, N>(acc, a, k, vx * vy, dx * vy, vx * dy, zero, 2usize, g);
        }
    }
}

/// φ̃_k = P_ix P_iy P_iz with k = (iz (p + 1) + iy) (p + 1) + ix.
#[cube]
pub(crate) fn hexahedron<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    acc: &mut Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] p: usize,
    #[comptime] g: usize,
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
            let dyz = dy * vz;
            let ydz = vy * dz;
            #[unroll]
            for ix in 0..p + 1 {
                let vx = lx[2 * ix];
                let dx = lx[2 * ix + 1];
                let k = comptime!((iz * (p + 1) + iy) * (p + 1) + ix);
                emit::<F, N>(acc, a, k, vx * vyz, dx * vyz, vx * dyz, vx * ydz, 3usize, g);
            }
        }
    }
}
