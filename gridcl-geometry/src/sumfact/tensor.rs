//! The sum-factorised factor kernel on quadrilaterals and hexahedra (docs/plans/M9h.md, T7a),
//! and on prisms through [`block_prism`] (T7c).
//!
//! Per cell and tensor block of the point set, the kernel contracts the E-vector with the 1D
//! tables of the outermost level first: on a hexahedron, for each point i_2 of level 2 it forms
//! A_2(g, a, b) = Σ_c φ_2(i_2, c) x(π(a, b, c), g) (and the same with φ_2'), for each point i_1
//! of level 1 it forms A_1(g, a) = Σ_b φ_1(i_1, b) A_2(g, a, b) (and the products with the
//! derivatives), and for each point i_0 it sums over a. That costs about 3 (p + 1)^4 products
//! per component at p + 1 points per direction, against (p + 1)^6 of the direct sum, and keeps
//! 2 G (p + 1)² + 3 G (p + 1) intermediate vectors per lane (local arrays of a comptime size).
//!
//! The node count per direction n is comptime (the array sizes); the point counts, strides and
//! table offsets are read from the table at run time, and no loop over nodes or points is
//! unrolled. Every sum starts from the product of its first term and runs in ascending order,
//! with the same operations in every lane, so the results are bitwise equal across vector sizes
//! on the CPU runtime.

use cubecl::prelude::*;

use super::LEVEL_LEN;
use super::prism::block_prism;
use crate::GeomLayout;
use crate::layout::{Factor, store_from_jacobian};

/// The comptime layout of the sum-factorised kernels for `l` and an element of `n` nodes per
/// direction: `nodes_per_cell` = n, and the point count 0, since the kernels take it at run
/// time.
pub(crate) fn sumfact_layout(l: GeomLayout, n: usize) -> GeomLayout {
    GeomLayout {
        nodes_per_cell: n,
        num_points: 0,
        ..l
    }
}

/// The integer at `tab[i]`.
#[cube]
pub(super) fn int<F: Float>(tab: &[F], i: usize) -> usize {
    u32::cast_from(tab[i]) as usize
}

/// `x` in every lane.
#[cube]
pub(super) fn bc<F: Float, N: Size>(x: F) -> Vector<F, N> {
    Vector::<F, N>::broadcast(x)
}

/// The tensor block whose header starts at `tab[h]` (TD = 2): the factor `f` at its points.
/// `xo` and `xs` are the lanes' E-vector offset and row stride in vectors, `base` the index of
/// the tile's first point in `out`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn block_2d<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    tab: &[F],
    extra: &[F],
    flags: &[u32],
    out: &mut [Vector<F, N>],
    h: usize,
    perm: usize,
    xo: usize,
    xs: usize,
    base: usize,
    vpt: usize,
    lane: usize,
    v: usize,
    first_cell: u32,
    last_cell: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
) {
    let n = comptime!(l.nodes_per_cell);
    let gd = comptime!(l.gdim);
    let len = comptime!(f.len(&l));
    let o = int::<F>(tab, h);
    let (m0, s0, t0, x0, d0) = (
        int::<F>(tab, h + 1),
        int::<F>(tab, h + 2),
        int::<F>(tab, h + 3),
        int::<F>(tab, h + 4),
        int::<F>(tab, h + 5),
    );
    let (m1, s1, t1, x1, d1) = (
        int::<F>(tab, h + 6),
        int::<F>(tab, h + 7),
        int::<F>(tab, h + 8),
        int::<F>(tab, h + 9),
        int::<F>(tab, h + 10),
    );
    // A_1(g, a) at g · n + a, with φ_1 (v) and φ_1' (d).
    let mut a1v = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut a1d = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut j = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    for i1 in 0..m1 {
        let r1 = t1 + i1 * n;
        let r1d = r1 + m1 * n;
        for a in 0..n {
            let lex = a * x0;
            let k = int::<F>(tab, perm + lex);
            let w = bc::<F, N>(tab[r1]);
            let wd = bc::<F, N>(tab[r1d]);
            #[unroll]
            for g in 0..gd {
                let x = xe[(k * gd + g) * xs + xo];
                a1v[g * n + a] = x * w;
                if comptime!(!f.is_points()) {
                    a1d[g * n + a] = x * wd;
                }
            }
            for b in 1..n {
                let k = int::<F>(tab, perm + lex + b * x1);
                let w = bc::<F, N>(tab[r1 + b]);
                let wd = bc::<F, N>(tab[r1d + b]);
                #[unroll]
                for g in 0..gd {
                    let x = xe[(k * gd + g) * xs + xo];
                    a1v[g * n + a] += x * w;
                    if comptime!(!f.is_points()) {
                        a1d[g * n + a] += x * wd;
                    }
                }
            }
        }
        for i0 in 0..m0 {
            let r0 = t0 + i0 * n;
            let r0d = r0 + m0 * n;
            let q = o + i0 * s0 + i1 * s1;
            let at = (base + q) * len;
            if comptime!(f.is_points()) {
                #[unroll]
                for g in 0..gd {
                    let mut acc = a1v[g * n] * bc::<F, N>(tab[r0]);
                    for a in 1..n {
                        acc += a1v[g * n + a] * bc::<F, N>(tab[r0 + a]);
                    }
                    out[(at + g) * vpt + lane] = acc + bc::<F, N>(extra[g]);
                }
            } else {
                #[unroll]
                for g in 0..gd {
                    let mut j0 = a1v[g * n] * bc::<F, N>(tab[r0d]);
                    let mut j1 = a1d[g * n] * bc::<F, N>(tab[r0]);
                    for a in 1..n {
                        j0 += a1v[g * n + a] * bc::<F, N>(tab[r0d + a]);
                        j1 += a1d[g * n + a] * bc::<F, N>(tab[r0 + a]);
                    }
                    j[d0 * gd + g] = j0;
                    j[d1 * gd + g] = j1;
                }
                store_from_jacobian::<F, N>(
                    &j, extra, flags, out, at, vpt, lane, v, q, first_cell, last_cell, l, f,
                );
            }
        }
    }
}

/// [`block_2d`] for TD = 3.
#[cube]
#[allow(clippy::too_many_arguments)]
fn block_3d<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    tab: &[F],
    extra: &[F],
    flags: &[u32],
    out: &mut [Vector<F, N>],
    h: usize,
    perm: usize,
    xo: usize,
    xs: usize,
    base: usize,
    vpt: usize,
    lane: usize,
    v: usize,
    first_cell: u32,
    last_cell: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
) {
    let n = comptime!(l.nodes_per_cell);
    let gd = comptime!(l.gdim);
    let len = comptime!(f.len(&l));
    let o = int::<F>(tab, h);
    let (m0, s0, t0, x0, d0) = (
        int::<F>(tab, h + 1),
        int::<F>(tab, h + 2),
        int::<F>(tab, h + 3),
        int::<F>(tab, h + 4),
        int::<F>(tab, h + 5),
    );
    let (m1, s1, t1, x1, d1) = (
        int::<F>(tab, h + 6),
        int::<F>(tab, h + 7),
        int::<F>(tab, h + 8),
        int::<F>(tab, h + 9),
        int::<F>(tab, h + 10),
    );
    let (m2, s2, t2, x2, d2) = (
        int::<F>(tab, h + 11),
        int::<F>(tab, h + 12),
        int::<F>(tab, h + 13),
        int::<F>(tab, h + 14),
        int::<F>(tab, h + 15),
    );
    // A_2(g, a, b) at (g · n + b) · n + a, with φ_2 (v) and φ_2' (d).
    let mut a2v =
        Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell * l.nodes_per_cell));
    let mut a2d =
        Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell * l.nodes_per_cell));
    // A_1(g, a) at g · n + a: values in levels 1 and 2 (vv), the level-1 derivative (vd), the
    // level-2 derivative (dv).
    let mut a1vv = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut a1vd = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut a1dv = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut j = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    for i2 in 0..m2 {
        let r2 = t2 + i2 * n;
        let r2d = r2 + m2 * n;
        for b in 0..n {
            for a in 0..n {
                let lex = a * x0 + b * x1;
                let k = int::<F>(tab, perm + lex);
                let w = bc::<F, N>(tab[r2]);
                let wd = bc::<F, N>(tab[r2d]);
                let at = b * n + a;
                #[unroll]
                for g in 0..gd {
                    let x = xe[(k * gd + g) * xs + xo];
                    a2v[g * n * n + at] = x * w;
                    if comptime!(!f.is_points()) {
                        a2d[g * n * n + at] = x * wd;
                    }
                }
                for c in 1..n {
                    let k = int::<F>(tab, perm + lex + c * x2);
                    let w = bc::<F, N>(tab[r2 + c]);
                    let wd = bc::<F, N>(tab[r2d + c]);
                    #[unroll]
                    for g in 0..gd {
                        let x = xe[(k * gd + g) * xs + xo];
                        a2v[g * n * n + at] += x * w;
                        if comptime!(!f.is_points()) {
                            a2d[g * n * n + at] += x * wd;
                        }
                    }
                }
            }
        }
        for i1 in 0..m1 {
            let r1 = t1 + i1 * n;
            let r1d = r1 + m1 * n;
            for a in 0..n {
                let w = bc::<F, N>(tab[r1]);
                let wd = bc::<F, N>(tab[r1d]);
                #[unroll]
                for g in 0..gd {
                    let y = a2v[g * n * n + a];
                    a1vv[g * n + a] = y * w;
                    if comptime!(!f.is_points()) {
                        a1vd[g * n + a] = y * wd;
                        a1dv[g * n + a] = a2d[g * n * n + a] * w;
                    }
                }
                for b in 1..n {
                    let w = bc::<F, N>(tab[r1 + b]);
                    let wd = bc::<F, N>(tab[r1d + b]);
                    #[unroll]
                    for g in 0..gd {
                        let y = a2v[g * n * n + b * n + a];
                        a1vv[g * n + a] += y * w;
                        if comptime!(!f.is_points()) {
                            a1vd[g * n + a] += y * wd;
                            a1dv[g * n + a] += a2d[g * n * n + b * n + a] * w;
                        }
                    }
                }
            }
            for i0 in 0..m0 {
                let r0 = t0 + i0 * n;
                let r0d = r0 + m0 * n;
                let q = o + i0 * s0 + i1 * s1 + i2 * s2;
                let at = (base + q) * len;
                if comptime!(f.is_points()) {
                    #[unroll]
                    for g in 0..gd {
                        let mut acc = a1vv[g * n] * bc::<F, N>(tab[r0]);
                        for a in 1..n {
                            acc += a1vv[g * n + a] * bc::<F, N>(tab[r0 + a]);
                        }
                        out[(at + g) * vpt + lane] = acc + bc::<F, N>(extra[g]);
                    }
                } else {
                    #[unroll]
                    for g in 0..gd {
                        let mut j0 = a1vv[g * n] * bc::<F, N>(tab[r0d]);
                        let mut j1 = a1vd[g * n] * bc::<F, N>(tab[r0]);
                        let mut j2 = a1dv[g * n] * bc::<F, N>(tab[r0]);
                        for a in 1..n {
                            j0 += a1vv[g * n + a] * bc::<F, N>(tab[r0d + a]);
                            j1 += a1vd[g * n + a] * bc::<F, N>(tab[r0 + a]);
                            j2 += a1dv[g * n + a] * bc::<F, N>(tab[r0 + a]);
                        }
                        j[d0 * gd + g] = j0;
                        j[d1 * gd + g] = j1;
                        j[d2 * gd + g] = j2;
                    }
                    store_from_jacobian::<F, N>(
                        &j, extra, flags, out, at, vpt, lane, v, q, first_cell, last_cell, l, f,
                    );
                }
            }
        }
    }
}

/// `layout::factor_kernel` by sum factorisation; `l` from [`sumfact_layout`], `tab` the table
/// of [`TensorFactors::table`](super::TensorFactors::table) narrowed to `F`, `num_points` the
/// points per cell, `prism` whether the cell is a prism ([`block_prism`]). The same buffers
/// (five) and arguments otherwise.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_sumfact_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    tab: &[F],
    extra: &[F],
    flags: &[u32],
    out: &mut [Vector<F, N>],
    x_offset: u32,
    x_stride: u32,
    out_first_tile: u32,
    first_cell: u32,
    last_cell: u32,
    num_vectors: u32,
    items_per_unit: u32,
    num_points: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
    #[comptime] prism: bool,
) {
    let levels = comptime!(if prism { 2 } else { l.tdim });
    let vpt = comptime!(l.tile) / N::value();
    let num_blocks = int::<F>(tab, 0);
    let perm = int::<F>(tab, 1);
    let xs = x_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt (C3).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            let base = (out_first_tile as usize + local_tile) * num_points as usize;
            let xo = x_offset as usize + v;
            for blk in 0..num_blocks {
                let h = 2 + blk * comptime!(1 + LEVEL_LEN * levels);
                if comptime!(prism) {
                    block_prism::<F, N>(
                        xe, tab, extra, flags, out, h, perm, xo, xs, base, vpt, lane, v,
                        first_cell, last_cell, l, f,
                    );
                } else if comptime!(l.tdim == 2) {
                    block_2d::<F, N>(
                        xe, tab, extra, flags, out, h, perm, xo, xs, base, vpt, lane, v,
                        first_cell, last_cell, l, f,
                    );
                } else {
                    block_3d::<F, N>(
                        xe, tab, extra, flags, out, h, perm, xo, xs, base, vpt, lane, v,
                        first_cell, last_cell, l, f,
                    );
                }
            }
        }
    }
}
