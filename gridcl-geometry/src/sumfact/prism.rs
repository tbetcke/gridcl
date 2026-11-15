//! Sum factorisation on prisms (docs/plans/M9h.md, T7c).
//!
//! The prism's Lagrange basis is the product of the triangle's (nt = n (n + 1) / 2 nodes) and
//! the interval's (n = p + 1 nodes) Lagrange bases, and its tensor point sets are products of
//! triangle points (level 0) and points in z (level 1). Per cell and tensor block, for each
//! triangle point i the kernel forms B(g, c) = Σ_t ψ_t(i) x(π(t, c), g) and the same with ∂_x ψ
//! and ∂_y ψ, then for each z point j it sums over c with φ_c(j) (and φ_c' for ∂_z). That costs
//! about 3 nt n per triangle point and 4 n per point and component, against 4 nt n per point of
//! the direct sum, and keeps 3 G n intermediate vectors per lane.
//!
//! As in [`tensor`](super::tensor), the node counts are comptime and the point counts, strides
//! and offsets run-time; every sum starts from the product of its first term and runs in
//! ascending order, with the same operations in every lane.

use cubecl::prelude::*;

use super::tensor::{bc, int};
use crate::GeomLayout;
use crate::layout::{Factor, store_from_jacobian};

/// The prism tensor block whose header starts at `tab[h]`: the factor `f` at its points, with
/// the arguments of the quadrilateral and hexahedron blocks of [`tensor`](super::tensor).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_prism<F: Float, N: Size>(
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
    let nt = comptime!(l.nodes_per_cell * (l.nodes_per_cell + 1) / 2);
    let gd = comptime!(l.gdim);
    let len = comptime!(f.len(&l));
    let o = int::<F>(tab, h);
    let (mt, st, tt) = (
        int::<F>(tab, h + 1),
        int::<F>(tab, h + 2),
        int::<F>(tab, h + 3),
    );
    let (mz, sz, tz) = (
        int::<F>(tab, h + 6),
        int::<F>(tab, h + 7),
        int::<F>(tab, h + 8),
    );
    // B(g, c) at g · n + c, with ψ (v), ∂_x ψ (x) and ∂_y ψ (y).
    let mut bv = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut bx = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut by = Array::<Vector<F, N>>::new(comptime!(l.gdim * l.nodes_per_cell));
    let mut j = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    for i in 0..mt {
        let r = tt + i * nt;
        let rx = r + mt * nt;
        let ry = rx + mt * nt;
        for c in 0..n {
            let lex = c * nt;
            let k = int::<F>(tab, perm + lex);
            let w = bc::<F, N>(tab[r]);
            let wx = bc::<F, N>(tab[rx]);
            let wy = bc::<F, N>(tab[ry]);
            #[unroll]
            for g in 0..gd {
                let x = xe[(k * gd + g) * xs + xo];
                bv[g * n + c] = x * w;
                if comptime!(!f.is_points()) {
                    bx[g * n + c] = x * wx;
                    by[g * n + c] = x * wy;
                }
            }
            for t in 1..nt {
                let k = int::<F>(tab, perm + lex + t);
                let w = bc::<F, N>(tab[r + t]);
                let wx = bc::<F, N>(tab[rx + t]);
                let wy = bc::<F, N>(tab[ry + t]);
                #[unroll]
                for g in 0..gd {
                    let x = xe[(k * gd + g) * xs + xo];
                    bv[g * n + c] += x * w;
                    if comptime!(!f.is_points()) {
                        bx[g * n + c] += x * wx;
                        by[g * n + c] += x * wy;
                    }
                }
            }
        }
        for jz in 0..mz {
            let r0 = tz + jz * n;
            let r0d = r0 + mz * n;
            let q = o + i * st + jz * sz;
            let at = (base + q) * len;
            if comptime!(f.is_points()) {
                #[unroll]
                for g in 0..gd {
                    let mut acc = bv[g * n] * bc::<F, N>(tab[r0]);
                    for c in 1..n {
                        acc += bv[g * n + c] * bc::<F, N>(tab[r0 + c]);
                    }
                    out[(at + g) * vpt + lane] = acc + bc::<F, N>(extra[g]);
                }
            } else {
                #[unroll]
                for g in 0..gd {
                    let w = bc::<F, N>(tab[r0]);
                    let mut j0 = bx[g * n] * w;
                    let mut j1 = by[g * n] * w;
                    let mut j2 = bv[g * n] * bc::<F, N>(tab[r0d]);
                    for c in 1..n {
                        let w = bc::<F, N>(tab[r0 + c]);
                        j0 += bx[g * n + c] * w;
                        j1 += by[g * n + c] * w;
                        j2 += bv[g * n + c] * bc::<F, N>(tab[r0d + c]);
                    }
                    j[g] = j0;
                    j[gd + g] = j1;
                    j[2 * gd + g] = j2;
                }
                store_from_jacobian::<F, N>(
                    &j, extra, flags, out, at, vpt, lane, v, q, first_cell, last_cell, l, f,
                );
            }
        }
    }
}
