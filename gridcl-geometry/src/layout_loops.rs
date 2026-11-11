//! The factor kernel of the `loops` strategy (docs/plans/M9h.md, T6a; ADR-0041): the node and
//! point counts are runtime arguments, so one kernel serves every degree of a (TD, GD, factor,
//! tile), and the operation count does not grow with the degree (C18, C20).
//!
//! The sums run over the nodes in ascending order from the product of node 0, as in the
//! `direct` kernel, so every lane and every vector size performs the same operations (no float
//! constant 0 or 1, C16), and the results are bitwise equal across vector sizes on the CPU
//! runtime.

use cubecl::prelude::*;

use crate::GeomLayout;
use crate::layout::{Factor, store_from_jacobian};

/// The comptime layout of the `loops` kernels for `l`: the node and point counts are 0, since
/// the kernels take them at run time.
pub(crate) fn loops_layout(l: GeomLayout) -> GeomLayout {
    GeomLayout {
        nodes_per_cell: 0,
        num_points: 0,
        ..l
    }
}

/// J at point q of the lanes' cells, as `layout::jacobian`, with `nodes` nodes at run time: for
/// each node, its coordinates are read once and added into every column of J.
#[cube]
#[allow(clippy::too_many_arguments)]
fn jacobian<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    dphi: &[F],
    x_offset: usize,
    x_stride: usize,
    v: usize,
    q: usize,
    nodes: usize,
    #[comptime] l: GeomLayout,
) -> Array<Vector<F, N>> {
    let td = l.tdim;
    let gd = l.gdim;
    let row = q * nodes;
    let mut j = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    #[unroll]
    for g in 0..gd {
        let x = xe[g * x_stride + x_offset + v];
        #[unroll]
        for d in 0..td {
            j[d * gd + g] = x * Vector::<F, N>::broadcast(dphi[row * td + d]);
        }
    }
    for k in 1..nodes {
        #[unroll]
        for g in 0..gd {
            let x = xe[(k * gd + g) * x_stride + x_offset + v];
            #[unroll]
            for d in 0..td {
                j[d * gd + g] += x * Vector::<F, N>::broadcast(dphi[(row + k) * td + d]);
            }
        }
    }
    j
}

/// `layout::factor_kernel` with runtime loops over `nodes_per_cell` nodes and `num_points`
/// points; `l` from [`loops_layout`]. The same buffers (five) and arguments otherwise.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_loops_kernel<F: Float, N: Size>(
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
    nodes_per_cell: u32,
    num_points: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
) {
    let vpt = comptime!(l.tile) / N::value();
    let len = comptime!(f.len(&l));
    let nodes = nodes_per_cell as usize;
    let x_offset = x_offset as usize;
    let x_stride = x_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt (C3).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            let tile = out_first_tile as usize + local_tile;
            for q in 0..num_points as usize {
                let o = (tile * num_points as usize + q) * len;
                match comptime!(f) {
                    Factor::Points => {
                        let mut x = Array::<Vector<F, N>>::new(comptime!(l.gdim));
                        let w = Vector::<F, N>::broadcast(tab[q * nodes]);
                        #[unroll]
                        for g in 0..l.gdim {
                            x[g] = xe[g * x_stride + x_offset + v] * w;
                        }
                        for k in 1..nodes {
                            let w = Vector::<F, N>::broadcast(tab[q * nodes + k]);
                            #[unroll]
                            for g in 0..l.gdim {
                                x[g] += xe[(k * l.gdim + g) * x_stride + x_offset + v] * w;
                            }
                        }
                        #[unroll]
                        for g in 0..l.gdim {
                            out[(o + g) * vpt + lane] = x[g] + Vector::<F, N>::broadcast(extra[g]);
                        }
                    }
                    _ => {
                        let j = jacobian::<F, N>(xe, tab, x_offset, x_stride, v, q, nodes, l);
                        store_from_jacobian::<F, N>(
                            &j, extra, flags, out, o, vpt, lane, v, q, first_cell, last_cell, l, f,
                        );
                    }
                }
            }
        }
    }
}
