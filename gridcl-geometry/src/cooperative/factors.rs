//! The factor kernel of the `cooperative` strategy (docs/plans/M9h.md, T10a): `loops`' sums
//! over the nodes of a cell, split among the units of a group by the tree of
//! [`crate::cooperative`].
//!
//! For each point, every unit sums its leaves of the nodes (X, or J for every other factor),
//! the group combines them in shared memory, and unit 0 stores the factor as `loops` does
//! (`layout::store_from_jacobian`). Where groups have several units, every slot of a cube
//! visits as many vectors as the cube's first slot, and slots past the launch's last vector of
//! cells read the last one and store nothing, so every unit reaches every `sync_cube`.

use cubecl::prelude::*;

use super::{Group, group_sum, position, stack_levels, unit_partial, unit_tree};
use crate::GeomLayout;
use crate::layout::{Factor, store_from_jacobian};

/// The components a factor kernel sums over the nodes: X (`Points`) or J (every other factor).
pub(crate) fn summed_components(l: &GeomLayout, f: Factor) -> usize {
    if f.is_points() {
        l.gdim
    } else {
        l.tdim * l.gdim
    }
}

/// The unit's partial of X (`points`) or J at point q of vector v's cells: its leaves
/// u · L/U + i for i < L/U, leaf λ summing the nodes λ, λ + L, … below `nodes` from the
/// product of node λ, each through [`unit_tree`].
#[cube]
#[allow(clippy::too_many_arguments)]
fn unit_sum<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    tab: &[F],
    x_offset: usize,
    x_stride: usize,
    v: usize,
    q: usize,
    nodes: usize,
    u: usize,
    #[comptime] l: GeomLayout,
    #[comptime] points: bool,
    #[comptime] g: Group,
) -> Array<Vector<F, N>> {
    let td = l.tdim;
    let gd = l.gdim;
    let comps = comptime!(if points { l.gdim } else { l.tdim * l.gdim });
    let per_unit = comptime!(g.leaves / g.units);
    let mut stack = Array::<Vector<F, N>>::new(comptime!(stack_levels(g) * comps));
    let mut leaf = Array::<Vector<F, N>>::new(comps);
    let row = q * nodes;
    for i in 0..per_unit {
        let first = u * per_unit + i;
        if comptime!(points) {
            let w = Vector::<F, N>::broadcast(tab[row + first]);
            #[unroll]
            for c in 0..gd {
                leaf[c] = xe[(first * gd + c) * x_stride + x_offset + v] * w;
            }
            let mut k = first + g.leaves;
            while k < nodes {
                let w = Vector::<F, N>::broadcast(tab[row + k]);
                #[unroll]
                for c in 0..gd {
                    leaf[c] += xe[(k * gd + c) * x_stride + x_offset + v] * w;
                }
                k += g.leaves;
            }
        } else {
            #[unroll]
            for c in 0..gd {
                let x = xe[(first * gd + c) * x_stride + x_offset + v];
                #[unroll]
                for d in 0..td {
                    leaf[d * gd + c] = x * Vector::<F, N>::broadcast(tab[(row + first) * td + d]);
                }
            }
            let mut k = first + g.leaves;
            while k < nodes {
                #[unroll]
                for c in 0..gd {
                    let x = xe[(k * gd + c) * x_stride + x_offset + v];
                    #[unroll]
                    for d in 0..td {
                        leaf[d * gd + c] += x * Vector::<F, N>::broadcast(tab[(row + k) * td + d]);
                    }
                }
                k += g.leaves;
            }
        }
        unit_tree::<F, N>(&mut stack, &mut leaf, i, comps);
    }
    unit_partial::<F, N>(&stack, comps, g)
}

/// `layout_loops::factor_loops_kernel` with each vector of cells taken by a group of
/// `g.units` units, its node sums split by the tree of [`crate::cooperative`]; `l` from
/// `loops_layout`. The same buffers (five) and arguments, and `nodes_per_cell` ≥ `g.leaves`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_cooperative_kernel<F: Float, N: Size>(
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
    #[comptime] g: Group,
) {
    let vpt = comptime!(l.tile) / N::value();
    let len = comptime!(f.len(&l));
    let comps = comptime!(summed_components(&l, f));
    let nodes = nodes_per_cell as usize;
    let x_offset = x_offset as usize;
    let x_stride = x_stride as usize;
    let num_points = num_points as usize;
    let (slot, u) = position(g);
    let num_vectors = num_vectors as usize;
    let last = num_vectors - 1;
    let items = items_per_unit as usize;
    let first = (CUBE_POS * g.slots + slot) * items;
    // The vectors a slot loops over: its own up to the last vector with one unit per group (no
    // `sync_cube`); else as many as the cube's first slot takes, the same in every slot, so
    // `sync_cube` stays uniform (CPU slots take blocks of many vectors, GPU slots one).
    let mut from = CUBE_POS * g.slots * items;
    if comptime!(g.units == 1) {
        from = first;
    }
    let count = min(items, num_vectors - min(from, num_vectors));
    for i in 0..count {
        let v = first + i;
        // Slots past the last vector read it and store nothing.
        let vr = min(v, last);
        // v − (v / vpt) · vpt, not v % vpt (C3).
        let local_tile = v / vpt;
        let lane = v - local_tile * vpt;
        let tile = out_first_tile as usize + local_tile;
        for q in 0..num_points {
            let part = unit_sum::<F, N>(
                xe,
                tab,
                x_offset,
                x_stride,
                vr,
                q,
                nodes,
                u,
                l,
                comptime!(f.is_points()),
                g,
            );
            let total = group_sum::<F, N>(&part, u, comps, g);
            if u == 0 && v < num_vectors {
                let o = (tile * num_points + q) * len;
                match comptime!(f) {
                    Factor::Points =>
                    {
                        #[unroll]
                        for c in 0..l.gdim {
                            out[(o + c) * vpt + lane] =
                                total[c] + Vector::<F, N>::broadcast(extra[c]);
                        }
                    }
                    _ => {
                        store_from_jacobian::<F, N>(
                            &total, extra, flags, out, o, vpt, lane, v, q, first_cell, last_cell,
                            l, f,
                        );
                    }
                }
            }
        }
    }
}
