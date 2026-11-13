//! The Newton kernel of the `cooperative` pull-back (docs/plans/M9h.md, T10b): `loops`' Newton
//! iterations with each step's contraction Σ_k a_k φ̃_k(ξ) split among the units of a group by
//! the tree of [`crate::cooperative`].
//!
//! The group shares one cell's set-up: `loops`' set-up kernel writes the coordinate map's
//! coefficients a(k, g) once per cell, and the units read only those of their own basis
//! functions ([`expansion_cooperative`]). Every unit receives the step's sum
//! ([`group_sum`]) and runs the same Newton update on it, so the iterates agree in every unit
//! without another exchange; unit 0 stores them. Every unit runs all `max_iterations` (there is
//! no early exit), every slot of a cube visits as many vectors as the cube's first slot, and
//! slots past the launch's last vector read the last one and store nothing, so every unit
//! reaches every `sync_cube`.

use cubecl::prelude::*;
use gridcl_dims::CellType;

use super::{Group, group_sum, position};
use crate::GeomLayout;
use crate::basis_eval::expansion_cooperative;
use crate::pull_back::{centroid, newton_update};

/// The components of the step's sum: the value and tdim derivatives of each of gdim columns.
pub(crate) fn step_components(l: &GeomLayout) -> usize {
    l.gdim * (l.tdim + 1)
}

/// `pull_back::pull_back_loops_kernel` with each vector of cells taken by a group of `g.units`
/// units, the sums of every Newton step split by the tree of [`crate::cooperative`]; `l` from
/// `loops_layout`, `cap` the evaluator's capacity. The same buffers (five) and arguments, and
/// `nodes` ≥ `g.leaves`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pull_back_cooperative_kernel<F: Float, N: Size>(
    x: &[Vector<F, N>],
    a: &[Vector<F, N>],
    coefficients: &[F],
    xi: &mut [Vector<F, N>],
    step: &mut [Vector<F, N>],
    x_first_tile: u32,
    a_stride: u32,
    xi_first_tile: u32,
    step_first_tile: u32,
    max_iterations: u32,
    tolerance_hi: f32,
    tolerance_lo: f32,
    num_vectors: u32,
    items_per_unit: u32,
    nodes: u32,
    num_points: u32,
    degree: u32,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
    #[comptime] cap: usize,
    #[comptime] g: Group,
) {
    let td = l.tdim;
    let gd = l.gdim;
    let comps = comptime!(step_components(&l));
    let nb = nodes as usize;
    let nq = num_points as usize;
    let vpt = comptime!(l.tile) / N::value();
    // The runtime zero and one of `basis_eval`.
    let zero = Vector::<F, N>::broadcast(coefficients[nb * nb]);
    let one = Vector::<F, N>::broadcast(coefficients[nb * nb + 1]);
    let tol = Vector::<F, N>::broadcast(F::cast_from(tolerance_hi) + F::cast_from(tolerance_lo));
    let big = Vector::<F, N>::broadcast(F::max_value());
    let no = Vector::<u32, N>::broadcast(0u32);
    let a_stride = a_stride as usize;
    let (slot, u) = position(g);
    let num_vectors = num_vectors as usize;
    let last = num_vectors - 1;
    let items = items_per_unit as usize;
    let first = (CUBE_POS * g.slots + slot) * items;
    // The vectors a slot loops over, as in `factors::factor_cooperative_kernel`.
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
        let read_tile = vr / vpt;
        let read_lane = vr - read_tile * vpt;
        let local_tile = v / vpt;
        let lane = v - local_tile * vpt;
        for q in 0..nq {
            // The point relative to the origin.
            let xo = ((x_first_tile as usize + read_tile) * nq + q) * gd;
            let mut y = Array::<Vector<F, N>>::new(comptime!(gd));
            #[unroll]
            for c in 0..gd {
                y[c] = x[(xo + c) * vpt + read_lane]
                    - Vector::<F, N>::broadcast(coefficients[nb * nb + 2 + c]);
            }
            // The centroid, times the runtime one (no operation on constants alone).
            let mut p = Array::<Vector<F, N>>::new(comptime!(td));
            #[unroll]
            for d in 0..td {
                p[d] = Vector::<F, N>::broadcast(F::new(comptime!(centroid(cell, d)))) * one;
            }
            let mut last_step = big;
            let mut frozen = no;
            for _ in 0..max_iterations {
                let part = expansion_cooperative::<F, N>(
                    &p,
                    a,
                    vr,
                    a_stride,
                    one,
                    zero,
                    degree as usize,
                    nb,
                    u,
                    cell,
                    gd,
                    cap,
                    g,
                );
                let e = group_sum::<F, N>(&part, u, comps, g);
                newton_update::<F, N>(&mut p, &e, &y, &mut last_step, &mut frozen, tol, l);
            }
            if u == 0 && v < num_vectors {
                let xo = ((xi_first_tile as usize + local_tile) * nq + q) * td;
                #[unroll]
                for d in 0..td {
                    xi[(xo + d) * vpt + lane] = p[d];
                }
                let so = (step_first_tile as usize + local_tile) * nq + q;
                step[so * vpt + lane] = last_step;
            }
        }
    }
}
