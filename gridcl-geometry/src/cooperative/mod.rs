//! The `cooperative` strategy (docs/plans/M9h.md, T10a and T10b; ADR-0042, item 4): a group of
//! units per vector of cells, planned by `LaunchPlan::per_group`.
//!
//! A sum over the K terms of a cell (the nodes of the factor kernels) is split into L leaves,
//! L the largest power of two not above min([`MAX_LEAVES`], K), so no leaf is empty: leaf l
//! sums the terms l, l + L, l + 2L, … in ascending order from the product of its first term,
//! as `loops` does. The group's `units_per_group` units take L / `units_per_group` consecutive
//! leaves each and reduce them pairwise ([`unit_tree`]); the units then combine their partials
//! pairwise in shared memory ([`group_sum`]). The tree over the L leaves is therefore the same
//! for every power-of-two `units_per_group` dividing L, including 1, and so are the results:
//! bitwise on the CPU runtime, whatever the vector size, since every lane runs the same
//! operations and no float constant enters (C16).
//!
//! On GPUs a launch asks `per_group` for L units per group and gets U ≤ L. The shared array
//! holds one vector per unit of the cube for each of the components reduced together; where
//! the cube's units times the components exceed the shared memory, the components go through
//! the tree in rounds of [`Group::round`] (at least one).
//!
//! On CPUs a launch asks for one unit per group. A CubeCL CPU unit is a thread, and
//! `sync_cube` a spin barrier on pinned workers (C21): with the barriers of a tree per point,
//! a cube of cooperating threads spends its time waiting whenever other work shares the cores.
//! One unit per group needs neither barriers nor shared memory (whose declaration alone makes
//! the CPU runtime fence every cube), and sums over the same tree; tests reach several units on
//! the CPU runtime with injected GPU-class properties.

pub(crate) mod factors;

use cubecl::prelude::*;
use gridcl_device::{DeviceClass, GridError, KernelKind, LaunchPlan, Result, device_properties};

/// The most leaves of a group's tree: a power of two.
pub(crate) const MAX_LEAVES: usize = 32;

/// The leaves of a sum over `terms` terms: the largest power of two not above
/// min([`MAX_LEAVES`], `terms`), and 1 for no term.
pub(crate) fn leaves(terms: usize) -> usize {
    1 << terms.clamp(1, MAX_LEAVES).ilog2()
}

/// The comptime shape of a cooperative launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Group {
    /// Units per group (`LaunchPlan::units_per_group`), a power of two dividing `leaves`.
    pub units: usize,
    /// Groups per cube: `cube_dim / units`.
    pub slots: usize,
    /// Leaves of the tree.
    pub leaves: usize,
    /// Components reduced together through the shared array (at least 1; no shared array
    /// with one unit per group).
    pub round: usize,
}

impl Group {
    /// The shape of no cooperative launch (for launchers that run another strategy).
    pub(crate) const NONE: Group = Group {
        units: 1,
        slots: 1,
        leaves: 1,
        round: 1,
    };

    /// Shared-memory vectors of the cube: one per unit and component of a round.
    pub(crate) fn shared_len(self) -> usize {
        self.slots * self.units * self.round
    }
}

/// The plan of a cooperative launch over `groups` groups (cells, padded to whole tiles) whose
/// sums have `terms` terms and `comps` components: `LaunchPlan::per_group` for
/// [`leaves`]`(terms)` units per group on GPUs and one on CPUs, and the [`Group`] of its
/// shape. With several units per group, the round is the most components whose shared array
/// (one vector per unit of the cube and component) fits the device's shared memory, at most
/// `comps`, and `per_group` checks that array's bytes.
///
/// # Errors
///
/// `TooLarge` where one component per unit of the cube exceeds the shared memory, and as
/// `per_group`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan(
    client: &Client,
    kind: KernelKind,
    groups: usize,
    elem_bytes: usize,
    max_vector: usize,
    live: usize,
    terms: usize,
    comps: usize,
) -> Result<(LaunchPlan, Group)> {
    let leaves = leaves(terms);
    let wanted = if DeviceClass::for_client(client) == DeviceClass::Gpu {
        leaves
    } else {
        1
    };
    // The shape first (no shared memory), then the plan with the round's bytes, which
    // `per_group` checks against the limit; the shape does not depend on them.
    let shape = LaunchPlan::per_group(
        client, kind, groups, wanted, elem_bytes, max_vector, live, 0,
    )?;
    let cube_dim = shape.cube_dim.num_elems() as usize;
    if shape.units_per_group == 1 {
        let g = Group {
            units: 1,
            slots: cube_dim,
            leaves,
            round: 1,
        };
        return Ok((shape, g));
    }
    let per_component = cube_dim
        .checked_mul(shape.vector_size)
        .and_then(|v| v.checked_mul(elem_bytes))
        .ok_or(GridError::TooLarge)?;
    let limit = device_properties(client).hardware.max_shared_memory_size;
    let round = (limit / per_component).min(comps.max(1));
    if round == 0 {
        return Err(GridError::TooLarge);
    }
    let plan = LaunchPlan::per_group(
        client,
        kind,
        groups,
        wanted,
        elem_bytes,
        max_vector,
        live,
        round * per_component,
    )?;
    let units = plan.units_per_group as usize;
    debug_assert_eq!(plan.cube_dim.num_elems() as usize, cube_dim);
    debug_assert!(leaves.is_multiple_of(units));
    let g = Group {
        units,
        slots: cube_dim / units,
        leaves,
        round,
    };
    Ok((plan, g))
}

/// The unit's position in its group: (slot of the cube, unit of the group).
#[cube]
pub(crate) fn position(#[comptime] g: Group) -> (usize, usize) {
    let slot = UNIT_POS as usize / g.units;
    (slot, UNIT_POS as usize - slot * g.units)
}

/// Adds `leaf` (C components), the unit's i-th leaf, into the pairwise tree over its leaves
/// held in `stack`: (log₂(per unit) + 1) · C vectors, level by level. After the last leaf
/// the unit's partial is at the top level ([`unit_partial`]).
#[cube]
pub(crate) fn unit_tree<F: Float, N: Size>(
    stack: &mut Array<Vector<F, N>>,
    leaf: &mut Array<Vector<F, N>>,
    i: usize,
    #[comptime] comps: usize,
) {
    let mut level = 0usize;
    let mut bits = i;
    while (bits & 1) == 1 {
        #[unroll]
        for c in 0..comps {
            leaf[c] = stack[level * comps + c] + leaf[c];
        }
        bits >>= 1;
        level += 1;
    }
    #[unroll]
    for c in 0..comps {
        stack[level * comps + c] = leaf[c];
    }
}

/// The unit's partial, the top level of `stack` after its L / U leaves went through
/// [`unit_tree`].
#[cube]
pub(crate) fn unit_partial<F: Float, N: Size>(
    stack: &Array<Vector<F, N>>,
    #[comptime] comps: usize,
    #[comptime] g: Group,
) -> Array<Vector<F, N>> {
    let top = comptime!((g.leaves / g.units).ilog2() as usize);
    let mut out = Array::<Vector<F, N>>::new(comps);
    #[unroll]
    for c in 0..comps {
        out[c] = stack[top * comps + c];
    }
    out
}

/// The levels of [`unit_tree`]'s stack: log₂(L / U) + 1.
pub(crate) fn stack_levels(g: Group) -> usize {
    (g.leaves / g.units).ilog2() as usize + 1
}

/// The sum of the group's partials, pairwise over its units (at strides t = 1, 2, …, unit u
/// with u mod 2t = 0 adds unit u + t's partial), valid in unit 0. The components go through
/// a shared array of [`Group::shared_len`] vectors (one per unit of the cube and component of
/// a round) in rounds of `g.round`. Every unit of the cube calls it, in uniform control flow:
/// it calls `sync_cube`. With one unit per group it returns `part` and declares no shared
/// memory. A kernel calls it once (one shared array).
#[cube]
pub(crate) fn group_sum<F: Float, N: Size>(
    part: &Array<Vector<F, N>>,
    u: usize,
    #[comptime] comps: usize,
    #[comptime] g: Group,
) -> Array<Vector<F, N>> {
    let mut total = Array::<Vector<F, N>>::new(comps);
    if comptime!(g.units == 1) {
        #[unroll]
        for c in 0..comps {
            total[c] = part[c];
        }
    } else {
        let mut shared = Shared::<[Vector<F, N>]>::new_slice(comptime!(g.shared_len()));
        let p = UNIT_POS as usize * g.round;
        #[unroll]
        for r in 0..comptime!(comps.div_ceil(g.round)) {
            let first = comptime!(r * g.round);
            let len = comptime!(g.round.min(comps - r * g.round));
            // The previous round's (or call's) reads of `shared` are done.
            sync_cube();
            #[unroll]
            for c in 0..len {
                shared[p + c] = part[first + c];
            }
            sync_cube();
            let mut t = 1usize;
            while t < g.units {
                if (u & (2 * t - 1)) == 0 {
                    let o = p + t * g.round;
                    #[unroll]
                    for c in 0..len {
                        shared[p + c] = shared[p + c] + shared[o + c];
                    }
                }
                sync_cube();
                t *= 2;
            }
            #[unroll]
            for c in 0..len {
                total[first + c] = shared[p + c];
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaves_are_powers_of_two_up_to_the_terms() {
        for (terms, want) in [
            (0, 1),
            (1, 1),
            (2, 2),
            (3, 2),
            (8, 8),
            (27, 16),
            (64, 32),
            (729, 32),
        ] {
            assert_eq!(leaves(terms), want, "{terms}");
        }
    }
}
