//! The `shared` strategy (docs/plans/M9h.md, T9; ADR-0042, items 3 and 4): the kernels of
//! `loops` and `sumfact` with their table staged in shared memory once per cube.
//!
//! A factor kernel stages its tabulation: the tensor table of `sumfact` where sum
//! factorisation applies (`PointSet::tabulate_tensor`), else the full tabulation of `loops`
//! (values or derivatives). The pull-back's set-up stages C′ (`loops`) or the tensor data of
//! `sumfact` (C₁′, the permutation and on prisms C′_tri); its Newton steps read only their
//! coefficients and are those of `loops` and `sumfact`. The E-vector and the bench evaluator
//! have no table, so `shared` falls back to `loops` for them, as it does where the table
//! exceeds the device's shared-memory limit (`strategy::choose`).
//!
//! The launches are planned by `LaunchPlan::per_group` with one unit per group, which checks
//! the shared memory per cube; with `units_per_group` = 1 the vector of slot s of cube c is
//! `(c · cube_dim + s) · items_per_unit + i`, which is `ABSOLUTE_POS · items_per_unit + i` as
//! in the `elementwise` kernels. Every unit of a cube copies a strided part of the table and
//! waits at one `sync_cube` before the cube's work, in uniform control flow. The arithmetic is
//! that of `loops` and `sumfact`, so the results are bitwise equal to theirs.

use cubecl::prelude::*;
use gridcl_device::{KernelKind, LaunchPlan, Result, device_properties};
use gridcl_dims::CellType;

use crate::GeomLayout;
use crate::layout::Factor;
use crate::layout_loops::factor_loops_body;
use crate::pull_back::pull_back_setup_body;
use crate::sumfact::pull_back::pull_back_sumfact_setup_body;
use crate::sumfact::tensor::factor_sumfact_body;

/// Shared-memory entries of a staged table of `len` entries of `elem_bytes` bytes: the next
/// power of two, so few kernels serve every degree, or the largest count within `limit`
/// bytes when that power exceeds it. `None` when the table itself exceeds `limit`.
pub(crate) fn capacity(len: usize, elem_bytes: usize, limit: usize) -> Option<usize> {
    let most = limit / elem_bytes;
    (len <= most).then(|| {
        len.max(1)
            .checked_next_power_of_two()
            .unwrap_or(most)
            .min(most)
    })
}

/// Whether a table of `len` entries of `elem_bytes` bytes fits `client`'s shared memory per
/// cube (`hardware.max_shared_memory_size`, the injected properties in tests).
pub(crate) fn fits(client: &Client, len: usize, elem_bytes: usize) -> bool {
    let limit = device_properties(client).hardware.max_shared_memory_size;
    capacity(len, elem_bytes, limit).is_some()
}

/// The plan and the shared capacity of a staged launch over `cells` cells (padded to whole
/// tiles) with a table of `len` entries: `LaunchPlan::per_group` with one unit per group and
/// the capacity's bytes of shared memory per cube.
///
/// # Errors
///
/// `TooLarge` where the table exceeds the shared memory (`strategy::choose` has checked it
/// fits) and as `per_group`.
pub(crate) fn plan(
    client: &Client,
    kind: KernelKind,
    cells: usize,
    elem_bytes: usize,
    tile: usize,
    live: usize,
    len: usize,
) -> Result<(LaunchPlan, usize)> {
    let limit = device_properties(client).hardware.max_shared_memory_size;
    let cap = capacity(len, elem_bytes, limit).ok_or(gridcl_device::GridError::TooLarge)?;
    let plan = LaunchPlan::per_group(
        client,
        kind,
        cells,
        1,
        elem_bytes,
        tile,
        live,
        cap * elem_bytes,
    )?;
    Ok((plan, cap))
}

/// `src[..len]` copied into a shared array of `cap` ≥ `len` entries by the units of the cube,
/// each taking every `CUBE_DIM`-th entry, behind one `sync_cube`. Called in uniform control
/// flow by every unit.
#[cube]
pub(crate) fn stage<F: Float>(src: &[F], len: u32, #[comptime] cap: usize) -> Shared<[F]> {
    let mut staged = Shared::<[F]>::new_slice(cap);
    let mut i = UNIT_POS;
    while i < len {
        staged[i as usize] = src[i as usize];
        i += CUBE_DIM;
    }
    sync_cube();
    staged
}

/// [`factor_sumfact_body`] with the table's first `tab_len` entries staged in a shared array
/// of `cap` entries. Five buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_sumfact_shared_kernel<F: Float, N: Size>(
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
    tab_len: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
    #[comptime] prism: bool,
    #[comptime] cap: usize,
) {
    let staged = stage::<F>(tab, tab_len, cap);
    factor_sumfact_body::<F, N>(
        xe,
        &staged,
        extra,
        flags,
        out,
        x_offset,
        x_stride,
        out_first_tile,
        first_cell,
        last_cell,
        num_vectors,
        items_per_unit,
        num_points,
        l,
        f,
        prism,
    );
}

/// [`factor_loops_body`] with the tabulation's first `tab_len` entries staged in a shared
/// array of `cap` entries. Five buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_loops_shared_kernel<F: Float, N: Size>(
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
    tab_len: u32,
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
    #[comptime] cap: usize,
) {
    let staged = stage::<F>(tab, tab_len, cap);
    factor_loops_body::<F, N>(
        xe,
        &staged,
        extra,
        flags,
        out,
        x_offset,
        x_stride,
        out_first_tile,
        first_cell,
        last_cell,
        num_vectors,
        items_per_unit,
        nodes_per_cell,
        num_points,
        l,
        f,
    );
}

/// [`pull_back_setup_body`] (`loops`' set-up) with the first `len` entries of `coefficients`
/// (C′, then the runtime zero and one and the origin) staged in a shared array of `cap`
/// entries. Three buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pull_back_setup_shared_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    coefficients: &[F],
    a: &mut [Vector<F, N>],
    xe_offset: u32,
    xe_stride: u32,
    a_stride: u32,
    num_vectors: u32,
    items_per_unit: u32,
    nodes: u32,
    len: u32,
    #[comptime] gd: usize,
    #[comptime] cap: usize,
) {
    let staged = stage::<F>(coefficients, len, cap);
    pull_back_setup_body::<F, N>(
        xe,
        &staged,
        a,
        xe_offset,
        xe_stride,
        a_stride,
        num_vectors,
        items_per_unit,
        nodes,
        gd,
    );
}

/// [`pull_back_sumfact_setup_body`] (`sumfact`'s set-up) with the first `len` entries of the
/// tensor data staged in a shared array of `cap` entries. Three buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pull_back_sumfact_setup_shared_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    data: &[F],
    a: &mut [Vector<F, N>],
    xe_offset: u32,
    xe_stride: u32,
    a_stride: u32,
    num_vectors: u32,
    items_per_unit: u32,
    nodes: u32,
    len: u32,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
    #[comptime] cap: usize,
) {
    let staged = stage::<F>(data, len, cap);
    pull_back_sumfact_setup_body::<F, N>(
        xe,
        &staged,
        a,
        xe_offset,
        xe_stride,
        a_stride,
        num_vectors,
        items_per_unit,
        nodes,
        l,
        cell,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacities_are_powers_of_two_within_the_limit() {
        // (len, elem_bytes, limit, want)
        for (len, bytes, limit, want) in [
            (0, 4, 1024, Some(1)),
            (1, 4, 1024, Some(1)),
            (3, 4, 1024, Some(4)),
            (256, 4, 1024, Some(256)),
            (257, 4, 1024, None),
            (200, 4, 1000, Some(250)),
            (130, 8, 1024, None),
            (100, 8, 1024, Some(128)),
            (5000, 4, 32 << 10, Some(8192)),
            (5000, 8, 48 << 10, Some(6144)),
            (usize::MAX, 4, usize::MAX, None),
        ] {
            assert_eq!(
                capacity(len, bytes, limit),
                want,
                "{len} × {bytes} within {limit}"
            );
        }
    }
}
