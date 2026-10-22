//! `measure_block` (Listing B11).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, Result, debug_assert_bindings,
};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, PointSet, QuadField, ScalarShape};

/// Per-cell sums Σ_q det J(q) · w_q, starting with q = 0, of the cells `c < num_cells` of one
/// tiled det J chunk (one component per point); cell c of the chunk is written to
/// `out[first + c]`. Scalar: one cell per item.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn cell_sums_kernel<F: Float>(
    detj: &[F],
    weights: &[F],
    out: &mut [F],
    first: u32,
    num_cells: u32,
    items_per_unit: u32,
    #[comptime] num_points: usize,
    #[comptime] tile: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = begin + it;
        if c < num_cells as usize {
            let t = c / tile;
            let lane = c - t * tile;
            let base = t * num_points * tile + lane;
            let mut acc = detj[base] * weights[0];
            for q in 1..num_points {
                acc += detj[base + q * tile] * weights[q];
            }
            out[first as usize + c] = acc;
        }
    }
}

/// Values summed by one unit in each pass of [`sum`].
const BLOCK: usize = 64;

/// `out[i] = Σ_{j < BLOCK} x[i · BLOCK + j]` over the entries below `len`, starting with
/// j = 0: one pass of a fixed-order blocked sum. Scalar: one output per item.
#[cube(launch_unchecked)]
#[allow(clippy::needless_range_loop)] // kernel code indexes device slices
fn block_sum_kernel<F: Float>(
    x: &[F],
    out: &mut [F],
    len: u32,
    num_out: u32,
    items_per_unit: u32,
    #[comptime] block: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = begin + it;
        if i < num_out as usize {
            let start = i * block;
            let end = min(start + block, len as usize);
            let mut acc = x[start];
            for j in start + 1..end {
                acc += x[j];
            }
            out[i] = acc;
        }
    }
}

/// The sum of the non-empty `x` by passes of fixed-order blocked sums, read back once.
/// Deterministic: no atomics and no order that depends on the launch shape.
fn sum<F: Real>(mut x: DeviceBuffer<F>) -> Result<f64> {
    let client = x.client().clone();
    while x.len() > 1 {
        let num_out = x.len().div_ceil(BLOCK);
        let out = DeviceBuffer::<F>::uninit(&client, num_out)?;
        let plan =
            LaunchPlan::elementwise(&client, KernelKind::Reduce, num_out, size_of::<F>(), 1, 1)?;
        debug_assert_bindings(&client, 2);
        // SAFETY: outputs i < num_out read x[i · BLOCK .. min((i + 1) · BLOCK, len)], within
        // its len entries, and write out[i] < num_out, its length.
        unsafe {
            block_sum_kernel::launch_unchecked::<F>(
                &client,
                plan.cube_count,
                plan.cube_dim,
                x.arg(),
                out.arg(),
                fit(x.len())?,
                fit(num_out)?,
                plan.items_per_unit,
                BLOCK,
            );
        }
        x = out;
    }
    // The per-block read that finishes measure in f64 (CLAUDE.md).
    Ok(x.to_host()?[0].widen())
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The measure of the owned cells of block `b`: per-cell sums of det J · w on the device, a
/// fixed-order device sum, one read, finished in f64 on this rank (Listing B11). The caller
/// sums the blocks and over ranks.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::measure_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 2)?;
/// let area = measure_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// assert!((area - 1.0).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
pub fn measure_block<F, TD, GD, G>(g: &G, b: usize, pts: &PointSet<F, TD>) -> Result<f64>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let client = g.client();
    let owned = g.cell_blocks()[b].index_map.num_owned();
    if owned == 0 || pts.is_empty() {
        return Ok(0.0);
    }
    let detj: QuadField<F, ScalarShape> = launch_factor(g, b, pts, Factor::DetJ, None)?;
    let sums = DeviceBuffer::<F>::uninit(client, owned)?;
    let tile = detj.tile();
    for chunk in detj.chunks() {
        // Owned cells come first in the block, so a chunk holds owned cells up to `owned`.
        let cells = chunk.num_cells.min(owned.saturating_sub(chunk.first_cell));
        if cells == 0 {
            continue;
        }
        let plan =
            LaunchPlan::elementwise(client, KernelKind::Reduce, cells, size_of::<F>(), 1, 1)?;
        debug_assert_bindings(client, 3);
        // SAFETY: cells c < cells read det J at tile·num_points·tile + q·tile + lane, within the
        // chunk's padded cells · num_points values; the weights below num_points; and write
        // sums[first_cell + c] < owned, the length of `sums`.
        unsafe {
            cell_sums_kernel::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                chunk.data.arg(),
                pts.weights_device().arg(),
                sums.arg(),
                fit(chunk.first_cell)?,
                fit(cells)?,
                plan.items_per_unit,
                pts.len(),
                tile,
            );
        }
    }
    sum(sums)
}
