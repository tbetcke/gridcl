//! Pull-back: reference points of physical points by Newton iteration, generic over the
//! vector size (docs/design.md, "Pull-back"; Listing B4b; ADR-0039, items 13-15).

use cubecl::prelude::*;
use gridcl_device::{
    GridError, KernelKind, LaunchPlan, Real, Result, VECTOR_PAD, debug_assert_bindings,
};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};

use crate::basis_eval::expansion;
use crate::layout::pinv_of;
use crate::strategy::{self, StrategyFamily};
use crate::{BlockFields, DeviceGrid, GeomLayout, QuadChunk, QuadField, ScalarShape, VecShape};

/// Options of the pull-back (ADR-0039, item 13).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let opts = PullBackOptions::default();
/// assert_eq!((opts.tolerance, opts.max_iterations), (None, 16));
/// // A fixed tolerance and more iterations for strongly curved cells.
/// let opts = PullBackOptions { tolerance: Some(1e-10), max_iterations: 32 };
/// assert_eq!(opts.max_iterations, 32);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PullBackOptions {
    /// A lane freezes once |Δξ|∞ ≤ tolerance; `None` is 10^−13 in f64 and 10^−5 in f32.
    pub tolerance: Option<f64>,
    /// The number of iterations, always run in full: there is no early exit.
    pub max_iterations: u32,
}

impl Default for PullBackOptions {
    fn default() -> Self {
        PullBackOptions {
            tolerance: None,
            max_iterations: 16,
        }
    }
}

/// The result of a pull-back, one field per cell block, tiled like the input.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let pb = mesh.pull_back(&mesh.physical_points(&rule)?, &PullBackOptions::default())?;
/// // One block; the cell centres pull back to the reference centre, converged.
/// let xi = pb.reference.blocks[0].to_host_cell_major()?;
/// assert!(xi.iter().all(|p| (p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6));
/// let steps = pb.last_step.blocks[0].to_host_cell_major()?;
/// assert!(steps.iter().all(|&s| s <= 1e-5));
/// # Ok::<(), GridError>(())
/// ```
pub struct PullBack<F: Real, TD: TopoDim> {
    /// The reference point ξ per cell and point.
    pub reference: BlockFields<F, VecShape<TD>>,
    /// |Δξ|∞ of the last iteration; above the tolerance where the point has not converged.
    pub last_step: BlockFields<F, ScalarShape>,
}

/// Coordinate `d` of the vertex centroid of `cell`, the start of the iteration.
fn centroid(cell: CellType, d: usize) -> f32 {
    let v = gridcl_refcell::vertices(cell);
    (v.iter().map(|p| p[d]).sum::<f64>() / v.len() as f64) as f32
}

/// The pull-back of the points of the cell vectors `v < num_vectors` of one chunk range. The
/// lanes are N consecutive cells of one tile (N divides the tile), as in the factor kernels.
///   x             input QuadField chunk, gdim components per point, absolute coordinates
///   xe            E-vector chunk [k][g][cell], cells fastest; row (k, g) starts at
///                 (k·G + g)·xe_stride, relative to the origin
///   coefficients  C' (nodes × nodes), the runtime zero and one, the origin (`EVector::
///                 coefficients`)
///   xi, step      output QuadField chunks: ξ (tdim components) and |Δξ|∞ of the last step
/// Tile offsets count tiles from the start of each chunk; the E-vector offset and stride count
/// vectors. Per cell, the coordinate map's expansion coefficients a(k, g) = Σ_j C'(k, j) x(j, g)
/// are formed once; per point, every lane runs all `max_iterations` steps
/// ξ ← ξ + K(ξ)(x − X(ξ)) and freezes, through lane selects, once |Δξ|∞ ≤ tolerance (ADR-0039,
/// item 13). A non-finite component of Δξ counts as a step of `F::max_value()`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pull_back_kernel<F: Float, N: Size>(
    x: &[Vector<F, N>],
    xe: &[Vector<F, N>],
    coefficients: &[F],
    xi: &mut [Vector<F, N>],
    step: &mut [Vector<F, N>],
    x_first_tile: u32,
    xe_offset: u32,
    xe_stride: u32,
    xi_first_tile: u32,
    step_first_tile: u32,
    max_iterations: u32,
    tolerance_hi: f32,
    tolerance_lo: f32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
    #[comptime] degree: usize,
) {
    let td = l.tdim;
    let gd = l.gdim;
    let nb = l.nodes_per_cell;
    let vpt = comptime!(l.tile) / N::value();
    // The runtime zero and one of `basis_eval`.
    let zero = Vector::<F, N>::broadcast(coefficients[nb * nb]);
    let one = Vector::<F, N>::broadcast(coefficients[nb * nb + 1]);
    let tol = Vector::<F, N>::broadcast(F::cast_from(tolerance_hi) + F::cast_from(tolerance_lo));
    let big = Vector::<F, N>::broadcast(F::max_value());
    let no = Vector::<u32, N>::broadcast(0u32);
    let yes = Vector::<u32, N>::broadcast(1u32);
    let xe_offset = xe_offset as usize;
    let xe_stride = xe_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt (CubeCL folds `x % 1` on the CPU runtime).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            // The coordinate map's coefficients a(k, g) at k · G + g, starting each sum with
            // j = 0 rather than 0 (`layout::jacobian`).
            let mut a = Array::<Vector<F, N>>::new(comptime!(nb * gd));
            for k in 0..nb {
                #[unroll]
                for g in 0..gd {
                    let mut acc = Vector::<F, N>::broadcast(coefficients[k * nb])
                        * xe[g * xe_stride + xe_offset + v];
                    for j in 1..nb {
                        acc += Vector::<F, N>::broadcast(coefficients[k * nb + j])
                            * xe[(j * gd + g) * xe_stride + xe_offset + v];
                    }
                    a[k * gd + g] = acc;
                }
            }
            for q in 0..l.num_points {
                // The point relative to the origin.
                let xo = ((x_first_tile as usize + local_tile) * l.num_points + q) * gd;
                let mut y = Array::<Vector<F, N>>::new(comptime!(gd));
                #[unroll]
                for g in 0..gd {
                    y[g] = x[(xo + g) * vpt + lane]
                        - Vector::<F, N>::broadcast(coefficients[nb * nb + 2 + g]);
                }
                // The centroid, times the runtime one (no operation on constants alone).
                let mut p = Array::<Vector<F, N>>::new(comptime!(td));
                #[unroll]
                for d in 0..td {
                    p[d] = Vector::<F, N>::broadcast(F::new(comptime!(centroid(cell, d)))) * one;
                }
                let mut last = big;
                let mut frozen = no;
                for _ in 0..max_iterations {
                    // X at entry g (1 + T), ∂X_g/∂ξ_d at g (1 + T) + 1 + d.
                    let e = expansion::<F, N>(&p, &a, one, zero, cell, degree, gd);
                    let mut j = Array::<Vector<F, N>>::new(comptime!(td * gd));
                    #[unroll]
                    for d in 0..td {
                        #[unroll]
                        for g in 0..gd {
                            j[d * gd + g] = e[g * (td + 1) + 1 + d];
                        }
                    }
                    let k = pinv_of::<F, N>(&j, l);
                    let mut r = Array::<Vector<F, N>>::new(comptime!(gd));
                    #[unroll]
                    for g in 0..gd {
                        r[g] = y[g] - e[g * (td + 1)];
                    }
                    // Δ = K r; |Δ|∞ and the number of non-finite components (comparisons
                    // with NaN are false).
                    let mut delta = Array::<Vector<F, N>>::new(comptime!(td));
                    let mut s = big;
                    let mut bad = no;
                    #[unroll]
                    for t in 0..td {
                        let mut acc = k[t] * r[0];
                        #[unroll]
                        for g in 1..gd {
                            acc += k[g * td + t] * r[g];
                        }
                        delta[t] = acc;
                        let m = acc.abs();
                        if comptime!(t == 0) {
                            s = m;
                        } else {
                            s = select_many(m.greater_than(&s), m, s);
                        }
                        bad += select_many(m.less_equal(&big), no, yes);
                    }
                    s = select_many(bad.equal(&no), s, big);
                    // Frozen lanes keep ξ and the step by selects, so no value of a diverging
                    // lane reaches them.
                    let keep = frozen.equal(&yes);
                    #[unroll]
                    for d in 0..td {
                        p[d] = select_many(keep, p[d], p[d] + delta[d]);
                    }
                    last = select_many(keep, last, s);
                    frozen = select_many(keep, frozen, select_many(s.less_equal(&tol), yes, no));
                }
                let xo = ((xi_first_tile as usize + local_tile) * l.num_points + q) * td;
                #[unroll]
                for d in 0..td {
                    xi[(xo + d) * vpt + lane] = p[d];
                }
                let so = (step_first_tile as usize + local_tile) * l.num_points + q;
                step[so * vpt + lane] = last;
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// A tile-aligned cell range that lies in one chunk of each of the four chunkings.
struct Range {
    /// The global index of the range's first cell.
    first_cell: usize,
    /// The range's cells rounded up to whole tiles.
    padded_cells: usize,
    /// The chunk of each chunking that holds the range.
    chunks: [usize; 4],
}

/// The ranges, in cell order, between consecutive chunk starts of the four chunkings, which
/// all cover the same cells. Every chunk starts on a multiple of `tile` (field chunks on whole
/// tiles, E-vector chunks on multiples of `VECTOR_PAD`), so every range does; a range ends at a
/// chunk start, a multiple of `tile`, or at the last cell, so its padded end lies inside the
/// padded allocation of each of its chunks.
fn ranges<F: Real>(parts: [&[QuadChunk<F>]; 4], tile: usize) -> Vec<Range> {
    let end = parts[0].last().map_or(0, |c| c.first_cell + c.num_cells);
    let mut cuts: Vec<usize> = parts
        .iter()
        .flat_map(|p| p.iter().map(|c| c.first_cell))
        .chain([end])
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut chunks = [0; 4];
    let mut out = Vec::new();
    for w in cuts.windows(2) {
        for (i, part) in parts.iter().enumerate() {
            while part[chunks[i]].first_cell + part[chunks[i]].num_cells <= w[0] {
                chunks[i] += 1;
            }
        }
        out.push(Range {
            first_cell: w[0],
            padded_cells: (w[1] - w[0]).next_multiple_of(tile),
            chunks,
        });
    }
    out
}

/// The reference points of the physical points `x` (absolute coordinates per cell and point
/// of block `b`, as `physical_points_block` returns them), and |Δξ|∞ of the last iteration,
/// both tiled like `x`.
///
/// Iteration starts at the reference cell's vertex centroid with Δ = K(ξ)(x − X(ξ)), K the
/// pseudo-inverse: Newton in codimension 0, Gauss–Newton otherwise. A lane freezes once
/// |Δ|∞ ≤ tolerance; the loop always runs `max_iterations`, so the result is the same at
/// every vector size (ADR-0039, items 13 and 14). Where the iteration has not converged, the
/// last step exceeds the tolerance: a step with a non-finite component counts as
/// `F::MAX`, and so does the step of `max_iterations = 0`. X(ξ) and J(ξ) come from the
/// device evaluation of the coordinate element's basis; pyramid points are clamped to
/// ζ ≤ 1 − 2^−10 (ADR-0039, item 15). Absolute input coordinates lose about |origin| · ε_F,
/// as `physical_points_block` does.
///
/// Plans use `KernelKind::PullBack` with `max_vector` = the tile width of `x`. The first call
/// for a block uploads its expansion coefficients, cached with the E-vector.
///
/// # Errors
///
/// `LengthMismatch` if `x` does not have one cell per local cell of block `b`; the element
/// errors of the block's coordinate element; allocation and device errors; `NotYetImplemented`
/// where the strategy chosen for the block does not run yet (ADR-0042, item 3).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::{physical_points_block, pull_back_block};
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Triangle, 2, 2)
///     .perturbed(0.05)
///     .build::<f32, D2, D2>(&client)?;
/// let pts = QuadRule::for_grid(&mesh, 2)?;
/// let pts = pts.get(Cell2::Triangle)?;
/// let x = physical_points_block(&mesh, 0, pts)?;
/// let (xi, step) = pull_back_block(&mesh, 0, &x, &PullBackOptions::default())?;
/// // Every Gauss point comes back, and every lane has converged.
/// let want = pts.points();
/// for (i, p) in xi.to_host_cell_major()?.iter().enumerate() {
///     let q = i % pts.len();
///     assert!((f64::from(p[0]) - want[2 * q]).abs() < 1e-5);
///     assert!((f64::from(p[1]) - want[2 * q + 1]).abs() < 1e-5);
/// }
/// assert!(step.to_host_cell_major()?.iter().all(|&s| s <= 1e-5));
/// # Ok::<(), GridError>(())
/// ```
#[allow(clippy::type_complexity)] // Listing B4b's signature
pub fn pull_back_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    x: &QuadField<F, VecShape<GD>>,
    opts: &PullBackOptions,
) -> Result<(QuadField<F, VecShape<TD>>, QuadField<F, ScalarShape>)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let client = g.client();
    let num_cells = g.cell_blocks()[b].index_map.num_local();
    if x.num_cells() != num_cells {
        return Err(GridError::LengthMismatch {
            expected: num_cells,
            found: x.num_cells(),
        });
    }
    let (tile, num_points) = (x.tile(), x.num_points());
    let xi = QuadField::<F, VecShape<TD>>::uninit(client, num_cells, num_points, tile)?;
    let step = QuadField::<F, ScalarShape>::uninit(client, num_cells, num_points, tile)?;
    if num_cells == 0 || num_points == 0 {
        return Ok((xi, step));
    }
    let gb = g.geometry_block(b);
    let cell: CellType = gb.element.cell.into();
    // Only `direct` runs until T6b (docs/plans/M9h.md); `choose` rejects the others.
    strategy::choose(
        client,
        StrategyFamily::PullBack,
        cell,
        gb.element.degree,
        size_of::<F>(),
    )?;
    let layout = GeomLayout::new::<TD, GD>(&gb.element, num_points, tile);
    let xe = g.evector(b)?;
    let coefficients = xe.coefficients(
        cell,
        gb.element.degree,
        gb.element.family,
        g.origin().as_ref(),
    )?;
    // The tolerance as an f32 pair whose sum in F rounds like the tolerance itself (a float
    // scalar argument of generic F is not a CubeCL launch argument).
    let tolerance = opts
        .tolerance
        .unwrap_or(if size_of::<F>() == 8 { 1e-13 } else { 1e-5 });
    let tolerance_hi = tolerance as f32;
    let tolerance_lo = (tolerance - f64::from(tolerance_hi)) as f32;
    let (td, gd, nb) = (layout.tdim, layout.gdim, layout.nodes_per_cell);
    let live = nb * gd + 3 * td * gd + 2 * gd + 3 * td + 6;
    for r in ranges([xe.chunks(), x.chunks(), xi.chunks(), step.chunks()], tile) {
        let [ce, cx, cxi, cs] = r.chunks;
        let (ce, cx) = (&xe.chunks()[ce], &x.chunks()[cx]);
        let (cxi, cs) = (&xi.chunks()[cxi], &step.chunks()[cs]);
        let plan = LaunchPlan::elementwise(
            client,
            KernelKind::PullBack,
            r.padded_cells,
            size_of::<F>(),
            tile,
            live,
        )?;
        let n = plan.vector_size;
        debug_assert_bindings(client, 5);
        // SAFETY: n divides the tile, which divides r.padded_cells, VECTOR_PAD, every chunk's
        // first cell and the E-vector stride, so the vector offsets below are exact. Every
        // range starts in its four chunks and its padded end lies inside each chunk's padded
        // allocation (`ranges`). The kernel handles vectors v < padded_cells / n, and for each:
        // - it reads E-vector vectors (k·G + g)·stride/n + offset/n + v < K·G·stride/n
        //   (offset + padded_cells ≤ stride, the chunk's cell count padded to VECTOR_PAD);
        // - it reads `coefficients` below K² + 2 + G, the length of the cached buffer;
        // - it reads `x` and writes `xi` and `step` in tile first_tile + v/(tile/n), inside the
        //   padded cells of each field chunk, at vector offsets below
        //   padded_cells · num_points · LEN / n past the range's first tile.
        unsafe {
            pull_back_kernel::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                n,
                cx.data.vector_arg(n),
                ce.data.vector_arg(n),
                coefficients.arg(),
                cxi.data.vector_arg(n),
                cs.data.vector_arg(n),
                fit((r.first_cell - cx.first_cell) / tile)?,
                fit((r.first_cell - ce.first_cell) / n)?,
                fit(ce.num_cells.next_multiple_of(VECTOR_PAD) / n)?,
                fit((r.first_cell - cxi.first_cell) / tile)?,
                fit((r.first_cell - cs.first_cell) / tile)?,
                opts.max_iterations,
                tolerance_hi,
                tolerance_lo,
                fit(r.padded_cells / n)?,
                plan.items_per_unit,
                layout,
                cell,
                gb.element.degree as usize,
            );
        }
    }
    Ok((xi, step))
}
