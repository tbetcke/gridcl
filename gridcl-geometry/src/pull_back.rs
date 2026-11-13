//! Pull-back: reference points of physical points by Newton iteration, generic over the
//! vector size (docs/design.md, "Pull-back"; Listing B4b; ADR-0039, items 13-15).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, Result, VECTOR_PAD,
    debug_assert_bindings,
};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};

use crate::basis_eval::{expansion, expansion_loops, loops_capacity};
use crate::layout::pinv_of;
use crate::layout_loops::loops_layout;
use crate::strategy::Strategy;
use crate::strategy::{self, StrategyFamily};
use crate::sumfact::pull_back::{
    pull_back_sumfact_kernel, pull_back_sumfact_setup_kernel, sumfact_pull_back_live,
    sumfact_setup_live,
};
use crate::sumfact::tensor::sumfact_layout;
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
pub(crate) fn centroid(cell: CellType, d: usize) -> f32 {
    let v = gridcl_refcell::vertices(cell);
    (v.iter().map(|p| p[d]).sum::<f64>() / v.len() as f64) as f32
}

/// The lower and upper bound of the box [−0.1, 1.1]^TD that confines the iterates: every
/// reference cell lies in [0, 1]^TD (ADR-0046). The margins are 0.1 rounded to f32.
const BOX: [f32; 2] = [-0.1, 1.1];

/// One Newton iteration of the lanes at `p`, from the coordinate map's expansion `e` at `p`
/// (X_g at entry g (1 + TD), ∂X_g/∂ξ_d at g (1 + TD) + 1 + d) and the target `y` (relative to
/// the origin): Δ = K(ξ)(y − X(ξ)), and |Δ|∞ with a non-finite component counting as
/// `F::max_value()`. Lanes not yet frozen move to ξ + Δ clamped to [`BOX`] by two selects
/// (ADR-0046), record |Δ|∞ in `last` and freeze once it is at most `tol`; frozen lanes keep ξ
/// and the step by selects, so no value of a diverging lane reaches them (ADR-0039, item 13).
/// The step is the unclamped one, so a point whose preimage lies outside the box does not
/// converge. Shared by the pull-back kernels of every strategy.
#[cube]
pub(crate) fn newton_update<F: Float, N: Size>(
    p: &mut Array<Vector<F, N>>,
    e: &Array<Vector<F, N>>,
    y: &Array<Vector<F, N>>,
    last: &mut Vector<F, N>,
    frozen: &mut Vector<u32, N>,
    tol: Vector<F, N>,
    #[comptime] l: GeomLayout,
) {
    let td = l.tdim;
    let gd = l.gdim;
    let big = Vector::<F, N>::broadcast(F::max_value());
    let lo = Vector::<F, N>::broadcast(F::new(comptime!(BOX[0])));
    let hi = Vector::<F, N>::broadcast(F::new(comptime!(BOX[1])));
    let no = Vector::<u32, N>::broadcast(0u32);
    let yes = Vector::<u32, N>::broadcast(1u32);
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
    // Δ = K r; |Δ|∞ and the number of non-finite components (comparisons with NaN are false).
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
    let keep = frozen.equal(&yes);
    #[unroll]
    for d in 0..td {
        let mut q = p[d] + delta[d];
        q = select_many(q.less_than(&lo), lo, q);
        q = select_many(q.greater_than(&hi), hi, q);
        p[d] = select_many(keep, p[d], q);
    }
    *last = select_many(keep, *last, s);
    *frozen = select_many(keep, *frozen, select_many(s.less_equal(&tol), yes, no));
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
/// are formed once; per point, every lane runs all `max_iterations` steps of [`newton_update`].
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
                    let e = expansion::<F, N>(&p, &a, one, zero, cell, degree, gd);
                    newton_update::<F, N>(&mut p, &e, &y, &mut last, &mut frozen, tol, l);
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

/// The set-up of the `loops` pull-back (docs/plans/M9h.md, T6b): the coordinate map's expansion
/// coefficients a(k, g) = Σ_j C'(k, j) x(j, g) of the cell vectors `v < num_vectors` of one
/// range, with `nodes` nodes at run time, summed over j in the order of [`pull_back_kernel`].
///   xe            E-vector chunk, as in [`pull_back_kernel`]
///   coefficients  C' (nodes × nodes), then the runtime zero and one and the origin
///   a             `[k][g][cell]`: a(k, g) of the range's cell vector v at (k·G + g)·a_stride + v
/// Three buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pull_back_setup_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    coefficients: &[F],
    a: &mut [Vector<F, N>],
    xe_offset: u32,
    xe_stride: u32,
    a_stride: u32,
    num_vectors: u32,
    items_per_unit: u32,
    nodes: u32,
    #[comptime] gd: usize,
) {
    let nb = nodes as usize;
    let xe_offset = xe_offset as usize;
    let xe_stride = xe_stride as usize;
    let a_stride = a_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            for k in 0..nb {
                let mut acc = Array::<Vector<F, N>>::new(comptime!(gd));
                let c = Vector::<F, N>::broadcast(coefficients[k * nb]);
                #[unroll]
                for g in 0..gd {
                    acc[g] = c * xe[g * xe_stride + xe_offset + v];
                }
                for j in 1..nb {
                    let c = Vector::<F, N>::broadcast(coefficients[k * nb + j]);
                    #[unroll]
                    for g in 0..gd {
                        acc[g] += c * xe[(j * gd + g) * xe_stride + xe_offset + v];
                    }
                }
                #[unroll]
                for g in 0..gd {
                    a[(k * gd + g) * a_stride + v] = acc[g];
                }
            }
        }
    }
}

/// The Newton iterations of the `loops` pull-back (docs/plans/M9h.md, T6b): as
/// [`pull_back_kernel`], with the expansion coefficients from [`pull_back_setup_kernel`]
/// (`a`, range-local cell vectors, row stride `a_stride` vectors), the node and point counts
/// and the degree at run time, and the basis evaluator of [`expansion_loops`] (`cap` its
/// capacity, `l` from `loops_layout`). Five buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pull_back_loops_kernel<F: Float, N: Size>(
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
) {
    let td = l.tdim;
    let gd = l.gdim;
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
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt (CubeCL folds `x % 1` on the CPU runtime).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            for q in 0..nq {
                // The point relative to the origin.
                let xo = ((x_first_tile as usize + local_tile) * nq + q) * gd;
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
                    let e = expansion_loops::<F, N>(
                        &p,
                        a,
                        v,
                        a_stride,
                        one,
                        zero,
                        degree as usize,
                        cell,
                        gd,
                        cap,
                    );
                    newton_update::<F, N>(&mut p, &e, &y, &mut last, &mut frozen, tol, l);
                }
                let xo = ((xi_first_tile as usize + local_tile) * nq + q) * td;
                #[unroll]
                for d in 0..td {
                    xi[(xo + d) * vpt + lane] = p[d];
                }
                let so = (step_first_tile as usize + local_tile) * nq + q;
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
/// pseudo-inverse: Newton in codimension 0, Gauss–Newton otherwise. Each iterate is clamped to
/// [−0.1, 1.1]^TD, the box of every reference cell widened by 0.1, while the freeze test and
/// the last step use the unclamped Δ, so a point whose preimage lies outside that box does not
/// converge (ADR-0046). A lane freezes once
/// |Δ|∞ ≤ tolerance; the loop always runs `max_iterations`, so the result is the same at
/// every vector size (ADR-0039, items 13 and 14). Where the iteration has not converged, the
/// last step exceeds the tolerance: a step with a non-finite component counts as
/// `F::MAX`, and so does the step of `max_iterations = 0`. X(ξ) and J(ξ) come from the
/// device evaluation of the coordinate element's basis; pyramid points are clamped to
/// ζ ≤ 1 − 2^−10 (ADR-0039, item 15). Absolute input coordinates lose about |origin| · ε_F,
/// as `physical_points_block` does.
///
/// Plans use `KernelKind::PullBack` with `max_vector` = the tile width of `x`. The first call
/// for a block uploads its expansion coefficients, cached with the E-vector. Under the `loops`
/// strategy a set-up kernel first writes each cell's coordinate-map coefficients to a scratch
/// buffer of the size of the largest range of the E-vector chunks, which the iterations read.
/// Under `sumfact` (quadrilaterals, hexahedra and prisms) the set-up forms the same
/// coefficients by sum factorisation, and each step evaluates the 1D Legendre polynomials at ξ
/// and contracts them one direction at a time; the first call uploads the interval's
/// coefficients, the node permutation and on prisms the triangle's coefficients instead of the
/// expansion coefficients (ADR-0049).
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
    let s = strategy::choose(
        client,
        StrategyFamily::PullBack,
        cell,
        gb.element.degree,
        size_of::<F>(),
        true,
    )?;
    let layout = GeomLayout::new::<TD, GD>(&gb.element, num_points, tile);
    let xe = g.evector(b)?;
    let (degree, family, origin) = (gb.element.degree, gb.element.family, g.origin());
    // The tolerance as an f32 pair whose sum in F rounds like the tolerance itself (a float
    // scalar argument of generic F is not a CubeCL launch argument).
    let tolerance = opts
        .tolerance
        .unwrap_or(if size_of::<F>() == 8 { 1e-13 } else { 1e-5 });
    let tolerance_hi = tolerance as f32;
    let tolerance_lo = (tolerance - f64::from(tolerance_hi)) as f32;
    let (td, gd, nb) = (layout.tdim, layout.gdim, layout.nodes_per_cell);
    let ranges = ranges([xe.chunks(), x.chunks(), xi.chunks(), step.chunks()], tile);
    // The scratch of the strategies with a set-up: the coordinate map's coefficients of the
    // largest range, nodes × gdim per cell (the size of one E-vector chunk range).
    let scratch = || -> Result<DeviceBuffer<F>> {
        let cells = ranges.iter().map(|r| r.padded_cells).max().unwrap_or(0);
        let per_cell = nb.checked_mul(gd).ok_or(GridError::TooLarge)?;
        DeviceBuffer::<F>::uninit(
            client,
            per_cell.checked_mul(cells).ok_or(GridError::TooLarge)?,
        )
    };
    if s == Strategy::SumFactorised {
        let data = xe.tensor_coefficients(cell, degree, family, origin.as_ref())?;
        // The node count per direction at run time, the arrays' capacity at comptime.
        let n1 = degree as usize + 1;
        let cap = loops_capacity(degree);
        let l = sumfact_layout(layout, cap);
        let a = scratch()?;
        for r in &ranges {
            let [ce, cx, cxi, cs] = r.chunks;
            let (ce, cx) = (&xe.chunks()[ce], &x.chunks()[cx]);
            let (cxi, cs) = (&xi.chunks()[cxi], &step.chunks()[cs]);
            let setup = LaunchPlan::elementwise(
                client,
                KernelKind::PullBack,
                r.padded_cells,
                size_of::<F>(),
                tile,
                sumfact_setup_live(cell, cap, td),
            )?;
            let n = setup.vector_size;
            debug_assert_bindings(client, 3);
            // SAFETY: as for `loops`' set-up below: the kernel handles vectors
            // v < padded_cells / n, reads E-vector vectors (k·G + g)·stride/n + offset/n + v
            // < K·G·stride/n for the nodes k < K of the permutation, `data` below its length
            // (`pull_back_data`: 2 + G + n², the permutation of K entries, on prisms nt²), and
            // writes `a` at (k·G + g)·padded_cells/n + v < K·G·padded_cells/n for k < K (n^TD
            // coefficients on quadrilaterals and hexahedra, nt·n on prisms, both K), within the
            // scratch of K·G·scratch_cells elements.
            unsafe {
                pull_back_sumfact_setup_kernel::launch_unchecked::<F>(
                    client,
                    setup.cube_count,
                    setup.cube_dim,
                    n,
                    ce.data.vector_arg(n),
                    data.arg(),
                    a.vector_arg(n),
                    fit((r.first_cell - ce.first_cell) / n)?,
                    fit(ce.num_cells.next_multiple_of(VECTOR_PAD) / n)?,
                    fit(r.padded_cells / n)?,
                    fit(r.padded_cells / n)?,
                    setup.items_per_unit,
                    fit(n1)?,
                    l,
                    cell,
                );
            }
            let plan = LaunchPlan::elementwise(
                client,
                KernelKind::PullBack,
                r.padded_cells,
                size_of::<F>(),
                tile,
                sumfact_pull_back_live(cell, cap, td, gd),
            )?;
            let n = plan.vector_size;
            debug_assert_bindings(client, 5);
            // SAFETY: as for `loops` below: the kernel reads `a` at (k·G + g)·padded_cells/n + v
            // for k < K, inside the scratch the set-up kernel wrote (stream order), `data` at
            // its first 2 + G entries, and `x`, `xi` and `step` as `direct` does, with
            // num_points at run time.
            unsafe {
                pull_back_sumfact_kernel::launch_unchecked::<F>(
                    client,
                    plan.cube_count,
                    plan.cube_dim,
                    n,
                    cx.data.vector_arg(n),
                    a.vector_arg(n),
                    data.arg(),
                    cxi.data.vector_arg(n),
                    cs.data.vector_arg(n),
                    fit((r.first_cell - cx.first_cell) / tile)?,
                    fit(r.padded_cells / n)?,
                    fit((r.first_cell - cxi.first_cell) / tile)?,
                    fit((r.first_cell - cs.first_cell) / tile)?,
                    opts.max_iterations,
                    tolerance_hi,
                    tolerance_lo,
                    fit(r.padded_cells / n)?,
                    plan.items_per_unit,
                    fit(num_points)?,
                    fit(n1)?,
                    l,
                    cell,
                );
            }
        }
        return Ok((xi, step));
    }
    let coefficients = xe.coefficients(cell, degree, family, origin.as_ref())?;
    if s == Strategy::Loops {
        let cap = loops_capacity(degree);
        // The evaluator's sums, 1D values and recurrences, then the Newton step's values.
        let live = gd * (td + 1) + 3 * cap + 18 + 3 * td * gd + 2 * gd + 3 * td + 6;
        let a = scratch()?;
        for r in ranges {
            let [ce, cx, cxi, cs] = r.chunks;
            let (ce, cx) = (&xe.chunks()[ce], &x.chunks()[cx]);
            let (cxi, cs) = (&xi.chunks()[cxi], &step.chunks()[cs]);
            let setup = LaunchPlan::elementwise(
                client,
                KernelKind::PullBack,
                r.padded_cells,
                size_of::<F>(),
                tile,
                2 * gd + 2,
            )?;
            let n = setup.vector_size;
            debug_assert_bindings(client, 3);
            // SAFETY: n divides the tile, which divides r.padded_cells, VECTOR_PAD, every
            // chunk's first cell and the E-vector stride. The kernel handles vectors
            // v < padded_cells / n; it reads E-vector vectors (k·G + g)·stride/n + offset/n + v
            // < K·G·stride/n (offset + padded_cells ≤ stride), `coefficients` below K², and
            // writes `a` at (k·G + g)·padded_cells/n + v < K·G·padded_cells/n, within the
            // scratch of K·G·scratch_cells elements (padded_cells ≤ scratch_cells).
            unsafe {
                pull_back_setup_kernel::launch_unchecked::<F>(
                    client,
                    setup.cube_count,
                    setup.cube_dim,
                    n,
                    ce.data.vector_arg(n),
                    coefficients.arg(),
                    a.vector_arg(n),
                    fit((r.first_cell - ce.first_cell) / n)?,
                    fit(ce.num_cells.next_multiple_of(VECTOR_PAD) / n)?,
                    fit(r.padded_cells / n)?,
                    fit(r.padded_cells / n)?,
                    setup.items_per_unit,
                    fit(nb)?,
                    gd,
                );
            }
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
            // SAFETY: as for `direct` below, with `a` in place of the E-vector: the kernel
            // reads `a` at (k·G + g)·padded_cells/n + v < K·G·padded_cells/n for k < K, inside
            // the scratch the set-up kernel wrote (stream order), `coefficients` below
            // K² + 2 + G, and `x`, `xi` and `step` as `direct` does.
            unsafe {
                pull_back_loops_kernel::launch_unchecked::<F>(
                    client,
                    plan.cube_count,
                    plan.cube_dim,
                    n,
                    cx.data.vector_arg(n),
                    a.vector_arg(n),
                    coefficients.arg(),
                    cxi.data.vector_arg(n),
                    cs.data.vector_arg(n),
                    fit((r.first_cell - cx.first_cell) / tile)?,
                    fit(r.padded_cells / n)?,
                    fit((r.first_cell - cxi.first_cell) / tile)?,
                    fit((r.first_cell - cs.first_cell) / tile)?,
                    opts.max_iterations,
                    tolerance_hi,
                    tolerance_lo,
                    fit(r.padded_cells / n)?,
                    plan.items_per_unit,
                    fit(nb)?,
                    fit(num_points)?,
                    degree,
                    loops_layout(layout),
                    cell,
                    cap,
                );
            }
        }
        return Ok((xi, step));
    }
    debug_assert!(
        s == Strategy::Direct,
        "`choose` returns only the strategies that run"
    );
    let live = nb * gd + 3 * td * gd + 2 * gd + 3 * td + 6;
    for r in ranges {
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
                degree as usize,
            );
        }
    }
    Ok((xi, step))
}
