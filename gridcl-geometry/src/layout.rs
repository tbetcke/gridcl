//! `GeomLayout` (Listing B5).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, Result, VectorPolicy,
    debug_assert_bindings,
};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};

use crate::field::chunk_pairs;
use crate::{CoordinateElement, DeviceGrid, FieldShape, PointSet, QuadField};

/// The comptime layout of a factor kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeomLayout {
    /// The topological dimension.
    pub tdim: usize,
    /// The geometric dimension.
    pub gdim: usize,
    /// Nodes of the coordinate element.
    pub nodes_per_cell: usize,
    /// Points per cell.
    pub num_points: usize,
    /// Cells per `QuadField` tile: 1, 2, 4, 8 or 16.
    pub tile: usize,
}

impl GeomLayout {
    /// The layout of element `e` at `points` points per cell in tiles of `tile` cells.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, D2, D3};
    /// use gridcl_geometry::{CoordinateElement, GeomLayout, NodeFamily};
    ///
    /// let e = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 2, family: NodeFamily::Equispaced };
    /// let l = GeomLayout::new::<D2, D3>(&e, 7, 16);
    /// assert_eq!((l.tdim, l.gdim, l.nodes_per_cell, l.num_points, l.tile), (2, 3, 6, 7, 16));
    /// ```
    pub fn new<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
        e: &CoordinateElement<TD>,
        points: usize,
        tile: usize,
    ) -> Self {
        let cell: CellType = e.cell.into();
        GeomLayout {
            tdim: TD::N,
            gdim: GD::N,
            nodes_per_cell: gridcl_refcell::num_lagrange_nodes(cell, e.degree),
            num_points: points,
            tile,
        }
    }
}

/// The factor a factor kernel computes; a comptime parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Factor {
    /// Physical points X, with the origin added back.
    Points,
    /// The Jacobian J, column-major.
    Jacobian,
    /// det J, or the pseudo-determinant sqrt(det(JᵀJ)) when TD < GD.
    DetJ,
    /// K = (JᵀJ)⁻¹Jᵀ, column-major TD × GD.
    Pinv,
    /// Outward unit facet normals Kᵀ n_ref / |Kᵀ n_ref| (codimension 0).
    FacetNormals,
    /// Unit normals s (J0 × J1) / |J0 × J1| or s (J0_y, −J0_x) / |J0| (codimension 1).
    Normals,
}

impl Factor {
    /// Components per point.
    pub(crate) fn len(self, l: &GeomLayout) -> usize {
        match self {
            Factor::Points | Factor::FacetNormals | Factor::Normals => l.gdim,
            Factor::Jacobian | Factor::Pinv => l.tdim * l.gdim,
            Factor::DetJ => 1,
        }
    }
}

/// J at point q of the lanes' cells: entry (g, d) at component d · G + g, the sum over the
/// nodes k of x(k, g) · ∂_d φ_k(q) (Listing B5), starting with k = 0 rather than with 0, whose
/// `0 + x` an optimizer may fold and so flip the sign of a zero; the same operation sequence
/// in every lane as for N = 1.
#[cube]
fn jacobian<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    dphi: &[F],
    x_offset: usize,
    x_stride: usize,
    v: usize,
    q: usize,
    #[comptime] l: GeomLayout,
) -> Array<Vector<F, N>> {
    let mut j = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    #[unroll]
    for d in 0..l.tdim {
        #[unroll]
        for g in 0..l.gdim {
            let mut acc = xe[g * x_stride + x_offset + v]
                * Vector::<F, N>::broadcast(dphi[q * l.nodes_per_cell * l.tdim + d]);
            #[unroll]
            for k in 1..l.nodes_per_cell {
                let x = xe[(k * l.gdim + g) * x_stride + x_offset + v];
                let w = Vector::<F, N>::broadcast(dphi[(q * l.nodes_per_cell + k) * l.tdim + d]);
                acc += x * w;
            }
            j[d * l.gdim + g] = acc;
        }
    }
    j
}

/// Σ_g J(g, a) · J(g, b) of the column-major `j`, starting with g = 0.
#[cube]
fn column_dot<F: Float, N: Size>(
    j: &Array<Vector<F, N>>,
    #[comptime] a: usize,
    #[comptime] b: usize,
    #[comptime] gdim: usize,
) -> Vector<F, N> {
    let mut acc = j[a * gdim] * j[b * gdim];
    #[unroll]
    for g in 1..gdim {
        acc += j[a * gdim + g] * j[b * gdim + g];
    }
    acc
}

/// det J of a square column-major `j` (J(g, d) at d · D + g): J00 (1×1), J00·J11 − J01·J10
/// (2×2), cofactors along the first row (3×3).
#[cube]
#[allow(unused_assignments)] // every comptime branch assigns `det`
fn square_det<F: Float, N: Size>(j: &Array<Vector<F, N>>, #[comptime] dim: usize) -> Vector<F, N> {
    let mut det = j[0];
    if comptime!(dim == 2) {
        det = j[0] * j[3] - j[2] * j[1];
    } else if comptime!(dim == 3) {
        det = j[0] * (j[4] * j[8] - j[7] * j[5]) - j[3] * (j[1] * j[8] - j[7] * j[2])
            + j[6] * (j[1] * j[5] - j[4] * j[2]);
    }
    det
}

/// det J, or the pseudo-determinant sqrt(det(JᵀJ)) when TD < GD.
#[cube]
#[allow(unused_assignments)] // every comptime branch assigns `det`
fn detj_of<F: Float, N: Size>(j: &Array<Vector<F, N>>, #[comptime] l: GeomLayout) -> Vector<F, N> {
    let mut det = j[0];
    if comptime!(l.tdim == l.gdim) {
        det = square_det::<F, N>(j, l.tdim);
    } else if comptime!(l.tdim == 1) {
        det = column_dot::<F, N>(j, 0usize, 0usize, l.gdim).sqrt();
    } else {
        let a = column_dot::<F, N>(j, 0usize, 0usize, l.gdim);
        let b = column_dot::<F, N>(j, 0usize, 1usize, l.gdim);
        let c = column_dot::<F, N>(j, 1usize, 1usize, l.gdim);
        det = (a * c - b * b).sqrt();
    }
    det
}

/// K = (JᵀJ)⁻¹Jᵀ of the column-major `j`, K(t, g) at component g · T + t: the adjugate times
/// 1 / det in codimension 0; J(g, 0) · (1/a) for TD = 1 < GD; Ginv · Jᵀ with
/// Ginv = (c·inv, −b·inv; −b·inv, a·inv), inv = 1/(a·c − b·b), for TD = 2, GD = 3.
#[cube]
fn pinv_of<F: Float, N: Size>(
    j: &Array<Vector<F, N>>,
    #[comptime] l: GeomLayout,
) -> Array<Vector<F, N>> {
    let gd = l.gdim;
    let td = l.tdim;
    let one = Vector::<F, N>::broadcast(F::from_int(1));
    let mut k = Array::<Vector<F, N>>::new(comptime!(l.tdim * l.gdim));
    if comptime!(l.tdim == l.gdim) {
        let inv = one / square_det::<F, N>(j, l.tdim);
        if comptime!(l.tdim == 1) {
            k[0] = inv;
        } else if comptime!(l.tdim == 2) {
            k[0] = j[3] * inv;
            k[2] = -j[2] * inv;
            k[1] = -j[1] * inv;
            k[3] = j[0] * inv;
        } else {
            k[0] = (j[4] * j[8] - j[7] * j[5]) * inv;
            k[3] = (j[6] * j[5] - j[3] * j[8]) * inv;
            k[6] = (j[3] * j[7] - j[6] * j[4]) * inv;
            k[1] = (j[7] * j[2] - j[1] * j[8]) * inv;
            k[4] = (j[0] * j[8] - j[6] * j[2]) * inv;
            k[7] = (j[6] * j[1] - j[0] * j[7]) * inv;
            k[2] = (j[1] * j[5] - j[4] * j[2]) * inv;
            k[5] = (j[3] * j[2] - j[0] * j[5]) * inv;
            k[8] = (j[0] * j[4] - j[3] * j[1]) * inv;
        }
    } else if comptime!(l.tdim == 1) {
        let inv = one / column_dot::<F, N>(j, 0usize, 0usize, l.gdim);
        #[unroll]
        for g in 0..gd {
            k[g] = j[g] * inv;
        }
    } else {
        let a = column_dot::<F, N>(j, 0usize, 0usize, l.gdim);
        let b = column_dot::<F, N>(j, 0usize, 1usize, l.gdim);
        let c = column_dot::<F, N>(j, 1usize, 1usize, l.gdim);
        let inv = one / (a * c - b * b);
        let g00 = c * inv;
        let g01 = -b * inv;
        let g11 = a * inv;
        #[unroll]
        for g in 0..gd {
            k[g * td] = g00 * j[g] + g01 * j[gd + g];
            k[g * td + 1] = g01 * j[g] + g11 * j[gd + g];
        }
    }
    k
}

/// A geometric factor at every point of the cell vectors `v < num_vectors` of one chunk pair.
/// The lanes are N consecutive cells of one tile (N divides the tile).
///   xe     E-vector chunk [k][g][cell], cells fastest; row (k, g) starts at (k·G + g)·x_stride
///   tab    φ [q][k] (Points) or ∂φ [q][k][d] (all others); broadcast to every lane
///   out    tiled QuadField chunk
///   extra  the origin [gdim] (Points) or the reference normals [q][tdim] (FacetNormals)
///   flags  cell flags of the block (Normals): bit 0 flips the normal; lanes read cell
///          min(first_cell + v·N + lane, last_cell)
/// Offsets and strides count vectors.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn factor_kernel<F: Float, N: Size>(
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
    #[comptime] l: GeomLayout,
    #[comptime] f: Factor,
) {
    let vpt = comptime!(l.tile) / N::value();
    let len = comptime!(f.len(&l));
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt: CubeCL 0.11 folds `x % 1` with a comptime 1
            // to x on the CPU runtime (vpt = 1 at tile width 1).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            let tile = out_first_tile as usize + local_tile;
            for q in 0..l.num_points {
                let o = (tile * l.num_points + q) * len;
                match comptime!(f) {
                    Factor::Points =>
                    {
                        #[unroll]
                        for g in 0..l.gdim {
                            let mut acc = xe[g * x_stride as usize + x_offset as usize + v]
                                * Vector::<F, N>::broadcast(tab[q * l.nodes_per_cell]);
                            #[unroll]
                            for k in 1..l.nodes_per_cell {
                                let x = xe
                                    [(k * l.gdim + g) * x_stride as usize + x_offset as usize + v];
                                let w = Vector::<F, N>::broadcast(tab[q * l.nodes_per_cell + k]);
                                acc += x * w;
                            }
                            out[(o + g) * vpt + lane] = acc + Vector::<F, N>::broadcast(extra[g]);
                        }
                    }
                    Factor::Jacobian => {
                        let j = jacobian::<F, N>(
                            xe,
                            tab,
                            x_offset as usize,
                            x_stride as usize,
                            v,
                            q,
                            l,
                        );
                        #[unroll]
                        for c in 0..comptime!(l.tdim * l.gdim) {
                            out[(o + c) * vpt + lane] = j[c];
                        }
                    }
                    Factor::DetJ => {
                        let j = jacobian::<F, N>(
                            xe,
                            tab,
                            x_offset as usize,
                            x_stride as usize,
                            v,
                            q,
                            l,
                        );
                        out[o * vpt + lane] = detj_of::<F, N>(&j, l);
                    }
                    Factor::Pinv => {
                        let j = jacobian::<F, N>(
                            xe,
                            tab,
                            x_offset as usize,
                            x_stride as usize,
                            v,
                            q,
                            l,
                        );
                        let k = pinv_of::<F, N>(&j, l);
                        #[unroll]
                        for c in 0..comptime!(l.tdim * l.gdim) {
                            out[(o + c) * vpt + lane] = k[c];
                        }
                    }
                    Factor::FacetNormals => {
                        let j = jacobian::<F, N>(
                            xe,
                            tab,
                            x_offset as usize,
                            x_stride as usize,
                            v,
                            q,
                            l,
                        );
                        let k = pinv_of::<F, N>(&j, l);
                        let td = l.tdim;
                        let mut m = Array::<Vector<F, N>>::new(comptime!(l.gdim));
                        #[unroll]
                        for g in 0..l.gdim {
                            let mut acc = k[g * td] * Vector::<F, N>::broadcast(extra[q * td]);
                            #[unroll]
                            for t in 1..td {
                                acc += k[g * td + t] * Vector::<F, N>::broadcast(extra[q * td + t]);
                            }
                            m[g] = acc;
                        }
                        let mut s2 = m[0] * m[0];
                        #[unroll]
                        for g in 1..l.gdim {
                            s2 += m[g] * m[g];
                        }
                        let norm = s2.sqrt();
                        #[unroll]
                        for g in 0..l.gdim {
                            out[(o + g) * vpt + lane] = m[g] / norm;
                        }
                    }
                    Factor::Normals => {
                        let j = jacobian::<F, N>(
                            xe,
                            tab,
                            x_offset as usize,
                            x_stride as usize,
                            v,
                            q,
                            l,
                        );
                        let mut n = Array::<Vector<F, N>>::new(comptime!(l.gdim));
                        if comptime!(l.gdim == 2) {
                            n[0] = j[1];
                            n[1] = -j[0];
                        } else {
                            n[0] = j[1] * j[5] - j[2] * j[4];
                            n[1] = j[2] * j[3] - j[0] * j[5];
                            n[2] = j[0] * j[4] - j[1] * j[3];
                        }
                        let mut s2 = n[0] * n[0];
                        #[unroll]
                        for g in 1..l.gdim {
                            s2 += n[g] * n[g];
                        }
                        let norm = s2.sqrt();
                        // The sign of each lane's cell, by arithmetic masks.
                        let mut bit = Vector::<u32, N>::broadcast(0u32);
                        #[unroll]
                        for ln in 0..N::value() {
                            let c = min(first_cell + (v * N::value() + ln) as u32, last_cell);
                            bit.insert(ln, flags[c as usize] & 1u32);
                        }
                        let s = select_many(
                            bit.equal(&Vector::<u32, N>::broadcast(1u32)),
                            Vector::<F, N>::broadcast(F::from_int(-1)),
                            Vector::<F, N>::broadcast(F::from_int(1)),
                        );
                        #[unroll]
                        for g in 0..l.gdim {
                            out[(o + g) * vpt + lane] = (s * n[g]) / norm;
                        }
                    }
                }
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The factor `f` of block `b` of `g` at the points of `pts`: one launch per chunk pair, each
/// with its own `LaunchPlan` (GeomFactors, `max_vector` = the tile width; ADR-0012).
pub(crate) fn launch_factor<F, TD, GD, G, S>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
    f: Factor,
    extra: Option<&DeviceBuffer<F>>,
) -> Result<QuadField<F, S>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
    S: FieldShape,
{
    let client = g.client();
    let gb = g.geometry_block(b);
    let tile = VectorPolicy::for_client(client).quad_tile();
    let layout = GeomLayout::new::<TD, GD>(&gb.element, pts.len(), tile);
    assert_eq!(S::LEN, f.len(&layout), "field shape and factor disagree");
    let num_cells = g.cell_blocks()[b].index_map.num_local();
    let out = QuadField::<F, S>::uninit(client, num_cells, pts.len(), tile)?;
    if num_cells == 0 || pts.is_empty() {
        return Ok(out);
    }
    let xe = g.evector(b)?;
    let tab = match f {
        Factor::Points => pts.tabulate(&gb.element)?,
        _ => pts.tabulate_derivatives(&gb.element)?,
    };
    let live = 2 * layout.tdim * layout.gdim + 2;
    let flags = &g.cell_blocks()[b].flags;
    for r in chunk_pairs(xe, &out) {
        let plan = LaunchPlan::elementwise(
            client,
            KernelKind::GeomFactors,
            r.padded_cells,
            size_of::<F>(),
            tile,
            live,
        )?;
        let n = plan.vector_size;
        debug_assert_bindings(client, 5);
        // SAFETY: n divides the tile, which divides r.padded_cells, VECTOR_PAD and the
        // E-vector stride, so the vector offsets below are exact. The kernel handles vectors
        // v < padded_cells / n: it reads E-vector vectors (k·G + g)·stride/n + x_first_cell/n
        // + v < K·G·stride/n, inside the E-vector chunk (x_first_cell + padded_cells ≤
        // stride); it reads `tab` below num_points · K · tdim, `extra` below gdim (the origin)
        // or num_points · tdim (reference normals), and `flags` at cells ≤ num_cells − 1; it
        // writes
        // tiles
        // out_first_cell/tile + v/(tile/n), which lie inside the field chunk's padded cells, at
        // vector offsets below the chunk's padded_cells · num_points · LEN / n.
        unsafe {
            factor_kernel::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                n,
                r.x.vector_arg(n),
                tab.arg(),
                extra.unwrap_or(xe.origin()).arg(),
                flags.arg(),
                r.out.vector_arg(n),
                fit(r.x_first_cell / n)?,
                fit(r.x_stride_cells / n)?,
                fit(r.out_first_cell / tile)?,
                fit(r.first_cell)?,
                fit(num_cells - 1)?,
                fit(r.padded_cells / n)?,
                plan.items_per_unit,
                layout,
                f,
            );
        }
    }
    Ok(out)
}
