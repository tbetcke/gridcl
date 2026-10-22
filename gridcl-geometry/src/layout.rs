//! `GeomLayout` (Listing B5).

use cubecl::prelude::*;
use gridcl_device::{
    GridError, KernelKind, LaunchPlan, Real, Result, VectorPolicy, debug_assert_bindings,
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
}

impl Factor {
    /// Components per point.
    pub(crate) fn len(self, l: &GeomLayout) -> usize {
        match self {
            Factor::Points => l.gdim,
            Factor::Jacobian => l.tdim * l.gdim,
        }
    }
}

/// J at point q of the lanes' cells: entry (g, d) at component d · G + g, each a sum over the
/// nodes k ascending from 0 of x(k, g) · ∂_d φ_k(q) (Listing B5); the same operation sequence
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
            let mut acc = Vector::<F, N>::broadcast(F::from_int(0));
            #[unroll]
            for k in 0..l.nodes_per_cell {
                let x = xe[(k * l.gdim + g) * x_stride + x_offset + v];
                let w = Vector::<F, N>::broadcast(dphi[(q * l.nodes_per_cell + k) * l.tdim + d]);
                acc += x * w;
            }
            j[d * l.gdim + g] = acc;
        }
    }
    j
}

/// A geometric factor at every point of the cell vectors `v < num_vectors` of one chunk pair.
/// The lanes are N consecutive cells of one tile (N divides the tile).
///   xe     E-vector chunk [k][g][cell], cells fastest; row (k, g) starts at (k·G + g)·x_stride
///   tab    φ [q][k] (Points) or ∂φ [q][k][d] (all others); broadcast to every lane
///   out    tiled QuadField chunk
///   extra  the origin [gdim] (Points)
/// Offsets and strides count vectors.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn factor_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    tab: &[F],
    extra: &[F],
    out: &mut [Vector<F, N>],
    x_offset: u32,
    x_stride: u32,
    out_first_tile: u32,
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
                            let mut acc = Vector::<F, N>::broadcast(F::from_int(0));
                            #[unroll]
                            for k in 0..l.nodes_per_cell {
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
        debug_assert_bindings(client, 4);
        // SAFETY: n divides the tile, which divides r.padded_cells, VECTOR_PAD and the
        // E-vector stride, so the vector offsets below are exact. The kernel handles vectors
        // v < padded_cells / n: it reads E-vector vectors (k·G + g)·stride/n + x_first_cell/n
        // + v < K·G·stride/n, inside the E-vector chunk (x_first_cell + padded_cells ≤
        // stride); it reads `tab` below num_points · K · tdim and `extra` below gdim; it writes
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
                xe.origin().arg(),
                r.out.vector_arg(n),
                fit(r.x_first_cell / n)?,
                fit(r.x_stride_cells / n)?,
                fit(r.out_first_cell / tile)?,
                fit(r.padded_cells / n)?,
                plan.items_per_unit,
                layout,
                f,
            );
        }
    }
    Ok(out)
}
