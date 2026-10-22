//! Facet normals (codimension 0) and normals (codimension 1).

use gridcl_device::{Real, Result};
use gridcl_dims::{Codim0, Codim1, Dim, TopoDim};

use crate::{DeviceGrid, FacetPointSet, PointSet, QuadField, VecShape};

/// Outward unit normals n = Kᵀ n_ref / |Kᵀ n_ref| at the facet points of `pts` (facet-major)
/// for every local cell of block `b`.
///
/// # Errors
///
/// Allocation and device errors.
pub fn facet_normals_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &FacetPointSet<F, TD>,
) -> Result<QuadField<F, VecShape<GD>>>
where
    F: Real,
    TD: TopoDim + Codim0<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let _ = (g, b, pts);
    todo!()
}

/// Unit normals of a codimension-1 block at the points of `pts`, with the sign s = −1 where
/// bit 0 of `CellBlock::flags` is set: n = s (J0 × J1) / |J0 × J1| for surfaces in 3D and
/// n = s (J0_y, −J0_x) / |J0| for curves in 2D ("Manifold orientation").
///
/// # Errors
///
/// Allocation and device errors.
pub fn normals_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
) -> Result<QuadField<F, VecShape<GD>>>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let _ = (g, b, pts);
    todo!()
}
