//! The det J and pseudo-inverse kernels; the path is chosen at comptime per (TD, GD).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, PinvShape, PointSet, QuadField, ScalarShape};

/// det J, or the pseudo-determinant sqrt(det(JᵀJ)) when TD < GD, at the points of `pts` for
/// every local cell of block `b`.
///
/// # Errors
///
/// Allocation and device errors.
pub fn detj_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
) -> Result<QuadField<F, ScalarShape>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    launch_factor(g, b, pts, Factor::DetJ)
}

/// K = (JᵀJ)⁻¹Jᵀ (TD × GD, column-major), which equals J⁻¹ in codimension 0, at the points of
/// `pts` for every local cell of block `b`.
///
/// # Errors
///
/// Allocation and device errors.
pub fn pseudo_inverse_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
) -> Result<QuadField<F, PinvShape<TD, GD>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    launch_factor(g, b, pts, Factor::Pinv)
}
