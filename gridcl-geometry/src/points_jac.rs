//! The physical-points and Jacobian kernels and their launchers (Listing B5).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::{DeviceGrid, JacShape, PointSet, QuadField, VecShape};

/// Physical points X at the points of `pts` for every local cell of block `b`, with the origin
/// added back.
///
/// # Errors
///
/// Allocation and device errors.
pub fn physical_points_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
) -> Result<QuadField<F, VecShape<GD>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let _ = (g, b, pts);
    todo!()
}

/// The Jacobian J (GD × TD, column-major) at the points of `pts` for every local cell of block
/// `b` (Listing B5).
///
/// # Errors
///
/// Allocation and device errors.
pub fn jacobians_block<F, TD, GD, G>(
    g: &G,
    b: usize,
    pts: &PointSet<F, TD>,
) -> Result<QuadField<F, JacShape<TD, GD>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let _ = (g, b, pts);
    todo!()
}
