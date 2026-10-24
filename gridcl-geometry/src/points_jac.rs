//! The physical-points and Jacobian kernels and their launchers (Listing B5).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, JacShape, PointSet, QuadField, VecShape};

/// Physical points X at the points of `pts` for every local cell of block `b`, with the origin
/// added back.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::physical_points_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let x = physical_points_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // The one Gauss point of the unit square is its centre.
/// let [x0, y0] = x.to_host_cell_major()?[0];
/// assert!((x0 - 0.5).abs() < 1e-6 && (y0 - 0.5).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
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
    launch_factor(g, b, pts, Factor::Points, None)
}

/// The Jacobian J (GD × TD, column-major) at the points of `pts` for every local cell of block
/// `b` (Listing B5).
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::jacobians_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let jac = jacobians_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // Squares of side 1/2: J = I / 2, column-major.
/// for j in jac.to_host_cell_major()? {
///     assert!((j[0][0] - 0.5).abs() < 1e-6 && (j[1][1] - 0.5).abs() < 1e-6);
///     assert!(j[0][1].abs() < 1e-6 && j[1][0].abs() < 1e-6);
/// }
/// # Ok::<(), GridError>(())
/// ```
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
    launch_factor(g, b, pts, Factor::Jacobian, None)
}
