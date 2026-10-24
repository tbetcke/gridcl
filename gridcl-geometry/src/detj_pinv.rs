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
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::detj_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 2)?;
/// let detj = detj_block(&mesh, 0, rule.get(Cell2::Triangle)?)?;
/// // Two positive triangles of area 1/2 on the reference triangle of area 1/2.
/// assert!(detj.to_host_cell_major()?.iter().all(|d| (d - 1.0).abs() < 1e-6));
/// # Ok::<(), GridError>(())
/// ```
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
    launch_factor(g, b, pts, Factor::DetJ, None)
}

/// K = (JᵀJ)⁻¹Jᵀ (TD × GD, column-major), which equals J⁻¹ in codimension 0, at the points of
/// `pts` for every local cell of block `b`.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::pseudo_inverse_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let k = pseudo_inverse_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // J = I / 2, so K = J⁻¹ = 2 I.
/// for k in k.to_host_cell_major()? {
///     assert!((k[0][0] - 2.0).abs() < 1e-5 && (k[1][1] - 2.0).abs() < 1e-5);
///     assert!(k[0][1].abs() < 1e-5 && k[1][0].abs() < 1e-5);
/// }
/// # Ok::<(), GridError>(())
/// ```
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
    launch_factor(g, b, pts, Factor::Pinv, None)
}
