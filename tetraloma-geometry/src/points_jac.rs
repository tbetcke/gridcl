//! The physical-points and Jacobian kernels and their launchers (Listing B5).

use tetraloma_device::{Real, Result};
use tetraloma_dims::{Dim, EmbeddedIn, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, JacShape, PointSet, QuadField, VecShape};

// Design: "Geometric factors", Listing B5.
/// The physical points X(ξ_q) of the reference points ξ_q of `pts`, for every local cell of
/// block `b` of `g`: GD coordinates per cell and point, absolute (the origin is added back).
///
/// X(ξ) = Σ_k x_k φ_k(ξ) is the map of the cell, with x_k its nodes and φ_k the basis of its
/// coordinate element ([The reference-to-physical map]). `Mesh::physical_points` calls it for
/// every block ([Points]).
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::physical_points_block;
///
/// // `mesh`: the unit square in one quadrilateral, on the test device; the tutorial
/// // "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let x = physical_points_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // The one Gauss point of the unit square is its centre. The copy to the host is for
/// // inspection.
/// let [x0, y0] = x.to_host_cell_major()?[0];
/// assert!((x0 - 0.5).abs() < 1e-6 && (y0 - 0.5).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
///
/// [The reference-to-physical map]: https://tetracl.github.io/tetraloma/concepts/reference-cells.html#the-reference-to-physical-map
/// [Points]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#points
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

// Design: "Geometric factors", Listing B5.
/// The Jacobian J at the points of `pts` for every local cell of block `b` of `g`: a GD × TD
/// matrix per point, column-major.
///
/// J_γd = ∂X_γ/∂ξ_d = Σ_k x_k,γ ∂φ_k/∂ξ_d; column d is the tangent vector of the cell along
/// reference direction d ([The Jacobian]). `Mesh::jacobians` calls it for every block
/// ([Jacobians]).
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::jacobians_block;
///
/// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
/// // "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let jac = jacobians_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // Squares of side 1/2: J = I / 2, column-major. The copy to the host is for inspection.
/// for j in jac.to_host_cell_major()? {
///     assert!((j[0][0] - 0.5).abs() < 1e-6 && (j[1][1] - 0.5).abs() < 1e-6);
///     assert!(j[0][1].abs() < 1e-6 && j[1][0].abs() < 1e-6);
/// }
/// # Ok::<(), GridError>(())
/// ```
///
/// [The Jacobian]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-jacobian
/// [Jacobians]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#jacobians
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
