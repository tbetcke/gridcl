//! The det J and pseudo-inverse kernels; the path is chosen at comptime per (TD, GD).

use tetraloma_device::{Real, Result};
use tetraloma_dims::{Dim, EmbeddedIn, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, PinvShape, PointSet, QuadField, ScalarShape};

// Design: "Geometric factors", Listing B5.
/// The determinant det J at the points of `pts` for every local cell of block `b` of `g`, or
/// the pseudo-determinant √det(JᵀJ) when the cells have a lower dimension than the space (a
/// curve or a surface).
///
/// In codimension 0 the value is signed: negative where the map mirrors the reference cell.
/// It is the volume factor of the change of variables, dx = det J dξ ([The determinant]).
/// `Mesh::detj` calls it for every block ([Determinants]).
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::detj_block;
///
/// // `mesh`: the unit square in two triangles, on the test device; the tutorial
/// // "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 2)?;
/// let detj = detj_block(&mesh, 0, rule.get(Cell2::Triangle)?)?;
/// // Two triangles of area 1/2 on the reference triangle of area 1/2: det J = 1. The copy to
/// // the host is for inspection.
/// assert!(detj.to_host_cell_major()?.iter().all(|d| (d - 1.0).abs() < 1e-5));
/// # Ok::<(), GridError>(())
/// ```
///
/// [The determinant]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-determinant
/// [Determinants]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#determinants
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

// Design: "Geometric factors", Listing B5.
/// The pseudo-inverse K = (JᵀJ)⁻¹Jᵀ of the Jacobian at the points of `pts` for every local
/// cell of block `b` of `g`: a TD × GD matrix per point, column-major.
///
/// In codimension 0, K = J⁻¹. On a curve or surface, K J = I, and the surface gradient of a
/// function u with u(X(ξ)) = û(ξ) is ∇_Γ u = Kᵀ ∇_ξ û ([The pseudo-inverse]).
/// `Mesh::pseudo_inverse_jacobians` calls it for every block ([Pseudo-inverses]).
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::pseudo_inverse_block;
///
/// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
/// // "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let k = pseudo_inverse_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // J = I / 2, so K = J⁻¹ = 2 I. The copy to the host is for inspection.
/// for k in k.to_host_cell_major()? {
///     assert!((k[0][0] - 2.0).abs() < 1e-5 && (k[1][1] - 2.0).abs() < 1e-5);
///     assert!(k[0][1].abs() < 1e-5 && k[1][0].abs() < 1e-5);
/// }
/// # Ok::<(), GridError>(())
/// ```
///
/// [The pseudo-inverse]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-pseudo-inverse
/// [Pseudo-inverses]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#pseudo-inverses
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
