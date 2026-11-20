//! Facet normals (codimension 0) and normals (codimension 1).

use tetraloma_device::{Real, Result};
use tetraloma_dims::{Codim0, Codim1, Dim, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, FacetPointSet, PointSet, QuadField, VecShape};

// Design: "Geometric factors", Listing B5.
/// The outward unit normals n = Kᵀ n̂ / |Kᵀ n̂| at the facet points of `pts`, facet-major, for
/// every local cell of block `b` of `g`, a block that fills its space (codimension 0).
///
/// n̂ is the outward unit normal of the reference cell on the point's facet and K = J⁻¹. The
/// normal points out of the cell whatever the sign of det J ([Facet normals]).
/// `Mesh::facet_normals` calls it for every block.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::facet_normals_block;
///
/// // `mesh`: the unit square in one quadrilateral, on the test device; the tutorial
/// // "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = FacetQuadRule::for_grid(&mesh, 1)?;
/// let n = facet_normals_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // One point per side of the unit square: the four outward axis directions. The copy to
/// // the host is for inspection.
/// let mut n: Vec<[i32; 2]> = n
///     .to_host_cell_major()?
///     .iter()
///     .map(|v| [v[0].round() as i32, v[1].round() as i32])
///     .collect();
/// n.sort();
/// assert_eq!(n, [[-1, 0], [0, -1], [0, 1], [1, 0]]);
/// # Ok::<(), GridError>(())
/// ```
///
/// [Facet normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#facet-normals
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
    launch_factor(
        g,
        b,
        pts.points(),
        Factor::FacetNormals,
        Some(pts.normals_device()),
    )
}

// Design: "Geometric factors", "Manifold orientation", Listing B5.
/// The unit normals at the points of `pts` for every local cell of block `b` of `g`, a block
/// one dimension lower than its space (a surface in 3D or a curve in 2D).
///
/// n = s (J₀ × J₁) / |J₀ × J₁| on a surface in 3D and n = s (J₁₀, −J₀₀) / |J₀| on a curve in
/// 2D, where J_d is column d of the Jacobian and s = −1 where bit 0 of the cell's
/// `CellBlock::flags` is set, +1 otherwise ([Normals]). `Mesh::orient` sets the flags so that
/// neighbouring normals agree ([Manifold orientation]); `Mesh::normals` calls this function
/// for every block.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
/// use tetraloma::{normals_block, physical_points_block};
///
/// // `mesh`: an icosahedron, 20 flat triangles around the origin, on the test device; the
/// // tutorial "Building a mesh" shows how to make a mesh.
/// # let client = tetraloma_testing::test_client();
/// # let mesh = tetraloma_testing::sphere(0, 1).build::<f32, D2, D3>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let pts = rule.get(Cell2::Triangle)?;
/// // The copies to the host are for inspection.
/// let n = normals_block(&mesh, 0, pts)?.to_host_cell_major()?;
/// let x = physical_points_block(&mesh, 0, pts)?.to_host_cell_major()?;
/// assert_eq!(n.len(), 20);
/// // The normal at a face's centroid is a unit vector along the centroid.
/// for (n, x) in n.iter().zip(&x) {
///     let dot: f32 = (0..3).map(|i| n[i] * x[i]).sum();
///     let len = x.iter().map(|c| c * c).sum::<f32>().sqrt();
///     assert!(((dot / len).abs() - 1.0).abs() < 1e-5);
/// }
/// # Ok::<(), GridError>(())
/// ```
///
/// [Normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#normals
/// [Manifold orientation]: https://tetracl.github.io/tetraloma/concepts/orientation.html#manifold-orientation
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
    launch_factor(g, b, pts, Factor::Normals, None)
}
