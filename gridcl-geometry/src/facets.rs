//! Facet normals (codimension 0) and normals (codimension 1).

use gridcl_device::{Real, Result};
use gridcl_dims::{Codim0, Codim1, Dim, TopoDim};

use crate::layout::{Factor, launch_factor};
use crate::{DeviceGrid, FacetPointSet, PointSet, QuadField, VecShape};

/// Outward unit normals n = Kᵀ n_ref / |Kᵀ n_ref| at the facet points of `pts` (facet-major)
/// for every local cell of block `b`.
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::facet_normals_block;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
/// let rule = FacetQuadRule::for_grid(&mesh, 1)?;
/// let n = facet_normals_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
/// // One point per side of the unit square: the four outward axis directions.
/// let mut n: Vec<[i32; 2]> = n
///     .to_host_cell_major()?
///     .iter()
///     .map(|v| [v[0].round() as i32, v[1].round() as i32])
///     .collect();
/// n.sort();
/// assert_eq!(n, [[-1, 0], [0, -1], [0, 1], [1, 0]]);
/// # Ok::<(), GridError>(())
/// ```
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

/// Unit normals of a codimension-1 block at the points of `pts`, with the sign s = −1 where
/// bit 0 of `CellBlock::flags` is set: n = s (J0 × J1) / |J0 × J1| for surfaces in 3D and
/// n = s (J0_y, −J0_x) / |J0| for curves in 2D ("Manifold orientation").
///
/// # Errors
///
/// Allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl::{normals_block, physical_points_block};
///
/// let client = gridcl_testing::test_client();
/// // An icosahedron: 20 flat triangles around the origin.
/// let mesh = gridcl_testing::sphere(0, 1).build::<f32, D2, D3>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 1)?;
/// let pts = rule.get(Cell2::Triangle)?;
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
