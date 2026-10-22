//! Reference facet normals and facet-to-cell point maps (ADR-0012), for facet quadrature rules
//! and the facet normals of gridcl-geometry.

use gridcl_dims::CellType;

/// The outward unit normal of every facet of `cell`, in facet order: `tdim(cell)` entries per
/// facet, in reference coordinates.
///
/// The facets are the sub-entities of dimension `tdim − 1` (A.2, A.3); an interval's facets
/// are its two vertices.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::reference_facet_normals;
///
/// // Facet 0 of the triangle is the edge (0, 1), on y = 0.
/// let n = reference_facet_normals(CellType::Triangle);
/// assert_eq!(n.len(), 3 * 2);
/// assert_eq!(&n[0..2], &[0.0, -1.0]);
/// ```
///
/// # Panics
///
/// If `cell` is the point cell, which has no facets.
pub fn reference_facet_normals(cell: CellType) -> Vec<f64> {
    let _ = cell;
    todo!()
}

/// Maps `points` on the reference cell of facet `facet` of `cell` into cell coordinates, with
/// A.5's affine rule v0 + Σ t_k (v_k − v0) over the facet's vertex list.
///
/// `points` are flat with `tdim(cell) − 1` coordinates per point; the result has `tdim(cell)`
/// per point. For an interval, whose facets are points, `points` is empty and stands for the
/// facet's one point.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::facet_points_to_cell;
///
/// // Facet 2 of the triangle is the edge (1, 2): t = 0.25 lies at (0.75, 0.25).
/// assert_eq!(facet_points_to_cell(CellType::Triangle, 2, &[0.25]), vec![0.75, 0.25]);
/// ```
///
/// # Panics
///
/// If `facet` is not a facet of `cell`, or `points.len()` is not a multiple of the facet's
/// dimension.
pub fn facet_points_to_cell(cell: CellType, facet: usize, points: &[f64]) -> Vec<f64> {
    let _ = (cell, facet, points);
    todo!()
}
