//! Reference facet normals and facet-to-cell point maps (ADR-0012), for facet quadrature rules
//! and the facet normals of gridcl-geometry.

use gridcl_dims::CellType;

use crate::{sub_entities, sub_entity_types, tdim, vertices};

/// The centroid of the vertices `list` of `cell`.
fn centroid(cell: CellType, list: &[usize]) -> Vec<f64> {
    let verts = vertices(cell);
    let td = tdim(cell);
    let mut c = vec![0.0; td];
    for &v in list {
        for (ci, x) in c.iter_mut().zip(verts[v]) {
            *ci += x;
        }
    }
    c.iter().map(|x| x / list.len() as f64).collect()
}

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
    let td = tdim(cell);
    assert!(
        td >= 1,
        "reference_facet_normals: the point cell has no facets"
    );
    let verts = vertices(cell);
    let all: Vec<usize> = (0..verts.len()).collect();
    let middle = centroid(cell, &all);
    let mut out = Vec::new();
    for facet in sub_entities(cell, td - 1) {
        let v = |k: usize| verts[facet[k]];
        let diff =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x - y).collect() };
        // A normal of the facet's affine hull, of either sign.
        let mut n = match td {
            1 => vec![1.0],
            2 => {
                let t = diff(v(1), v(0));
                vec![t[1], -t[0]]
            }
            _ => {
                let (a, b) = (diff(v(1), v(0)), diff(v(2), v(0)));
                vec![
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            }
        };
        let outward: f64 = n
            .iter()
            .zip(diff(&centroid(cell, facet), &middle))
            .map(|(a, b)| a * b)
            .sum();
        let len = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if outward > 0.0 { 1.0 / len } else { -1.0 / len };
        n.iter_mut().for_each(|x| *x *= scale);
        out.extend(n);
    }
    out
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
    let td = tdim(cell);
    assert!(
        td >= 1,
        "facet_points_to_cell: the point cell has no facets"
    );
    let facets = sub_entities(cell, td - 1);
    assert!(
        facet < facets.len(),
        "facet_points_to_cell: {cell:?} has {} facets, not {}",
        facets.len(),
        facet + 1
    );
    let list = facets[facet];
    let fd = tdim(sub_entity_types(cell, td - 1)[facet]);
    let verts = vertices(cell);
    let v0 = verts[list[0]];
    if fd == 0 {
        assert!(
            points.is_empty(),
            "facet_points_to_cell: a point facet takes no coordinates"
        );
        return v0.to_vec();
    }
    assert!(
        points.len().is_multiple_of(fd),
        "facet_points_to_cell: {} coordinates are not points of dimension {fd}",
        points.len()
    );
    // The facet's axes run from v0 to its vertices 1..=fd: simplices list them in order, and
    // quadrilaterals in tensor order (v1 − v0 and v2 − v0).
    let mut out = Vec::with_capacity(points.len() / fd * td);
    for t in points.chunks_exact(fd) {
        for i in 0..td {
            let x = v0[i]
                + (0..fd)
                    .map(|k| t[k] * (verts[list[k + 1]][i] - v0[i]))
                    .sum::<f64>();
            out.push(x);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use CellType::*;

    const CELLS: [CellType; 7] = [
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ];

    #[test]
    fn normals_are_unit_and_outward() {
        for cell in CELLS {
            let td = tdim(cell);
            let normals = reference_facet_normals(cell);
            let facets = sub_entities(cell, td - 1);
            assert_eq!(normals.len(), facets.len() * td, "{cell:?}");
            let all: Vec<usize> = (0..vertices(cell).len()).collect();
            let middle = centroid(cell, &all);
            for (f, facet) in facets.iter().enumerate() {
                let n = &normals[f * td..(f + 1) * td];
                let len = n.iter().map(|x| x * x).sum::<f64>().sqrt();
                assert!((len - 1.0).abs() < 1e-15, "{cell:?} facet {f}: |n| = {len}");
                let c = centroid(cell, facet);
                let dot: f64 = n
                    .iter()
                    .zip(c.iter().zip(&middle))
                    .map(|(a, (x, m))| a * (x - m))
                    .sum();
                assert!(dot > 0.0, "{cell:?} facet {f} points inward");
                // Normal to every edge of the facet.
                for &v in facet.iter() {
                    let d: f64 = n
                        .iter()
                        .zip(vertices(cell)[v].iter().zip(vertices(cell)[facet[0]]))
                        .map(|(a, (x, y))| a * (x - y))
                        .sum();
                    assert!(
                        d.abs() < 1e-15,
                        "{cell:?} facet {f} is not normal to its vertices"
                    );
                }
            }
        }
    }

    #[test]
    fn known_normals() {
        let s = 1.0 / 3f64.sqrt();
        assert_eq!(reference_facet_normals(Interval), [-1.0, 1.0]);
        // Tetrahedron face 3 is (1, 2, 3), on x + y + z = 1.
        let n = reference_facet_normals(Tetrahedron);
        for (a, b) in n[9..12].iter().zip([s, s, s]) {
            assert!((a - b).abs() < 1e-15);
        }
        // Hexahedron face 0 is z = 0, face 5 is z = 1.
        let n = reference_facet_normals(Hexahedron);
        assert_eq!(&n[0..3], &[0.0, 0.0, -1.0]);
        assert_eq!(&n[15..18], &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn mapped_points_lie_on_their_facet() {
        for cell in CELLS {
            let td = tdim(cell);
            let normals = reference_facet_normals(cell);
            for (f, (facet, ty)) in sub_entities(cell, td - 1)
                .iter()
                .zip(sub_entity_types(cell, td - 1))
                .enumerate()
            {
                // The facet's vertices map to the cell's vertices.
                let ref_verts: Vec<f64> = vertices(*ty).iter().flat_map(|v| v.to_vec()).collect();
                let mapped = if tdim(*ty) == 0 {
                    facet_points_to_cell(cell, f, &[])
                } else {
                    facet_points_to_cell(cell, f, &ref_verts)
                };
                let want: Vec<f64> = facet
                    .iter()
                    .flat_map(|&v| vertices(cell)[v].to_vec())
                    .collect();
                assert_eq!(mapped, want, "{cell:?} facet {f}: vertices");
                // Gauss points of the facet land on the facet's plane.
                if tdim(*ty) > 0 {
                    let rule = crate::gauss_rule(*ty, 3);
                    let pts = facet_points_to_cell(cell, f, &rule.points);
                    let n = &normals[f * td..(f + 1) * td];
                    let v0 = vertices(cell)[facet[0]];
                    for x in pts.chunks_exact(td) {
                        let d: f64 = n
                            .iter()
                            .zip(x.iter().zip(v0))
                            .map(|(a, (p, q))| a * (p - q))
                            .sum();
                        assert!(d.abs() < 1e-14, "{cell:?} facet {f}: point off the facet");
                    }
                }
            }
        }
    }
}
