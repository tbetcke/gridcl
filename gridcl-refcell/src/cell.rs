//! Reference-cell tables A.1–A.3 (docs/design.md, Appendix A).
//!
//! The tables are `const` literals, so they are usable on the host, in `comptime!` blocks and
//! in tests. Sub-entities of each dimension are numbered by their sorted vertex lists in
//! lexicographic order (DefElement Convention 2); a unit test requires the lexicographic
//! generator in `generator.rs` to reproduce every table, and gridcl-conformance checks them
//! against ndelement.

use gridcl_dims::CellType;

use CellType::{Hexahedron, Interval, Point, Prism, Pyramid, Quadrilateral, Tetrahedron, Triangle};

/// The topological dimension of `cell`: 0 for a point, up to 3 for volume cells.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// assert_eq!(gridcl_refcell::tdim(CellType::Point), 0);
/// assert_eq!(gridcl_refcell::tdim(CellType::Prism), 3);
/// ```
pub const fn tdim(cell: CellType) -> usize {
    match cell {
        Point => 0,
        Interval => 1,
        Triangle | Quadrilateral => 2,
        Tetrahedron | Hexahedron | Prism | Pyramid => 3,
    }
}

/// The vertex coordinates of the reference cell (A.1), in local vertex order.
///
/// Each vertex has [`tdim`] coordinates; the point cell has one vertex with none.
/// Quadrilaterals and hexahedra use tensor (lexicographic) order.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// let v = gridcl_refcell::vertices(CellType::Quadrilateral);
/// assert_eq!(v, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
/// ```
pub const fn vertices(cell: CellType) -> &'static [&'static [f64]] {
    match cell {
        Point => &[&[]],
        Interval => &[&[0.0], &[1.0]],
        Triangle => &[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0]],
        Quadrilateral => &[&[0.0, 0.0], &[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]],
        Tetrahedron => &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ],
        Hexahedron => &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
        ],
        Prism => &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
            &[1.0, 0.0, 1.0],
            &[0.0, 1.0, 1.0],
        ],
        Pyramid => &[
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            &[1.0, 1.0, 0.0],
            &[0.0, 0.0, 1.0],
        ],
    }
}

/// The local vertex lists of the sub-entities of dimension `dim` (A.2, A.3), numbered by
/// DefElement Convention 2: sorted vertex lists in lexicographic order.
///
/// Dimension 0 gives the vertices as one-element lists, dimension [`tdim`] the cell itself.
/// Quadrilateral faces list their vertices in the face's tensor order, which is their sorted
/// order. Empty if `dim > tdim`.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::{sub_entities, sub_entity_types};
///
/// // Triangle edge i is opposite vertex 2 − i.
/// assert_eq!(sub_entities(CellType::Triangle, 1), [[0, 1], [0, 2], [1, 2]]);
/// // Prism faces are mixed.
/// assert_eq!(sub_entities(CellType::Prism, 2)[1], [0, 1, 3, 4]);
/// assert_eq!(sub_entity_types(CellType::Prism, 2)[1], CellType::Quadrilateral);
/// ```
pub const fn sub_entities(cell: CellType, dim: usize) -> &'static [&'static [usize]] {
    match (cell, dim) {
        (Point, 0) => &[&[0]],
        (Interval, 0) => &[&[0], &[1]],
        (Interval, 1) => &[&[0, 1]],
        (Triangle, 0) => &[&[0], &[1], &[2]],
        (Triangle, 1) => &[&[0, 1], &[0, 2], &[1, 2]],
        (Triangle, 2) => &[&[0, 1, 2]],
        (Quadrilateral, 0) => &[&[0], &[1], &[2], &[3]],
        (Quadrilateral, 1) => &[&[0, 1], &[0, 2], &[1, 3], &[2, 3]],
        (Quadrilateral, 2) => &[&[0, 1, 2, 3]],
        (Tetrahedron, 0) => &[&[0], &[1], &[2], &[3]],
        (Tetrahedron, 1) => &[&[0, 1], &[0, 2], &[0, 3], &[1, 2], &[1, 3], &[2, 3]],
        (Tetrahedron, 2) => &[&[0, 1, 2], &[0, 1, 3], &[0, 2, 3], &[1, 2, 3]],
        (Tetrahedron, 3) => &[&[0, 1, 2, 3]],
        (Hexahedron, 0) => &[&[0], &[1], &[2], &[3], &[4], &[5], &[6], &[7]],
        (Hexahedron, 1) => &[
            &[0, 1],
            &[0, 2],
            &[0, 4],
            &[1, 3],
            &[1, 5],
            &[2, 3],
            &[2, 6],
            &[3, 7],
            &[4, 5],
            &[4, 6],
            &[5, 7],
            &[6, 7],
        ],
        (Hexahedron, 2) => &[
            &[0, 1, 2, 3],
            &[0, 1, 4, 5],
            &[0, 2, 4, 6],
            &[1, 3, 5, 7],
            &[2, 3, 6, 7],
            &[4, 5, 6, 7],
        ],
        (Hexahedron, 3) => &[&[0, 1, 2, 3, 4, 5, 6, 7]],
        (Prism, 0) => &[&[0], &[1], &[2], &[3], &[4], &[5]],
        (Prism, 1) => &[
            &[0, 1],
            &[0, 2],
            &[0, 3],
            &[1, 2],
            &[1, 4],
            &[2, 5],
            &[3, 4],
            &[3, 5],
            &[4, 5],
        ],
        (Prism, 2) => &[
            &[0, 1, 2],
            &[0, 1, 3, 4],
            &[0, 2, 3, 5],
            &[1, 2, 4, 5],
            &[3, 4, 5],
        ],
        (Prism, 3) => &[&[0, 1, 2, 3, 4, 5]],
        (Pyramid, 0) => &[&[0], &[1], &[2], &[3], &[4]],
        (Pyramid, 1) => &[
            &[0, 1],
            &[0, 2],
            &[0, 4],
            &[1, 3],
            &[1, 4],
            &[2, 3],
            &[2, 4],
            &[3, 4],
        ],
        (Pyramid, 2) => &[
            &[0, 1, 2, 3],
            &[0, 1, 4],
            &[0, 2, 4],
            &[1, 3, 4],
            &[2, 3, 4],
        ],
        (Pyramid, 3) => &[&[0, 1, 2, 3, 4]],
        _ => &[],
    }
}

/// The cell types of the sub-entities of dimension `dim`, in the order of [`sub_entities`].
/// Empty if `dim > tdim`.
pub const fn sub_entity_types(cell: CellType, dim: usize) -> &'static [CellType] {
    match (cell, dim) {
        (_, 0) => match cell {
            Point => &[Point],
            Interval => &[Point; 2],
            Triangle => &[Point; 3],
            Quadrilateral | Tetrahedron => &[Point; 4],
            Pyramid => &[Point; 5],
            Prism => &[Point; 6],
            Hexahedron => &[Point; 8],
        },
        (Interval, 1) => &[Interval],
        (Triangle, 1) => &[Interval; 3],
        (Quadrilateral, 1) => &[Interval; 4],
        (Tetrahedron, 1) => &[Interval; 6],
        (Pyramid, 1) => &[Interval; 8],
        (Prism, 1) => &[Interval; 9],
        (Hexahedron, 1) => &[Interval; 12],
        (Triangle, 2) => &[Triangle],
        (Quadrilateral, 2) => &[Quadrilateral],
        (Tetrahedron, 2) => &[Triangle; 4],
        (Hexahedron, 2) => &[Quadrilateral; 6],
        (Prism, 2) => &[
            Triangle,
            Quadrilateral,
            Quadrilateral,
            Quadrilateral,
            Triangle,
        ],
        (Pyramid, 2) => &[Quadrilateral, Triangle, Triangle, Triangle, Triangle],
        (Tetrahedron, 3) => &[Tetrahedron],
        (Hexahedron, 3) => &[Hexahedron],
        (Prism, 3) => &[Prism],
        (Pyramid, 3) => &[Pyramid],
        _ => &[],
    }
}

/// The number of sub-entities of dimension `dim`; 0 if `dim > tdim`.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// let counts: Vec<usize> =
///     (0..4).map(|d| gridcl_refcell::num_sub_entities(CellType::Pyramid, d)).collect();
/// assert_eq!(counts, [5, 8, 5, 1]);
/// ```
pub const fn num_sub_entities(cell: CellType, dim: usize) -> usize {
    sub_entities(cell, dim).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CELLS: [CellType; 8] = [
        Point,
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ];

    #[test]
    fn tables_are_consistent() {
        for cell in CELLS {
            let td = tdim(cell);
            assert!(vertices(cell).iter().all(|v| v.len() == td), "{cell:?}");
            assert_eq!(num_sub_entities(cell, 0), vertices(cell).len(), "{cell:?}");
            assert_eq!(
                sub_entities(cell, td),
                [(0..vertices(cell).len()).collect::<Vec<_>>()]
            );
            assert_eq!(sub_entity_types(cell, td), [cell]);
            for dim in 0..=3 {
                assert_eq!(
                    sub_entity_types(cell, dim).len(),
                    num_sub_entities(cell, dim)
                );
                for (e, t) in sub_entities(cell, dim)
                    .iter()
                    .zip(sub_entity_types(cell, dim))
                {
                    assert_eq!(tdim(*t), dim, "{cell:?}");
                    assert_eq!(e.len(), vertices(*t).len(), "{cell:?}, {e:?}");
                    assert!(e.is_sorted(), "{cell:?}, {e:?}");
                }
                assert!(
                    sub_entities(cell, dim).is_sorted(),
                    "{cell:?}, dimension {dim}"
                );
            }
            assert!(sub_entities(cell, td + 1).is_empty());
        }
    }

    /// A.2 and A.3: triangle edge i is opposite vertex 2 − i; tetrahedron face i is opposite
    /// vertex 3 − i.
    #[test]
    fn simplex_sub_entities_are_opposite_vertices() {
        for (i, edge) in sub_entities(Triangle, 1).iter().enumerate() {
            assert!(!edge.contains(&(2 - i)), "edge {i}");
        }
        for (i, face) in sub_entities(Tetrahedron, 2).iter().enumerate() {
            assert!(!face.contains(&(3 - i)), "face {i}");
        }
    }

    /// A.3: hexahedron faces 0–5 lie on z = 0, y = 0, x = 0, x = 1, y = 1, z = 1.
    #[test]
    fn hexahedron_face_planes() {
        let planes = [(2, 0.0), (1, 0.0), (0, 0.0), (0, 1.0), (1, 1.0), (2, 1.0)];
        for (face, (axis, value)) in sub_entities(Hexahedron, 2).iter().zip(planes) {
            for &v in *face {
                assert_eq!(vertices(Hexahedron)[v][axis], value, "face {face:?}");
            }
        }
    }

    /// Quadrilateral faces are in tensor order: the first vertex and the last are opposite, so
    /// the boundary cycle is 0–1–3–2 in face-local slots.
    #[test]
    fn quadrilateral_faces_are_tensor_ordered() {
        for cell in [Quadrilateral, Hexahedron, Prism, Pyramid] {
            for (face, t) in sub_entities(cell, 2).iter().zip(sub_entity_types(cell, 2)) {
                if *t != Quadrilateral {
                    continue;
                }
                let x = |slot: usize| vertices(cell)[face[slot]];
                // v3 − v2 = v1 − v0 for a parallelogram in tensor order.
                for d in 0..3.min(x(0).len()) {
                    assert_eq!(
                        x(3)[d] - x(2)[d],
                        x(1)[d] - x(0)[d],
                        "{cell:?} face {face:?}"
                    );
                }
            }
        }
    }
}
