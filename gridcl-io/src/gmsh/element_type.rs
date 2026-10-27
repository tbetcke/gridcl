//! The Gmsh element table (A.7), generated from A.7's type-id lists and gridcl-refcell's node
//! counts.

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;

use super::FORMAT;

/// A Gmsh element type of A.7.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::element_type;
///
/// let ty = element_type(118)?;
/// assert_eq!((ty.cell, ty.degree, ty.num_nodes, ty.serendipity), (CellType::Pyramid, 3, 30, false));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElementType {
    /// The Gmsh type id.
    pub id: i32,
    /// The cell type.
    pub cell: CellType,
    /// The degree (2 for the serendipity types; 1 for the point).
    pub degree: u32,
    /// The number of nodes per element.
    pub num_nodes: usize,
    /// Whether this is a serendipity type (16–19), upgraded to complete Lagrange on read.
    pub serendipity: bool,
}

/// The complete Lagrange type ids of A.7, degree 1 first, per cell type.
const LAGRANGE: [(CellType, &[i32]); 8] = [
    (CellType::Point, &[15]),
    (CellType::Interval, &[1, 8, 26, 27, 28, 62, 63, 64, 65, 66]),
    (CellType::Triangle, &[2, 9, 21, 23, 25, 42, 43, 44, 45, 46]),
    (
        CellType::Quadrilateral,
        &[3, 10, 36, 37, 38, 47, 48, 49, 50, 51],
    ),
    (
        CellType::Tetrahedron,
        &[4, 11, 29, 30, 31, 71, 72, 73, 74, 75],
    ),
    (CellType::Hexahedron, &[5, 12, 92, 93, 94, 95, 96, 97, 98]),
    (CellType::Prism, &[6, 13, 90, 91, 106, 107, 108, 109, 110]),
    (
        CellType::Pyramid,
        &[7, 14, 118, 119, 120, 121, 122, 123, 124],
    ),
];

/// The serendipity types of A.7: (id, cell, node count).
const SERENDIPITY: [(i32, CellType, usize); 4] = [
    (16, CellType::Quadrilateral, 8),
    (17, CellType::Hexahedron, 20),
    (18, CellType::Prism, 15),
    (19, CellType::Pyramid, 13),
];

/// The name of a cell type in errors.
pub(crate) fn cell_name(cell: CellType) -> &'static str {
    match cell {
        CellType::Point => "Point",
        CellType::Interval => "Interval",
        CellType::Triangle => "Triangle",
        CellType::Quadrilateral => "Quadrilateral",
        CellType::Tetrahedron => "Tetrahedron",
        CellType::Hexahedron => "Hexahedron",
        CellType::Prism => "Prism",
        CellType::Pyramid => "Pyramid",
    }
}

/// The complete Lagrange type `ids[degree - 1]` of `cell`.
fn lagrange(cell: CellType, ids: &[i32], degree: u32) -> Option<ElementType> {
    let id = *ids.get(usize::try_from(degree).ok()?.checked_sub(1)?)?;
    Some(ElementType {
        id,
        cell,
        degree,
        num_nodes: gridcl_refcell::num_lagrange_nodes(cell, degree),
        serendipity: false,
    })
}

/// The element type of Gmsh type id `id`.
///
/// # Errors
///
/// `UnsupportedElementType` for every id not in A.7, among them Gmsh's incomplete triangles,
/// tetrahedra, quadrilaterals, hexahedra and prisms and the incomplete pyramids 125–131.
///
/// # Example
///
/// ```
/// use gridcl_device::GridError;
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::element_type;
///
/// let ty = element_type(17)?;
/// assert_eq!((ty.cell, ty.degree, ty.num_nodes, ty.serendipity), (CellType::Hexahedron, 2, 20, true));
/// assert!(matches!(element_type(125), Err(GridError::UnsupportedElementType { type_id: 125, .. })));
/// # Ok::<(), GridError>(())
/// ```
pub fn element_type(id: i32) -> Result<ElementType> {
    for (cell, ids) in LAGRANGE {
        if let Some(k) = ids.iter().position(|&x| x == id) {
            let degree = u32::try_from(k + 1).expect("at most ten degrees");
            return Ok(lagrange(cell, ids, degree).expect("an A.7 degree"));
        }
    }
    SERENDIPITY
        .iter()
        .find(|s| s.0 == id)
        .map(|&(id, cell, num_nodes)| ElementType {
            id,
            cell,
            degree: 2,
            num_nodes,
            serendipity: true,
        })
        .ok_or(GridError::UnsupportedElementType {
            format: FORMAT,
            type_id: id,
        })
}

/// The complete Lagrange element type of `cell` and `degree`. The point has degree 1 only.
///
/// # Errors
///
/// `UnsupportedDegree` if Gmsh has no complete Lagrange type of this cell and degree.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::gmsh::lagrange_element_type;
///
/// assert_eq!(lagrange_element_type(CellType::Triangle, 2)?.id, 9);
/// assert_eq!(lagrange_element_type(CellType::Hexahedron, 9)?.num_nodes, 1000);
/// assert!(lagrange_element_type(CellType::Hexahedron, 10).is_err());
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn lagrange_element_type(cell: CellType, degree: u32) -> Result<ElementType> {
    let (_, ids) = LAGRANGE
        .iter()
        .find(|(c, _)| *c == cell)
        .expect("every cell type has an A.7 row");
    lagrange(cell, ids, degree).ok_or(GridError::UnsupportedDegree {
        cell: cell_name(cell),
        degree,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every id of A.7 with its node count, as A.7 lists it.
    const A7: [(i32, usize); 63] = [
        (15, 1),
        (1, 2),
        (8, 3),
        (26, 4),
        (27, 5),
        (28, 6),
        (62, 7),
        (63, 8),
        (64, 9),
        (65, 10),
        (66, 11),
        (2, 3),
        (9, 6),
        (21, 10),
        (23, 15),
        (25, 21),
        (42, 28),
        (43, 36),
        (44, 45),
        (45, 55),
        (46, 66),
        (3, 4),
        (10, 9),
        (36, 16),
        (37, 25),
        (38, 36),
        (47, 49),
        (48, 64),
        (49, 81),
        (50, 100),
        (51, 121),
        (16, 8),
        (4, 4),
        (11, 10),
        (29, 20),
        (30, 35),
        (31, 56),
        (71, 84),
        (72, 120),
        (73, 165),
        (74, 220),
        (75, 286),
        (5, 8),
        (12, 27),
        (92, 64),
        (93, 125),
        (94, 216),
        (95, 343),
        (96, 512),
        (97, 729),
        (98, 1000),
        (17, 20),
        (6, 6),
        (13, 18),
        (90, 40),
        (91, 75),
        (106, 126),
        (107, 196),
        (108, 288),
        (109, 405),
        (110, 550),
        (18, 15),
        (7, 5),
    ];

    /// The pyramid rows of A.7 beyond the linear one.
    const A7_PYRAMID: [(i32, usize); 9] = [
        (14, 14),
        (118, 30),
        (119, 55),
        (120, 91),
        (121, 140),
        (122, 204),
        (123, 285),
        (124, 385),
        (19, 13),
    ];

    #[test]
    fn element_type_every_a7_id_round_trips() {
        for &(id, nodes) in A7.iter().chain(&A7_PYRAMID) {
            let ty = element_type(id).unwrap();
            assert_eq!((ty.id, ty.num_nodes), (id, nodes), "type {id}");
            if ty.serendipity {
                assert_eq!(ty.degree, 2, "type {id}");
            } else {
                assert_eq!(
                    ty.num_nodes,
                    gridcl_refcell::num_lagrange_nodes(ty.cell, ty.degree),
                    "type {id}"
                );
                assert_eq!(lagrange_element_type(ty.cell, ty.degree).unwrap(), ty);
            }
        }
        assert_eq!(A7.len() + A7_PYRAMID.len(), 72);
    }

    #[test]
    fn element_type_rejects_other_ids() {
        let known: Vec<i32> = A7.iter().chain(&A7_PYRAMID).map(|t| t.0).collect();
        for id in -1..=200 {
            if !known.contains(&id) {
                assert!(
                    matches!(
                        element_type(id),
                        Err(GridError::UnsupportedElementType { format: "Gmsh", type_id }) if type_id == id
                    ),
                    "type {id}"
                );
            }
        }
        // Incomplete triangles, the incomplete pyramids 125-131 and the other incomplete types.
        for id in [
            20, 22, 24, 125, 126, 127, 128, 129, 130, 131, 32, 33, 39, 99, 111,
        ] {
            assert!(element_type(id).is_err(), "type {id}");
        }
    }

    #[test]
    fn element_type_lagrange_degrees() {
        for (cell, ids) in LAGRANGE {
            for (k, &id) in ids.iter().enumerate() {
                let ty = lagrange_element_type(cell, k as u32 + 1).unwrap();
                assert_eq!(ty.id, id);
            }
            let past = ids.len() as u32 + 1;
            assert!(matches!(
                lagrange_element_type(cell, past),
                Err(GridError::UnsupportedDegree { degree, .. }) if degree == past
            ));
            assert!(lagrange_element_type(cell, 0).is_err());
        }
    }
}
