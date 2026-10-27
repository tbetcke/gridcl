//! The VTK cell types of A.8.

use gridcl_dims::CellType;

/// The VTK cell type gridcl writes for `cell` of degree `degree` (A.8): the linear type for
/// p = 1 or `lagrange == false`, else the Lagrange type (68–73). Pyramids of degree p ≥ 2 give
/// 14: they are written from their vertices (ADR-0024).
///
/// | Cell | Linear | Lagrange (p ≥ 2) |
/// | --- | --- | --- |
/// | Point | 1 | 1 |
/// | Interval | 3 | 68 |
/// | Triangle | 5 | 69 |
/// | Quadrilateral | 9 | 70 |
/// | Tetrahedron | 10 | 71 |
/// | Hexahedron | 12 | 72 |
/// | Prism | 13 | 73 |
/// | Pyramid | 14 | 14 (vertices only) |
///
/// # Panics
///
/// If `degree == 0`: degrees start at 1. [`to_vtu`](super::to_vtu) reports a block of degree 0
/// as `UnsupportedDegree` before it gets here.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::vtk::vtk_cell_type;
///
/// assert_eq!(vtk_cell_type(CellType::Hexahedron, 1, true), 12);
/// assert_eq!(vtk_cell_type(CellType::Hexahedron, 2, true), 72);
/// assert_eq!(vtk_cell_type(CellType::Hexahedron, 2, false), 12);
/// // VTK has no usable Lagrange pyramid: high-order pyramids are written from their vertices.
/// assert_eq!(vtk_cell_type(CellType::Pyramid, 3, true), 14);
/// ```
pub fn vtk_cell_type(cell: CellType, degree: u32, lagrange: bool) -> u8 {
    assert!(
        degree >= 1,
        "{cell:?} cells of degree 0: degrees start at 1"
    );
    let (linear, high_order) = match cell {
        CellType::Point => (1, 1),
        CellType::Interval => (3, 68),
        CellType::Triangle => (5, 69),
        CellType::Quadrilateral => (9, 70),
        CellType::Tetrahedron => (10, 71),
        CellType::Hexahedron => (12, 72),
        CellType::Prism => (13, 73),
        CellType::Pyramid => (14, 14),
    };
    if lagrange && degree >= 2 {
        high_order
    } else {
        linear
    }
}

/// Whether a cell of `cell` and `degree` is written from its vertices only, although its
/// degree is 2 or more: with `lagrange == false`, and for pyramids (A.8, ADR-0024).
pub(crate) fn written_linear(cell: CellType, degree: u32, lagrange: bool) -> bool {
    !lagrange || degree == 1 || cell == CellType::Pyramid
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of A.8: (cell, linear type, the writer's type for p ≥ 2), written out from the
    /// table.
    const A8: [(CellType, u8, u8); 8] = [
        (CellType::Point, 1, 1),
        (CellType::Interval, 3, 68),
        (CellType::Triangle, 5, 69),
        (CellType::Quadrilateral, 9, 70),
        (CellType::Tetrahedron, 10, 71),
        (CellType::Hexahedron, 12, 72),
        (CellType::Prism, 13, 73),
        (CellType::Pyramid, 14, 14),
    ];

    #[test]
    fn vtk_cell_type_every_a8_row() {
        for (cell, linear, high_order) in A8 {
            for lagrange in [false, true] {
                assert_eq!(vtk_cell_type(cell, 1, lagrange), linear, "{cell:?} p = 1");
            }
            for p in 2..=6 {
                assert_eq!(vtk_cell_type(cell, p, false), linear, "{cell:?} p = {p}");
                assert_eq!(vtk_cell_type(cell, p, true), high_order, "{cell:?} p = {p}");
            }
            assert_eq!(vtk_cell_type(cell, u32::MAX, true), high_order);
        }
    }

    #[test]
    fn vtk_cell_type_linear_means_corners() {
        for (cell, linear, _) in A8 {
            for p in 1..=4 {
                for lagrange in [false, true] {
                    let ty = vtk_cell_type(cell, p, lagrange);
                    // Point cells are type 1 at every degree, but have no other nodes to drop.
                    if cell != CellType::Point {
                        assert_eq!(
                            written_linear(cell, p, lagrange),
                            ty == linear,
                            "{cell:?} p = {p}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "degree 0")]
    fn vtk_cell_type_degree_zero_panics() {
        vtk_cell_type(CellType::Triangle, 0, true);
    }
}
