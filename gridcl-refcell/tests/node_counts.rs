//! M1b acceptance: node counts match A.4 for p ≤ 10.

use gridcl_dims::CellType;
use gridcl_refcell::{
    NodeFamily, lagrange_points, num_interior_lagrange_nodes, num_lagrange_nodes, tdim,
};

const CELLS: [CellType; 8] = [
    CellType::Point,
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

/// A.4: (N(p), interior nodes); the point cell has one node and no interior
/// (docs/plans/M1b.md, question 12).
fn a4(cell: CellType, p: i64) -> (i64, i64) {
    let q = p - 1;
    match cell {
        CellType::Point => (1, 0),
        CellType::Interval => (p + 1, q),
        CellType::Triangle => ((p + 1) * (p + 2) / 2, q * (p - 2) / 2),
        CellType::Quadrilateral => ((p + 1).pow(2), q.pow(2)),
        CellType::Tetrahedron => ((p + 1) * (p + 2) * (p + 3) / 6, q * (p - 2) * (p - 3) / 6),
        CellType::Hexahedron => ((p + 1).pow(3), q.pow(3)),
        CellType::Prism => ((p + 1).pow(2) * (p + 2) / 2, q.pow(2) * (p - 2) / 2),
        CellType::Pyramid => (
            (p + 1) * (p + 2) * (2 * p + 3) / 6,
            q * (p - 2) * (2 * p - 3) / 6,
        ),
    }
}

#[test]
fn node_counts_match_a4() {
    for cell in CELLS {
        for p in 1..=10u32 {
            let (n, interior) = a4(cell, i64::from(p));
            // A.4's interior formulas are non-negative for every p ≥ 1.
            let (n, interior) = (n as usize, interior as usize);
            assert_eq!(num_lagrange_nodes(cell, p), n, "N({p}) of {cell:?}");
            assert_eq!(
                num_interior_lagrange_nodes(cell, p),
                interior,
                "interior({p}) of {cell:?}"
            );

            // The layout has N(p) points; the point cell's single node has no coordinates.
            let td = tdim(cell);
            if td > 0 {
                let points = lagrange_points(cell, p, NodeFamily::Equispaced).unwrap();
                assert_eq!(points.len() % td, 0, "{cell:?}, p = {p}");
                assert_eq!(points.len() / td, n, "lagrange_points of {cell:?}, p = {p}");
            }
        }
    }
}
