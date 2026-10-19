//! M1b acceptance: the generators reproduce every A.9 row and DOLFINx's Gmsh P3 tables
//! (tet20 after A.11; quad16, hex64, prism40 and pyramid30 unchanged).
//!
//! Direction (A.9): entry k is the internal index of the file's k-th node,
//! `internal[map[k]] = file[k]`.

use gridcl_dims::CellType;
use gridcl_refcell::{RefCellError, gmsh_to_internal, vtk_to_internal};

/// The A.9 rows (Gmsh, VTK Lagrange) as (cell, degree, map); "identity" rows written out.
#[rustfmt::skip]
fn a9_gmsh() -> Vec<(CellType, u32, Vec<usize>)> {
    use CellType::*;
    vec![
        (Interval, 1, vec![0, 1]),
        (Interval, 2, vec![0, 1, 2]),
        (Triangle, 1, vec![0, 1, 2]),
        (Triangle, 2, vec![0, 1, 2, 3, 5, 4]),
        (Quadrilateral, 1, vec![0, 1, 3, 2]),
        (Quadrilateral, 2, vec![0, 1, 3, 2, 4, 6, 7, 5, 8]),
        (Tetrahedron, 1, vec![0, 1, 2, 3]),
        (Tetrahedron, 2, vec![0, 1, 2, 3, 4, 7, 5, 6, 9, 8]),
        (Hexahedron, 1, vec![0, 1, 3, 2, 4, 5, 7, 6]),
        (Hexahedron, 2, vec![0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 15, 14, 16, 17, 18, 19,
                             20, 21, 22, 23, 24, 25, 26]),
        (Prism, 1, (0..6).collect()),
        (Prism, 2, (0..18).collect()),
        (Pyramid, 1, vec![0, 1, 3, 2, 4]),
        (Pyramid, 2, vec![0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13]),
    ]
}

#[rustfmt::skip]
fn a9_vtk() -> Vec<(CellType, u32, Vec<usize>)> {
    use CellType::*;
    vec![
        (Interval, 1, vec![0, 1]),
        (Interval, 2, vec![0, 1, 2]),
        (Triangle, 1, vec![0, 1, 2]),
        (Triangle, 2, vec![0, 1, 2, 3, 5, 4]),
        (Quadrilateral, 1, vec![0, 1, 3, 2]),
        (Quadrilateral, 2, vec![0, 1, 3, 2, 4, 6, 7, 5, 8]),
        (Tetrahedron, 1, vec![0, 1, 2, 3]),
        (Tetrahedron, 2, vec![0, 1, 2, 3, 4, 7, 5, 6, 8, 9]),
        (Hexahedron, 1, vec![0, 1, 3, 2, 4, 5, 7, 6]),
        (Hexahedron, 2, vec![0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14,
                             22, 23, 21, 24, 20, 25, 26]),
        (Prism, 1, (0..6).collect()),
        (Prism, 2, vec![0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16]),
        (Pyramid, 1, vec![0, 1, 3, 2, 4]),
    ]
}

#[test]
fn gmsh_reproduces_a9() {
    for (cell, degree, want) in a9_gmsh() {
        assert_eq!(
            gmsh_to_internal(cell, degree).unwrap(),
            want,
            "Gmsh {cell:?}, p = {degree}"
        );
    }
}

#[test]
fn vtk_reproduces_a9() {
    for (cell, degree, want) in a9_vtk() {
        assert_eq!(
            vtk_to_internal(cell, degree).unwrap(),
            want,
            "VTK {cell:?}, p = {degree}"
        );
    }
    // A.9 "Pyramid (14): not written (A.8)".
    assert_eq!(
        vtk_to_internal(CellType::Pyramid, 2),
        Err(RefCellError::NotInFormat {
            format: "VTK",
            cell: CellType::Pyramid,
            degree: 2
        })
    );
}

// DOLFINx `cpp/dolfinx/io/cells.cpp`, functions gmsh_tetrahedron, gmsh_quadrilateral,
// gmsh_hexahedron, gmsh_prism and gmsh_pyramid, at main
// be8352e677303b686d6862b70923bab086b279ff (2026-10-19; the P3 hexahedron, prism and pyramid
// arrays entered main in c9e0341f3f8bafc9cc29148b0816ac4f5cacee65, ADR-0007).
const DOLFINX_TET10: [usize; 10] = [0, 1, 2, 3, 9, 6, 8, 7, 4, 5];
#[rustfmt::skip]
const DOLFINX_TET20: [usize; 20] = [0, 1, 2, 3, 14, 15, 8, 9, 13, 12, 11, 10, 5, 4, 7, 6, 19, 18, 17, 16];
#[rustfmt::skip]
const DOLFINX_QUAD16: [usize; 16] = [0, 1, 3, 2, 4, 5, 8, 9, 11, 10, 7, 6, 12, 13, 15, 14];
#[rustfmt::skip]
const DOLFINX_HEX64: [usize; 64] = [
    0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 19, 18, 22, 23, 20, 21, 24, 25, 26, 27, 28, 29, 31, 30,
    32, 34, 35, 33, 36, 37, 39, 38, 40, 42, 43, 41, 44, 45, 47, 46,
    49, 48, 50, 51, 52, 53, 55, 54, 56, 57, 59, 58, 60, 61, 63, 62,
];
#[rustfmt::skip]
const DOLFINX_PRISM40: [usize; 40] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 37, 25, 26,
    28, 27, 29, 31, 32, 30, 33, 34, 36, 35, 38, 39,
];
#[rustfmt::skip]
const DOLFINX_PYRAMID30: [usize; 30] = [
    0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    16, 15, 19, 20, 17, 18, 25, 26, 27, 28, 21, 23, 24, 22, 29,
];

/// A.11 for a tetrahedron of degree p: a Basix/DOLFINx node index to the gridcl index.
/// Vertices are unchanged; Basix edge e is gridcl edge 5 − e and Basix face f is gridcl face
/// 3 − f; the order inside each entity's block is unchanged. Interior nodes (p ≥ 4) are
/// unchanged.
fn tetrahedron_basix_to_gridcl(p: usize, basix: usize) -> usize {
    let per_edge = p - 1;
    let per_face = (p - 1) * (p - 2) / 2;
    let edges0 = 4;
    let faces0 = edges0 + 6 * per_edge;
    let interior0 = faces0 + 4 * per_face;
    if basix < edges0 || basix >= interior0 {
        basix
    } else if basix < faces0 {
        let (e, j) = ((basix - edges0) / per_edge, (basix - edges0) % per_edge);
        edges0 + (5 - e) * per_edge + j
    } else {
        let (f, j) = ((basix - faces0) / per_face, (basix - faces0) % per_face);
        faces0 + (3 - f) * per_face + j
    }
}

#[test]
fn gmsh_p3_matches_dolfinx() {
    let tet = |p, map: &[usize]| -> Vec<usize> {
        map.iter()
            .map(|&i| tetrahedron_basix_to_gridcl(p, i))
            .collect()
    };
    // The conversion reproduces A.9's tetrahedron row from DOLFINx's tet10 array.
    assert_eq!(tet(2, &DOLFINX_TET10), [0, 1, 2, 3, 4, 7, 5, 6, 9, 8]);

    assert_eq!(
        gmsh_to_internal(CellType::Tetrahedron, 3).unwrap(),
        tet(3, &DOLFINX_TET20)
    );
    assert_eq!(
        gmsh_to_internal(CellType::Quadrilateral, 3).unwrap(),
        DOLFINX_QUAD16
    );
    assert_eq!(
        gmsh_to_internal(CellType::Hexahedron, 3).unwrap(),
        DOLFINX_HEX64
    );
    assert_eq!(
        gmsh_to_internal(CellType::Prism, 3).unwrap(),
        DOLFINX_PRISM40
    );
    assert_eq!(
        gmsh_to_internal(CellType::Pyramid, 3).unwrap(),
        DOLFINX_PYRAMID30
    );
}
