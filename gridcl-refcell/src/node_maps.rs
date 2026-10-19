//! Node maps between file formats and the internal Lagrange order, A.9 (docs/design.md,
//! "High-order node orderings").
//!
//! Entry k of a map is the internal index of the file's k-th node:
//! `internal[map[k]] = file[k]`.
//!
//! Each format's order is generated from its recursive rule as points p·x of gridcl's
//! reference cell on the integer lattice, and [`match_lattice`] matches them to
//! [`lagrange_points`] (equispaced). Equispaced nodes are lattice points, so the matching is
//! exact, with no tolerance.

mod gmsh;
mod vtk;

use std::collections::HashMap;

use gridcl_dims::CellType;

use crate::{NodeFamily, RefCellError, lagrange_points, num_lagrange_nodes, tdim};

/// The map from Gmsh's node order to the internal order for `cell` of degree `degree`.
///
/// Generated for any p ≥ 1 from Gmsh's recursive rule (Gmsh 4.15.2,
/// `src/numeric/pointsGenerators.cpp`): vertices, edge interiors, face interiors as
/// lower-order elements, then the volume interior. The point cell gives `[0]`.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::gmsh_to_internal;
///
/// // A.9: Gmsh's quadrilateral runs round the boundary, gridcl's is tensor ordered.
/// let map = gmsh_to_internal(CellType::Quadrilateral, 1).unwrap();
/// assert_eq!(map, [0, 1, 3, 2]);
///
/// // Direction: entry k is the internal index of the file's k-th node.
/// let file = ["a", "b", "c", "d"];
/// let mut internal = [""; 4];
/// for (k, &i) in map.iter().enumerate() {
///     internal[i] = file[k];
/// }
/// assert_eq!(internal, ["a", "b", "d", "c"]);
/// ```
///
/// # Errors
///
/// [`RefCellError::UnsupportedDegree`] if `degree == 0`.
pub fn gmsh_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError> {
    if degree == 0 {
        return Err(RefCellError::UnsupportedDegree { cell, degree });
    }
    Ok(match_lattice(
        cell,
        degree,
        &gmsh::lattice_points(cell, degree),
    ))
}

/// The map from VTK's Lagrange node order to the internal order for `cell` of degree `degree`.
///
/// Generated for any p ≥ 1 from VTK's index functions (VTK 9.7.1,
/// `Common/DataModel/vtkHigherOrder*.cxx`): vertices, then edges and faces with VTK's own
/// directions, then the interior; triangle and tetrahedron interiors are recursive, the
/// others lexicographic (the triangle faces and layers of a prism too). The pyramid has only
/// the linear VTK pyramid (A.8), and the point cell gives `[0]`.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::{RefCellError, vtk_to_internal};
///
/// // A.9: VTK's quadrilateral runs counter-clockwise, then the edges, then the interior.
/// let map = vtk_to_internal(CellType::Quadrilateral, 2).unwrap();
/// assert_eq!(map, [0, 1, 3, 2, 4, 6, 7, 5, 8]);
///
/// // Direction: entry k is the internal index of the file's k-th node.
/// let file: Vec<usize> = (0..9).collect();
/// let mut internal = [0; 9];
/// for (k, &i) in map.iter().enumerate() {
///     internal[i] = file[k];
/// }
/// assert_eq!(internal, [0, 1, 3, 2, 4, 7, 5, 6, 8]);
///
/// // VTK has no usable Lagrange pyramid (A.8).
/// assert_eq!(
///     vtk_to_internal(CellType::Pyramid, 2),
///     Err(RefCellError::NotInFormat { format: "VTK", cell: CellType::Pyramid, degree: 2 })
/// );
/// ```
///
/// # Errors
///
/// - [`RefCellError::UnsupportedDegree`] if `degree == 0`;
/// - [`RefCellError::NotInFormat`] for pyramids of degree 2 or more (A.8).
pub fn vtk_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError> {
    if degree == 0 {
        return Err(RefCellError::UnsupportedDegree { cell, degree });
    }
    if cell == CellType::Pyramid && degree >= 2 {
        return Err(RefCellError::NotInFormat {
            format: "VTK",
            cell,
            degree,
        });
    }
    Ok(match_lattice(
        cell,
        degree,
        &vtk::lattice_points(cell, degree),
    ))
}

/// The internal index of each file node: entry k is the index in
/// [`lagrange_points`]`(cell, degree, Equispaced)` of `file[k]`.
///
/// `file[k]` is the file's k-th node as the lattice point p·x of gridcl's reference cell
/// (A.1); coordinates beyond [`tdim`] are zero. Shared by the Gmsh and VTK generators.
///
/// # Panics
///
/// If `degree == 0`, or if `file` is not a permutation of the equispaced nodes (a generator
/// bug); the message names the cell, the degree and the offending node.
fn match_lattice(cell: CellType, degree: u32, file: &[[u32; 3]]) -> Vec<usize> {
    let n = num_lagrange_nodes(cell, degree);
    assert_eq!(
        file.len(),
        n,
        "{cell:?} p = {degree}: {} file nodes, {n} Lagrange nodes",
        file.len()
    );
    let td = tdim(cell);
    let index: HashMap<[u32; 3], usize> = if td == 0 {
        HashMap::from([([0; 3], 0)])
    } else {
        let points = lagrange_points(cell, degree, NodeFamily::Equispaced)
            .expect("equispaced nodes exist for every cell and p >= 1");
        let p = f64::from(degree);
        points
            .chunks_exact(td)
            .enumerate()
            .map(|(i, x)| {
                let mut key = [0; 3];
                for (k, &xj) in key.iter_mut().zip(x) {
                    let scaled = (xj * p).round();
                    debug_assert!((xj * p - scaled).abs() < 1e-6, "node off the lattice");
                    *k = scaled as u32;
                }
                (key, i)
            })
            .collect()
    };
    let mut seen = vec![false; n];
    file.iter()
        .map(|node| {
            let &i = index.get(node).unwrap_or_else(|| {
                panic!("{cell:?} p = {degree}: file node {node:?} is not a Lagrange node")
            });
            assert!(
                !std::mem::replace(&mut seen[i], true),
                "{cell:?} p = {degree}: file node {node:?} appears twice"
            );
            i
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use CellType::*;
    use proptest::prelude::*;

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

    /// A.9's vertex rows (Gmsh → internal), indexed like `CELLS`.
    const GMSH_VERTEX_MAPS: [&[usize]; 8] = [
        &[0],
        &[0, 1],
        &[0, 1, 2],
        &[0, 1, 3, 2],
        &[0, 1, 2, 3],
        &[0, 1, 3, 2, 4, 5, 7, 6],
        &[0, 1, 2, 3, 4, 5],
        &[0, 1, 3, 2, 4],
    ];

    /// A.9's vertex rows (VTK → internal), indexed like `CELLS`.
    const VTK_VERTEX_MAPS: [&[usize]; 8] = [
        &[0],
        &[0, 1],
        &[0, 1, 2],
        &[0, 1, 3, 2],
        &[0, 1, 2, 3],
        &[0, 1, 3, 2, 4, 5, 7, 6],
        &[0, 1, 2, 3, 4, 5],
        &[0, 1, 3, 2, 4],
    ];

    #[test]
    fn degree_zero_is_rejected() {
        for cell in CELLS {
            let err = Err(RefCellError::UnsupportedDegree { cell, degree: 0 });
            assert_eq!(gmsh_to_internal(cell, 0), err);
            assert_eq!(vtk_to_internal(cell, 0), err);
        }
    }

    #[test]
    fn point_cell_has_one_node() {
        for degree in 1..=4 {
            assert_eq!(gmsh_to_internal(Point, degree).unwrap(), [0]);
            assert_eq!(vtk_to_internal(Point, degree).unwrap(), [0]);
        }
    }

    /// A.8: VTK has only the linear pyramid.
    #[test]
    fn vtk_has_only_the_linear_pyramid() {
        assert_eq!(vtk_to_internal(Pyramid, 1).unwrap(), [0, 1, 3, 2, 4]);
        for degree in 2..=5 {
            assert_eq!(
                vtk_to_internal(Pyramid, degree),
                Err(RefCellError::NotInFormat {
                    format: "VTK",
                    cell: Pyramid,
                    degree
                })
            );
        }
    }

    /// DOLFINx's Gmsh P3 triangle (`gmsh_triangle(10)` in `cpp/dolfinx/io/cells.cpp` at
    /// be8352e) after A.11: Basix edge e is gridcl edge 2 − e.
    #[test]
    fn p3_triangle_matches_dolfinx() {
        let dolfinx = [0, 1, 2, 7, 8, 3, 4, 6, 5, 9];
        let a11 = |b: usize| match b {
            3..=8 => 3 + (2 - (b - 3) / 2) * 2 + (b - 3) % 2,
            _ => b,
        };
        let want: Vec<usize> = dolfinx.into_iter().map(a11).collect();
        assert_eq!(gmsh_to_internal(Triangle, 3).unwrap(), want);
    }

    #[test]
    #[should_panic(expected = "is not a Lagrange node")]
    fn matching_rejects_points_off_the_layout() {
        match_lattice(Interval, 2, &[[0, 0, 0], [2, 0, 0], [3, 0, 0]]);
    }

    #[test]
    #[should_panic(expected = "appears twice")]
    fn matching_rejects_repeated_points() {
        match_lattice(Interval, 2, &[[0, 0, 0], [2, 0, 0], [2, 0, 0]]);
    }

    proptest! {
        /// Every Gmsh map is a bijection onto 0..N that starts with A.9's vertex map.
        #[test]
        fn gmsh_maps_are_permutations(c in 0..CELLS.len(), degree in 1u32..=6) {
            let cell = CELLS[c];
            let map = gmsh_to_internal(cell, degree).unwrap();
            let n = num_lagrange_nodes(cell, degree);
            let mut sorted = map.clone();
            sorted.sort_unstable();
            prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            prop_assert_eq!(&map[..GMSH_VERTEX_MAPS[c].len()], GMSH_VERTEX_MAPS[c]);
        }

        /// Every VTK map (pyramids p = 1 only) is a bijection onto 0..N that starts with A.9's
        /// vertex map.
        #[test]
        fn vtk_maps_are_permutations(c in 0..CELLS.len(), degree in 1u32..=6) {
            let cell = CELLS[c];
            let degree = if cell == Pyramid { 1 } else { degree };
            let map = vtk_to_internal(cell, degree).unwrap();
            let n = num_lagrange_nodes(cell, degree);
            let mut sorted = map.clone();
            sorted.sort_unstable();
            prop_assert_eq!(sorted, (0..n).collect::<Vec<_>>());
            prop_assert_eq!(&map[..VTK_VERTEX_MAPS[c].len()], VTK_VERTEX_MAPS[c]);
        }
    }
}
