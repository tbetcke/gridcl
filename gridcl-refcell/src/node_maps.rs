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
/// # Errors
///
/// - [`RefCellError::UnsupportedDegree`] if `degree == 0`;
/// - [`RefCellError::NotInFormat`] for pyramids of degree 2 or more (A.8).
// T9 stub: the allowance goes with the body.
#[allow(unused_variables)]
pub fn vtk_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError> {
    todo!()
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

    #[test]
    fn degree_zero_is_rejected() {
        for cell in CELLS {
            assert_eq!(
                gmsh_to_internal(cell, 0),
                Err(RefCellError::UnsupportedDegree { cell, degree: 0 })
            );
        }
    }

    #[test]
    fn point_cell_has_one_node() {
        for degree in 1..=4 {
            assert_eq!(gmsh_to_internal(Point, degree).unwrap(), [0]);
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
    }
}
