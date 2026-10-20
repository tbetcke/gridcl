//! Gmsh's node order for Lagrange elements of any degree, on gridcl's integer lattice.
//!
//! Gmsh numbers the nodes of a degree-p element by the integer "monomials" of
//! `src/numeric/pointsGenerators.cpp` (functions `gmshGenerateMonomialsLine`, `…Triangle`,
//! `…Quadrangle`, `…Tetrahedron`, `…Hexahedron`, `…Prism` and `…Pyramid`; Lagrange pyramids
//! go through `gmshGenerateMonomialsPyramidGeneral`, which calls `…Pyramid` for them). The
//! rule, ported here line by line:
//!
//! 1. the vertices;
//! 2. the edge interiors, edge by edge from the edge's first vertex to its second, with the
//!    edge lists of `MTriangle::edges_tri`, `MQuadrangle::edges_quad`,
//!    `MTetrahedron::edges_tetra`, `MHexahedron::edges_hexa`, `MPrism::edges_prism` and
//!    `MPyramid::edges_pyramid`;
//! 3. the face interiors, face by face (`faces_tetra`, `faces_hexa`, `faces_prism`,
//!    `faces_pyramid`): each is a lower-order Gmsh triangle (degree p − 3) or quadrilateral
//!    (degree p − 2) shifted by one lattice step and placed with its u axis from the face's
//!    vertex 0 to its vertex 1 and its v axis from vertex 0 to vertex 2 (triangles) or
//!    vertex 3 (quadrilaterals);
//! 4. the volume interior: a Gmsh tetrahedron of degree p − 4, hexahedron of degree p − 2 or
//!    pyramid of degree p − 3, shifted inwards; for prisms the product of a Gmsh triangle of
//!    degree p − 3 (outer loop) and a Gmsh line of degree p − 2 (inner loop). Two-dimensional
//!    interiors are recursive in the same way (a triangle of degree p − 3, a quadrilateral of
//!    degree p − 2).
//!
//! Sources: Gmsh 4.15.2 (tag `gmsh_4_15_2`, GitHub mirror live-clones/gmsh),
//! `src/numeric/pointsGenerators.cpp` and the edge and face tables in `src/geo/MTriangle.h`,
//! `MQuadrangle.h`, `MTetrahedron.h`, `MHexahedron.h`, `MPrism.h` and `MPyramid.h`.
//!
//! Gmsh's monomial m of a degree-p element is the node's position on the lattice of its
//! reference element: x = m/p on simplices and in the triangle directions of the prism,
//! x = −1 + 2m/p on the line, quadrilateral, hexahedron and prism axis, and for the pyramid
//! x = (2m₀ − m₂)/p, y = (2m₁ − m₂)/p, z = 1 − m₂/p. The affine map from Gmsh's reference
//! element onto gridcl's (A.1) that sends Gmsh vertex k to internal vertex A.9 `map[k]` turns
//! m into gridcl's lattice point p·x: the identity for every cell except the pyramid, where it
//! is (m₀, m₁, p − m₂) ([`to_internal_lattice`]).

use gridcl_dims::CellType;

/// A point of Gmsh's monomial lattice; unused coordinates are zero.
type Monomial = [i64; 3];

/// `MTriangle::edges_tri`.
const EDGES_TRIANGLE: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];
/// `MQuadrangle::edges_quad`.
const EDGES_QUADRILATERAL: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];
/// `MTetrahedron::edges_tetra`.
const EDGES_TETRAHEDRON: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [3, 0], [3, 2], [3, 1]];
/// `MTetrahedron::faces_tetra`.
const FACES_TETRAHEDRON: [&[usize]; 4] = [&[0, 2, 1], &[0, 1, 3], &[0, 3, 2], &[3, 1, 2]];
/// `MHexahedron::edges_hexa`.
const EDGES_HEXAHEDRON: [[usize; 2]; 12] = [
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 5],
    [2, 3],
    [2, 6],
    [3, 7],
    [4, 5],
    [4, 7],
    [5, 6],
    [6, 7],
];
/// `MHexahedron::faces_hexa`.
const FACES_HEXAHEDRON: [&[usize]; 6] = [
    &[0, 3, 2, 1],
    &[0, 1, 5, 4],
    &[0, 4, 7, 3],
    &[1, 2, 6, 5],
    &[2, 3, 7, 6],
    &[4, 5, 6, 7],
];
/// `MPrism::edges_prism`.
const EDGES_PRISM: [[usize; 2]; 9] = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [1, 4],
    [2, 5],
    [3, 4],
    [3, 5],
    [4, 5],
];
/// `MPrism::faces_prism` (Gmsh's `-1` fourth vertex of a triangle dropped).
const FACES_PRISM: [&[usize]; 5] = [
    &[0, 2, 1],
    &[3, 4, 5],
    &[0, 1, 4, 3],
    &[0, 3, 5, 2],
    &[1, 2, 5, 4],
];
/// `MPyramid::edges_pyramid`.
const EDGES_PYRAMID: [[usize; 2]; 8] = [
    [0, 1],
    [0, 3],
    [0, 4],
    [1, 2],
    [1, 4],
    [2, 3],
    [2, 4],
    [3, 4],
];
/// `MPyramid::faces_pyramid` (Gmsh's `-1` fourth vertex of a triangle dropped).
const FACES_PYRAMID: [&[usize]; 5] = [
    &[0, 1, 4],
    &[3, 0, 4],
    &[1, 2, 4],
    &[2, 3, 4],
    &[0, 3, 2, 1],
];

/// The nodes of Gmsh's degree-`degree` Lagrange element on `cell`, in Gmsh's order, as
/// points p·x of gridcl's reference cell (unused coordinates zero).
///
/// `degree ≥ 1`; the point cell has the single node 0.
pub(super) fn lattice_points(cell: CellType, degree: u32) -> Vec<[u32; 3]> {
    debug_assert!(degree >= 1, "Gmsh node order needs p >= 1");
    let p = i64::from(degree);
    let monomials = match cell {
        CellType::Point => vec![[0; 3]],
        CellType::Interval => line(p).into_iter().map(|m| [m, 0, 0]).collect(),
        CellType::Triangle => triangle(p),
        CellType::Quadrilateral => quadrilateral(p),
        CellType::Tetrahedron => tetrahedron(p),
        CellType::Hexahedron => hexahedron(p),
        CellType::Prism => prism(p),
        CellType::Pyramid => pyramid(p),
    };
    monomials
        .into_iter()
        .map(|m| {
            to_internal_lattice(cell, p, m).map(|c| {
                u32::try_from(c).expect("Gmsh lattice point outside gridcl's reference cell")
            })
        })
        .collect()
}

/// Gmsh's monomial `m` of a degree-`p` element on `cell` as gridcl's lattice point p·x: the
/// affine map that sends Gmsh vertex k to internal vertex A.9 `map[k]` (module docs).
fn to_internal_lattice(cell: CellType, p: i64, m: Monomial) -> Monomial {
    match cell {
        CellType::Pyramid => [m[0], m[1], p - m[2]],
        _ => m,
    }
}

/// Every point moved by `shift` lattice steps.
fn shifted(points: Vec<Monomial>, shift: [i64; 3]) -> impl Iterator<Item = Monomial> {
    points
        .into_iter()
        .map(move |m| [m[0] + shift[0], m[1] + shift[1], m[2] + shift[2]])
}

/// Appends the interior nodes of every edge, from its first vertex to its second.
fn push_edges(out: &mut Vec<Monomial>, p: i64, edges: &[[usize; 2]]) {
    for &[i0, i1] in edges {
        let (a, b) = (out[i0], out[i1]);
        let u = [0, 1, 2].map(|j| (b[j] - a[j]) / p);
        for i in 1..p {
            out.push([0, 1, 2].map(|j| a[j] + i * u[j]));
        }
    }
}

/// Appends a face interior: `inner` (a lower-order element shifted by one lattice step) with
/// its u axis along vertex 0 → vertex 1 of `face` and its v axis along vertex 0 → `v_end`.
fn push_face(out: &mut Vec<Monomial>, p: i64, face: &[usize], v_end: usize, inner: &[Monomial]) {
    let (o, a, b) = (out[face[0]], out[face[1]], out[v_end]);
    let u = [0, 1, 2].map(|j| (a[j] - o[j]) / p);
    let v = [0, 1, 2].map(|j| (b[j] - o[j]) / p);
    for d in inner {
        out.push([0, 1, 2].map(|j| o[j] + u[j] * d[0] + v[j] * d[1]));
    }
}

/// Appends the face interiors of a volume cell; triangle faces are skipped for p ≤ 2.
fn push_faces(out: &mut Vec<Monomial>, p: i64, faces: &[&[usize]]) {
    let quad: Vec<Monomial> = shifted(quadrilateral(p - 2), [1, 1, 0]).collect();
    let tri: Vec<Monomial> = if p > 2 {
        shifted(triangle(p - 3), [1, 1, 0]).collect()
    } else {
        Vec::new()
    };
    for face in faces {
        match face.len() {
            4 => push_face(out, p, face, face[3], &quad),
            _ => push_face(out, p, face, face[2], &tri),
        }
    }
}

/// `gmshGenerateMonomialsLine`: 0, p, 1, …, p − 1.
fn line(p: i64) -> Vec<i64> {
    if p == 0 {
        return vec![0];
    }
    [0, p].into_iter().chain(1..p).collect()
}

/// `gmshGenerateMonomialsTriangle`.
fn triangle(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, 0]];
    if p == 0 {
        return out;
    }
    out.extend([[p, 0, 0], [0, p, 0]]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_TRIANGLE);
        if p > 2 {
            out.extend(shifted(triangle(p - 3), [1, 1, 0]));
        }
    }
    out
}

/// `gmshGenerateMonomialsQuadrangle`.
fn quadrilateral(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, 0]];
    if p == 0 {
        return out;
    }
    out.extend([[p, 0, 0], [p, p, 0], [0, p, 0]]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_QUADRILATERAL);
        out.extend(shifted(quadrilateral(p - 2), [1, 1, 0]));
    }
    out
}

/// `gmshGenerateMonomialsTetrahedron`.
fn tetrahedron(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, 0]];
    if p == 0 {
        return out;
    }
    out.extend([[p, 0, 0], [0, p, 0], [0, 0, p]]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_TETRAHEDRON);
        if p > 2 {
            push_faces(&mut out, p, &FACES_TETRAHEDRON);
            if p > 3 {
                out.extend(shifted(tetrahedron(p - 4), [1, 1, 1]));
            }
        }
    }
    out
}

/// `gmshGenerateMonomialsHexahedron`.
fn hexahedron(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, 0]];
    if p == 0 {
        return out;
    }
    out.extend([
        [p, 0, 0],
        [p, p, 0],
        [0, p, 0],
        [0, 0, p],
        [p, 0, p],
        [p, p, p],
        [0, p, p],
    ]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_HEXAHEDRON);
        push_faces(&mut out, p, &FACES_HEXAHEDRON);
        out.extend(shifted(hexahedron(p - 2), [1, 1, 1]));
    }
    out
}

/// `gmshGenerateMonomialsPrism`.
fn prism(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, 0]];
    if p == 0 {
        return out;
    }
    out.extend([[p, 0, 0], [0, p, 0], [0, 0, p], [p, 0, p], [0, p, p]]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_PRISM);
        push_faces(&mut out, p, &FACES_PRISM);
        if p > 2 {
            let line = line(p - 2);
            for t in triangle(p - 3) {
                for &l in &line {
                    out.push([1 + t[0], 1 + t[1], 1 + l]);
                }
            }
        }
    }
    out
}

/// `gmshGenerateMonomialsPyramid`: the base at m₂ = p, the apex at the origin.
fn pyramid(p: i64) -> Vec<Monomial> {
    let mut out = vec![[0, 0, p]];
    if p == 0 {
        return out;
    }
    out.extend([[p, 0, p], [p, p, p], [0, p, p], [0, 0, 0]]);
    if p > 1 {
        push_edges(&mut out, p, &EDGES_PYRAMID);
        push_faces(&mut out, p, &FACES_PYRAMID);
        if p > 2 {
            out.extend(shifted(pyramid(p - 3), [1, 1, 2]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{num_lagrange_nodes, vertices};
    use CellType::*;

    /// A.9's vertex rows (Gmsh → internal).
    const VERTEX_MAPS: [(CellType, &[usize]); 8] = [
        (Point, &[0]),
        (Interval, &[0, 1]),
        (Triangle, &[0, 1, 2]),
        (Quadrilateral, &[0, 1, 3, 2]),
        (Tetrahedron, &[0, 1, 2, 3]),
        (Hexahedron, &[0, 1, 3, 2, 4, 5, 7, 6]),
        (Prism, &[0, 1, 2, 3, 4, 5]),
        (Pyramid, &[0, 1, 3, 2, 4]),
    ];

    /// The lattice map sends Gmsh's vertex k to internal vertex A.9 `map[k]` for every degree,
    /// so it is the affine map of the vertex correspondence.
    #[test]
    fn lattice_map_follows_the_vertex_correspondence() {
        for (cell, map) in VERTEX_MAPS {
            for degree in 1..=5 {
                let points = lattice_points(cell, degree);
                for (k, &internal) in map.iter().enumerate() {
                    let want: Vec<u32> = vertices(cell)[internal]
                        .iter()
                        .map(|&x| x as u32 * degree)
                        .collect();
                    assert_eq!(
                        points[k][..want.len()],
                        want[..],
                        "{cell:?} p = {degree} vertex {k}"
                    );
                }
            }
        }
    }

    #[test]
    fn node_counts_match_a4() {
        for (cell, _) in VERTEX_MAPS {
            for degree in 1..=7 {
                assert_eq!(
                    lattice_points(cell, degree).len(),
                    num_lagrange_nodes(cell, degree),
                    "{cell:?} p = {degree}"
                );
            }
        }
    }

    /// The Gmsh line order of `gmshGenerateMonomialsLine`.
    #[test]
    fn line_lists_end_points_first() {
        assert_eq!(line(0), [0]);
        assert_eq!(line(1), [0, 1]);
        assert_eq!(line(4), [0, 4, 1, 2, 3]);
    }

    /// Gmsh's second-order triangle and quadrilateral (reference manual, "Node ordering").
    #[test]
    fn second_order_surface_elements() {
        assert_eq!(
            triangle(2),
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]
            ]
        );
        assert_eq!(
            quadrilateral(2),
            [
                [0, 0, 0],
                [2, 0, 0],
                [2, 2, 0],
                [0, 2, 0],
                [1, 0, 0],
                [2, 1, 0],
                [1, 2, 0],
                [0, 1, 0],
                [1, 1, 0]
            ]
        );
    }
}
