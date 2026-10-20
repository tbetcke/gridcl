//! VTK's node order for Lagrange cells of any degree, on gridcl's integer lattice.
//!
//! VTK numbers the nodes of its Lagrange cells (types 68 to 74) by index functions that take
//! a node's integer coordinates on the cell's lattice to its position in the connectivity.
//! They are ported here line by line, and the order is obtained by placing every lattice point
//! at the position its index function gives:
//!
//! - curve: `vtkHigherOrderInterpolation::AppendCurveCollocationPoints`: the two vertices,
//!   then the interior from x = 0 to x = 1;
//! - triangle: `vtkHigherOrderTriangle::Index`: recursive, as in Gmsh: vertices, the edges
//!   0 → 1, 1 → 2, 2 → 0, then the interior as a triangle of degree p − 3;
//! - quadrilateral: `vtkHigherOrderQuadrilateral::PointIndexFromIJK`;
//! - tetrahedron: `vtkHigherOrderTetra::Index`: vertices, the edges 0 → 1, 1 → 2, 2 → 0,
//!   0 → 3, 1 → 3, 2 → 3, the faces (0, 1, 3), (1, 2, 3), (0, 2, 3), (0, 1, 2), each a
//!   recursive triangle in barycentric coordinates (`FaceBCoords`), then the interior as a
//!   tetrahedron of degree p − 4;
//! - hexahedron: `vtkHigherOrderHexahedron::PointIndexFromIJK`;
//! - wedge (prism): `vtkHigherOrderWedge::PointIndexFromIJK`;
//! - pyramid: the linear `VTK_PYRAMID` (14) only. VTK's Lagrange pyramid (74) is a
//!   placeholder, so higher degrees are not in the format (A.8).
//!
//! The quadrilateral, hexahedron and wedge follow VTK's axis-aligned conventions: edges run
//! along +x, +y or +z (the wedge's third bottom and top edges run from vertex 2 to vertex 0
//! and from 5 to 3, its 1 → 2 edges along +y), the hexahedron's faces come in the order
//! x = 0, x = 1, y = 0, y = 1, z = 0, z = 1, and face and volume interiors are lexicographic,
//! the first free axis fastest. The wedge's triangular faces and the triangle layers of its
//! interior are lexicographic too (`triangleDOFOffset`: x fastest, then y), not recursive; its
//! quadrilateral faces come in the order y = 0, x + y = 1, x = 0, with the in-plane axis
//! fastest and z slowest. The hexahedron order is VTK 9's (VTK XML files of version 2.2 or
//! later); older files swap the two top y-edges (`NodeNumberingMappingFromVTK8To9`).
//!
//! Sources: VTK v9.7.1 (tag `v9.7.1`, github.com/Kitware/VTK), `Common/DataModel/`
//! `vtkHigherOrderTriangle.cxx`, `vtkHigherOrderQuadrilateral.cxx`, `vtkHigherOrderTetra.cxx`,
//! `vtkHigherOrderHexahedron.cxx`, `vtkHigherOrderWedge.cxx` and
//! `vtkHigherOrderInterpolation.cxx`.
//!
//! VTK's parametric coordinates of these cells are on [0, 1] with gridcl's geometry (A.1): a
//! node with VTK lattice coordinates (i, j, k) (barycentric indices (b₀, b₁, b₂, …) for the
//! simplices, whose first entries are p·x) is gridcl's lattice point (i, j, k). VTK's
//! quadrilateral and hexahedron vertices run counter-clockwise, which is only a renumbering of
//! the same points.

use gridcl_dims::CellType;

/// The nodes of VTK's degree-`degree` Lagrange cell on `cell`, in VTK's order, as points p·x
/// of gridcl's reference cell (unused coordinates zero).
///
/// `degree ≥ 1`, and `degree == 1` for the pyramid; the point cell has the single node 0.
pub(super) fn lattice_points(cell: CellType, degree: u32) -> Vec<[u32; 3]> {
    debug_assert!(degree >= 1, "VTK node order needs p >= 1");
    let p = i64::from(degree);
    let range = move || 0..=p;
    match cell {
        CellType::Point => vec![[0; 3]],
        CellType::Interval => [0, p]
            .into_iter()
            .chain(1..p)
            .map(|i| to_u32([i, 0, 0]))
            .collect(),
        CellType::Triangle => scatter(
            cell,
            degree,
            range().flat_map(|j| (0..=p - j).map(move |i| [i, j, 0])),
            |[i, j, _]| triangle_index([i, j, p - i - j], p),
        ),
        CellType::Quadrilateral => scatter(
            cell,
            degree,
            range().flat_map(|j| range().map(move |i| [i, j, 0])),
            |[i, j, _]| quadrilateral_index(i, j, p),
        ),
        CellType::Tetrahedron => scatter(
            cell,
            degree,
            range().flat_map(|k| {
                (0..=p - k).flat_map(move |j| (0..=p - k - j).map(move |i| [i, j, k]))
            }),
            |[i, j, k]| tetrahedron_index([i, j, k, p - i - j - k], p),
        ),
        CellType::Hexahedron => scatter(
            cell,
            degree,
            range().flat_map(|k| range().flat_map(move |j| range().map(move |i| [i, j, k]))),
            |[i, j, k]| hexahedron_index(i, j, k, p),
        ),
        CellType::Prism => scatter(
            cell,
            degree,
            range().flat_map(|k| (0..=p).flat_map(move |j| (0..=p - j).map(move |i| [i, j, k]))),
            |[i, j, k]| wedge_index(i, j, k, p),
        ),
        CellType::Pyramid => {
            assert_eq!(degree, 1, "VTK has no Lagrange pyramid (A.8)");
            // VTK_PYRAMID: the base counter-clockwise, then the apex.
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]
                .map(to_u32)
                .to_vec()
        }
    }
}

/// Places every lattice point at the position VTK's `index` gives it.
///
/// # Panics
///
/// If an index is out of range or two points share one (a port bug).
fn scatter(
    cell: CellType,
    degree: u32,
    points: impl Iterator<Item = [i64; 3]>,
    index: impl Fn([i64; 3]) -> i64,
) -> Vec<[u32; 3]> {
    let points: Vec<[i64; 3]> = points.collect();
    let mut out: Vec<Option<[u32; 3]>> = vec![None; points.len()];
    for x in points {
        let k = usize::try_from(index(x))
            .ok()
            .filter(|&k| k < out.len())
            .unwrap_or_else(|| panic!("{cell:?} p = {degree}: VTK index of {x:?} out of range"));
        assert!(
            out[k].replace(to_u32(x)).is_none(),
            "{cell:?} p = {degree}: two nodes at VTK index {k}"
        );
    }
    out.into_iter()
        .map(|x| x.expect("as many indices as points, all distinct"))
        .collect()
}

fn to_u32(x: [i64; 3]) -> [u32; 3] {
    x.map(|c| u32::try_from(c).expect("VTK lattice point outside gridcl's reference cell"))
}

/// `vtkHigherOrderTriangle::Index`: the position of the node with barycentric index `b`
/// (b₀ = p·x, b₁ = p·y, b₂ = p − b₀ − b₁).
fn triangle_index(b: [i64; 3], order: i64) -> i64 {
    debug_assert_eq!(b.iter().sum::<i64>(), order);
    let (mut order, mut index, mut max, mut min) = (order, 0, order, 0);
    let bmin = b.into_iter().min().expect("three entries");
    // Scope into the correct triangle.
    while bmin > min {
        index += 3 * order;
        max -= 2;
        min += 1;
        order -= 3;
    }
    for dim in 0..3 {
        if b[(dim + 2) % 3] == max {
            // A vertex.
            return index;
        }
        index += 1;
    }
    for dim in 0..3 {
        if b[(dim + 1) % 3] == min {
            // An edge.
            return index + b[dim] - (min + 1);
        }
        index += max - (min + 1);
    }
    index
}

/// `VertexMaxCoords` of `vtkHigherOrderTetra.cxx`.
const TET_VERTEX_MAX_COORDS: [usize; 4] = [3, 0, 1, 2];
/// `EdgeMinCoords`.
const TET_EDGE_MIN_COORDS: [[usize; 2]; 6] = [[1, 2], [2, 3], [0, 2], [0, 1], [1, 3], [0, 3]];
/// `EdgeCountingCoord`.
const TET_EDGE_COUNTING_COORD: [usize; 6] = [0, 1, 3, 2, 2, 2];
/// `FaceBCoords`.
const TET_FACE_B_COORDS: [[usize; 3]; 4] = [[0, 2, 3], [2, 0, 1], [2, 1, 3], [1, 0, 3]];
/// `FaceMinCoord`.
const TET_FACE_MIN_COORD: [usize; 4] = [1, 3, 0, 2];

/// `vtkHigherOrderTetra::Index`: the position of the node with barycentric index `b`
/// (b₀ = p·x, b₁ = p·y, b₂ = p·z, b₃ = p − b₀ − b₁ − b₂).
fn tetrahedron_index(b: [i64; 4], order: i64) -> i64 {
    debug_assert_eq!(b.iter().sum::<i64>(), order);
    let (mut order, mut index, mut max, mut min) = (order, 0, order, 0);
    let bmin = b.into_iter().min().expect("four entries");
    // Scope into the correct tetrahedron.
    while bmin > min {
        index += 2 * (order * order + 1);
        max -= 3;
        min += 1;
        order -= 4;
    }
    for vertex in 0..4 {
        if b[TET_VERTEX_MAX_COORDS[vertex]] == max {
            // A vertex.
            return index;
        }
        index += 1;
    }
    for edge in 0..6 {
        let [c0, c1] = TET_EDGE_MIN_COORDS[edge];
        if b[c0] == min && b[c1] == min {
            // An edge.
            return index + b[TET_EDGE_COUNTING_COORD[edge]] - (min + 1);
        }
        index += max - (min + 1);
    }
    for face in 0..4 {
        if b[TET_FACE_MIN_COORD[face]] == min {
            // A face: subtract the face's vertices and edges, 3·order nodes.
            let projected = TET_FACE_B_COORDS[face].map(|c| b[c] - min);
            return index + triangle_index(projected, order) - 3 * order;
        }
        index += (order + 1) * (order + 2) / 2 - 3 * order;
    }
    index
}

/// `vtkHigherOrderQuadrilateral::PointIndexFromIJK` with `order = [p, p]`.
fn quadrilateral_index(i: i64, j: i64, p: i64) -> i64 {
    let ibdy = i == 0 || i == p;
    let jbdy = j == 0 || j == p;
    match (ibdy, jbdy) {
        // A vertex.
        (true, true) => match (i != 0, j != 0) {
            (false, false) => 0,
            (true, false) => 1,
            (true, true) => 2,
            (false, true) => 3,
        },
        // An edge along the i axis.
        (false, true) => 4 + (i - 1) + if j != 0 { 2 * (p - 1) } else { 0 },
        // An edge along the j axis.
        (true, false) => 4 + (j - 1) + if i != 0 { p - 1 } else { 3 * (p - 1) },
        // The interior.
        (false, false) => 4 + 4 * (p - 1) + (i - 1) + (p - 1) * (j - 1),
    }
}

/// `vtkHigherOrderHexahedron::PointIndexFromIJK` with `order = [p, p, p]`.
fn hexahedron_index(i: i64, j: i64, k: i64, p: i64) -> i64 {
    let bdy = [i, j, k].map(|c| c == 0 || c == p);
    let (ibdy, jbdy, kbdy) = (bdy[0], bdy[1], bdy[2]);
    let nbdy = bdy.into_iter().filter(|&b| b).count();
    let q = p - 1;
    // The vertex number of a corner of the k = 0 square, counter-clockwise.
    let corner = |i: i64, j: i64| match (i != 0, j != 0) {
        (false, false) => 0,
        (true, false) => 1,
        (true, true) => 2,
        (false, true) => 3,
    };
    if nbdy == 3 {
        return corner(i, j) + if k != 0 { 4 } else { 0 };
    }
    let mut offset = 8;
    if nbdy == 2 {
        let top = if k != 0 { 4 * q } else { 0 };
        if !ibdy {
            // An edge along the i axis.
            return (i - 1) + if j != 0 { 2 * q } else { 0 } + top + offset;
        }
        if !jbdy {
            // An edge along the j axis.
            return (j - 1) + if i != 0 { q } else { 3 * q } + top + offset;
        }
        // An edge along the k axis.
        offset += 8 * q;
        return (k - 1) + q * corner(i, j) + offset;
    }
    offset += 12 * q;
    if nbdy == 1 {
        if ibdy {
            // An i-normal face.
            return (j - 1) + q * (k - 1) + if i != 0 { q * q } else { 0 } + offset;
        }
        offset += 2 * q * q;
        if jbdy {
            // A j-normal face.
            return (i - 1) + q * (k - 1) + if j != 0 { q * q } else { 0 } + offset;
        }
        offset += 2 * q * q;
        debug_assert!(kbdy);
        // A k-normal face.
        return (i - 1) + q * (j - 1) + if k != 0 { q * q } else { 0 } + offset;
    }
    // The interior.
    offset += 6 * q * q;
    offset + (i - 1) + q * ((j - 1) + q * (k - 1))
}

/// `triangleDOFOffset` of `vtkHigherOrderWedge.cxx`: the lexicographic position of the
/// interior triangle node (i, j), i fastest.
fn triangle_dof_offset(order: i64, i: i64, j: i64) -> i64 {
    i + order * (j - 1) - (j * (j + 1)) / 2
}

/// `vtkHigherOrderWedge::PointIndexFromIJK` with `order = [p, p, p]`.
fn wedge_index(i: i64, j: i64, k: i64, p: i64) -> i64 {
    let (rs_order, t_order) = (p, p);
    let (rm1, tm1) = (rs_order - 1, t_order - 1);
    let ibdy = i == 0;
    let jbdy = j == 0;
    let ijbdy = i + j == rs_order;
    let kbdy = k == 0 || k == t_order;
    let nbdy = [ibdy, jbdy, ijbdy, kbdy].into_iter().filter(|&b| b).count();
    // The vertex number of a corner of the bottom triangle.
    let corner = || {
        if ibdy && jbdy {
            0
        } else if jbdy && ijbdy {
            1
        } else {
            2
        }
    };
    if nbdy == 3 {
        return corner() + if k != 0 { 3 } else { 0 };
    }
    let mut offset = 6;
    if nbdy == 2 {
        if !kbdy {
            // A vertical edge: two of ibdy, jbdy, ijbdy hold.
            offset += 6 * rm1;
            return offset + (k - 1) + corner() * tm1;
        }
        // A horizontal edge: kbdy and one of ibdy, jbdy, ijbdy hold. Skip the bottom edges on
        // the top face.
        offset += if k == t_order { 3 * rm1 } else { 0 };
        if jbdy {
            return offset + i - 1;
        }
        offset += rm1;
        if ijbdy {
            return offset + j - 1;
        }
        offset += rm1;
        return offset + (rs_order - j - 1);
    }
    offset += 6 * rm1 + 3 * tm1;
    // Nodes inside one triangular and one quadrilateral face.
    let ntfdof = (rm1 - 1) * rm1 / 2;
    let nqfdof = rm1 * tm1;
    if nbdy == 1 {
        if kbdy {
            // A triangular face.
            if k > 0 {
                offset += ntfdof;
            }
            return offset + triangle_dof_offset(rs_order, i, j);
        }
        offset += 2 * ntfdof;
        // Quadrilateral faces: y = 0, then x + y = 1, then x = 0.
        if jbdy {
            return offset + (i - 1) + rm1 * (k - 1);
        }
        offset += nqfdof;
        if ijbdy {
            return offset + (j - 1) + rm1 * (k - 1);
        }
        offset += nqfdof;
        return offset + (rs_order - j - 1) + rm1 * (k - 1);
    }
    // The interior.
    offset += 2 * ntfdof + 3 * nqfdof;
    offset + triangle_dof_offset(rs_order, i, j) + ntfdof * (k - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{num_lagrange_nodes, vertices};
    use CellType::*;

    /// A.9's vertex rows (VTK → internal).
    const VERTEX_MAPS: [(CellType, &[usize]); 7] = [
        (Point, &[0]),
        (Interval, &[0, 1]),
        (Triangle, &[0, 1, 2]),
        (Quadrilateral, &[0, 1, 3, 2]),
        (Tetrahedron, &[0, 1, 2, 3]),
        (Hexahedron, &[0, 1, 3, 2, 4, 5, 7, 6]),
        (Prism, &[0, 1, 2, 3, 4, 5]),
    ];

    #[test]
    fn vertices_come_first_in_vtk_order() {
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

    /// VTK's triangle is recursive with Gmsh's rule (vertices, edges 0 → 1 → 2 → 0, then a
    /// triangle of degree p − 3), and `vtkHigherOrderTriangle::SetParametricCoords` lists the
    /// same points.
    #[test]
    fn triangle_is_gmsh_recursive() {
        for degree in 1..=9 {
            assert_eq!(
                lattice_points(Triangle, degree),
                super::super::gmsh::lattice_points(Triangle, degree),
                "p = {degree}"
            );
        }
    }

    /// `vtkHigherOrderTriangle::BarycentricIndex`, the inverse of `Index`, ported separately.
    fn triangle_barycentric_index(mut index: i64, mut order: i64) -> [i64; 3] {
        let (mut max, mut min) = (order, 0);
        while index != 0 && index >= 3 * order {
            index -= 3 * order;
            max -= 2;
            min += 1;
            order -= 3;
        }
        let mut b = [0; 3];
        if index < 3 {
            let i = index as usize;
            b[i] = min;
            b[(i + 1) % 3] = min;
            b[(i + 2) % 3] = max;
        } else {
            let index = index - 3;
            let dim = index / (order - 1);
            let offset = index - dim * (order - 1);
            let dim = dim as usize;
            b[(dim + 1) % 3] = min;
            b[(dim + 2) % 3] = (max - 1) - offset;
            b[dim] = (min + 1) + offset;
        }
        b
    }

    /// `LinearVertices` and `EdgeVertices` of `vtkHigherOrderTetra.cxx`.
    const TET_LINEAR_VERTICES: [[i64; 4]; 4] =
        [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
    const TET_EDGE_VERTICES: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

    /// `vtkHigherOrderTetra::BarycentricIndex`, the inverse of `Index`, ported separately.
    fn tetrahedron_barycentric_index(mut index: i64, mut order: i64) -> [i64; 4] {
        let (mut max, mut min) = (order, 0);
        while index >= 2 * (order * order + 1) && index != 0 && order > 3 {
            index -= 2 * (order * order + 1);
            max -= 3;
            min += 1;
            order -= 4;
        }
        let mut b = [0; 4];
        if index < 4 {
            for (c, bc) in b.iter_mut().enumerate() {
                *bc = if c == TET_VERTEX_MAX_COORDS[index as usize] {
                    max
                } else {
                    min
                };
            }
        } else if index - 4 < 6 * (order - 1) {
            let edge = ((index - 4) / (order - 1)) as usize;
            let v = (index - 4) % (order - 1);
            let [a, e] = TET_EDGE_VERTICES[edge];
            for (c, bc) in b.iter_mut().enumerate() {
                *bc = min
                    + TET_LINEAR_VERTICES[a][c] * (max - min - 1 - v)
                    + TET_LINEAR_VERTICES[e][c] * (1 + v);
            }
        } else {
            let per_face = (order - 2) * (order - 1) / 2;
            let face = ((index - 4 - 6 * (order - 1)) / per_face) as usize;
            let v = (index - 4 - 6 * (order - 1)) % per_face;
            let projected = if order == 3 {
                [0; 3]
            } else {
                triangle_barycentric_index(v, order - 3)
            };
            for (i, &c) in TET_FACE_B_COORDS[face].iter().enumerate() {
                b[c] = min + 1 + projected[i];
            }
            b[TET_FACE_MIN_COORD[face]] = min;
        }
        b
    }

    /// `Index` and `BarycentricIndex` (which `SetParametricCoords` uses to place the nodes)
    /// are inverse to each other, for the triangle and the tetrahedron.
    #[test]
    fn simplex_index_inverts_barycentric_index() {
        for degree in 1..=9u32 {
            let p = i64::from(degree);
            for (n, x) in lattice_points(Triangle, degree).into_iter().enumerate() {
                let [b0, b1, _] = triangle_barycentric_index(n as i64, p);
                assert_eq!(
                    [b0, b1, 0],
                    x.map(i64::from),
                    "triangle p = {degree}, node {n}"
                );
            }
            for (n, x) in lattice_points(Tetrahedron, degree).into_iter().enumerate() {
                let [b0, b1, b2, _] = tetrahedron_barycentric_index(n as i64, p);
                assert_eq!(
                    [b0, b1, b2],
                    x.map(i64::from),
                    "tetrahedron p = {degree}, node {n}"
                );
            }
        }
    }

    /// `wedgeCorner`, `wedgeEdgeCorners` and `wedgeFaceCorners` of
    /// `vtkHigherOrderInterpolation.cxx` (corner columns only).
    const WEDGE_CORNERS: [[i64; 3]; 6] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
    ];
    const WEDGE_EDGES: [[usize; 2]; 9] = [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
        [0, 3],
        [1, 4],
        [2, 5],
    ];
    const WEDGE_FACES: [&[usize]; 5] = [
        &[0, 1, 2],
        &[3, 4, 5],
        &[0, 1, 4, 3],
        &[1, 2, 5, 4],
        &[2, 0, 3, 5],
    ];

    /// `vtkHigherOrderInterpolation::AppendWedgeCollocationPoints` (which places the nodes of
    /// `vtkHigherOrderWedge`), ported separately on the lattice.
    fn wedge_collocation_points(p: i64) -> Vec<[i64; 3]> {
        let corner = |v: usize| WEDGE_CORNERS[v].map(|c| c * p);
        // f0 + (r·(f1 − f0) + s·(f3 − f0)) / p for lattice steps r, s.
        let at = |f0: [i64; 3], f1: [i64; 3], f3: [i64; 3], r: i64, s: i64| {
            [0, 1, 2].map(|c| f0[c] + (r * (f1[c] - f0[c]) + s * (f3[c] - f0[c])) / p)
        };
        let mut out: Vec<[i64; 3]> = (0..6).map(corner).collect();
        for [a, b] in WEDGE_EDGES {
            for r in 1..p {
                out.push(at(corner(a), corner(b), corner(a), r, 0));
            }
        }
        for face in &WEDGE_FACES[..2] {
            // VTK's f3 is the face's third corner.
            let (f0, f1, f3) = (corner(face[0]), corner(face[1]), corner(face[2]));
            for s in 1..p {
                for r in 1..p - s {
                    out.push(at(f0, f1, f3, r, s));
                }
            }
        }
        for face in &WEDGE_FACES[2..] {
            let (f0, f1, f3) = (corner(face[0]), corner(face[1]), corner(face[3]));
            for s in 1..p {
                for r in 1..p {
                    out.push(at(f0, f1, f3, r, s));
                }
            }
        }
        for k in 1..p {
            for j in 1..p {
                for i in 1..p - j {
                    out.push([i, j, k]);
                }
            }
        }
        out
    }

    /// The wedge's two descriptions in VTK, `PointIndexFromIJK` and the collocation points,
    /// give the same order.
    #[test]
    fn wedge_index_matches_collocation_points() {
        for degree in 1..=9u32 {
            let want = wedge_collocation_points(i64::from(degree));
            let got: Vec<[i64; 3]> = lattice_points(Prism, degree)
                .into_iter()
                .map(|x| x.map(i64::from))
                .collect();
            assert_eq!(got, want, "p = {degree}");
        }
    }

    /// The wedge's triangular faces are lexicographic: for p = 5 the six nodes of the bottom
    /// face come row by row, where Gmsh's recursive rule would list (1, 1), (3, 1), (1, 3)
    /// first.
    #[test]
    fn wedge_triangle_faces_are_lexicographic() {
        let p = 5;
        let first = 6 + 9 * (p as usize - 1);
        let bottom: Vec<[u32; 3]> = lattice_points(Prism, p)[first..first + 6].to_vec();
        assert_eq!(
            bottom,
            [
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0],
                [1, 2, 0],
                [2, 2, 0],
                [1, 3, 0]
            ]
        );
    }

    /// The VTK 9 Lagrange quadrilateral of degree 3 (16 nodes): vertices counter-clockwise,
    /// edges along +x or +y, the interior row by row.
    #[test]
    fn cubic_quadrilateral() {
        let want: [[u32; 3]; 16] = [
            [0, 0, 0],
            [3, 0, 0],
            [3, 3, 0],
            [0, 3, 0],
            [1, 0, 0],
            [2, 0, 0],
            [3, 1, 0],
            [3, 2, 0],
            [1, 3, 0],
            [2, 3, 0],
            [0, 1, 0],
            [0, 2, 0],
            [1, 1, 0],
            [2, 1, 0],
            [1, 2, 0],
            [2, 2, 0],
        ];
        assert_eq!(lattice_points(Quadrilateral, 3), want);
    }
}
