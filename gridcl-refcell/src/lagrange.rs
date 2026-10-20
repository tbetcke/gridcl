//! Lagrange node counts and layouts, A.4 and A.5 (docs/design.md, Appendix A).
//!
//! Blocks are ordered vertices, edges, faces, interior; inside each sub-entity the nodes follow
//! that entity's own vertex list (A.5). The interval, triangle, quadrilateral, tetrahedron and
//! hexahedron layouts are exactly ndelement's; the prism and pyramid rules are gridcl's and
//! coincide with Basix's equispaced lattices.

use core::ops::Range;

use gridcl_dims::CellType;

use crate::{RefCellError, sub_entities, sub_entity_types, tdim, vertices};

/// The 1D point family of a Lagrange layout ("Reference cells and conventions", node families).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeFamily {
    /// Equispaced points, on every cell.
    Equispaced,
    /// Gauss–Lobatto–Legendre points, on intervals, quadrilaterals and hexahedra only.
    Gll,
}

/// `f(p)` in checked u64 arithmetic, converted to usize; panics naming the cell and degree.
fn checked_count(cell: CellType, degree: u32, f: impl Fn(u64) -> Option<u64>) -> usize {
    f(u64::from(degree))
        .and_then(|n| usize::try_from(n).ok())
        .unwrap_or_else(|| {
            panic!("Lagrange node count of {cell:?} cells of degree {degree} overflows usize")
        })
}

/// The number of Lagrange nodes N(p) of degree `degree` on `cell` (A.4).
///
/// The point cell has one node. Degree 0 gives one node on every cell.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::num_lagrange_nodes;
///
/// assert_eq!(num_lagrange_nodes(CellType::Prism, 3), 40);
/// assert_eq!(num_lagrange_nodes(CellType::Pyramid, 3), 30);
/// ```
///
/// # Panics
///
/// If the count overflows `usize`; the message names the cell and the degree.
pub fn num_lagrange_nodes(cell: CellType, degree: u32) -> usize {
    checked_count(cell, degree, |p| {
        let a = p.checked_add(1)?;
        let b = p.checked_add(2)?;
        match cell {
            CellType::Point => Some(1),
            CellType::Interval => Some(a),
            CellType::Triangle => Some(a.checked_mul(b)? / 2),
            CellType::Quadrilateral => a.checked_mul(a),
            CellType::Tetrahedron => Some(a.checked_mul(b)?.checked_mul(p.checked_add(3)?)? / 6),
            CellType::Hexahedron => a.checked_mul(a)?.checked_mul(a),
            CellType::Prism => Some(a.checked_mul(a)?.checked_mul(b)? / 2),
            CellType::Pyramid => {
                let c = p.checked_mul(2)?.checked_add(3)?;
                Some(a.checked_mul(b)?.checked_mul(c)? / 6)
            }
        }
    })
}

/// The number of Lagrange nodes in the interior of `cell` (A.4); 0 below the first degree with
/// interior nodes, for degree 0, and for the point cell.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::num_interior_lagrange_nodes;
///
/// assert_eq!(num_interior_lagrange_nodes(CellType::Tetrahedron, 3), 0);
/// assert_eq!(num_interior_lagrange_nodes(CellType::Tetrahedron, 4), 1);
/// ```
///
/// # Panics
///
/// If the count overflows `usize`; the message names the cell and the degree.
pub fn num_interior_lagrange_nodes(cell: CellType, degree: u32) -> usize {
    checked_count(cell, degree, |p| {
        // The first degree with interior nodes: every factor below is then non-negative.
        let first = match cell {
            CellType::Point => return Some(0),
            CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron => 2,
            CellType::Triangle | CellType::Prism | CellType::Pyramid => 3,
            CellType::Tetrahedron => 4,
        };
        if p < first {
            return Some(0);
        }
        let (a, b) = (p - 1, p - 2);
        match cell {
            CellType::Interval => Some(a),
            CellType::Triangle => Some(a * b / 2),
            CellType::Quadrilateral => a.checked_mul(a),
            CellType::Tetrahedron => Some(a.checked_mul(b)?.checked_mul(p - 3)? / 6),
            CellType::Hexahedron => a.checked_mul(a)?.checked_mul(a),
            CellType::Prism => Some(a.checked_mul(a)?.checked_mul(b)? / 2),
            CellType::Pyramid => Some(a.checked_mul(b)?.checked_mul(2 * p - 3)? / 6),
            CellType::Point => unreachable!(),
        }
    })
}

/// The number of nodes of one sub-entity block: 1 for a vertex, else the interior count.
fn block_size(entity: CellType, degree: u32) -> usize {
    match entity {
        CellType::Point => 1,
        _ => num_interior_lagrange_nodes(entity, degree),
    }
}

/// The Lagrange nodes of degree `degree` on `cell` in internal order (A.5), flat and
/// node-major with [`tdim`](crate::tdim) coordinates per node.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::{NodeFamily, lagrange_points};
///
/// // Vertices, then the edges (0,1), (0,2), (1,2) from their first vertex to their second.
/// let points = lagrange_points(CellType::Triangle, 2, NodeFamily::Equispaced).unwrap();
/// assert_eq!(points, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5]);
/// ```
///
/// # Errors
///
/// - [`RefCellError::UnsupportedDegree`] if `degree == 0`;
/// - [`RefCellError::UnsupportedNodeFamily`] for GLL on a cell other than an interval,
///   quadrilateral or hexahedron.
///
/// # Panics
///
/// If the Newton iteration for a GLL point does not converge (tested for `degree` ≤ 39).
pub fn lagrange_points(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<f64>, RefCellError> {
    if degree == 0 {
        return Err(RefCellError::UnsupportedDegree { cell, degree });
    }
    let p = degree as usize;
    // The interior 1D points along an edge, as fractions of the edge.
    let line: Vec<f64> = match family {
        NodeFamily::Equispaced => (1..p).map(|i| i as f64 / p as f64).collect(),
        NodeFamily::Gll => match cell {
            CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron => {
                let gll = crate::jacobi::gll_points(p + 1);
                gll[1..p].to_vec()
            }
            _ => return Err(RefCellError::UnsupportedNodeFamily { cell, family }),
        },
    };
    let td = tdim(cell);
    let coords = vertices(cell);
    let mut out = Vec::with_capacity(num_lagrange_nodes(cell, degree) * td);
    for dim in 0..=td {
        for (entity, &entity_type) in sub_entities(cell, dim)
            .iter()
            .zip(sub_entity_types(cell, dim))
        {
            let v = |k: usize| coords[entity[k]];
            // v0 + Σ_k t_k (v_{k+1} − v0), in ndelement's order of operations.
            let mut push = |t: &[f64], axes: &[usize]| {
                for j in 0..td {
                    let mut x = v(0)[j];
                    for (tk, &a) in t.iter().zip(axes) {
                        x += tk * (v(a)[j] - v(0)[j]);
                    }
                    out.push(x);
                }
            };
            let frac = |i: usize| i as f64 / p as f64;
            match entity_type {
                CellType::Point => push(&[], &[]),
                CellType::Interval => {
                    for &t in &line {
                        push(&[t], &[1]);
                    }
                }
                CellType::Triangle => {
                    for i1 in 1..p {
                        for i0 in 1..p - i1 {
                            push(&[frac(i0), frac(i1)], &[1, 2]);
                        }
                    }
                }
                CellType::Quadrilateral => {
                    for &t1 in &line {
                        for &t0 in &line {
                            push(&[t0, t1], &[1, 2]);
                        }
                    }
                }
                CellType::Tetrahedron => {
                    for i2 in 1..p {
                        for i1 in 1..p - i2 {
                            for i0 in 1..p - i2 - i1 {
                                push(&[frac(i0), frac(i1), frac(i2)], &[1, 2, 3]);
                            }
                        }
                    }
                }
                CellType::Hexahedron => {
                    for &t2 in &line {
                        for &t1 in &line {
                            for &t0 in &line {
                                push(&[t0, t1, t2], &[1, 2, 4]);
                            }
                        }
                    }
                }
                CellType::Prism => {
                    for i2 in 1..p {
                        for i1 in 1..p {
                            for i0 in 1..p - i1 {
                                push(&[frac(i0), frac(i1), frac(i2)], &[1, 2, 3]);
                            }
                        }
                    }
                }
                CellType::Pyramid => {
                    // (i0/p, i1/p, i2/p) on the reference pyramid itself (A.5).
                    for i2 in 1..p.saturating_sub(1) {
                        for i1 in 1..p - i2 {
                            for i0 in 1..p - i2 {
                                out.extend([frac(i0), frac(i1), frac(i2)]);
                            }
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), num_lagrange_nodes(cell, degree) * td);
    Ok(out)
}

/// The node indices (a range of the internal order) of the Lagrange nodes inside sub-entity
/// `index` of dimension `dim`, excluding its boundary (A.5 block layout). For `dim = 0` this
/// is the vertex's single node.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::lagrange_entity_nodes;
///
/// // A prism of degree 3: 6 vertices, 9 edges of 2 nodes, then triangle face 0 (1 node) and
/// // quadrilateral face 1 (4 nodes).
/// assert_eq!(lagrange_entity_nodes(CellType::Prism, 3, 1, 0), 6..8);
/// assert_eq!(lagrange_entity_nodes(CellType::Prism, 3, 2, 0), 24..25);
/// assert_eq!(lagrange_entity_nodes(CellType::Prism, 3, 2, 1), 25..29);
/// ```
///
/// # Panics
///
/// If `degree == 0`, `dim > tdim(cell)` or `index` is not a sub-entity of that dimension.
pub fn lagrange_entity_nodes(
    cell: CellType,
    degree: u32,
    dim: usize,
    index: usize,
) -> Range<usize> {
    assert!(
        degree >= 1,
        "lagrange_entity_nodes: degree 0 has no block layout"
    );
    assert!(
        dim <= tdim(cell),
        "lagrange_entity_nodes: {cell:?} has no sub-entities of dimension {dim}"
    );
    let types = sub_entity_types(cell, dim);
    assert!(
        index < types.len(),
        "lagrange_entity_nodes: {cell:?} has {} sub-entities of dimension {dim}, not {}",
        types.len(),
        index + 1
    );
    let before: usize = (0..dim)
        .flat_map(|d| sub_entity_types(cell, d))
        .chain(&types[..index])
        .map(|&t| block_size(t, degree))
        .sum();
    before..before + block_size(types[index], degree)
}

#[cfg(test)]
mod tests {
    use super::*;
    use CellType::*;

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
    const VOLUME_CELLS: [CellType; 7] = [
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ];

    fn families(cell: CellType) -> Vec<NodeFamily> {
        match cell {
            Interval | Quadrilateral | Hexahedron => vec![NodeFamily::Equispaced, NodeFamily::Gll],
            _ => vec![NodeFamily::Equispaced],
        }
    }

    fn point(points: &[f64], td: usize, k: usize) -> &[f64] {
        &points[k * td..(k + 1) * td]
    }

    #[test]
    fn blocks_tile_the_node_range() {
        for cell in CELLS {
            for p in 1..=6 {
                let mut next = 0;
                for dim in 0..=tdim(cell) {
                    for index in 0..sub_entities(cell, dim).len() {
                        let r = lagrange_entity_nodes(cell, p, dim, index);
                        assert_eq!(r.start, next, "{cell:?} p = {p}, entity ({dim}, {index})");
                        next = r.end;
                    }
                }
                assert_eq!(next, num_lagrange_nodes(cell, p), "{cell:?} p = {p}");
                let interior = lagrange_entity_nodes(cell, p, tdim(cell), 0);
                if cell != Point {
                    assert_eq!(interior.len(), num_interior_lagrange_nodes(cell, p));
                }
            }
        }
    }

    #[test]
    fn vertices_come_first() {
        for cell in VOLUME_CELLS {
            for family in families(cell) {
                let points = lagrange_points(cell, 3, family).unwrap();
                let td = tdim(cell);
                for (k, v) in vertices(cell).iter().enumerate() {
                    assert_eq!(point(&points, td, k), *v, "{cell:?} {family:?}");
                }
            }
        }
    }

    /// Each node of an entity block lies on that entity and not on any of its vertices: on the
    /// segment for an edge, on the face's plane for a face of a three-dimensional cell.
    #[test]
    fn points_lie_on_their_entity() {
        for cell in VOLUME_CELLS {
            let td = tdim(cell);
            for family in families(cell) {
                for p in 1..=5 {
                    let points = lagrange_points(cell, p, family).unwrap();
                    for dim in 1..td {
                        for (index, entity) in sub_entities(cell, dim).iter().enumerate() {
                            let v: Vec<&[f64]> =
                                entity.iter().map(|&k| vertices(cell)[k]).collect();
                            for k in lagrange_entity_nodes(cell, p, dim, index) {
                                let x = point(&points, td, k);
                                let d: Vec<f64> = (0..td).map(|j| x[j] - v[0][j]).collect();
                                let e1: Vec<f64> = (0..td).map(|j| v[1][j] - v[0][j]).collect();
                                if dim == 1 {
                                    // d parallel to e1 with 0 < t < 1.
                                    let t = (0..td).map(|j| d[j] * e1[j]).sum::<f64>()
                                        / (0..td).map(|j| e1[j] * e1[j]).sum::<f64>();
                                    let cross: f64 =
                                        (0..td).map(|j| (d[j] - t * e1[j]).abs()).sum();
                                    assert!(
                                        cross < 1e-14 && t > 0.0 && t < 1.0,
                                        "{cell:?} edge {entity:?}: {x:?}"
                                    );
                                } else {
                                    let e2: Vec<f64> = (0..td).map(|j| v[2][j] - v[0][j]).collect();
                                    let n = [
                                        e1[1] * e2[2] - e1[2] * e2[1],
                                        e1[2] * e2[0] - e1[0] * e2[2],
                                        e1[0] * e2[1] - e1[1] * e2[0],
                                    ];
                                    let dot: f64 = (0..3).map(|j| n[j] * d[j]).sum();
                                    assert!(dot.abs() < 1e-14, "{cell:?} face {entity:?}: {x:?}");
                                }
                                assert!(
                                    v.iter().all(|w| *w != x),
                                    "{cell:?}: node {k} is a vertex"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// A.5's in-entity rules, written out for small degrees.
    #[test]
    fn in_entity_rules_of_a5() {
        let eq = NodeFamily::Equispaced;
        let block = |cell, p, dim, index| {
            let points = lagrange_points(cell, p, eq).unwrap();
            let td = tdim(cell);
            lagrange_entity_nodes(cell, p, dim, index)
                .flat_map(|k| point(&points, td, k).to_vec())
                .collect::<Vec<f64>>()
        };
        // Edge (1, 3) of a quadrilateral, from vertex 1 towards vertex 3.
        assert_eq!(
            block(Quadrilateral, 3, 1, 2),
            [1.0, 1.0 / 3.0, 1.0, 2.0 / 3.0]
        );
        // Triangle interior, i1 outer and i0 inner.
        assert_eq!(block(Triangle, 4, 2, 0), [0.25, 0.25, 0.5, 0.25, 0.25, 0.5]);
        // Hexahedron face 3 (1, 3, 5, 7) on x = 1: first face coordinate (towards 3, i.e. y)
        // fastest.
        let third = 1.0 / 3.0;
        let two = 2.0 / 3.0;
        assert_eq!(
            block(Hexahedron, 3, 2, 3),
            [
                1.0, third, third, 1.0, two, third, 1.0, third, two, 1.0, two, two
            ]
        );
        // Triangle face 3 (1, 2, 3) of a tetrahedron, p = 3: v1 + (1/3)(v2 − v1) + (1/3)(v3 − v1).
        let face = block(Tetrahedron, 3, 2, 3);
        assert_eq!(face.len(), 3);
        for (got, want) in face.iter().zip([third, third, third]) {
            assert!((got - want).abs() < 1e-15);
        }
        // Tetrahedron interior, p = 5: i2 outermost, i0 innermost.
        let want = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]];
        let got = block(Tetrahedron, 5, 3, 0);
        assert_eq!(got.len(), 12);
        for (x, w) in got.chunks(3).zip(want) {
            for j in 0..3 {
                assert!(
                    (x[j] - f64::from(w[j]) / 5.0).abs() < 1e-15,
                    "{x:?} vs {w:?}"
                );
            }
        }
        // Prism interior, p = 4: 1 ≤ i2 ≤ 3, i0 + i1 ≤ 3, i0 innermost.
        let got = block(Prism, 4, 3, 0);
        let want: Vec<f64> = (1..4)
            .flat_map(|i2| [[1, 1, i2], [2, 1, i2], [1, 2, i2]])
            .flat_map(|w| w.map(|i| f64::from(i) / 4.0))
            .collect();
        assert_eq!(got, want);
        // Pyramid interior, p = 4: i2 = 1 gives i0, i1 in 1..=2; i2 = 2 gives (1, 1).
        let got = block(Pyramid, 4, 3, 0);
        let want: Vec<f64> = [[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2]]
            .iter()
            .flat_map(|w| w.map(|i| f64::from(i) / 4.0))
            .collect();
        assert_eq!(got, want);
    }

    #[test]
    fn gll_only_on_tensor_cells() {
        for cell in [Triangle, Tetrahedron, Prism, Pyramid, Point] {
            assert_eq!(
                lagrange_points(cell, 2, NodeFamily::Gll),
                Err(RefCellError::UnsupportedNodeFamily {
                    cell,
                    family: NodeFamily::Gll
                })
            );
        }
        // GLL points of degree 2 are the equispaced ones; they differ from degree 3 on.
        for cell in [Interval, Quadrilateral, Hexahedron] {
            let gll = lagrange_points(cell, 2, NodeFamily::Gll).unwrap();
            let eq = lagrange_points(cell, 2, NodeFamily::Equispaced).unwrap();
            assert!(gll.iter().zip(&eq).all(|(a, b)| (a - b).abs() < 1e-15));
            assert_ne!(
                lagrange_points(cell, 3, NodeFamily::Gll),
                lagrange_points(cell, 3, NodeFamily::Equispaced)
            );
        }
    }

    #[test]
    fn degree_zero_is_unsupported() {
        for cell in CELLS {
            assert_eq!(
                lagrange_points(cell, 0, NodeFamily::Equispaced),
                Err(RefCellError::UnsupportedDegree { cell, degree: 0 })
            );
            assert_eq!(num_lagrange_nodes(cell, 0), 1);
            assert_eq!(num_interior_lagrange_nodes(cell, 0), 0);
        }
    }

    #[test]
    #[should_panic(expected = "Hexahedron cells of degree 4294967295 overflows")]
    fn node_count_overflow_panics() {
        num_lagrange_nodes(Hexahedron, u32::MAX);
    }
}
