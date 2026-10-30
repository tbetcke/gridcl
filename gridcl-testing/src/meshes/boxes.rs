//! Structured boxes of every cell type, the mixed tetrahedron/pyramid/hexahedron box, the
//! hexahedron/prism/pyramid box and the P1 triangle strip.
//!
//! Every box lives on the integer lattice of resolution 2·n·p: subcell corners are multiples of
//! 2p, so the centres of hexahedral subcells (the pyramid apices) and every equispaced Lagrange
//! node of degree p are lattice points. Cells are numbered with x fastest, then y, then z, and
//! the cells of one subcell are consecutive.

use gridcl_dims::{Cell2, Cell3, CellType};

use super::HostMesh;
use super::lattice::{Assembler, Key, orient};

/// The hexahedron faces as vertex lists in tensor order (sub-entities of dimension 2, A.3).
fn hex_faces() -> &'static [&'static [usize]] {
    gridcl_refcell::sub_entities(CellType::Hexahedron, 2)
}

/// The face x = 0 of the reference hexahedron.
const HEX_FACE_X0: [usize; 4] = [0, 2, 4, 6];

/// Panics unless `n` and `p` are positive.
fn check_args(n: usize, p: u32) {
    assert!(n > 0, "a box needs n ≥ 1 cells per direction");
    assert!(p > 0, "a box needs degree p ≥ 1");
}

/// The lattice resolution 2·n·p.
fn resolution(n: usize, p: u32) -> i64 {
    2i64.checked_mul(i64::try_from(n).expect("n fits i64"))
        .and_then(|r| r.checked_mul(i64::from(p)))
        .expect("lattice resolution overflows i64")
}

/// The keys of the 8 corners of hexahedral subcell (i, j, k), in tensor order, and its centre.
fn subcell(i: usize, j: usize, k: usize, p: u32) -> ([Key; 8], Key) {
    let h = 2 * i64::from(p);
    let o = [i as i64 * h, j as i64 * h, k as i64 * h];
    let corners = core::array::from_fn(|v| {
        let bit = |b: usize| ((v >> b) & 1) as i64;
        [o[0] + bit(0) * h, o[1] + bit(1) * h, o[2] + bit(2) * h]
    });
    let half = i64::from(p);
    (corners, [o[0] + half, o[1] + half, o[2] + half])
}

/// The pyramid on `face` of a hexahedral subcell with apex at its centre, positively oriented.
fn pyramid(corners: &[Key; 8], face: &[usize], centre: Key) -> [Key; 5] {
    let mut verts = [
        corners[face[0]],
        corners[face[1]],
        corners[face[2]],
        corners[face[3]],
        centre,
    ];
    orient(CellType::Pyramid, &mut verts);
    verts
}

/// Adds the two tetrahedra that split the pyramid `pyr` along its base diagonal v0–v3.
fn split_pyramid(mesh: &mut Assembler, p: u32, pyr: &[Key; 5]) {
    let [b0, b1, b2, b3, apex] = *pyr;
    for mut tet in [[b0, b1, b3, apex], [b0, b3, b2, apex]] {
        orient(CellType::Tetrahedron, &mut tet);
        mesh.cell(CellType::Tetrahedron, p, &tet);
    }
}

/// The unit interval [0, 1] with `n` cells of degree `p` (gdim 1).
///
/// # Panics
///
/// If `n` or `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::interval_box(4, 2);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (1, 4, 9));
/// ```
pub fn interval_box(n: usize, p: u32) -> HostMesh {
    check_args(n, p);
    let mut mesh = Assembler::new(1, resolution(n, p));
    let h = 2 * i64::from(p);
    for i in 0..n as i64 {
        mesh.cell(CellType::Interval, p, &[[i * h, 0, 0], [(i + 1) * h, 0, 0]]);
    }
    mesh.finish()
}

/// The unit square with n × n quadrilaterals of degree `p`, or 2n² triangles, each square split
/// along its diagonal (gdim 2).
///
/// # Panics
///
/// If `n` or `p` is 0.
///
/// # Example
///
/// ```
/// use gridcl_dims::Cell2;
///
/// let quads = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1);
/// assert_eq!((quads.num_cells(), quads.num_nodes()), (4, 9));
/// let triangles = gridcl_testing::square_box(Cell2::Triangle, 2, 2);
/// assert_eq!((triangles.num_cells(), triangles.num_nodes()), (8, 25));
/// ```
pub fn square_box(cell: Cell2, n: usize, p: u32) -> HostMesh {
    check_args(n, p);
    let mut mesh = Assembler::new(2, resolution(n, p));
    for j in 0..n {
        for i in 0..n {
            // The bottom face of the subcell (i, j, 0) is the square, in tensor order.
            let (c, _) = subcell(i, j, 0, p);
            match cell {
                Cell2::Quadrilateral => {
                    mesh.cell(CellType::Quadrilateral, p, &[c[0], c[1], c[2], c[3]]);
                }
                Cell2::Triangle => {
                    // The diagonal from (0, 0) to (1, 1) in every square.
                    for mut tri in [[c[0], c[1], c[3]], [c[0], c[3], c[2]]] {
                        orient(CellType::Triangle, &mut tri);
                        mesh.cell(CellType::Triangle, p, &tri);
                    }
                }
            }
        }
    }
    mesh.finish()
}

/// The unit cube with n³ hexahedra of degree `p`: 6n³ tetrahedra (Kuhn split), 2n³ prisms
/// (each hexahedron split along a diagonal of its z-faces) or 6n³ pyramids (each hexahedron
/// split into six around its centre) (gdim 3).
///
/// # Panics
///
/// If `n` or `p` is 0.
///
/// # Example
///
/// ```
/// use gridcl_dims::Cell3;
///
/// let hexes = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 1);
/// assert_eq!((hexes.num_cells(), hexes.num_nodes()), (8, 27));
/// let tets = gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1);
/// assert_eq!(tets.num_cells(), 48);
/// let prisms = gridcl_testing::cube_box(Cell3::Prism, 1, 2);
/// assert_eq!((prisms.num_cells(), prisms.num_nodes()), (2, 27));
/// // The pyramids add the centre of every hexahedron.
/// let pyramids = gridcl_testing::cube_box(Cell3::Pyramid, 1, 1);
/// assert_eq!((pyramids.num_cells(), pyramids.num_nodes()), (6, 9));
/// ```
pub fn cube_box(cell: Cell3, n: usize, p: u32) -> HostMesh {
    check_args(n, p);
    let mut mesh = Assembler::new(3, resolution(n, p));
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (c, centre) = subcell(i, j, k, p);
                match cell {
                    Cell3::Hexahedron => mesh.cell(CellType::Hexahedron, p, &c),
                    Cell3::Tetrahedron => {
                        // Kuhn: one tetrahedron per path 0 → e_a → e_a + e_b → 7 along the main
                        // diagonal, the same in every hexahedron, so face diagonals conform.
                        for (a, b) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)] {
                            let ea = 1 << a;
                            let mut tet = [c[0], c[ea], c[ea | (1 << b)], c[7]];
                            orient(CellType::Tetrahedron, &mut tet);
                            mesh.cell(CellType::Tetrahedron, p, &tet);
                        }
                    }
                    Cell3::Prism => {
                        // The z-faces split along the diagonal (0, 0)–(1, 1), extruded in z.
                        for [a, b, d] in [[0, 1, 3], [0, 3, 2]] {
                            let mut prism = [c[a], c[b], c[d], c[a + 4], c[b + 4], c[d + 4]];
                            orient(CellType::Prism, &mut prism);
                            mesh.cell(CellType::Prism, p, &prism);
                        }
                    }
                    Cell3::Pyramid => {
                        for face in hex_faces() {
                            let pyr = pyramid(&c, face, centre);
                            mesh.cell(CellType::Pyramid, p, &pyr);
                        }
                    }
                }
            }
        }
    }
    mesh.finish()
}

/// A conforming tetrahedron/pyramid/hexahedron box on the unit cube with n³ hexahedral
/// subcells of degree `p`: subcells with an even index sum are split into six pyramids around
/// their centre, and those pyramids whose base lies on the face x = 0 into two tetrahedra; the
/// other subcells stay hexahedra.
///
/// # Panics
///
/// If `n` or `p` is 0.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// // Subcells (0, 0, 0) and (1, 1, 0), (1, 0, 1), (0, 1, 1) are split; two of them touch x = 0.
/// let h = gridcl_testing::mixed_box(2, 1);
/// let blocks: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
/// assert_eq!(
///     blocks,
///     [
///         (CellType::Tetrahedron, 4),
///         (CellType::Hexahedron, 4),
///         (CellType::Pyramid, 22),
///     ]
/// );
/// ```
pub fn mixed_box(n: usize, p: u32) -> HostMesh {
    check_args(n, p);
    let mut mesh = Assembler::new(3, resolution(n, p));
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (c, centre) = subcell(i, j, k, p);
                if (i + j + k) % 2 == 1 {
                    mesh.cell(CellType::Hexahedron, p, &c);
                    continue;
                }
                for face in hex_faces() {
                    let pyr = pyramid(&c, face, centre);
                    if i == 0 && *face == HEX_FACE_X0 {
                        split_pyramid(&mut mesh, p, &pyr);
                    } else {
                        mesh.cell(CellType::Pyramid, p, &pyr);
                    }
                }
            }
        }
    }
    mesh.finish()
}

/// A conforming hexahedron/prism/pyramid box on the unit cube with n³ hexahedral subcells of
/// degree `p`, the M7 acceptance mesh (docs/plans/M7.md, question 14): the subcells of the
/// columns i < ⌊n/3⌋ are split into two prisms each, extruded in z like `cube_box`'s prisms,
/// so triangles lie only between prisms and on z = 0, 1; of the other subcells, those with an
/// even index sum are split into six pyramids around their centre and the rest stay
/// hexahedra. Neighbouring subcells of different kinds meet in whole quadrilaterals.
///
/// # Panics
///
/// If `n` or `p` is 0.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// // n = 3: one column of prism subcells (18 prisms); of the other 18 subcells, 9 have an even
/// // index sum.
/// let h = gridcl_testing::hex_prism_pyramid_box(3, 1);
/// let blocks: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
/// assert_eq!(
///     blocks,
///     [
///         (CellType::Hexahedron, 9),
///         (CellType::Prism, 18),
///         (CellType::Pyramid, 54),
///     ]
/// );
/// ```
pub fn hex_prism_pyramid_box(n: usize, p: u32) -> HostMesh {
    check_args(n, p);
    let prism_columns = n / 3;
    let mut mesh = Assembler::new(3, resolution(n, p));
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let (c, centre) = subcell(i, j, k, p);
                if i < prism_columns {
                    for [a, b, d] in [[0, 1, 3], [0, 3, 2]] {
                        let mut prism = [c[a], c[b], c[d], c[a + 4], c[b + 4], c[d + 4]];
                        orient(CellType::Prism, &mut prism);
                        mesh.cell(CellType::Prism, p, &prism);
                    }
                } else if (i + j + k) % 2 == 0 {
                    for face in hex_faces() {
                        let pyr = pyramid(&c, face, centre);
                        mesh.cell(CellType::Pyramid, p, &pyr);
                    }
                } else {
                    mesh.cell(CellType::Hexahedron, p, &c);
                }
            }
        }
    }
    mesh.finish()
}

/// `num_cells` P1 triangles in a strip along the x axis, [0, ⌈num_cells/2⌉] × [0, 1] (gdim 2).
///
/// Square i holds triangles 2i and 2i + 1; with an odd count the last square holds one.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::triangle_strip(5);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (2, 5, 8));
/// ```
pub fn triangle_strip(num_cells: usize) -> HostMesh {
    // Direct construction (the strip is large): node (x, y) is 2x + y. The squares are split
    // along the diagonal (0, 0)–(1, 1) like `square_box`; both triangles are positive.
    let squares = num_cells.div_ceil(2);
    let num_nodes = if num_cells == 0 { 0 } else { 2 * (squares + 1) };
    let node = |x: usize, y: usize| u32::try_from(2 * x + y).expect("node index fits u32");
    let nodes = (0..num_nodes)
        .flat_map(|i| [(i / 2) as f64, (i % 2) as f64])
        .collect();
    let mut cells = Vec::with_capacity(6 * squares);
    for x in 0..squares {
        let (q0, q1, q2, q3) = (node(x, 0), node(x + 1, 0), node(x, 1), node(x + 1, 1));
        cells.extend([q0, q1, q3, q0, q3, q2]);
    }
    let full = HostMesh {
        gdim: 2,
        nodes,
        node_ids: None,
        blocks: vec![super::HostBlock {
            cell: CellType::Triangle,
            degree: 1,
            cells,
            cell_ids: None,
        }],
    };
    // Drop the second triangle of the last square for an odd count, and with it the node
    // (0, 1) if it was the only cell.
    let mut h = full.first_cells(num_cells);
    h.node_ids = Some((0..h.num_nodes() as i64).collect());
    h.blocks[0].cell_ids = Some((0..num_cells as i64).collect());
    h
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gridcl::{D2, D3, GridError};
    use gridcl_dims::{Cell2, Cell3, CellType};
    use gridcl_refcell::{
        LagrangeElement, NodeFamily, gauss_rule, lagrange_entity_nodes,
        num_interior_lagrange_nodes, sub_entities, sub_entity_types, tdim,
    };

    use super::super::HostMesh;
    use super::super::lattice::{PointGrid, SplitMix64};
    use super::{
        cube_box, hex_prism_pyramid_box, interval_box, mixed_box, square_box, triangle_strip,
    };

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
        Mixed,
        HexPrismPyramid,
    }

    const ALL: [Kind; 9] = [
        Kind::Interval,
        Kind::Triangle,
        Kind::Quadrilateral,
        Kind::Tetrahedron,
        Kind::Hexahedron,
        Kind::Prism,
        Kind::Pyramid,
        Kind::Mixed,
        Kind::HexPrismPyramid,
    ];

    fn host(kind: Kind, n: usize, p: u32) -> HostMesh {
        match kind {
            Kind::Interval => interval_box(n, p),
            Kind::Triangle => square_box(Cell2::Triangle, n, p),
            Kind::Quadrilateral => square_box(Cell2::Quadrilateral, n, p),
            Kind::Tetrahedron => cube_box(Cell3::Tetrahedron, n, p),
            Kind::Hexahedron => cube_box(Cell3::Hexahedron, n, p),
            Kind::Prism => cube_box(Cell3::Prism, n, p),
            Kind::Pyramid => cube_box(Cell3::Pyramid, n, p),
            Kind::Mixed => mixed_box(n, p),
            Kind::HexPrismPyramid => hex_prism_pyramid_box(n, p),
        }
    }

    fn kind_tdim(kind: Kind) -> usize {
        match kind {
            Kind::Interval => 1,
            Kind::Triangle | Kind::Quadrilateral => 2,
            _ => 3,
        }
    }

    /// The closed-form counts of gridcl/tests/common/mod.rs (`box_counts`): entities per
    /// dimension, and (dimension, type, count) for dimensions 1..tdim.
    fn box_counts(kind: Kind, n: usize) -> (Vec<usize>, Vec<(usize, CellType, usize)>) {
        use CellType::{Interval, Quadrilateral, Triangle};
        let v1 = n + 1;
        let hex_v = v1 * v1 * v1;
        let hex_e = 3 * n * v1 * v1;
        let hex_f = 3 * n * n * v1;
        let n3 = n * n * n;
        match kind {
            Kind::Interval => (vec![v1, n], vec![]),
            Kind::Quadrilateral => {
                let e = 2 * n * v1;
                (vec![v1 * v1, e, n * n], vec![(1, Interval, e)])
            }
            Kind::Triangle => {
                let e = 2 * n * v1 + n * n;
                (vec![v1 * v1, e, 2 * n * n], vec![(1, Interval, e)])
            }
            Kind::Hexahedron => (
                vec![hex_v, hex_e, hex_f, n3],
                vec![(1, Interval, hex_e), (2, Quadrilateral, hex_f)],
            ),
            Kind::Tetrahedron => {
                let e = hex_e + hex_f + n3;
                let f = 2 * hex_f + 6 * n3;
                (
                    vec![hex_v, e, f, 6 * n3],
                    vec![(1, Interval, e), (2, Triangle, f)],
                )
            }
            Kind::Prism => {
                let z_faces = n * n * v1;
                let e = hex_e + z_faces;
                let tri = 2 * z_faces;
                let quad = 2 * n * n * v1 + n3;
                (
                    vec![hex_v, e, tri + quad, 2 * n3],
                    vec![
                        (1, Interval, e),
                        (2, Triangle, tri),
                        (2, Quadrilateral, quad),
                    ],
                )
            }
            Kind::Pyramid => {
                let e = hex_e + 8 * n3;
                let tri = 12 * n3;
                (
                    vec![hex_v + n3, e, hex_f + tri, 6 * n3],
                    vec![
                        (1, Interval, e),
                        (2, Triangle, tri),
                        (2, Quadrilateral, hex_f),
                    ],
                )
            }
            Kind::Mixed => {
                let split = n3.div_ceil(2);
                let at_x0 = (n * n).div_ceil(2);
                let e = hex_e + 8 * split + at_x0;
                let tri = 12 * split + 3 * at_x0;
                let quad = hex_f - at_x0;
                let cells = n3 + 5 * split + at_x0;
                (
                    vec![hex_v + split, e, tri + quad, cells],
                    vec![
                        (1, Interval, e),
                        (2, Triangle, tri),
                        (2, Quadrilateral, quad),
                    ],
                )
            }
            Kind::HexPrismPyramid => {
                // m prism columns; `split` subcells with an even index sum outside them.
                let m = n / 3;
                let split = (m..n)
                    .flat_map(|i| (0..n * n).map(move |jk| i + jk / n + jk % n))
                    .filter(|s| s % 2 == 0)
                    .count();
                let z_diagonals = m * n * v1;
                let e = hex_e + z_diagonals + 8 * split;
                let tri = 2 * z_diagonals + 12 * split;
                let quad = hex_f - z_diagonals + m * n * n;
                let cells = 2 * m * n * n + 6 * split + (n - m) * n * n - split;
                (
                    vec![hex_v + split, e, tri + quad, cells],
                    vec![
                        (1, Interval, e),
                        (2, Triangle, tri),
                        (2, Quadrilateral, quad),
                    ],
                )
            }
        }
    }

    /// The node row of cell `c` of block `b`.
    fn row(h: &HostMesh, b: usize, c: usize) -> &[u32] {
        let k = h.blocks[b].nodes_per_cell();
        &h.blocks[b].cells[c * k..(c + 1) * k]
    }

    /// The coordinates of node `i`.
    fn x(h: &HostMesh, i: u32) -> &[f64] {
        let i = i as usize;
        &h.nodes[i * h.gdim..(i + 1) * h.gdim]
    }

    /// Every cell as (block, cell index).
    fn cells(h: &HostMesh) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..h.blocks.len()).flat_map(move |b| (0..h.blocks[b].num_cells()).map(move |c| (b, c)))
    }

    /// The distinct sub-entities of dimension `dim` (sorted vertex nodes) and their types.
    fn entities(h: &HostMesh, dim: usize) -> HashMap<Vec<u32>, CellType> {
        let mut out = HashMap::new();
        for (b, c) in cells(h) {
            let cell = h.blocks[b].cell;
            let nodes = row(h, b, c);
            for (ent, &ty) in sub_entities(cell, dim)
                .iter()
                .zip(sub_entity_types(cell, dim))
            {
                let mut key: Vec<u32> = ent.iter().map(|&v| nodes[v]).collect();
                key.sort_unstable();
                if let Some(old) = out.insert(key, ty) {
                    assert_eq!(old, ty, "one entity with two types");
                }
            }
        }
        out
    }

    #[test]
    fn entity_counts_match_the_closed_form() {
        for kind in ALL {
            for n in [1, 2, 4] {
                for p in [1, 2] {
                    let h = host(kind, n, p);
                    let td = kind_tdim(kind);
                    let (per_dim, per_type) = box_counts(kind, n);
                    let found: Vec<usize> = (0..=td).map(|d| entities(&h, d).len()).collect();
                    assert_eq!(found, per_dim, "{kind:?}, n = {n}, p = {p}");
                    assert_eq!(h.num_cells(), per_dim[td], "{kind:?}, n = {n}");
                    for (d, ty, count) in per_type {
                        let found = entities(&h, d).values().filter(|&&t| t == ty).count();
                        assert_eq!(found, count, "{kind:?}, n = {n}: {ty:?} of dim {d}");
                    }
                }
            }
        }
    }

    #[test]
    fn node_counts_match_the_entities_and_every_node_is_used() {
        for kind in ALL {
            for n in [1, 2] {
                for p in 1..=4 {
                    let h = host(kind, n, p);
                    let td = kind_tdim(kind);
                    let want: usize = (0..=td)
                        .map(|d| {
                            entities(&h, d)
                                .values()
                                .map(|&ty| {
                                    if d == 0 {
                                        1
                                    } else {
                                        num_interior_lagrange_nodes(ty, p)
                                    }
                                })
                                .sum::<usize>()
                        })
                        .sum();
                    assert_eq!(h.num_nodes(), want, "{kind:?}, n = {n}, p = {p}");
                    let used: HashSet<u32> = h
                        .blocks
                        .iter()
                        .flat_map(|b| b.cells.iter().copied())
                        .collect();
                    assert_eq!(used.len(), h.num_nodes(), "{kind:?}, n = {n}, p = {p}");
                }
            }
        }
    }

    #[test]
    fn node_coordinates_are_distinct_and_in_the_unit_box() {
        for kind in ALL {
            for n in [1, 2, 3] {
                for p in 1..=4 {
                    let h = host(kind, n, p);
                    assert!(h.nodes.iter().all(|&x| (0.0..=1.0).contains(&x)));
                    let distinct: HashSet<Vec<u64>> = h
                        .nodes
                        .chunks_exact(h.gdim)
                        .map(|x| x.iter().map(|v| v.to_bits()).collect())
                        .collect();
                    assert_eq!(distinct.len(), h.num_nodes(), "{kind:?}, n = {n}, p = {p}");
                }
            }
        }
    }

    #[test]
    fn interior_facets_are_shared_by_two_cells_and_the_rest_lie_on_the_boundary() {
        for kind in ALL {
            for n in [1, 2, 3] {
                for p in 1..=3 {
                    let h = host(kind, n, p);
                    let td = kind_tdim(kind);
                    // Facet vertex set → the sorted node lists of the facet in each cell.
                    let mut facets: HashMap<Vec<u32>, Vec<Vec<u32>>> = HashMap::new();
                    for (b, c) in cells(&h) {
                        let cell = h.blocks[b].cell;
                        let nodes = row(&h, b, c);
                        for facet in sub_entities(cell, td - 1) {
                            let mut key: Vec<u32> = facet.iter().map(|&v| nodes[v]).collect();
                            key.sort_unstable();
                            let mut on_facet = Vec::new();
                            for d in 0..td {
                                for (i, ent) in sub_entities(cell, d).iter().enumerate() {
                                    if ent.iter().all(|v| facet.contains(v)) {
                                        let range = lagrange_entity_nodes(cell, p, d, i);
                                        on_facet.extend(range.map(|k| nodes[k]));
                                    }
                                }
                            }
                            on_facet.sort_unstable();
                            facets.entry(key).or_default().push(on_facet);
                        }
                    }
                    for (key, lists) in &facets {
                        let what = format!("{kind:?}, n = {n}, p = {p}, facet {key:?}");
                        match lists.len() {
                            1 => {
                                let on_boundary = (0..h.gdim).any(|g| {
                                    [0.0, 1.0]
                                        .into_iter()
                                        .any(|side| lists[0].iter().all(|&v| x(&h, v)[g] == side))
                                });
                                assert!(on_boundary, "{what}: one cell, but not on the boundary");
                            }
                            2 => assert_eq!(lists[0], lists[1], "{what}: nodes differ"),
                            m => panic!("{what}: shared by {m} cells"),
                        }
                    }
                }
            }
        }
    }

    /// det J at the points of a Gauss rule of degree `q` in every cell, and Σ w det J.
    fn det_j(h: &HostMesh, q: u32) -> (Vec<f64>, f64) {
        let gd = h.gdim;
        let mut dets = Vec::new();
        let mut volume = 0.0;
        for (b, block) in h.blocks.iter().enumerate() {
            let td = tdim(block.cell);
            assert_eq!(td, gd, "det J of a full-dimensional cell");
            let rule = gauss_rule(block.cell, q);
            let element = LagrangeElement::new(block.cell, block.degree, NodeFamily::Equispaced)
                .expect("element");
            let tab = element.tabulate(&rule.points, 1).expect("tabulate");
            for c in 0..block.num_cells() {
                let nodes = row(h, b, c);
                for (iq, w) in rule.weights.iter().enumerate() {
                    // j[g][d] = Σ_i x_i[g] ∂φ_i/∂r_d
                    let mut j = [[0.0f64; 3]; 3];
                    for (i, &node) in nodes.iter().enumerate() {
                        for (xg, jg) in x(h, node).iter().zip(&mut j) {
                            for (d, jgd) in jg.iter_mut().enumerate().take(td) {
                                *jgd += xg * tab.get(1 + d, iq, i);
                            }
                        }
                    }
                    let det = match td {
                        1 => j[0][0],
                        2 => j[0][0] * j[1][1] - j[0][1] * j[1][0],
                        _ => {
                            j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                                - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                                + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0])
                        }
                    };
                    dets.push(det);
                    volume += w * det;
                }
            }
        }
        (dets, volume)
    }

    #[test]
    fn det_j_is_positive_and_the_volume_is_one() {
        for kind in ALL {
            for n in [1, 2, 3] {
                for p in 1..=4 {
                    let plain = host(kind, n, p);
                    let moved = plain.perturbed(0.05);
                    for (what, h) in [("plain", &plain), ("perturbed", &moved)] {
                        // Degree 3p integrates det J of every map here exactly.
                        let (dets, volume) = det_j(h, 3 * p);
                        let what = format!("{kind:?}, n = {n}, p = {p}, {what}");
                        let min = dets.iter().fold(f64::INFINITY, |m, &d| m.min(d));
                        assert!(min > 0.0, "{what}: det J = {min}");
                        assert!((volume - 1.0).abs() < 1e-12, "{what}: volume {volume}");
                    }
                }
            }
        }
    }

    #[test]
    fn perturbed_mixed_boxes_stay_valid_at_fine_n() {
        // The hexahedra of the mixed box have every face pinned; without the cut-off at pyramid
        // nodes they fold at n = 3, p = 4. Here det J must stay within a factor 2 of the
        // unperturbed value in every block.
        for n in [3, 4, 5] {
            let p = 4;
            let plain = mixed_box(n, p);
            let moved = plain.perturbed(0.05);
            for b in 0..plain.blocks.len() {
                let one = |h: &HostMesh| HostMesh {
                    blocks: vec![h.blocks[b].clone()],
                    ..h.clone()
                };
                let (want, _) = det_j(&one(&plain), 3 * p);
                let (got, _) = det_j(&one(&moved), 3 * p);
                for (g, w) in got.iter().zip(&want) {
                    assert!(g / w > 0.5 && g / w < 2.0, "n = {n}, block {b}: {g} vs {w}");
                }
                if plain.blocks[b].cell != CellType::Pyramid {
                    assert_ne!(got, want, "n = {n}, block {b}: not perturbed");
                }
            }
        }
    }

    #[test]
    fn point_grid_distances_match_brute_force() {
        let mut rng = SplitMix64::new(11);
        let mut uniform = || (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        for gdim in 1..=3 {
            for num in [0, 1, 7, 200] {
                let points: Vec<f64> = (0..num * gdim).map(|_| uniform()).collect();
                let grid = PointGrid::new(gdim, &points);
                for _ in 0..50 {
                    let x: Vec<f64> = (0..gdim).map(|_| 1.4 * uniform() - 0.2).collect();
                    let brute = points
                        .chunks_exact(gdim)
                        .map(|y| x.iter().zip(y).map(|(u, v)| (u - v) * (u - v)).sum::<f64>())
                        .fold(f64::INFINITY, f64::min)
                        .sqrt();
                    for cap in [0.05, 1.0] {
                        assert_eq!(grid.distance(&x, cap), brute.min(cap), "{gdim}D, {num}");
                    }
                }
            }
        }
    }

    #[test]
    fn perturbation_moves_only_interior_nodes_off_pyramids() {
        for kind in ALL {
            let h = host(kind, 2, 3);
            let moved = h.perturbed(0.05);
            let pyramid_nodes: HashSet<u32> = h
                .blocks
                .iter()
                .filter(|b| b.cell == CellType::Pyramid)
                .flat_map(|b| b.cells.iter().copied())
                .collect();
            let mut num_moved = 0;
            for i in 0..h.num_nodes() as u32 {
                let (a, b) = (x(&h, i), x(&moved, i));
                let boundary = a.iter().any(|&v| v == 0.0 || v == 1.0);
                if boundary || pyramid_nodes.contains(&i) {
                    assert_eq!(a, b, "{kind:?}: node {i} moved");
                } else {
                    assert_ne!(a, b, "{kind:?}: node {i} stayed");
                    num_moved += 1;
                }
            }
            assert_eq!(num_moved > 0, kind != Kind::Pyramid, "{kind:?}");
            assert_eq!(moved.blocks, h.blocks);
        }
    }

    #[test]
    fn generators_number_nodes_and_cells_by_index() {
        for kind in ALL {
            let h = host(kind, 2, 2);
            let n = h.num_nodes() as i64;
            assert_eq!(h.node_ids, Some((0..n).collect()));
            let ids: Vec<i64> = h
                .blocks
                .iter()
                .flat_map(|b| b.cell_ids.clone().expect("cell ids"))
                .collect();
            assert_eq!(ids, (0..h.num_cells() as i64).collect::<Vec<_>>());
            let types: Vec<CellType> = h.blocks.iter().map(|b| b.cell).collect();
            let mut sorted = types.clone();
            sorted.sort();
            sorted.dedup();
            assert_eq!(types, sorted, "{kind:?}: one block per type, in type order");
        }
    }

    /// The multiset of (id, coordinate bits) of the nodes.
    fn node_multiset(h: &HostMesh) -> Vec<(i64, Vec<u64>)> {
        let mut out: Vec<_> = (0..h.num_nodes())
            .map(|i| {
                let bits = x(h, i as u32).iter().map(|v| v.to_bits()).collect();
                (h.node_id(i), bits)
            })
            .collect();
        out.sort();
        out
    }

    /// Every cell as its list of node ids.
    fn cell_id_lists(h: &HostMesh) -> Vec<Vec<i64>> {
        cells(h)
            .map(|(b, c)| {
                row(h, b, c)
                    .iter()
                    .map(|&i| h.node_id(i as usize))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn shuffled_nodes_are_a_permutation() {
        for kind in ALL {
            let h = host(kind, 2, 2).perturbed(0.05);
            for seed in [0, 1, 0xdead_beef] {
                let s = h.with_shuffled_nodes(seed);
                assert_eq!(node_multiset(&s), node_multiset(&h), "{kind:?}");
                assert_eq!(cell_id_lists(&s), cell_id_lists(&h), "{kind:?}");
                assert_eq!(s, h.with_shuffled_nodes(seed), "deterministic");
                for (a, b) in s.blocks.iter().zip(&h.blocks) {
                    assert_eq!(a.cell_ids, b.cell_ids);
                }
            }
            assert_ne!(
                h.with_shuffled_nodes(1).nodes,
                h.nodes,
                "{kind:?}: not shuffled"
            );
            // Index-based ids are materialised.
            let bare = HostMesh {
                node_ids: None,
                ..h.clone()
            };
            let s = bare.with_shuffled_nodes(5);
            assert_eq!(node_multiset(&s), node_multiset(&h));
            assert_eq!(cell_id_lists(&s), cell_id_lists(&h));
        }
    }

    #[test]
    fn shuffled_ids_are_injective_and_in_range() {
        const LIMIT: i64 = 1 << 40;
        for kind in ALL {
            let h = host(kind, 2, 2).with_shuffled_nodes(9);
            let s = h.with_shuffled_ids(42);
            assert_eq!(s.nodes, h.nodes);
            assert_eq!(s, h.with_shuffled_ids(42), "deterministic");
            assert_ne!(s.node_ids, h.with_shuffled_ids(43).node_ids);
            let node_ids = s.node_ids.clone().expect("node ids");
            assert_eq!(node_ids.len(), h.num_nodes());
            let cell_ids: Vec<i64> = s
                .blocks
                .iter()
                .flat_map(|b| b.cell_ids.clone().expect("cell ids"))
                .collect();
            assert_eq!(cell_ids.len(), h.num_cells());
            for ids in [&node_ids, &cell_ids] {
                assert!(ids.iter().all(|id| (-LIMIT..=LIMIT).contains(id)));
                let distinct: HashSet<i64> = ids.iter().copied().collect();
                assert_eq!(distinct.len(), ids.len(), "{kind:?}: duplicate ids");
            }
            assert!(
                node_ids.iter().any(|&id| id < 0),
                "{kind:?}: no negative ids"
            );
            assert!(
                node_ids.iter().any(|&id| id >= 1 << 32),
                "{kind:?}: no wide ids"
            );
            // The same cells, under the new ids of their nodes.
            let new_of_old: HashMap<i64, i64> = (0..h.num_nodes())
                .map(|i| (h.node_id(i), s.node_id(i)))
                .collect();
            let mapped: Vec<Vec<i64>> = cell_id_lists(&h)
                .into_iter()
                .map(|l| l.into_iter().map(|id| new_of_old[&id]).collect())
                .collect();
            assert_eq!(mapped, cell_id_lists(&s));
            for (a, b) in s.blocks.iter().zip(&h.blocks) {
                assert_eq!((a.cell, a.degree, &a.cells), (b.cell, b.degree, &b.cells));
            }
        }
    }

    #[test]
    fn first_cells_keeps_the_first_cells_and_their_nodes() {
        for kind in ALL {
            let h = host(kind, 2, 2).with_shuffled_ids(3).with_shuffled_nodes(4);
            for keep in [0, 1, 3, 1000] {
                let f = h.first_cells(keep);
                assert_eq!(f.blocks.len(), h.blocks.len());
                for (b, (fb, hb)) in f.blocks.iter().zip(&h.blocks).enumerate() {
                    let m = keep.min(hb.num_cells());
                    assert_eq!(fb.num_cells(), m);
                    let want_ids = hb.cell_ids.as_deref().map(|ids| &ids[..m]);
                    assert_eq!(fb.cell_ids.as_deref(), want_ids);
                    for c in 0..m {
                        for (&a, &o) in row(&f, b, c).iter().zip(row(&h, b, c)) {
                            assert_eq!(x(&f, a), x(&h, o));
                            assert_eq!(f.node_id(a as usize), h.node_id(o as usize));
                        }
                    }
                }
                let used: HashSet<u32> = f
                    .blocks
                    .iter()
                    .flat_map(|b| b.cells.iter().copied())
                    .collect();
                assert_eq!(used.len(), f.num_nodes(), "{kind:?}: unused nodes");
                // Relative order kept: the ids appear in the order they had.
                let pos: HashMap<i64, usize> =
                    (0..h.num_nodes()).map(|i| (h.node_id(i), i)).collect();
                let order: Vec<usize> = (0..f.num_nodes()).map(|i| pos[&f.node_id(i)]).collect();
                assert!(order.windows(2).all(|w| w[0] < w[1]), "{kind:?}: order");
            }
            assert_eq!(h.first_cells(usize::MAX), h);
        }
    }

    #[test]
    fn the_triangle_strip_has_the_requested_cells() {
        for num in [0, 1, 2, 5, 16, 17, 1001] {
            let h = triangle_strip(num);
            assert_eq!((h.gdim, h.num_cells()), (2, num));
            assert_eq!(h.blocks.len(), 1);
            assert_eq!(
                (h.blocks[0].cell, h.blocks[0].degree),
                (CellType::Triangle, 1)
            );
            let squares = num.div_ceil(2);
            let want_nodes = match num {
                0 => 0,
                1 => 3,
                _ => 2 * (squares + 1),
            };
            assert_eq!(h.num_nodes(), want_nodes, "{num} cells");
            for (b, c) in cells(&h) {
                let v: Vec<&[f64]> = row(&h, b, c).iter().map(|&i| x(&h, i)).collect();
                let area2 = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1])
                    - (v[2][0] - v[0][0]) * (v[1][1] - v[0][1]);
                assert_eq!(area2, 1.0, "{num} cells: cell {c}");
                let inside = |p: &&[f64]| p[0] <= squares as f64 && (0.0..=1.0).contains(&p[1]);
                assert!(v.iter().all(inside));
            }
            if num > 0 {
                let (dets, volume) = det_j(&h, 1);
                assert!(dets.iter().all(|&d| d > 0.0));
                assert!((volume - num as f64 / 2.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn the_large_triangle_strip_of_the_invariance_tests() {
        let num = 1_000_003;
        let h = triangle_strip(num);
        assert_eq!(h.num_cells(), num);
        assert_eq!(h.num_nodes(), 2 * (num.div_ceil(2) + 1));
    }

    #[test]
    fn the_hex_prism_pyramid_box_has_the_closed_form_cell_types() {
        for n in 1..=7 {
            let m = n / 3;
            let split = (m..n)
                .flat_map(|i| (0..n * n).map(move |jk| i + jk / n + jk % n))
                .filter(|s| s % 2 == 0)
                .count();
            let want: Vec<(CellType, usize)> = [
                (CellType::Hexahedron, (n - m) * n * n - split),
                (CellType::Prism, 2 * m * n * n),
                (CellType::Pyramid, 6 * split),
            ]
            .into_iter()
            .filter(|&(_, count)| count > 0)
            .collect();
            let h = hex_prism_pyramid_box(n, 2);
            let found: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
            assert_eq!(found, want, "n = {n}");
            // Every prism lies in the columns x ≤ m / n.
            if let Some(b) = h.blocks.iter().position(|b| b.cell == CellType::Prism) {
                let bound = m as f64 / n as f64;
                for &node in &h.blocks[b].cells {
                    assert!(
                        x(&h, node)[0] <= bound,
                        "n = {n}: a prism node beyond x = {bound}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_mixed_box_splits_even_subcells() {
        // n = 1: one even subcell, six pyramids, the one on x = 0 split in two.
        let h = mixed_box(1, 1);
        let blocks: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
        assert_eq!(blocks, [(CellType::Tetrahedron, 2), (CellType::Pyramid, 5)]);
        // Every tetrahedron has three vertices on x = 0 and its fourth at a subcell centre.
        let h = mixed_box(3, 2);
        let b = h
            .blocks
            .iter()
            .position(|b| b.cell == CellType::Tetrahedron);
        let b = b.expect("tetrahedra");
        assert_eq!(h.blocks[b].num_cells(), 2 * 5);
        for c in 0..h.blocks[b].num_cells() {
            let verts = &row(&h, b, c)[..4];
            let at_x0 = verts.iter().filter(|&&v| x(&h, v)[0] == 0.0).count();
            assert_eq!(at_x0, 3);
        }
    }

    #[test]
    fn build_checks_dimensions_first() {
        let square = square_box(Cell2::Triangle, 1, 1);
        assert!(matches!(
            square.elements::<D2, D3>(),
            Err(GridError::DimensionMismatch {
                expected: (2, 2),
                found: (2, 3)
            })
        ));
        let elements = square.elements::<D2, D2>().expect("elements");
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].cell, Cell2::Triangle);
        let mut cube = cube_box(Cell3::Tetrahedron, 1, 2);
        cube.gdim = 2;
        assert!(matches!(
            cube.elements::<D2, D2>(),
            Err(GridError::WrongCellDim {
                found: "Tetrahedron",
                expected_tdim: 2
            })
        ));
    }
}
