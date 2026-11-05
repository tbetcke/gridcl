//! The manifold meshes of M9 (docs/plans/M9.md, T0a): a Möbius strip, a book of triangles on
//! one edge, a closed curve, a Y junction of intervals, an open cap of the sphere, and
//! [`HostMesh::with_flipped_cells`].

use std::f64::consts::PI;

use gridcl_dims::CellType;
use gridcl_refcell::NodeFamily;

use super::lattice::SplitMix64;
use super::{HostBlock, HostMesh, sphere};

/// A mesh of one block of degree-`degree` cells with index-based node and cell ids.
fn single_block(
    gdim: usize,
    nodes: Vec<f64>,
    cell: CellType,
    degree: u32,
    cells: Vec<u32>,
) -> HostMesh {
    let mut h = HostMesh {
        gdim,
        nodes,
        node_ids: None,
        blocks: vec![HostBlock {
            cell,
            degree,
            cells,
            cell_ids: None,
        }],
    };
    h.node_ids = Some((0..h.num_nodes() as i64).collect());
    h.blocks[0].cell_ids = Some((0..h.blocks[0].num_cells() as i64).collect());
    h
}

/// The equispaced reference points of an interval of degree `p`, in A.5 order.
fn interval_points(p: u32) -> Vec<f64> {
    gridcl_refcell::lagrange_points(CellType::Interval, p, NodeFamily::Equispaced)
        .expect("interval Lagrange points")
}

/// A Möbius strip of 2n P1 triangles (gdim 3): n squares along the centre circle of radius 1,
/// of width 1, with a half twist.
///
/// Square i spans the parameter range u ∈ [2πi/n, 2π(i+1)/n] of
/// x(u, v) = ((1 + v cos(u/2)/2) cos u, (1 + v cos(u/2)/2) sin u, v sin(u/2)/2), v = ±1, and
/// holds the triangles (b_i, b_{i+1}, t_{i+1}) and (b_i, t_{i+1}, t_i), where t_i (node i) is
/// at v = 1 and b_i (node n + i) at v = −1. The twist identifies t_n with b_0 and b_n with
/// t_0, so V = 2n, E = 4n, F = 2n (Euler characteristic 0), and the boundary is one closed
/// curve of 2n edges. Cell orientations agree within the parameter domain and disagree across
/// the seam u = 0, as they must.
///
/// # Panics
///
/// If `n` < 3.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::mobius_strip(8);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (3, 16, 16));
/// ```
pub fn mobius_strip(n: usize) -> HostMesh {
    assert!(n >= 3, "a Möbius strip needs n ≥ 3 squares");
    let node = |i: usize, top: bool| {
        let (i, top) = if i == n { (0, !top) } else { (i, top) };
        u32::try_from(if top { i } else { n + i }).expect("node index fits u32")
    };
    let mut nodes = Vec::with_capacity(6 * n);
    for v in [1.0, -1.0] {
        for i in 0..n {
            let u = 2.0 * PI * i as f64 / n as f64;
            let r = 1.0 + 0.5 * v * (0.5 * u).cos();
            nodes.extend([r * u.cos(), r * u.sin(), 0.5 * v * (0.5 * u).sin()]);
        }
    }
    let mut cells = Vec::with_capacity(6 * n);
    for i in 0..n {
        let (b0, b1, t0, t1) = (
            node(i, false),
            node(i + 1, false),
            node(i, true),
            node(i + 1, true),
        );
        cells.extend([b0, b1, t1, b0, t1, t0]);
    }
    single_block(3, nodes, CellType::Triangle, 1, cells)
}

/// A book of `pages` P1 triangles on one edge (gdim 3): every triangle k is
/// (node 0, node 1, node 2 + k), with the spine from node 0 at (0, 0, 0) to node 1 at
/// (0, 0, 1) and the tip of page k at (cos α_k, sin α_k, 1/2), α_k = 2πk/pages.
///
/// The spine has `pages` cells: three or more make the mesh non-manifold. V = pages + 2,
/// E = 2 · pages + 1, F = pages (Euler characteristic 1).
///
/// # Panics
///
/// If `pages` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::book(3);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (3, 3, 5));
/// assert_eq!(h.blocks[0].cells, [0, 1, 2, 0, 1, 3, 0, 1, 4]);
/// ```
pub fn book(pages: usize) -> HostMesh {
    assert!(pages > 0, "a book needs a page");
    let mut nodes = vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut cells = Vec::with_capacity(3 * pages);
    for k in 0..pages {
        let a = 2.0 * PI * k as f64 / pages as f64;
        nodes.extend([a.cos(), a.sin(), 0.5]);
        cells.extend([0, 1, u32::try_from(2 + k).expect("node index fits u32")]);
    }
    single_block(3, nodes, CellType::Triangle, 1, cells)
}

/// The unit circle as `n` intervals of degree `p` (gdim 2), oriented counter-clockwise, so the
/// normals (J0_y, −J0_x) of "Geometric factors" point outward.
///
/// The circle has n · p nodes: node k is at angle 2πk/(n · p), so the nodes of cell i sit at
/// the angles 2π(i + r)/n of its reference points r, and the cells are curved for p ≥ 2.
///
/// # Panics
///
/// If `n` < 3 or `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::circle(4, 2);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (2, 4, 8));
/// // Cell 3 runs from angle 3π/2 back to 0, through 7π/4.
/// assert_eq!(&h.blocks[0].cells[9..12], [6, 0, 7]);
/// ```
pub fn circle(n: usize, p: u32) -> HostMesh {
    assert!(n >= 3, "a circle needs n ≥ 3 intervals");
    assert!(p > 0, "a circle needs degree p ≥ 1");
    let pu = p as usize;
    let total = n.checked_mul(pu).expect("node count overflows");
    let nodes = (0..total)
        .flat_map(|k| {
            let a = 2.0 * PI * k as f64 / total as f64;
            [a.cos(), a.sin()]
        })
        .collect();
    let refs = interval_points(p);
    let mut cells = Vec::with_capacity(n * (pu + 1));
    for i in 0..n {
        for &r in &refs {
            let k = (i * pu + (r * f64::from(p)).round() as usize) % total;
            cells.push(u32::try_from(k).expect("node index fits u32"));
        }
    }
    single_block(2, nodes, CellType::Interval, p, cells)
}

/// Three straight intervals of degree `p` from a common point (gdim 2): node 0 at the origin,
/// and arm a from it to the unit vector at angle π/2 + 2πa/3, its tip and interior nodes
/// following in A.5 order. The origin has three cells: the mesh is non-manifold.
///
/// # Panics
///
/// If `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::y_junction(2);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (2, 3, 7));
/// assert_eq!(h.blocks[0].cells, [0, 1, 2, 0, 3, 4, 0, 5, 6]);
/// ```
pub fn y_junction(p: u32) -> HostMesh {
    assert!(p > 0, "a junction needs degree p ≥ 1");
    let refs = interval_points(p);
    let mut nodes = vec![0.0, 0.0];
    let mut cells = Vec::with_capacity(3 * refs.len());
    for a in 0..3 {
        let angle = PI / 2.0 + 2.0 * PI * f64::from(a) / 3.0;
        let tip = [angle.cos(), angle.sin()];
        cells.push(0);
        for &r in &refs[1..] {
            cells.push(u32::try_from(nodes.len() / 2).expect("node index fits u32"));
            nodes.extend(tip.map(|t| r * t));
        }
    }
    single_block(2, nodes, CellType::Interval, p, cells)
}

/// An open cap of the unit sphere: the cells of [`sphere`]`(refinements, p)` whose vertex
/// centroid c has c · d > 0, for the unit vector d along (0.1, 0.2, 1) (a direction in no
/// symmetry plane of the icosahedron, so no centroid lies on the cutting plane).
///
/// The cap is a disk (Euler characteristic 1) with one boundary curve, close to a hemisphere.
/// Nodes and cells keep their origin ids from the sphere, and their orientation: J0 × J1
/// points away from the origin.
///
/// # Panics
///
/// If `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::hemisphere(1, 1);
/// assert_eq!(h.gdim, 3);
/// assert!(h.num_cells() > 30 && h.num_cells() < 50);
/// ```
pub fn hemisphere(refinements: u32, p: u32) -> HostMesh {
    let s = sphere(refinements, p);
    let norm = (0.01f64 + 0.04 + 1.0).sqrt();
    let d = [0.1 / norm, 0.2 / norm, 1.0 / norm];
    let k = s.blocks[0].nodes_per_cell();
    let side = |c: usize| {
        let row = &s.blocks[0].cells[c * k..c * k + 3];
        let centroid: Vec<f64> = (0..3)
            .map(|g| {
                row.iter()
                    .map(|&n| s.nodes[3 * n as usize + g])
                    .sum::<f64>()
                    / 3.0
            })
            .collect();
        centroid.iter().zip(d).map(|(c, d)| c * d).sum::<f64>()
    };
    s.retain_cells(|_, c| side(c) > 0.0)
}

/// The permutation of the reference nodes of `cell` at `degree` (equispaced) that reverses the
/// cell's orientation: `perm[i] = j` where reference point j is σ(reference point i), with
/// σ(r) = 1 − r on intervals and σ(r0, r1) = (r1, r0) on triangles and quadrilaterals (vertices
/// 1 and 2 swap).
fn reversal(cell: CellType, degree: u32) -> Vec<usize> {
    let points = gridcl_refcell::lagrange_points(cell, degree, NodeFamily::Equispaced)
        .expect("equispaced Lagrange points");
    let td = gridcl_refcell::tdim(cell);
    let sigma = |r: &[f64]| -> Vec<f64> {
        match cell {
            CellType::Interval => vec![1.0 - r[0]],
            CellType::Triangle | CellType::Quadrilateral => vec![r[1], r[0]],
            _ => panic!(
                "with_flipped_cells reverses intervals, triangles and quadrilaterals, not {cell:?}"
            ),
        }
    };
    let pts: Vec<&[f64]> = points.chunks_exact(td).collect();
    pts.iter()
        .map(|r| {
            let s = sigma(r);
            pts.iter()
                .position(|q| q.iter().zip(&s).all(|(a, b)| (a - b).abs() < 1e-12))
                .unwrap_or_else(|| panic!("{cell:?} p={degree}: no image of {r:?}"))
        })
        .collect()
}

impl HostMesh {
    /// The mesh with a random subset of cells reversed: each cell is flipped with probability
    /// `fraction` (a SplitMix64 stream seeded by `seed`, one draw per cell in block order).
    ///
    /// A flipped cell keeps its nodes, origin id and image, but its node list is permuted by the
    /// orientation-reversing reference map: r ↦ 1 − r on intervals, (r0, r1) ↦ (r1, r0) on
    /// triangles and quadrilaterals, for every node of every degree. Its Jacobian columns change
    /// sign (intervals) or swap (triangles, quadrilaterals), so its normal reverses.
    ///
    /// # Panics
    ///
    /// If a block holds cells other than intervals, triangles and quadrilaterals.
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::circle(3, 2);
    /// let all = h.with_flipped_cells(1, 1.0);
    /// // Cell 0 runs from node 0 to node 2 through node 1; flipped, from 2 to 0.
    /// assert_eq!(&h.blocks[0].cells[..3], [0, 2, 1]);
    /// assert_eq!(&all.blocks[0].cells[..3], [2, 0, 1]);
    /// assert_eq!(h.with_flipped_cells(1, 0.0), h);
    /// ```
    pub fn with_flipped_cells(&self, seed: u64, fraction: f64) -> HostMesh {
        let mut rng = SplitMix64::new(seed);
        let mut out = self.clone();
        for block in &mut out.blocks {
            let perm = reversal(block.cell, block.degree);
            for row in block.cells.chunks_exact_mut(perm.len()) {
                let draw = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                if draw < fraction {
                    let old = row.to_vec();
                    for (r, &j) in row.iter_mut().zip(&perm) {
                        *r = old[j];
                    }
                }
            }
        }
        out
    }
}

/// Host tests: counts, Euler characteristics, orientation of flipped cells; on the device, that
/// flipping keeps the measure.
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use gridcl::{D1, D2, D3};
    use gridcl_refcell::{LagrangeElement, NodeFamily, gauss_rule, sub_entities};

    use super::super::{HostMesh, mixed_surface, sphere};
    use super::{book, circle, hemisphere, mobius_strip, y_junction};

    /// The cells of every facet (sorted vertex nodes) over all blocks.
    fn facet_cells(h: &HostMesh) -> HashMap<Vec<u32>, usize> {
        let mut out = HashMap::new();
        for block in &h.blocks {
            let td = gridcl_refcell::tdim(block.cell);
            let k = block.nodes_per_cell();
            for row in block.cells.chunks_exact(k) {
                for ent in sub_entities(block.cell, td - 1) {
                    let mut key: Vec<u32> = ent.iter().map(|&v| row[v]).collect();
                    key.sort_unstable();
                    *out.entry(key).or_default() += 1;
                }
            }
        }
        out
    }

    /// The number of distinct sub-entities of dimension `dim` (sorted vertex nodes).
    fn num_entities(h: &HostMesh, dim: usize) -> usize {
        let mut set = std::collections::HashSet::new();
        for block in &h.blocks {
            let k = block.nodes_per_cell();
            for row in block.cells.chunks_exact(k) {
                for ent in sub_entities(block.cell, dim) {
                    let mut key: Vec<u32> = ent.iter().map(|&v| row[v]).collect();
                    key.sort_unstable();
                    set.insert(key);
                }
            }
        }
        set.len()
    }

    /// V − E + F of a surface.
    fn euler(h: &HostMesh) -> i64 {
        num_entities(h, 0) as i64 - num_entities(h, 1) as i64 + h.num_cells() as i64
    }

    #[test]
    fn mobius_strip_counts_and_euler_characteristic() {
        for n in [3, 4, 9, 32] {
            let h = mobius_strip(n);
            assert_eq!((h.num_cells(), h.num_nodes()), (2 * n, 2 * n));
            assert_eq!(num_entities(&h, 1), 4 * n, "n = {n}: edges");
            assert_eq!(euler(&h), 0, "n = {n}");
            let cells = facet_cells(&h);
            assert_eq!(
                cells.values().filter(|&&c| c == 1).count(),
                2 * n,
                "n = {n}: boundary"
            );
            assert!(cells.values().all(|&c| c <= 2), "n = {n}: manifold");
        }
    }

    #[test]
    fn book_counts_and_euler_characteristic() {
        for pages in 1..=5 {
            let h = book(pages);
            assert_eq!((h.num_cells(), h.num_nodes()), (pages, pages + 2));
            assert_eq!(num_entities(&h, 1), 2 * pages + 1);
            assert_eq!(euler(&h), 1, "pages = {pages}");
            assert_eq!(facet_cells(&h)[&vec![0, 1]], pages, "the spine");
        }
    }

    #[test]
    fn circle_and_junction_counts() {
        for n in [3, 5, 16] {
            for p in 1..=4 {
                let h = circle(n, p);
                assert_eq!((h.num_cells(), h.num_nodes()), (n, n * p as usize));
                assert!(facet_cells(&h).values().all(|&c| c == 2), "closed");
                assert!(
                    h.nodes
                        .chunks(2)
                        .all(|x| (x[0].hypot(x[1]) - 1.0).abs() < 1e-15),
                    "on the circle"
                );
            }
        }
        for p in 1..=4 {
            let h = y_junction(p);
            assert_eq!((h.num_cells(), h.num_nodes()), (3, 1 + 3 * p as usize));
            let cells = facet_cells(&h);
            assert_eq!(cells[&vec![0]], 3, "the junction");
            assert_eq!(cells.values().filter(|&&c| c == 1).count(), 3, "the tips");
        }
    }

    #[test]
    fn hemisphere_is_a_disk() {
        for r in 0..=3 {
            for p in 1..=2 {
                let h = hemisphere(r, p);
                let total = sphere(r, p).num_cells();
                assert!(
                    h.num_cells() > total / 3 && h.num_cells() < 2 * total / 3,
                    "r = {r}: {} of {total} cells",
                    h.num_cells()
                );
                assert_eq!(euler(&h), 1, "r = {r}, p = {p}");
                let cells = facet_cells(&h);
                assert!(cells.values().all(|&c| c <= 2), "manifold");
                // One boundary curve: every boundary vertex is on exactly two boundary edges.
                let mut degree: HashMap<u32, usize> = HashMap::new();
                for key in cells.iter().filter(|&(_, &c)| c == 1).map(|(k, _)| k) {
                    for &v in key {
                        *degree.entry(v).or_default() += 1;
                    }
                }
                assert!(!degree.is_empty() && degree.values().all(|&d| d == 2));
            }
        }
    }

    /// Per cell, the signs of n · x at every Gauss point of degree 2p, where n is J0 × J1
    /// (surfaces in 3D) or (J0_y, −J0_x) (curves in 2D).
    fn normal_signs(h: &HostMesh) -> Vec<Vec<f64>> {
        let gd = h.gdim;
        let mut out = Vec::new();
        for block in &h.blocks {
            let rule = gauss_rule(block.cell, 2 * block.degree);
            let element =
                LagrangeElement::new(block.cell, block.degree, NodeFamily::Equispaced).unwrap();
            let tab = element.tabulate(&rule.points, 1).unwrap();
            let td = gridcl_refcell::tdim(block.cell);
            for row in block.cells.chunks_exact(block.nodes_per_cell()) {
                let mut signs = Vec::new();
                for iq in 0..rule.weights.len() {
                    let (mut x, mut j) = ([0.0; 3], [[0.0; 3]; 2]);
                    for (i, &node) in row.iter().enumerate() {
                        let xn = &h.nodes[gd * node as usize..gd * (node as usize + 1)];
                        for g in 0..gd {
                            x[g] += tab.get(0, iq, i) * xn[g];
                            for (d, jd) in j.iter_mut().enumerate().take(td) {
                                jd[g] += tab.get(1 + d, iq, i) * xn[g];
                            }
                        }
                    }
                    let n = if gd == 2 {
                        [j[0][1], -j[0][0], 0.0]
                    } else {
                        let [a, b] = j;
                        [
                            a[1] * b[2] - a[2] * b[1],
                            a[2] * b[0] - a[0] * b[2],
                            a[0] * b[1] - a[1] * b[0],
                        ]
                    };
                    signs.push(n.iter().zip(x).map(|(n, x)| n * x).sum::<f64>().signum());
                }
                out.push(signs);
            }
        }
        out
    }

    #[test]
    fn flipped_cells_point_inward() {
        let meshes = [
            ("sphere", sphere(1, 1)),
            ("sphere p=3", sphere(1, 3)),
            ("mixed surface p=2", mixed_surface(1, 2)),
            ("circle p=3", circle(24, 3)),
        ];
        for (name, h) in meshes {
            let flipped = h.with_flipped_cells(5, 0.4);
            let mut count = 0;
            for (b, (blk, fblk)) in h.blocks.iter().zip(&flipped.blocks).enumerate() {
                let k = blk.nodes_per_cell();
                for (c, (row, frow)) in blk.cells.chunks(k).zip(fblk.cells.chunks(k)).enumerate() {
                    let mut a = row.to_vec();
                    let mut f = frow.to_vec();
                    a.sort_unstable();
                    f.sort_unstable();
                    assert_eq!(a, f, "{name}: block {b} cell {c} keeps its nodes");
                    count += usize::from(row != frow);
                }
            }
            let total = h.num_cells();
            assert!(
                count > total / 5 && count < 3 * total / 5,
                "{name}: {count} of {total} flipped"
            );
            let before = normal_signs(&h);
            let after = normal_signs(&flipped);
            let mut c = 0;
            for blk in h.blocks.iter().zip(&flipped.blocks) {
                let k = blk.0.nodes_per_cell();
                for (row, frow) in blk.0.cells.chunks(k).zip(blk.1.cells.chunks(k)) {
                    assert!(
                        before[c].iter().all(|&s| s > 0.0),
                        "{name}: cell {c} outward"
                    );
                    let want = if row == frow { 1.0 } else { -1.0 };
                    assert!(
                        after[c].iter().all(|&s| s == want),
                        "{name}: cell {c} sign {want}"
                    );
                    c += 1;
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "not Tetrahedron")]
    fn flipping_rejects_volume_cells() {
        let _ = super::super::cube_box(gridcl_dims::Cell3::Tetrahedron, 1, 1)
            .with_flipped_cells(0, 0.5);
    }

    /// Flipping permutes the nodes of a cell, not its image: the measure changes only by the
    /// quadrature error, since collapsed triangle rules are not symmetric under (r0, r1) ↦
    /// (r1, r0). A Gauss degree of 4p + 6 makes that error negligible in f64.
    #[test]
    fn flipping_keeps_the_measure() {
        let client = crate::test_client();
        let double = crate::supports::<f64>(&client);
        let measure = |h: &HostMesh, q| -> f64 {
            match (h.gdim, double) {
                (2, true) => h.build::<f64, D1, D2>(&client).unwrap().measure(q).unwrap(),
                (2, false) => h.build::<f32, D1, D2>(&client).unwrap().measure(q).unwrap(),
                (_, true) => h.build::<f64, D2, D3>(&client).unwrap().measure(q).unwrap(),
                (_, false) => h.build::<f32, D2, D3>(&client).unwrap().measure(q).unwrap(),
            }
        };
        let tol = if double { 1e-10 } else { crate::TOL_F32 };
        for (h, p) in [
            (sphere(1, 2), 2),
            (mixed_surface(1, 3), 3),
            (circle(5, 3), 3),
        ] {
            let q = 4 * p + 6;
            let a = measure(&h, q);
            let b = measure(&h.with_flipped_cells(9, 0.5), q);
            assert!((a - b).abs() <= tol * a, "{a} vs {b}");
        }
    }
}
