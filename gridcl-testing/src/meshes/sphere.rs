//! Curved sphere surfaces from icosahedral refinement, nodes projected onto the unit sphere.
//!
//! Each face of the icosahedron is split into 4^refinements flat triangles (refinement by edge
//! midpoints, without intermediate projection). A node of the flat mesh is an integer
//! combination of icosahedron vertices with weights summing to 2^refinements · p; its key is the
//! sorted list of its non-zero (vertex, weight) pairs, so nodes shared between faces are found
//! exactly. Every node is then projected onto the unit sphere.

use gridcl_dims::CellType;

use super::HostMesh;
use super::lattice::{Assembler, Key};

/// A node of the refined icosahedron: up to three (vertex, weight) pairs with positive weights,
/// sorted by vertex, padded with (−1, 0).
type SphereKey = [i64; 6];

/// The 12 icosahedron vertices (not normalised): (0, ±1, ±φ) and its cyclic permutations.
fn icosahedron_vertices() -> [[f64; 3]; 12] {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let mut out = [[0.0; 3]; 12];
    let mut n = 0;
    for axis in 0..3 {
        for s1 in [-1.0, 1.0] {
            for s2 in [-phi, phi] {
                let mut v = [0.0; 3];
                v[(axis + 1) % 3] = s1;
                v[(axis + 2) % 3] = s2;
                out[n] = v;
                n += 1;
            }
        }
    }
    out
}

/// The 20 icosahedron faces as vertex triples with outward orientation (det(v0, v1, v2) > 0),
/// in lexicographic order of their vertex sets.
fn icosahedron_faces(v: &[[f64; 3]; 12]) -> Vec<[usize; 3]> {
    let dist2 = |a: usize, b: usize| (0..3).map(|g| (v[a][g] - v[b][g]).powi(2)).sum::<f64>();
    let edge = |a: usize, b: usize| (dist2(a, b) - 4.0).abs() < 1e-9;
    let det = |[a, b, c]: [usize; 3]| {
        let (a, b, c) = (v[a], v[b], v[c]);
        a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
    };
    let mut faces = Vec::with_capacity(20);
    for a in 0..12 {
        for b in a + 1..12 {
            for c in b + 1..12 {
                if edge(a, b) && edge(b, c) && edge(a, c) {
                    let face = if det([a, b, c]) > 0.0 {
                        [a, b, c]
                    } else {
                        [a, c, b]
                    };
                    faces.push(face);
                }
            }
        }
    }
    assert_eq!(faces.len(), 20, "the icosahedron has 20 faces");
    faces
}

/// The key of the point with face-local lattice coordinates (a, b) on `face`, at `resolution`
/// lattice steps per face edge.
fn sphere_key(face: [usize; 3], resolution: i64, local: Key) -> SphereKey {
    let [a, b, _] = local;
    let weights = [resolution - a - b, a, b];
    assert!(weights.iter().all(|&w| w >= 0), "point outside the face");
    let mut pairs: Vec<(i64, i64)> = face
        .iter()
        .zip(weights)
        .filter(|&(_, w)| w > 0)
        .map(|(&v, w)| (v as i64, w))
        .collect();
    pairs.sort_unstable();
    let mut key = [-1, 0, -1, 0, -1, 0];
    for (slot, (v, w)) in pairs.into_iter().enumerate() {
        key[2 * slot] = v;
        key[2 * slot + 1] = w;
    }
    key
}

/// The unit sphere as 20 · 4^refinements triangles of degree `p`: the icosahedron refined
/// `refinements` times, with every node projected onto the sphere (gdim 3).
///
/// Every triangle is oriented with its normal J0 × J1 pointing away from the origin.
///
/// # Panics
///
/// If `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::sphere(1, 2);
/// assert_eq!((h.gdim, h.num_cells()), (3, 80));
/// // Vertices and edge midpoints of 80 triangles: V + E = 42 + 120.
/// assert_eq!(h.num_nodes(), 162);
/// assert!(h.nodes.chunks(3).all(|x| (x.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-14));
/// ```
pub fn sphere(refinements: u32, p: u32) -> HostMesh {
    assert!(p > 0, "a sphere needs degree p ≥ 1");
    let n = 1i64
        .checked_shl(refinements)
        .filter(|&n| n > 0)
        .expect("too many refinements");
    let step = i64::from(p);
    let resolution = n.checked_mul(step).expect("lattice resolution overflows");
    let v = icosahedron_vertices();
    let mut mesh = Assembler::<SphereKey>::with_keys(3);
    for face in icosahedron_faces(&v) {
        let canon = |k: Key| sphere_key(face, resolution, k);
        for j in 0..n {
            for i in 0..n - j {
                let at = |a: i64, b: i64| [a * step, b * step, 0];
                let up = [at(i, j), at(i + 1, j), at(i, j + 1)];
                mesh.cell_with(CellType::Triangle, p, &up, canon);
                if i + j + 1 < n {
                    let down = [at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
                    mesh.cell_with(CellType::Triangle, p, &down, canon);
                }
            }
        }
    }
    mesh.finish_with(|key| {
        let mut x = [0.0; 3];
        for pair in key.as_chunks::<2>().0.iter().filter(|pair| pair[0] >= 0) {
            for (xg, vg) in x.iter_mut().zip(v[pair[0] as usize]) {
                *xg += pair[1] as f64 * vg;
            }
        }
        project(x)
    })
}

/// `x / |x|`.
pub(super) fn project(x: [f64; 3]) -> [f64; 3] {
    let norm = x.iter().map(|c| c * c).sum::<f64>().sqrt();
    x.map(|c| c / norm)
}

/// Tests of the three curved generators (sphere, mixed surface, ball) on host data.
#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use gridcl_dims::CellType;
    use gridcl_refcell::{
        LagrangeElement, NodeFamily, gauss_rule, lagrange_entity_nodes, sub_entities,
    };

    use super::super::{HostMesh, ball, mixed_surface, sphere};

    const PI: f64 = std::f64::consts::PI;

    fn row(h: &HostMesh, b: usize, c: usize) -> &[u32] {
        let k = h.blocks[b].nodes_per_cell();
        &h.blocks[b].cells[c * k..(c + 1) * k]
    }

    fn x(h: &HostMesh, i: u32) -> &[f64] {
        let i = i as usize;
        &h.nodes[i * 3..(i + 1) * 3]
    }

    fn cells(h: &HostMesh) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..h.blocks.len()).flat_map(move |b| (0..h.blocks[b].num_cells()).map(move |c| (b, c)))
    }

    fn norm(v: &[f64]) -> f64 {
        v.iter().map(|c| c * c).sum::<f64>().sqrt()
    }

    /// The number of distinct sub-entities of dimension `dim` (sorted vertex nodes).
    fn num_entities(h: &HostMesh, dim: usize) -> usize {
        let mut set = HashSet::new();
        for (b, c) in cells(h) {
            let nodes = row(h, b, c);
            for ent in sub_entities(h.blocks[b].cell, dim) {
                let mut key: Vec<u32> = ent.iter().map(|&v| nodes[v]).collect();
                key.sort_unstable();
                set.insert(key);
            }
        }
        set.len()
    }

    /// Basic invariants: distinct coordinates, every node used, index-based ids.
    fn check_nodes(h: &HostMesh, what: &str) {
        let distinct: HashSet<Vec<u64>> = h
            .nodes
            .as_chunks::<3>()
            .0
            .iter()
            .map(|x| x.iter().map(|v| v.to_bits()).collect())
            .collect();
        assert_eq!(
            distinct.len(),
            h.num_nodes(),
            "{what}: duplicate coordinates"
        );
        let used: HashSet<u32> = h
            .blocks
            .iter()
            .flat_map(|b| b.cells.iter().copied())
            .collect();
        assert_eq!(used.len(), h.num_nodes(), "{what}: unused nodes");
        assert_eq!(h.node_ids, Some((0..h.num_nodes() as i64).collect()));
        let ids: Vec<i64> = h
            .blocks
            .iter()
            .flat_map(|b| b.cell_ids.clone().expect("cell ids"))
            .collect();
        assert_eq!(ids, (0..h.num_cells() as i64).collect::<Vec<_>>(), "{what}");
    }

    /// Every edge of a closed surface is shared by exactly two cells with the same nodes, and
    /// V − E + F = 2.
    fn check_closed_surface(h: &HostMesh, what: &str) {
        let mut edges: HashMap<Vec<u32>, Vec<Vec<u32>>> = HashMap::new();
        for (b, c) in cells(h) {
            let block = &h.blocks[b];
            let nodes = row(h, b, c);
            for (e, ent) in sub_entities(block.cell, 1).iter().enumerate() {
                let mut key: Vec<u32> = ent.iter().map(|&v| nodes[v]).collect();
                key.sort_unstable();
                let mut on_edge: Vec<u32> = lagrange_entity_nodes(block.cell, block.degree, 1, e)
                    .map(|k| nodes[k])
                    .chain(key.iter().copied())
                    .collect();
                on_edge.sort_unstable();
                edges.entry(key).or_default().push(on_edge);
            }
        }
        for (key, lists) in &edges {
            assert_eq!(lists.len(), 2, "{what}: edge {key:?}");
            assert_eq!(lists[0], lists[1], "{what}: edge {key:?} nodes differ");
        }
        let chi = num_entities(h, 0) as i64 - num_entities(h, 1) as i64 + h.num_cells() as i64;
        assert_eq!(chi, 2, "{what}: Euler characteristic");
    }

    /// The area Σ w |J0 × J1| of a surface mesh at a Gauss rule of degree `q`, and the minimum
    /// over all quadrature points of n̂ · x̂ (1 for an outward normal on the sphere).
    fn surface_area(h: &HostMesh, q: u32) -> (f64, f64) {
        let mut area = 0.0;
        let mut min_cos = f64::INFINITY;
        for (b, block) in h.blocks.iter().enumerate() {
            let rule = gauss_rule(block.cell, q);
            let element = LagrangeElement::new(block.cell, block.degree, NodeFamily::Equispaced)
                .expect("element");
            let tab = element.tabulate(&rule.points, 1).expect("tabulate");
            for c in 0..block.num_cells() {
                let nodes = row(h, b, c);
                for (iq, w) in rule.weights.iter().enumerate() {
                    let (mut p, mut j0, mut j1) = ([0.0; 3], [0.0; 3], [0.0; 3]);
                    for (i, &node) in nodes.iter().enumerate() {
                        let phi = tab.get(0, iq, i);
                        let (d0, d1) = (tab.get(1, iq, i), tab.get(2, iq, i));
                        for (g, xg) in x(h, node).iter().enumerate() {
                            p[g] += phi * xg;
                            j0[g] += d0 * xg;
                            j1[g] += d1 * xg;
                        }
                    }
                    let n = [
                        j0[1] * j1[2] - j0[2] * j1[1],
                        j0[2] * j1[0] - j0[0] * j1[2],
                        j0[0] * j1[1] - j0[1] * j1[0],
                    ];
                    let dot: f64 = n.iter().zip(&p).map(|(a, b)| a * b).sum();
                    min_cos = min_cos.min(dot / (norm(&n) * norm(&p)));
                    area += w * norm(&n);
                }
            }
        }
        (area, min_cos)
    }

    /// The volume Σ w det J of a volume mesh at a Gauss rule of degree `q`, and min det J.
    fn volume(h: &HostMesh, q: u32) -> (f64, f64) {
        let mut vol = 0.0;
        let mut min_det = f64::INFINITY;
        for (b, block) in h.blocks.iter().enumerate() {
            let rule = gauss_rule(block.cell, q);
            let element = LagrangeElement::new(block.cell, block.degree, NodeFamily::Equispaced)
                .expect("element");
            let tab = element.tabulate(&rule.points, 1).expect("tabulate");
            for c in 0..block.num_cells() {
                let nodes = row(h, b, c);
                for (iq, w) in rule.weights.iter().enumerate() {
                    let mut j = [[0.0f64; 3]; 3];
                    for (i, &node) in nodes.iter().enumerate() {
                        for (xg, jg) in x(h, node).iter().zip(&mut j) {
                            for (d, jgd) in jg.iter_mut().enumerate() {
                                *jgd += xg * tab.get(1 + d, iq, i);
                            }
                        }
                    }
                    let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
                        - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
                        + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);
                    min_det = min_det.min(det);
                    vol += w * det;
                }
            }
        }
        (vol, min_det)
    }

    /// The least-squares slope of y against x.
    fn slope(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let sxx: f64 = x.iter().map(|a| (a - mx) * (a - mx)).sum();
        sxy / sxx
    }

    #[test]
    fn sphere_counts_and_topology() {
        for r in 0..=2u32 {
            let f = 20 * 4usize.pow(r);
            let (v, e) = (f / 2 + 2, 3 * f / 2);
            for p in 1..=4u32 {
                let h = sphere(r, p);
                let what = format!("sphere({r}, {p})");
                assert_eq!(h.num_cells(), f, "{what}");
                assert_eq!(h.blocks.len(), 1);
                assert_eq!(
                    (h.blocks[0].cell, h.blocks[0].degree),
                    (CellType::Triangle, p)
                );
                let pm = p as usize - 1;
                let interior = pm * pm.saturating_sub(1) / 2;
                assert_eq!(h.num_nodes(), v + e * pm + f * interior, "{what}");
                check_nodes(&h, &what);
                check_closed_surface(&h, &what);
                for xi in h.nodes.as_chunks::<3>().0.iter() {
                    assert!(
                        (norm(xi) - 1.0).abs() <= 1e-14,
                        "{what}: |x| = {}",
                        norm(xi)
                    );
                }
            }
        }
    }

    #[test]
    fn mixed_surface_counts_and_topology() {
        for r in 0..=2u32 {
            let squares = 6 * 4usize.pow(r);
            let split = squares / 3;
            for p in 1..=3u32 {
                let h = mixed_surface(r, p);
                let what = format!("mixed_surface({r}, {p})");
                let blocks: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
                assert_eq!(
                    blocks,
                    [
                        (CellType::Triangle, 2 * split),
                        (CellType::Quadrilateral, squares - split)
                    ],
                    "{what}"
                );
                check_nodes(&h, &what);
                check_closed_surface(&h, &what);
                for xi in h.nodes.as_chunks::<3>().0.iter() {
                    assert!(
                        (norm(xi) - 1.0).abs() <= 1e-14,
                        "{what}: |x| = {}",
                        norm(xi)
                    );
                }
            }
        }
    }

    #[test]
    fn sphere_area_converges_at_rate_p_plus_one() {
        let exact = 4.0 * PI;
        for p in 1..=4u32 {
            let mut log_h = Vec::new();
            let mut log_err = Vec::new();
            for r in 0..=3u32 {
                let (area, min_cos) = surface_area(&sphere(r, p), 4 * p + 2);
                assert!(
                    min_cos > 0.5,
                    "sphere({r}, {p}): inward normal, cos {min_cos}"
                );
                log_h.push(-f64::from(r) * 2f64.ln());
                log_err.push(((area - exact).abs() / exact).ln());
            }
            let rate = slope(&log_h, &log_err);
            eprintln!("sphere area, p = {p}: rate {rate:.3}, log errors {log_err:.3?}");
            assert!(rate >= f64::from(p) + 0.8, "p = {p}: rate {rate}");
        }
    }

    #[test]
    fn mixed_surface_area_converges_monotonically() {
        let exact = 4.0 * PI;
        for p in 1..=3u32 {
            let mut last = f64::INFINITY;
            for r in 0..=3u32 {
                let (area, min_cos) = surface_area(&mixed_surface(r, p), 2 * p + 2);
                assert!(
                    min_cos > 0.5,
                    "mixed_surface({r}, {p}): normal, cos {min_cos}"
                );
                let err = (area - exact).abs() / exact;
                eprintln!("mixed surface area, p = {p}, r = {r}: error {err:.3e}");
                assert!(
                    err < last,
                    "mixed_surface({r}, {p}): error {err} after {last}"
                );
                last = err;
            }
        }
    }

    #[test]
    fn ball_nodes_cells_and_boundary() {
        for r in 0..=1u32 {
            for p in 1..=3u32 {
                let h = ball(r, p);
                let what = format!("ball({r}, {p})");
                let n = 2 * 2usize.pow(r);
                assert_eq!(h.num_cells(), n * n * n, "{what}");
                assert_eq!(
                    (h.blocks[0].cell, h.blocks[0].degree),
                    (CellType::Hexahedron, p)
                );
                let m = n * p as usize;
                assert_eq!(h.num_nodes(), (m + 1).pow(3), "{what}");
                check_nodes(&h, &what);
                let on_sphere = h
                    .nodes
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .filter(|&xi| (norm(xi) - 1.0).abs() <= 1e-14)
                    .count();
                assert_eq!(on_sphere, (m + 1).pow(3) - (m - 1).pow(3), "{what}");
                assert!(
                    h.nodes
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .all(|xi| norm(xi) <= 1.0 + 1e-14)
                );
            }
        }
    }

    #[test]
    fn ball_volume_converges_monotonically_with_positive_det_j() {
        let exact = 4.0 * PI / 3.0;
        for p in 1..=4u32 {
            let mut last = f64::INFINITY;
            for r in 0..=2u32 {
                let (vol, min_det) = volume(&ball(r, p), 3 * p + 1);
                assert!(min_det > 0.0, "ball({r}, {p}): det J = {min_det}");
                let err = (vol - exact).abs() / exact;
                eprintln!(
                    "ball volume, p = {p}, r = {r}: error {err:.3e}, min det J {min_det:.3e}"
                );
                assert!(err < last, "ball({r}, {p}): error {err} after {last}");
                last = err;
            }
        }
    }

    /// The curved meshes build in `F`.
    fn curved_meshes_build_in<F: gridcl::Real>(client: &cubecl::prelude::Client) {
        use gridcl::prelude::*;
        let s = sphere(1, 2).build::<F, D2, D3>(client).expect("sphere");
        assert_eq!((s.tdim(), s.gdim()), (2, 3));
        let m = mixed_surface(0, 2)
            .build::<F, D2, D3>(client)
            .expect("mixed surface");
        let cells: Vec<Cell2> = m.cell_blocks().iter().map(|b| b.cell).collect();
        assert_eq!(cells.len(), 2);
        assert!(cells.contains(&Cell2::Triangle) && cells.contains(&Cell2::Quadrilateral));
        let b = ball(0, 2).build::<F, D3, D3>(client).expect("ball");
        assert_eq!((b.tdim(), b.gdim()), (3, 3));
    }

    #[test]
    fn curved_meshes_build() {
        let client = crate::test_client();
        curved_meshes_build_in::<f32>(&client);
        if crate::supports::<f64>(&client) {
            curved_meshes_build_in::<f64>(&client);
        }
    }
}
