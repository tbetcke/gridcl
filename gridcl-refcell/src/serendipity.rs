//! Gmsh's serendipity elements (types 16–19) and their upgrade to complete P2 (ADR-0022).
//!
//! Gmsh's 8-node quadrilateral, 20-node hexahedron and 15-node prism interpolate in the
//! serendipity spaces of degree 2 (Gmsh `gmshGenerateMonomialsQuadSerendipity`,
//! `…HexaSerendipity`, `…PrismSerendipity`):
//!
//! - quadrilateral: P2 + {x²y, xy²};
//! - hexahedron: the monomials of superlinear degree ≤ 2 (P2, xyz, x²y and its five
//!   permutations, x²yz, xy²z, xyz²);
//! - prism: P2(x, y) ⊗ {1, z} + P1(x, y) ⊗ {z²}.
//!
//! These spaces are invariant under axis-wise affine maps, so they are the same on Gmsh's and
//! on gridcl's reference cells. The 13-node pyramid's basis is rational, but on the base it
//! reduces to the 8-node quadrilateral's, and the base centre is the only node it lacks; its
//! weights come from that trace.
//!
//! The serendipity nodes are the first nodes of Gmsh's complete-P2 element (vertices, then
//! edge midpoints), so the missing nodes are the face centres and, for the hexahedron, the
//! volume centre, at the positions of Gmsh's P2 element.

use gridcl_dims::CellType;

use crate::lu::Lu;
use crate::{RefCellError, gmsh_to_internal, lagrange_points, num_lagrange_nodes, tdim};

/// The weights that give the missing nodes of a complete-P2 element from the nodes of Gmsh's
/// serendipity element of the same cell: 8 → 9 (quadrilateral), 20 → 27 (hexahedron), 15 → 18
/// (prism), 13 → 14 (pyramid).
///
/// Both node lists are in Gmsh order. The serendipity nodes are the first
/// [`num_file_nodes`](Self::num_file_nodes) nodes of Gmsh's complete-P2 element; missing node
/// m (Gmsh P2 index `num_file_nodes + m`) is Σⱼ `weights[m * num_file_nodes + j]` · node j.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::serendipity_upgrade;
///
/// // The centre of an 8-node quadrilateral: −1/4 of each vertex, 1/2 of each edge midpoint.
/// let up = serendipity_upgrade(CellType::Quadrilateral).unwrap();
/// assert_eq!((up.num_file_nodes, up.num_nodes), (8, 9));
/// for (j, w) in up.weights.iter().enumerate() {
///     let want = if j < 4 { -0.25 } else { 0.5 };
///     assert!((w - want).abs() < 1e-14);
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct SerendipityUpgrade {
    /// The node count of the serendipity element: 8, 20, 15 or 13.
    pub num_file_nodes: usize,
    /// The node count of the complete-P2 element: 9, 27, 18 or 14.
    pub num_nodes: usize,
    /// `[missing node][file node]`, row-major.
    pub weights: Vec<f64>,
}

/// Exponents of the monomials of a serendipity space; unused coordinates are 0.
type Exponents = [u32; 3];

const QUADRILATERAL: [Exponents; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [2, 0, 0],
    [0, 2, 0],
    [2, 1, 0],
    [1, 2, 0],
];

const HEXAHEDRON: [Exponents; 20] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [2, 1, 0],
    [2, 0, 1],
    [1, 2, 0],
    [0, 2, 1],
    [1, 0, 2],
    [0, 1, 2],
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
];

const PRISM: [Exponents; 15] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [2, 0, 0],
    [1, 1, 0],
    [0, 2, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [2, 0, 1],
    [1, 1, 1],
    [0, 2, 1],
    [0, 0, 2],
    [1, 0, 2],
    [0, 1, 2],
];

/// The monomials `exps` at the point `x` (`x.len()` coordinates).
fn monomials(exps: &[Exponents], x: &[f64]) -> Vec<f64> {
    exps.iter()
        .map(|e| x.iter().zip(e).map(|(c, &k)| c.powi(k as i32)).product())
        .collect()
}

/// A nodal basis of the span of `exps` at `nodes` ([node][dim]): φᵢ(nodeⱼ) = δᵢⱼ.
pub(crate) struct NodalBasis {
    exps: Vec<Exponents>,
    dim: usize,
    /// The factors of Vᵀ, Vᵢⱼ = mⱼ(nodeᵢ): φ(x) solves Vᵀ φ = m(x).
    vt: Lu,
}

impl NodalBasis {
    pub(crate) fn new(exps: &[Exponents], nodes: &[f64], dim: usize) -> Result<Self, RefCellError> {
        let n = exps.len();
        assert_eq!(nodes.len(), n * dim, "one node per monomial");
        let mut vt = vec![0.0; n * n];
        for (i, x) in nodes.chunks_exact(dim).enumerate() {
            for (j, m) in monomials(exps, x).into_iter().enumerate() {
                vt[j * n + i] = m;
            }
        }
        Ok(NodalBasis {
            exps: exps.to_vec(),
            dim,
            vt: Lu::factor(vt, n)?,
        })
    }

    /// The basis functions at `x`.
    pub(crate) fn eval(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        let mut phi = monomials(&self.exps, x);
        self.vt.solve(&mut phi);
        phi
    }
}

/// The node count of Gmsh's serendipity element on `cell`.
fn num_serendipity_nodes(cell: CellType) -> Option<usize> {
    match cell {
        CellType::Quadrilateral => Some(8),
        CellType::Hexahedron => Some(20),
        CellType::Prism => Some(15),
        CellType::Pyramid => Some(13),
        _ => None,
    }
}

/// The points of Gmsh's complete-P2 element on `cell`, in Gmsh order, on gridcl's reference
/// cell: [node][tdim].
fn gmsh_p2_points(cell: CellType) -> Vec<f64> {
    let d = tdim(cell);
    let internal =
        lagrange_points(cell, 2, crate::NodeFamily::Equispaced).expect("P2 exists on every cell");
    gmsh_to_internal(cell, 2)
        .expect("Gmsh has P2 on every cell")
        .iter()
        .flat_map(|&i| internal[i * d..(i + 1) * d].to_vec())
        .collect()
}

/// The private basis of Gmsh's serendipity element on `cell` (quadrilateral, hexahedron and
/// prism; the pyramid's is rational and not needed), at the serendipity nodes in Gmsh order.
pub(crate) fn serendipity_basis(cell: CellType) -> Option<NodalBasis> {
    let exps: &[Exponents] = match cell {
        CellType::Quadrilateral => &QUADRILATERAL,
        CellType::Hexahedron => &HEXAHEDRON,
        CellType::Prism => &PRISM,
        _ => return None,
    };
    let d = tdim(cell);
    let points = gmsh_p2_points(cell);
    let nodes = &points[..exps.len() * d];
    let exps: Vec<Exponents> = exps
        .iter()
        .map(|e| {
            let mut e = *e;
            e[d..].fill(0);
            e
        })
        .collect();
    Some(NodalBasis::new(&exps, nodes, d).expect("the serendipity nodes are unisolvent"))
}

/// The upgrade of Gmsh's serendipity element on `cell` (host, f64): its serendipity map
/// evaluated at the missing complete-P2 nodes.
///
/// # Errors
///
/// `NotInFormat` (format `"Gmsh serendipity"`, degree 2) for cells without a Gmsh serendipity
/// type: all but quadrilaterals, hexahedra, prisms and pyramids.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::serendipity_upgrade;
///
/// // A 20-node hexahedron lacks six face centres and the volume centre.
/// let up = serendipity_upgrade(CellType::Hexahedron).unwrap();
/// assert_eq!((up.num_file_nodes, up.num_nodes, up.weights.len()), (20, 27, 7 * 20));
/// // Every missing node is an affine combination of the file's nodes.
/// for row in up.weights.chunks(20) {
///     assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-14);
/// }
/// assert!(serendipity_upgrade(CellType::Triangle).is_err());
/// ```
pub fn serendipity_upgrade(cell: CellType) -> Result<SerendipityUpgrade, RefCellError> {
    let num_file_nodes = num_serendipity_nodes(cell).ok_or(RefCellError::NotInFormat {
        format: "Gmsh serendipity",
        cell,
        degree: 2,
    })?;
    let num_nodes = num_lagrange_nodes(cell, 2);
    let d = tdim(cell);
    let points = gmsh_p2_points(cell);
    let missing = points[num_file_nodes * d..].chunks_exact(d);
    let mut weights = Vec::with_capacity((num_nodes - num_file_nodes) * num_file_nodes);
    match serendipity_basis(cell) {
        Some(basis) => {
            for x in missing {
                weights.extend(basis.eval(x));
            }
        }
        None => {
            // The pyramid: its basis on the base (z = 0) is the 8-node quadrilateral's.
            let base: Vec<usize> = (0..num_file_nodes)
                .filter(|&j| points[j * d + 2] == 0.0)
                .collect();
            assert_eq!(
                base.len(),
                8,
                "four vertices and four edge midpoints on the base"
            );
            let nodes: Vec<f64> = base
                .iter()
                .flat_map(|&j| points[j * d..j * d + 2].to_vec())
                .collect();
            let quad =
                NodalBasis::new(&QUADRILATERAL, &nodes, 2).expect("the base nodes are unisolvent");
            for x in missing {
                assert_eq!(x[2], 0.0, "the pyramid's missing node is the base centre");
                let mut row = vec![0.0; num_file_nodes];
                for (&j, w) in base.iter().zip(quad.eval(&x[..2])) {
                    row[j] = w;
                }
                weights.extend(row);
            }
        }
    }
    Ok(SerendipityUpgrade {
        num_file_nodes,
        num_nodes,
        weights,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const CELLS: [CellType; 3] = [
        CellType::Quadrilateral,
        CellType::Hexahedron,
        CellType::Prism,
    ];

    /// φᵢ(nodeⱼ) = δᵢⱼ and Σ φᵢ = 1 at random points.
    #[test]
    fn serendipity_bases_are_nodal_and_a_partition_of_unity() {
        for cell in CELLS {
            let basis = serendipity_basis(cell).unwrap();
            let d = tdim(cell);
            let n = num_serendipity_nodes(cell).unwrap();
            let points = gmsh_p2_points(cell);
            for (j, x) in points[..n * d].chunks_exact(d).enumerate() {
                for (i, phi) in basis.eval(x).iter().enumerate() {
                    let want = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (phi - want).abs() < 1e-13,
                        "{cell:?}: φ{i}(node {j}) = {phi}"
                    );
                }
            }
            for x in [[0.1, 0.2, 0.3], [0.7, 0.05, 0.9], [0.3, 0.3, 0.3]] {
                let sum: f64 = basis.eval(&x[..d]).iter().sum();
                assert!((sum - 1.0).abs() < 1e-13, "{cell:?}: Σφ = {sum}");
            }
        }
    }

    /// The quadrilateral centre: −1/4 per vertex, 1/2 per edge; the hexahedron centre: −1/4 per
    /// vertex, 1/4 per edge; face centres of hexahedra, prisms and the pyramid base: −1/4 per
    /// face vertex, 1/2 per face edge, 0 elsewhere.
    #[test]
    fn serendipity_centre_weights() {
        let quad = serendipity_upgrade(CellType::Quadrilateral).unwrap();
        for (j, w) in quad.weights.iter().enumerate() {
            let want = if j < 4 { -0.25 } else { 0.5 };
            assert!((w - want).abs() < 1e-14, "quadrilateral node {j}: {w}");
        }
        let hex = serendipity_upgrade(CellType::Hexahedron).unwrap();
        let centre = &hex.weights[6 * 20..];
        for (j, w) in centre.iter().enumerate() {
            let want = if j < 8 { -0.25 } else { 0.25 };
            assert!((w - want).abs() < 1e-14, "hexahedron centre node {j}: {w}");
        }
        for cell in [CellType::Hexahedron, CellType::Prism, CellType::Pyramid] {
            let up = serendipity_upgrade(cell).unwrap();
            let n = up.num_file_nodes;
            let nv = crate::vertices(cell).len();
            let faces = if cell == CellType::Hexahedron {
                6
            } else {
                up.num_nodes - n
            };
            for row in up.weights.chunks(n).take(faces) {
                let mut nonzero: Vec<(usize, f64)> = row
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, w)| w.abs() > 1e-14)
                    .collect();
                assert_eq!(
                    nonzero.len(),
                    8,
                    "{cell:?}: a face centre uses its 8 face nodes"
                );
                nonzero.sort_by_key(|&(j, _)| j);
                for (j, w) in nonzero {
                    let want = if j < nv { -0.25 } else { 0.5 };
                    assert!((w - want).abs() < 1e-14, "{cell:?}: node {j}: {w}");
                }
            }
        }
    }

    #[test]
    fn serendipity_upgrade_only_for_serendipity_cells() {
        for cell in [
            CellType::Point,
            CellType::Interval,
            CellType::Triangle,
            CellType::Tetrahedron,
        ] {
            assert!(matches!(
                serendipity_upgrade(cell),
                Err(RefCellError::NotInFormat { degree: 2, .. })
            ));
        }
        for (cell, n, m) in [
            (CellType::Quadrilateral, 8, 9),
            (CellType::Hexahedron, 20, 27),
            (CellType::Prism, 15, 18),
            (CellType::Pyramid, 13, 14),
        ] {
            let up = serendipity_upgrade(cell).unwrap();
            assert_eq!((up.num_file_nodes, up.num_nodes), (n, m));
            assert_eq!(up.weights.len(), (m - n) * n);
        }
    }

    proptest! {
        /// Every function of the serendipity space is reproduced: interpolating it at the
        /// serendipity nodes and upgrading gives its values at the missing nodes.
        #[test]
        fn serendipity_reproduces_its_space(
            coeffs in proptest::collection::vec(-10.0f64..10.0, 20),
        ) {
            for cell in CELLS {
                let exps: &[Exponents] = match cell {
                    CellType::Quadrilateral => &QUADRILATERAL,
                    CellType::Hexahedron => &HEXAHEDRON,
                    _ => &PRISM,
                };
                let d = tdim(cell);
                let f = |x: &[f64]| -> f64 {
                    let mut padded = [0.0; 3];
                    padded[..d].copy_from_slice(x);
                    monomials(exps, &padded).iter().zip(&coeffs).map(|(m, c)| m * c).sum()
                };
                let up = serendipity_upgrade(cell).unwrap();
                let points = gmsh_p2_points(cell);
                let n = up.num_file_nodes;
                let values: Vec<f64> = points[..n * d].chunks_exact(d).map(f).collect();
                for (m, x) in points[n * d..].chunks_exact(d).enumerate() {
                    let got: f64 = up.weights[m * n..(m + 1) * n]
                        .iter()
                        .zip(&values)
                        .map(|(w, v)| w * v)
                        .sum();
                    let want = f(x);
                    prop_assert!((got - want).abs() < 1e-11 * (1.0 + want.abs()),
                        "{:?} missing node {}: {} vs {}", cell, m, got, want);
                }
            }
        }
    }
}
