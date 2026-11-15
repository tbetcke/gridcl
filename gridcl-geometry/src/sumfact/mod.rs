//! Sum factorisation (docs/plans/M9h.md, T7a–T8; ADR-0042, items 3 and 5).
//!
//! On quadrilaterals and hexahedra the Lagrange basis of a coordinate element is the tensor
//! product of the 1D Lagrange basis of the same degree and node family, under a permutation of
//! the nodes; a point set that is a tensor product of 1D point sets (a Gauss rule) records its
//! factors ([`TensorFactors`]). A sum-factorised kernel then contracts the E-vector with the 1D
//! tabulations one direction at a time, instead of summing over every node at every point.
//! On prisms the basis is the product of the triangle's and the interval's Lagrange bases
//! (Finding 3 of the plan), and the point sets are products of triangle points and points in
//! z: the kernel contracts the triangle factor, then z ([`prism`]).
//!
//! The kernels read one table per (element, point set) ([`TensorFactors::table`]), uploaded once
//! under CLAUDE.md's clause for tabulations: a header describing the tensor blocks of the point
//! set, the 1D tabulations of each direction, and the lexicographic-to-gridcl node permutation,
//! all as values of the scalar type (integers below 2^24, exact in f32), so no kernel binds a
//! sixth buffer (ADR-0042, item 5).
//!
//! The pull-back works on modal coefficients instead: a set-up contracts the E-vector with the
//! interval's coefficients one direction at a time, and each Newton step evaluates the 1D
//! Legendre polynomials at its changing point ([`pull_back`]; ADR-0049).

pub(crate) mod prism;
pub(crate) mod pull_back;
pub(crate) mod tensor;

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, NodeFamily};

use crate::names::element_error;

/// One factor of a [`TensorBlock`]: a direction, or the triangle (directions 0 and 1) of a prism.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Level {
    /// The reference directions (coordinates) the level sets, ascending.
    pub(crate) dims: Vec<usize>,
    /// The points of the level, `[i][dims]`.
    pub(crate) points: Vec<f64>,
    /// The step of the point index per step along the level.
    pub(crate) stride: usize,
}

impl Level {
    /// A level along the single direction `dim`.
    fn line(dim: usize, points: Vec<f64>, stride: usize) -> Self {
        Level {
            dims: vec![dim],
            points,
            stride,
        }
    }

    /// The number of points.
    pub(crate) fn len(&self) -> usize {
        self.points.len() / self.dims.len()
    }
}

/// Points `offset + Σ_ℓ i_ℓ · stride_ℓ` with coordinates `dims_ℓ` equal to `points_ℓ[i_ℓ]`, for
/// every multi-index (i_0, …) of the levels; level 0 is the innermost loop of the kernels.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorBlock {
    /// The index of the block's point with multi-index 0.
    pub(crate) offset: usize,
    /// One level per reference direction.
    pub(crate) levels: Vec<Level>,
}

/// The tensor structure of a point set: blocks that together hold each point once (one block
/// for a cell rule, one per facet for a facet rule). On quadrilaterals and hexahedra each level
/// is a direction; on prisms level 0 is the triangle (directions 0 and 1) and level 1 is z.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorFactors {
    pub(crate) blocks: Vec<TensorBlock>,
}

/// Header entries per block per level: points, point stride, table offset of the 1D values,
/// lexicographic node stride and reference direction.
pub(crate) const LEVEL_LEN: usize = 5;

/// The largest integer a table holds: every one is exact in f32.
const MAX_TABLE_INT: usize = 1 << 24;

impl TensorFactors {
    /// The factors of a tensor rule on [0, 1]^td with the 1D points `points` in every direction,
    /// the first direction running fastest (gridcl-refcell's `gauss_rule` on quadrilaterals and
    /// hexahedra).
    pub(crate) fn cube(td: usize, points: &[f64]) -> Self {
        let m = points.len();
        let levels = (0..td)
            .map(|d| Level::line(d, points.to_vec(), m.pow(d as u32)))
            .collect();
        TensorFactors {
            blocks: vec![TensorBlock { offset: 0, levels }],
        }
    }

    /// The factors of the prism rule with the triangle points `triangle` (`[i][2]`) and the
    /// points `line` in z, the triangle running fastest (gridcl-refcell's `gauss_rule` on
    /// prisms).
    pub(crate) fn prism(triangle: &[f64], line: &[f64]) -> Self {
        let mt = triangle.len() / 2;
        TensorFactors {
            blocks: vec![TensorBlock {
                offset: 0,
                levels: vec![
                    Level {
                        dims: vec![0, 1],
                        points: triangle.to_vec(),
                        stride: 1,
                    },
                    Level::line(2, line.to_vec(), mt),
                ],
            }],
        }
    }

    /// The factors of the facet rule of `cell` (a quadrilateral, hexahedron or prism) whose
    /// facets carry the Gauss rules of `degree`, mapped facet by facet into cell coordinates by
    /// `facet_points_to_cell`, facet-major (`FacetQuadRule::gauss`): one block per facet.
    /// - Quadrilaterals and hexahedra: the levels follow the facet's axes and end with the
    ///   direction normal to the facet, a level of one point.
    /// - Prisms: a triangle facet is its triangle rule times one point in z; a quadrilateral
    ///   facet runs along a triangle edge (its first axis, a level of points in directions 0
    ///   and 1) and along z (its second axis).
    ///
    /// # Panics
    ///
    /// If the cell is not a quadrilateral, hexahedron or prism.
    pub(crate) fn facets(cell: CellType, degree: u32) -> Self {
        if cell == CellType::Prism {
            return Self::prism_facets(degree);
        }
        assert!(
            matches!(cell, CellType::Quadrilateral | CellType::Hexahedron),
            "facet sum factorisation on {cell:?}"
        );
        let line = &gridcl_refcell::gauss_rule(CellType::Interval, degree).points;
        let td = gridcl_refcell::tdim(cell);
        let fd = td - 1;
        let m = line.len();
        let per_facet = m.pow(fd as u32);
        // The facet points, the first facet direction fastest: gauss_rule on the facet type.
        let facet_points: Vec<f64> = (0..per_facet)
            .flat_map(|flat| (0..fd).map(move |j| line[flat / m.pow(j as u32) % m]))
            .collect();
        // The facet's origin and its axes' ends: 0 and the unit points e_j.
        let corners: Vec<f64> = (0..=fd)
            .flat_map(|k| (0..fd).map(move |j| if k == j + 1 { 1.0 } else { 0.0 }))
            .collect();
        let num_facets = gridcl_refcell::sub_entity_types(cell, fd).len();
        let blocks = (0..num_facets)
            .map(|f| {
                let ends = gridcl_refcell::facet_points_to_cell(cell, f, &corners);
                let x = gridcl_refcell::facet_points_to_cell(cell, f, &facet_points);
                // The cell direction along facet axis j: the one coordinate it changes.
                let axis = |j: usize| {
                    let moved: Vec<usize> = (0..td)
                        .filter(|&d| ends[(j + 1) * td + d] != ends[d])
                        .collect();
                    assert_eq!(
                        moved.len(),
                        1,
                        "{cell:?} facet {f}: axis {j} is not a direction"
                    );
                    moved[0]
                };
                let mut levels: Vec<Level> = (0..fd)
                    .map(|j| {
                        let dim = axis(j);
                        let stride = m.pow(j as u32);
                        let points = (0..m).map(|i| x[i * stride * td + dim]).collect();
                        Level::line(dim, points, stride)
                    })
                    .collect();
                let normal = (0..td)
                    .find(|d| levels.iter().all(|l| l.dims[0] != *d))
                    .expect("a direction normal to the facet");
                levels.push(Level::line(normal, vec![x[normal]], 0));
                TensorBlock {
                    offset: f * per_facet,
                    levels,
                }
            })
            .collect();
        TensorFactors { blocks }
    }

    /// [`facets`](TensorFactors::facets) of the prism.
    fn prism_facets(degree: u32) -> Self {
        let cell = CellType::Prism;
        let mut offset = 0;
        let mut blocks = Vec::new();
        for (f, &facet) in gridcl_refcell::sub_entity_types(cell, 2).iter().enumerate() {
            let rule = gridcl_refcell::gauss_rule(facet, degree);
            let x = gridcl_refcell::facet_points_to_cell(cell, f, &rule.points);
            let count = x.len() / 3;
            // (x, y) of points 0..k and z of every `step`-th point.
            let xy = |k: usize| (0..k).flat_map(|i| [x[3 * i], x[3 * i + 1]]).collect();
            let z = |k: usize, step: usize| (0..k).map(|i| x[3 * i * step + 2]).collect();
            let levels = if facet == CellType::Triangle {
                // z is the same at every point of a facet z = 0 or z = 1.
                vec![
                    Level {
                        dims: vec![0, 1],
                        points: xy(count),
                        stride: 1,
                    },
                    Level::line(2, z(1, 1), 0),
                ]
            } else {
                // The facet's first axis runs along a triangle edge, its second along z (the
                // prism's facets list a top vertex third, `sub_entities`).
                let m = count.isqrt();
                vec![
                    Level {
                        dims: vec![0, 1],
                        points: xy(m),
                        stride: 1,
                    },
                    Level::line(2, z(m, m), m),
                ]
            };
            blocks.push(TensorBlock { offset, levels });
            offset += count;
        }
        TensorFactors { blocks }
    }

    /// The points `[q][td]` the factors describe, or `None` unless the blocks hold each of
    /// `num_points` points exactly once.
    pub(crate) fn points(&self, td: usize, num_points: usize) -> Option<Vec<f64>> {
        let mut out = vec![0.0; num_points * td];
        let mut seen = vec![false; num_points];
        for block in &self.blocks {
            let dims: Vec<usize> = block.levels.iter().flat_map(|l| l.dims.clone()).collect();
            if !(0..td).all(|d| dims.contains(&d)) || dims.len() != td {
                return None;
            }
            let count: usize = block.levels.iter().map(Level::len).product();
            for flat in 0..count {
                let mut rest = flat;
                let mut q = block.offset;
                let mut x = vec![0.0; td];
                for level in &block.levels {
                    let m = level.len();
                    let i = rest % m;
                    rest /= m;
                    q += i * level.stride;
                    let k = level.dims.len();
                    for (&d, &c) in level.dims.iter().zip(&level.points[i * k..(i + 1) * k]) {
                        x[d] = c;
                    }
                }
                if q >= num_points || std::mem::replace(&mut seen[q], true) {
                    return None;
                }
                out[q * td..(q + 1) * td].copy_from_slice(&x);
            }
        }
        seen.iter().all(|&s| s).then_some(out)
    }

    /// The table of the sum-factorised kernels for the element of `degree` and `family` on
    /// `cell` (a quadrilateral, hexahedron or prism), in f64:
    /// - `[0]` the number of blocks, `[1]` the offset of the permutation;
    /// - per block, its point offset, then per level ([`LEVEL_LEN`] entries): the number of
    ///   points m, the point stride, the offset of the level's values `[i][a]` (m × n, the
    ///   derivatives `[i][a]` follow), the lexicographic stride of its node index and its first
    ///   direction;
    /// - the tables, level by level;
    /// - the permutation: the gridcl node of each lexicographic multi-index of the factors'
    ///   nodes.
    ///
    /// On quadrilaterals and hexahedra each level is a direction with n = p + 1 nodes of
    /// gridcl-refcell's interval element of the same degree and family, its lexicographic stride
    /// n^dim. On prisms level 0 is the triangle: its nt = n (n + 1) / 2 nodes of the triangle
    /// element, the values `[i][t]` followed by the x and y derivatives, lexicographic stride 1;
    /// level 1 is z, as on hexahedra, with lexicographic stride nt. The permutation's entry
    /// t + nt c is the prism node at triangle node t and interval node c.
    ///
    /// # Errors
    ///
    /// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element gridcl-refcell does not
    /// offer; `TooLarge` if an entry reaches 2^24.
    ///
    /// # Panics
    ///
    /// If the cell is not a quadrilateral, hexahedron or prism, the levels do not have the
    /// cell's form, or the cell's nodes are not the product of the factors' nodes.
    pub(crate) fn table(
        &self,
        cell: CellType,
        degree: u32,
        family: NodeFamily,
    ) -> Result<Vec<f64>> {
        let td = gridcl_refcell::tdim(cell);
        let prism = cell == CellType::Prism;
        assert!(
            prism || matches!(cell, CellType::Quadrilateral | CellType::Hexahedron),
            "sum factorisation on {cell:?}"
        );
        let line =
            LagrangeElement::new(CellType::Interval, degree, family).map_err(element_error)?;
        let triangle = if prism {
            Some(LagrangeElement::new(CellType::Triangle, degree, family).map_err(element_error)?)
        } else {
            None
        };
        let n = line.dim();
        let nt = triangle.as_ref().map_or(0, LagrangeElement::dim);
        let num_levels = if prism { 2 } else { td };
        let header = 2 + self.blocks.len() * (1 + LEVEL_LEN * num_levels);
        let mut out = vec![0.0; header];
        out[0] = self.blocks.len() as f64;
        let mut h = 2;
        for block in &self.blocks {
            assert_eq!(
                block.levels.len(),
                num_levels,
                "{cell:?}: tensor block levels"
            );
            out[h] = block.offset as f64;
            h += 1;
            for (l, level) in block.levels.iter().enumerate() {
                let m = level.len();
                let at = out.len();
                let (dim, lex) = match (&triangle, l) {
                    (Some(tri), 0) => {
                        assert_eq!(level.dims, [0, 1], "prism level 0 is the triangle");
                        let tab = tri.tabulate(&level.points, 1).map_err(element_error)?;
                        for deriv in 0..3 {
                            for i in 0..m {
                                out.extend((0..nt).map(|t| tab.get(deriv, i, t)));
                            }
                        }
                        (0, 1)
                    }
                    _ => {
                        let [dim] = level.dims[..] else {
                            panic!("{cell:?}: level {l} is not a direction")
                        };
                        assert!(!prism || dim == 2, "prism level 1 is z");
                        let tab = line.tabulate(&level.points, 1).map_err(element_error)?;
                        for deriv in 0..2 {
                            for i in 0..m {
                                out.extend((0..n).map(|a| tab.get(deriv, i, a)));
                            }
                        }
                        (dim, if prism { nt } else { n.pow(dim as u32) })
                    }
                };
                let entry = [m, level.stride, at, lex, dim];
                for (k, v) in entry.into_iter().enumerate() {
                    out[h + k] = v as f64;
                }
                h += LEVEL_LEN;
            }
        }
        out[1] = out.len() as f64;
        let perm = match &triangle {
            Some(tri) => prism_permutation(degree, family, tri, &line)?,
            None => permutation(cell, degree, family, &line)?,
        };
        out.extend(perm.into_iter().map(|k| k as f64));
        if out.len() >= MAX_TABLE_INT
            || self.blocks.iter().any(|b| {
                b.offset >= MAX_TABLE_INT || b.levels.iter().any(|l| l.stride >= MAX_TABLE_INT)
            })
        {
            return Err(GridError::TooLarge);
        }
        Ok(out)
    }
}

/// The gridcl node of each lexicographic multi-index of the interval element `line`'s nodes on
/// `cell`, the first direction fastest.
///
/// # Panics
///
/// If a tensor-grid point is not a node of the cell's element.
fn permutation(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    line: &LagrangeElement,
) -> Result<Vec<usize>> {
    let td = gridcl_refcell::tdim(cell);
    let element = LagrangeElement::new(cell, degree, family).map_err(element_error)?;
    let nodes = element.points();
    let t = line.points();
    let n = t.len();
    // The 1D node nearest to coordinate x.
    let nearest = |x: f64| {
        (0..n)
            .min_by(|&a, &b| (t[a] - x).abs().total_cmp(&(t[b] - x).abs()))
            .filter(|&a| (t[a] - x).abs() < 1e-12)
            .unwrap_or_else(|| panic!("{cell:?} node coordinate {x} is no 1D node"))
    };
    let mut perm = vec![usize::MAX; n.pow(td as u32)];
    for (k, x) in nodes.chunks_exact(td).enumerate() {
        let lex = x.iter().rev().fold(0, |acc, &c| acc * n + nearest(c));
        assert_eq!(
            perm[lex],
            usize::MAX,
            "{cell:?}: two nodes at one tensor point"
        );
        perm[lex] = k;
    }
    Ok(perm)
}

/// The prism node at each lexicographic pair (t, c) of the triangle element `triangle`'s nodes
/// and the interval element `line`'s nodes, at entry t + nt c.
///
/// # Panics
///
/// If a product point is not a node of the prism's element.
fn prism_permutation(
    degree: u32,
    family: NodeFamily,
    triangle: &LagrangeElement,
    line: &LagrangeElement,
) -> Result<Vec<usize>> {
    let element = LagrangeElement::new(CellType::Prism, degree, family).map_err(element_error)?;
    let (s, t) = (triangle.points(), line.points());
    let (nt, n) = (s.len() / 2, t.len());
    // The index of the entry of `pts` (of `k` coordinates) at `x`.
    let nearest = |pts: &[f64], k: usize, x: &[f64]| {
        (0..pts.len() / k)
            .find(|&a| (0..k).all(|d| (pts[a * k + d] - x[d]).abs() < 1e-12))
            .unwrap_or_else(|| panic!("prism node {x:?} is no product node"))
    };
    let mut perm = vec![usize::MAX; nt * n];
    for (k, x) in element.points().as_chunks::<3>().0.iter().enumerate() {
        let lex = nearest(s, 2, &x[..2]) + nt * nearest(t, 1, &x[2..]);
        assert_eq!(
            perm[lex],
            usize::MAX,
            "prism: two nodes at one product point"
        );
        perm[lex] = k;
    }
    Ok(perm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The factors describe `points`, and the tensor product of the 1D tables of `table`,
    /// permuted, is the element's tabulation there, block by block.
    fn check_table(
        cell: CellType,
        degree: u32,
        family: NodeFamily,
        factors: &TensorFactors,
        points: &[f64],
    ) {
        let td = gridcl_refcell::tdim(cell);
        let num_points = points.len() / td;
        assert_eq!(factors.points(td, num_points).unwrap(), points);
        let t = factors.table(cell, degree, family).unwrap();
        let element = LagrangeElement::new(cell, degree, family).unwrap();
        let want = element.tabulate(points, 1).unwrap();
        let n = degree as usize + 1;
        let perm = t[1] as usize;
        assert_eq!(t[0] as usize, factors.blocks.len());
        for (blk, block) in factors.blocks.iter().enumerate() {
            let h = 2 + blk * (1 + LEVEL_LEN * td);
            assert_eq!(t[h] as usize, block.offset);
            let level = |l: usize| &t[h + 1 + l * LEVEL_LEN..h + 1 + (l + 1) * LEVEL_LEN];
            let count: usize = (0..td).map(|l| level(l)[0] as usize).product();
            for flat in 0..count {
                // The multi-index of the point, level 0 fastest.
                let mut rest = flat;
                let mut q = block.offset;
                let mut i = vec![0; td];
                for (l, il) in i.iter_mut().enumerate() {
                    let (m, stride, dim) =
                        (level(l)[0] as usize, level(l)[1] as usize, level(l)[4]);
                    assert_eq!(level(l)[3] as usize, n.pow(dim as u32));
                    *il = rest % m;
                    rest /= m;
                    q += *il * stride;
                }
                for lex in 0..n.pow(td as u32) {
                    let a: Vec<usize> = (0..td).map(|d| lex / n.pow(d as u32) % n).collect();
                    let k = t[perm + lex] as usize;
                    // ∂_e of Π_d φ_{a_d}(x_d): the derivative table in direction e.
                    for deriv in 0..=td {
                        let got: f64 = (0..td)
                            .map(|l| {
                                let lv = level(l);
                                let (m, at, dim) = (lv[0] as usize, lv[2] as usize, lv[4] as usize);
                                let row = if deriv == dim + 1 { at + m * n } else { at };
                                t[row + i[l] * n + a[dim]]
                            })
                            .product();
                        let w = want.get(deriv, q, k);
                        assert!(
                            (got - w).abs() < 1e-10 * (1.0 + w.abs()),
                            "{cell:?} p={degree} {family:?} q={q} k={k} deriv={deriv}: {got} vs {w}"
                        );
                    }
                }
            }
        }
    }

    /// The tensor Gauss rule of `rule_degree` on `cell`.
    fn check_products(cell: CellType, degree: u32, family: NodeFamily, rule_degree: u32) {
        let td = gridcl_refcell::tdim(cell);
        let rule = gridcl_refcell::gauss_rule(cell, rule_degree);
        let line = gridcl_refcell::gauss_rule(CellType::Interval, rule_degree).points;
        check_table(
            cell,
            degree,
            family,
            &TensorFactors::cube(td, &line),
            &rule.points,
        );
    }

    /// The facet Gauss rules of `rule_degree` on `cell`, as `FacetQuadRule::gauss` maps them.
    fn check_facet_products(cell: CellType, degree: u32, family: NodeFamily, rule_degree: u32) {
        let td = gridcl_refcell::tdim(cell);
        let facet_types = gridcl_refcell::sub_entity_types(cell, td - 1);
        let points: Vec<f64> = facet_types
            .iter()
            .enumerate()
            .flat_map(|(f, &facet)| {
                let rule = gridcl_refcell::gauss_rule(facet, rule_degree);
                gridcl_refcell::facet_points_to_cell(cell, f, &rule.points)
            })
            .collect();
        let factors = TensorFactors::facets(cell, rule_degree);
        if cell == CellType::Prism {
            check_prism_table(degree, &factors, &points);
        } else {
            check_table(cell, degree, family, &factors, &points);
        }
    }

    /// As [`check_table`] on the prism: the triangle table times the z table, permuted.
    fn check_prism_table(degree: u32, factors: &TensorFactors, points: &[f64]) {
        let family = NodeFamily::Equispaced;
        let num_points = points.len() / 3;
        assert_eq!(factors.points(3, num_points).unwrap(), points);
        let t = factors.table(CellType::Prism, degree, family).unwrap();
        let element = LagrangeElement::new(CellType::Prism, degree, family).unwrap();
        let want = element.tabulate(points, 1).unwrap();
        let n = degree as usize + 1;
        let nt = n * (n + 1) / 2;
        let perm = t[1] as usize;
        assert_eq!(t[0] as usize, factors.blocks.len());
        for (blk, block) in factors.blocks.iter().enumerate() {
            let h = 2 + blk * (1 + LEVEL_LEN * 2);
            assert_eq!(t[h] as usize, block.offset);
            let tri = &t[h + 1..h + 1 + LEVEL_LEN];
            let z = &t[h + 1 + LEVEL_LEN..h + 1 + 2 * LEVEL_LEN];
            assert_eq!((tri[3], tri[4], z[3], z[4]), (1.0, 0.0, nt as f64, 2.0));
            let (mt, mz) = (tri[0] as usize, z[0] as usize);
            for (i, j) in (0..mz).flat_map(|j| (0..mt).map(move |i| (i, j))) {
                let q = block.offset + i * tri[1] as usize + j * z[1] as usize;
                for (lex, &k) in t[perm..perm + nt * n].iter().enumerate() {
                    let (a, c) = (lex % nt, lex / nt);
                    let psi = |d: usize| t[tri[2] as usize + (d * mt + i) * nt + a];
                    let phi = |d: usize| t[z[2] as usize + (d * mz + j) * n + c];
                    let got = [
                        psi(0) * phi(0),
                        psi(1) * phi(0),
                        psi(2) * phi(0),
                        psi(0) * phi(1),
                    ];
                    for (deriv, got) in got.into_iter().enumerate() {
                        let w = want.get(deriv, q, k as usize);
                        assert!(
                            (got - w).abs() < 1e-10 * (1.0 + w.abs()),
                            "prism p={degree} q={q} k={k} deriv={deriv}: {got} vs {w}"
                        );
                    }
                }
            }
        }
    }

    /// The prism's Gauss rule of `rule_degree`.
    fn check_prism_products(degree: u32, rule_degree: u32) {
        let rule = gridcl_refcell::gauss_rule(CellType::Prism, rule_degree);
        let triangle = gridcl_refcell::gauss_rule(CellType::Triangle, rule_degree).points;
        let line = gridcl_refcell::gauss_rule(CellType::Interval, rule_degree).points;
        check_prism_table(
            degree,
            &TensorFactors::prism(&triangle, &line),
            &rule.points,
        );
    }

    #[test]
    fn tensor_tables_reproduce_the_element() {
        for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
            for degree in 1..=4 {
                check_products(CellType::Quadrilateral, degree, family, 2 * degree);
                check_products(CellType::Hexahedron, degree, family, 2 * degree);
            }
            check_products(CellType::Quadrilateral, 8, family, 3);
            check_products(CellType::Hexahedron, 6, family, 1);
        }
    }

    #[test]
    fn facet_tables_reproduce_the_element() {
        for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
            for degree in 1..=4 {
                check_facet_products(CellType::Quadrilateral, degree, family, 2 * degree);
                check_facet_products(CellType::Hexahedron, degree, family, 2 * degree);
            }
            // One point per facet direction.
            check_facet_products(CellType::Quadrilateral, 8, family, 1);
            check_facet_products(CellType::Hexahedron, 5, family, 1);
        }
    }

    #[test]
    fn prism_tables_reproduce_the_element() {
        for degree in 1..=4 {
            check_prism_products(degree, 2 * degree);
            check_facet_products(CellType::Prism, degree, NodeFamily::Equispaced, 2 * degree);
        }
        check_prism_products(8, 1);
        check_facet_products(CellType::Prism, 6, NodeFamily::Equispaced, 1);
    }

    #[test]
    fn points_need_every_point_once() {
        let f = TensorFactors::cube(2, &[0.25, 0.75]);
        assert_eq!(
            f.points(2, 4).unwrap(),
            [0.25, 0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.75]
        );
        assert!(f.points(2, 5).is_none());
        assert!(f.points(2, 3).is_none());
        assert!(f.points(3, 4).is_none());
        let mut twice = f.clone();
        twice.blocks.push(f.blocks[0].clone());
        assert!(twice.points(2, 4).is_none());
    }
}
