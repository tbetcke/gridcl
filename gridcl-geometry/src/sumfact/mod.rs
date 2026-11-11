//! Sum factorisation (docs/plans/M9h.md, T7a–T8; ADR-0042, items 3 and 5).
//!
//! On quadrilaterals and hexahedra the Lagrange basis of a coordinate element is the tensor
//! product of the 1D Lagrange basis of the same degree and node family, under a permutation of
//! the nodes; a point set that is a tensor product of 1D point sets (a Gauss rule) records its
//! factors ([`TensorFactors`]). A sum-factorised kernel then contracts the E-vector with the 1D
//! tabulations one direction at a time, instead of summing over every node at every point.
//!
//! The kernels read one table per (element, point set) ([`TensorFactors::table`]), uploaded once
//! under CLAUDE.md's clause for tabulations: a header describing the tensor blocks of the point
//! set, the 1D tabulations of each direction, and the lexicographic-to-gridcl node permutation,
//! all as values of the scalar type (integers below 2^24, exact in f32), so no kernel binds a
//! sixth buffer (ADR-0042, item 5).

pub(crate) mod tensor;

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, NodeFamily};

use crate::names::element_error;

/// One direction of a [`TensorBlock`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Level {
    /// The reference direction (coordinate) the level runs along.
    pub(crate) dim: usize,
    /// The 1D points of the direction.
    pub(crate) points: Vec<f64>,
    /// The step of the point index per step along the direction.
    pub(crate) stride: usize,
}

/// Points `offset + Σ_ℓ i_ℓ · stride_ℓ` with coordinate `dim_ℓ` equal to `points_ℓ[i_ℓ]`, for every
/// multi-index (i_0, …) of the levels; level 0 is the innermost loop of the kernels.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorBlock {
    /// The index of the block's point with multi-index 0.
    pub(crate) offset: usize,
    /// One level per reference direction.
    pub(crate) levels: Vec<Level>,
}

/// The tensor structure of a point set: blocks that together hold each point once (one block
/// for a cell rule).
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
            .map(|d| Level {
                dim: d,
                points: points.to_vec(),
                stride: m.pow(d as u32),
            })
            .collect();
        TensorFactors {
            blocks: vec![TensorBlock { offset: 0, levels }],
        }
    }

    /// The points `[q][td]` the factors describe, or `None` unless the blocks hold each of
    /// `num_points` points exactly once.
    pub(crate) fn points(&self, td: usize, num_points: usize) -> Option<Vec<f64>> {
        let mut out = vec![0.0; num_points * td];
        let mut seen = vec![false; num_points];
        for block in &self.blocks {
            if block.levels.len() != td {
                return None;
            }
            let count: usize = block.levels.iter().map(|l| l.points.len()).product();
            for flat in 0..count {
                let mut rest = flat;
                let mut q = block.offset;
                let mut x = vec![0.0; td];
                for level in &block.levels {
                    let m = level.points.len();
                    let i = rest % m;
                    rest /= m;
                    q += i * level.stride;
                    *x.get_mut(level.dim)? = level.points[i];
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
    /// `cell` (a quadrilateral or hexahedron), in f64:
    /// - `[0]` the number of blocks, `[1]` the offset of the permutation;
    /// - per block, its point offset, then per level ([`LEVEL_LEN`] entries): the number of
    ///   points m, the point stride, the offset of the 1D values `[i][a]` (m × n, the
    ///   derivatives `[i][a]` follow), the lexicographic stride n^dim and the direction;
    /// - the 1D tables, level by level;
    /// - the permutation: entry a_0 + n a_1 + n² a_2 is the gridcl node at the 1D nodes
    ///   (a_0, a_1, a_2), indices of gridcl-refcell's interval element of the same degree and
    ///   family.
    ///
    /// # Errors
    ///
    /// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element gridcl-refcell does not
    /// offer; `TooLarge` if an entry reaches 2^24.
    ///
    /// # Panics
    ///
    /// If the cell is not a quadrilateral or hexahedron, or its nodes are not the tensor grid of
    /// the interval element's nodes.
    pub(crate) fn table(
        &self,
        cell: CellType,
        degree: u32,
        family: NodeFamily,
    ) -> Result<Vec<f64>> {
        assert!(
            matches!(cell, CellType::Quadrilateral | CellType::Hexahedron),
            "sum factorisation on {cell:?}"
        );
        let td = gridcl_refcell::tdim(cell);
        let line =
            LagrangeElement::new(CellType::Interval, degree, family).map_err(element_error)?;
        let n = line.dim();
        let header = 2 + self.blocks.len() * (1 + LEVEL_LEN * td);
        let mut out = vec![0.0; header];
        out[0] = self.blocks.len() as f64;
        let mut h = 2;
        for block in &self.blocks {
            out[h] = block.offset as f64;
            h += 1;
            for level in &block.levels {
                let m = level.points.len();
                let tab = line.tabulate(&level.points, 1).map_err(element_error)?;
                let at = out.len();
                for deriv in 0..2 {
                    for i in 0..m {
                        out.extend((0..n).map(|a| tab.get(deriv, i, a)));
                    }
                }
                let entry = [m, level.stride, at, n.pow(level.dim as u32), level.dim];
                for (k, v) in entry.into_iter().enumerate() {
                    out[h + k] = v as f64;
                }
                h += LEVEL_LEN;
            }
        }
        out[1] = out.len() as f64;
        out.extend(
            permutation(cell, degree, family, &line)?
                .into_iter()
                .map(|k| k as f64),
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The tensor product of the 1D tables of `table`, permuted, is the element's tabulation.
    fn check_products(cell: CellType, degree: u32, family: NodeFamily, rule_degree: u32) {
        let td = gridcl_refcell::tdim(cell);
        let rule = gridcl_refcell::gauss_rule(cell, rule_degree);
        let line = gridcl_refcell::gauss_rule(CellType::Interval, rule_degree).points;
        let factors = TensorFactors::cube(td, &line);
        assert_eq!(factors.points(td, rule.weights.len()).unwrap(), rule.points);
        let t = factors.table(cell, degree, family).unwrap();
        let element = LagrangeElement::new(cell, degree, family).unwrap();
        let want = element.tabulate(&rule.points, 1).unwrap();
        let n = degree as usize + 1;
        let m = line.len();
        let perm = t[1] as usize;
        let level = |l: usize| &t[3 + l * LEVEL_LEN..3 + (l + 1) * LEVEL_LEN];
        for q in 0..rule.weights.len() {
            // The multi-index of point q, level ℓ = direction ℓ.
            let i: Vec<usize> = (0..td).map(|d| q / m.pow(d as u32) % m).collect();
            for lex in 0..n.pow(td as u32) {
                let a: Vec<usize> = (0..td).map(|d| lex / n.pow(d as u32) % n).collect();
                let k = t[perm + lex] as usize;
                // ∂_e of Π_d φ_{a_d}(x_d): the derivative table in direction e.
                for deriv in 0..=td {
                    let got: f64 = (0..td)
                        .map(|d| {
                            let l = level(d);
                            let (mm, at) = (l[0] as usize, l[2] as usize);
                            let row = if deriv == d + 1 { at + mm * n } else { at };
                            t[row + i[d] * n + a[d]]
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
