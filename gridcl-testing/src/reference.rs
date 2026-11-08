//! Host f64 references of the geometric factors and the pull-back, and the degree-scaled
//! tolerances of the M9h acceptance tests (docs/plans/M9h.md, T0a; ADR-0042, item 2).
//!
//! The device functions ([`points`], [`jacobians`], [`detj`], [`pseudo_inverses`], [`normals`],
//! [`facet_normals`], [`pull_back`]) evaluate a block of a `DeviceGrid` with gridcl-refcell's
//! f64 tabulation of the block's coordinate element, at the node coordinates the grid stores:
//! its node L-vector widened to f64 plus its origin, as `Mesh::read_node` reads them before it
//! narrows. They read the nodes and the block's dofmap to the host (test code only). Results
//! are flat `[cell][point][component]` in the block's local cell order, the order of
//! `QuadField::to_host_cell_major`, with the components of the device factors:
//!
//! - X: g; J: d · GD + g (column-major GD × TD); det J: one; K: g · TD + t (column-major
//!   TD × GD); normals and facet normals: g.
//!
//! [`host`] evaluates X, J and det J of a [`HostMesh`] block directly.
//!
//! Tolerances (ADR-0042, item 2): the f32 and f64 tolerances scale by max(1, A / A_ref), where
//! A is an amplification factor of the element at the test's points ([`Amplification`]) and
//! A_ref the same factor of the equispaced element of degree 4 (3 on pyramids) of that cell at
//! the same points. Errors are relative per quantity over a block ([`rel_err`]).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use gridcl::{DeviceGrid, FacetPointSet, FieldShape, NodeFamily, QuadField, Real, Result};
use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, Tabulation};

use crate::{HostMesh, TOL_F32, TOL_F64};

// ---------------------------------------------------------------------------------------------
// Cell coordinates.

/// The node coordinates of every cell of a block, `[cell][node][gdim]`, with its element.
struct BlockNodes {
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    gd: usize,
    k: usize,
    coords: Vec<f64>,
}

impl BlockNodes {
    /// Block `b` of a device grid: stored nodes plus the origin, through the dofmap.
    fn of_grid<G: DeviceGrid>(g: &G, b: usize) -> Result<Self> {
        let gb = g.geometry_block(b);
        let cell: CellType = gb.element.cell.into();
        let k = gb.nodes_per_cell();
        assert_eq!(gb.dofmap.stride as usize, k, "dofmap stride");
        let gd = <G::GD as gridcl::Dim>::N;
        let nodes = g.nodes().to_host()?;
        let origin = g.origin();
        let dofmap = gb.dofmap.data.to_host()?;
        let coords = dofmap
            .iter()
            .flat_map(|&n| {
                let n = n as usize;
                (0..gd).map(move |i| (n, i))
            })
            .map(|(n, i)| nodes[n * gd + i].widen() + origin.as_ref()[i])
            .collect();
        Ok(BlockNodes {
            cell,
            degree: gb.element.degree,
            family: gb.element.family,
            gd,
            k,
            coords,
        })
    }

    /// Block `b` of a host mesh.
    fn of_host(h: &HostMesh, b: usize) -> Self {
        let block = &h.blocks[b];
        let gd = h.gdim;
        let coords = block
            .cells
            .iter()
            .flat_map(|&n| {
                h.nodes[n as usize * gd..(n as usize + 1) * gd]
                    .iter()
                    .copied()
            })
            .collect();
        BlockNodes {
            cell: block.cell,
            degree: block.degree,
            family: block.family,
            gd,
            k: block.nodes_per_cell(),
            coords,
        }
    }

    fn td(&self) -> usize {
        gridcl_refcell::tdim(self.cell)
    }

    fn num_cells(&self) -> usize {
        self.coords.len().checked_div(self.k * self.gd).unwrap_or(0)
    }

    fn element(&self) -> LagrangeElement {
        LagrangeElement::new(self.cell, self.degree, self.family).expect("element")
    }

    /// The nodes of cell `c`, `[node][gdim]`.
    fn cell_coords(&self, c: usize) -> &[f64] {
        let n = self.k * self.gd;
        &self.coords[c * n..(c + 1) * n]
    }

    /// X (`[cell][point][g]`) at the points of `tab` (row 0 of the tabulation).
    fn points(&self, tab: &Tabulation) -> Vec<f64> {
        let (gd, nq) = (self.gd, tab.num_points);
        let mut out = Vec::with_capacity(self.num_cells() * nq * gd);
        for c in 0..self.num_cells() {
            let x = self.cell_coords(c);
            for q in 0..nq {
                for g in 0..gd {
                    out.push((0..self.k).map(|i| x[i * gd + g] * tab.get(0, q, i)).sum());
                }
            }
        }
        out
    }

    /// J (`[cell][point][d · GD + g]`) at the points of `tab`, which has derivatives.
    fn jacobians(&self, tab: &Tabulation) -> Vec<f64> {
        let (gd, td, nq) = (self.gd, self.td(), tab.num_points);
        let mut out = Vec::with_capacity(self.num_cells() * nq * td * gd);
        for c in 0..self.num_cells() {
            let x = self.cell_coords(c);
            for q in 0..nq {
                for d in 0..td {
                    for g in 0..gd {
                        out.push(
                            (0..self.k)
                                .map(|i| x[i * gd + g] * tab.get(1 + d, q, i))
                                .sum(),
                        );
                    }
                }
            }
        }
        out
    }
}

/// The tabulation of `nodes`' element at `points` with first derivatives.
fn tabulate(nodes: &BlockNodes, points: &[f64], nderivs: u32) -> Tabulation {
    nodes.element().tabulate(points, nderivs).expect("tabulate")
}

// ---------------------------------------------------------------------------------------------
// Per-point maths on column-major J.

/// det J (TD = GD) or the pseudo-determinant sqrt(det(JᵀJ)) of the column-major GD × TD `j`.
pub fn detj_of(j: &[f64], td: usize, gd: usize) -> f64 {
    let col = |d: usize| &j[d * gd..(d + 1) * gd];
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    match (td, gd) {
        (1, 1) => j[0],
        (2, 2) => j[0] * j[3] - j[2] * j[1],
        (3, 3) => {
            let m = |g: usize, d: usize| j[d * 3 + g];
            m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
        }
        (1, _) => dot(col(0), col(0)).sqrt(),
        (2, 3) => {
            let (a, b, c) = (
                dot(col(0), col(0)),
                dot(col(0), col(1)),
                dot(col(1), col(1)),
            );
            (a * c - b * b).sqrt()
        }
        _ => panic!("detj_of: no (TD, GD) = ({td}, {gd})"),
    }
}

/// K = (JᵀJ)⁻¹Jᵀ (J⁻¹ when TD = GD) of the column-major GD × TD `j`, column-major TD × GD:
/// K(t, g) at g · TD + t.
pub fn pinv_of(j: &[f64], td: usize, gd: usize) -> Vec<f64> {
    // G = JᵀJ (TD × TD), its inverse, then K = G⁻¹ Jᵀ.
    let jj = |a: usize, b: usize| (0..gd).map(|g| j[a * gd + g] * j[b * gd + g]).sum::<f64>();
    let ginv: Vec<f64> = match td {
        1 => vec![1.0 / jj(0, 0)],
        2 => {
            let (a, b, c) = (jj(0, 0), jj(0, 1), jj(1, 1));
            let inv = 1.0 / (a * c - b * b);
            vec![c * inv, -b * inv, -b * inv, a * inv]
        }
        3 => {
            let m: Vec<f64> = (0..9).map(|i| jj(i / 3, i % 3)).collect();
            let cof = |r0: usize, r1: usize, c0: usize, c1: usize| {
                m[r0 * 3 + c0] * m[r1 * 3 + c1] - m[r0 * 3 + c1] * m[r1 * 3 + c0]
            };
            let adj = [
                cof(1, 2, 1, 2),
                -cof(0, 2, 1, 2),
                cof(0, 1, 1, 2),
                -cof(1, 2, 0, 2),
                cof(0, 2, 0, 2),
                -cof(0, 1, 0, 2),
                cof(1, 2, 0, 1),
                -cof(0, 2, 0, 1),
                cof(0, 1, 0, 1),
            ];
            let det = m[0] * adj[0] + m[1] * adj[3] + m[2] * adj[6];
            adj.iter().map(|a| a / det).collect()
        }
        _ => panic!("pinv_of: TD = {td}"),
    };
    let mut k = vec![0.0; td * gd];
    for g in 0..gd {
        for t in 0..td {
            k[g * td + t] = (0..td).map(|s| ginv[t * td + s] * j[s * gd + g]).sum();
        }
    }
    k
}

/// The Frobenius condition number ‖J‖_F ‖K‖_F of the column-major GD × TD `j`: at least the
/// 2-norm condition number and at most TD times it.
pub fn condition_of(j: &[f64], td: usize, gd: usize) -> f64 {
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    norm(&j[..td * gd]) * norm(&pinv_of(j, td, gd))
}

/// `f` of every point's J, flattened.
fn per_point(j: &[f64], td: usize, gd: usize, f: impl Fn(&[f64]) -> Vec<f64>) -> Vec<f64> {
    j.chunks_exact(td * gd).flat_map(f).collect()
}

// ---------------------------------------------------------------------------------------------
// Device-grid references.

/// X at `points` (reference coordinates, `[q][tdim]`) for every local cell of block `b`:
/// `[cell][q][g]`.
///
/// # Errors
///
/// Device errors of the reads.
pub fn points<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<Vec<f64>> {
    let nodes = BlockNodes::of_grid(g, b)?;
    Ok(nodes.points(&tabulate(&nodes, points, 0)))
}

/// J at `points` for every local cell of block `b`: `[cell][q][d · GD + g]`.
///
/// # Errors
///
/// Device errors of the reads.
pub fn jacobians<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<Vec<f64>> {
    let nodes = BlockNodes::of_grid(g, b)?;
    Ok(nodes.jacobians(&tabulate(&nodes, points, 1)))
}

/// det J (the pseudo-determinant when TD < GD) at `points`: `[cell][q]`.
///
/// # Errors
///
/// Device errors of the reads.
pub fn detj<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<Vec<f64>> {
    let (td, gd) = dims::<G>();
    let j = jacobians(g, b, points)?;
    Ok(per_point(&j, td, gd, |j| vec![detj_of(j, td, gd)]))
}

/// K at `points`: `[cell][q][g · TD + t]`.
///
/// # Errors
///
/// Device errors of the reads.
pub fn pseudo_inverses<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<Vec<f64>> {
    let (td, gd) = dims::<G>();
    let j = jacobians(g, b, points)?;
    Ok(per_point(&j, td, gd, |j| pinv_of(j, td, gd)))
}

/// Unit normals of a codimension-1 block at `points`: s (J0_y, −J0_x) / |J0| for curves in 2D,
/// s (J0 × J1) / |J0 × J1| for surfaces in 3D, with s = −1 where bit 0 of the cell's flags is
/// set: `[cell][q][g]`.
///
/// # Errors
///
/// Device errors of the reads.
///
/// # Panics
///
/// If the block is not of codimension 1.
pub fn normals<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<Vec<f64>> {
    let (td, gd) = dims::<G>();
    assert_eq!(td + 1, gd, "normals of a block of codimension {}", gd - td);
    let flags = g.cell_blocks()[b].flags.to_host()?;
    let j = jacobians(g, b, points)?;
    let nq = points.len() / td;
    let mut out = Vec::with_capacity(j.len() / td);
    for (i, j) in j.chunks_exact(td * gd).enumerate() {
        let v = match gd {
            2 => vec![j[1], -j[0]],
            _ => vec![
                j[1] * j[5] - j[2] * j[4],
                j[2] * j[3] - j[0] * j[5],
                j[0] * j[4] - j[1] * j[3],
            ],
        };
        let s = if flags[i / nq] & 1 == 1 { -1.0 } else { 1.0 };
        let len = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.extend(v.iter().map(|x| s * x / len));
    }
    Ok(out)
}

/// Outward unit facet normals Kᵀ n_ref / |Kᵀ n_ref| of a codimension-0 block at the points of
/// `fps` (facet-major): `[cell][q][g]`.
///
/// # Errors
///
/// Device errors of the reads.
pub fn facet_normals<G: DeviceGrid>(
    g: &G,
    b: usize,
    fps: &FacetPointSet<G::F, G::TD>,
) -> Result<Vec<f64>> {
    let (td, gd) = dims::<G>();
    let pts = fps.points().points();
    let nq = pts.len() / td;
    let k = pseudo_inverses(g, b, pts)?;
    let mut out = Vec::with_capacity(k.len() / td);
    for (i, k) in k.chunks_exact(td * gd).enumerate() {
        let n_ref = fps.reference_normal(i % nq);
        let m: Vec<f64> = (0..gd)
            .map(|g| (0..td).map(|t| k[g * td + t] * n_ref[t]).sum())
            .collect();
        let len = m.iter().map(|x| x * x).sum::<f64>().sqrt();
        out.extend(m.iter().map(|x| x / len));
    }
    Ok(out)
}

/// The reference points ξ of the physical points `x` (`[cell][q][g]`, `nq` per local cell of
/// block `b`) by a host Newton iteration in f64 (Gauss–Newton when TD < GD), from the centroid
/// of the reference cell, until a step is below 1e-15 or after 100 steps: `[cell][q][t]`.
///
/// Each step tabulates the element at one point: use it on small blocks.
///
/// # Errors
///
/// Device errors of the reads.
///
/// # Panics
///
/// If `x` does not hold `nq` points of every local cell.
pub fn pull_back<G: DeviceGrid>(g: &G, b: usize, x: &[f64], nq: usize) -> Result<Vec<f64>> {
    let nodes = BlockNodes::of_grid(g, b)?;
    let (td, gd) = (nodes.td(), nodes.gd);
    assert_eq!(
        x.len(),
        nodes.num_cells() * nq * gd,
        "pull_back: x has the wrong length"
    );
    let element = nodes.element();
    let verts = gridcl_refcell::vertices(nodes.cell);
    let centroid: Vec<f64> = (0..td)
        .map(|t| verts.iter().map(|v| v[t]).sum::<f64>() / verts.len() as f64)
        .collect();
    let mut out = Vec::with_capacity(nodes.num_cells() * nq * td);
    for (i, x) in x.chunks_exact(gd).enumerate() {
        let cx = nodes.cell_coords(i / nq);
        let mut xi = centroid.clone();
        for _ in 0..100 {
            let tab = element.tabulate(&xi, 1).expect("tabulate");
            let at = |row: usize, gg: usize| -> f64 {
                (0..nodes.k)
                    .map(|n| cx[n * gd + gg] * tab.get(row, 0, n))
                    .sum()
            };
            let r: Vec<f64> = (0..gd).map(|gg| x[gg] - at(0, gg)).collect();
            let j: Vec<f64> = (0..td * gd).map(|c| at(1 + c / gd, c % gd)).collect();
            let k = pinv_of(&j, td, gd);
            let step: Vec<f64> = (0..td)
                .map(|t| (0..gd).map(|gg| k[gg * td + t] * r[gg]).sum())
                .collect();
            xi.iter_mut().zip(&step).for_each(|(a, s)| *a += s);
            if step.iter().map(|s| s * s).sum::<f64>().sqrt() < 1e-15 {
                break;
            }
        }
        out.extend(xi);
    }
    Ok(out)
}

/// A device field read back in `[cell][point][component]` order and widened to f64, the layout
/// of the references above.
///
/// # Errors
///
/// Device errors of `to_host_cell_major`.
pub fn field_f64<F: Real, S: FieldShape>(field: &QuadField<F, S>) -> Result<Vec<f64>> {
    let host = field.to_host_cell_major()?;
    assert_eq!(size_of::<S::Host<F>>(), S::LEN * size_of::<F>());
    // SAFETY: the host value of every gridcl shape is an array, possibly nested, of exactly
    // S::LEN values of F (FieldShape's contract; checked by size above), so the vector's memory
    // is host.len() · LEN initialised values of F, aligned for F.
    let flat =
        unsafe { core::slice::from_raw_parts(host.as_ptr().cast::<F>(), host.len() * S::LEN) };
    Ok(flat.iter().map(|x| x.widen()).collect())
}

fn dims<G: DeviceGrid>() -> (usize, usize) {
    (<G::TD as gridcl::Dim>::N, <G::GD as gridcl::Dim>::N)
}

// ---------------------------------------------------------------------------------------------
// Host-mesh references.

/// X, J and det J of a [`HostMesh`] block in f64, with the block's node family.
pub mod host {
    use super::{BlockNodes, detj_of, per_point, tabulate};
    use crate::HostMesh;

    /// X at `points` (`[q][tdim]`) for every cell of block `b`: `[cell][q][g]`.
    pub fn points(h: &HostMesh, b: usize, points: &[f64]) -> Vec<f64> {
        let nodes = BlockNodes::of_host(h, b);
        nodes.points(&tabulate(&nodes, points, 0))
    }

    /// J at `points` for every cell of block `b`: `[cell][q][d · GD + g]`.
    pub fn jacobians(h: &HostMesh, b: usize, points: &[f64]) -> Vec<f64> {
        let nodes = BlockNodes::of_host(h, b);
        nodes.jacobians(&tabulate(&nodes, points, 1))
    }

    /// det J (the pseudo-determinant when TD < GD) at `points`: `[cell][q]`.
    pub fn detj(h: &HostMesh, b: usize, points: &[f64]) -> Vec<f64> {
        let td = gridcl_refcell::tdim(h.blocks[b].cell);
        let gd = h.gdim;
        per_point(&jacobians(h, b, points), td, gd, |j| {
            vec![detj_of(j, td, gd)]
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Amplification factors and tolerances (ADR-0042, item 2).

/// An amplification factor of a Lagrange element at a point x (ADR-0042, item 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Amplification {
    /// Σ_j |ψ_j(x)|, the Lebesgue function: for X.
    Values,
    /// max_d Σ_j |∂_d ψ_j(x)|: for J and the quantities derived from it.
    Derivatives,
    /// Σ_k |φ_k(x)| Σ_j |C_kj|, with φ the orthonormal set and C the expansion coefficients
    /// (`LagrangeElement::coefficients`): for a pull-back through coefficients.
    Coefficients,
}

/// The degree of the reference element A_ref of `cell`: 4, and 3 on pyramids, whose tests stop
/// at p = 3 (ADR-0042, Problem 2).
pub fn reference_degree(cell: CellType) -> u32 {
    if cell == CellType::Pyramid { 3 } else { 4 }
}

/// The amplification factor `kind` of the element (`cell`, `degree`, `family`) at every point
/// of `points` (`[q][tdim]`).
///
/// # Example
///
/// ```
/// use gridcl::NodeFamily;
/// use gridcl_dims::CellType;
/// use gridcl_testing::reference::{Amplification, amplification};
///
/// // P1 hat functions are non-negative and sum to one.
/// let a = amplification(CellType::Interval, 1, NodeFamily::Equispaced, &[0.3], Amplification::Values);
/// assert!((a[0] - 1.0).abs() < 1e-15);
/// ```
pub fn amplification(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    points: &[f64],
    kind: Amplification,
) -> Vec<f64> {
    let element = LagrangeElement::new(cell, degree, family).expect("element");
    let n = element.dim();
    let td = gridcl_refcell::tdim(cell);
    match kind {
        Amplification::Values => {
            let tab = element.tabulate(points, 0).expect("tabulate");
            (0..tab.num_points)
                .map(|q| (0..n).map(|j| tab.get(0, q, j).abs()).sum())
                .collect()
        }
        Amplification::Derivatives => {
            let tab = element.tabulate(points, 1).expect("tabulate");
            (0..tab.num_points)
                .map(|q| {
                    (0..td)
                        .map(|d| (0..n).map(|j| tab.get(1 + d, q, j).abs()).sum::<f64>())
                        .fold(0.0, f64::max)
                })
                .collect()
        }
        Amplification::Coefficients => {
            let data = coefficient_data(cell, degree, family);
            let tab = element.tabulate(points, 0).expect("tabulate");
            // φ_k(x) = Σ_j ψ_j(x) V_jk with V = C⁻¹ (the Lagrange interpolant of φ_k is φ_k).
            (0..tab.num_points)
                .map(|q| {
                    let psi = &tab.data[q * n..(q + 1) * n];
                    (0..n)
                        .map(|k| {
                            let phi: f64 = (0..n).map(|j| psi[j] * data.v[j * n + k]).sum();
                            phi.abs() * data.row_sums[k]
                        })
                        .sum()
                })
                .collect()
        }
    }
}

/// max(1, A / A_ref): A the largest factor `kind` of (`cell`, `degree`, `family`) over
/// `points`, and A_ref that of the equispaced element of [`reference_degree`] at the same
/// points.
///
/// # Example
///
/// ```
/// use gridcl::NodeFamily;
/// use gridcl_dims::CellType;
/// use gridcl_refcell::gauss_rule;
/// use gridcl_testing::reference::{Amplification, tolerance_scale};
///
/// let pts = gauss_rule(CellType::Interval, 4).points;
/// // At p ≤ 4 the scale is (about) 1.
/// let s = tolerance_scale(CellType::Interval, 2, NodeFamily::Equispaced, &pts, Amplification::Derivatives);
/// assert_eq!(s, 1.0);
/// ```
pub fn tolerance_scale(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    points: &[f64],
    kind: Amplification,
) -> f64 {
    let max = |v: Vec<f64>| v.into_iter().fold(0.0, f64::max);
    let a = max(amplification(cell, degree, family, points, kind));
    let a_ref = max(amplification(
        cell,
        reference_degree(cell),
        NodeFamily::Equispaced,
        points,
        kind,
    ));
    (a / a_ref).max(1.0)
}

/// The tolerance scale of a pull-back of block `b` from the images of `points`:
/// max(1, A / A_ref) with A = max over cells and points of κ(J) times the
/// [`Amplification::Coefficients`] factor, and A_ref the same with the reference element's
/// factor ([`condition_of`] gives κ).
///
/// # Errors
///
/// Device errors of the reads.
pub fn pull_back_scale<G: DeviceGrid>(g: &G, b: usize, points: &[f64]) -> Result<f64> {
    let nodes = BlockNodes::of_grid(g, b)?;
    let (td, gd) = (nodes.td(), nodes.gd);
    let j = nodes.jacobians(&tabulate(&nodes, points, 1));
    let kappa: Vec<f64> = j
        .chunks_exact(td * gd)
        .map(|j| condition_of(j, td, gd))
        .collect();
    let kind = Amplification::Coefficients;
    let a = amplification(nodes.cell, nodes.degree, nodes.family, points, kind);
    let a_ref = amplification(
        nodes.cell,
        reference_degree(nodes.cell),
        NodeFamily::Equispaced,
        points,
        kind,
    );
    let nq = a.len();
    let weighted = |f: &[f64]| {
        kappa
            .iter()
            .enumerate()
            .map(|(i, k)| k * f[i % nq])
            .fold(0.0, f64::max)
    };
    if kappa.is_empty() {
        return Ok(1.0);
    }
    Ok((weighted(&a) / weighted(&a_ref)).max(1.0))
}

/// The relative tolerance of `F` scaled by `scale`: `scale` · 1e-5 (f32) or `scale` · 1e-12
/// (f64).
pub fn tolerance<F: Real>(scale: f64) -> f64 {
    scale
        * if size_of::<F>() == 4 {
            TOL_F32
        } else {
            TOL_F64
        }
}

/// The relative error of a quantity over a block (ADR-0042, item 2): max |got − want| /
/// max |want| over every entry and component, or the absolute error if `want` is all zero.
///
/// Each quantity (X, J, det J, K, normals, ξ) is measured on its own, so det J ~ h³ stays
/// relative; the components of one quantity share its largest value, because rounding errors
/// scale with the quantity's size, not with each entry's (ADR-0042, item 2, "per component"
/// read per quantity by the maintainer, 2026-11-08).
///
/// # Example
///
/// ```
/// use gridcl_testing::reference::rel_err;
///
/// // A small off-diagonal entry is measured against the largest entry.
/// let want = [1.0, 1e-3, 1e-3, 2.0];
/// let got = [1.0, 1.1e-3, 1e-3, 2.0];
/// assert!((rel_err(&got, &want) - 5e-5).abs() < 1e-15);
/// assert_eq!(rel_err(&[1e-3], &[0.0]), 1e-3);
/// ```
///
/// # Panics
///
/// If the lengths differ.
pub fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "rel_err: lengths");
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    if scale > 0.0 { err / scale } else { err }
}

/// V = C⁻¹ (row-major) and the row sums Σ_j |C_kj| of an element, cached.
struct CoefficientData {
    v: Vec<f64>,
    row_sums: Vec<f64>,
}

fn coefficient_data(cell: CellType, degree: u32, family: NodeFamily) -> Arc<CoefficientData> {
    type Cache = Mutex<HashMap<(CellType, u32, NodeFamily), Arc<CoefficientData>>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some(d) = cache.lock().expect("cache").get(&(cell, degree, family)) {
        return d.clone();
    }
    let element = LagrangeElement::new(cell, degree, family).expect("element");
    let n = element.dim();
    let c = element.coefficients();
    let row_sums = (0..n)
        .map(|k| c[k * n..(k + 1) * n].iter().map(|x| x.abs()).sum())
        .collect();
    let data = Arc::new(CoefficientData {
        v: invert(c, n),
        row_sums,
    });
    cache
        .lock()
        .expect("cache")
        .insert((cell, degree, family), data.clone());
    data
}

/// The inverse of the row-major n × n matrix `a`, by Gauss–Jordan elimination with partial
/// pivoting.
fn invert(a: &[f64], n: usize) -> Vec<f64> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&x, &y| m[x * n + col].abs().total_cmp(&m[y * n + col].abs()))
            .expect("rows");
        assert!(m[pivot * n + col] != 0.0, "invert: singular matrix");
        if pivot != col {
            for j in 0..n {
                m.swap(col * n + j, pivot * n + j);
                inv.swap(col * n + j, pivot * n + j);
            }
        }
        let d = 1.0 / m[col * n + col];
        for j in 0..n {
            m[col * n + j] *= d;
            inv[col * n + j] *= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                m[r * n + j] -= f * m[col * n + j];
                inv[r * n + j] -= f * inv[col * n + j];
            }
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_refcell::gauss_rule;

    #[test]
    fn invert_gives_the_inverse() {
        let a = [4.0, 1.0, 2.0, 0.0, 3.0, 1.0, 1.0, 0.0, 2.0];
        let v = invert(&a, 3);
        for i in 0..3 {
            for j in 0..3 {
                let p: f64 = (0..3).map(|k| a[i * 3 + k] * v[k * 3 + j]).sum();
                assert!((p - if i == j { 1.0 } else { 0.0 }).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn pinv_is_a_left_inverse() {
        for (td, gd, j) in [
            (1, 1, vec![2.0]),
            (1, 3, vec![1.0, 2.0, 0.5]),
            (2, 2, vec![1.0, 0.2, -0.3, 2.0]),
            (2, 3, vec![1.0, 0.2, 0.1, -0.3, 2.0, 0.4]),
            (3, 3, vec![1.0, 0.2, 0.1, -0.3, 2.0, 0.4, 0.5, 0.1, 3.0]),
        ] {
            let k = pinv_of(&j, td, gd);
            for a in 0..td {
                for b in 0..td {
                    let kj: f64 = (0..gd).map(|g| k[g * td + a] * j[b * gd + g]).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (kj - want).abs() < 1e-14,
                        "({td}, {gd}): KJ[{a}][{b}] = {kj}"
                    );
                }
            }
            assert!(condition_of(&j, td, gd) >= 1.0 - 1e-14);
        }
    }

    /// ADR-0042, Problem 2: the absolute factors at hexahedron p = 8 (464, 19,019 and 6,910) and
    /// the ratios to p = 4 of a few rows (two significant digits), at the Gauss points of
    /// degree 2p.
    #[test]
    fn amplification_reproduces_the_adr_table() {
        use Amplification::*;
        use CellType::*;
        use NodeFamily::*;
        let max = |v: Vec<f64>| v.into_iter().fold(0.0, f64::max);
        let rows: &[(CellType, NodeFamily, u32, [f64; 3])] = if cfg!(debug_assertions) {
            &[
                (Interval, Equispaced, 8, [3.6, 8.8, 2.6]),
                (Quadrilateral, Gll, 8, [0.7, 1.4, 2.4]),
                (Triangle, Equispaced, 6, [2.3, 3.9, 2.2]),
            ]
        } else {
            &[
                (Interval, Equispaced, 8, [3.6, 8.8, 2.6]),
                (Quadrilateral, Gll, 8, [0.7, 1.4, 2.4]),
                (Triangle, Equispaced, 6, [2.3, 3.9, 2.2]),
                (Hexahedron, Equispaced, 8, [46.0, 112.0, 17.0]),
                (Prism, Equispaced, 7, [8.5, 18.0, 7.1]),
            ]
        };
        for &(cell, family, p, want) in rows {
            let pts = gauss_rule(cell, 2 * p).points;
            for (kind, want) in [Values, Derivatives, Coefficients].into_iter().zip(want) {
                let a = max(amplification(cell, p, family, &pts, kind));
                let a_ref = max(amplification(cell, 4, Equispaced, &pts, kind));
                let ratio = a / a_ref;
                assert!(
                    (ratio - want).abs() <= 0.051 * want.max(1.0),
                    "{cell:?} {family:?} p = {p} {kind:?}: {ratio} vs {want}"
                );
                assert_eq!(tolerance_scale(cell, p, family, &pts, kind), ratio.max(1.0));
                if cell == Hexahedron {
                    let abs = [464.0, 19_019.0, 6_910.0][kind as usize];
                    assert!((a - abs).abs() < 0.01 * abs, "{kind:?}: {a} vs {abs}");
                }
            }
        }
    }

    #[test]
    fn scales_are_one_at_low_degree() {
        for cell in [
            CellType::Triangle,
            CellType::Quadrilateral,
            CellType::Tetrahedron,
        ] {
            for p in 1..=4 {
                let pts = gauss_rule(cell, 2 * p).points;
                for kind in [Amplification::Values, Amplification::Derivatives] {
                    let s = tolerance_scale(cell, p, NodeFamily::Equispaced, &pts, kind);
                    assert!(s <= 1.01, "{cell:?} p = {p} {kind:?}: {s}");
                }
            }
        }
    }
}
