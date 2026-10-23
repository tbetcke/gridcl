//! Scalar L-vector reference kernels of the geometric factors (M3 acceptance, last bullet).
//!
//! One unit per cell reads the node L-vector through the dofmap, with no E-vector and no tiles,
//! and writes plain [cell][point][component] output. Each factor performs the operation
//! sequence the library's factor kernels perform in every lane, so results agree bitwise on the
//! CPU runtime:
//!
//! Every sum starts with its first term (not with 0, whose `0 + x` an optimizer may fold and
//! so flip the sign of a zero) and adds the others in ascending order:
//!
//! - X(g) = (Σ_k x_k(g) · φ_k) + origin(g);
//! - J(g, d) = Σ_k x_k(g) · ∂_d φ_k; component d · G + g;
//! - det J: J00 (1×1); J00·J11 − J01·J10 (2×2); cofactors along the first row (3×3);
//!   sqrt(a) with a = Σ_g J(g,0)² (TD = 1 < GD); sqrt(a·c − b·b) with a = J0·J0, b = J0·J1,
//!   c = J1·J1 (TD = 2, GD = 3); dot products over g;
//! - K (component g · T + t): the adjugate times inv = 1 / det in codimension 0; J(g,0) · (1/a)
//!   for TD = 1 < GD; Ginv · Jᵀ with Ginv = (c·inv, −b·inv; −b·inv, a·inv) and inv = 1/(a·c − b·b)
//!   for TD = 2, GD = 3;
//! - facet normals: m(g) = Σ_t K(t, g) · n_ref(t), |m| = sqrt(Σ_g m(g)²),
//!   n(g) = m(g) / |m|;
//! - normals: v = (J(1,0), −J(0,0)) for curves in 2D, v = J0 × J1 for surfaces, |v| as above,
//!   n(g) = (s · v(g)) / |v| with s = −1 where bit 0 of the cell's flags is set, else 1.

use cubecl::prelude::*;
use gridcl::prelude::*;
use gridcl::{FacetPointSet, PointSet};
use gridcl_device::{KernelKind, LaunchPlan};

/// The factor a reference kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RefFactor {
    Points,
    Jacobian,
    DetJ,
    Pinv,
    FacetNormals,
    Normals,
}

/// The comptime layout of a reference launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RefLayout {
    pub tdim: usize,
    pub gdim: usize,
    pub nodes_per_cell: usize,
    pub num_points: usize,
    pub factor: RefFactor,
}

impl RefLayout {
    /// Components per point.
    pub fn len(&self) -> usize {
        match self.factor {
            RefFactor::Points | RefFactor::FacetNormals | RefFactor::Normals => self.gdim,
            RefFactor::Jacobian | RefFactor::Pinv => self.gdim * self.tdim,
            RefFactor::DetJ => 1,
        }
    }
}

/// Σ_g J(g, a) · J(g, b) of the column-major `j`, starting with g = 0.
#[cube]
fn column_dot<F: Float>(
    j: &Array<F>,
    #[comptime] a: usize,
    #[comptime] b: usize,
    #[comptime] gd: usize,
) -> F {
    let mut acc = j[a * gd] * j[b * gd];
    #[unroll]
    for g in 1..gd {
        acc += j[a * gd + g] * j[b * gd + g];
    }
    acc
}

/// J of cell `c` at point `q`, column-major (component d · G + g).
#[cube]
fn jacobian_at<F: Float>(
    nodes: &[F],
    dofmap: &[u32],
    dphi: &[F],
    c: usize,
    q: usize,
    #[comptime] l: RefLayout,
) -> Array<F> {
    let mut j = Array::<F>::new(comptime!(l.tdim * l.gdim));
    #[unroll]
    for d in 0..l.tdim {
        #[unroll]
        for g in 0..l.gdim {
            let node0 = dofmap[c * l.nodes_per_cell] as usize;
            let mut acc = nodes[node0 * l.gdim + g] * dphi[q * l.nodes_per_cell * l.tdim + d];
            #[unroll]
            for k in 1..l.nodes_per_cell {
                let node = dofmap[c * l.nodes_per_cell + k] as usize;
                acc += nodes[node * l.gdim + g] * dphi[(q * l.nodes_per_cell + k) * l.tdim + d];
            }
            j[d * l.gdim + g] = acc;
        }
    }
    j
}

/// det J, or the pseudo-determinant, of the column-major `j`.
#[allow(unused_assignments)] // every comptime branch assigns `det`
#[cube]
fn detj_of<F: Float>(j: &Array<F>, #[comptime] l: RefLayout) -> F {
    let gd = l.gdim;
    let mut det = F::from_int(0);
    if comptime!(l.tdim == l.gdim) {
        if comptime!(l.tdim == 1) {
            det = j[0];
        } else if comptime!(l.tdim == 2) {
            det = j[0] * j[3] - j[2] * j[1];
        } else {
            // m(g, d) = j[d * 3 + g]
            det = j[0] * (j[4] * j[8] - j[7] * j[5]) - j[3] * (j[1] * j[8] - j[7] * j[2])
                + j[6] * (j[1] * j[5] - j[4] * j[2]);
        }
    } else if comptime!(l.tdim == 1) {
        det = F::sqrt(column_dot::<F>(j, 0usize, 0usize, gd));
    } else {
        let a = column_dot::<F>(j, 0usize, 0usize, gd);
        let b = column_dot::<F>(j, 0usize, 1usize, gd);
        let c = column_dot::<F>(j, 1usize, 1usize, gd);
        det = F::sqrt(a * c - b * b);
    }
    det
}

/// K = (JᵀJ)⁻¹Jᵀ of the column-major `j`, component g · T + t.
#[cube]
fn pinv_of<F: Float>(j: &Array<F>, #[comptime] l: RefLayout) -> Array<F> {
    let gd = l.gdim;
    let td = l.tdim;
    let mut k = Array::<F>::new(comptime!(l.tdim * l.gdim));
    if comptime!(l.tdim == l.gdim) {
        if comptime!(l.tdim == 1) {
            k[0] = F::from_int(1) / j[0];
        } else if comptime!(l.tdim == 2) {
            let inv = F::from_int(1) / (j[0] * j[3] - j[2] * j[1]);
            // K(t, g) at g * 2 + t; J(g, d) at d * 2 + g.
            k[0] = j[3] * inv;
            k[2] = -j[2] * inv;
            k[1] = -j[1] * inv;
            k[3] = j[0] * inv;
        } else {
            let det = j[0] * (j[4] * j[8] - j[7] * j[5]) - j[3] * (j[1] * j[8] - j[7] * j[2])
                + j[6] * (j[1] * j[5] - j[4] * j[2]);
            let inv = F::from_int(1) / det;
            // m(g, d) = j[d * 3 + g]; K(t, g) at g * 3 + t.
            k[0] = (j[4] * j[8] - j[7] * j[5]) * inv; // K00 = m11 m22 - m12 m21
            k[3] = (j[6] * j[5] - j[3] * j[8]) * inv; // K01 = m02 m21 - m01 m22
            k[6] = (j[3] * j[7] - j[6] * j[4]) * inv; // K02 = m01 m12 - m02 m11
            k[1] = (j[7] * j[2] - j[1] * j[8]) * inv; // K10 = m12 m20 - m10 m22
            k[4] = (j[0] * j[8] - j[6] * j[2]) * inv; // K11 = m00 m22 - m02 m20
            k[7] = (j[6] * j[1] - j[0] * j[7]) * inv; // K12 = m02 m10 - m00 m12
            k[2] = (j[1] * j[5] - j[4] * j[2]) * inv; // K20 = m10 m21 - m11 m20
            k[5] = (j[3] * j[2] - j[0] * j[5]) * inv; // K21 = m01 m20 - m00 m21
            k[8] = (j[0] * j[4] - j[3] * j[1]) * inv; // K22 = m00 m11 - m01 m10
        }
    } else if comptime!(l.tdim == 1) {
        let inv = F::from_int(1) / column_dot::<F>(j, 0usize, 0usize, gd);
        #[unroll]
        for g in 0..gd {
            k[g] = j[g] * inv;
        }
    } else {
        let a = column_dot::<F>(j, 0usize, 0usize, gd);
        let b = column_dot::<F>(j, 0usize, 1usize, gd);
        let c = column_dot::<F>(j, 1usize, 1usize, gd);
        let inv = F::from_int(1) / (a * c - b * b);
        let g00 = c * inv;
        let g01 = -b * inv;
        let g11 = a * inv;
        #[unroll]
        for g in 0..gd {
            k[g * td] = g00 * j[g] + g01 * j[gd + g];
            k[g * td + 1] = g01 * j[g] + g11 * j[gd + g];
        }
    }
    k
}

/// The reference kernel: one unit per cell, `items_per_unit` cells per unit.
///   nodes  L-vector [node][gdim]
///   dofmap [cell][k]
///   tab    φ [q][k] (Points) or ∂φ [q][k][d] (all others)
///   extra  origin [gdim] (Points), n_ref [q][tdim] (FacetNormals), unused otherwise
///   flags  cell flags (Normals), unused otherwise
///   out    [cell][q][len]
#[cube(launch_unchecked)]
fn reference_kernel<F: Float>(
    nodes: &[F],
    dofmap: &[u32],
    tab: &[F],
    extra: &[F],
    flags: &[u32],
    out: &mut [F],
    num_cells: u32,
    items_per_unit: u32,
    #[comptime] l: RefLayout,
) {
    let gd = l.gdim;
    let td = l.tdim;
    let len = comptime!(l.len());
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let c = first + i;
        if c < num_cells as usize {
            for q in 0..l.num_points {
                let o = (c * l.num_points + q) * len;
                match comptime!(l.factor) {
                    RefFactor::Points =>
                    {
                        #[unroll]
                        for g in 0..gd {
                            let node0 = dofmap[c * l.nodes_per_cell] as usize;
                            let mut acc = nodes[node0 * gd + g] * tab[q * l.nodes_per_cell];
                            #[unroll]
                            for k in 1..l.nodes_per_cell {
                                let node = dofmap[c * l.nodes_per_cell + k] as usize;
                                acc += nodes[node * gd + g] * tab[q * l.nodes_per_cell + k];
                            }
                            out[o + g] = acc + extra[g];
                        }
                    }
                    RefFactor::Jacobian => {
                        let j = jacobian_at::<F>(nodes, dofmap, tab, c, q, l);
                        #[unroll]
                        for e in 0..comptime!(l.tdim * l.gdim) {
                            out[o + e] = j[e];
                        }
                    }
                    RefFactor::DetJ => {
                        let j = jacobian_at::<F>(nodes, dofmap, tab, c, q, l);
                        out[o] = detj_of::<F>(&j, l);
                    }
                    RefFactor::Pinv => {
                        let j = jacobian_at::<F>(nodes, dofmap, tab, c, q, l);
                        let k = pinv_of::<F>(&j, l);
                        #[unroll]
                        for e in 0..comptime!(l.tdim * l.gdim) {
                            out[o + e] = k[e];
                        }
                    }
                    RefFactor::FacetNormals => {
                        let j = jacobian_at::<F>(nodes, dofmap, tab, c, q, l);
                        let k = pinv_of::<F>(&j, l);
                        let mut m = Array::<F>::new(comptime!(l.gdim));
                        #[unroll]
                        for g in 0..gd {
                            let mut acc = k[g * td] * extra[q * td];
                            #[unroll]
                            for t in 1..td {
                                acc += k[g * td + t] * extra[q * td + t];
                            }
                            m[g] = acc;
                        }
                        let mut s2 = m[0] * m[0];
                        #[unroll]
                        for g in 1..gd {
                            s2 += m[g] * m[g];
                        }
                        let norm = F::sqrt(s2);
                        #[unroll]
                        for g in 0..gd {
                            out[o + g] = m[g] / norm;
                        }
                    }
                    RefFactor::Normals => {
                        let j = jacobian_at::<F>(nodes, dofmap, tab, c, q, l);
                        let mut v = Array::<F>::new(comptime!(l.gdim));
                        if comptime!(l.gdim == 2) {
                            v[0] = j[1];
                            v[1] = -j[0];
                        } else {
                            v[0] = j[1] * j[5] - j[2] * j[4];
                            v[1] = j[2] * j[3] - j[0] * j[5];
                            v[2] = j[0] * j[4] - j[1] * j[3];
                        }
                        let mut s2 = v[0] * v[0];
                        #[unroll]
                        for g in 1..gd {
                            s2 += v[g] * v[g];
                        }
                        let norm = F::sqrt(s2);
                        let flip = (flags[c] & 1u32) == 1u32;
                        let s = select(flip, F::from_int(-1), F::from_int(1));
                        #[unroll]
                        for g in 0..gd {
                            out[o + g] = (s * v[g]) / norm;
                        }
                    }
                }
            }
        }
    }
}

/// Runs the reference kernel for `factor` on block `b` of `g` with the point set `pts`;
/// `extra` holds the origin (Points) or the reference normals [q][tdim] (FacetNormals).
/// Returns [cell][q][len].
fn run<G: DeviceGrid>(
    g: &G,
    b: usize,
    pts: &PointSet<G::F, G::TD>,
    factor: RefFactor,
    extra: &[f64],
) -> Vec<G::F> {
    let client = g.client();
    let gb = g.geometry_block(b);
    let cell: CellType = gb.element.cell.into();
    let l = RefLayout {
        tdim: <G::TD as Dim>::N,
        gdim: <G::GD as Dim>::N,
        nodes_per_cell: gridcl_refcell::num_lagrange_nodes(cell, gb.element.degree),
        num_points: pts.len(),
        factor,
    };
    let num_cells = g.cell_blocks()[b].index_map.num_local();
    if num_cells == 0 {
        return Vec::new();
    }
    let tab = match factor {
        RefFactor::Points => pts.tabulate(&gb.element),
        _ => pts.tabulate_derivatives(&gb.element),
    }
    .expect("tabulate");
    let extra: Vec<G::F> = if extra.is_empty() {
        vec![<G::F as Real>::narrow(0.0)]
    } else {
        extra.iter().map(|&x| <G::F as Real>::narrow(x)).collect()
    };
    let extra = DeviceBuffer::from_host(client, &extra).expect("upload");
    let flags = &g.cell_blocks()[b].flags;
    let out =
        DeviceBuffer::<G::F>::uninit(client, num_cells * l.num_points * l.len()).expect("uninit");
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::GeomFactors,
        num_cells,
        size_of::<G::F>(),
        1,
        1,
    )
    .expect("plan");
    gridcl_device::debug_assert_bindings(client, 6);
    // SAFETY: every buffer is bound with its logical length; the kernel reads cells below
    // num_cells only: dofmap rows c · K + k < num_cells · K, nodes at dofmap entries times gdim
    // (the dofmap indexes the L-vector), tab entries below num_points · K · tdim, extra below
    // gdim or num_points · tdim, flags below num_cells; it writes out below
    // num_cells · num_points · len, the length of `out`.
    unsafe {
        reference_kernel::launch_unchecked::<G::F>(
            client,
            plan.cube_count,
            plan.cube_dim,
            g.nodes().arg(),
            gb.dofmap.data.arg(),
            tab.arg(),
            extra.arg(),
            flags.arg(),
            out.arg(),
            num_cells as u32,
            plan.items_per_unit,
            l,
        );
    }
    out.to_host().expect("read")
}

/// The reference physical points of block `b`.
pub fn points<G: DeviceGrid>(g: &G, b: usize, pts: &PointSet<G::F, G::TD>) -> Vec<G::F> {
    let origin = g.origin();
    run(g, b, pts, RefFactor::Points, origin.as_ref())
}

/// The reference factor `factor` (Jacobian, DetJ, Pinv or Normals) of block `b`.
pub fn factor<G: DeviceGrid>(
    g: &G,
    b: usize,
    pts: &PointSet<G::F, G::TD>,
    factor: RefFactor,
) -> Vec<G::F> {
    assert!(!matches!(
        factor,
        RefFactor::Points | RefFactor::FacetNormals
    ));
    run(g, b, pts, factor, &[])
}

/// The reference facet normals of block `b`.
pub fn facet_normals<G: DeviceGrid>(
    g: &G,
    b: usize,
    fps: &FacetPointSet<G::F, G::TD>,
) -> Vec<G::F> {
    let n = fps.points().len();
    let nref: Vec<f64> = (0..n)
        .flat_map(|q| fps.reference_normal(q).to_vec())
        .collect();
    run(g, b, fps.points(), RefFactor::FacetNormals, &nref)
}
