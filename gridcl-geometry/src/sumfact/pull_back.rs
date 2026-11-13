//! The tensor pull-back (docs/plans/M9h.md, T8; ADR-0042, items 1 and 5; ADR-0049).
//!
//! On quadrilaterals and hexahedra C' is the tensor product of the interval's C₁′ under the
//! lexicographic node permutation (Finding 3 of the plan), so the coordinate map's coefficients
//! over the tensor Legendre basis, a = C'x, are formed by sum factorisation: a set-up kernel
//! ([`pull_back_sumfact_setup_kernel`]) contracts the E-vector with C₁′ one direction at a time,
//! 3 (p + 1)⁴ products per component of a hexahedron against (p + 1)⁶ of `loops`' set-up. Each
//! Newton step ([`pull_back_sumfact_kernel`]) evaluates the 1D Legendre polynomials and their
//! derivatives at each coordinate of ξ by the recurrences of the `loops` evaluator and contracts
//! a one direction at a time, about 2 (p + 1)³ products per component. On prisms C' is
//! C'_tri ⊗ C₁′: the set-up contracts the triangle with C'_tri and z with C₁′, and the step
//! combines the Dubiner set with Legendre in z.
//!
//! The coefficients are those of `loops` and `direct`, evaluated in the same basis, so the
//! conditioning outside the reference cell is theirs; contracting the nodal values with the
//! Lagrange bases instead amplifies f32 rounding by the Lebesgue function of each direction,
//! about 296 at ADR-0046's box margin for equispaced p = 8 (ADR-0049).
//!
//! The kernels read one buffer per geometry block ([`pull_back_data`]), uploaded on first use
//! and cached with the E-vector (ADR-0042, item 5): the runtime zero and one, the origin, C₁′,
//! the lexicographic-to-gridcl node permutation and, on prisms, C'_tri. The node count per
//! direction and the point count are run-time arguments, and only the capacity of the local
//! arrays is comptime (`loops_capacity`: 4, 8 or 16), so one kernel serves several degrees and no
//! loop has a constant trip count for the compiler to unroll: with the node count at comptime,
//! CUDA's first use of the hexahedron set-up and step took 820 s at p = 8 (6 s at p = 4), against
//! under 0.4 s for `loops` (Locust, docs/plans/M9h.md, T8). Every sum starts from the runtime zero
//! and runs in ascending order, and every lane performs the same operations, so the results are
//! bitwise equal across vector sizes on the CPU runtime.

use cubecl::prelude::*;
use gridcl_device::Result;
use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, NodeFamily};

use super::tensor::{bc, int};
use super::{permutation, prism_permutation};
use crate::GeomLayout;
use crate::basis_eval::{device_coefficients, dubiner_into, legendre_into};
use crate::names::element_error;
use crate::pull_back::{centroid, newton_update};

/// The data of the tensor pull-back of the element (cell, degree, family), a quadrilateral,
/// hexahedron or prism, in f64:
/// - `[0]` and `[1]` the runtime zero and one, then the grid's `origin` (gdim entries);
/// - C₁′ of the interval element of the same degree and family, row-major n × n with
///   n = p + 1 (`basis_eval::device_coefficients` without its zero and one): the Lagrange
///   function of 1D node a is Σ_k C₁′[k n + a] P_k(2t − 1);
/// - the permutation: the gridcl node of each lexicographic multi-index, entry
///   a_0 + n a_1 + n² a_2 on quadrilaterals and hexahedra (`sumfact::permutation`), t + nt c on
///   prisms (`sumfact::prism_permutation`, nt = n (n + 1) / 2 triangle nodes);
/// - on prisms, C'_tri of the triangle element, row-major nt × nt.
///
/// # Errors
///
/// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element gridcl-refcell does not offer.
///
/// # Panics
///
/// If the cell is not a quadrilateral, hexahedron or prism.
pub(crate) fn pull_back_data(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    origin: &[f64],
) -> Result<Vec<f64>> {
    let line = LagrangeElement::new(CellType::Interval, degree, family).map_err(element_error)?;
    let n = line.dim();
    let c1 = device_coefficients(CellType::Interval, degree, family)?;
    let mut out: Vec<f64> = [0.0, 1.0]
        .into_iter()
        .chain(origin.iter().copied())
        .chain(c1[..n * n].iter().copied())
        .collect();
    match cell {
        CellType::Quadrilateral | CellType::Hexahedron => {
            out.extend(
                permutation(cell, degree, family, &line)?
                    .into_iter()
                    .map(|k| k as f64),
            );
        }
        CellType::Prism => {
            let triangle =
                LagrangeElement::new(CellType::Triangle, degree, family).map_err(element_error)?;
            out.extend(
                prism_permutation(degree, family, &triangle, &line)?
                    .into_iter()
                    .map(|k| k as f64),
            );
            let nt = triangle.dim();
            let c = device_coefficients(CellType::Triangle, degree, family)?;
            out.extend_from_slice(&c[..nt * nt]);
        }
        _ => panic!("tensor pull-back on {cell:?}"),
    }
    Ok(out)
}

/// The coefficient `data[at]` in every lane.
#[cube]
fn coef<F: Float, N: Size>(data: &[F], at: usize) -> Vector<F, N> {
    bc::<F, N>(data[at])
}

/// `a[at] = w · t` when `first`, else `a[at] += w · t`: the accumulation over the outermost
/// direction of the set-up, in ascending order.
#[cube]
fn accumulate<F: Float, N: Size>(
    a: &mut [Vector<F, N>],
    at: usize,
    w: Vector<F, N>,
    t: Vector<F, N>,
    first: bool,
) {
    if first {
        a[at] = w * t;
    } else {
        a[at] += w * t;
    }
}

/// The set-up of the tensor pull-back: the coordinate map's coefficients a = C'x of the cell
/// vectors `v < num_vectors` of one range, by sum factorisation (module docs), into `a` as
/// `[k][g][cell]` with row stride `a_stride` vectors, k in the order of `basis_eval::expansion`
/// (k_0 + n k_1 + n² k_2; prisms k_tri + nt k_z):
///   xe    E-vector chunk [k][g][cell], row (k, g) at (k·G + g)·xe_stride + xe_offset + v
///   data  [`pull_back_data`] narrowed to `F`
/// The components run one at a time, and per plane of the outermost direction the two inner
/// contractions keep 2 n² (prisms nt) values per lane; the outermost one accumulates into `a`.
/// `nodes` is n, the nodes per direction, and `l` is `sumfact_layout`'s with the capacity of
/// the local arrays (at least n) as its node count. Three buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pull_back_sumfact_setup_kernel<F: Float, N: Size>(
    xe: &[Vector<F, N>],
    data: &[F],
    a: &mut [Vector<F, N>],
    xe_offset: u32,
    xe_stride: u32,
    a_stride: u32,
    num_vectors: u32,
    items_per_unit: u32,
    nodes: u32,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
) {
    let n = nodes as usize;
    let gd = comptime!(l.gdim);
    let c_at = comptime!(2 + l.gdim);
    let perm_at = c_at + n * n;
    let zero = bc::<F, N>(data[0]);
    let xs = xe_stride as usize;
    let st = a_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            let xo = xe_offset as usize + v;
            if comptime!(cell == CellType::Prism) {
                let nt = n * (n + 1) / 2;
                let ctri_at = perm_at + nt * n;
                // T(k) = Σ_t C'_tri(k, t) x(π(t, c), g) of one z node c.
                let mut t1 = Array::<Vector<F, N>>::new(comptime!(
                    l.nodes_per_cell * (l.nodes_per_cell + 1) / 2
                ));
                for g in 0..gd {
                    for c in 0..n {
                        for k in 0..nt {
                            let mut s = zero;
                            for t in 0..nt {
                                let node = int::<F>(data, perm_at + t + nt * c);
                                s += coef::<F, N>(data, ctri_at + k * nt + t)
                                    * xe[(node * gd + g) * xs + xo];
                            }
                            t1[k] = s;
                        }
                        for kz in 0..n {
                            let w = coef::<F, N>(data, c_at + kz * n + c);
                            for k in 0..nt {
                                let at = ((k + nt * kz) * gd + g) * st + v;
                                accumulate::<F, N>(a, at, w, t1[k], c == 0);
                            }
                        }
                    }
                }
            } else if comptime!(l.tdim == 2) {
                // T(k_0) = Σ_a C₁′(k_0, a) x(π(a, b), g) of one node b along direction 1.
                let mut t1 = Array::<Vector<F, N>>::new(comptime!(l.nodes_per_cell));
                for g in 0..gd {
                    for b in 0..n {
                        for k0 in 0..n {
                            let mut s = zero;
                            for i in 0..n {
                                let node = int::<F>(data, perm_at + i + n * b);
                                s += coef::<F, N>(data, c_at + k0 * n + i)
                                    * xe[(node * gd + g) * xs + xo];
                            }
                            t1[k0] = s;
                        }
                        for k1 in 0..n {
                            let w = coef::<F, N>(data, c_at + k1 * n + b);
                            for k0 in 0..n {
                                let at = ((k0 + n * k1) * gd + g) * st + v;
                                accumulate::<F, N>(a, at, w, t1[k0], b == 0);
                            }
                        }
                    }
                }
            } else {
                // Of one node c along direction 2: T1(k_0, b) at b n + k_0 and T2(k_0, k_1) at
                // k_1 n + k_0.
                let mut t1 =
                    Array::<Vector<F, N>>::new(comptime!(l.nodes_per_cell * l.nodes_per_cell));
                let mut t2 =
                    Array::<Vector<F, N>>::new(comptime!(l.nodes_per_cell * l.nodes_per_cell));
                for g in 0..gd {
                    for c in 0..n {
                        for b in 0..n {
                            for k0 in 0..n {
                                let mut s = zero;
                                for i in 0..n {
                                    let node = int::<F>(data, perm_at + i + n * b + n * n * c);
                                    s += coef::<F, N>(data, c_at + k0 * n + i)
                                        * xe[(node * gd + g) * xs + xo];
                                }
                                t1[b * n + k0] = s;
                            }
                        }
                        for k1 in 0..n {
                            for k0 in 0..n {
                                let mut s = zero;
                                for b in 0..n {
                                    s += coef::<F, N>(data, c_at + k1 * n + b) * t1[b * n + k0];
                                }
                                t2[k1 * n + k0] = s;
                            }
                        }
                        for k2 in 0..n {
                            let w = coef::<F, N>(data, c_at + k2 * n + c);
                            for k1 in 0..n {
                                for k0 in 0..n {
                                    let at = ((k0 + n * k1 + n * n * k2) * gd + g) * st + v;
                                    accumulate::<F, N>(a, at, w, t2[k1 * n + k0], c == 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The coordinate map at ξ in the layout of `basis_eval::expansion` (X_g at g (1 + TD), ∂X_g/∂ξ_d
/// at g (1 + TD) + 1 + d), relative to the origin, from the coefficients of the set-up (row
/// (k, g) of the lanes' cells at (k · G + g) · `st` + `v`): Σ_k a_k φ̃_k(ξ) contracted one
/// direction at a time.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tensor_map<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &[Vector<F, N>],
    v: usize,
    st: usize,
    n: usize,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
) -> Array<Vector<F, N>> {
    let gd = comptime!(l.gdim);
    let td = comptime!(l.tdim);
    let mut e = Array::<Vector<F, N>>::new(comptime!(l.gdim * (l.tdim + 1)));
    #[unroll]
    for c in 0..gd * (td + 1) {
        e[c] = zero;
    }
    // P_k(2t − 1) at 2k and its derivative along t at 2k + 1, per direction.
    let mut lz = Array::<Vector<F, N>>::new(comptime!(2 * l.nodes_per_cell));
    legendre_into::<F, N>(&mut lz, xi[td - 1], one, zero, n - 1);
    // The sums over the inner directions with values (v) and derivatives along ξ_0 (x) and ξ_1
    // (y) of the inner factor.
    let mut yv = Array::<Vector<F, N>>::new(comptime!(l.gdim));
    let mut yx = Array::<Vector<F, N>>::new(comptime!(l.gdim));
    let mut yy = Array::<Vector<F, N>>::new(comptime!(l.gdim));
    if comptime!(cell == CellType::Prism) {
        let nt = n * (n + 1) / 2;
        let mut dub = Array::<Vector<F, N>>::new(comptime!(
            3 * l.nodes_per_cell * (l.nodes_per_cell + 1) / 2
        ));
        dubiner_into::<F, N>(&mut dub, xi[0], xi[1], one, zero, n - 1);
        for kz in 0..n {
            #[unroll]
            for g in 0..gd {
                yv[g] = zero;
                yx[g] = zero;
                yy[g] = zero;
            }
            for k in 0..nt {
                let (d0, d1, d2) = (dub[3 * k], dub[3 * k + 1], dub[3 * k + 2]);
                #[unroll]
                for g in 0..gd {
                    let w = a[((k + nt * kz) * gd + g) * st + v];
                    yv[g] += d0 * w;
                    yx[g] += d1 * w;
                    yy[g] += d2 * w;
                }
            }
            let (p, dp) = (lz[2 * kz], lz[2 * kz + 1]);
            #[unroll]
            for g in 0..gd {
                e[4 * g] += p * yv[g];
                e[4 * g + 1] += p * yx[g];
                e[4 * g + 2] += p * yy[g];
                e[4 * g + 3] += dp * yv[g];
            }
        }
    } else {
        let mut lx = Array::<Vector<F, N>>::new(comptime!(2 * l.nodes_per_cell));
        legendre_into::<F, N>(&mut lx, xi[0], one, zero, n - 1);
        // B(g): the sums over direction 0 of one line, with P (v) and P' (x).
        let mut bv = Array::<Vector<F, N>>::new(comptime!(l.gdim));
        let mut bx = Array::<Vector<F, N>>::new(comptime!(l.gdim));
        if comptime!(td == 2) {
            for k1 in 0..n {
                #[unroll]
                for g in 0..gd {
                    bv[g] = zero;
                    bx[g] = zero;
                }
                for k0 in 0..n {
                    let (p, dp) = (lx[2 * k0], lx[2 * k0 + 1]);
                    #[unroll]
                    for g in 0..gd {
                        let w = a[((k0 + n * k1) * gd + g) * st + v];
                        bv[g] += p * w;
                        bx[g] += dp * w;
                    }
                }
                let (p, dp) = (lz[2 * k1], lz[2 * k1 + 1]);
                #[unroll]
                for g in 0..gd {
                    e[3 * g] += p * bv[g];
                    e[3 * g + 1] += p * bx[g];
                    e[3 * g + 2] += dp * bv[g];
                }
            }
        } else {
            let mut ly = Array::<Vector<F, N>>::new(comptime!(2 * l.nodes_per_cell));
            legendre_into::<F, N>(&mut ly, xi[1], one, zero, n - 1);
            for k2 in 0..n {
                #[unroll]
                for g in 0..gd {
                    yv[g] = zero;
                    yx[g] = zero;
                    yy[g] = zero;
                }
                for k1 in 0..n {
                    #[unroll]
                    for g in 0..gd {
                        bv[g] = zero;
                        bx[g] = zero;
                    }
                    for k0 in 0..n {
                        let (p, dp) = (lx[2 * k0], lx[2 * k0 + 1]);
                        #[unroll]
                        for g in 0..gd {
                            let w = a[((k0 + n * k1 + n * n * k2) * gd + g) * st + v];
                            bv[g] += p * w;
                            bx[g] += dp * w;
                        }
                    }
                    let (p, dp) = (ly[2 * k1], ly[2 * k1 + 1]);
                    #[unroll]
                    for g in 0..gd {
                        yv[g] += p * bv[g];
                        yx[g] += p * bx[g];
                        yy[g] += dp * bv[g];
                    }
                }
                let (p, dp) = (lz[2 * k2], lz[2 * k2 + 1]);
                #[unroll]
                for g in 0..gd {
                    e[4 * g] += p * yv[g];
                    e[4 * g + 1] += p * yx[g];
                    e[4 * g + 2] += p * yy[g];
                    e[4 * g + 3] += dp * yv[g];
                }
            }
        }
    }
    e
}

/// The Newton iterations of the tensor pull-back: as `pull_back::pull_back_loops_kernel`, with
/// the coefficients of [`pull_back_sumfact_setup_kernel`] (`a`, range-local cell vectors, row
/// stride `a_stride` vectors) evaluated by [`tensor_map`], `data` of [`pull_back_data`] for the
/// runtime zero and one and the origin, `num_points` and `nodes` (n) at run time and `l` as for
/// the set-up.
/// Five buffers.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pull_back_sumfact_kernel<F: Float, N: Size>(
    x: &[Vector<F, N>],
    a: &[Vector<F, N>],
    data: &[F],
    xi: &mut [Vector<F, N>],
    step: &mut [Vector<F, N>],
    x_first_tile: u32,
    a_stride: u32,
    xi_first_tile: u32,
    step_first_tile: u32,
    max_iterations: u32,
    tolerance_hi: f32,
    tolerance_lo: f32,
    num_vectors: u32,
    items_per_unit: u32,
    num_points: u32,
    nodes: u32,
    #[comptime] l: GeomLayout,
    #[comptime] cell: CellType,
) {
    let td = l.tdim;
    let n = nodes as usize;
    let gd = l.gdim;
    let nq = num_points as usize;
    let vpt = comptime!(l.tile) / N::value();
    // The runtime zero and one of `basis_eval`.
    let zero = bc::<F, N>(data[0]);
    let one = bc::<F, N>(data[1]);
    let tol = bc::<F, N>(F::cast_from(tolerance_hi) + F::cast_from(tolerance_lo));
    let big = bc::<F, N>(F::max_value());
    let no = Vector::<u32, N>::broadcast(0u32);
    let st = a_stride as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            // v − (v / vpt) · vpt, not v % vpt (CubeCL folds `x % 1` on the CPU runtime).
            let local_tile = v / vpt;
            let lane = v - local_tile * vpt;
            for q in 0..nq {
                // The point relative to the origin.
                let at = ((x_first_tile as usize + local_tile) * nq + q) * gd;
                let mut y = Array::<Vector<F, N>>::new(comptime!(gd));
                #[unroll]
                for g in 0..gd {
                    y[g] = x[(at + g) * vpt + lane] - bc::<F, N>(data[2 + g]);
                }
                // The centroid, times the runtime one (no operation on constants alone).
                let mut p = Array::<Vector<F, N>>::new(comptime!(td));
                #[unroll]
                for d in 0..td {
                    p[d] = bc::<F, N>(F::new(comptime!(centroid(cell, d)))) * one;
                }
                let mut last = big;
                let mut frozen = no;
                for _ in 0..max_iterations {
                    let e = tensor_map::<F, N>(&p, a, v, st, n, one, zero, l, cell);
                    newton_update::<F, N>(&mut p, &e, &y, &mut last, &mut frozen, tol, l);
                }
                let at = ((xi_first_tile as usize + local_tile) * nq + q) * td;
                #[unroll]
                for d in 0..td {
                    xi[(at + d) * vpt + lane] = p[d];
                }
                let so = (step_first_tile as usize + local_tile) * nq + q;
                step[so * vpt + lane] = last;
            }
        }
    }
}

/// The vectors the set-up kernel keeps live for local arrays of capacity n per direction on
/// `cell`.
pub(crate) fn sumfact_setup_live(cell: CellType, n: usize, td: usize) -> usize {
    let inner = match (cell, td) {
        (CellType::Prism, _) => n * (n + 1) / 2,
        (_, 2) => n,
        _ => 2 * n * n,
    };
    inner + 4
}

/// The vectors the step kernel keeps live for local arrays of capacity n per direction on `cell`:
/// the 1D Legendre values (and the Dubiner set on prisms), the sums, the map and the Newton step
/// of `pull_back`.
pub(crate) fn sumfact_pull_back_live(cell: CellType, n: usize, td: usize, gd: usize) -> usize {
    let bases = if cell == CellType::Prism {
        2 * n + 3 * n * (n + 1) / 2
    } else {
        2 * td * n
    };
    bases + 5 * gd + gd * (td + 1) + 3 * td * gd + 3 * gd + 3 * td + 6
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tensor product of C₁′ (and on prisms C'_tri ⊗ C₁′), permuted, is the cell's C' of
    /// `device_coefficients`: the set-up's coefficients are those of `loops` and `direct`.
    #[test]
    fn factors_reproduce_the_coefficients() {
        let origin = [0.25, -1.5, 3.0];
        for (cell, family, degrees) in [
            (CellType::Quadrilateral, NodeFamily::Equispaced, 1..=8),
            (CellType::Quadrilateral, NodeFamily::Gll, 1..=8),
            (CellType::Hexahedron, NodeFamily::Equispaced, 1..=4),
            (CellType::Hexahedron, NodeFamily::Gll, 1..=4),
            (CellType::Prism, NodeFamily::Equispaced, 1..=4),
        ] {
            let td = gridcl_refcell::tdim(cell);
            for degree in degrees {
                let data = pull_back_data(cell, degree, family, &origin[..td]).unwrap();
                assert_eq!((data[0], data[1]), (0.0, 1.0));
                assert_eq!(&data[2..2 + td], &origin[..td]);
                let want = device_coefficients(cell, degree, family).unwrap();
                let nb = gridcl_refcell::num_lagrange_nodes(cell, degree);
                let n = degree as usize + 1;
                let c1 = &data[2 + td..2 + td + n * n];
                let perm = &data[2 + td + n * n..];
                let scale = want[..nb * nb].iter().fold(0.0f64, |m, x| m.max(x.abs()));
                let mut seen = vec![false; nb];
                let mut check = |k: usize, lex: usize, got: f64| {
                    let j = perm[lex] as usize;
                    seen[j] = true;
                    let w = want[k * nb + j];
                    assert!(
                        (got - w).abs() <= 1e-12 * scale,
                        "{cell:?} p={degree} {family:?} k={k} node {j}: {got} vs {w}"
                    );
                };
                if cell == CellType::Prism {
                    let nt = n * (n + 1) / 2;
                    let ctri = &perm[nt * n..];
                    assert_eq!(ctri.len(), nt * nt);
                    for (k, lex) in (0..nb).flat_map(|k| (0..nb).map(move |lex| (k, lex))) {
                        let (kt, kz, t, c) = (k % nt, k / nt, lex % nt, lex / nt);
                        check(k, lex, ctri[kt * nt + t] * c1[kz * n + c]);
                    }
                } else {
                    assert_eq!(perm.len(), nb);
                    for (k, lex) in (0..nb).flat_map(|k| (0..nb).map(move |lex| (k, lex))) {
                        let digit = |i: usize, d: usize| i / n.pow(d as u32) % n;
                        let got = (0..td)
                            .map(|d| c1[digit(k, d) * n + digit(lex, d)])
                            .product();
                        check(k, lex, got);
                    }
                }
                assert!(seen.iter().all(|&s| s), "{cell:?} p={degree}: permutation");
            }
        }
    }
}
