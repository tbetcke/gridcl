//! Device evaluation of the coordinate-element bases at changing points, for the pull-back
//! (docs/design.md, "Tabulation and quadrature"; ADR-0039, item 15).
//!
//! The `#[cube]` helpers evaluate the orthogonal sets of "Tabulation" and their first
//! derivatives by the host's recurrences (`gridcl-refcell/src/polynomials.rs`), in the same
//! basis order, specialised at comptime by cell type and degree: Legendre products on
//! intervals, quadrilaterals and hexahedra (`tensor.rs`), Dubiner on triangles and tetrahedra
//! and Dubiner times Legendre on prisms (`simplex.rs`), the orthonormal rationomials on
//! pyramids (`pyramid.rs`). Every loop is unrolled, so every recurrence constant is an
//! integer known at comptime, exact in `F`.
//!
//! The helpers leave out the normalisation φ_k = c_k φ̃_k (c_k a square root of an integer).
//! [`device_coefficients`] folds it into [`LagrangeElement::coefficients`] in f64 before the
//! narrowing to `F`: ψ_j = Σ_k C'[k·N + j] φ̃_k with C'[k·N + j] = c_k C[k·N + j]. The device
//! needs no square root, and the narrowed C' carries the same relative rounding as C.
//!
//! **Equal operations at every vector size.** CubeCL 0.11 folds x·1, x + 0, x − 0, x/1 and
//! operations on constants alone for one-lane values only, and then fuses multiply-adds of
//! what is left (docs/cubecl-upstream.md, C16): a constant 1 or 0, or an operation between
//! constants, would make the kernels at vector size 1 fuse other products than at larger sizes
//! and round differently. So these helpers never use the constants 0 and 1: they take a
//! runtime `zero` and `one`, read from the end of the coefficient buffer
//! ([`device_coefficients`]); every other constant is an integer or half-integer other than
//! 0 and 1, and every operation has a runtime operand.

mod loops;
mod pyramid;
mod simplex;
mod tensor;

#[cfg(any(test, feature = "bench"))]
pub(crate) use loops::broadcast_rows;
pub(crate) use loops::{
    dubiner_into, expansion_cooperative, expansion_loops, legendre_into, loops_capacity,
};

use cubecl::prelude::*;
use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_refcell::{LagrangeElement, NodeFamily, num_lagrange_nodes};

use crate::names::element_error;

/// The integer constants (c0, c1, c2, c3, c4) of step m → m + 1 of the homogeneous Jacobi
/// recurrence of gridcl-refcell's `scaled_jacobi`:
/// Q_{m+1} = (c1 (c2 x + c3 s) Q_m − c4 s² Q_{m−1}) / c0. For m ≥ 1 and a, b ≥ 0, c0, c1,
/// c2 and c4 are at least 3; c3 = a² − b² may be 0 or 1.
fn recurrence(m: usize, a: i64, b: i64) -> [i64; 5] {
    let m = m as i64;
    [
        2 * (m + 1) * (m + a + b + 1) * (2 * m + a + b),
        2 * m + a + b + 1,
        (2 * m + a + b + 2) * (2 * m + a + b),
        a * a - b * b,
        2 * (m + a) * (m + b) * (2 * m + a + b + 2),
    ]
}

/// `c` as a vector constant; `c` is neither 0 nor 1 (module docs).
#[cube]
fn cst<F: Float, N: Size>(#[comptime] c: i64) -> Vector<F, N> {
    comptime!(assert!(c != 0 && c != 1, "use the runtime zero and one"));
    Vector::<F, N>::broadcast(F::from_int(c))
}

/// h / 2 times the runtime `one` (h ≥ 0): `zero` for h = 0, `one` for h = 2, else the product
/// of the constant h / 2 (exact in `F` for the small h of the recurrences) and `one`.
#[cube]
fn halves<F: Float, N: Size>(
    #[comptime] h: i64,
    one: Vector<F, N>,
    zero: Vector<F, N>,
) -> Vector<F, N> {
    let mut out = zero;
    if comptime!(h == 2) {
        out = one;
    } else if comptime!(h != 0) {
        out = Vector::<F, N>::broadcast(F::new(comptime!(h as f32 * 0.5))) * one;
    }
    out
}

/// The integer `c` as a vector: `zero` for 0, `one` for 1, else the constant.
#[cube]
fn int<F: Float, N: Size>(
    #[comptime] c: i64,
    one: Vector<F, N>,
    zero: Vector<F, N>,
) -> Vector<F, N> {
    let mut out = zero;
    if comptime!(c == 1) {
        out = one;
    } else if comptime!(c != 0) {
        out = cst::<F, N>(c);
    }
    out
}

/// Q_0, …, Q_n(x, s) = s^m P_m^(a, b)(x / s) and their partial derivatives, by the
/// recurrence of gridcl-refcell's `scaled_jacobi`: entry 3m the value of Q_m, 3m + 1 ∂/∂x and
/// 3m + 2 ∂/∂s (`zero` unless `with_ds`).
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scaled_jacobi<F: Float, N: Size>(
    x: Vector<F, N>,
    s: Vector<F, N>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] n: usize,
    #[comptime] a: i64,
    #[comptime] b: i64,
    #[comptime] with_ds: bool,
) -> Array<Vector<F, N>> {
    let mut q = Array::<Vector<F, N>>::new(comptime!(3 * (n + 1)));
    q[0] = one;
    q[1] = zero;
    q[2] = zero;
    if comptime!(n > 0) {
        let half = Vector::<F, N>::broadcast(F::new(0.5f32));
        let ab2 = cst::<F, N>(comptime!(a + b + 2));
        let amb = int::<F, N>(comptime!(a - b), one, zero);
        q[3] = half * (ab2 * x + amb * s);
        q[4] = halves::<F, N>(comptime!(a + b + 2), one, zero);
        q[5] = zero;
        if comptime!(with_ds) {
            q[5] = halves::<F, N>(comptime!(a - b), one, zero);
        }
        #[unroll]
        for m in 1..n {
            let c = comptime!(recurrence(m, a, b));
            let c0 = cst::<F, N>(comptime!(c[0]));
            let c1 = cst::<F, N>(comptime!(c[1]));
            let c2 = cst::<F, N>(comptime!(c[2]));
            let c3 = int::<F, N>(comptime!(c[3]), one, zero);
            let c4 = cst::<F, N>(comptime!(c[4]));
            let p1 = q[3 * m];
            let p0 = q[3 * m - 3];
            let p1x = q[3 * m + 1];
            let p0x = q[3 * m - 2];
            let lin = c2 * x + c3 * s;
            let ss = s * s;
            q[3 * m + 3] = (c1 * lin * p1 - c4 * ss * p0) / c0;
            q[3 * m + 4] = (c1 * (c2 * p1 + lin * p1x) - c4 * ss * p0x) / c0;
            q[3 * m + 5] = zero;
            if comptime!(with_ds) {
                let p1s = q[3 * m + 2];
                let p0s = q[3 * m - 1];
                let two = cst::<F, N>(2i64);
                q[3 * m + 5] = (c1 * (c3 * p1 + lin * p1s) - c4 * (two * s * p0 + ss * p0s)) / c0;
            }
        }
    }
    q
}

/// The unnormalised Legendre polynomials Q_i(2t − 1, 1) = P_i(2t − 1) on [0, 1], i ≤ n:
/// entry 2i the value, 2i + 1 the derivative along t.
#[cube]
pub(crate) fn legendre<F: Float, N: Size>(
    t: Vector<F, N>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] n: usize,
) -> Array<Vector<F, N>> {
    let two = cst::<F, N>(2i64);
    let q = scaled_jacobi::<F, N>(two * t - one, one, one, zero, n, 0i64, 0i64, false);
    let mut out = Array::<Vector<F, N>>::new(comptime!(2 * (n + 1)));
    #[unroll]
    for i in 0..n + 1 {
        out[2 * i] = q[3 * i];
        out[2 * i + 1] = two * q[3 * i + 1];
    }
    out
}

/// Adds basis function k to the sums: `acc[c (1 + td) + d] += a[k g + c] · v_d` for each of
/// the `g` columns c of `a`, with v_0 the value of φ̃_k and v_{1+d} its derivative along ξ_d
/// (the v beyond `td` are unused). Each sum adds a product of runtime values, so the fusion
/// is the same at every vector size.
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit<F: Float, N: Size>(
    acc: &mut Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    #[comptime] k: usize,
    v0: Vector<F, N>,
    v1: Vector<F, N>,
    v2: Vector<F, N>,
    v3: Vector<F, N>,
    #[comptime] td: usize,
    #[comptime] g: usize,
) {
    #[unroll]
    for c in 0..g {
        let w = a[k * g + c];
        let o = c * (td + 1);
        acc[o] += w * v0;
        acc[o + 1] += w * v1;
        if comptime!(td >= 2) {
            acc[o + 2] += w * v2;
        }
        if comptime!(td >= 3) {
            acc[o + 3] += w * v3;
        }
    }
}

/// Σ_k a_k φ̃_k(ξ) and its first derivatives, for `g` columns of expansion coefficients:
/// `a[k g + c]` is a_k of column c, k in the order of gridcl-refcell's `orthonormal_at`
/// (the unnormalised orthogonal set of degree `degree` on `cell`, module docs). Entry
/// c (1 + tdim) of the result is the sum of column c and c (1 + tdim) + 1 + d its derivative
/// along ξ_d. The set is never stored: each φ̃_k is added to the sums as the recurrences
/// produce it, so a kernel keeps g (1 + tdim) sums live rather than a table of every basis
/// function, which slowed the CUDA compiler by minutes at p = 4 and broke the launch of the
/// f64 hexahedron at vector size 2 (M9 T10a). With a_k = C'[k N + j] (`device_coefficients`)
/// the result is the Lagrange function ψ_j; with a_k = Σ_j C'[k N + j] x_j it is the
/// coordinate map. `zero` and `one` are the runtime constants of the module docs. The pyramid
/// clamps ζ to at most 1 − 2^−10 first (ADR-0039, item 15).
#[cube]
pub(crate) fn expansion<F: Float, N: Size>(
    xi: &Array<Vector<F, N>>,
    a: &Array<Vector<F, N>>,
    one: Vector<F, N>,
    zero: Vector<F, N>,
    #[comptime] cell: CellType,
    #[comptime] degree: usize,
    #[comptime] g: usize,
) -> Array<Vector<F, N>> {
    let td = comptime!(gridcl_refcell::tdim(cell));
    let mut acc = Array::<Vector<F, N>>::new(comptime!(g * (td + 1)));
    #[unroll]
    for c in 0..g * (td + 1) {
        acc[c] = zero;
    }
    if comptime!(cell == CellType::Interval) {
        tensor::interval::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Quadrilateral) {
        tensor::quadrilateral::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Hexahedron) {
        tensor::hexahedron::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Triangle) {
        simplex::triangle::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Tetrahedron) {
        simplex::tetrahedron::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Prism) {
        simplex::prism::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else if comptime!(cell == CellType::Pyramid) {
        pyramid::pyramid::<F, N>(xi, a, &mut acc, one, zero, degree, g);
    } else {
        // The point cell: one basis function, the constant 1.
        #[unroll]
        for c in 0..g {
            acc[c] = a[c];
        }
    }
    acc
}

/// The normalisation c_k of every basis function, φ_k = c_k φ̃_k, in the order of
/// [`expansion`].
fn normalisation(cell: CellType, p: usize) -> Vec<f64> {
    let sqrt = |n: usize| (n as f64).sqrt();
    let triangle = |out: &mut Vec<f64>, scale: f64| {
        for i in 0..=p {
            for j in 0..=p - i {
                out.push(scale * sqrt((2 * i + 1) * (2 * i + 2 * j + 2)));
            }
        }
    };
    let mut out = Vec::with_capacity(num_lagrange_nodes(cell, p as u32));
    match cell {
        CellType::Point => out.push(1.0),
        CellType::Interval => out.extend((0..=p).map(|i| sqrt(2 * i + 1))),
        CellType::Quadrilateral => {
            for iy in 0..=p {
                for ix in 0..=p {
                    out.push(sqrt((2 * ix + 1) * (2 * iy + 1)));
                }
            }
        }
        CellType::Hexahedron => {
            for iz in 0..=p {
                for iy in 0..=p {
                    for ix in 0..=p {
                        out.push(sqrt((2 * ix + 1) * (2 * iy + 1) * (2 * iz + 1)));
                    }
                }
            }
        }
        CellType::Triangle => triangle(&mut out, 1.0),
        CellType::Tetrahedron => {
            for i in 0..=p {
                for j in 0..=p - i {
                    for k in 0..=p - i - j {
                        out.push(sqrt(
                            (2 * i + 1) * (2 * i + 2 * j + 2) * (2 * i + 2 * j + 2 * k + 3),
                        ));
                    }
                }
            }
        }
        CellType::Prism => {
            for iz in 0..=p {
                triangle(&mut out, sqrt(2 * iz + 1));
            }
        }
        CellType::Pyramid => {
            for i in 0..=p {
                for j in 0..=p {
                    let m = i.max(j);
                    for k in 0..=p - m {
                        out.push(sqrt((2 * i + 1) * (2 * j + 1) * (2 * k + 2 * m + 3)));
                    }
                }
            }
        }
    }
    out
}

/// C' of the module docs, row-major N × N, in f64: the coefficients of the Lagrange basis of
/// (cell, degree, family) over [`expansion`], followed by the runtime zero and one
/// (N² + 2 entries).
///
/// # Errors
///
/// `UnsupportedDegree` and `UnsupportedNodeFamily` of the element.
pub(crate) fn device_coefficients(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<f64>> {
    let element = LagrangeElement::new(cell, degree, family).map_err(element_error)?;
    let n = element.dim();
    let norms = normalisation(cell, degree as usize);
    if norms.len() != n {
        return Err(GridError::LengthMismatch {
            expected: n,
            found: norms.len(),
        });
    }
    let mut out: Vec<f64> = element
        .coefficients()
        .chunks_exact(n)
        .zip(&norms)
        .flat_map(|(row, &c)| row.iter().map(move |x| c * x))
        .collect();
    // The runtime zero and one of the module docs.
    out.extend([0.0, 1.0]);
    Ok(out)
}

#[cfg(test)]
mod tests;
