//! The device evaluation against gridcl-refcell's host tabulation (ADR-0039, item 15).

use cubecl::prelude::*;
use gridcl_device::testing::{record_plans, with_vector_size};
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, VECTOR_PAD, VectorSizeOverride,
    debug_assert_bindings,
};
use gridcl_dims::CellType::{self, *};
use gridcl_refcell::{LagrangeElement, NodeFamily, num_lagrange_nodes, tdim, vertices};

use super::{device_coefficients, normalisation, orthogonal_basis};

/// ψ_j and ∂_d ψ_j of the Lagrange basis at each lane's point:
///   points        `[d][point]`, rows of `stride` vectors
///   coefficients  C', N × N, then the runtime zero and one
///   out           `[(deriv, j)][point]`, deriv 0 the value and 1 + d the derivative along ξ_d
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn lagrange_kernel<F: Float, N: Size>(
    points: &[Vector<F, N>],
    coefficients: &[F],
    out: &mut [Vector<F, N>],
    stride: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] cell: CellType,
    #[comptime] degree: usize,
) {
    let td = comptime!(tdim(cell));
    let nb = comptime!(num_lagrange_nodes(cell, degree as u32));
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            let mut xi = Array::<Vector<F, N>>::new(comptime!(td));
            #[unroll]
            for d in 0..td {
                xi[d] = points[d * stride as usize + v];
            }
            let zero = Vector::<F, N>::broadcast(coefficients[nb * nb]);
            let one = Vector::<F, N>::broadcast(coefficients[nb * nb + 1]);
            let phi = orthogonal_basis::<F, N>(&xi, one, zero, cell, degree);
            for j in 0..nb {
                #[unroll]
                for d in 0..td + 1 {
                    let mut acc = phi[d] * Vector::<F, N>::broadcast(coefficients[j]);
                    for k in 1..nb {
                        acc += phi[k * (td + 1) + d]
                            * Vector::<F, N>::broadcast(coefficients[k * nb + j]);
                    }
                    out[(d * nb + j) * stride as usize + v] = acc;
                }
            }
        }
    }
}

/// The device's ψ table of (cell, degree, family) at `points` (flat, tdim per point), as
/// `[(deriv, j)][point]` without padding.
fn evaluate<F: Real>(
    cell: CellType,
    degree: u32,
    family: NodeFamily,
    points: &[F],
) -> Result<Vec<F>, GridError> {
    let client = gridcl_testing::test_client();
    let td = tdim(cell);
    let nb = num_lagrange_nodes(cell, degree);
    let count = points.len() / td;
    let padded = count.next_multiple_of(VECTOR_PAD);
    // Padding lanes repeat the last point.
    let mut rows = vec![F::narrow(0.0); td * padded];
    for d in 0..td {
        for q in 0..padded {
            rows[d * padded + q] = points[q.min(count - 1) * td + d];
        }
    }
    let rows = DeviceBuffer::from_host(&client, &rows)?;
    let c: Vec<F> = device_coefficients(cell, degree, family)?
        .into_iter()
        .map(F::narrow)
        .collect();
    let c = DeviceBuffer::from_host(&client, &c)?;
    let out = DeviceBuffer::<F>::uninit(&client, (1 + td) * nb * padded)?;
    let plan = LaunchPlan::elementwise(
        &client,
        KernelKind::PullBack,
        padded,
        size_of::<F>(),
        VECTOR_PAD,
        2,
    )?;
    let n = plan.vector_size;
    debug_assert_bindings(&client, 3);
    // SAFETY: n divides VECTOR_PAD and so `padded`; the kernel handles vectors
    // v < num_vectors = padded / n and reads `rows` at d · padded / n + v < td · padded / n,
    // `c` below nb² + 2, and writes `out` at (d · nb + j) · padded / n + v < (1 + td) · nb ·
    // padded / n.
    unsafe {
        lagrange_kernel::launch_unchecked::<F>(
            &client,
            plan.cube_count,
            plan.cube_dim,
            n,
            rows.vector_arg(n),
            c.arg(),
            out.vector_arg(n),
            (padded / n) as u32,
            plan.num_vectors,
            plan.items_per_unit,
            cell,
            degree as usize,
        );
    }
    let all = out.to_host()?;
    Ok((0..(1 + td) * nb)
        .flat_map(|r| all[r * padded..r * padded + count].to_vec())
        .collect())
}

/// A splitmix64 stream in [0, 1).
struct Uniform(u64);

impl Uniform {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The cell's vertices, then `n` random points inside it (flat, tdim per point); for the
/// pyramid also points above the clamp ζ = 1 − 2^−10, the apex among them.
fn sample(cell: CellType, n: usize, seed: u64) -> Vec<f64> {
    let mut u = Uniform(seed);
    let mut out: Vec<f64> = vertices(cell).iter().flat_map(|v| v.to_vec()).collect();
    for _ in 0..n {
        let point: Vec<f64> = match cell {
            Interval => vec![u.next()],
            Quadrilateral => vec![u.next(), u.next()],
            Hexahedron => vec![u.next(), u.next(), u.next()],
            Triangle | Prism => {
                let (mut x, mut y) = (u.next(), u.next());
                if x + y > 1.0 {
                    (x, y) = (1.0 - x, 1.0 - y);
                }
                let mut p = vec![x, y];
                if cell == Prism {
                    p.push(u.next());
                }
                p
            }
            Tetrahedron => loop {
                let p = [u.next(), u.next(), u.next()];
                if p.iter().sum::<f64>() < 1.0 {
                    break p.to_vec();
                }
            },
            Pyramid => {
                let z = u.next();
                vec![u.next() * (1.0 - z), u.next() * (1.0 - z), z]
            }
            Point => unreachable!(),
        };
        out.extend(point);
    }
    if cell == Pyramid {
        let near = 1.0 - 2f64.powi(-12);
        out.extend([0.0, 0.0, 1.0, 0.5 * (1.0 - near), 0.25 * (1.0 - near), near]);
    }
    out
}

/// Every coordinate element of the tests: degrees 1–4 (pyramids 1–3), equispaced, and GLL
/// where offered.
fn elements() -> Vec<(CellType, u32, NodeFamily)> {
    let mut out = vec![];
    for cell in [
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ] {
        let top = if cell == Pyramid { 3 } else { 4 };
        for p in 1..=top {
            out.push((cell, p, NodeFamily::Equispaced));
            if matches!(cell, Interval | Quadrilateral | Hexahedron) {
                out.push((cell, p, NodeFamily::Gll));
            }
        }
    }
    out
}

/// The device table against `LagrangeElement::tabulate` at the same (narrowed, pyramid
/// clamped) points: relative `tol`, against 1 for entries below 1 in magnitude.
fn matches_host<F: Real>(tol: f64) {
    for (cell, p, family) in elements() {
        let td = tdim(cell);
        let points: Vec<F> = sample(cell, 40, 17 + u64::from(p))
            .into_iter()
            .map(F::narrow)
            .collect();
        let got = evaluate::<F>(cell, p, family, &points).unwrap();
        let mut host: Vec<f64> = points.iter().map(|x| x.widen()).collect();
        if cell == Pyramid {
            for x in host.as_chunks_mut::<3>().0 {
                x[2] = x[2].min(1.0 - 2f64.powi(-10));
            }
        }
        let e = LagrangeElement::new(cell, p, family).unwrap();
        let t = e.tabulate(&host, 1).unwrap();
        let (nb, count) = (e.dim(), t.num_points);
        for d in 0..=td {
            for j in 0..nb {
                for q in 0..count {
                    let want = t.get(d, q, j);
                    let have = got[(d * nb + j) * count + q].widen();
                    assert!(
                        (have - want).abs() <= tol * want.abs().max(1.0),
                        "{cell:?} p = {p} {family:?}, deriv {d}, basis {j}, point {:?}: {have} vs {want}",
                        &host[q * td..(q + 1) * td]
                    );
                }
            }
        }
    }
}

#[test]
fn values_and_derivatives_match_the_host() {
    matches_host::<f32>(gridcl_testing::TOL_F32);
    if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
        matches_host::<f64>(gridcl_testing::TOL_F64);
    } else {
        eprintln!("values_and_derivatives_match_the_host: f64 is not supported; skipped");
    }
}

/// Every vector size the device offers gives the scalar result: bitwise on the CPU runtime,
/// to the tolerance elsewhere. The plans are `KernelKind::PullBack` at the forced size.
fn vector_size_independent<F: Real>(tol: f64) {
    let client = gridcl_testing::test_client();
    let widest = client
        .io_optimized_vector_sizes(size_of::<F>())
        .next()
        .unwrap_or(1);
    for (cell, p, family) in elements() {
        let points: Vec<F> = sample(cell, 37, 5).into_iter().map(F::narrow).collect();
        let scalar = with_vector_size(VectorSizeOverride::Scalar, || {
            evaluate::<F>(cell, p, family, &points)
        })
        .unwrap();
        for n in [2, 4, 8, 16].into_iter().filter(|&n| n <= widest) {
            let (vector, plans) = record_plans(|| {
                with_vector_size(VectorSizeOverride::Fixed(n), || {
                    evaluate::<F>(cell, p, family, &points)
                })
            });
            assert!(
                plans
                    .iter()
                    .all(|(k, plan)| *k == KernelKind::PullBack && plan.vector_size == n),
                "{plans:?}"
            );
            for (i, (a, b)) in scalar.iter().zip(&vector.unwrap()).enumerate() {
                let (a, b) = (a.widen(), b.widen());
                if gridcl_testing::bitwise_device() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{cell:?} p = {p}, n = {n}, entry {i}"
                    );
                } else {
                    assert!(
                        (a - b).abs() <= tol * a.abs().max(1.0),
                        "{cell:?} p = {p}, n = {n}, entry {i}: {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn every_vector_size_gives_the_scalar_result() {
    vector_size_independent::<f32>(gridcl_testing::TOL_F32);
    if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
        vector_size_independent::<f64>(gridcl_testing::TOL_F64);
    } else {
        eprintln!("every_vector_size_gives_the_scalar_result: f64 is not supported; skipped");
    }
}

/// The normalisation has one factor per basis function, and folding it in gives
/// `coefficients` scaled row by row.
#[test]
fn normalisation_matches_the_basis_size() {
    for cell in [
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ] {
        for p in 1..=6 {
            assert_eq!(
                normalisation(cell, p).len(),
                num_lagrange_nodes(cell, p as u32),
                "{cell:?} p = {p}"
            );
        }
    }
    let c = device_coefficients(Interval, 1, NodeFamily::Equispaced).unwrap();
    // φ_1 = √3 φ̃_1: ψ_0 = φ_0 / 2 − φ_1 / (2√3) = φ̃_0 / 2 − φ̃_1 / 2; then 0 and 1.
    assert_eq!(c.len(), 6);
    assert!((c[0] - 0.5).abs() < 1e-15 && (c[2] + 0.5).abs() < 1e-15);
    assert_eq!(c[4..], [0.0, 1.0]);
    assert!(matches!(
        device_coefficients(Triangle, 3, NodeFamily::Gll),
        Err(GridError::UnsupportedNodeFamily { .. })
    ));
}

/// The coefficients are uploaded once per geometry block and cached with its E-vector.
#[test]
fn coefficients_are_cached_per_block() {
    use gridcl_device::DeviceStrided;
    use gridcl_dims::{Cell1, D1, D2};
    use gridcl_parallel::{IndexMap, Serial};

    use crate::{CoordinateElement, Geometry, GeometryBlock};

    let client = gridcl_testing::test_client();
    let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0, 0.0, 0.1]).unwrap();
    let element = CoordinateElement::<D1> {
        cell: Cell1::Interval,
        degree: 2,
        family: NodeFamily::Gll,
    };
    let dofmap = DeviceStrided {
        data: DeviceBuffer::from_host(&client, &[0u32, 1, 2]).unwrap(),
        stride: 3,
    };
    let geometry = Geometry::<f32, D1, D2>::new(
        vec![GeometryBlock { element, dofmap }],
        nodes,
        IndexMap::new(&Serial, 3).unwrap(),
        [0.0, 0.0],
    )
    .unwrap();
    let before = geometry.footprint_bytes();
    let xe = geometry.evector(0).unwrap();
    let with_evector = geometry.footprint_bytes();
    let first = xe.coefficients(Interval, 2, NodeFamily::Gll).unwrap();
    let second = xe.coefficients(Interval, 2, NodeFamily::Gll).unwrap();
    assert!(core::ptr::eq(first, second));
    let want: Vec<f32> = device_coefficients(Interval, 2, NodeFamily::Gll)
        .unwrap()
        .into_iter()
        .map(f32::narrow)
        .collect();
    assert_eq!(first.to_host().unwrap(), want);
    assert!(before < with_evector);
    assert_eq!(
        geometry.footprint_bytes(),
        with_evector + first.footprint_bytes()
    );
}
