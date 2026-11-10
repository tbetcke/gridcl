//! Benchmark hooks of the degree sweep (docs/plans/M9h.md, T1; ADR-0038, ADR-0042). Only with
//! the Cargo feature `bench`, which is off by default and not part of the stable API;
//! gridcl-bench turns it on.
//!
//! - [`Interpolant`]: the device basis evaluator of the pull-back (`basis_eval::expansion`) on
//!   its own: a Lagrange interpolant and its first derivatives at given reference points.
//!
//! The pull-back's set-up and Newton step need no hook: `pull_back_block` always runs
//! `max_iterations` steps, so the sweep times it at 1 and 9 iterations and takes the step as
//! the difference over 8 (ADR-0042, item 1).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, Result, VECTOR_PAD,
    debug_assert_bindings,
};
use gridcl_dims::CellType;
use gridcl_refcell::{NodeFamily, num_lagrange_nodes, tdim};

use crate::basis_eval::{device_coefficients, expansion};
use crate::strategy::{self, StrategyFamily};

/// The interpolant Σ_j w_j ψ_j and its derivatives at each lane's point:
///   points        `[d][point]`, rows of `stride` vectors
///   coefficients  a_k = Σ_j C'[k N + j] w_j (N entries), then the runtime zero and one
///   out           `[deriv][point]`, deriv 0 the value and 1 + d the derivative along ξ_d
#[cube(launch_unchecked)]
fn interpolant_kernel<F: Float, N: Size>(
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
    let stride = stride as usize;
    let zero = Vector::<F, N>::broadcast(coefficients[nb]);
    let one = Vector::<F, N>::broadcast(coefficients[nb + 1]);
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            let mut xi = Array::<Vector<F, N>>::new(comptime!(td));
            #[unroll]
            for d in 0..td {
                xi[d] = points[d * stride + v];
            }
            let mut a = Array::<Vector<F, N>>::new(comptime!(nb));
            for k in 0..nb {
                a[k] = Vector::<F, N>::broadcast(coefficients[k]);
            }
            let e = expansion::<F, N>(&xi, &a, one, zero, cell, degree, 1usize);
            #[unroll]
            for d in 0..td + 1 {
                out[d * stride + v] = e[d];
            }
        }
    }
}

/// A Lagrange interpolant of one element on the device, evaluated by the pull-back's basis
/// evaluator.
///
/// # Example
///
/// ```
/// use gridcl_device::DeviceBuffer;
/// use gridcl_dims::CellType;
/// use gridcl_geometry::NodeFamily;
/// use gridcl_geometry::bench::Interpolant;
///
/// let client = gridcl_testing::test_client();
/// // The P1 interval interpolant of (1, 3) is 1 + 2t.
/// let f = Interpolant::<f32>::new(&client, CellType::Interval, 1, NodeFamily::Equispaced, &[1.0, 3.0])?;
/// let mut points = vec![0.0f32; 16];
/// points[0] = 0.25;
/// let out = f.evaluate(&DeviceBuffer::from_host(&client, &points)?, 16)?.to_host()?;
/// assert!((out[0] - 1.5).abs() < 1e-6 && (out[16] - 2.0).abs() < 1e-6);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub struct Interpolant<F: Real> {
    cell: CellType,
    degree: u32,
    coefficients: DeviceBuffer<F>,
}

impl<F: Real> Interpolant<F> {
    /// The interpolant of `values`, one per node in gridcl-refcell's node order, on the element
    /// (`cell`, `degree`, `family`). The coefficients a_k = Σ_j C'[k N + j] w_j are formed in f64
    /// and narrowed to `F`.
    ///
    /// # Errors
    ///
    /// `UnsupportedDegree` and `UnsupportedNodeFamily` of the element; `LengthMismatch` if
    /// `values` does not have one entry per node; allocation errors.
    pub fn new(
        client: &Client,
        cell: CellType,
        degree: u32,
        family: NodeFamily,
        values: &[f64],
    ) -> Result<Self> {
        let c = device_coefficients(cell, degree, family)?;
        let n = num_lagrange_nodes(cell, degree);
        if values.len() != n {
            return Err(GridError::LengthMismatch {
                expected: n,
                found: values.len(),
            });
        }
        let coefficients: Vec<F> = (0..n)
            .map(|k| (0..n).map(|j| c[k * n + j] * values[j]).sum::<f64>())
            .chain([c[n * n], c[n * n + 1]])
            .map(F::narrow)
            .collect();
        Ok(Interpolant {
            cell,
            degree,
            coefficients: DeviceBuffer::from_host(client, &coefficients)?,
        })
    }

    /// The value and the tdim first derivatives at every point of `points`, rows `[d][point]`
    /// of `stride` entries (padding points included), as rows `[deriv][point]` of `stride`
    /// entries. One launch, planned with `KernelKind::PullBack`.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `stride` is not a multiple of `VECTOR_PAD` or `points` does not hold
    /// tdim rows of it; `TooLarge` and allocation errors; `NotYetImplemented` where the
    /// strategy chosen for the PullBack family does not run yet.
    pub fn evaluate(&self, points: &DeviceBuffer<F>, stride: usize) -> Result<DeviceBuffer<F>> {
        let td = tdim(self.cell);
        if !stride.is_multiple_of(VECTOR_PAD) || points.len() != td * stride {
            return Err(GridError::LengthMismatch {
                expected: td * stride.next_multiple_of(VECTOR_PAD),
                found: points.len(),
            });
        }
        let client = self.coefficients.client();
        let out = DeviceBuffer::<F>::uninit(client, (1 + td) * stride)?;
        if stride == 0 {
            return Ok(out);
        }
        // Only `direct` runs until T6b (docs/plans/M9h.md); `choose` rejects the others.
        strategy::choose(
            client,
            StrategyFamily::PullBack,
            self.cell,
            self.degree,
            size_of::<F>(),
        )?;
        let nb = num_lagrange_nodes(self.cell, self.degree);
        let plan = LaunchPlan::elementwise(
            client,
            KernelKind::PullBack,
            stride,
            size_of::<F>(),
            VECTOR_PAD,
            nb + 3 * td + 4,
        )?;
        let n = plan.vector_size;
        let rows = u32::try_from(stride / n).map_err(|_| GridError::TooLarge)?;
        debug_assert_bindings(client, 3);
        // SAFETY: n divides VECTOR_PAD and so `stride`; the kernel handles vectors
        // v < num_vectors = stride / n and reads `points` at d · stride / n + v < td · stride / n
        // (its length, checked above), `coefficients` below nb + 2 (its length), and writes
        // `out` at d · stride / n + v < (1 + td) · stride / n, its length.
        unsafe {
            interpolant_kernel::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                n,
                points.vector_arg(n),
                self.coefficients.arg(),
                out.vector_arg(n),
                rows,
                plan.num_vectors,
                plan.items_per_unit,
                self.cell,
                self.degree as usize,
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use gridcl_refcell::LagrangeElement;

    use super::*;

    /// Against gridcl-refcell's tabulation of the interpolant, at p = 1–4 on every cell type
    /// (pyramids to 3) and at a few points, in f32 and, where supported, f64: within 10^−10
    /// (f64) or 10^−3 (f32) of max(1, |value|), loose bounds for the rounding through the
    /// coefficients (the accuracy tests of the pull-back are the acceptance check).
    #[test]
    fn interpolant_matches_the_host_tabulation() {
        let client = gridcl_testing::test_client();
        matches_host::<f32>(&client, 1e-3);
        if gridcl_testing::supports::<f64>(&client) {
            matches_host::<f64>(&client, 1e-10);
        }
    }

    fn matches_host<F: Real>(client: &Client, tol: f64) {
        let cells = [
            CellType::Interval,
            CellType::Triangle,
            CellType::Quadrilateral,
            CellType::Tetrahedron,
            CellType::Hexahedron,
            CellType::Prism,
            CellType::Pyramid,
        ];
        for cell in cells {
            let td = tdim(cell);
            let top = if cell == CellType::Pyramid { 3 } else { 4 };
            for p in 1..=top {
                let e = LagrangeElement::new(cell, p, NodeFamily::Equispaced).unwrap();
                let nb = e.dim();
                let values: Vec<f64> = (0..nb).map(|j| 1.0 + (j % 7) as f64 / 8.0).collect();
                let f = Interpolant::<F>::new(client, cell, p, NodeFamily::Equispaced, &values)
                    .unwrap();
                // Three points inside every reference cell: the scaled vertex centroid.
                let pts: Vec<f64> = [0.2, 0.5, 0.7]
                    .iter()
                    .flat_map(|&s| (0..td).map(move |_| s / td as f64))
                    .collect();
                let mut rows = vec![F::narrow(0.0); td * VECTOR_PAD];
                for (q, x) in pts.chunks(td).enumerate() {
                    for d in 0..td {
                        rows[d * VECTOR_PAD + q] = F::narrow(x[d]);
                    }
                }
                let rows = DeviceBuffer::from_host(client, &rows).unwrap();
                // The points the device sees.
                let pts: Vec<f64> = (0..3)
                    .flat_map(|q| (0..td).map(move |d| (q, d)))
                    .map(|(q, d)| F::narrow(pts[q * td + d]).widen())
                    .collect();
                let out = f.evaluate(&rows, VECTOR_PAD).unwrap().to_host().unwrap();
                let tab = e.tabulate(&pts, 1).unwrap();
                for q in 0..3 {
                    for d in 0..=td {
                        let want: f64 = (0..nb).map(|j| tab.get(d, q, j) * values[j]).sum();
                        let got = out[d * VECTOR_PAD + q].widen();
                        assert!(
                            (got - want).abs() <= tol * want.abs().max(1.0),
                            "{cell:?} p={p} point {q} deriv {d}: {got} vs {want}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn bad_lengths_are_rejected() {
        let client = gridcl_testing::test_client();
        let cell = CellType::Triangle;
        let f =
            Interpolant::<f32>::new(&client, cell, 1, NodeFamily::Equispaced, &[1.0; 3]).unwrap();
        let rows = DeviceBuffer::<f32>::zeros(&client, 2 * VECTOR_PAD).unwrap();
        assert!(f.evaluate(&rows, VECTOR_PAD).is_ok());
        assert!(f.evaluate(&rows, 2 * VECTOR_PAD).is_err());
        assert!(f.evaluate(&rows, VECTOR_PAD / 2 * 3).is_err());
        assert!(
            Interpolant::<f32>::new(&client, cell, 1, NodeFamily::Equispaced, &[1.0; 4]).is_err()
        );
    }
}
