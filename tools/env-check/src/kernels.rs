//! Sample kernels and their host references.
//!
//! The element-wise add runs at vector size 1 and at the widest IO-optimized size;
//! the u32 histogram uses `Atomic<u32>::fetch_add`. Results are compared bitwise
//! with the host. The cube count comes from `calculate_cube_count_elemwise`
//! (docs/plans/E0.md, T1: `LaunchPlan` arrives in M2, and env-check is not library code).

use std::time::Instant;

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;

use crate::report::{CheckStatus, KernelCheck};

/// Element count of the sample kernels (10^7).
pub const DEFAULT_ELEMENTS: usize = 10_000_000;

/// Bins of the atomic histogram.
pub const HISTOGRAM_BINS: usize = 256;

/// A floating-point type the element-wise add runs in.
pub trait Precision: Float + CubeElement + Copy + PartialEq + std::fmt::Debug {
    /// `f32` or `f64`.
    const NAME: &'static str;
    /// Size in bytes, for `io_optimized_vector_sizes`.
    const BYTES: usize;
    /// Exact conversion of a small integer.
    fn from_u32(value: u32) -> Self;
    /// Rounding conversion from f64 (exact for f64).
    fn from_f64(value: f64) -> Self;
    fn add(self, other: Self) -> Self;
    fn to_bits_u64(self) -> u64;
}

impl Precision for f32 {
    const NAME: &'static str = "f32";
    const BYTES: usize = 4;
    fn from_u32(value: u32) -> Self {
        debug_assert!(value < (1 << 24), "{value} is not exact in f32");
        value as f32
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Precision for f64 {
    const NAME: &'static str = "f64";
    const BYTES: usize = 8;
    fn from_u32(value: u32) -> Self {
        f64::from(value)
    }
    fn from_f64(value: f64) -> Self {
        value
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}

/// `out = a + b`, one vector of `N` lanes per unit; units past the end do nothing.
///
/// Every lane is an independent element, so the result at any `N` equals the scalar sum.
#[cube(launch_unchecked)]
fn elementwise_add_kernel<F: Float, N: Size>(
    a: &[Vector<F, N>],
    b: &[Vector<F, N>],
    out: &mut [Vector<F, N>],
) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
    }
}

/// `bins[values[i]] += 1` with a u32 atomic add; values outside the bins are skipped.
#[cube(launch_unchecked)]
fn atomic_histogram_kernel(values: &[u32], bins: &[Atomic<u32>]) {
    if ABSOLUTE_POS < values.len() {
        let bin = values[ABSOLUTE_POS] as usize;
        if bin < bins.len() {
            bins[bin].fetch_add(1u32);
        }
    }
}

/// The two inputs of the element-wise add, deterministic and not all integers.
///
/// Most sums round, so the bitwise comparison checks round-to-nearest addition.
pub fn add_inputs<F: Precision>(elements: usize) -> (Vec<F>, Vec<F>) {
    let a = (0..elements)
        .map(|i| F::from_f64((i % 8191) as f64 * 0.1 - 300.0))
        .collect();
    let b = (0..elements)
        .map(|i| F::from_f64(((i * 7) % 4099) as f64 / 3.0))
        .collect();
    (a, b)
}

/// Host reference of the element-wise add.
pub fn host_add<F: Precision>(a: &[F], b: &[F]) -> Vec<F> {
    assert_eq!(a.len(), b.len(), "inputs of different lengths");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| Precision::add(x, y))
        .collect()
}

/// Deterministic histogram input with values in `0..bins`.
pub fn histogram_input(elements: usize, bins: usize) -> Vec<u32> {
    assert!(
        bins > 0 && u32::try_from(bins).is_ok(),
        "bins must be in 1..=u32::MAX"
    );
    let bins = bins as u64;
    (0..elements as u64)
        // Knuth's multiplicative hash spreads the values unevenly but deterministically.
        .map(|i| ((i.wrapping_mul(2_654_435_761) >> 13) % bins) as u32)
        .collect()
}

/// Host reference of the histogram.
pub fn host_histogram(values: &[u32], bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; bins];
    for &value in values {
        if let Some(count) = counts.get_mut(value as usize) {
            *count += 1;
        }
    }
    counts
}

/// Index and values of the first element where `got` and `want` differ bitwise,
/// or a length mismatch.
pub fn first_mismatch<F: Precision>(got: &[F], want: &[F]) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!(
            "length {} differs from the expected {}",
            got.len(),
            want.len()
        ));
    }
    let (index, (g, w)) = got
        .iter()
        .zip(want)
        .enumerate()
        .find(|(_, (g, w))| g.to_bits_u64() != w.to_bits_u64())?;
    let mismatches = got
        .iter()
        .zip(want)
        .filter(|(g, w)| g.to_bits_u64() != w.to_bits_u64())
        .count();
    Some(format!(
        "{mismatches} of {} elements differ; first at index {index}: got {g:?} ({:#x}), want {w:?} ({:#x})",
        got.len(),
        g.to_bits_u64(),
        w.to_bits_u64()
    ))
}

/// The vector sizes the element-wise add runs at: 1 and the widest IO-optimized size.
pub fn add_vector_sizes(client: &Client, elem_bytes: usize) -> Vec<usize> {
    let widest = client
        .io_optimized_vector_sizes(elem_bytes)
        .next()
        .unwrap_or(1);
    if widest > 1 { vec![1, widest] } else { vec![1] }
}

/// A launch shape for `units` units with one item each.
struct Shape {
    cube_count: CubeCount,
    cube_dim: CubeDim,
}

/// The cube dim and the cube count from `calculate_cube_count_elemwise`, checking in usize
/// arithmetic that the launch covers every unit and that the counts fit the kernel's u32s.
fn elementwise_shape(client: &Client, units: usize) -> Result<Shape, String> {
    if units == 0 {
        return Err("no work items".to_string());
    }
    u32::try_from(units).map_err(|_| format!("{units} work items exceed u32"))?;
    let cube_dim = CubeDim::new(client, units);
    let cube_count = calculate_cube_count_elemwise(client, units, cube_dim);
    let CubeCount::Static(x, y, z) = cube_count else {
        return Err("calculate_cube_count_elemwise returned a dynamic count".to_string());
    };
    let max = client.properties().hardware.max_cube_count;
    if x > max.0 || y > max.1 || z > max.2 {
        return Err(format!(
            "cube count ({x}, {y}, {z}) exceeds the device maximum {max:?}"
        ));
    }
    let launched = [x, y, z, cube_dim.num_elems()]
        .into_iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(n as usize))
        .ok_or_else(|| "launched unit count overflows usize".to_string())?;
    if launched < units {
        return Err(format!(
            "the launch covers {launched} units, fewer than the {units} needed"
        ));
    }
    Ok(Shape {
        cube_count,
        cube_dim,
    })
}

/// `count` elements of `elem_bytes` bytes, checked against overflow and `max_page_size`.
fn buffer_bytes(client: &Client, count: usize, elem_bytes: usize) -> Result<usize, String> {
    let bytes = count
        .checked_mul(elem_bytes)
        .ok_or_else(|| format!("{count} x {elem_bytes} bytes overflows usize"))?;
    let max = client.properties().memory.max_page_size;
    if bytes as u64 > max {
        return Err(format!(
            "a buffer of {bytes} bytes exceeds max_page_size {max}"
        ));
    }
    Ok(bytes)
}

/// Milliseconds since `start`, rounded to microseconds so the report round-trips exactly.
fn millis_since(start: Instant) -> f64 {
    (start.elapsed().as_secs_f64() * 1e6).round() / 1e3
}

/// Runs the element-wise add in precision `F` at `vector_size` and compares it with the host.
///
/// When `elements` is not a multiple of `vector_size`, the buffers are padded with zeros to
/// whole vectors and only the first `elements` results are compared.
pub fn check_elementwise_add<F: Precision>(
    client: &Client,
    elements: usize,
    vector_size: usize,
) -> KernelCheck {
    let mut check = KernelCheck {
        kernel: "elementwise_add".to_string(),
        precision: F::NAME.to_string(),
        vector_size,
        elements,
        status: CheckStatus::Fail,
        millis: None,
        detail: None,
    };
    match run_elementwise_add::<F>(client, elements, vector_size) {
        Ok((millis, mismatch)) => {
            check.millis = Some(millis);
            check.status = match mismatch {
                None => CheckStatus::Pass,
                Some(_) => CheckStatus::Fail,
            };
            check.detail = mismatch;
        }
        Err(error) => check.detail = Some(error),
    }
    check
}

/// The add's run time and its first mismatch with the host, or why it could not run.
fn run_elementwise_add<F: Precision>(
    client: &Client,
    elements: usize,
    vector_size: usize,
) -> Result<(f64, Option<String>), String> {
    if vector_size == 0 {
        return Err("vector size 0".to_string());
    }
    let num_vectors = elements.div_ceil(vector_size);
    let padded = num_vectors
        .checked_mul(vector_size)
        .ok_or("padded element count overflows usize")?;
    buffer_bytes(client, padded, F::BYTES)?;
    let shape = elementwise_shape(client, num_vectors)?;

    let (mut a, mut b) = add_inputs::<F>(elements);
    let want = host_add(&a, &b);
    a.resize(padded, F::from_u32(0));
    b.resize(padded, F::from_u32(0));
    let a_handle = client.create_from_slice(F::as_bytes(&a));
    let b_handle = client.create_from_slice(F::as_bytes(&b));
    let out_handle = client.create_from_slice(F::as_bytes(&vec![F::from_u32(0); padded]));

    let start = Instant::now();
    // SAFETY: each buffer holds `padded = num_vectors * vector_size` elements of `F`, so its
    // length is `num_vectors` vectors of `vector_size` lanes, as declared to from_raw_parts.
    // The kernel reads and writes index ABSOLUTE_POS only when it is below out.len(), the
    // shortest of the three lengths, and has no loops.
    unsafe {
        elementwise_add_kernel::launch_unchecked::<F>(
            client,
            shape.cube_count,
            shape.cube_dim,
            vector_size,
            BufferArg::from_raw_parts(a_handle, num_vectors),
            BufferArg::from_raw_parts(b_handle, num_vectors),
            BufferArg::from_raw_parts(out_handle.clone(), num_vectors),
        );
    }
    let bytes = client
        .read_one(out_handle)
        .map_err(|e| format!("kernel or read-back failed: {e}"))?;
    let millis = millis_since(start);

    let got = F::from_bytes(&bytes);
    let got = got
        .get(..elements)
        .ok_or_else(|| format!("read back {} elements, fewer than {elements}", got.len()))?;
    Ok((millis, first_mismatch(got, &want)))
}

/// Runs the u32 atomic histogram and compares it with the host.
pub fn check_atomic_histogram(client: &Client, elements: usize, bins: usize) -> KernelCheck {
    let mut check = KernelCheck {
        kernel: "atomic_histogram".to_string(),
        precision: "u32".to_string(),
        vector_size: 1,
        elements,
        status: CheckStatus::Fail,
        millis: None,
        detail: None,
    };
    match run_atomic_histogram(client, elements, bins) {
        Ok((millis, mismatch)) => {
            check.millis = Some(millis);
            check.status = match mismatch {
                None => CheckStatus::Pass,
                Some(_) => CheckStatus::Fail,
            };
            check.detail = mismatch;
        }
        Err(error) => check.detail = Some(error),
    }
    check
}

/// The histogram's run time and its first mismatch with the host, or why it could not run.
fn run_atomic_histogram(
    client: &Client,
    elements: usize,
    bins: usize,
) -> Result<(f64, Option<String>), String> {
    // Every count is at most `elements`, which must fit a u32 bin.
    u32::try_from(elements).map_err(|_| format!("{elements} elements overflow a u32 bin"))?;
    if bins == 0 {
        return Err("no bins".to_string());
    }
    buffer_bytes(client, elements, size_of::<u32>())?;
    buffer_bytes(client, bins, size_of::<u32>())?;
    let shape = elementwise_shape(client, elements)?;

    let values = histogram_input(elements, bins);
    let want = host_histogram(&values, bins);
    let values_handle = client.create_from_slice(u32::as_bytes(&values));
    let bins_handle = client.create_from_slice(u32::as_bytes(&vec![0u32; bins]));

    let start = Instant::now();
    // SAFETY: `values` holds `elements` u32s and `bins` holds `bins` u32s, the lengths given
    // to from_raw_parts. The kernel reads values[ABSOLUTE_POS] only below values.len() and
    // adds to bins[bin] only below bins.len(), and has no loops.
    unsafe {
        atomic_histogram_kernel::launch_unchecked(
            client,
            shape.cube_count,
            shape.cube_dim,
            BufferArg::from_raw_parts(values_handle, elements),
            BufferArg::from_raw_parts(bins_handle.clone(), bins),
        );
    }
    let bytes = client
        .read_one(bins_handle)
        .map_err(|e| format!("kernel or read-back failed: {e}"))?;
    let millis = millis_since(start);

    let got = u32::from_bytes(&bytes);
    let got = got
        .get(..bins)
        .ok_or_else(|| format!("read back {} bins, fewer than {bins}", got.len()))?;
    let mismatch = if got.len() != want.len() {
        Some(format!("{} bins, want {}", got.len(), want.len()))
    } else {
        got.iter()
            .zip(&want)
            .enumerate()
            .find(|(_, (g, w))| g != w)
            .map(|(bin, (g, w))| format!("bin {bin}: got {g}, want {w}"))
    };
    Ok((millis, mismatch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_inputs_are_deterministic_and_fractional() {
        let (a1, b1) = add_inputs::<f64>(10_000);
        let (a2, b2) = add_inputs::<f64>(10_000);
        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert!(a1.iter().any(|x| x.fract() != 0.0));
        assert!(b1.iter().any(|x| x.fract() != 0.0));
        let (a32, _) = add_inputs::<f32>(100);
        assert_eq!(a32[3], (3.0f64 * 0.1 - 300.0) as f32);
    }

    #[test]
    fn host_add_rounds_like_ieee() {
        let (a, b) = add_inputs::<f32>(5000);
        let sum = host_add(&a, &b);
        for i in 0..a.len() {
            assert_eq!(sum[i].to_bits(), (a[i] + b[i]).to_bits());
        }
        // Many sums are inexact, so the comparison is meaningful.
        let inexact = (0..a.len())
            .filter(|&i| f64::from(sum[i]) != f64::from(a[i]) + f64::from(b[i]))
            .count();
        assert!(inexact > 100, "only {inexact} inexact sums");
        assert_eq!(host_add(&[0.1f64], &[0.2f64]), [0.1 + 0.2]);
    }

    #[test]
    fn histogram_reference_counts_every_value() {
        let values = histogram_input(100_000, HISTOGRAM_BINS);
        assert!(values.iter().all(|&v| (v as usize) < HISTOGRAM_BINS));
        let counts = host_histogram(&values, HISTOGRAM_BINS);
        assert_eq!(counts.iter().map(|&c| c as usize).sum::<usize>(), 100_000);
        assert!(counts.iter().all(|&c| c > 0), "every bin is hit");
        assert_eq!(host_histogram(&[0, 1, 1, 3, 9], 4), [1, 2, 0, 1]);
        assert_eq!(histogram_input(64, 8), histogram_input(64, 8));
    }

    #[test]
    fn first_mismatch_is_bitwise() {
        assert_eq!(first_mismatch(&[1.0f32, 2.0], &[1.0, 2.0]), None);
        let detail = first_mismatch(&[1.0f32, 2.0, 3.5], &[1.0, 2.5, 3.0]).expect("mismatch");
        assert!(detail.starts_with("2 of 3 elements differ; first at index 1"));
        // +0 and -0 compare equal but differ bitwise.
        assert!(first_mismatch(&[-0.0f64], &[0.0]).is_some());
        assert!(first_mismatch(&[f64::NAN], &[f64::NAN]).is_none());
        assert!(
            first_mismatch(&[1.0f64], &[1.0, 2.0])
                .expect("length")
                .contains("length")
        );
    }

    #[test]
    fn precision_conversions() {
        assert_eq!(<f32 as Precision>::from_u32(7), 7.0);
        assert_eq!(<f64 as Precision>::from_u32(u32::MAX), 4_294_967_295.0);
        assert_eq!(1.5f32.to_bits_u64(), u64::from(1.5f32.to_bits()));
        assert_eq!(Precision::add(0.5f64, 0.25), 0.75);
    }
}
