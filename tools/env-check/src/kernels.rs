//! Sample kernels and their host references.
//!
//! The element-wise add runs at vector size 1 and at the widest IO-optimized size;
//! the u32 histogram uses `Atomic<u32>::fetch_add`. Results are compared bitwise
//! with the host. The cube count comes from `calculate_cube_count_elemwise`
//! (docs/plans/E0.md, T1: `LaunchPlan` arrives in M2, and env-check is not library code).

use cubecl::prelude::Client;

use crate::report::KernelCheck;

/// Element count of the sample kernels (10^7).
pub const DEFAULT_ELEMENTS: usize = 10_000_000;

/// Bins of the atomic histogram.
pub const HISTOGRAM_BINS: usize = 256;

/// A floating-point type the element-wise add runs in.
pub trait Precision: Copy + PartialEq + std::fmt::Debug {
    /// `f32` or `f64`.
    const NAME: &'static str;
    /// Size in bytes, for `io_optimized_vector_sizes`.
    const BYTES: usize;
    /// Exact conversion of a small integer.
    fn from_u32(value: u32) -> Self;
    fn add(self, other: Self) -> Self;
    fn to_bits_u64(self) -> u64;
}

impl Precision for f32 {
    const NAME: &'static str = "f32";
    const BYTES: usize = 4;
    fn from_u32(value: u32) -> Self {
        let _ = value;
        todo!()
    }
    fn add(self, other: Self) -> Self {
        let _ = other;
        todo!()
    }
    fn to_bits_u64(self) -> u64 {
        todo!()
    }
}

impl Precision for f64 {
    const NAME: &'static str = "f64";
    const BYTES: usize = 8;
    fn from_u32(value: u32) -> Self {
        let _ = value;
        todo!()
    }
    fn add(self, other: Self) -> Self {
        let _ = other;
        todo!()
    }
    fn to_bits_u64(self) -> u64 {
        todo!()
    }
}

/// The two inputs of the element-wise add, deterministic and not all integers.
pub fn add_inputs<F: Precision>(elements: usize) -> (Vec<F>, Vec<F>) {
    let _ = elements;
    todo!()
}

/// Host reference of the element-wise add.
pub fn host_add<F: Precision>(a: &[F], b: &[F]) -> Vec<F> {
    let _ = (a, b);
    todo!()
}

/// Deterministic histogram input with values in `0..bins`.
pub fn histogram_input(elements: usize, bins: usize) -> Vec<u32> {
    let _ = (elements, bins);
    todo!()
}

/// Host reference of the histogram.
pub fn host_histogram(values: &[u32], bins: usize) -> Vec<u32> {
    let _ = (values, bins);
    todo!()
}

/// Index and values of the first element where `got` and `want` differ bitwise,
/// or a length mismatch.
pub fn first_mismatch<F: Precision>(got: &[F], want: &[F]) -> Option<String> {
    let _ = (got, want);
    todo!()
}

/// The vector sizes the element-wise add runs at: 1 and the widest IO-optimized size.
pub fn add_vector_sizes(client: &Client, elem_bytes: usize) -> Vec<usize> {
    let _ = (client, elem_bytes);
    todo!()
}

/// Runs the element-wise add in precision `F` at `vector_size` and compares it with the host.
///
/// `elements` must be a multiple of `vector_size`.
pub fn check_elementwise_add<F: Precision>(
    client: &Client,
    elements: usize,
    vector_size: usize,
) -> KernelCheck {
    let _ = (client, elements, vector_size);
    todo!()
}

/// Runs the u32 atomic histogram and compares it with the host.
pub fn check_atomic_histogram(client: &Client, elements: usize, bins: usize) -> KernelCheck {
    let _ = (client, elements, bins);
    todo!()
}
