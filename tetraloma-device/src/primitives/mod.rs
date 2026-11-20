// Listing S1b.
//! Parallel algorithms on device buffers, re-exported at the crate root: fill and copy, gather
//! and scatter, pack and unpack, histogram, reductions, prefix sums, a stable radix sort,
//! run-length encoding and grouping of sorted keys.
//!
//! tetraloma builds its meshes with them, and you may use them on buffers of your own. Every
//! primitive takes its launch shape from one [`LaunchPlan`](crate::LaunchPlan), uses no plane
//! operations, no 64-bit atomics and no floating-point atomics, and gives the same bits on every
//! run on a device
//! ([Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#determinism)). An empty
//! input never launches a kernel.

use cubecl::prelude::*;

mod copy;
mod fill;
mod gather;
mod histogram;
mod pack;
mod radix_sort;
mod reduce;
mod scan;
mod segmented;
mod sort_keys;
mod unique;

pub use copy::copy;
pub use fill::fill;
pub(crate) use fill::launch_fill;
pub use gather::{gather, scatter};
pub use histogram::histogram;
pub use pack::{pack, unpack};
pub use radix_sort::{radix_sort, radix_sort_owned};
pub use reduce::reduce;
pub use scan::{exclusive_scan, inclusive_scan};
pub use segmented::segmented_reduce;
pub use sort_keys::SortKeys;
pub use unique::{Runs, Unique, run_length, unique_by_key, unique_by_key_runs};

mod sealed {
    /// Seals `DeviceScalar`. Its supertraits are the launch bounds the kernels need, which
    /// `DeviceScalar` thereby implies without naming them in its public declaration.
    pub trait Sealed: cubecl::frontend::LaunchArg<RuntimeArg = Self> {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
}

/// The scalar types of the arithmetic primitives: `f32`, `f64`, `i32` and `u32`.
///
/// The trait is sealed: no other type can implement it. [`reduce`], the prefix sums and
/// [`segmented_reduce`] combine values in a fixed order for each of these types
/// ([Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#determinism)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceScalar, ReduceOp, reduce};
///
/// fn total<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Vec<T> {
///     reduce(x, ReduceOp::Sum).unwrap().to_host().unwrap()
/// }
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// assert_eq!(total(&DeviceBuffer::from_host(&client, &[1i32, 2, 3]).unwrap()), vec![6]);
/// assert_eq!(total(&DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap()), vec![6]);
/// ```
pub trait DeviceScalar: CubeElement + Numeric + sealed::Sealed {}

impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}
impl DeviceScalar for i32 {}
impl DeviceScalar for u32 {}

/// The operation of a reduction: the sum, the minimum or the maximum.
///
/// [`reduce`], [`segmented_reduce`] and the collectives of tetraloma-parallel take it. Sums of
/// floating-point values are taken in a fixed order, so they give the same bits on every run on a
/// device ([Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#determinism)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, ReduceOp, reduce};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let x = DeviceBuffer::from_host(&client, &[3.0f32, -1.0, 2.0]).unwrap();
/// assert_eq!(reduce(&x, ReduceOp::Sum).unwrap().to_host().unwrap(), vec![4.0]);
/// assert_eq!(reduce(&x, ReduceOp::Min).unwrap().to_host().unwrap(), vec![-1.0]);
/// assert_eq!(reduce(&x, ReduceOp::Max).unwrap().to_host().unwrap(), vec![3.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    /// The sum, in a fixed order.
    Sum,
    /// The smallest value.
    Min,
    /// The largest value.
    Max,
}
