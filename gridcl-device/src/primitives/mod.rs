//! Device primitives (Listing S1b), re-exported at the crate root.
//!
//! Every primitive takes its launch shape from one `LaunchPlan` (one partition per primitive),
//! uses no plane operations, no 64-bit atomics and no float atomics, and is bitwise
//! deterministic. An empty input never launches a kernel.

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
pub use radix_sort::radix_sort;
pub use reduce::reduce;
pub use scan::{exclusive_scan, inclusive_scan};
pub use segmented::segmented_reduce;
pub use sort_keys::SortKeys;
pub use unique::{Runs, Unique, run_length, unique_by_key};

mod sealed {
    /// Seals `DeviceScalar`. Its supertraits are the launch bounds the kernels need, which
    /// `DeviceScalar` thereby implies without naming them in its public declaration.
    pub trait Sealed: cubecl::frontend::LaunchArg<RuntimeArg = Self> {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
}

/// The scalars of the typed primitives (sealed): f32, f64, i32, u32.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceScalar, ReduceOp, reduce};
///
/// fn total<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Vec<T> {
///     reduce(x, ReduceOp::Sum).unwrap().to_host().unwrap()
/// }
///
/// let client = gridcl_testing::test_client();
/// assert_eq!(total(&DeviceBuffer::from_host(&client, &[1i32, 2, 3]).unwrap()), vec![6]);
/// assert_eq!(total(&DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap()), vec![6]);
/// ```
pub trait DeviceScalar: CubeElement + Numeric + sealed::Sealed {}

impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}
impl DeviceScalar for i32 {}
impl DeviceScalar for u32 {}

/// Sum, min or max. Re-exported by gridcl-parallel, so `gridcl_parallel::ReduceOp` stays valid.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, ReduceOp, reduce};
///
/// let client = gridcl_testing::test_client();
/// let x = DeviceBuffer::from_host(&client, &[3.0f32, -1.0, 2.0]).unwrap();
/// assert_eq!(reduce(&x, ReduceOp::Sum).unwrap().to_host().unwrap(), vec![4.0]);
/// assert_eq!(reduce(&x, ReduceOp::Min).unwrap().to_host().unwrap(), vec![-1.0]);
/// assert_eq!(reduce(&x, ReduceOp::Max).unwrap().to_host().unwrap(), vec![3.0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
}
