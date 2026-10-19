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
mod radix_sort;
mod reduce;
mod scan;
mod segmented;
mod sort_keys;
mod unique;

pub use copy::copy;
pub use fill::fill;
pub use gather::{gather, scatter};
pub use histogram::histogram;
pub use radix_sort::radix_sort;
pub use reduce::reduce;
pub use scan::{exclusive_scan, inclusive_scan};
pub use segmented::segmented_reduce;
pub use sort_keys::SortKeys;
pub use unique::{Runs, Unique, run_length, unique_by_key};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
}

/// The scalars of the typed primitives (sealed): f32, f64, i32, u32.
pub trait DeviceScalar: CubeElement + Numeric + sealed::Sealed {}

impl DeviceScalar for f32 {}
impl DeviceScalar for f64 {}
impl DeviceScalar for i32 {}
impl DeviceScalar for u32 {}

/// Sum, min or max. Re-exported by gridcl-parallel, so `gridcl_parallel::ReduceOp` stays valid.
#[derive(Clone, Copy, Debug)]
pub enum ReduceOp {
    Sum,
    Min,
    Max,
}
