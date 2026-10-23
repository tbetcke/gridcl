//! The `DeviceBuffer` family, capability and limit checks, launch plans and the vector policy,
//! device primitives, pack/unpack kernels and GPU-aware pointer access.
//!
//! The only gridcl crate that uses CubeCL runtime types (docs/design.md, "Dependency rules" 3).

mod buffer;
mod error;
mod plan;
mod policy;
mod precision;
pub mod primitives;

#[cfg(feature = "testing")]
pub mod testing;
#[cfg(not(feature = "testing"))]
mod testing;

pub use buffer::{DeviceBuffer, DeviceCsr, DeviceStrided, VECTOR_PAD, pad_len};
pub use error::{ConnKey, GridError, Result};
pub use plan::{LaunchPlan, debug_assert_bindings};
pub use policy::{DeviceClass, KernelKind, VectorPolicy, VectorSizeOverride};
pub use precision::{Real, check_precision};
pub use primitives::*;
pub use testing::device_properties;
