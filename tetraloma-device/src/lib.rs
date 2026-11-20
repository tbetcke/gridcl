// Design: "Dependency rules" 3 (the only tetraloma crate that uses CubeCL runtime types);
// Listings S1, S1b, S2, B5.
//! Device buffers, device limits, launch plans and parallel primitives: the device layer of
//! tetraloma.
//!
//! - [`DeviceBuffer`] is a typed array in device memory, with the explicit copies to and from
//!   the host; [`DeviceStrided`] and [`DeviceCsr`] hold the rows of a connectivity.
//! - [`Real`] is the scalar type of a mesh, `f32` or `f64`, and [`check_precision`] tells
//!   whether a device supports it.
//! - [`LaunchPlan`] chooses the launch shape of tetraloma's kernels from the device's limits and
//!   the built-in [`VectorPolicy`].
//! - [`primitives`] holds the deterministic device algorithms tetraloma is built on: fill,
//!   copy, gather and scatter, pack and unpack, reductions, prefix sums, a radix sort and
//!   run-length encoding.
//! - [`GridError`] and [`Result`] are the error and result types of every tetraloma crate.
//!
//! The facade crate `tetraloma` re-exports what a program needs from here. The chapter
//! [Compute devices](https://tetracl.github.io/tetraloma/devices.html) explains how to open a
//! device, which precisions it supports and when data moves between host and device.

mod buffer;
mod cores;
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
// ADR-0026, item 12: re-exported for tetraloma-parallel, which has no cubecl dependency.
/// CubeCL's client and element trait, re-exported: a [`Client`] drives one device, and
/// [`CubeElement`] is a type that device buffers can hold. The chapter [Compute
/// devices](https://tetracl.github.io/tetraloma/devices.html#choosing-a-device-at-run-time)
/// shows how to get a client.
pub use cubecl::prelude::{Client, CubeElement};
pub use error::{ConnKey, GridError, Result};
pub use plan::{LaunchPlan, debug_assert_bindings};
pub use policy::{CpuUnits, DeviceClass, KernelKind, VectorPolicy, VectorSizeOverride};
pub use precision::{Real, check_precision};
pub use primitives::*;
pub use testing::device_properties;
