//! Test hooks (ADR-0007). Only with the Cargo feature `testing`, which is off by default and not
//! part of the stable API; gridcl-testing turns it on for every workspace test.
//!
//! The overrides are scoped and thread-local, because gridcl runs one host thread per mesh.
//! Precedence: the scoped override, then the environment variable (`GRIDCL_VECTOR_SIZE`,
//! `GRIDCL_QUAD_TILE`), then vector-policy.toml.

#[cfg(feature = "testing")]
use cubecl::ir::DeviceProperties;

#[cfg(feature = "testing")]
use crate::{KernelKind, LaunchPlan, VectorSizeOverride};

/// Replaces the properties gridcl-device reads (limits, device class, vector sizes) for buffers
/// and plans made inside `f`; tests clone `client.properties()` and modify it.
#[cfg(feature = "testing")]
pub fn with_properties<R>(_props: DeviceProperties, _f: impl FnOnce() -> R) -> R {
    todo!()
}

/// Forces the vector size of plans made inside `f`.
#[cfg(feature = "testing")]
pub fn with_vector_size<R>(_v: VectorSizeOverride, _f: impl FnOnce() -> R) -> R {
    todo!()
}

/// Forces `VectorPolicy::quad_tile` inside `f`.
#[cfg(feature = "testing")]
pub fn with_quad_tile<R>(_tile: usize, _f: impl FnOnce() -> R) -> R {
    todo!()
}

/// The plans made inside `f`, so a test can assert the vector size it ran at.
#[cfg(feature = "testing")]
pub fn record_plans<R>(_f: impl FnOnce() -> R) -> (R, Vec<(KernelKind, LaunchPlan)>) {
    todo!()
}
