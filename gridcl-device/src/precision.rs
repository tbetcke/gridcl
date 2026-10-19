//! `Real` (Listing B3) and `check_precision` (Listing S1).

use cubecl::prelude::*;

use crate::Result;

/// Scalar of mesh geometry, re-exported by gridcl. Host arithmetic uses `widen`/`narrow` (or
/// f64 directly), never method calls on `Self`: CubeCL's `Float` and `num_traits::Float` share
/// method names (sqrt, abs, min, ...), and CubeCL's versions are device-only stubs that panic
/// on the host.
pub trait Real: Float + CubeElement {
    /// The value as f64 (exact).
    fn widen(self) -> f64;
    /// `x` rounded to `Self`.
    fn narrow(x: f64) -> Self;
}

impl Real for f32 {
    fn widen(self) -> f64 {
        todo!()
    }
    fn narrow(_x: f64) -> Self {
        todo!()
    }
}

impl Real for f64 {
    fn widen(self) -> f64 {
        todo!()
    }
    fn narrow(_x: f64) -> Self {
        todo!()
    }
}

/// `Ok` if the device supports `F`, else `GridError::UnsupportedPrecision`. Called whenever a
/// mesh is created on a client, so precision errors surface early.
pub fn check_precision<F: Real>(_client: &Client) -> Result<()> {
    todo!()
}
