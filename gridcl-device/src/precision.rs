//! `Real` (Listing B3) and `check_precision` (Listing S1).

use cubecl::prelude::*;

use crate::{GridError, Result, testing};

/// Scalar of mesh geometry, re-exported by gridcl. Host arithmetic uses `widen`/`narrow` (or
/// f64 directly), never method calls on `Self`: CubeCL's `Float` and `num_traits::Float` share
/// method names (sqrt, abs, min, ...), and CubeCL's versions are device-only stubs that panic
/// on the host.
///
/// # Example
///
/// ```
/// use gridcl_device::Real;
///
/// fn mean<F: Real>(xs: &[F]) -> F {
///     F::narrow(xs.iter().map(|x| x.widen()).sum::<f64>() / xs.len() as f64)
/// }
/// assert_eq!(mean(&[1.0f32, 2.0, 6.0]), 3.0);
/// assert_eq!(0.1f32.widen(), 0.1f32 as f64);
/// ```
pub trait Real: Float + CubeElement {
    /// The value as f64 (exact).
    fn widen(self) -> f64;
    /// `x` rounded to `Self`.
    fn narrow(x: f64) -> Self;
}

impl Real for f32 {
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn narrow(x: f64) -> Self {
        x as f32
    }
}

impl Real for f64 {
    fn widen(self) -> f64 {
        self
    }
    fn narrow(x: f64) -> Self {
        x
    }
}

/// `Ok` if the device supports `F`, else `GridError::UnsupportedPrecision`. Called whenever a
/// mesh is created on a client, so precision errors surface early. An f64 request on Metal
/// fails.
///
/// # Example
///
/// ```
/// use gridcl_device::{GridError, check_precision};
///
/// let client = gridcl_testing::test_client();
/// check_precision::<f32>(&client).unwrap();
/// match check_precision::<f64>(&client) {
///     Ok(()) => {}
///     Err(GridError::UnsupportedPrecision { scalar, .. }) => assert_eq!(scalar, "f64"),
///     Err(e) => panic!("{e}"),
/// }
/// ```
pub fn check_precision<F: Real>(client: &Client) -> Result<()> {
    if testing::properties(client).supports_type(F::cube_type()) {
        return Ok(());
    }
    Err(GridError::UnsupportedPrecision {
        scalar: F::type_name(),
        device: client.name(),
    })
}
