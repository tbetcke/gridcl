//! `Real` (Listing B3) and `check_precision` (Listing S1).

use cubecl::prelude::*;

use crate::{GridError, Result, testing};

// Listing B3.
/// The scalar type of a mesh: `f32` or `f64`.
///
/// Write code that works in both precisions generic over `F: Real`. On the host, compute with
/// `f64`: [`widen`](Real::widen) converts a value to `f64` exactly, and
/// [`narrow`](Real::narrow) rounds an `f64` back to `F`. Do not call arithmetic methods such as
/// `sqrt`, `abs` or `min` on `F` in host code. `F` gets them from CubeCL's `Float`, whose
/// versions are meant for kernels and panic on the host.
/// [Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#f32-and-f64)
/// explains what changes between the two types.
///
/// # Example
///
/// ```
/// use tetraloma_device::Real;
///
/// fn mean<F: Real>(xs: &[F]) -> F {
///     F::narrow(xs.iter().map(|x| x.widen()).sum::<f64>() / xs.len() as f64)
/// }
/// assert_eq!(mean(&[1.0f32, 2.0, 6.0]), 3.0);
/// assert_eq!(0.1f32.widen(), 0.1f32 as f64);
/// ```
pub trait Real: Float + CubeElement {
    /// The value as `f64`, exactly: every `f32` is an `f64` too. See
    /// [Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#f32-and-f64).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::Real;
    ///
    /// assert_eq!(0.1f32.widen(), f64::from(0.1f32));
    /// assert_eq!(0.1f64.widen(), 0.1);
    /// ```
    fn widen(self) -> f64;
    /// `x` rounded to the nearest value of `Self`: unchanged for `f64`, rounded for `f32`. See
    /// [Precision](https://tetracl.github.io/tetraloma/concepts/precision.html#f32-and-f64).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::Real;
    ///
    /// assert_eq!(f32::narrow(0.1), 0.1f32);
    /// assert_eq!(f64::narrow(0.1), 0.1);
    /// // A round trip through f64 is exact.
    /// assert_eq!(f32::narrow(1.5f32.widen()), 1.5);
    /// ```
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

// Listing S1.
/// Checks that `client`'s device supports the scalar type `F`.
///
/// tetraloma calls it whenever a mesh is created, so a mesh of an unsupported type fails before
/// any kernel runs. Call it yourself to choose `F` for a device: Metal has no `f64`, the CPU
/// runtime and CUDA have both. [Compute
/// devices](https://tetracl.github.io/tetraloma/devices.html#f64-support) lists the backends.
///
/// # Errors
///
/// `GridError::UnsupportedPrecision`, naming the type and the device, if the device does not
/// support `F`.
///
/// # Example
///
/// ```
/// use tetraloma_device::{GridError, check_precision};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
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
