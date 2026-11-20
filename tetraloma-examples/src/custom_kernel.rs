//! A downstream CubeCL kernel over a mesh (the book, tutorial "Writing kernels").

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Integrates a function over a mesh in `F` on `client` with a CubeCL kernel over `DeviceGrid` and `GeometryBlock`, and checks the result against `measure`.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client) -> tetraloma::Result<()> {
    let _ = client;
    todo!()
}
