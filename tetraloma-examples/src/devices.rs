//! The device a build opens, and what the book's chapter "Compute devices" describes of it.

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Prints the runtime, the name and the properties of `client`'s device (f64 support, plane size, memory), and runs a small mesh in `F` on it.
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
