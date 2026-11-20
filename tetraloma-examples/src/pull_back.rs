//! Pulling physical points back to reference coordinates (the book, tutorial "Pulling back points").

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Pulls points of a curved mesh in `F` on `client` back to reference coordinates and checks that the forward map returns them.
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
