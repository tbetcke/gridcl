//! Iterating over blocks, cells, entities and connectivity (the book, tutorial "Iterating over a mesh").

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds a mesh in `F` on `client` and walks its blocks, cells, entities and connectivity, reading them back to the host for inspection.
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
