//! A distributed mesh: partitioners, ghost modes and cell ordering (the book, tutorial "Distributed meshes").

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds a distributed mesh in `F` on `client` over `comm` and prints its partition, ghosts and ordering. Collective over `comm`.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real, C: Comm>(client: &Client, comm: C) -> tetraloma::Result<()> {
    let _ = (client, comm);
    todo!()
}
