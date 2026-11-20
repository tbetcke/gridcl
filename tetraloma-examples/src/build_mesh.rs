//! Building meshes from arrays (the book, tutorial "Building a mesh").

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds meshes from host arrays in `F` on `client`: one cell type, several cell types, and higher-degree geometry, and checks their sizes and measures.
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
