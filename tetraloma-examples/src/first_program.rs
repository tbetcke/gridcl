//! A first program: a mesh, its geometry and a VTU file on one device (the book, "A first program").

use std::path::Path;

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds a small mesh in `F` on `client`, computes its geometric data and writes a VTU file to `out_dir` (which must exist).
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client, out_dir: &Path) -> tetraloma::Result<()> {
    let _ = (client, out_dir);
    todo!()
}
