//! Parallel I/O: Gmsh in; PVTU and, with the feature `hdf5`, VTKHDF out (the book, tutorial "Parallel I/O").

use std::path::Path;

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Reads a Gmsh file in parallel in `F` on `client` over `comm`, and writes it to `out_dir` (which must exist) as PVTU and, with the feature `hdf5`, as VTKHDF. Collective over `comm`.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real, C: Comm>(client: &Client, comm: C, out_dir: &Path) -> tetraloma::Result<()> {
    let _ = (client, comm, out_dir);
    todo!()
}
