//! VTU output of a mesh and a point field (Listing B13).

use std::path::Path;

use cubecl::prelude::Client;

/// Writes a mesh with a point field to `out_dir` on `client`, in f64 where the device supports
/// it and f32 otherwise.
pub fn run(_client: &Client, _out_dir: &Path) -> gridcl::Result<()> {
    todo!("M10 T11")
}
