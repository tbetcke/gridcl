//! Reading a Gmsh file and fixing the dimensions once (Listing B8).

use std::path::Path;

use cubecl::prelude::Client;

/// Reads the volume mesh in `path` on `client`, in f64 where the device supports it and f32
/// otherwise, and prints its owned cells.
pub fn run(_client: &Client, _path: &Path) -> gridcl::Result<()> {
    todo!("M10 T11")
}
