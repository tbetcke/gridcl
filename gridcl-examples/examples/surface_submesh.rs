//! A boundary surface mesh with its normals, Jacobians and area (Listing B9).
//!
//! ```sh
//! cargo run --release -p gridcl-examples --example surface_submesh
//! ```

use gridcl_examples::{device, surface_submesh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    surface_submesh::run(&client)?;
    Ok(())
}
