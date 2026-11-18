//! VTU output of a mesh and a point field (Listing B13).
//!
//! ```sh
//! cargo run --release -p gridcl-examples --example vtk_output [-- OUT_DIR]
//! ```
//!
//! Without an argument it writes to `target/gridcl-examples/vtk_output` in the workspace.

use std::path::PathBuf;

use gridcl_examples::{device, vtk_output};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args_os().nth(1).map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../target/gridcl-examples/vtk_output"
            ))
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&out_dir)?;
    let client = device(0)?.client();
    vtk_output::run(&client, &out_dir)?;
    Ok(())
}
