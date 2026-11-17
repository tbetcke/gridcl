//! Reads a Gmsh file and fixes the dimensions once (Listing B8).
//!
//! ```sh
//! cargo run --release -p gridcl-examples --example read_gmsh [-- FILE.msh]
//! ```
//!
//! Without an argument it reads `gridcl-examples/data/two_regions.msh`.

use std::path::PathBuf;

use gridcl_examples::{device, read_gmsh};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os().nth(1).map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh")),
        PathBuf::from,
    );
    let client = device(0)?.client();
    read_gmsh::run(&client, &path)?;
    Ok(())
}
