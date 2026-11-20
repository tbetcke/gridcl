//! A first program: a mesh, its geometry and a VTU file (the book, "A first program").
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example first_program [-- OUT_DIR]
//! ```
//!
//! Without an argument it writes to `target/tetraloma-examples/first_program` in the workspace.

use std::path::PathBuf;

use tetraloma_examples::{device, first_program, has_f64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args_os().nth(1).map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../target/tetraloma-examples/first_program"
            ))
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&out_dir)?;
    let client = device(0)?.client();
    if has_f64(&client) {
        first_program::run::<f64>(&client, &out_dir)?;
    } else {
        first_program::run::<f32>(&client, &out_dir)?;
    }
    Ok(())
}
