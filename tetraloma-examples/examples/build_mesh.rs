//! Builds meshes from arrays (the book, tutorial "Building a mesh").
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example build_mesh
//! ```

use tetraloma_examples::{build_mesh, device, has_f64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    if has_f64(&client) {
        build_mesh::run::<f64>(&client)?;
    } else {
        build_mesh::run::<f32>(&client)?;
    }
    Ok(())
}
