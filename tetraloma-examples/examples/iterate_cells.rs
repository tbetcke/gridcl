//! Iterates over blocks, cells, entities and connectivity (the book, tutorial "Iterating over a mesh").
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example iterate_cells
//! ```

use tetraloma_examples::{device, has_f64, iterate_cells};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    if has_f64(&client) {
        iterate_cells::run::<f64>(&client)?;
    } else {
        iterate_cells::run::<f32>(&client)?;
    }
    Ok(())
}
