//! Pulls physical points back to reference coordinates (the book, tutorial "Pulling back points").
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example pull_back
//! ```

use tetraloma_examples::{device, has_f64, pull_back};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    if has_f64(&client) {
        pull_back::run::<f64>(&client)?;
    } else {
        pull_back::run::<f32>(&client)?;
    }
    Ok(())
}
