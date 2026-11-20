//! Geometric factors at p = 1, 4 and 8 on a curved sphere.
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example geometric_factors
//! ```

use tetraloma_examples::{device, geometric_factors};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    geometric_factors::run(&client)?;
    Ok(())
}
