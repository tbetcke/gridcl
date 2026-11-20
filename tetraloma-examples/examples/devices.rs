//! Opens the device that the runtime features select and prints what the book's chapter "Compute devices" describes.
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example devices
//! ```

use tetraloma_examples::{device, devices, has_f64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    if has_f64(&client) {
        devices::run::<f64>(&client)?;
    } else {
        devices::run::<f32>(&client)?;
    }
    Ok(())
}
