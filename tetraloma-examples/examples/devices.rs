//! Opens the device that the runtime features and `TETRALOMA_TEST_DEVICE` select and prints what
//! the book's chapter "Compute devices" describes.
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example devices
//! TETRALOMA_TEST_DEVICE=metal cargo run --release -p tetraloma-examples --features metal --example devices
//! ```

use tetraloma::prelude::*;
use tetraloma_examples::devices;

// ANCHOR: main
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The device named by the environment, or the first the build can open.
    let name = std::env::var(devices::DEVICE_VAR).unwrap_or_default();
    // One process, so GPU 0; under MPI, local_gpu_index(&comm) gives each rank its own.
    let gpu = local_gpu_index(&Serial);
    let client = devices::open(&name, gpu)?.client();

    // f64 where the device supports it, f32 otherwise.
    if tetraloma_examples::has_f64(&client) {
        devices::run::<f64>(&client)?;
    } else {
        devices::run::<f32>(&client)?;
    }
    Ok(())
}
// ANCHOR_END: main
