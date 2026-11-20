//! A downstream CubeCL kernel over a mesh (the book, tutorial "Writing kernels").
//!
//! ```sh
//! cargo run --release -p tetraloma-examples --example custom_kernel
//! ```

use tetraloma_examples::{custom_kernel, device, has_f64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = device(0)?.client();
    if has_f64(&client) {
        custom_kernel::run::<f64>(&client)?;
    } else {
        custom_kernel::run::<f32>(&client)?;
    }
    Ok(())
}
