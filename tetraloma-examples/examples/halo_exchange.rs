//! `forward` and `reverse` on a mesh's node index map (Listing B13), under MPI.
//!
//! ```sh
//! cargo build --release -p tetraloma-examples --features mpi --example halo_exchange
//! mpirun -n 2 target/release/examples/halo_exchange
//! ```
//!
//! Each rank opens the GPU of its node-local rank (`local_gpu_index`, ADR-0060) where the
//! build links a GPU runtime that takes an index (CUDA).

use tetraloma::prelude::*;
use tetraloma_examples::{device, halo_exchange};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().ok_or("MPI already initialised")?; // one host thread per rank
    let comm = Mpi::world(&universe);
    let gpu = local_gpu_index(&comm); // node-local rank (ADR-0026, ADR-0060)
    let client = device(gpu)?.client();
    halo_exchange::run(&client, comm)?;
    Ok(())
}
