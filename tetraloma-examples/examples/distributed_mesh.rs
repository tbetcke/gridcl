//! A distributed mesh under MPI (the book, tutorial "Distributed meshes").
//!
//! ```sh
//! cargo build --release -p tetraloma-examples --features mpi --example distributed_mesh
//! mpirun -n 4 target/release/examples/distributed_mesh
//! ```
//!
//! Each rank opens the GPU of its node-local rank (`local_gpu_index`) where the build links a
//! GPU runtime that takes an index (CUDA).

// ANCHOR: main
use tetraloma::prelude::*;
use tetraloma_examples::{device, distributed_mesh, has_f64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().ok_or("MPI already initialised")?; // one host thread per rank
    let comm = Mpi::world(&universe);
    let gpu = local_gpu_index(&comm); // the node-local rank
    let client = device(gpu)?.client();
    if has_f64(&client) {
        distributed_mesh::run::<f64, _>(&client, comm)?;
    } else {
        distributed_mesh::run::<f32, _>(&client, comm)?;
    }
    Ok(())
}
// ANCHOR_END: main
