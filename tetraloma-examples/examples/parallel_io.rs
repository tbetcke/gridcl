//! Parallel Gmsh input and PVTU and VTKHDF output under MPI (the book, tutorial "Parallel I/O").
//!
//! ```sh
//! cargo build --release -p tetraloma-examples --features mpi,hdf5 --example parallel_io
//! mpirun -n 4 target/release/examples/parallel_io [OUT_DIR]
//! ```
//!
//! Without an argument it writes to `target/tetraloma-examples/parallel_io` in the workspace.
//! Without the feature `hdf5` it writes PVTU only.

// ANCHOR: main
use std::path::PathBuf;

use tetraloma::prelude::*;
use tetraloma_examples::{device, has_f64, parallel_io};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let universe = mpi::initialize().ok_or("MPI already initialised")?; // one host thread per rank
    let comm = Mpi::world(&universe);
    tetraloma_examples::check_world_size(&comm)?; // stop if each rank was started alone
    let out_dir = std::env::args_os().nth(1).map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../target/tetraloma-examples/parallel_io"
            ))
        },
        PathBuf::from,
    );
    std::fs::create_dir_all(&out_dir)?;
    let gpu = local_gpu_index(&comm); // the node-local rank
    let client = device(gpu)?.client();
    if has_f64(&client) {
        parallel_io::run::<f64, _>(&client, comm, &out_dir)?;
    } else {
        parallel_io::run::<f32, _>(&client, comm, &out_dir)?;
    }
    Ok(())
}
// ANCHOR_END: main
