//! M10 bullet 6 (ADR-0060 item 6) and D1 (ADR-0067): every example's `run` on the test device
//! (`TETRALOMA_TEST_DEVICE`): the CPU runtime here and in CI, Metal and CUDA on the GPU targets.
//! The book's programs run in f32 and, where the device supports it, f64. Also: the committed
//! figures of the book are what the generator writes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use cubecl::prelude::Client;
use tetraloma::Serial;
use tetraloma_examples::{
    build_mesh, custom_kernel, devices, distributed_mesh, figures, first_program,
    geometric_factors, halo_exchange, has_f64, iterate_cells, parallel_io, pull_back, read_gmsh,
    surface_submesh, vtk_output,
};
use tetraloma_testing::test_client;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// An empty output directory of `name` under the target directory.
fn out_dir(name: &str) -> PathBuf {
    let dir = root().join("../target/tetraloma-examples-tests").join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the output directory");
    dir
}

#[test]
fn read_gmsh_runs() {
    let path = root().join("data/two_regions.msh");
    read_gmsh::run(&test_client(), &path).expect("read_gmsh");
}

#[test]
fn surface_submesh_runs() {
    surface_submesh::run(&test_client()).expect("surface_submesh");
}

#[test]
fn geometric_factors_runs() {
    geometric_factors::run(&test_client()).expect("geometric_factors");
}

#[test]
fn halo_exchange_runs() {
    halo_exchange::run(&test_client(), Serial).expect("halo_exchange");
}

#[test]
fn vtk_output_runs() {
    let dir = out_dir("vtk_output");
    vtk_output::run(&test_client(), &dir).expect("vtk_output");
    let written: Vec<_> = std::fs::read_dir(&dir).expect("read the output").collect();
    assert!(
        !written.is_empty(),
        "vtk_output wrote nothing to {}",
        dir.display()
    );
}

/// The test client if its device supports f64; `None`, with a note, otherwise.
fn f64_client(test: &str) -> Option<Client> {
    let client = test_client();
    if has_f64(&client) {
        Some(client)
    } else {
        eprintln!("{test}: skipped, {} has no f64", client.name());
        None
    }
}

#[test]
fn first_program_runs_f32() {
    let client = test_client();
    let dir = out_dir("first_program_runs_f32");
    first_program::run::<f32>(&client, &dir).expect("first_program");
}

#[test]
fn first_program_runs_f64() {
    if let Some(client) = f64_client("first_program_runs_f64") {
        let dir = out_dir("first_program_runs_f64");
        first_program::run::<f64>(&client, &dir).expect("first_program");
    }
}

#[test]
fn devices_runs_f32() {
    let client = test_client();
    devices::run::<f32>(&client).expect("devices");
}

#[test]
fn devices_runs_f64() {
    if let Some(client) = f64_client("devices_runs_f64") {
        devices::run::<f64>(&client).expect("devices");
    }
}

#[test]
fn build_mesh_runs_f32() {
    let client = test_client();
    build_mesh::run::<f32>(&client).expect("build_mesh");
}

#[test]
fn build_mesh_runs_f64() {
    if let Some(client) = f64_client("build_mesh_runs_f64") {
        build_mesh::run::<f64>(&client).expect("build_mesh");
    }
}

#[test]
fn iterate_cells_runs_f32() {
    let client = test_client();
    iterate_cells::run::<f32>(&client).expect("iterate_cells");
}

#[test]
fn iterate_cells_runs_f64() {
    if let Some(client) = f64_client("iterate_cells_runs_f64") {
        iterate_cells::run::<f64>(&client).expect("iterate_cells");
    }
}

#[test]
fn pull_back_runs_f32() {
    let client = test_client();
    pull_back::run::<f32>(&client).expect("pull_back");
}

#[test]
fn pull_back_runs_f64() {
    if let Some(client) = f64_client("pull_back_runs_f64") {
        pull_back::run::<f64>(&client).expect("pull_back");
    }
}

#[test]
fn custom_kernel_runs_f32() {
    let client = test_client();
    custom_kernel::run::<f32>(&client).expect("custom_kernel");
}

#[test]
fn custom_kernel_runs_f64() {
    if let Some(client) = f64_client("custom_kernel_runs_f64") {
        custom_kernel::run::<f64>(&client).expect("custom_kernel");
    }
}

#[test]
fn distributed_mesh_runs_f32() {
    let client = test_client();
    distributed_mesh::run::<f32, _>(&client, Serial).expect("distributed_mesh");
}

#[test]
fn distributed_mesh_runs_f64() {
    if let Some(client) = f64_client("distributed_mesh_runs_f64") {
        distributed_mesh::run::<f64, _>(&client, Serial).expect("distributed_mesh");
    }
}

#[test]
fn parallel_io_runs_f32() {
    let client = test_client();
    let dir = out_dir("parallel_io_runs_f32");
    parallel_io::run::<f32, _>(&client, Serial, &dir).expect("parallel_io");
}

#[test]
fn parallel_io_runs_f64() {
    if let Some(client) = f64_client("parallel_io_runs_f64") {
        let dir = out_dir("parallel_io_runs_f64");
        parallel_io::run::<f64, _>(&client, Serial, &dir).expect("parallel_io");
    }
}

/// The committed figures of the book (docs/user/src/figures/*.svg) are what
/// `figures::svg_files` writes, with no file missing and none left over. After a change, run
/// `cargo run -p tetraloma-examples --bin figures`.
#[test]
fn figures_are_generated() {
    let dir = root().join("../docs/user/src/figures");
    let generated: BTreeMap<String, String> = figures::svg_files().into_iter().collect();
    let mut committed = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries {
            let path = entry.expect("read the figures directory").path();
            if path.extension().is_some_and(|e| e == "svg") {
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                let text = std::fs::read_to_string(&path).expect("read a figure");
                committed.insert(name, text);
            }
        }
    }
    assert!(!generated.is_empty(), "the generator writes no figures");
    let stale: Vec<_> = generated
        .iter()
        .filter(|(name, svg)| committed.get(*name) != Some(*svg))
        .map(|(name, _)| name.as_str())
        .collect();
    let extra: Vec<_> = committed
        .keys()
        .filter(|name| !generated.contains_key(*name))
        .collect();
    assert!(
        stale.is_empty() && extra.is_empty(),
        "figures out of date (run `cargo run -p tetraloma-examples --bin figures`): \
         missing or different {stale:?}, not generated {extra:?}"
    );
}
