//! M10 bullet 6 (ADR-0060 item 6): every example's `run` on the test device
//! (`TETRALOMA_TEST_DEVICE`): the CPU runtime here and in CI, Metal and CUDA on the GPU targets.

use std::path::PathBuf;

use tetraloma::Serial;
use tetraloma_examples::{
    geometric_factors, halo_exchange, read_gmsh, surface_submesh, vtk_output,
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
