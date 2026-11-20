//! VTU output of a mesh and a point field (Listing B13).

use std::path::Path;

use cubecl::prelude::Client;
use tetraloma::prelude::*;
use tetraloma::vtk::VtkScalar;

/// Writes a mesh with a point field to `out_dir` on `client`, in f64 where the device supports
/// it and f32 otherwise.
///
/// The mesh is the unit sphere as 80 triangles of degree 2. `run` writes `scalp.vtu` with
/// Lagrange cells and the point field `position` (each node's coordinates), and
/// `scalp_linear.vtu` with linear cells through the erased path (`write_any`), as Listing B13
/// does. It checks that both files are VTK XML files. `out_dir` must exist.
///
/// # Errors
///
/// The errors of mesh creation and of [`VtkWriter::write`], `Io` if a written file cannot be
/// read back; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run(client: &Client, out_dir: &Path) -> tetraloma::Result<()> {
    if crate::has_f64(client) {
        write::<f64>(client, out_dir)
    } else {
        write::<f32>(client, out_dir)
    }
}

fn write<F: Real + VtkScalar>(client: &Client, out_dir: &Path) -> tetraloma::Result<()> {
    let scalp: SurfaceMesh<F> = tetraloma_testing::sphere(1, 2).build(client)?;
    println!(
        "sphere: {} triangles of degree 2 on {} in {}",
        scalp.cell_blocks()[0].index_map.num_owned(),
        client.name(),
        crate::scalar_name::<F>()
    );

    // ANCHOR: write
    // One vector per geometry node: its coordinates, with the mesh's origin added back.
    let origin = scalp.origin();
    let position: Vec<F> = scalp
        .nodes()
        .to_host()?
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|x| (0..3).map(move |g| F::narrow(x[g].widen() + origin[g])))
        .collect();
    let position = DeviceBuffer::from_host(scalp.client(), &position)?;

    let lagrange = out_dir.join("scalp");
    let linear = out_dir.join("scalp_linear");
    VtkWriter::vtu(&lagrange)
        .lagrange(true)
        .point_data("position", &position, 3)
        .write(&scalp)?; // generic over TD, GD
    VtkWriter::vtu(&linear)
        .lagrange(false)
        .write_any(AnyMeshRef::Surface(&scalp))?; // erased path, no clone
    // ANCHOR_END: write

    for stem in [&lagrange, &linear] {
        let path = stem.with_extension("vtu");
        let bytes =
            std::fs::read(&path).map_err(|e| GridError::Io(format!("{}: {e}", path.display())))?;
        assert!(
            bytes.starts_with(b"<?xml") || bytes.starts_with(b"<VTKFile"),
            "{} is not a VTK XML file",
            path.display()
        );
        println!("  wrote {} ({} bytes)", path.display(), bytes.len());
    }
    Ok(())
}
