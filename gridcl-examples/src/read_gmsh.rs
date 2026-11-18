//! Reading a Gmsh file and fixing the dimensions once (Listing B8).

use std::path::Path;

use cubecl::prelude::Client;
use gridcl::prelude::*;

/// Reads the volume mesh in `path` on `client`, in f64 where the device supports it and f32
/// otherwise, and prints its owned cells.
///
/// The example file `gridcl-examples/data/two_regions.msh` holds two regions of the box
/// [0, 2] × [0, 1]²: hexahedra (`hex`) in [0, 1]³, and tetrahedra with the pyramids beside the
/// hexahedra (`tet`) in [1, 2] × [0, 1]². `run` checks that the two regions cover the mesh and,
/// for that file, that its volume is 2.
///
/// # Errors
///
/// The errors of [`gmsh::read`], `DimensionMismatch` if the file holds no volume mesh in 3D,
/// `UnknownTag` if it has no `hex` or `tet` region; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run(client: &Client, path: &Path) -> gridcl::Result<()> {
    if crate::has_f64(client) {
        read::<f64>(client, path)
    } else {
        read::<f32>(client, path)
    }
}

fn read<F: Real>(client: &Client, path: &Path) -> gridcl::Result<()> {
    // The file decides D_TOP and D_GEO at run time ...
    let any: AnyMesh<F> = gmsh::read(client, Serial, path, GmshOptions::default())?;
    println!(
        "{}: tdim {}, gdim {}",
        path.display(),
        any.tdim(),
        any.gdim()
    );

    // ... the program fixes them once.
    let head: VolumeMesh<F> = any.try_into()?;
    let owned: usize = head
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.num_owned())
        .sum();
    println!(
        "{owned} owned cells on rank {}, running on {} in {}",
        head.comm().rank(),
        client.name(),
        crate::scalar_name::<F>()
    );
    for block in head.cell_blocks() {
        println!("  {:?}: {} cells", block.cell, block.index_map.num_owned());
    }

    // The physical groups of the cells, by name.
    let mut in_regions = 0;
    for name in ["hex", "tet"] {
        let region = head.tags().named(Codim::Vol, name)?;
        println!("  region {name:?}: {} cells", region.indices.len());
        in_regions += region.indices.len();
    }
    assert_eq!(in_regions, owned, "the regions do not cover the mesh");

    let volume = head.measure(1)?;
    println!("  volume {volume}");
    if path.ends_with("two_regions.msh") {
        let err = (volume - 2.0).abs() / 2.0;
        assert!(
            err <= 10.0 * crate::tol::<F>(),
            "volume {volume}, expected 2"
        );
    }
    Ok(())
}
