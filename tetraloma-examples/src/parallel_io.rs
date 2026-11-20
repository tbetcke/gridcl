//! Parallel I/O: Gmsh in; PVTU and, with the feature `hdf5`, VTKHDF out (the book, tutorial
//! "Parallel I/O").
//!
//! The input is `tetraloma-examples/data/two_regions.msh`, an ASCII Gmsh file of the box
//! [0, 2] × [0, 1]² in 52 cells: 8 hexahedra in [0, 1]³, and 40 tetrahedra and 4 pyramids in
//! [1, 2] × [0, 1]². The program reads it on every rank, writes it back as one binary Gmsh file
//! and reads that, which takes the byte-range path of the parallel reader, and writes the mesh
//! as PVTU and, with the feature `hdf5`, as VTKHDF.

use std::path::Path;

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// The example file: the box [0, 2] × [0, 1]² in two regions, in ASCII.
const INPUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");

/// The cells of the example file: 8 hexahedra, 40 tetrahedra and 4 pyramids.
const CELLS: u64 = 52;

/// Reads a Gmsh file in parallel in `F` on `client` over `comm`, and writes it to `out_dir`
/// (which must exist) as PVTU and, with the feature `hdf5`, as VTKHDF. Collective over `comm`.
///
/// `run` reads `data/two_regions.msh` with the [`Block`] partitioner and one layer of ghost
/// cells across facets, writes it to `out_dir/two_regions_binary.msh` as binary Gmsh (gathered
/// on rank 0) and reads that file back. After each read it checks that the ranks own the 52
/// cells once between them and that the volume is 2. It then writes `out_dir/two_regions.pvtu`
/// with one piece per rank and, with the feature `hdf5`, `out_dir/two_regions.vtkhdf`, each
/// with the cell data `owner` (the rank that owns each cell) and `OriginId`, and checks on rank
/// 0 that the files are there.
///
/// # Errors
///
/// The errors of [`gmsh::read`], of the Gmsh and VTK writers and of `measure`;
/// `DimensionMismatch` if the file holds no volume mesh in 3D; allocation, device and
/// communication errors.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real, C: Comm>(client: &Client, comm: C, out_dir: &Path) -> tetraloma::Result<()> {
    let (rank, size) = (comm.rank(), comm.size());
    if rank == 0 {
        println!(
            "parallel I/O on {size} rank(s), on {} in {}",
            client.name(),
            crate::scalar_name::<F>()
        );
    }

    // ANCHOR: read
    // Every rank calls `read`; each ends up with its owned cells and their ghosts.
    let options = GmshOptions::default()
        .partitioner(Block)
        .ghosts(GhostMode::SharedFacet);
    let any: AnyMesh<F, C> = gmsh::read(client, comm.clone(), INPUT, options.clone())?;
    let mesh: VolumeMesh<F, C> = any.try_into()?;
    // ANCHOR_END: read
    check(&mesh, "ASCII")?;

    // ANCHOR: binary
    // Rank 0 writes the gathered mesh as binary Gmsh; then every rank reads its byte range.
    let binary = out_dir.join("two_regions_binary.msh");
    gmsh::GmshWriter::new(&binary).binary(true).write(&mesh)?;
    let any = gmsh::read::<F, C>(client, comm.clone(), &binary, options)?;
    let mesh: VolumeMesh<F, C> = any.try_into()?;
    // ANCHOR_END: binary
    check(&mesh, "binary")?;

    // ANCHOR: owner
    // The rank that owns each local cell, in the writers' cell order: block by block, the
    // owned cells (this rank) and then the ghost cells (their owners).
    let mut owner: Vec<i32> = Vec::new();
    for block in mesh.cell_blocks() {
        let map = &block.index_map;
        owner.extend(std::iter::repeat_n(rank as i32, map.num_owned()));
        owner.extend_from_slice(map.ghost_owners());
    }
    let owner = DeviceBuffer::from_host(mesh.client(), &owner)?;
    // ANCHOR_END: owner

    // ANCHOR: pvtu
    // two_regions.pvtu (rank 0) and one piece two_regions_<rank>.vtu per rank.
    VtkWriter::pvtu(out_dir.join("two_regions"))
        .origin_ids(true)
        .cell_data("owner", &owner, 1)
        .write(&mesh)?;
    // ANCHOR_END: pvtu
    let ghosts: usize = mesh
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.num_ghosts())
        .sum();
    let any_ghosts = mesh.comm().sum(ghosts as u64) > 0;
    if rank == 0 {
        let header = std::fs::read_to_string(out_dir.join("two_regions.pvtu"))
            .expect("rank 0 wrote the PVTU header");
        for r in 0..size {
            assert!(
                header.contains(&format!("<Piece Source=\"two_regions_{r}.vtu\"/>")),
                "the header does not name the piece of rank {r}"
            );
            assert!(
                out_dir.join(format!("two_regions_{r}.vtu")).is_file(),
                "rank {r} wrote no piece"
            );
        }
        // ParaView hides the ghost cells of a header with GhostLevel 1.
        let level = u8::from(any_ghosts);
        assert!(
            header.contains(&format!("GhostLevel=\"{level}\"")),
            "the header's GhostLevel is not {level}"
        );
        println!(
            "wrote {} with {size} piece(s)",
            out_dir.join("two_regions.pvtu").display()
        );
    }

    #[cfg(feature = "hdf5")]
    {
        // ANCHOR: vtkhdf
        // One file two_regions.vtkhdf, with one partition per rank.
        VtkWriter::vtkhdf(out_dir.join("two_regions"))
            .origin_ids(true)
            .cell_data("owner", &owner, 1)
            .write(&mesh)?;
        // ANCHOR_END: vtkhdf
        if rank == 0 {
            let path = out_dir.join("two_regions.vtkhdf");
            let bytes = std::fs::read(&path).expect("the VTKHDF file was written");
            assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n", "not an HDF5 file");
            println!("wrote {}", path.display());
        }
    }
    Ok(())
}

/// Checks that `mesh`, read from the `what` file, is the whole box: the ranks own the 52 cells
/// once between them, and the volume is 2. Prints this rank's owned and ghost cells.
fn check<F: Real, C: Comm>(mesh: &VolumeMesh<F, C>, what: &str) -> tetraloma::Result<()> {
    let comm = mesh.comm();
    let blocks = mesh.cell_blocks();
    let owned: usize = blocks.iter().map(|b| b.index_map.num_owned()).sum();
    let ghosts: usize = blocks.iter().map(|b| b.index_map.num_ghosts()).sum();
    println!(
        "rank {}: {what} file read, {owned} owned and {ghosts} ghost cells",
        comm.rank()
    );
    assert_eq!(
        comm.sum(owned as u64),
        CELLS,
        "the ranks do not own each cell once"
    );
    let volume = mesh.measure(1)?; // owned cells only, summed over the ranks
    assert!(
        (volume - 2.0).abs() / 2.0 <= 10.0 * crate::tol::<F>(),
        "volume {volume}, expected 2"
    );
    Ok(())
}
