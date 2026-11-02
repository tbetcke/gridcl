//! M8 acceptance, VTKHDF (docs/plans/M8.md, T0b; ADR-0035, rules 8 and 12): VTK reads the
//! VTKHDF file as the PVTU, and after it removes the ghost cells every cell appears exactly
//! once. Feature `hdf5` (a parallel HDF5; the file is written collectively through MPI-IO).

use std::path::Path;
use std::process::ExitCode;

use gridcl::{GhostMode, VtkWriter};
use gridcl_testing::io_parallel::{self as io, VtkFormat, remove_shared_dir, shared_dir};

fn vtkhdf(path: &Path) -> VtkWriter<'static> {
    VtkWriter::vtkhdf(path)
}

fn main() -> ExitCode {
    gridcl_mpi_tests::run("vtkhdf_parallel", |comm| {
        let client = gridcl_testing::test_client();
        let dir = shared_dir(comm, "vtkhdf-parallel");
        io::vtkhdf_equals_pvtu(comm, &client, vtkhdf, &dir);
        for ghosts in [
            GhostMode::SharedFacet,
            GhostMode::SharedVertex { layers: 1 },
        ] {
            io::ghosts_once(comm, &client, ghosts, VtkFormat::Vtkhdf(vtkhdf), &dir);
        }
        remove_shared_dir(comm, &dir);
    })
}
