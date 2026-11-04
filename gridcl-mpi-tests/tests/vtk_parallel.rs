//! M8 acceptance, VTU and PVTU (docs/plans/M8.md, T0b; ADR-0035, rule 8): after VTK removes the
//! ghost cells of the PVTU every cell appears exactly once (shared-facet and one-layer
//! shared-vertex ghosts), the gathered VTU equals the one-rank VTU after sorting by origin id,
//! and failed writes are agreed. The oracle checks run on rank 0 unless `GRIDCL_SKIP_PYTHON=1`.

use std::process::ExitCode;

use gridcl::GhostMode;
use gridcl_testing::io_parallel::{self as io, VtkFormat, remove_shared_dir, shared_dir};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("vtk_parallel", |comm| {
        let client = gridcl_testing::test_client();
        let dir = shared_dir(comm, "vtk-parallel");
        for ghosts in [
            GhostMode::SharedFacet,
            GhostMode::SharedVertex { layers: 1 },
        ] {
            io::ghosts_once(comm, &client, ghosts, VtkFormat::Pvtu, &dir);
        }
        io::vtu_gathered_equals_serial(comm, &client, &dir);
        io::collective_write_errors(comm, &client, &dir);
        remove_shared_dir(comm, &dir);
    })
}
