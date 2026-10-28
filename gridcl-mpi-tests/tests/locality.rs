//! M6 acceptance: on a randomly permuted structured box, locality ordering at least halves the
//! mean over cells of (max − min local node index in the cell).

use std::process::ExitCode;

use gridcl_testing::parallel;

fn main() -> ExitCode {
    gridcl_mpi_tests::run("locality", |comm| {
        let client = gridcl_testing::test_client();
        parallel::locality(comm, &client);
    })
}
