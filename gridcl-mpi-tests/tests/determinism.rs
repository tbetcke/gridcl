//! M6 acceptance: two runs of the pipeline, entity creation and the scatters give bitwise
//! identical device arrays on every rank.

use std::process::ExitCode;

use gridcl_testing::parallel;

fn main() -> ExitCode {
    gridcl_mpi_tests::run("determinism", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::determinism_cases() {
            parallel::determinism(comm, &client, &case);
        }
    })
}
