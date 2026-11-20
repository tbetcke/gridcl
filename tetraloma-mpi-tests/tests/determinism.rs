//! M6 acceptance: two runs of the pipeline, entity creation and the scatters give bitwise
//! identical device arrays on every rank.

use std::process::ExitCode;

use tetraloma_testing::parallel;

fn main() -> ExitCode {
    tetraloma_mpi_tests::run("determinism", |comm| {
        let client = tetraloma_testing::test_client();
        for case in parallel::determinism_cases() {
            parallel::determinism(comm, &client, &case);
        }
    })
}
