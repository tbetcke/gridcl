//! M6 acceptance (ADR-0027): on a randomly permuted structured box, locality ordering at least
//! halves the mean over owned cells of (max − min local index of the cell's owned nodes).

use std::process::ExitCode;

use tetraloma_testing::parallel;

fn main() -> ExitCode {
    tetraloma_mpi_tests::run("locality", |comm| {
        let client = tetraloma_testing::test_client();
        parallel::locality(comm, &client);
    })
}
