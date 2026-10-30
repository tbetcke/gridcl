//! M6 acceptance: a forward scatter of global ids reproduces the ghost global ids, and a
//! reverse Add of ones equals the number of ranks holding each index, for the cell, vertex, node
//! and entity maps at block sizes 1 and 3.

use std::process::ExitCode;

use gridcl_testing::parallel;

fn main() -> ExitCode {
    gridcl_mpi_tests::run("scatter", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::scatter_cases() {
            parallel::scatter(comm, &client, &case);
        }
    })
}
