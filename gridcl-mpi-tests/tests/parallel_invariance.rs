//! M6 acceptance: global entity counts, tag counts and total volume, canonical entity vertex
//! lists and permutation bits equal one rank's, for every test-mesh family at p = 1 and 2, in
//! every ghost mode and input distribution (docs/plans/M6.md, acceptance coverage).

use std::process::ExitCode;

use gridcl_testing::parallel::{self, Coverage};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("parallel_invariance", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::cases(Coverage::Full) {
            parallel::invariance(comm, &client, &case);
        }
    })
}
