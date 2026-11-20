//! M9 acceptance: `orient` recovers randomly flipped closed surfaces and curves with the flags
//! of a one-rank build, orients the open hemisphere `AwayFrom` and `Towards` a point, and gives
//! agreed `NeedsReferencePoint`, `NonOrientable` and `NonManifold` errors, for every orient
//! family at p = 1 and 2 in every ghost mode (docs/plans/M9.md, acceptance coverage).

use std::process::ExitCode;

use tetraloma_testing::parallel::{self, Coverage};

fn main() -> ExitCode {
    tetraloma_mpi_tests::run("orient_parallel", |comm| {
        let client = tetraloma_testing::test_client();
        for case in parallel::orient_cases(Coverage::Full) {
            parallel::orient(comm, &client, &case);
        }
    })
}
