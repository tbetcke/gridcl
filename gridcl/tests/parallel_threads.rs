//! The M6 acceptance checks on `ThreadComm` with 1–4 ranks, so `cargo test` covers them
//! (docs/plans/M6.md, T0); the MPI versions are in gridcl-mpi-tests.

use gridcl_testing::ThreadComm;
use gridcl_testing::parallel::{self, Coverage};

const RANKS: [usize; 4] = [1, 2, 3, 4];

#[test]
fn parallel_invariance() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::cases(Coverage::Quick) {
                parallel::invariance(&comm, &client, &case);
            }
        });
    }
}

#[test]
fn scatter() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::scatter_cases().into_iter().step_by(3) {
                parallel::scatter(&comm, &client, &case);
            }
        });
    }
}

#[test]
fn determinism() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::determinism_cases() {
                parallel::determinism(&comm, &client, &case);
            }
        });
    }
}

#[test]
fn locality() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            parallel::locality(&comm, &client);
        });
    }
}
