//! M9h acceptance (docs/plans/M9h.md, rule 9; ADR-0042, item 9): geometric factors and
//! pull-back results per (cell origin id, point) equal one rank's, at p ∈ {1, 4, 8}, for every
//! high-order test-mesh family (pyramids p ≤ 3), every ghost mode at p ≤ 2 and one per case
//! above.

use std::process::ExitCode;

use gridcl_testing::parallel::{self, HIGH_ORDER_MPI_DEGREES};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("high_order_parallel", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::high_order_cases(&HIGH_ORDER_MPI_DEGREES) {
            parallel::high_order_invariance(comm, &client, &case);
        }
    })
}
