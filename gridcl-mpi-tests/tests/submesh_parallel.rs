//! M9 acceptance: boundary, facet, cell and edge submeshes equal a one-rank build's, with
//! outward normals on the ball and the boxes, ∮ x·n = gdim · V, a consistent `EntityMap` and an
//! agreed `InteriorFacet`, for every submesh family at p = 1 and 2 in every ghost mode
//! (docs/plans/M9.md, acceptance coverage).

use std::process::ExitCode;

use gridcl_testing::parallel::{self, Coverage};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("submesh_parallel", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::submesh_cases(Coverage::Full) {
            parallel::submesh(comm, &client, &case);
        }
    })
}
