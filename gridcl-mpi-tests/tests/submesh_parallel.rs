//! M9 acceptance: boundary, facet, cell and edge submeshes equal a one-rank build's, with
//! outward normals on the ball and the boxes, ∮ x·n = gdim · V, a consistent `EntityMap` and an
//! agreed `InteriorFacet`, for every submesh family at p = 1 and 2 in every ghost mode, and an
//! agreed `NonManifold` on a book (docs/plans/M9.md, acceptance coverage).

use std::process::ExitCode;

use gridcl_testing::parallel::{self, Coverage, Family};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("submesh_parallel", |comm| {
        let client = gridcl_testing::test_client();
        for case in parallel::submesh_cases(Coverage::Full) {
            parallel::submesh(comm, &client, &case);
        }
        for case in parallel::orient_cases(Coverage::Full)
            .into_iter()
            .filter(|c| c.family == Family::Book)
        {
            parallel::submesh_non_manifold(comm, &client, &case);
        }
    })
}
