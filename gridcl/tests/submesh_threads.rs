//! M9 submeshes on `ThreadComm` with 1–4 ranks (docs/plans/M9.md, acceptance coverage): the
//! boundary, facet, cell and edge submeshes equal a one-rank build's, with outward normals,
//! ∮ x·n = gdim · V, a consistent `EntityMap` and an agreed `InteriorFacet`
//! (`gridcl_testing::parallel::submesh`). The MPI version is gridcl-mpi-tests' `submesh_parallel`.

use gridcl_testing::ThreadComm;
use gridcl_testing::parallel::{self, Coverage, Family};

const RANKS: [usize; 4] = [1, 2, 3, 4];

/// The boundary of the ball in every ghost mode.
#[test]
fn ball_boundary_normals_point_outward() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::submesh_cases(Coverage::Full)
                .into_iter()
                .filter(|c| c.family == Family::Ball)
            {
                parallel::submesh(&comm, &client, &case);
            }
        });
    }
}

/// Every other submesh family, one ghost mode per case.
#[test]
fn submeshes_match_one_rank() {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::submesh_cases(Coverage::Quick)
                .into_iter()
                .filter(|c| c.family != Family::Ball)
            {
                parallel::submesh(&comm, &client, &case);
            }
        });
    }
}
