//! M9 `orient` on `ThreadComm` with 1–4 ranks (docs/plans/M9.md, acceptance coverage): flags
//! per cell origin id equal a one-rank build's on every local cell, normals take the requested
//! direction, and errors are agreed (`gridcl_testing::parallel::orient`). The MPI version is
//! gridcl-mpi-tests' `orient_parallel`.

use gridcl_testing::ThreadComm;
use gridcl_testing::parallel::{self, Case, Coverage, Family};

const RANKS: [usize; 4] = [1, 2, 3, 4];

/// Runs the orient check of `cases` on 1–4 ranks.
fn run(cases: &[Case]) {
    for n in RANKS {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in cases {
                parallel::orient(&comm, &client, case);
            }
        });
    }
}

/// The cases of `families` at `coverage`.
fn cases(coverage: Coverage, families: &[Family]) -> Vec<Case> {
    parallel::orient_cases(coverage)
        .into_iter()
        .filter(|c| families.contains(&c.family))
        .collect()
}

/// A flipped sphere in every ghost mode; the mixed surface and the circle in one per case.
#[test]
fn outward_recovers_flipped_sphere() {
    let mut all = cases(Coverage::Full, &[Family::Sphere]);
    all.extend(cases(
        Coverage::Quick,
        &[Family::MixedSurface, Family::Circle],
    ));
    run(&all);
}

#[test]
fn three_cells_on_an_edge_is_non_manifold() {
    run(&cases(Coverage::Full, &[Family::Book, Family::YJunction]));
}

#[test]
fn mobius_strip_is_non_orientable() {
    run(&cases(Coverage::Full, &[Family::Mobius]));
}

/// `Outward` fails on the hemisphere; `AwayFrom` and `Towards` the origin succeed.
#[test]
fn outward_on_open_surface_needs_reference_point() {
    run(&cases(Coverage::Full, &[Family::Hemisphere]));
}

/// The circle (12 cells) on 16 ranks, so that some ranks own no cells: every rank must take
/// part in each collective (ranks without owned cells skipped the agreed Gauss degree of the
/// direction step, and MPI runs at 16 ranks hung).
#[test]
fn ranks_without_cells_take_part() {
    let all = cases(Coverage::Quick, &[Family::Circle]);
    ThreadComm::run(16, |comm| {
        let client = gridcl_testing::test_client();
        for case in &all {
            parallel::orient(&comm, &client, case);
        }
    });
}
