//! `IndexMap::with_ghosts` on `ThreadComm` with 1–5 ranks (docs/plans/M6.md, T5): random ghost
//! layouts against a host reference, and the agreed errors.

use gridcl_testing::{ThreadComm, comm_checks};
use proptest::prelude::*;

#[test]
fn with_ghosts_without_ghosts_equals_new_and_errors_are_agreed() {
    for n in 1..=5 {
        ThreadComm::run(n, |comm| comm_checks::index_map_basics(&comm));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_ghost_layouts(n in 1..=5_usize, seed in any::<u64>()) {
        ThreadComm::run(n, |comm| comm_checks::index_map_layout(&comm, seed));
    }
}
