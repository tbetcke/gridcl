//! The M8 acceptance checks on `ThreadComm` (docs/plans/M8.md, T0b; ADR-0035), so that
//! `cargo test` covers them; the MPI versions are in gridcl-mpi-tests. The checks are in
//! `gridcl_testing::io_parallel`. The memory bullet needs one process per rank and runs only
//! under MPI (`read_memory`).

use gridcl::prelude::*;
use gridcl_testing::ThreadComm;
use gridcl_testing::io_parallel::{
    self as io, BLOCK, Choice, Coverage, VtkFormat, read_cases, remove_shared_dir, shared_dir,
};

/// Bullet 1 on `ranks` threads: every case of `coverage` with `partitioners`.
fn global_mesh_identical_on(ranks: usize, coverage: Coverage, partitioners: &[Choice]) {
    ThreadComm::run(ranks, |comm| {
        let client = gridcl_testing::test_client();
        let dir = shared_dir(&comm, "global-mesh");
        for case in read_cases(coverage, partitioners) {
            io::global_mesh_identical(&comm, &client, &case, &dir);
        }
        remove_shared_dir(&comm, &dir);
    });
}

/// Bullet 1, the full corpus at 1–4 and 8 ranks with the default partitioner (`Block` on
/// threads, `Metis` with feature `metis`; rule 6), one test per rank count so that hosted CI's
/// shards can share them out. With `GRIDCL_IO_THREADS_SUBSET=1` the subset instead: hosted
/// CI's CPU jobs, where the full corpus in both precisions takes about 55 min (the full corpus
/// runs locally, on Locust and in the MPI jobs).
mod global_mesh_identical {
    use super::*;

    fn on(ranks: usize) {
        let coverage = if std::env::var_os("GRIDCL_IO_THREADS_SUBSET").is_some_and(|v| v == "1") {
            Coverage::Subset
        } else {
            Coverage::Full
        };
        global_mesh_identical_on(ranks, coverage, &[Choice::Default]);
    }

    #[test]
    fn ranks_1() {
        on(1);
    }

    #[test]
    fn ranks_2() {
        on(2);
    }

    #[test]
    fn ranks_3() {
        on(3);
    }

    #[test]
    fn ranks_4() {
        on(4);
    }

    #[test]
    fn ranks_8() {
        on(8);
    }
}

/// Bullet 1, the subset at 5–7 ranks, with the default partitioner and `Block`; one test per
/// rank count.
mod global_mesh_identical_subset {
    use super::*;

    fn on(ranks: usize) {
        global_mesh_identical_on(ranks, Coverage::Subset, &[Choice::Default, BLOCK]);
    }

    #[test]
    fn ranks_5() {
        on(5);
    }

    #[test]
    fn ranks_6() {
        on(6);
    }

    #[test]
    fn ranks_7() {
        on(7);
    }
}

/// Rule 5 at 1–4 ranks (the fixtures have 2 and 3 partitions).
#[test]
fn block_honours_partitions() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            io::block_honours_partitions(&comm, &gridcl_testing::test_client());
        });
    }
}

/// Rule 7 at 1–4 ranks.
#[test]
fn partitioned_round_trip() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let dir = shared_dir(&comm, "round-trip");
            io::partitioned_round_trip(&comm, &gridcl_testing::test_client(), &dir);
            remove_shared_dir(&comm, &dir);
        });
    }
}

/// Bullet 4 on the PVTU at 1–4 ranks, with shared-facet and one-layer shared-vertex ghosts.
#[test]
fn pvtu_ghosts_once() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            let dir = shared_dir(&comm, "pvtu-ghosts");
            for ghosts in [
                GhostMode::SharedFacet,
                GhostMode::SharedVertex { layers: 1 },
            ] {
                io::ghosts_once(&comm, &client, ghosts, VtkFormat::Pvtu, &dir);
            }
            remove_shared_dir(&comm, &dir);
        });
    }
}

/// Rule 8: the gathered VTU at 1–4 ranks equals the one-rank VTU after sorting by origin id.
#[test]
fn vtu_gathered_equals_serial() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let dir = shared_dir(&comm, "vtu-gathered");
            io::vtu_gathered_equals_serial(&comm, &gridcl_testing::test_client(), &dir);
            remove_shared_dir(&comm, &dir);
        });
    }
}

/// Rules 7–8: failed writes are agreed, at 1–3 ranks.
#[test]
fn collective_write_errors() {
    for n in 1..=3 {
        ThreadComm::run(n, |comm| {
            let dir = shared_dir(&comm, "write-errors");
            io::collective_write_errors(&comm, &gridcl_testing::test_client(), &dir);
            remove_shared_dir(&comm, &dir);
        });
    }
}

/// `VtkWriter::vtkhdf` for the checks.
#[cfg(feature = "hdf5")]
fn vtkhdf(path: &std::path::Path) -> VtkWriter<'static> {
    VtkWriter::vtkhdf(path)
}

/// Bullet 3 at 1–4 ranks (feature `hdf5`).
#[cfg(feature = "hdf5")]
#[test]
fn vtkhdf_equals_pvtu() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let dir = shared_dir(&comm, "vtkhdf");
            io::vtkhdf_equals_pvtu(&comm, &gridcl_testing::test_client(), vtkhdf, &dir);
            remove_shared_dir(&comm, &dir);
        });
    }
}

/// Bullet 4 on the VTKHDF file at 1–4 ranks (feature `hdf5`).
#[cfg(feature = "hdf5")]
#[test]
fn vtkhdf_ghosts_once() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            let dir = shared_dir(&comm, "vtkhdf-ghosts");
            for ghosts in [
                GhostMode::SharedFacet,
                GhostMode::SharedVertex { layers: 1 },
            ] {
                io::ghosts_once(&comm, &client, ghosts, VtkFormat::Vtkhdf(vtkhdf), &dir);
            }
            remove_shared_dir(&comm, &dir);
        });
    }
}
