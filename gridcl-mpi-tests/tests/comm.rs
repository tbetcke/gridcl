//! `Mpi` against the expected values of the `ThreadComm` tests (gridcl_testing::comm_checks):
//! every `CommDyn` and `CommExt` method, an asymmetric neighbourhood, and neighbourhood
//! exchange on 64 random graphs (docs/plans/M6.md, T3); `IndexMap::with_ghosts` on 16 random
//! layouts (T5); `local_gpu_index` against the host names (M10; ADR-0060 item 1).

use std::process::ExitCode;

use gridcl::{CommDyn, CommExt};
use gridcl_testing::comm_checks;

fn main() -> ExitCode {
    gridcl_mpi_tests::run("comm", |comm| {
        if comm.size() == 1 {
            comm_checks::matches_serial(comm);
        } else {
            comm_checks::wrong_counts(comm);
        }
        comm_checks::identities(comm);
        comm_checks::ring(comm);
        // The seeds of rank 0, so every rank builds the same graphs.
        let mut seed = comm.all_gather(0x6d70_6974_6573_7473_u64)[0];
        for _ in 0..64 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            comm_checks::neighbour_exchange(comm, seed);
        }
        comm_checks::index_map_basics(comm);
        for _ in 0..16 {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            comm_checks::index_map_layout(comm, seed);
        }
        let dup = comm.duplicate();
        assert_eq!((dup.rank(), dup.size()), (comm.rank(), comm.size()));
        assert_eq!(dup.sum(1_u32) as usize, comm.size());
        local_gpu_index_is_the_node_local_rank(comm);
    })
}

/// `local_gpu_index` is the rank among the ranks on this host, computed here independently of
/// the shared-memory split: an all-gather of a hash of each rank's host name
/// (`MPI_Get_processor_name`), counting the lower ranks on the same host. It holds on one node
/// and on Kathleen's two.
fn local_gpu_index_is_the_node_local_rank(comm: &gridcl::Mpi) {
    let name = mpi::environment::processor_name().expect("MPI_Get_processor_name");
    // FNV-1a.
    let hash = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    let hashes = comm.all_gather(hash);
    let want = hashes[..comm.rank()].iter().filter(|&&h| h == hash).count();
    assert_eq!(
        gridcl::local_gpu_index(comm),
        want,
        "rank {} on {name}: local_gpu_index",
        comm.rank()
    );
}
