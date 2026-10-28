//! `Mpi` against the expected values of the `ThreadComm` tests (gridcl_testing::comm_checks):
//! every `CommDyn` and `CommExt` method, an asymmetric neighbourhood, and neighbourhood
//! exchange on 64 random graphs (docs/plans/M6.md, T3).

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
        let dup = comm.duplicate();
        assert_eq!((dup.rank(), dup.size()), (comm.rank(), comm.size()));
        assert_eq!(dup.sum(1_u32) as usize, comm.size());
    })
}
