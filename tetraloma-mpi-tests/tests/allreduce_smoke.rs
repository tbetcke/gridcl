//! M0 acceptance test: every rank opens the test device, and the sum-allreduce of the rank
//! ids equals n(n−1)/2 (through `CommExt::sum` since M6).

use std::process::ExitCode;

use tetraloma::{CommDyn, CommExt};

fn main() -> ExitCode {
    tetraloma_mpi_tests::run("allreduce_smoke", |comm| {
        let _client = tetraloma_testing::test_client();
        let (rank, size) = (comm.rank() as i64, comm.size() as i64);
        let sum = comm.sum(rank);
        assert_eq!(
            sum,
            size * (size - 1) / 2,
            "rank {rank} of {size}: sum of rank ids"
        );
    })
}
