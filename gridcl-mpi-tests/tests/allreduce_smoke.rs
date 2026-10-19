//! M0 acceptance test: every rank opens the test device, and the sum-allreduce of the rank
//! ids equals n(n−1)/2.

use std::process::ExitCode;

use mpi::collective::SystemOperation;
use mpi::traits::*;

fn main() -> ExitCode {
    let universe = mpi::initialize().expect("MPI is already initialised");
    let world = universe.world();
    let rank = world.rank();
    let size = world.size();

    let _client = gridcl_testing::test_client();

    let mut sum = 0i64;
    world.all_reduce_into(&i64::from(rank), &mut sum, SystemOperation::sum());
    let want = i64::from(size) * (i64::from(size) - 1) / 2;
    if sum != want {
        eprintln!("allreduce_smoke: rank {rank} of {size}: sum of rank ids is {sum}, want {want}");
        return ExitCode::FAILURE;
    }
    if rank == 0 {
        println!("allreduce_smoke: {size} ranks passed");
    }
    ExitCode::SUCCESS
}
