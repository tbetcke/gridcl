//! MPI integration tests of gridcl.
//!
//! Each test is a `[[test]]` target with `harness = false` and the `mpi` feature, run on 1–4
//! ranks by `scripts/mpi-test.sh`. A test exits with a non-zero status on any rank that fails.
//!
//! `run` (feature `mpi`) initialises MPI and installs a panic hook that aborts the whole job,
//! so that a panic on one rank (a failed check, or a `todo!()`) ends every rank instead of
//! leaving the others waiting in the next collective. The checks themselves are in
//! `gridcl_testing::parallel`, shared with the `ThreadComm` tests of gridcl.

#[cfg(feature = "mpi")]
use std::process::ExitCode;

/// Initialises MPI, installs the abort hook, runs `f` on the world communicator and reports
/// success on rank 0.
#[cfg(feature = "mpi")]
pub fn run(name: &str, f: impl FnOnce(&gridcl::Mpi)) -> ExitCode {
    use gridcl::CommDyn;

    let universe = mpi::initialize().expect("MPI is already initialised");
    install_abort_hook();
    let comm = gridcl::Mpi::world(&universe);
    f(&comm);
    comm.barrier();
    if comm.rank() == 0 {
        println!("{name}: {} ranks passed", comm.size());
    }
    ExitCode::SUCCESS
}

/// Prints the panic as usual, then aborts every rank with `MPI_Abort` (rsmpi's safe
/// `Communicator::abort`).
#[cfg(feature = "mpi")]
pub fn install_abort_hook() {
    use mpi::traits::Communicator;

    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        mpi::topology::SimpleCommunicator::world().abort(1);
    }));
}
