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

/// A partitioner of the M7 targets with its seed.
#[cfg(feature = "partitioners")]
pub type MakePartitioner = fn(u32) -> Box<dyn gridcl::Partitioner>;

/// The external partitioners of the M7 targets, Scotch (the reference) first: Scotch,
/// PT-Scotch, and METIS with the feature `metis` or ParMETIS with `parmetis` (ADR-0029, item 11;
/// ADR-0033).
#[cfg(feature = "partitioners")]
pub fn partitioners() -> Vec<(&'static str, MakePartitioner)> {
    #[cfg_attr(not(any(feature = "metis", feature = "parmetis")), allow(unused_mut))]
    let mut out: Vec<(&'static str, MakePartitioner)> = vec![
        ("scotch", |s| Box::new(gridcl::Scotch::default().seed(s))),
        ("ptscotch", |s| {
            Box::new(gridcl::PtScotch::default().seed(s))
        }),
    ];
    #[cfg(feature = "metis")]
    out.push(("metis", |s| Box::new(gridcl::Metis::default().seed(s))));
    #[cfg(feature = "parmetis")]
    out.push(("parmetis", |s| {
        Box::new(gridcl::ParMetis::default().seed(s))
    }));
    out
}

/// The pipeline's dual graph of the 3D mesh `h` in even chunks on `comm`, as `build_on` would
/// give it to the partitioner (`MeshBuilder::dual_graph_on`, ADR-0029, item 15), on the test
/// device.
///
/// # Panics
///
/// On an error, with its message.
#[cfg(feature = "partitioners")]
pub fn pipeline_graph(comm: &gridcl::Mpi, h: &gridcl_testing::HostMesh) -> gridcl::DistGraph {
    use gridcl::CommDyn;

    let client = gridcl_testing::test_client();
    h.chunk(comm.rank(), comm.size())
        .builder::<f32, gridcl::D3, gridcl::D3>(&client)
        .and_then(|b| b.dual_graph_on(comm.clone()))
        .unwrap_or_else(|e| panic!("dual graph, rank {}: {e}", comm.rank()))
}
