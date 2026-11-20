//! MPI integration tests of tetraloma.
//!
//! Each test is a `[[test]]` target with `harness = false` and the `mpi` feature, run on 1–4
//! ranks by `scripts/mpi-test.sh`. A test exits with a non-zero status on any rank that fails.
//!
//! `run` (feature `mpi`) initialises MPI and installs a panic hook that aborts the whole job,
//! so that a panic on one rank (a failed check, or a `todo!()`) ends every rank instead of
//! leaving the others waiting in the next collective. Before the test it applies the rank-count
//! guard: under `scripts/mpi-test.sh` (which exports `TETRALOMA_EXPECT_RANKS`) a world of another
//! size than `mpirun -n` requested fails at once (ADR-0068). The checks themselves are in
//! `tetraloma_testing::parallel`, shared with the `ThreadComm` tests of tetraloma.

#[cfg(feature = "mpi")]
use std::process::ExitCode;

/// Initialises MPI, installs the abort hook, checks the world size against
/// `TETRALOMA_EXPECT_RANKS` ([`tetraloma_testing::check_world_size`]), runs `f` on the world
/// communicator and reports success on rank 0.
///
/// A world size that differs from `TETRALOMA_EXPECT_RANKS` prints the guard's message on every
/// rank and returns `ExitCode::FAILURE` before any collective: every rank sees the same
/// mismatch, so none is left waiting.
#[cfg(feature = "mpi")]
pub fn run(name: &str, f: impl FnOnce(&tetraloma::Mpi)) -> ExitCode {
    use tetraloma::CommDyn;

    let universe = mpi::initialize().expect("MPI is already initialised");
    install_abort_hook();
    let comm = tetraloma::Mpi::world(&universe);
    if let Err(message) = tetraloma_testing::check_world_size(comm.size()) {
        // One write of the whole line, so that the ranks' lines never interleave
        // (scripts/check-rank-guard.sh looks for every rank's line).
        let line = format!("{name}: rank {}: {message}\n", comm.rank());
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
        return ExitCode::FAILURE;
    }
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
pub type MakePartitioner = fn(u32) -> Box<dyn tetraloma::Partitioner>;

/// The external partitioners of the M7 targets, Scotch (the reference) first: Scotch,
/// PT-Scotch, and METIS with the feature `metis` or ParMETIS with `parmetis` (ADR-0029, item 11;
/// ADR-0033).
#[cfg(feature = "partitioners")]
pub fn partitioners() -> Vec<(&'static str, MakePartitioner)> {
    #[cfg_attr(not(any(feature = "metis", feature = "parmetis")), allow(unused_mut))]
    let mut out: Vec<(&'static str, MakePartitioner)> = vec![
        ("scotch", |s| Box::new(tetraloma::Scotch::default().seed(s))),
        ("ptscotch", |s| {
            Box::new(tetraloma::PtScotch::default().seed(s))
        }),
    ];
    #[cfg(feature = "metis")]
    out.push(("metis", |s| Box::new(tetraloma::Metis::default().seed(s))));
    #[cfg(feature = "parmetis")]
    out.push(("parmetis", |s| {
        Box::new(tetraloma::ParMetis::default().seed(s))
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
pub fn pipeline_graph(
    comm: &tetraloma::Mpi,
    h: &tetraloma_testing::HostMesh,
) -> tetraloma::DistGraph {
    use tetraloma::CommDyn;

    let client = tetraloma_testing::test_client();
    h.chunk(comm.rank(), comm.size())
        .builder::<f32, tetraloma::D3, tetraloma::D3>(&client)
        .and_then(|b| b.dual_graph_on(comm.clone()))
        .unwrap_or_else(|e| panic!("dual graph, rank {}: {e}", comm.rank()))
}
