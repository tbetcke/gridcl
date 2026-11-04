//! M8 acceptance, Gmsh (docs/plans/M8.md, T0b; ADR-0035, rules 5, 7, 11): the global mesh read
//! on 1–8 ranks equals the one-rank read (the full corpus with the default partitioner at 1–4
//! and 8 ranks, else the subset; the subset with `Block` and every enabled external
//! partitioner), `Block` honours partitioned files, partitioned and gathered writes round-trip,
//! and failed writes are agreed.

use std::process::ExitCode;

use gridcl::CommDyn;
use gridcl_testing::io_parallel::{
    self as io, BLOCK, Choice, Coverage, coverage_of, read_cases, remove_shared_dir, shared_dir,
};

/// `Block` and the external partitioners of the build: Scotch and PT-Scotch with
/// `partitioners`, METIS with `metis`, ParMETIS with `parmetis`.
fn partitioners() -> Vec<Choice> {
    #[allow(unused_mut)]
    let mut out = vec![BLOCK];
    #[cfg(feature = "partitioners")]
    {
        out.push(Choice::Named("scotch", || {
            Box::new(gridcl::Scotch::default())
        }));
        out.push(Choice::Named("ptscotch", || {
            Box::new(gridcl::PtScotch::default())
        }));
    }
    #[cfg(feature = "metis")]
    out.push(Choice::Named(
        "metis",
        || Box::new(gridcl::Metis::default()),
    ));
    #[cfg(feature = "parmetis")]
    out.push(Choice::Named("parmetis", || {
        Box::new(gridcl::ParMetis::default())
    }));
    out
}

fn main() -> ExitCode {
    gridcl_mpi_tests::run("gmsh_parallel", |comm| {
        let client = gridcl_testing::test_client();
        let dir = shared_dir(comm, "gmsh-parallel");
        let mut cases = read_cases(coverage_of(comm.size()), &[Choice::Default]);
        cases.extend(read_cases(Coverage::Subset, &partitioners()));
        for case in &cases {
            io::global_mesh_identical(comm, &client, case, &dir);
        }
        io::block_honours_partitions(comm, &client);
        io::partitioned_round_trip(comm, &client, &dir);
        io::collective_write_errors(comm, &client, &dir);
        remove_shared_dir(comm, &dir);
    })
}
