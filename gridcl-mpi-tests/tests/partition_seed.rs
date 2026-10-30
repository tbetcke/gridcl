//! M7 acceptance (ADR-0029, item 11): a fixed seed reproduces the partition for a fixed rank
//! count. Per external partitioner and acceptance mesh, seed 1 twice with seed 2 in between
//! gives identical parts on every rank; rank 0 prints the hash of the parts, which must agree
//! between two `mpirun` runs.

use std::process::ExitCode;

use gridcl::CommDyn;
use gridcl_testing::partition::{acceptance_meshes, parts_hash};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("partition_seed", |comm| {
        let rank = comm.rank();
        let nparts = comm.size();
        for mesh in acceptance_meshes() {
            let g = gridcl_mpi_tests::pipeline_graph(comm, &mesh.host(1));
            for (name, make) in gridcl_mpi_tests::partitioners() {
                let at = format!("{} ranks={nparts} {name}", mesh.name);
                let run = |seed| {
                    make(seed)
                        .partition(comm, &g, nparts)
                        .unwrap_or_else(|e| panic!("{at} seed={seed}, rank {rank}: {e}"))
                };
                let first = run(1);
                let other = run(2);
                let again = run(1);
                assert_eq!(
                    first, again,
                    "{at}, rank {rank}: seed 1 gives different parts"
                );
                let hash = parts_hash(comm, &first);
                let other_hash = parts_hash(comm, &other);
                if rank == 0 {
                    println!("parts_hash {at} seed=1: {hash:016x} (seed=2: {other_hash:016x})");
                }
            }
        }
    })
}
