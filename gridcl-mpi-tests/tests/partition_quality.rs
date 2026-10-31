//! M7 acceptance (ADR-0029, item 11) on the tetrahedral box of 1 053 696 cells and the
//! hexahedron/prism/pyramid box of 208 444 cells, one part per rank, on the pipeline's dual
//! graph:
//!
//! - every external partitioner's weighted imbalance is at most 5%;
//! - cut(PtScotch) ≤ 1.10 · cut(Scotch) and cut(Metis) ≤ 1.10 · cut(Scotch);
//! - every cut, Scotch's included, is at most twice the slab bound (ADR-0030); rank 0 prints
//!   every cut, its ratio to the slab bound and the ratios to Scotch's.

use std::process::ExitCode;
use std::time::Instant;

use gridcl::CommDyn;
use gridcl_testing::partition::{acceptance_meshes, edge_cut, slab_cut_bound, weighted_imbalance};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("partition_quality", |comm| {
        let rank = comm.rank();
        let nparts = comm.size();
        for mesh in acceptance_meshes() {
            let g = gridcl_mpi_tests::pipeline_graph(comm, &mesh.host(1));
            let bound = slab_cut_bound(&mesh, nparts);
            let mut cuts = Vec::new();
            for (name, make) in gridcl_mpi_tests::partitioners() {
                let at = format!("{} ranks={nparts} {name}", mesh.name);
                let start = Instant::now();
                let part = make(0)
                    .partition(comm, &g, nparts)
                    .unwrap_or_else(|e| panic!("{at}, rank {rank}: {e}"));
                let seconds = start.elapsed().as_secs_f64();
                let imbalance = weighted_imbalance(comm, &g, &part, nparts).expect("imbalance");
                let cut = edge_cut(comm, &g, &part).expect("edge cut");
                if rank == 0 {
                    println!(
                        "partition_quality {at}: cut {cut} (slab bound {bound}, ratio {:.3}), \
                         weighted imbalance {imbalance:.4}, {seconds:.2} s",
                        cut as f64 / bound.max(1) as f64
                    );
                }
                assert!(
                    imbalance <= 0.05,
                    "{at}: weighted imbalance {imbalance} > 0.05"
                );
                assert!(
                    cut <= 2 * bound,
                    "{at}: cut {cut} above twice the slab bound {bound} (ADR-0030)"
                );
                cuts.push((name, cut));
            }
            let scotch = cuts[0].1;
            for &(name, cut) in &cuts[1..] {
                let ratio = cut as f64 / scotch.max(1) as f64;
                if rank == 0 {
                    println!(
                        "partition_quality {} ranks={nparts}: cut({name}) / cut(scotch) = \
                         {ratio:.4}",
                        mesh.name
                    );
                }
                if name == "ptscotch" || name == "metis" {
                    assert!(
                        cut * 100 <= 110 * scotch,
                        "{} ranks={nparts}: cut({name}) = {cut} > 1.10 · cut(scotch) = 1.10 · \
                         {scotch}",
                        mesh.name
                    );
                }
            }
        }
    })
}
