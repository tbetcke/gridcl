//! `build_on` with every external partitioner (docs/plans/M7.md, T6, T9), in f32 and f64 (where
//! the device supports it): each rank owns the cells the partitioner sends it on the pipeline's
//! graph, and the volume is 1.

use std::process::ExitCode;

use gridcl::{CommDyn, CommExt, D3, DeviceGrid, Mesh, Real};
use gridcl_testing::{HostMesh, TOL_F32, TOL_F64};

/// Builds `h` with the partitioner and checks the owned cells against `want` (the cells the
/// partitioner sends to this rank) and the volume.
fn check<F: Real>(
    comm: &gridcl::Mpi,
    h: &HostMesh,
    want: usize,
    make: gridcl_mpi_tests::MakePartitioner,
    at: &str,
) {
    let rank = comm.rank();
    let client = gridcl_testing::test_client();
    if std::mem::size_of::<F>() == 8 && !gridcl_testing::supports::<f64>(&client) {
        return;
    }
    let mesh: Mesh<F, D3, D3, gridcl::Mpi> = h
        .chunk(rank, comm.size())
        .builder::<F, D3, D3>(&client)
        .and_then(|b| {
            b.partitioner(BoxPartitioner(make(3)))
                .build_on(comm.clone())
        })
        .unwrap_or_else(|e| panic!("{at}, rank {rank}: {e}"));
    let owned: usize = mesh
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.num_owned())
        .sum();
    assert_eq!(owned, want, "{at}, rank {rank}: owned cells");
    let volume = mesh.measure(2).expect("measure");
    let tol = if std::mem::size_of::<F>() == 8 {
        TOL_F64
    } else {
        TOL_F32
    };
    assert!((volume - 1.0).abs() <= 10.0 * tol, "{at}: volume {volume}");
}

/// A boxed partitioner as a `Partitioner + 'static` for `MeshBuilder::partitioner`.
struct BoxPartitioner(Box<dyn gridcl::Partitioner>);

impl gridcl::Partitioner for BoxPartitioner {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn partition(
        &self,
        comm: &dyn CommDyn,
        g: &gridcl::DistGraph,
        nparts: usize,
    ) -> gridcl::Result<Vec<i32>> {
        self.0.partition(comm, g, nparts)
    }
}

fn main() -> ExitCode {
    gridcl_mpi_tests::run("partition_build", |comm| {
        let rank = comm.rank();
        let size = comm.size();
        let meshes = [
            (
                "hex/prism/pyramid n=6",
                gridcl_testing::hex_prism_pyramid_box(6, 1),
            ),
            ("mixed n=4 p=2", gridcl_testing::mixed_box(4, 2)),
        ];
        for (mesh, h) in &meshes {
            let g = gridcl_mpi_tests::pipeline_graph(comm, h);
            for (name, make) in gridcl_mpi_tests::partitioners() {
                let at = format!("{mesh} ranks={size} {name}");
                // On one rank build_on does not call the partitioner: every cell is owned.
                let want = if size == 1 {
                    h.num_cells()
                } else {
                    let part = make(3)
                        .partition(comm, &g, size)
                        .unwrap_or_else(|e| panic!("{at}, rank {rank}: {e}"));
                    // The cells every rank sends to each rank, summed.
                    let counts: Vec<u64> = (0..size)
                        .map(|r| comm.sum(part.iter().filter(|&&d| d as usize == r).count() as u64))
                        .collect();
                    counts[rank] as usize
                };
                check::<f32>(comm, h, want, make, &format!("{at} f32"));
                check::<f64>(comm, h, want, make, &format!("{at} f64"));
            }
        }
    })
}
