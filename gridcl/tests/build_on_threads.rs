//! `MeshBuilder::build_on` on `ThreadComm` with 1–4 ranks (M6 T12): every test-mesh family
//! builds, and the global cell, vertex and node counts, the owned cell ids and the volume agree
//! with one rank.

use gridcl::prelude::*;
use gridcl::{CommDyn, GhostMode};
use gridcl_device::Client;

use gridcl_testing::parallel::{self, Case, Coverage, Distribution, MeshCheck, RoundRobin};
use gridcl_testing::{HostChunk, HostMesh, ThreadComm};

/// The global counts and the volume of a mesh, and the sorted origin ids of the cells this
/// rank owns, gathered on rank 0.
fn summary<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>, degree: u32) -> (Vec<i64>, f64, Vec<i64>)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut counts: Vec<i64> = mesh
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.size_global())
        .collect();
    counts.push(mesh.vertex_index_map().size_global());
    counts.push(mesh.node_index_map().size_global());
    let mut owned = Vec::new();
    for b in mesh.cell_blocks() {
        let ids = gridcl_topology::decode_origin_ids(&b.origin_ids.to_host().unwrap());
        owned.extend_from_slice(&ids[..b.index_map.num_owned()]);
    }
    let mut ids = parallel::gather_to_root(mesh.comm(), &owned).unwrap();
    ids.sort_unstable();
    (counts, mesh.measure(2 * degree).unwrap(), ids)
}

struct Build<'a> {
    comm: &'a ThreadComm,
    client: &'a Client,
}

impl MeshCheck for Build<'_> {
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let rank = self.comm.rank();
        let at = format!("{case}, rank {rank}");
        let serial: Mesh<F, TD, GD> = host
            .chunk(0, 1)
            .builder(self.client)
            .unwrap()
            .build()
            .unwrap();
        let chunk = match case.distribution {
            Distribution::RankZero if rank != 0 => HostChunk::empty(host.gdim),
            Distribution::RankZero => host.chunk(0, 1),
            _ => host.chunk(rank, self.comm.size()),
        };
        let mut builder = chunk
            .builder::<F, TD, GD>(self.client)
            .unwrap()
            .ghosts(case.ghosts);
        if case.distribution == Distribution::RoundRobin {
            builder = builder.partitioner(RoundRobin);
        }
        let mesh = builder
            .build_on(self.comm.clone())
            .unwrap_or_else(|e| panic!("{at}: {e}"));

        // Local layout: [owned | ghost] per map; ghost cells only with a ghost mode.
        for b in mesh.cell_blocks() {
            let m = &b.index_map;
            assert_eq!(m.num_local(), m.num_owned() + m.num_ghosts(), "{at}");
            if case.ghosts == GhostMode::None {
                assert_eq!(m.num_ghosts(), 0, "{at}");
            }
        }
        assert_eq!(
            mesh.origin().as_ref(),
            serial.origin().as_ref(),
            "{at}: origin"
        );

        let (counts, volume, ids) = summary(&mesh, case.degree);
        let (want_counts, want_volume, want_ids) = summary(&serial, case.degree);
        assert_eq!(counts, want_counts, "{at}: cell, vertex and node counts");
        if rank == 0 {
            assert_eq!(ids, want_ids, "{at}: owned cells");
        }
        let tol = if size_of::<F>() == 8 { 1e-12 } else { 1e-5 };
        let err = (volume - want_volume).abs() / want_volume.abs();
        assert!(err <= tol, "{at}: volume {volume} vs {want_volume}");
    }
}

#[test]
fn every_family_builds_on_one_to_four_ranks() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            for case in parallel::cases(Coverage::Quick) {
                parallel::dispatch(
                    &client,
                    &case,
                    &mut Build {
                        comm: &comm,
                        client: &client,
                    },
                );
            }
        });
    }
}
