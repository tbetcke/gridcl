//! `Block` on `ThreadComm` with 1–7 ranks (docs/plans/M6.md, T8): balanced parts for any
//! distribution of the vertices, fewer vertices than ranks included, and agreed errors.
//!
//! M7 (docs/plans/M7.md): the vertex weights of the pipeline's dual graph (`weights`), the
//! rank-0 partitioners `Metis` and `Scotch` on 1–4 ranks (features `metis`, `scotch`), and the
//! distributed partitioners' error without an MPI communicator (features `ptscotch`,
//! `parmetis`, with `gridcl-testing/mpi`).

use gridcl::{Block, Comm, CommDyn, CommExt, D1, D2, D3, DistGraph, GridError, Partitioner};
use gridcl_testing::partition::host_dual_graph;
use gridcl_testing::{HostMesh, ThreadComm};

/// A graph without edges with `counts[r]` vertices on rank r.
fn graph(rank: usize, counts: &[i64]) -> DistGraph {
    let vtxdist: Vec<i64> = std::iter::once(0)
        .chain(counts.iter().scan(0, |at, &c| {
            *at += c;
            Some(*at)
        }))
        .collect();
    DistGraph {
        vtxdist,
        xadj: vec![0; counts[rank] as usize + 1],
        adjncy: vec![],
        vwgt: None,
    }
}

#[test]
fn parts_are_balanced_within_one() {
    for p in 1..=7_usize {
        // Fewer vertices than ranks, everything on one rank, and uneven chunks.
        let layouts: Vec<Vec<i64>> = vec![
            (0..p as i64)
                .map(|r| i64::from(r == 0) * (p as i64 - 1))
                .collect(),
            (0..p as i64)
                .map(|r| if r == p as i64 - 1 { 50 } else { 0 })
                .collect(),
            (0..p as i64).map(|r| 3 * r + 1).collect(),
            (0..p as i64).map(|r| (r * 7) % 5).collect(),
        ];
        for counts in &layouts {
            ThreadComm::run(p, |comm| {
                let r = comm.rank();
                let g = graph(r, counts);
                let dest = Block.partition(&comm, &g, p).unwrap();
                let total = *g.vtxdist.last().unwrap();
                let first = g.vtxdist[r];
                for (k, &d) in dest.iter().enumerate() {
                    let i = first + k as i64;
                    assert_eq!(i64::from(d), i * p as i64 / total);
                }
                let sizes: Vec<i64> = (0..p)
                    .map(|part| {
                        comm.sum(dest.iter().filter(|&&d| d as usize == part).count() as i64)
                    })
                    .collect();
                assert_eq!(sizes.iter().sum::<i64>(), total);
                let (lo, hi) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
                assert!(hi - lo <= 1, "p = {p}, counts {counts:?}: sizes {sizes:?}");
            });
        }
    }
}

#[test]
fn errors_are_agreed() {
    ThreadComm::run(3, |comm| {
        let mut g = graph(comm.rank(), &[2, 2, 2]);
        if comm.rank() == 1 {
            g.xadj.pop();
        }
        match Block.partition(&comm, &g, 3) {
            Err(GridError::LengthMismatch {
                expected: 3,
                found: 2,
            }) if comm.rank() == 1 => {}
            Err(GridError::RemoteFailure { rank: 1 }) if comm.rank() != 1 => {}
            other => panic!("rank {}: {other:?}", comm.rank()),
        }
    });
}

/// The pipeline's dual graph of the even chunks of `h` (stages 1–2).
fn pipeline_graph<C: Comm>(comm: C, h: &HostMesh) -> DistGraph {
    let client = gridcl_testing::test_client();
    let chunk = h.chunk(comm.rank(), comm.size());
    let tdim = gridcl_refcell::tdim(h.blocks[0].cell);
    let graph = match (tdim, h.gdim) {
        (1, 1) => chunk
            .builder::<f32, D1, D1>(&client)
            .and_then(|b| b.dual_graph_on(comm)),
        (2, 2) => chunk
            .builder::<f32, D2, D2>(&client)
            .and_then(|b| b.dual_graph_on(comm)),
        (2, 3) => chunk
            .builder::<f32, D2, D3>(&client)
            .and_then(|b| b.dual_graph_on(comm)),
        (3, 3) => chunk
            .builder::<f32, D3, D3>(&client)
            .and_then(|b| b.dual_graph_on(comm)),
        dims => panic!("no mesh of (tdim, gdim) = {dims:?}"),
    };
    graph.expect("dual graph")
}

/// The pipeline's graph carries the Lagrange node count of every cell as its weight
/// (ADR-0029, item 2) and otherwise equals the serial facet graph, on 1–4 ranks, for every mesh
/// family at p = 1 and 2.
#[test]
fn weights() {
    use gridcl_testing::parallel::FAMILIES;

    let mut hosts: Vec<(String, HostMesh)> = Vec::new();
    for p in 1..=2 {
        for family in FAMILIES {
            hosts.push((format!("{family:?} p={p}"), family.host(p)));
        }
        hosts.push((
            format!("hex/prism/pyramid p={p}"),
            gridcl_testing::hex_prism_pyramid_box(3, p),
        ));
    }
    for (name, h) in &hosts {
        let want = host_dual_graph(h);
        for size in 1..=4 {
            ThreadComm::run(size, |comm| {
                let r = comm.rank();
                let g = pipeline_graph(comm, h);
                assert_eq!(g.vtxdist, chunk_vtxdist(h.num_cells(), size), "{name}");
                let (begin, end) = (g.vtxdist[r] as usize, g.vtxdist[r + 1] as usize);
                let w = g.vwgt.as_ref().expect("weights");
                assert_eq!(
                    w[..],
                    want.vwgt.as_ref().unwrap()[begin..end],
                    "{name}, rank {r} of {size}"
                );
                for (i, u) in (begin..end).enumerate() {
                    let got = &g.adjncy[g.xadj[i] as usize..g.xadj[i + 1] as usize];
                    let exp = &want.adjncy[want.xadj[u] as usize..want.xadj[u + 1] as usize];
                    assert_eq!(got, exp, "{name}, rank {r} of {size}, vertex {u}");
                }
            });
        }
    }
}

/// `dual_graph_on` returns the graph that `build_on` gives the partitioner, on 2–4 ranks.
#[test]
fn dual_graph_on_equals_the_partitioner_input() {
    use std::sync::{Arc, Mutex};

    /// `Block`, keeping the graph it receives.
    struct Recording(Arc<Mutex<Option<DistGraph>>>);
    impl Partitioner for Recording {
        fn name(&self) -> &str {
            "recording"
        }
        fn partition(
            &self,
            comm: &dyn CommDyn,
            g: &DistGraph,
            nparts: usize,
        ) -> gridcl::Result<Vec<i32>> {
            *self.0.lock().unwrap() = Some(g.clone());
            Block.partition(comm, g, nparts)
        }
    }

    let hosts = [
        ("mixed", gridcl_testing::mixed_box(2, 1)),
        (
            "hex/prism/pyramid p=2",
            gridcl_testing::hex_prism_pyramid_box(2, 2),
        ),
    ];
    for (name, h) in &hosts {
        for size in 2..=4 {
            ThreadComm::run(size, |comm| {
                let client = gridcl_testing::test_client();
                let seen = Arc::new(Mutex::new(None));
                let chunk = h.chunk(comm.rank(), size);
                chunk
                    .builder::<f32, D3, D3>(&client)
                    .unwrap()
                    .partitioner(Recording(seen.clone()))
                    .build_on(comm.clone())
                    .expect("mesh");
                let want = pipeline_graph(comm, h);
                assert_eq!(
                    seen.lock().unwrap().as_ref(),
                    Some(&want),
                    "{name} on {size} ranks"
                );
            });
        }
    }
}

/// `vtxdist` of `n` cells in even chunks over `size` ranks (`HostMesh::chunk`).
fn chunk_vtxdist(n: usize, size: usize) -> Vec<i64> {
    (0..=size).map(|r| (r * n / size) as i64).collect()
}

/// Rank `rank`'s share of the one-rank graph `g` when rank r holds `counts[r]` vertices.
#[cfg(any(
    feature = "metis",
    feature = "scotch",
    feature = "ptscotch",
    feature = "parmetis"
))]
fn split(g: &DistGraph, counts: &[usize], rank: usize) -> DistGraph {
    let mut vtxdist = vec![0_i64];
    for &c in counts {
        vtxdist.push(vtxdist.last().unwrap() + c as i64);
    }
    let (begin, end) = (vtxdist[rank] as usize, vtxdist[rank + 1] as usize);
    let (lo, hi) = (g.xadj[begin], g.xadj[end]);
    DistGraph {
        vtxdist,
        xadj: g.xadj[begin..=end].iter().map(|x| x - lo).collect(),
        adjncy: g.adjncy[lo as usize..hi as usize].to_vec(),
        vwgt: g.vwgt.as_ref().map(|w| w[begin..end].to_vec()),
    }
}

/// The rank-0 partitioners on ThreadComm: destinations in range, weighted imbalance ≤ 5%, the
/// cut within the slab bound (ADR-0029, item 11), and the same seed giving the same parts with
/// another seed's call in between, for even, uneven and empty-rank layouts on 1–4 ranks.
#[cfg(any(feature = "metis", feature = "scotch"))]
mod rank0 {
    use super::*;
    use gridcl_testing::partition::{
        BoxKind, PartitionMesh, edge_cut, slab_cut_bound, weighted_imbalance,
    };

    /// The vertex counts per rank: even, uneven, and everything on the last rank.
    fn layouts(n: usize, size: usize) -> Vec<Vec<usize>> {
        let even = (0..size)
            .map(|r| (r + 1) * n / size - r * n / size)
            .collect();
        let total: usize = (1..=size).map(|r| 3 * r - 2).sum();
        let mut uneven: Vec<usize> = (0..size).map(|r| (3 * r + 1) * n / total).collect();
        uneven[0] += n - uneven.iter().sum::<usize>();
        let last = (0..size)
            .map(|r| if r + 1 == size { n } else { 0 })
            .collect();
        vec![even, uneven, last]
    }

    pub(super) fn check<P: Partitioner + Sync>(
        name: &str,
        make: impl Fn(u32) -> P + Sync,
        mesh: PartitionMesh,
        p: u32,
    ) {
        let h = mesh.host(p);
        let g = host_dual_graph(&h);
        let n = h.num_cells();
        for size in 1..=4 {
            for counts in layouts(n, size) {
                let at = format!(
                    "{name} on {} p={p}, {size} ranks, counts {counts:?}",
                    mesh.name
                );
                ThreadComm::run(size, |comm| {
                    let r = comm.rank();
                    let local = split(&g, &counts, r);
                    let dest = make(7).partition(&comm, &local, size).expect(&at);
                    assert_eq!(dest.len(), counts[r], "{at}");
                    assert!(dest.iter().all(|&d| (0..size as i32).contains(&d)), "{at}");
                    let imbalance = weighted_imbalance(&comm, &local, &dest, size).unwrap();
                    assert!(imbalance <= 0.05, "{at}: weighted imbalance {imbalance}");
                    let cut = edge_cut(&comm, &local, &dest).unwrap();
                    let bound = slab_cut_bound(&mesh, size);
                    assert!(cut <= bound, "{at}: cut {cut} above the slab bound {bound}");
                    let other = make(8).partition(&comm, &local, size).expect(&at);
                    assert_eq!(other.len(), counts[r], "{at}");
                    let again = make(7).partition(&comm, &local, size).expect(&at);
                    assert_eq!(again, dest, "{at}: seed 7 twice gives different parts");
                });
            }
        }
    }

    pub(super) const TETS: PartitionMesh = PartitionMesh {
        name: "tetrahedra n=12",
        kind: BoxKind::Tetrahedra,
        n: 12,
    };
    pub(super) const MIXED: PartitionMesh = PartitionMesh {
        name: "hex/prism/pyramid n=12",
        kind: BoxKind::HexPrismPyramid,
        n: 12,
    };
}

#[cfg(feature = "metis")]
#[test]
fn metis_on_tetrahedra() {
    rank0::check(
        "metis",
        |s| gridcl::Metis::default().seed(s),
        rank0::TETS,
        1,
    );
}

#[cfg(feature = "metis")]
#[test]
fn metis_on_hex_prism_pyramid() {
    rank0::check(
        "metis",
        |s| gridcl::Metis::default().seed(s),
        rank0::MIXED,
        2,
    );
}

#[cfg(feature = "scotch")]
#[test]
fn scotch_on_tetrahedra() {
    rank0::check(
        "scotch",
        |s| gridcl::Scotch::default().seed(s),
        rank0::TETS,
        1,
    );
}

#[cfg(feature = "scotch")]
#[test]
fn scotch_on_hex_prism_pyramid() {
    rank0::check(
        "scotch",
        |s| gridcl::Scotch::default().seed(s),
        rank0::MIXED,
        2,
    );
}

/// Without an MPI handle on more than one rank, the distributed partitioners fail on every
/// rank (ADR-0029, item 6).
#[cfg(any(feature = "ptscotch", feature = "parmetis"))]
#[test]
fn distributed_partitioners_need_mpi() {
    let partitioners: Vec<(&str, Box<dyn Partitioner + Sync>)> = vec![
        #[cfg(feature = "ptscotch")]
        ("ptscotch", Box::new(gridcl::PtScotch::default())),
        #[cfg(feature = "parmetis")]
        ("parmetis", Box::new(gridcl::ParMetis::default())),
    ];
    let h = gridcl_testing::cube_box(gridcl::Cell3::Hexahedron, 2, 1);
    let g = host_dual_graph(&h);
    for (name, partitioner) in &partitioners {
        let out = ThreadComm::run(2, |comm| {
            partitioner.partition(&comm, &split(&g, &[4, 4], comm.rank()), 2)
        });
        for r in out {
            assert!(
                matches!(r, Err(GridError::NeedsMpiCommunicator(n)) if n == *name),
                "{name}: {r:?}"
            );
        }
    }
}
