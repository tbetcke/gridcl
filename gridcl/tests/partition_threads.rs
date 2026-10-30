//! `Block` on `ThreadComm` with 1–7 ranks (docs/plans/M6.md, T8): balanced parts for any
//! distribution of the vertices, fewer vertices than ranks included, and agreed errors.

use gridcl::{Block, CommDyn, CommExt, DistGraph, GridError, Partitioner};
use gridcl_testing::ThreadComm;

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
