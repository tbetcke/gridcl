//! Checks of a communicator's `CommDyn` and `CommExt` methods, shared by the `ThreadComm` tests
//! and the MPI target `comm` (docs/plans/M6.md, T2 and T3), so both assert the same values.
//! Every function is collective: every rank of `comm` calls it.

use gridcl::{CommDyn, CommExt, DType, GridError, Neighbourhood, Serial};
use gridcl_device::ReduceOp;

/// On one rank, every byte-level method agrees with `Serial`.
///
/// # Panics
///
/// If `comm` has more than one rank, or a result differs.
pub fn matches_serial(comm: &dyn CommDyn) {
    let serial: &dyn CommDyn = &Serial;
    assert_eq!((comm.rank(), comm.size()), (0, 1));
    comm.barrier();
    for op in [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max] {
        let mut a = [1.5_f64.to_ne_bytes(), (-2.0_f64).to_ne_bytes()].concat();
        let mut b = a.clone();
        comm.all_reduce_bytes(&mut a, DType::F64, op);
        serial.all_reduce_bytes(&mut b, DType::F64, op);
        assert_eq!(a, b);
        for dtype in [DType::I32, DType::U64, DType::F32] {
            let mut a = vec![7_u8; 8];
            let mut b = a.clone();
            comm.exscan_bytes(&mut a, dtype, op);
            serial.exscan_bytes(&mut b, dtype, op);
            assert_eq!(a, b, "{dtype:?} {op:?}");
        }
    }
    let send = 9_u32.to_ne_bytes();
    let (mut a, mut b) = ([0_u8; 4], [0_u8; 4]);
    comm.all_to_all_v_bytes(&send, &[1], &mut a, &[1], DType::U32)
        .unwrap();
    serial
        .all_to_all_v_bytes(&send, &[1], &mut b, &[1], DType::U32)
        .unwrap();
    assert_eq!(a, b);
    let nb = Neighbourhood::new(comm, vec![], vec![]).unwrap();
    comm.neighbour_all_to_all_v_bytes(&nb, &[], &[], &mut [], &[], DType::F32)
        .unwrap();
    assert!(Neighbourhood::new(comm, vec![0], vec![]).is_err());
    assert_eq!(comm.duplicate().size(), 1);
    assert_eq!(comm.min(3_i64), 3);
    assert_eq!(comm.all_gather(4_u32), [4]);
    assert_eq!(comm.exscan_sum(5_u64), 0);
}

/// Reductions, scans (with the rank-0 identities of ADR-0011), gathers and a variable
/// all-to-all with known results on any number of ranks.
pub fn identities(comm: &dyn CommDyn) {
    let (r, n) = (comm.rank(), comm.size());
    assert_eq!(comm.sum(r as u64), (n * (n - 1) / 2) as u64);
    assert_eq!(comm.min(r as i32 - 1), -1);
    assert_eq!(comm.max(r as f32), (n - 1) as f32);
    assert_eq!(comm.sum(0.5_f64), 0.5 * n as f64);
    assert_eq!(
        comm.exscan_sum(r as i64),
        (r * r.saturating_sub(1) / 2) as i64
    );
    assert_eq!(
        comm.all_gather(10 * r as u32),
        (0..n as u32).map(|i| 10 * i).collect::<Vec<_>>()
    );
    let mut lo = (r as f64).to_ne_bytes();
    comm.exscan_bytes(&mut lo, DType::F64, ReduceOp::Min);
    assert_eq!(f64::from_ne_bytes(lo), if r == 0 { f64::MAX } else { 0.0 });
    let mut hi = (r as i32).to_ne_bytes();
    comm.exscan_bytes(&mut hi, DType::I32, ReduceOp::Max);
    assert_eq!(
        i32::from_ne_bytes(hi),
        if r == 0 { i32::MIN } else { r as i32 - 1 }
    );
    let mut words = [r as u32 + 1, 7].map(u32::to_ne_bytes).concat();
    comm.all_reduce_bytes(&mut words, DType::U32, ReduceOp::Max);
    assert_eq!(words, [n as u32, 7].map(u32::to_ne_bytes).concat());
    // Rank r sends s + 1 copies of 100 r + s to rank s.
    let counts: Vec<usize> = (0..n).map(|s| s + 1).collect();
    let send: Vec<u32> = (0..n)
        .flat_map(|s| vec![(100 * r + s) as u32; s + 1])
        .collect();
    let (recv, recv_counts) = comm.all_to_all_v(&send, &counts).unwrap();
    assert_eq!(recv_counts, vec![r + 1; n]);
    let want: Vec<u32> = (0..n)
        .flat_map(|src| vec![(100 * src + r) as u32; r + 1])
        .collect();
    assert_eq!(recv, want);
    // Empty messages.
    let (recv, recv_counts) = comm.all_to_all_v::<f64>(&[], &vec![0; n]).unwrap();
    assert!(recv.is_empty() && recv_counts == vec![0; n]);
    comm.barrier();
}

/// An asymmetric neighbourhood: rank r sends r + 1 values to r + 1 and receives from r − 1
/// (no wrap-around, so rank 0 has no source and the last rank no destination).
pub fn ring(comm: &dyn CommDyn) {
    let (r, n) = (comm.rank(), comm.size());
    let destinations: Vec<usize> = (r + 1 < n).then_some(r + 1).into_iter().collect();
    let sources: Vec<usize> = r.checked_sub(1).into_iter().collect();
    let nb = Neighbourhood::new(comm, sources.clone(), destinations.clone()).unwrap();
    assert_eq!(
        (nb.sources(), nb.destinations()),
        (&sources[..], &destinations[..])
    );
    let send: Vec<f32> = if destinations.is_empty() {
        Vec::new()
    } else {
        (0..=r).map(|k| (10 * r + k) as f32).collect()
    };
    let counts: Vec<usize> = destinations.iter().map(|_| r + 1).collect();
    let (recv, recv_counts) = comm.neighbour_all_to_all_v(&nb, &send, &counts).unwrap();
    match r.checked_sub(1) {
        Some(src) => {
            assert_eq!(recv_counts, [r]);
            assert_eq!(
                recv,
                (0..r).map(|k| (10 * src + k) as f32).collect::<Vec<_>>()
            );
        }
        None => assert!(recv.is_empty() && recv_counts.is_empty()),
    }
    // A self-loop and an out-of-range rank are agreed errors.
    let bad = Neighbourhood::new(comm, vec![], if r == 0 { vec![n] } else { vec![] });
    match bad {
        Err(GridError::IndexOutOfRange { index, len }) if r == 0 => {
            assert_eq!((index, len), (n, n))
        }
        Err(GridError::RemoteFailure { rank: 0 }) if r > 0 => {}
        Err(e) => panic!("rank {r}: unexpected {e}"),
        Ok(_) => panic!("rank {r}: an out-of-range neighbour was accepted"),
    }
}

/// Count slices of the wrong length or sum are `LengthMismatch`, on every rank. Needs at least
/// two ranks.
pub fn wrong_counts(comm: &dyn CommDyn) {
    let n = comm.size();
    assert!(n >= 2, "wrong_counts needs two ranks");
    assert!(matches!(
        comm.all_to_all_v(&[1_i32], &[1]),
        Err(GridError::LengthMismatch { expected, found: 1 }) if expected == n
    ));
    let mut counts = vec![0; n];
    counts[0] = 1;
    assert!(matches!(
        comm.all_to_all_v(&[1_i32, 2], &counts),
        Err(GridError::LengthMismatch {
            expected: 1,
            found: 2
        })
    ));
}

/// A random directed graph on `n` ranks without self-loops, from the bits of `seed`.
fn graph(n: usize, seed: u64) -> Vec<Vec<bool>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| i != j && (seed >> ((i * n + j) % 64)) & 1 == 1)
                .collect()
        })
        .collect()
}

/// Neighbourhood exchange on the random graph of `seed` (the same on every rank): rank i sends
/// ((j + 1) · (i + 1)) mod 4 values 1000 i + j + k to each destination j.
pub fn neighbour_exchange(comm: &dyn CommDyn, seed: u64) {
    let (r, n) = (comm.rank(), comm.size());
    let g = graph(n, seed);
    let count = |i: usize, j: usize| ((j + 1) * (i + 1)) % 4;
    let dsts: Vec<usize> = (0..n).filter(|&j| g[r][j]).collect();
    let srcs: Vec<usize> = (0..n).filter(|&i| g[i][r]).collect();
    let nb = Neighbourhood::new(comm, srcs.clone(), dsts.clone()).unwrap();
    let counts: Vec<usize> = dsts.iter().map(|&j| count(r, j)).collect();
    let send: Vec<i64> = dsts
        .iter()
        .flat_map(|&j| (0..count(r, j)).map(move |k| (1000 * r + j + k) as i64))
        .collect();
    let (recv, recv_counts) = comm.neighbour_all_to_all_v(&nb, &send, &counts).unwrap();
    let want_counts: Vec<usize> = srcs.iter().map(|&i| count(i, r)).collect();
    let want: Vec<i64> = srcs
        .iter()
        .flat_map(|&i| (0..count(i, r)).map(move |k| (1000 * i + r + k) as i64))
        .collect();
    assert_eq!(recv_counts, want_counts, "rank {r}, seed {seed:#x}");
    assert_eq!(recv, want, "rank {r}, seed {seed:#x}");
}
