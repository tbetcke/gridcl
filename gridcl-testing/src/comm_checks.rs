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

/// The random index-map layout of `seed` on `n` ranks: per rank the owned count, and its ghosts
/// as (global index, owner) in ghost order. Every rank computes the same layout.
pub fn ghost_layout(n: usize, seed: u64) -> (Vec<usize>, Vec<Vec<(i64, i32)>>) {
    ghost_layout_sized(n, seed, 20)
}

/// [`ghost_layout`] with fewer than `max_owned` owned indices per rank.
pub fn ghost_layout_sized(
    n: usize,
    seed: u64,
    max_owned: u64,
) -> (Vec<usize>, Vec<Vec<(i64, i32)>>) {
    let mut state = seed;
    let mut next = move |m: u64| {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) % m.max(1)) as usize
    };
    let owned: Vec<usize> = (0..n).map(|_| next(max_owned)).collect();
    let first: Vec<usize> = owned
        .iter()
        .scan(0, |at, &c| {
            let f = *at;
            *at += c;
            Some(f)
        })
        .collect();
    let ghosts = (0..n)
        .map(|r| {
            let mut g: Vec<(i64, i32)> = (0..n)
                .filter(|&s| s != r)
                .flat_map(|s| (0..owned[s]).map(move |i| (s, i)))
                .filter(|_| next(3) == 0)
                .map(|(s, i)| ((first[s] + i) as i64, s as i32))
                .collect();
            for i in (1..g.len()).rev() {
                g.swap(i, next(i as u64 + 1));
            }
            g
        })
        .collect();
    (owned, ghosts)
}

/// `IndexMap::with_ghosts` on the layout of `seed`: owned ranges, ghosts, the symmetry of the
/// forward neighbourhoods, and the send and receive lists against a host reference.
pub fn index_map_layout(comm: &dyn CommDyn, seed: u64) {
    let (r, n) = (comm.rank(), comm.size());
    let (owned, ghosts) = ghost_layout(n, seed);
    let first: i64 = owned[..r].iter().sum::<usize>() as i64;
    let mine = &ghosts[r];
    let map = gridcl::IndexMap::with_ghosts(
        comm,
        owned[r],
        mine.iter().map(|g| g.0).collect(),
        mine.iter().map(|g| g.1).collect(),
    )
    .unwrap();
    let at = format!("rank {r} of {n}, seed {seed:#x}");
    assert_eq!(map.owned_range(), (first, first + owned[r] as i64), "{at}");
    assert_eq!(
        map.size_global(),
        owned.iter().sum::<usize>() as i64,
        "{at}"
    );
    assert_eq!(map.num_local(), owned[r] + mine.len(), "{at}");
    assert_eq!(
        map.ghosts(),
        mine.iter().map(|g| g.0).collect::<Vec<_>>(),
        "{at}"
    );

    // Send lists: for each rank s ghosting this rank's indices, in ascending s, the local
    // indices in s's ghost order.
    let (mut dsts, mut counts, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for (s, theirs) in ghosts.iter().enumerate() {
        let want: Vec<u32> = theirs
            .iter()
            .filter(|g| g.1 as usize == r)
            .map(|g| (g.0 - first) as u32)
            .collect();
        if !want.is_empty() {
            dsts.push(s);
            counts.push(want.len());
            idx.extend(want);
        }
    }
    let (d, c, i) = map.send_lists();
    assert_eq!(
        (d, c, i),
        (&dsts[..], &counts[..], &idx[..]),
        "{at}: send lists"
    );

    // Receive lists: owners in ascending order, ghost positions ascending within each.
    let (mut srcs, mut counts, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..n {
        let want: Vec<u32> = (0..mine.len())
            .filter(|&k| mine[k].1 as usize == s)
            .map(|k| (owned[r] + k) as u32)
            .collect();
        if !want.is_empty() {
            srcs.push(s);
            counts.push(want.len());
            idx.extend(want);
        }
    }
    let (s, c, i) = map.recv_lists();
    assert_eq!(
        (s, c, i),
        (&srcs[..], &counts[..], &idx[..]),
        "{at}: receive lists"
    );

    // Symmetry: s sends to r exactly when r receives from s.
    let sends: Vec<u64> = (0..n)
        .map(|s| u64::from(map.send_lists().0.contains(&s)))
        .collect();
    let (all, _) = comm.all_to_all_v(&sends, &vec![1; n]).unwrap();
    for (s, &sent) in all.iter().enumerate() {
        assert_eq!(
            sent == 1,
            map.recv_lists().0.contains(&s),
            "{at}: symmetry with {s}"
        );
    }

    // A clone shares the pattern.
    let copy = map.clone();
    assert_eq!(copy.send_lists(), map.send_lists());
    assert_eq!(copy.ghost_owners(), map.ghost_owners());
}

/// `with_ghosts` without ghosts equals `new`, and a request for an index the owner does not
/// own is an agreed `IndexOutOfRange` on the owner. Needs at least two ranks for the error.
pub fn index_map_basics(comm: &dyn CommDyn) {
    let (r, n) = (comm.rank(), comm.size());
    let a = gridcl::IndexMap::new(comm, r + 2).unwrap();
    let b = gridcl::IndexMap::with_ghosts(comm, r + 2, vec![], vec![]).unwrap();
    assert_eq!(
        (a.owned_range(), a.size_global(), a.num_ghosts()),
        (b.owned_range(), b.size_global(), b.num_ghosts())
    );
    assert!(b.send_lists().0.is_empty() && b.recv_lists().0.is_empty());
    if n < 2 {
        return;
    }
    // Rank 0 asks rank 1 for an index far past the global size.
    let (ghosts, owners) = if r == 0 {
        (vec![1000], vec![1])
    } else {
        (vec![], vec![])
    };
    match gridcl::IndexMap::with_ghosts(comm, 1, ghosts, owners) {
        Err(GridError::IndexOutOfRange { index: 1000, len }) if r == 1 => assert_eq!(len, n),
        Err(GridError::RemoteFailure { rank: 1 }) if r != 1 => {}
        Err(e) => panic!("rank {r}: unexpected {e}"),
        Ok(_) => panic!("rank {r}: an unowned index was accepted"),
    }
}
