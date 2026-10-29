//! `Scatterer` on `ThreadComm` with 1–4 ranks (docs/plans/M6.md, T7): forward and every
//! `ReverseOp` against a host reference on random ghost layouts, the scatter-consistency
//! property, and bitwise repeatability.

use std::fmt::Debug;

use gridcl::{CommDyn, DeviceBuffer, GridError, IndexMap, ReverseOp, ScatterNumeric, Transport};
use gridcl_device::Client;
use gridcl_testing::{ThreadComm, comm_checks::ghost_layout_sized};
use proptest::prelude::*;

/// The scalars under test, with the host arithmetic of the reference.
trait Val: ScatterNumeric + PartialEq + Debug {
    fn of(v: i64) -> Self;
    fn plus(self, o: Self) -> Self;
    fn below(&self, o: &Self) -> bool;
}

macro_rules! val {
    ($($t:ty),*) => {$(
        impl Val for $t {
            fn of(v: i64) -> Self {
                v as $t
            }
            fn plus(self, o: Self) -> Self {
                self + o
            }
            fn below(&self, o: &Self) -> bool {
                self < o
            }
        }
    )*};
}
val!(f32, f64, i32, u32);

/// One rank's map of the layout of `seed`, with the layout itself.
struct Layout {
    map: IndexMap,
    owned: Vec<usize>,
    ghosts: Vec<Vec<(i64, i32)>>,
}

impl Layout {
    fn new(comm: &dyn CommDyn, seed: u64, max_owned: u64) -> Layout {
        let (owned, ghosts) = ghost_layout_sized(comm.size(), seed, max_owned);
        let mine = &ghosts[comm.rank()];
        let map = IndexMap::with_ghosts(
            comm,
            owned[comm.rank()],
            mine.iter().map(|g| g.0).collect(),
            mine.iter().map(|g| g.1).collect(),
        )
        .unwrap();
        Layout { map, owned, ghosts }
    }
}

/// The value the owner of global index `g` holds in component `c`.
fn owner_value(g: i64, c: usize) -> i64 {
    3 * g + c as i64 + 1
}

/// The value rank `s` holds in component `c` of its ghost of `g`.
fn ghost_value(g: i64, s: usize, c: usize) -> i64 {
    5 * g + 7 * s as i64 + c as i64 + 2
}

/// This rank's input: owner values on owned indices, ghost values on ghosts.
fn input<T: Val>(comm: &dyn CommDyn, l: &Layout, block: usize) -> Vec<T> {
    let r = comm.rank();
    let first = l.map.owned_range().0;
    let mut x = Vec::with_capacity(l.map.num_local() * block);
    for i in 0..l.owned[r] {
        x.extend((0..block).map(|c| T::of(owner_value(first + i as i64, c))));
    }
    for &(g, _) in &l.ghosts[r] {
        x.extend((0..block).map(|c| T::of(ghost_value(g, r, c))));
    }
    x
}

/// Forward and every reverse op against the host reference.
fn check<T: Val>(comm: &dyn CommDyn, client: &Client, l: &Layout, block: usize) {
    let (r, n) = (comm.rank(), comm.size());
    let at = format!(
        "rank {r} of {n}, block {block}, {}",
        std::any::type_name::<T>()
    );
    let first = l.map.owned_range().0;
    let num_owned = l.owned[r];
    let x0 = input::<T>(comm, l, block);
    let scatterer = l.map.scatterer(client, block).unwrap();
    assert_eq!(scatterer.transport(), Transport::HostStaged);

    // Forward: ghosts take their owner's values.
    let mut x = DeviceBuffer::from_host(client, &x0).unwrap();
    scatterer.forward(&mut x).unwrap();
    let mut want = x0.clone();
    for (k, &(g, _)) in l.ghosts[r].iter().enumerate() {
        for c in 0..block {
            want[(num_owned + k) * block + c] = T::of(owner_value(g, c));
        }
    }
    assert_eq!(x.to_host().unwrap(), want, "{at}: forward");

    // Reverse: contributions by ascending neighbour rank (each rank ghosts an index once).
    for op in [
        ReverseOp::Add,
        ReverseOp::Min,
        ReverseOp::Max,
        ReverseOp::Insert,
    ] {
        let mut x = DeviceBuffer::from_host(client, &x0).unwrap();
        scatterer.reverse(&mut x, op).unwrap();
        let mut want = x0.clone();
        for i in 0..num_owned {
            let g = first + i as i64;
            let holders: Vec<usize> = (0..n)
                .filter(|&s| l.ghosts[s].iter().any(|&(h, _)| h == g))
                .collect();
            if holders.is_empty() {
                continue;
            }
            for c in 0..block {
                let own = T::of(owner_value(g, c));
                let mut contributions = holders.iter().map(|&s| T::of(ghost_value(g, s, c)));
                want[i * block + c] = match op {
                    ReverseOp::Add => contributions.fold(own, T::plus),
                    ReverseOp::Min => {
                        contributions.fold(own, |a, b| if b.below(&a) { b } else { a })
                    }
                    ReverseOp::Max => {
                        contributions.fold(own, |a, b| if a.below(&b) { b } else { a })
                    }
                    ReverseOp::Insert => contributions.next().unwrap(),
                };
            }
        }
        assert_eq!(x.to_host().unwrap(), want, "{at}: reverse {op:?}");
    }
}

fn check_all_types(comm: &dyn CommDyn, l: &Layout, block: usize) {
    let client = gridcl_testing::test_client();
    check::<f32>(comm, &client, l, block);
    check::<i32>(comm, &client, l, block);
    check::<u32>(comm, &client, l, block);
    if gridcl_testing::supports::<f64>(&client) {
        check::<f64>(comm, &client, l, block);
    } else {
        let s = l.map.scatterer(&client, block).unwrap();
        let mut x = DeviceBuffer::from_host(&client, &input::<f64>(comm, l, block)).unwrap();
        // f64 moves as u32 words, so forward works on every device.
        s.forward(&mut x).unwrap();
        assert!(matches!(
            s.reverse(&mut x, ReverseOp::Add),
            Err(GridError::UnsupportedPrecision { scalar: "f64", .. })
        ));
    }
}

#[test]
fn forward_and_reverse_match_the_host_reference() {
    for n in 1..=4 {
        for seed in [1_u64, 2, 3] {
            ThreadComm::run(n, |comm| {
                let l = Layout::new(&comm, seed, 20);
                for block in [1, 3] {
                    check_all_types(&comm, &l, block);
                }
            });
        }
    }
}

#[test]
fn maps_without_ghosts_and_bad_inputs() {
    ThreadComm::run(2, |comm| {
        let client = gridcl_testing::test_client();
        // IndexMap::new has no exchange; with_ghosts without ghosts has an empty one.
        for map in [
            IndexMap::new(&comm, 3).unwrap(),
            IndexMap::with_ghosts(&comm, 3, vec![], vec![]).unwrap(),
        ] {
            let s = map.scatterer(&client, 2).unwrap();
            let host: Vec<i32> = (0..6).collect();
            let mut x = DeviceBuffer::from_host(&client, &host).unwrap();
            s.forward(&mut x).unwrap();
            s.reverse(&mut x, ReverseOp::Add).unwrap();
            assert_eq!(x.to_host().unwrap(), host);
            let mut short = DeviceBuffer::from_host(&client, &host[..5]).unwrap();
            assert!(matches!(
                s.forward(&mut short),
                Err(GridError::LengthMismatch {
                    expected: 6,
                    found: 5
                })
            ));
            assert!(matches!(
                map.scatterer(&client, 0),
                Err(GridError::LengthMismatch { .. })
            ));
        }
    });
}

#[test]
fn clones_and_second_scatterers_agree() {
    ThreadComm::run(3, |comm| {
        let client = gridcl_testing::test_client();
        let l = Layout::new(&comm, 7, 20);
        let x0 = input::<i32>(&comm, &l, 2);
        let run = |map: &IndexMap| {
            let mut x = DeviceBuffer::from_host(&client, &x0).unwrap();
            map.scatterer(&client, 2)
                .unwrap()
                .reverse(&mut x, ReverseOp::Add)
                .unwrap();
            x.to_host().unwrap()
        };
        // The first scatterer fills the device cache, the second reuses it, the clone has none.
        let a = run(&l.map);
        assert_eq!(run(&l.map), a);
        assert_eq!(run(&l.map.clone()), a);
    });
}

#[test]
fn two_runs_are_bitwise_equal() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            let l = Layout::new(&comm, 11, 200);
            // Values whose sums round, so a change of order would show.
            let x0: Vec<f32> = (0..l.map.num_local() * 3)
                .map(|i| 0.1 + (i as f32).sqrt() * 1e-3 * (comm.rank() + 1) as f32)
                .collect();
            let run = || {
                let mut x = DeviceBuffer::from_host(&client, &x0).unwrap();
                let s = l.map.scatterer(&client, 3).unwrap();
                s.reverse(&mut x, ReverseOp::Add).unwrap();
                s.forward(&mut x).unwrap();
                x.to_host().unwrap()
            };
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(run()), bits(run()));
        });
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// IndexMap scatter consistency: after a forward, every ghost equals its owner; a reverse
    /// `Add` of ones counts the holders of each owned index; a reverse `Insert` of the rank
    /// gives the lowest ghosting rank. Up to 10^5 indices.
    #[test]
    fn consistency(n in 1..=4_usize, seed in any::<u64>(), scale in 0..=4_u32, block in 1..=3_usize) {
        let max_owned = (10_u64.pow(scale) * 25 / n as u64).max(1);
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            let l = Layout::new(&comm, seed, max_owned);
            let (r, local) = (comm.rank(), l.map.num_local());
            let first = l.map.owned_range().0;
            let s = l.map.scatterer(&client, block).unwrap();

            let ids: Vec<i32> = (0..local * block)
                .map(|e| if e / block < l.owned[r] { (first as usize * block + e) as i32 } else { -1 })
                .collect();
            let mut x = DeviceBuffer::from_host(&client, &ids).unwrap();
            s.forward(&mut x).unwrap();
            let x = x.to_host().unwrap();
            for (k, &(g, _)) in l.ghosts[r].iter().enumerate() {
                for c in 0..block {
                    assert_eq!(x[(l.owned[r] + k) * block + c], (g as usize * block + c) as i32);
                }
            }

            let mut ones = DeviceBuffer::from_host(&client, &vec![1_u32; local * block]).unwrap();
            s.reverse(&mut ones, ReverseOp::Add).unwrap();
            let mut ranks = DeviceBuffer::from_host(&client, &vec![r as u32; local * block]).unwrap();
            s.reverse(&mut ranks, ReverseOp::Insert).unwrap();
            let (ones, ranks) = (ones.to_host().unwrap(), ranks.to_host().unwrap());
            let mut holders = vec![Vec::new(); l.owned[r]];
            for (t, theirs) in l.ghosts.iter().enumerate() {
                for &(g, o) in theirs {
                    if o as usize == r {
                        holders[(g - first) as usize].push(t as u32);
                    }
                }
            }
            for e in 0..local * block {
                let i = e / block;
                let (count, lowest) = match holders.get(i) {
                    Some(h) => (1 + h.len() as u32, h.first().copied().unwrap_or(r as u32)),
                    None => (1, r as u32),
                };
                assert_eq!(ones[e], count, "rank {r}: Add at {e}");
                assert_eq!(ranks[e], lowest, "rank {r}: Insert at {e}");
            }
        });
    }
}
