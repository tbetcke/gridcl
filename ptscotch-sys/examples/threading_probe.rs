//! The threading probe of ADR-0029, item 13: PT-Scotch under MPI's `Threading::Single`.
//!
//! A PT-Scotch built with `SCOTCH_PTHREAD_MPI` needs `MPI_THREAD_MULTIPLE` and fails in
//! `SCOTCH_dgraphInit`. This program initialises MPI at `Threading::Single`, builds a ring of 8
//! vertices split over the ranks in blocks and partitions it into 2 parts with
//! `SCOTCH_dgraphPart`. Every call must succeed and the parts must hold 4 vertices each.
//!
//! ```text
//! cargo build -p ptscotch-sys --features mpi --example threading_probe
//! mpirun -n 2 target/debug/examples/threading_probe
//! ```

use std::mem::MaybeUninit;
use std::ptr;

use mpi::Threading;
use mpi::collective::SystemOperation;
use mpi::raw::AsRaw;
use mpi::traits::*;
use ptscotch_sys::*;

const VERTICES: SCOTCH_Num = 8;
const PARTS: SCOTCH_Num = 2;

fn main() {
    let (universe, threading) =
        mpi::initialize_with_threading(Threading::Single).expect("MPI initialisation");
    let world = universe.world();
    let (rank, size) = (
        SCOTCH_Num::from(world.rank()),
        SCOTCH_Num::from(world.size()),
    );

    // Vertices first..last of the ring, each joined to its two neighbours (global numbers,
    // base 0).
    let first = VERTICES * rank / size;
    let last = VERTICES * (rank + 1) / size;
    let mut vertloctab: Vec<SCOTCH_Num> = (0..=last - first).map(|i| 2 * i).collect();
    let mut edgeloctab: Vec<SCOTCH_Num> = (first..last)
        .flat_map(|v| [(v + VERTICES - 1) % VERTICES, (v + 1) % VERTICES])
        .collect();
    let vertlocnbr = last - first;
    let edgelocnbr = 2 * vertlocnbr;
    let mut partloctab: Vec<SCOTCH_Num> = vec![-1; (last - first) as usize];

    let mut dgraph = MaybeUninit::<SCOTCH_Dgraph>::uninit();
    let mut strat = MaybeUninit::<SCOTCH_Strat>::uninit();
    // SAFETY: dgraph is writable storage of the size the header gives, initialised here before
    // any other use; world's raw handle is a valid communicator of every rank for the program's
    // lifetime.
    let status = unsafe { SCOTCH_dgraphInit(dgraph.as_mut_ptr(), world.as_raw()) };
    assert_eq!(
        status, 0,
        "rank {rank}: SCOTCH_dgraphInit under {threading:?}"
    );
    // SAFETY: dgraph was initialised above; the arrays hold vertlocnbr + 1 and edgelocnbr
    // entries, outlive the graph (exited below) and describe a valid symmetric distributed
    // graph; null optional arrays mean compact vertices, unit loads and no labels.
    let status = unsafe {
        SCOTCH_dgraphBuild(
            dgraph.as_mut_ptr(),
            0,
            vertlocnbr,
            vertlocnbr,
            vertloctab.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
            ptr::null_mut(),
            edgelocnbr,
            edgelocnbr,
            edgeloctab.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };
    assert_eq!(status, 0, "rank {rank}: SCOTCH_dgraphBuild");
    // SAFETY: dgraph is a built graph.
    let status = unsafe { SCOTCH_dgraphCheck(dgraph.as_ptr()) };
    assert_eq!(status, 0, "rank {rank}: SCOTCH_dgraphCheck");
    // SAFETY: strat is writable storage of the header's size, initialised here first.
    let status = unsafe { SCOTCH_stratInit(strat.as_mut_ptr()) };
    assert_eq!(status, 0, "rank {rank}: SCOTCH_stratInit");
    // SAFETY: strat was initialised above.
    let status = unsafe {
        SCOTCH_stratDgraphMapBuild(strat.as_mut_ptr(), SCOTCH_STRATQUALITY, size, PARTS, 0.03)
    };
    assert_eq!(status, 0, "rank {rank}: SCOTCH_stratDgraphMapBuild");
    // SAFETY: dgraph and strat are initialised; partloctab holds one entry per local vertex.
    let status = unsafe {
        SCOTCH_dgraphPart(
            dgraph.as_mut_ptr(),
            PARTS,
            strat.as_mut_ptr(),
            partloctab.as_mut_ptr(),
        )
    };
    assert_eq!(status, 0, "rank {rank}: SCOTCH_dgraphPart");
    // SAFETY: both were initialised above and are not used afterwards.
    unsafe {
        SCOTCH_stratExit(strat.as_mut_ptr());
        SCOTCH_dgraphExit(dgraph.as_mut_ptr());
    }

    let mut counts = [0i64; PARTS as usize];
    for &part in &partloctab {
        assert!((0..PARTS).contains(&part), "rank {rank}: part {part}");
        counts[part as usize] += 1;
    }
    let mut totals = [0i64; PARTS as usize];
    world.all_reduce_into(&counts[..], &mut totals[..], SystemOperation::sum());
    assert_eq!(totals, [4, 4], "rank {rank}: part sizes");
    println!(
        "threading_probe: rank {rank} of {size} ok (threading {threading:?}, \
         SCOTCH_Num {SCOTCH_NUM_BITS} bits, parts {partloctab:?})"
    );
}
