//! `ThreadComm`: an in-process communicator with one thread per rank, so that multi-rank code
//! runs in `cargo test` (docs/plans/M6.md, T2).
//!
//! Every collective is built from point-to-point messages through one mailbox per (source,
//! destination) pair. Mailboxes are FIFO, so the messages of successive collectives match up as
//! long as every rank calls the same collectives in the same order, which `CommDyn` requires
//! anyway. Reductions and scans combine the contributions in rank order, so their results are
//! deterministic.

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::rc::Rc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use gridcl::{CommDyn, DType, GridError, Neighbourhood, Result};
use gridcl_device::ReduceOp;

/// The state the ranks of one [`ThreadComm::run`] share.
struct Shared {
    size: usize,
    state: Mutex<State>,
    wake: Condvar,
}

struct State {
    /// The messages from `src` to `dst` at `src * size + dst`, oldest first.
    queues: Vec<VecDeque<Vec<u8>>>,
    /// A rank has panicked: every waiting rank gives up.
    poisoned: bool,
}

/// The panic message of a rank that gives up because another rank panicked.
const POISONED: &str = "ThreadComm: another rank panicked";

/// One rank of an in-process group of ranks, each on its own thread; made by
/// [`ThreadComm::run`].
///
/// All `CommDyn` operations are supported: `raw()` (feature `mpi`) is `None`, so a
/// [`Neighbourhood`] on a `ThreadComm` keeps its rank lists, and `duplicate()` is a clone
/// that refers to the same group.
///
/// The `gridcl_device::testing::with_*` overrides are thread-local and are not propagated to
/// the rank threads: set them inside the closure, or use `GRIDCL_VECTOR_SIZE`.
///
/// # Example
///
/// ```
/// use gridcl::{CommDyn, CommExt};
/// use gridcl_testing::ThreadComm;
///
/// let sums = ThreadComm::run(3, |comm| comm.sum(comm.rank() as u64));
/// assert_eq!(sums, [3, 3, 3]);
/// let before = ThreadComm::run(3, |comm| comm.exscan_sum(1_i32));
/// assert_eq!(before, [0, 1, 2]);
/// ```
#[derive(Clone)]
pub struct ThreadComm {
    rank: usize,
    shared: Arc<Shared>,
}

impl ThreadComm {
    /// Runs `f` on `n` ranks, one thread each, and returns the results in rank order.
    ///
    /// # Panics
    ///
    /// If `n` is 0, or if `f` panics on any rank: the other ranks stop at their next
    /// communication, and the first panic that is not such a stop is resumed here.
    pub fn run<R: Send>(n: usize, f: impl Fn(ThreadComm) -> R + Sync) -> Vec<R> {
        assert!(n > 0, "ThreadComm::run needs at least one rank");
        let shared = Arc::new(Shared {
            size: n,
            state: Mutex::new(State {
                queues: (0..n * n).map(|_| VecDeque::new()).collect(),
                poisoned: false,
            }),
            wake: Condvar::new(),
        });
        let f = &f;
        let outcomes: Vec<std::thread::Result<R>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n)
                .map(|rank| {
                    let comm = ThreadComm {
                        rank,
                        shared: Arc::clone(&shared),
                    };
                    let shared = Arc::clone(&shared);
                    std::thread::Builder::new()
                        .name(format!("rank {rank}"))
                        .spawn_scoped(scope, move || {
                            let out = catch_unwind(AssertUnwindSafe(|| f(comm)));
                            if out.is_err() {
                                shared.lock().poisoned = true;
                                shared.wake.notify_all();
                            }
                            out
                        })
                        .expect("spawn a rank thread")
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("rank threads catch their panics"))
                .collect()
        });
        let mut results = Vec::with_capacity(n);
        let mut stop: Option<Box<dyn Any + Send>> = None;
        let mut first: Option<Box<dyn Any + Send>> = None;
        for outcome in outcomes {
            match outcome {
                Ok(r) => results.push(r),
                Err(p) if is_stop(&p) => stop = stop.or(Some(p)),
                Err(p) => first = first.or(Some(p)),
            }
        }
        if let Some(p) = first.or(stop) {
            resume_unwind(p);
        }
        results
    }

    /// Sends `bytes` to `dst`.
    fn send(&self, dst: usize, bytes: &[u8]) {
        let size = self.shared.size;
        self.shared.lock().queues[self.rank * size + dst].push_back(bytes.to_vec());
        self.shared.wake.notify_all();
    }

    /// The oldest message from `src`; waits for it.
    ///
    /// # Panics
    ///
    /// If another rank has panicked.
    fn recv(&self, src: usize) -> Vec<u8> {
        let size = self.shared.size;
        let mut state = self.shared.lock();
        loop {
            if let Some(m) = state.queues[src * size + self.rank].pop_front() {
                return m;
            }
            if state.poisoned {
                drop(state);
                panic!("{POISONED}");
            }
            state = self
                .shared
                .wake
                .wait(state)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Sends `buf` to every rank and returns every rank's buffer, in rank order.
    fn exchange_all(&self, buf: &[u8]) -> Vec<Vec<u8>> {
        for dst in 0..self.shared.size {
            self.send(dst, buf);
        }
        (0..self.shared.size).map(|src| self.recv(src)).collect()
    }
}

/// Whether a panic payload is a rank giving up after another rank's panic.
fn is_stop(payload: &Box<dyn Any + Send>) -> bool {
    payload
        .downcast_ref::<String>()
        .is_some_and(|m| m == POISONED)
        || payload
            .downcast_ref::<&str>()
            .is_some_and(|m| *m == POISONED)
}

impl Shared {
    /// The state, also after a panic on a thread that held the lock.
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// The size of one element of `dtype` in bytes.
fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
    }
}

/// `acc[i] = op(acc[i], x[i])` elementwise over native-order elements of `dtype`.
fn combine(acc: &mut [u8], x: &[u8], dtype: DType, op: ReduceOp) {
    macro_rules! fold {
        ($t:ty, $sum:expr) => {{
            const N: usize = size_of::<$t>();
            for (a, b) in acc.chunks_exact_mut(N).zip(x.chunks_exact(N)) {
                let u = <$t>::from_ne_bytes(a.try_into().unwrap());
                let v = <$t>::from_ne_bytes(b.try_into().unwrap());
                let r: $t = match op {
                    ReduceOp::Sum => $sum(u, v),
                    ReduceOp::Min => u.min(v),
                    ReduceOp::Max => u.max(v),
                };
                a.copy_from_slice(&r.to_ne_bytes());
            }
        }};
    }
    match dtype {
        DType::I32 => fold!(i32, i32::wrapping_add),
        DType::I64 => fold!(i64, i64::wrapping_add),
        DType::U32 => fold!(u32, u32::wrapping_add),
        DType::U64 => fold!(u64, u64::wrapping_add),
        DType::F32 => fold!(f32, |u: f32, v: f32| u + v),
        DType::F64 => fold!(f64, |u: f64, v: f64| u + v),
    }
}

/// Every element of `buf` set to the identity of `op` for `dtype`: 0 for `Sum`, the largest
/// finite value for `Min` and the smallest finite value for `Max` (ADR-0011).
fn fill_identity(buf: &mut [u8], dtype: DType, op: ReduceOp) {
    macro_rules! fill {
        ($t:ty) => {{
            let v: $t = match op {
                ReduceOp::Sum => 0 as $t,
                ReduceOp::Min => <$t>::MAX,
                ReduceOp::Max => <$t>::MIN,
            };
            for e in buf.chunks_exact_mut(size_of::<$t>()) {
                e.copy_from_slice(&v.to_ne_bytes());
            }
        }};
    }
    match dtype {
        DType::I32 => fill!(i32),
        DType::I64 => fill!(i64),
        DType::U32 => fill!(u32),
        DType::U64 => fill!(u64),
        DType::F32 => fill!(f32),
        DType::F64 => fill!(f64),
    }
}

/// `Ok` if `found == expected`, else `LengthMismatch`.
fn check_len(expected: usize, found: usize) -> Result<()> {
    if expected == found {
        Ok(())
    } else {
        Err(GridError::LengthMismatch { expected, found })
    }
}

/// The byte offsets of `counts` elements of `dtype`, and the total; `LengthMismatch` unless
/// the total is `len` bytes, `TooLarge` on overflow.
fn displacements(counts: &[usize], dtype: DType, len: usize) -> Result<Vec<usize>> {
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    let mut at = 0_usize;
    offsets.push(0);
    for &c in counts {
        let bytes = c
            .checked_mul(dtype_size(dtype))
            .ok_or(GridError::TooLarge)?;
        at = at.checked_add(bytes).ok_or(GridError::TooLarge)?;
        offsets.push(at);
    }
    check_len(at, len)?;
    Ok(offsets)
}

impl ThreadComm {
    /// Sends `send` split by `send_counts` to `dsts` and fills `recv` from `srcs` by
    /// `recv_counts`. The lengths are checked before any message is sent; a received message of
    /// the wrong length gives `LengthMismatch` (in elements).
    #[allow(clippy::too_many_arguments)]
    fn exchange_v(
        &self,
        dsts: &[usize],
        srcs: &[usize],
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()> {
        check_len(dsts.len(), send_counts.len())?;
        check_len(srcs.len(), recv_counts.len())?;
        let out = displacements(send_counts, dtype, send.len())?;
        let inn = displacements(recv_counts, dtype, recv.len())?;
        for (k, &dst) in dsts.iter().enumerate() {
            self.send(dst, &send[out[k]..out[k + 1]]);
        }
        let mut result = Ok(());
        for (k, &src) in srcs.iter().enumerate() {
            let m = self.recv(src);
            let slot = &mut recv[inn[k]..inn[k + 1]];
            if m.len() == slot.len() {
                slot.copy_from_slice(&m);
            } else if result.is_ok() {
                result = check_len(recv_counts[k], m.len() / dtype_size(dtype));
            }
        }
        result
    }
}

impl CommDyn for ThreadComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn size(&self) -> usize {
        self.shared.size
    }
    fn barrier(&self) {
        self.exchange_all(&[]);
    }
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        debug_assert!(buf.len().is_multiple_of(dtype_size(dtype)));
        let all = self.exchange_all(buf);
        buf.copy_from_slice(&all[0]);
        for x in &all[1..] {
            combine(buf, x, dtype, op);
        }
    }
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        debug_assert!(buf.len().is_multiple_of(dtype_size(dtype)));
        let all = self.exchange_all(buf);
        fill_identity(buf, dtype, op);
        if self.rank > 0 {
            buf.copy_from_slice(&all[0]);
            for x in &all[1..self.rank] {
                combine(buf, x, dtype, op);
            }
        }
    }
    fn all_to_all_v_bytes(
        &self,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()> {
        let ranks: Vec<usize> = (0..self.shared.size).collect();
        self.exchange_v(&ranks, &ranks, send, send_counts, recv, recv_counts, dtype)
    }
    fn neighbour_all_to_all_v_bytes(
        &self,
        nb: &Neighbourhood,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()> {
        self.exchange_v(
            nb.destinations(),
            nb.sources(),
            send,
            send_counts,
            recv,
            recv_counts,
            dtype,
        )
    }
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        None
    }
    fn duplicate(&self) -> Rc<dyn CommDyn> {
        Rc::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comm_checks;
    use proptest::prelude::*;

    #[test]
    fn one_rank_matches_serial() {
        ThreadComm::run(1, |comm| comm_checks::matches_serial(&comm));
    }

    #[test]
    fn identities_on_one_to_five_ranks() {
        for n in 1..=5 {
            let out = ThreadComm::run(n, |comm| {
                assert_eq!(comm.size(), n);
                comm_checks::identities(&comm);
                comm_checks::ring(&comm);
                comm.rank()
            });
            assert_eq!(out, (0..n).collect::<Vec<_>>());
        }
    }

    #[test]
    fn wrong_counts_are_length_mismatch() {
        ThreadComm::run(2, |comm| comm_checks::wrong_counts(&comm));
    }

    #[test]
    fn a_panic_on_one_rank_ends_every_rank() {
        let result = std::panic::catch_unwind(|| {
            ThreadComm::run(3, |comm| {
                if comm.rank() == 1 {
                    panic!("rank 1 fails");
                }
                // The others wait in a collective that rank 1 never joins.
                comm.barrier();
            })
        });
        let payload = result.expect_err("the panic is resumed");
        let message = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned());
        assert_eq!(message.as_deref(), Some("rank 1 fails"));
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn neighbour_exchange_on_random_graphs(n in 1..=5_usize, seed in any::<u64>()) {
            ThreadComm::run(n, |comm| comm_checks::neighbour_exchange(&comm, seed));
        }
    }
}
