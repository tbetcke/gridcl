//! `ThreadComm`: an in-process communicator with one thread per rank, so that multi-rank code
//! runs in `cargo test` (docs/plans/M6.md, T2).

use std::rc::Rc;

use gridcl::{CommDyn, DType, Neighbourhood, Result};
use gridcl_device::ReduceOp;

/// One rank of an in-process group of ranks, each on its own thread; made by
/// [`ThreadComm::run`].
///
/// The `gridcl_device::testing::with_*` overrides are thread-local and are not propagated to
/// the rank threads: set them inside the closure, or use `GRIDCL_VECTOR_SIZE`.
#[derive(Clone)]
pub struct ThreadComm {
    rank: usize,
    size: usize,
}

impl ThreadComm {
    /// Runs `f` on `n` ranks, one thread each, and returns the results in rank order. A panic
    /// on any rank ends every rank and is resumed here.
    pub fn run<R: Send>(n: usize, f: impl Fn(ThreadComm) -> R + Sync) -> Vec<R> {
        let _ = (n, f);
        todo!("M6 T2: ThreadComm::run")
    }
}

impl CommDyn for ThreadComm {
    fn rank(&self) -> usize {
        self.rank
    }
    fn size(&self) -> usize {
        self.size
    }
    fn barrier(&self) {
        todo!("M6 T2: ThreadComm::barrier")
    }
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let _ = (buf, dtype, op);
        todo!("M6 T2: ThreadComm::all_reduce_bytes")
    }
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let _ = (buf, dtype, op);
        todo!("M6 T2: ThreadComm::exscan_bytes")
    }
    fn all_to_all_v_bytes(
        &self,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()> {
        let _ = (send, send_counts, recv, recv_counts, dtype);
        todo!("M6 T2: ThreadComm::all_to_all_v_bytes")
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
        let _ = (nb, send, send_counts, recv, recv_counts, dtype);
        todo!("M6 T2: ThreadComm::neighbour_all_to_all_v_bytes")
    }
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        None
    }
    fn duplicate(&self) -> Rc<dyn CommDyn> {
        todo!("M6 T2: ThreadComm::duplicate")
    }
}
