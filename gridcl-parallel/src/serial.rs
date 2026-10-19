//! `Serial`: the one-rank communicator with no-op collectives.

use crate::{CommDyn, DType, Neighbourhood, ReduceOp, Result};

/// One rank; collectives are no-ops.
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

impl CommDyn for Serial {
    fn rank(&self) -> usize {
        todo!()
    }
    fn size(&self) -> usize {
        todo!()
    }
    fn barrier(&self) {
        todo!()
    }
    fn all_reduce_bytes(&self, _buf: &mut [u8], _dtype: DType, _op: ReduceOp) {
        todo!()
    }
    fn exscan_bytes(&self, _buf: &mut [u8], _dtype: DType, _op: ReduceOp) {
        todo!()
    }
    fn all_to_all_v_bytes(
        &self,
        _send: &[u8],
        _send_counts: &[usize],
        _recv: &mut [u8],
        _recv_counts: &[usize],
        _dtype: DType,
    ) -> Result<()> {
        todo!()
    }
    fn neighbour_all_to_all_v_bytes(
        &self,
        _nb: &Neighbourhood,
        _send: &[u8],
        _send_counts: &[usize],
        _recv: &mut [u8],
        _recv_counts: &[usize],
        _dtype: DType,
    ) -> Result<()> {
        todo!()
    }
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        todo!()
    }
}
