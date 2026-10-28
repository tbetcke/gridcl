//! A fake communicator for the unit tests of the rank checks.

use gridcl_device::Result;
use gridcl_parallel::{CommDyn, DType, Neighbourhood, ReduceOp};

/// A communicator of two ranks, rank 0, on which every collective panics: for the one-rank
/// checks of the readers and writers, which refuse it before any collective.
#[derive(Clone)]
pub(crate) struct TwoRanks;

impl CommDyn for TwoRanks {
    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        2
    }
    fn barrier(&self) {
        unreachable!()
    }
    fn all_reduce_bytes(&self, _: &mut [u8], _: DType, _: ReduceOp) {
        unreachable!()
    }
    fn exscan_bytes(&self, _: &mut [u8], _: DType, _: ReduceOp) {
        unreachable!()
    }
    fn all_to_all_v_bytes(
        &self,
        _: &[u8],
        _: &[usize],
        _: &mut [u8],
        _: &[usize],
        _: DType,
    ) -> Result<()> {
        unreachable!()
    }
    fn neighbour_all_to_all_v_bytes(
        &self,
        _: &Neighbourhood,
        _: &[u8],
        _: &[usize],
        _: &mut [u8],
        _: &[usize],
        _: DType,
    ) -> Result<()> {
        unreachable!()
    }
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        None
    }
}
