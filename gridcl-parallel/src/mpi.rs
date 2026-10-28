//! `Mpi`: the MPI communicator (feature `mpi`).

use std::rc::Rc;

use mpi::topology::SimpleCommunicator;

use crate::{CommDyn, DType, Neighbourhood, ReduceOp, Result};

/// An MPI communicator. Wraps it in an `Rc`, so it is deliberately not `Send` (docs/design.md,
/// "MPI bindings").
#[derive(Clone)]
pub struct Mpi(#[allow(dead_code)] /* read from M6 T3 */ Rc<SimpleCommunicator>);

impl Mpi {
    /// The world communicator of `universe`.
    pub fn world(universe: &mpi::environment::Universe) -> Mpi {
        let _ = universe;
        todo!("M6 T3: Mpi::world")
    }
    /// A communicator from rsmpi's.
    pub fn new(comm: SimpleCommunicator) -> Mpi {
        Mpi(Rc::new(comm))
    }
}

impl CommDyn for Mpi {
    fn rank(&self) -> usize {
        todo!("M6 T3: Mpi::rank")
    }
    fn size(&self) -> usize {
        todo!("M6 T3: Mpi::size")
    }
    fn barrier(&self) {
        todo!("M6 T3: Mpi::barrier")
    }
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let _ = (buf, dtype, op);
        todo!("M6 T3: Mpi::all_reduce_bytes")
    }
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let _ = (buf, dtype, op);
        todo!("M6 T3: Mpi::exscan_bytes")
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
        todo!("M6 T3: Mpi::all_to_all_v_bytes")
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
        todo!("M6 T3: Mpi::neighbour_all_to_all_v_bytes")
    }
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        todo!("M6 T3: Mpi::raw")
    }
    fn duplicate(&self) -> Rc<dyn CommDyn> {
        todo!("M6 T3: Mpi::duplicate")
    }
}
