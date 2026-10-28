//! `Mpi`: the MPI communicator (feature `mpi`).

use std::rc::Rc;

use mpi::collective::SystemOperation;
use mpi::datatype::{Equivalence, Partition, PartitionMut};
use mpi::ffi::MPI_Datatype;
use mpi::raw::AsRaw;
use mpi::topology::SimpleCommunicator;
use mpi::traits::{Communicator, CommunicatorCollectives};

use crate::comm::{dtype_size, identity_bytes};
use crate::counts::checked_counts;
use crate::{CommDyn, DType, GridError, Neighbourhood, ReduceOp, Result};

/// An MPI communicator. It wraps rsmpi's communicator in an `Rc`, so it is deliberately not
/// `Send` (docs/design.md, "MPI bindings"); clones and [`CommDyn::duplicate`] refer to the
/// same communicator.
///
/// - `exscan_bytes` writes the identity of the operation on rank 0 (ADR-0011).
/// - Counts or displacements above `i32::MAX` give `GridError::MessageTooLarge`.
/// - Neighbourhood exchanges run on the graph communicator of a [`Neighbourhood`] made on
///   this communicator (`MPI_Neighbor_alltoallv`).
///
/// # Example
///
/// ```no_run
/// use gridcl_parallel::{CommDyn, CommExt, Mpi};
///
/// let universe = mpi::initialize().expect("MPI is not initialised yet");
/// let comm = Mpi::world(&universe);
/// let n = comm.sum(1_u64);
/// assert_eq!(n as usize, comm.size());
/// ```
#[derive(Clone)]
pub struct Mpi(Rc<SimpleCommunicator>);

impl Mpi {
    /// The world communicator of `universe`.
    pub fn world(universe: &mpi::environment::Universe) -> Mpi {
        Mpi(Rc::new(universe.world()))
    }
    /// A communicator from rsmpi's.
    pub fn new(comm: SimpleCommunicator) -> Mpi {
        Mpi(Rc::new(comm))
    }
}

/// rsmpi's operation of `op`.
fn system_op(op: ReduceOp) -> SystemOperation {
    match op {
        ReduceOp::Sum => SystemOperation::sum(),
        ReduceOp::Min => SystemOperation::min(),
        ReduceOp::Max => SystemOperation::max(),
    }
}

/// Calls `$f::<T>($args)` with T the Rust type of `$dtype`.
macro_rules! with_type {
    ($dtype:expr, $f:ident($($arg:expr),*)) => {
        match $dtype {
            DType::I32 => $f::<i32>($($arg),*),
            DType::I64 => $f::<i64>($($arg),*),
            DType::U32 => $f::<u32>($($arg),*),
            DType::U64 => $f::<u64>($($arg),*),
            DType::F32 => $f::<f32>($($arg),*),
            DType::F64 => $f::<f64>($($arg),*),
        }
    };
}

/// The MPI datatype of `T`.
fn datatype_of<T: Equivalence>() -> MPI_Datatype {
    T::equivalent_datatype().as_raw()
}

/// The elements of `bytes`, which may be unaligned for `T`.
fn elements<T: bytemuck::Pod>(bytes: &[u8]) -> Vec<T> {
    bytes
        .chunks_exact(size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn all_reduce_typed<T: bytemuck::Pod + Equivalence>(
    comm: &SimpleCommunicator,
    buf: &mut [u8],
    op: ReduceOp,
) {
    let send: Vec<T> = elements(buf);
    let mut recv = send.clone();
    comm.all_reduce_into(&send[..], &mut recv[..], system_op(op));
    buf.copy_from_slice(bytemuck::cast_slice(&recv));
}

fn exscan_typed<T: bytemuck::Pod + Equivalence>(
    comm: &SimpleCommunicator,
    buf: &mut [u8],
    op: ReduceOp,
) {
    let send: Vec<T> = elements(buf);
    let mut recv = send.clone();
    comm.exclusive_scan_into(&send[..], &mut recv[..], system_op(op));
    buf.copy_from_slice(bytemuck::cast_slice(&recv));
}

/// `LengthMismatch` unless `counts` has `n` entries whose elements fill `len` bytes.
fn check_counts(counts: &[usize], n: usize, len: usize, dtype: DType) -> Result<()> {
    if counts.len() != n {
        return Err(GridError::LengthMismatch {
            expected: n,
            found: counts.len(),
        });
    }
    let elements = counts
        .iter()
        .try_fold(0_usize, |a, &c| a.checked_add(c))
        .ok_or(GridError::TooLarge)?;
    let bytes = elements
        .checked_mul(dtype_size(dtype))
        .ok_or(GridError::TooLarge)?;
    if bytes != len {
        return Err(GridError::LengthMismatch {
            expected: bytes,
            found: len,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn all_to_all_v_typed<T: bytemuck::Pod + Equivalence>(
    comm: &SimpleCommunicator,
    send: &[u8],
    send_counts: &[i32],
    send_displs: &[i32],
    recv: &mut [u8],
    recv_counts: &[i32],
    recv_displs: &[i32],
) {
    let send: Vec<T> = elements(send);
    let mut out = vec![T::zeroed(); recv.len() / size_of::<T>()];
    {
        let s = Partition::new(&send[..], send_counts, send_displs);
        let mut r = PartitionMut::new(&mut out[..], recv_counts, recv_displs);
        comm.all_to_all_varcount_into(&s, &mut r);
    }
    recv.copy_from_slice(bytemuck::cast_slice(&out));
}

impl CommDyn for Mpi {
    fn rank(&self) -> usize {
        self.0.rank() as usize
    }
    fn size(&self) -> usize {
        self.0.size() as usize
    }
    fn barrier(&self) {
        self.0.barrier();
    }
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        debug_assert!(buf.len().is_multiple_of(dtype_size(dtype)));
        with_type!(dtype, all_reduce_typed(&self.0, buf, op));
    }
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let size = dtype_size(dtype);
        debug_assert!(buf.len().is_multiple_of(size));
        with_type!(dtype, exscan_typed(&self.0, buf, op));
        // MPI leaves rank 0's result undefined; gridcl defines it as the identity (ADR-0011).
        if self.rank() == 0 {
            let identity = identity_bytes(dtype, op);
            for element in buf.chunks_exact_mut(size) {
                element.copy_from_slice(&identity[..size]);
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
        let p = self.size();
        check_counts(send_counts, p, send.len(), dtype)?;
        check_counts(recv_counts, p, recv.len(), dtype)?;
        let (sc, sd) = checked_counts(send_counts)?;
        let (rc, rd) = checked_counts(recv_counts)?;
        with_type!(
            dtype,
            all_to_all_v_typed(&self.0, send, &sc, &sd, recv, &rc, &rd)
        );
        Ok(())
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
        let (ns, nd) = (nb.sources().len(), nb.destinations().len());
        check_counts(send_counts, nd, send.len(), dtype)?;
        check_counts(recv_counts, ns, recv.len(), dtype)?;
        let Some(graph) = nb.graph() else {
            // Only the empty neighbourhood has no graph communicator: nothing to exchange.
            if ns + nd > 0 {
                return Err(GridError::LengthMismatch {
                    expected: 0,
                    found: ns + nd,
                });
            }
            return Ok(());
        };
        let (sc, sd) = checked_counts(send_counts)?;
        let (rc, rd) = checked_counts(recv_counts)?;
        graph.neighbour_all_to_all_v(
            ns,
            nd,
            send,
            &sc,
            &sd,
            recv,
            &rc,
            &rd,
            with_type!(dtype, datatype_of()),
            dtype_size(dtype),
        );
        Ok(())
    }
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        Some(self.0.as_raw())
    }
    fn duplicate(&self) -> Rc<dyn CommDyn> {
        Rc::new(self.clone())
    }
}
