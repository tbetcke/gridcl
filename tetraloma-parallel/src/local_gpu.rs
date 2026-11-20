//! `local_gpu_index` (ADR-0026 item 17, ADR-0060 item 1).

use crate::CommDyn;

/// Index of this rank's GPU on its node: the rank within the node. Collective.
///
/// With an MPI communicator, the rank in `MPI_Comm_split_type(…, MPI_COMM_TYPE_SHARED, …)`;
/// otherwise [`CommDyn::rank`]: 0 for [`Serial`](crate::Serial), and the ranks of an in-process
/// communicator share one node. It does not know the GPU count: a program that opens
/// `Device::cuda(local_gpu_index(&comm))` falls back to another device if there is no such GPU
/// (Listing B8).
///
/// # Example
///
/// ```
/// use tetraloma_parallel::{Serial, local_gpu_index};
///
/// // One rank, alone on its node.
/// assert_eq!(local_gpu_index(&Serial), 0);
/// ```
pub fn local_gpu_index(comm: &dyn CommDyn) -> usize {
    #[cfg(feature = "mpi")]
    if let Some(raw) = comm.raw() {
        return shared_rank(raw, comm.rank());
    }
    comm.rank()
}

/// This rank's rank in the shared-memory communicator of `parent`, the ranks of each node
/// ordered by their rank `rank` in `parent`. Collective over `parent`.
#[cfg(feature = "mpi")]
fn shared_rank(parent: mpi::ffi::MPI_Comm, rank: usize) -> usize {
    use std::os::raw::c_int;

    use mpi::ffi;

    let key = c_int::try_from(rank).expect("an MPI rank fits a C int");
    // SAFETY: reading an immutable handle constant that mpi-sys defines.
    let mut node = unsafe { ffi::RSMPI_COMM_NULL };
    // SAFETY: `parent` is a live intra-communicator (from `CommDyn::raw` of an `Mpi`), every
    // rank of it calls this function (`local_gpu_index` is collective), the split type and
    // info are constants mpi-sys defines, and `node` is a valid place for the new handle.
    let rc = unsafe {
        ffi::MPI_Comm_split_type(
            parent,
            ffi::RSMPI_COMM_TYPE_SHARED,
            key,
            ffi::RSMPI_INFO_NULL,
            &mut node,
        )
    };
    // The default error handler, MPI_ERRORS_ARE_FATAL, aborts on any error.
    debug_assert_eq!(rc, 0, "MPI_Comm_split_type");
    let mut local: c_int = 0;
    // SAFETY: `node` is the live communicator just created (MPI_COMM_TYPE_SHARED never gives
    // MPI_COMM_NULL: every rank passes the same split type), and `local` is a valid place for
    // the rank.
    let rc = unsafe { ffi::MPI_Comm_rank(node, &mut local) };
    debug_assert_eq!(rc, 0, "MPI_Comm_rank");
    // SAFETY: `node` is live, created above and freed only here; every rank of it frees it
    // (collective), and nothing else holds the handle.
    let rc = unsafe { ffi::MPI_Comm_free(&mut node) };
    debug_assert_eq!(rc, 0, "MPI_Comm_free");
    usize::try_from(local).expect("an MPI rank is not negative")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Serial;

    #[test]
    fn local_gpu_index_serial_is_zero() {
        assert_eq!(local_gpu_index(&Serial), 0);
    }
}
