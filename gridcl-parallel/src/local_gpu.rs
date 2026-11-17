//! `local_gpu_index` (ADR-0026 item 17, ADR-0060 item 1).

use crate::CommDyn;

/// Index of this rank's GPU on its node: the rank within the node. Collective.
///
/// With an MPI communicator, the rank in `MPI_Comm_split_type(…, MPI_COMM_TYPE_SHARED, …)`;
/// otherwise [`CommDyn::rank`]: 0 for [`Serial`](crate::Serial), and the ranks of an in-process
/// communicator share one node. It does not know the GPU count: a program that opens
/// `Device::cuda(local_gpu_index(&comm))` falls back to another device if there is no such GPU
/// (Listing B8).
pub fn local_gpu_index(_comm: &dyn CommDyn) -> usize {
    todo!("M10 T4")
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
