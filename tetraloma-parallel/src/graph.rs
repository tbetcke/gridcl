//! Safe wrappers over the MPI functions rsmpi does not offer: `MPI_Dist_graph_create_adjacent`
//! and `MPI_Neighbor_alltoallv` (docs/design.md, "MPI bindings"). Feature `mpi`.

use std::os::raw::c_int;

use mpi::ffi::{self, MPI_Comm, MPI_Datatype};

use crate::{GridError, Result};

/// A distributed-graph communicator, freed on drop (unless MPI is already finalised).
pub(crate) struct GraphComm(MPI_Comm);

/// `ranks` as MPI ranks.
fn mpi_ranks(ranks: &[usize]) -> Result<Vec<c_int>> {
    ranks
        .iter()
        .map(|&r| c_int::try_from(r).map_err(|_| GridError::TooLarge))
        .collect()
}

impl GraphComm {
    /// The distributed-graph communicator over `parent` in which this rank receives from
    /// `sources` and sends to `destinations`, in that order: unweighted, `reorder = 0`, so
    /// ranks keep their numbers. Collective over `parent`.
    pub(crate) fn create(
        parent: MPI_Comm,
        sources: &[usize],
        destinations: &[usize],
    ) -> Result<Self> {
        let sources = mpi_ranks(sources)?;
        let destinations = mpi_ranks(destinations)?;
        let indegree = c_int::try_from(sources.len()).map_err(|_| GridError::TooLarge)?;
        let outdegree = c_int::try_from(destinations.len()).map_err(|_| GridError::TooLarge)?;
        // SAFETY: reading an immutable handle constant that mpi-sys defines.
        let mut graph = unsafe { ffi::RSMPI_COMM_NULL };
        // SAFETY: `parent` is a live intra-communicator (from `Mpi`), every rank of it calls
        // this function (the caller is collective), `sources` and `destinations` hold
        // `indegree` and `outdegree` ranks and outlive the call, the weights are
        // MPI_UNWEIGHTED, and `graph` is a valid place for the new handle.
        let rc = unsafe {
            ffi::MPI_Dist_graph_create_adjacent(
                parent,
                indegree,
                sources.as_ptr(),
                ffi::RSMPI_UNWEIGHTED(),
                outdegree,
                destinations.as_ptr(),
                ffi::RSMPI_UNWEIGHTED(),
                ffi::RSMPI_INFO_NULL,
                0,
                &mut graph,
            )
        };
        // The default error handler, MPI_ERRORS_ARE_FATAL, aborts on any error.
        debug_assert_eq!(rc, 0, "MPI_Dist_graph_create_adjacent");
        Ok(GraphComm(graph))
    }

    /// `MPI_Neighbor_alltoallv` of elements of `datatype`, `elem` bytes each: block k of
    /// `send` (`send_counts[k]` elements at `send_displs[k]`) goes to destination k, and block
    /// k of `recv` comes from source k. Collective over the graph communicator.
    ///
    /// # Panics
    ///
    /// If a block lies outside its buffer, or the count slices do not have one entry per
    /// neighbour (the caller checks both).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn neighbour_all_to_all_v(
        &self,
        num_sources: usize,
        num_destinations: usize,
        send: &[u8],
        send_counts: &[c_int],
        send_displs: &[c_int],
        recv: &mut [u8],
        recv_counts: &[c_int],
        recv_displs: &[c_int],
        datatype: MPI_Datatype,
        elem: usize,
    ) {
        let within = |len: usize, counts: &[c_int], displs: &[c_int], n: usize| {
            counts.len() == n
                && displs.len() == n
                && counts
                    .iter()
                    .zip(displs)
                    .all(|(&c, &d)| (d as usize + c as usize) * elem <= len)
        };
        assert!(within(
            send.len(),
            send_counts,
            send_displs,
            num_destinations
        ));
        assert!(within(recv.len(), recv_counts, recv_displs, num_sources));
        // Empty slices may share one dangling address, which MPI reads as an aliased send and
        // receive buffer (Open MPI: MPI_ERR_ARG). Empty buffers point at distinct dummies.
        let (dummy_send, mut dummy_recv) = ([0_u8; 8], [0_u8; 8]);
        let send_ptr = if send.is_empty() {
            dummy_send.as_ptr()
        } else {
            send.as_ptr()
        };
        let recv_ptr = if recv.is_empty() {
            dummy_recv.as_mut_ptr()
        } else {
            recv.as_mut_ptr()
        };
        // SAFETY: the graph communicator is live (owned by `self`); the count and
        // displacement slices have one entry per destination and per source of the graph
        // (`num_destinations` and `num_sources` are the degrees it was created with, checked
        // by the caller); every block lies inside its buffer (asserted above), and an empty
        // buffer is a live dummy that no block reaches; `send_ptr` is only read and
        // `recv_ptr` only written, and they never alias.
        let rc = unsafe {
            ffi::MPI_Neighbor_alltoallv(
                send_ptr.cast(),
                send_counts.as_ptr(),
                send_displs.as_ptr(),
                datatype,
                recv_ptr.cast(),
                recv_counts.as_ptr(),
                recv_displs.as_ptr(),
                datatype,
                self.0,
            )
        };
        debug_assert_eq!(rc, 0, "MPI_Neighbor_alltoallv");
    }
}

impl Drop for GraphComm {
    fn drop(&mut self) {
        if mpi::environment::is_finalized() {
            return;
        }
        // SAFETY: `self.0` is the live communicator created in `create`, freed only here.
        unsafe {
            ffi::MPI_Comm_free(&mut self.0);
        }
    }
}
