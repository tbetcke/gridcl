// Design: "Dependency rules" 5 (no MPI dependency without the feature `mpi`); milestones M2
// (the communication core) and M6 (`Mpi`, `Scatterer`).
//! Communicators, index maps and halo exchange: the parallel layer of tetraloma.
//!
//! - [`CommDyn`], [`Comm`] and [`CommExt`] are the interface of a communicator. [`Serial`] is
//!   the communicator of one process; with the feature `mpi`, `Mpi` wraps an MPI communicator.
//! - [`IndexMap`] says which rank owns each index of a distributed set of entities, and which
//!   indices each rank holds as ghosts.
//! - [`Scatterer`] runs the halo exchange of an index map on device arrays: forward from the
//!   owners to the ghosts, and reverse from the ghosts to their owners.
//! - [`Neighbourhood`] lists the ranks a rank exchanges data with.
//!
//! Without the feature `mpi` the crate does not depend on MPI. The facade crate `tetraloma`
//! re-exports all of it. The tutorials [Distributed
//! meshes](https://tetracl.github.io/tetraloma/tutorials/distributed.html) and [Halo
//! exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html) show how to use
//! it.

mod comm;
mod counts;
#[cfg(feature = "mpi")]
mod graph;
mod hash;
mod index_map;
mod local_gpu;
#[cfg(feature = "mpi")]
mod mpi;
mod scatterer;
mod serial;

pub use comm::{Comm, CommDyn, CommExt, CommScalar, DType, Neighbourhood};
pub use hash::owner_hash;
pub use index_map::IndexMap;
pub use local_gpu::local_gpu_index;
#[cfg(feature = "mpi")]
pub use mpi::Mpi;
pub use scatterer::{ReverseOp, ScatterNumeric, Scatterer, Transport};
pub use serial::Serial;
pub use tetraloma_device::{GridError, ReduceOp, Result};
