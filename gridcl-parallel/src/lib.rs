//! `CommDyn`, `Comm`, `CommExt`, `Serial`, `Mpi` (feature `mpi`), `IndexMap`, `Scatterer` and
//! the neighbourhood exchange.
//!
//! Without the `mpi` feature the crate does not depend on MPI (docs/design.md, "Dependency
//! rules" 5). Milestone M2 builds the communication core; M6 adds `Mpi` and `Scatterer`.

mod comm;
mod counts;
#[cfg(feature = "mpi")]
mod graph;
mod hash;
mod index_map;
#[cfg(feature = "mpi")]
mod mpi;
mod scatterer;
mod serial;

pub use comm::{Comm, CommDyn, CommExt, CommScalar, DType, Neighbourhood};
pub use gridcl_device::{GridError, ReduceOp, Result};
pub use hash::owner_hash;
pub use index_map::IndexMap;
#[cfg(feature = "mpi")]
pub use mpi::Mpi;
pub use scatterer::{ReverseOp, ScatterNumeric, Scatterer, Transport};
pub use serial::Serial;
