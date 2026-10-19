//! `CommDyn`, `Comm`, `CommExt`, `Serial`, `Mpi` (feature `mpi`), `IndexMap`, `Scatterer` and
//! the neighbourhood exchange.
//!
//! Without the `mpi` feature the crate does not depend on MPI (docs/design.md, "Dependency
//! rules" 5). Milestone M2 builds the communication core; M6 adds `Mpi` and `Scatterer`.

mod comm;
mod index_map;
mod serial;

pub use comm::{Comm, CommDyn, CommExt, CommScalar, DType, Neighbourhood};
pub use gridcl_device::{GridError, ReduceOp, Result};
pub use index_map::IndexMap;
pub use serial::Serial;
