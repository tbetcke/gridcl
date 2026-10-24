//! Benchmark hooks of the private mesh families (ADR-0015, item 4). Only with the Cargo feature
//! `bench`, which gridcl-bench turns on; not part of the library API and without any stability
//! promise.
//!
//! Each hook runs exactly one family's launch sequence on inputs already on the device, the
//! cell-fastest connectivity copy included, and returns the outputs in an opaque owner, so the
//! caller can sync, time and drop them.

use gridcl_device::{DeviceBuffer, Result};
use gridcl_dims::TopoDim;

use crate::Topology;

/// The output buffers of one hook: held until dropped, so the timed launches stay alive until
/// the caller has synchronised.
pub struct BenchOutput {
    #[allow(dead_code)] // held only for its lifetime
    buffers: Vec<DeviceBuffer<u32>>,
}

impl BenchOutput {
    /// Words held, padding excluded.
    pub fn len(&self) -> usize {
        self.buffers.iter().map(DeviceBuffer::len).sum()
    }

    /// Whether no words are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The key emission of `create_entities(dim)` ("Entity creation", step 1) for every entity type
/// of dimension `dim`.
///
/// # Errors
///
/// `EntitiesNotCreated(dim)` unless 1 ≤ dim < tdim; allocation and device errors.
pub fn emit_keys<TD: TopoDim>(topology: &Topology<TD>, dim: usize) -> Result<BenchOutput> {
    let _ = (topology, dim);
    todo!("M3s T10: bench::emit_keys")
}

/// The PermutationInfo launch of cell block `b`.
///
/// # Errors
///
/// `IndexOutOfRange` if there is no block `b`; allocation and device errors.
pub fn permutation_info<TD: TopoDim>(topology: &Topology<TD>, b: usize) -> Result<BenchOutput> {
    let _ = (topology, b);
    todo!("M3s T10: bench::permutation_info")
}
