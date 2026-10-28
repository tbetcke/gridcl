//! `Scatterer`: the host-staged halo exchange with a deterministic reverse (docs/design.md,
//! "Transports"; ADR-0009, ADR-0026).

use gridcl_device::{CubeElement, DeviceBuffer, DeviceScalar};

use crate::Result;

/// Forward (owner → ghosts) and reverse (ghosts → owner) exchange of one `IndexMap`, built by
/// [`IndexMap::scatterer`](crate::IndexMap::scatterer).
pub struct Scatterer {
    transport: Transport,
}

impl Scatterer {
    /// Owner → ghosts (insert). T moves as u32 words; `size_of::<T>()` must be a multiple of 4.
    pub fn forward<T: CubeElement>(&self, x: &mut DeviceBuffer<T>) -> Result<()> {
        let _ = x;
        todo!("M6 T7: Scatterer::forward")
    }
    /// Ghosts → owner by a segmented reduce: deterministic and free of atomics.
    pub fn reverse<T: ScatterNumeric>(&self, x: &mut DeviceBuffer<T>, op: ReverseOp) -> Result<()> {
        let _ = (x, op);
        todo!("M6 T7: Scatterer::reverse")
    }
    /// The transport this scatterer exchanges through.
    pub fn transport(&self) -> Transport {
        self.transport
    }
}

/// How [`Scatterer::reverse`] combines the ghost contributions with the owner's value.
///
/// `Add`, `Min` and `Max` combine the owner's value with every ghost contribution in a fixed
/// order; `Insert` replaces it by the lowest neighbour rank's contribution (unchanged without
/// any). Ghost values are unchanged by reverse; empty segments use the ADR-0011 identities
/// (ADR-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReverseOp {
    /// The sum.
    Add,
    /// The minimum.
    Min,
    /// The maximum.
    Max,
    /// The contribution of the lowest neighbour rank.
    Insert,
}

/// The scalars of [`Scatterer::reverse`]: f32, f64, i32 and u32.
pub trait ScatterNumeric: DeviceScalar {}

impl ScatterNumeric for f32 {}
impl ScatterNumeric for f64 {}
impl ScatterNumeric for i32 {}
impl ScatterNumeric for u32 {}

/// How a [`Scatterer`] moves data between ranks. M6 always chooses `HostStaged`; M10 adds
/// `GpuAware` (ADR-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Pack on the device, exchange on the host, unpack on the device.
    HostStaged,
    /// Device pointers handed to a CUDA-aware MPI (M10).
    GpuAware,
}
