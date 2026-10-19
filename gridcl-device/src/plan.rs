//! `LaunchPlan` (Listing S2, ADR-0008).

use cubecl::prelude::*;

use crate::{KernelKind, Result};

/// Shape of one launch on the current device. Not `Copy`: `CubeCount` is not `Copy` in
/// CubeCL 0.11 (ADR-0008).
#[derive(Clone, Debug)]
pub struct LaunchPlan {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    /// Lanes per vector; 1 is a scalar launch.
    pub vector_size: usize,
    /// Consecutive vectors each unit processes: 1 on GPUs, a block on CPUs.
    pub items_per_unit: u32,
    /// Vectors covered, padding included.
    pub num_vectors: u32,
}

impl LaunchPlan {
    /// `items` scalar items; `max_vector` the widest size the data layout allows (a divisor of
    /// `VECTOR_PAD`); `live` the vectors the kernel keeps live, for the register budget.
    pub fn elementwise(
        _client: &Client,
        _kind: KernelKind,
        _items: usize,
        _elem_bytes: usize,
        _max_vector: usize,
        _live: usize,
    ) -> Result<Self> {
        todo!()
    }
}
