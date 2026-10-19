//! `KernelKind`, `DeviceClass`, `VectorSizeOverride` and `VectorPolicy` (Listing S2).

use cubecl::prelude::*;

/// Kernel families with an entry in the vector policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    Fill,
    Copy,
    Reduce,
    ScanTile,
    SortDigits,
    Gather,
    Pack,
    TopologyKeys,
    PermutationInfo,
    EVector,
    GeomFactors,
    PullBack,
}

/// Device classes the policy distinguishes, derived from `Client::properties()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    CpuSimd128,
    CpuSimd256,
    CpuSimd512,
    Gpu,
}

impl DeviceClass {
    /// A device with `hardware.num_cpu_cores` is a CPU, classed by `hardware.load_width` (512
    /// bits: CpuSimd512, 256: CpuSimd256, else CpuSimd128); every other device is Gpu.
    /// `load_width` alone does not tell a CPU from wgpu, which also reports 128 bits.
    pub fn for_client(_client: &Client) -> DeviceClass {
        todo!()
    }
}

/// Value of `GRIDCL_VECTOR_SIZE` (`scalar | policy | widest | <n>`) and of
/// `testing::with_vector_size`. `Widest` is the first of `io_optimized_vector_sizes(elem_bytes)`.
/// `Fixed(n)` and `Widest` skip the register budget but stay clamped by `max_vector`; `Policy`
/// applies the full rule of [`LaunchPlan::elementwise`](crate::LaunchPlan::elementwise).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSizeOverride {
    Scalar,
    Policy,
    Widest,
    Fixed(usize),
}

/// Vector sizes and tile widths chosen from benchmarks (docs/perf/vectorization.md), compiled in
/// from gridcl-device/vector-policy.toml. `GRIDCL_VECTOR_SIZE` and `GRIDCL_QUAD_TILE` override
/// it for tests and benchmarks.
pub struct VectorPolicy {
    _private: (),
}

impl VectorPolicy {
    /// The policy of the client's device class.
    pub fn for_client(_client: &Client) -> &'static VectorPolicy {
        todo!()
    }

    /// Preferred vector size for one kernel family and scalar size; 1 until benchmarked.
    pub fn vector_size(&self, _kind: KernelKind, _elem_bytes: usize) -> usize {
        todo!()
    }

    /// Tile width of QuadField and E-vector chunks: 1, 2, 4, 8 or 16.
    pub fn quad_tile(&self) -> usize {
        todo!()
    }
}
