//! The communicator traits (Listing S2c).

use crate::{ReduceOp, Result};

/// Object-safe communicator interface used by IndexMap, Scatterer and partitioners.
pub trait CommDyn: 'static {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;
    fn barrier(&self);
    /// In-place all-reduce of `buf`, read as elements of `dtype`.
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp);
    /// In-place exclusive scan; rank 0 receives the identity of `op`.
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp);
    /// Variable all-to-all; counts in elements.
    fn all_to_all_v_bytes(
        &self,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()>;
    /// Neighbourhood variant on a distributed-graph communicator.
    fn neighbour_all_to_all_v_bytes(
        &self,
        nb: &Neighbourhood,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()>;
    /// Raw handle for PT-Scotch and ParMETIS; None for Serial.
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm>;
}

/// The communicator stored in a Mesh. Not required to be Send or Sync.
pub trait Comm: CommDyn + Clone {}
impl<T: CommDyn + Clone> Comm for T {}

/// Typed helpers on any communicator, including `&dyn CommDyn`.
pub trait CommExt {
    /// The sum of `x` over all ranks.
    fn sum<T: CommScalar>(&self, x: T) -> T;
    /// The maximum of `x` over all ranks.
    fn max<T: CommScalar>(&self, x: T) -> T;
    /// The sum of `x` over the lower ranks; 0 on rank 0.
    fn exscan_sum<T: CommScalar>(&self, x: T) -> T;
}

impl<C: CommDyn + ?Sized> CommExt for C {
    fn sum<T: CommScalar>(&self, _x: T) -> T {
        todo!()
    }
    fn max<T: CommScalar>(&self, _x: T) -> T {
        todo!()
    }
    fn exscan_sum<T: CommScalar>(&self, _x: T) -> T {
        todo!()
    }
}

/// Element type of a byte-level collective.
#[derive(Clone, Copy, Debug)]
pub enum DType {
    I32,
    I64,
    U32,
    U64,
    F32,
    F64,
}

/// Scalars of the typed collectives: i32, i64, u32, u64, f32, f64.
pub trait CommScalar: bytemuck::Pod {
    const DTYPE: DType;
}

impl CommScalar for i32 {
    const DTYPE: DType = DType::I32;
}
impl CommScalar for i64 {
    const DTYPE: DType = DType::I64;
}
impl CommScalar for u32 {
    const DTYPE: DType = DType::U32;
}
impl CommScalar for u64 {
    const DTYPE: DType = DType::U64;
}
impl CommScalar for f32 {
    const DTYPE: DType = DType::F32;
}
impl CommScalar for f64 {
    const DTYPE: DType = DType::F64;
}

/// Source and destination ranks of a distributed-graph communicator. Opaque; Serial accepts
/// only the empty one. M6 adds construction over a graph communicator.
pub struct Neighbourhood {
    sources: Vec<usize>,
    destinations: Vec<usize>,
}

impl Neighbourhood {
    /// No sources and no destinations.
    pub fn empty() -> Self {
        todo!()
    }
    /// The ranks this rank receives from.
    pub fn sources(&self) -> &[usize] {
        &self.sources
    }
    /// The ranks this rank sends to.
    pub fn destinations(&self) -> &[usize] {
        &self.destinations
    }
}
