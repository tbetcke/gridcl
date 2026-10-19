//! The communicator traits (Listing S2c).

use crate::{ReduceOp, Result};

/// Object-safe communicator interface used by IndexMap, Scatterer and partitioners.
///
/// Byte buffers hold elements of a [`DType`] in native byte order; counts are in elements.
/// The identity of a [`ReduceOp`], which [`CommDyn::exscan_bytes`] writes on rank 0, follows
/// gridcl-device: 0 for `Sum`, the largest finite value of the type for `Min` (`f32::MAX`, not
/// infinity) and the smallest finite value for `Max` (`f32::MIN`, `i32::MIN`, 0 for unsigned).
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommDyn, DType, ReduceOp, Serial};
///
/// let comm: &dyn CommDyn = &Serial;
/// assert_eq!((comm.rank(), comm.size()), (0, 1));
///
/// let mut buf = 7_i32.to_ne_bytes();
/// comm.all_reduce_bytes(&mut buf, DType::I32, ReduceOp::Sum);
/// assert_eq!(i32::from_ne_bytes(buf), 7);
///
/// comm.exscan_bytes(&mut buf, DType::I32, ReduceOp::Min);
/// assert_eq!(i32::from_ne_bytes(buf), i32::MAX);
/// ```
pub trait CommDyn: 'static {
    /// This rank, in `0..size()`.
    fn rank(&self) -> usize;
    /// The number of ranks.
    fn size(&self) -> usize;
    /// Blocks until every rank has entered the barrier.
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
///
/// Every `CommDyn` that is `Clone` is a `Comm`.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{Comm, Serial};
///
/// fn ranks<C: Comm>(comm: &C) -> usize {
///     comm.clone().size()
/// }
/// assert_eq!(ranks(&Serial), 1);
/// ```
pub trait Comm: CommDyn + Clone {}
impl<T: CommDyn + Clone> Comm for T {}

/// Typed helpers on any communicator, including `&dyn CommDyn`.
///
/// They go through [`CommDyn::all_reduce_bytes`] and [`CommDyn::exscan_bytes`] on a
/// one-element buffer, so every rank must call them in the same order.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommDyn, CommExt, Serial};
///
/// let comm: &dyn CommDyn = &Serial;
/// assert_eq!(comm.sum(3_u64), 3);
/// assert_eq!(comm.max(-2.5_f64), -2.5);
/// assert_eq!(comm.exscan_sum(5_i32), 0);
/// ```
pub trait CommExt {
    /// The sum of `x` over all ranks.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, Serial};
    ///
    /// assert_eq!(Serial.sum(1.5_f32), 1.5);
    /// ```
    fn sum<T: CommScalar>(&self, x: T) -> T;
    /// The maximum of `x` over all ranks.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, Serial};
    ///
    /// assert_eq!(Serial.max(-4_i64), -4);
    /// ```
    fn max<T: CommScalar>(&self, x: T) -> T;
    /// The sum of `x` over the lower ranks; 0 on rank 0.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, Serial};
    ///
    /// assert_eq!(Serial.exscan_sum(9_u32), 0);
    /// ```
    fn exscan_sum<T: CommScalar>(&self, x: T) -> T;
}

impl<C: CommDyn + ?Sized> CommExt for C {
    fn sum<T: CommScalar>(&self, x: T) -> T {
        let mut v = x;
        self.all_reduce_bytes(bytemuck::bytes_of_mut(&mut v), T::DTYPE, ReduceOp::Sum);
        v
    }
    fn max<T: CommScalar>(&self, x: T) -> T {
        let mut v = x;
        self.all_reduce_bytes(bytemuck::bytes_of_mut(&mut v), T::DTYPE, ReduceOp::Max);
        v
    }
    fn exscan_sum<T: CommScalar>(&self, x: T) -> T {
        let mut v = x;
        self.exscan_bytes(bytemuck::bytes_of_mut(&mut v), T::DTYPE, ReduceOp::Sum);
        v
    }
}

/// Element type of a byte-level collective.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommScalar, DType};
///
/// assert!(matches!(<f64 as CommScalar>::DTYPE, DType::F64));
/// ```
#[derive(Clone, Copy, Debug)]
pub enum DType {
    /// `i32`.
    I32,
    /// `i64`.
    I64,
    /// `u32`.
    U32,
    /// `u64`.
    U64,
    /// `f32`.
    F32,
    /// `f64`.
    F64,
}

/// The size of one element of `dtype` in bytes.
pub(crate) fn dtype_size(dtype: DType) -> usize {
    match dtype {
        DType::I32 | DType::U32 | DType::F32 => 4,
        DType::I64 | DType::U64 | DType::F64 => 8,
    }
}

/// Picks the identity of `op` among the identities of `Sum`, `Min` and `Max`.
fn identity_of<T: bytemuck::Pod>(zero: T, min_identity: T, max_identity: T, op: ReduceOp) -> T {
    match op {
        ReduceOp::Sum => zero,
        ReduceOp::Min => min_identity,
        ReduceOp::Max => max_identity,
    }
}

/// The identity of `op` for `dtype` as native-order bytes in the first
/// `dtype_size(dtype)` entries of the returned array: 0 for `Sum`, the largest finite value
/// for `Min` and the smallest finite value for `Max` (the convention of gridcl-device).
pub(crate) fn identity_bytes(dtype: DType, op: ReduceOp) -> [u8; 8] {
    let mut out = [0_u8; 8];
    let mut put = |bytes: &[u8]| out[..bytes.len()].copy_from_slice(bytes);
    match dtype {
        DType::I32 => put(bytemuck::bytes_of(&identity_of(
            0_i32,
            i32::MAX,
            i32::MIN,
            op,
        ))),
        DType::I64 => put(bytemuck::bytes_of(&identity_of(
            0_i64,
            i64::MAX,
            i64::MIN,
            op,
        ))),
        DType::U32 => put(bytemuck::bytes_of(&identity_of(
            0_u32,
            u32::MAX,
            u32::MIN,
            op,
        ))),
        DType::U64 => put(bytemuck::bytes_of(&identity_of(
            0_u64,
            u64::MAX,
            u64::MIN,
            op,
        ))),
        DType::F32 => put(bytemuck::bytes_of(&identity_of(
            0_f32,
            f32::MAX,
            f32::MIN,
            op,
        ))),
        DType::F64 => put(bytemuck::bytes_of(&identity_of(
            0_f64,
            f64::MAX,
            f64::MIN,
            op,
        ))),
    }
    out
}

/// Scalars of the typed collectives: i32, i64, u32, u64, f32, f64.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommExt, CommScalar, Serial};
///
/// fn total<T: CommScalar>(x: T) -> T {
///     Serial.sum(x)
/// }
/// assert_eq!(total(2_u32), 2);
/// ```
pub trait CommScalar: bytemuck::Pod {
    /// The byte-level element type of `Self`.
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
///
/// # Example
///
/// ```
/// use gridcl_parallel::Neighbourhood;
///
/// let nb = Neighbourhood::empty();
/// assert!(nb.sources().is_empty());
/// assert!(nb.destinations().is_empty());
/// ```
pub struct Neighbourhood {
    sources: Vec<usize>,
    destinations: Vec<usize>,
}

impl Neighbourhood {
    /// No sources and no destinations.
    pub fn empty() -> Self {
        Neighbourhood {
            sources: Vec::new(),
            destinations: Vec::new(),
        }
    }
    /// A neighbourhood with the given ranks, for tests of the empty-only rule.
    #[cfg(test)]
    pub(crate) fn from_ranks(sources: Vec<usize>, destinations: Vec<usize>) -> Self {
        Neighbourhood {
            sources,
            destinations,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_sizes_match_the_rust_types() {
        assert_eq!(dtype_size(i32::DTYPE), size_of::<i32>());
        assert_eq!(dtype_size(i64::DTYPE), size_of::<i64>());
        assert_eq!(dtype_size(u32::DTYPE), size_of::<u32>());
        assert_eq!(dtype_size(u64::DTYPE), size_of::<u64>());
        assert_eq!(dtype_size(f32::DTYPE), size_of::<f32>());
        assert_eq!(dtype_size(f64::DTYPE), size_of::<f64>());
    }

    #[test]
    fn identities_follow_the_device_convention() {
        let f32_of =
            |op| f32::from_ne_bytes(identity_bytes(DType::F32, op)[..4].try_into().unwrap());
        assert_eq!(f32_of(ReduceOp::Sum), 0.0);
        assert_eq!(f32_of(ReduceOp::Min), f32::MAX);
        assert_eq!(f32_of(ReduceOp::Max), f32::MIN);
        let i64_of = |op| i64::from_ne_bytes(identity_bytes(DType::I64, op));
        assert_eq!(i64_of(ReduceOp::Sum), 0);
        assert_eq!(i64_of(ReduceOp::Min), i64::MAX);
        assert_eq!(i64_of(ReduceOp::Max), i64::MIN);
        let u32_of =
            |op| u32::from_ne_bytes(identity_bytes(DType::U32, op)[..4].try_into().unwrap());
        assert_eq!(u32_of(ReduceOp::Min), u32::MAX);
        assert_eq!(u32_of(ReduceOp::Max), 0);
    }

    #[test]
    fn empty_neighbourhood_has_no_ranks() {
        let nb = Neighbourhood::empty();
        assert!(nb.sources().is_empty() && nb.destinations().is_empty());
    }
}
