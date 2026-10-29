//! The communicator traits (Listing S2c).

use std::rc::Rc;

use crate::{GridError, ReduceOp, Result};

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
    /// Cheap clone that refers to the same communicator (for `Mpi` an `Rc` clone, not
    /// `MPI_Comm_dup`), so a [`Neighbourhood`] can hold it (ADR-0026).
    fn duplicate(&self) -> Rc<dyn CommDyn>;
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
    /// The minimum of `x` over all ranks.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, Serial};
    ///
    /// assert_eq!(Serial.min(2.5_f64), 2.5);
    /// ```
    fn min<T: CommScalar>(&self, x: T) -> T;
    /// `x` of every rank, in rank order.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, Serial};
    ///
    /// assert_eq!(Serial.all_gather(7_i32), vec![7]);
    /// ```
    fn all_gather<T: CommScalar>(&self, x: T) -> Vec<T>;
    /// Variable all-to-all: the first `send_counts[0]` values of `send` go to rank 0, the next
    /// `send_counts[1]` to rank 1, and so on. Exchanges the counts first; returns the received
    /// values, grouped by source rank, and the count from each rank.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` unless there is one count per rank and the counts add up to
    /// `send.len()`; `TooLarge` if a sum overflows; the communicator's errors (for `Mpi`,
    /// `MessageTooLarge`).
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, GridError, Serial};
    ///
    /// let (values, counts) = Serial.all_to_all_v(&[1_u64, 2], &[2])?;
    /// assert_eq!((values, counts), (vec![1, 2], vec![2]));
    /// # Ok::<(), GridError>(())
    /// ```
    fn all_to_all_v<T: CommScalar>(
        &self,
        send: &[T],
        send_counts: &[usize],
    ) -> Result<(Vec<T>, Vec<usize>)>;
    /// Neighbourhood all-to-all: `send_counts[i]` consecutive values of `send` go to
    /// `nb.destinations()[i]`. Exchanges the counts first; returns the received values, grouped
    /// in the order of `nb.sources()`, and the count from each source.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` unless there is one count per destination and the counts add up to
    /// `send.len()`; `TooLarge` if a sum overflows; the communicator's errors.
    ///
    /// ```
    /// use gridcl_parallel::{CommExt, GridError, Neighbourhood, Serial};
    ///
    /// let nb = Neighbourhood::new(&Serial, vec![], vec![])?;
    /// let (values, counts) = Serial.neighbour_all_to_all_v::<f32>(&nb, &[], &[])?;
    /// assert!(values.is_empty() && counts.is_empty());
    /// # Ok::<(), GridError>(())
    /// ```
    fn neighbour_all_to_all_v<T: CommScalar>(
        &self,
        nb: &Neighbourhood,
        send: &[T],
        send_counts: &[usize],
    ) -> Result<(Vec<T>, Vec<usize>)>;
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
    fn min<T: CommScalar>(&self, x: T) -> T {
        let mut v = x;
        self.all_reduce_bytes(bytemuck::bytes_of_mut(&mut v), T::DTYPE, ReduceOp::Min);
        v
    }
    fn all_gather<T: CommScalar>(&self, x: T) -> Vec<T> {
        let p = self.size();
        let send = vec![x; p];
        let mut recv = vec![x; p];
        let ones = vec![1; p];
        self.all_to_all_v_bytes(
            bytemuck::cast_slice(&send),
            &ones,
            bytemuck::cast_slice_mut(&mut recv),
            &ones,
            T::DTYPE,
        )
        .expect("all_gather: one element per rank always fits");
        recv
    }
    fn all_to_all_v<T: CommScalar>(
        &self,
        send: &[T],
        send_counts: &[usize],
    ) -> Result<(Vec<T>, Vec<usize>)> {
        let p = self.size();
        check_counts(send, send_counts, p)?;
        let ones = vec![1; p];
        let recv_counts = exchange_counts(send_counts, p, |s, r| {
            self.all_to_all_v_bytes(s, &ones, r, &ones, DType::U64)
        })?;
        let mut recv = vec![T::zeroed(); total(&recv_counts)?];
        self.all_to_all_v_bytes(
            bytemuck::cast_slice(send),
            send_counts,
            bytemuck::cast_slice_mut(&mut recv),
            &recv_counts,
            T::DTYPE,
        )?;
        Ok((recv, recv_counts))
    }
    fn neighbour_all_to_all_v<T: CommScalar>(
        &self,
        nb: &Neighbourhood,
        send: &[T],
        send_counts: &[usize],
    ) -> Result<(Vec<T>, Vec<usize>)> {
        check_counts(send, send_counts, nb.destinations().len())?;
        let out_ones = vec![1; nb.destinations().len()];
        let in_ones = vec![1; nb.sources().len()];
        let recv_counts = exchange_counts(send_counts, nb.sources().len(), |s, r| {
            self.neighbour_all_to_all_v_bytes(nb, s, &out_ones, r, &in_ones, DType::U64)
        })?;
        let mut recv = vec![T::zeroed(); total(&recv_counts)?];
        self.neighbour_all_to_all_v_bytes(
            nb,
            bytemuck::cast_slice(send),
            send_counts,
            bytemuck::cast_slice_mut(&mut recv),
            &recv_counts,
            T::DTYPE,
        )?;
        Ok((recv, recv_counts))
    }
}

/// `LengthMismatch` unless there are `n` counts and they add up to `send.len()`; `TooLarge`
/// if the sum overflows.
fn check_counts<T>(send: &[T], send_counts: &[usize], n: usize) -> Result<()> {
    if send_counts.len() != n {
        return Err(GridError::LengthMismatch {
            expected: n,
            found: send_counts.len(),
        });
    }
    let sum = total(send_counts)?;
    if sum != send.len() {
        return Err(GridError::LengthMismatch {
            expected: sum,
            found: send.len(),
        });
    }
    Ok(())
}

/// The sum of `counts`; `TooLarge` on overflow.
fn total(counts: &[usize]) -> Result<usize> {
    counts
        .iter()
        .try_fold(0_usize, |a, &c| a.checked_add(c))
        .ok_or(GridError::TooLarge)
}

/// Sends one u64 count per peer through `exchange` (one element to and from each peer) and
/// returns the `num_sources` counts received.
fn exchange_counts(
    send_counts: &[usize],
    num_sources: usize,
    exchange: impl FnOnce(&[u8], &mut [u8]) -> Result<()>,
) -> Result<Vec<usize>> {
    let send: Vec<u64> = send_counts.iter().map(|&c| c as u64).collect();
    let mut recv = vec![0_u64; num_sources];
    exchange(
        bytemuck::cast_slice(&send),
        bytemuck::cast_slice_mut(&mut recv),
    )?;
    recv.into_iter()
        .map(|c| usize::try_from(c).map_err(|_| GridError::TooLarge))
        .collect()
}

/// The agreed result of a collective step (ADR-0026, item 15): every rank learns the lowest
/// failing rank. That rank, and every other failing rank, returns its own error; the others
/// return `RemoteFailure { rank }`. Collective.
pub(crate) fn agree<T>(comm: &dyn CommDyn, local: Result<T>) -> Result<T> {
    let mine = if local.is_err() {
        comm.rank() as u64
    } else {
        u64::MAX
    };
    let mut lowest = mine;
    comm.all_reduce_bytes(
        bytemuck::bytes_of_mut(&mut lowest),
        DType::U64,
        ReduceOp::Min,
    );
    match local {
        Err(e) => Err(e),
        Ok(_) if lowest != u64::MAX => Err(GridError::RemoteFailure {
            rank: lowest as usize,
        }),
        Ok(v) => Ok(v),
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

/// Source and destination ranks of a distributed-graph communicator, with the communicator it
/// was made on (ADR-0026). Opaque; Serial accepts only the empty one.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{Neighbourhood, Serial};
///
/// let nb = Neighbourhood::empty();
/// assert!(nb.sources().is_empty());
/// assert!(nb.destinations().is_empty());
/// // Serial has no other rank to name.
/// assert!(Neighbourhood::new(&Serial, vec![], vec![]).is_ok());
/// assert!(Neighbourhood::new(&Serial, vec![0], vec![]).is_err());
/// ```
pub struct Neighbourhood {
    sources: Vec<usize>,
    destinations: Vec<usize>,
    /// The parent communicator; `None` for [`Neighbourhood::empty`].
    comm: Option<Rc<dyn CommDyn>>,
    /// The distributed-graph communicator, when the parent has an MPI handle.
    #[cfg(feature = "mpi")]
    graph: Option<Rc<crate::graph::GraphComm>>,
}

impl Neighbourhood {
    /// No sources and no destinations.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::Neighbourhood;
    ///
    /// let nb = Neighbourhood::empty();
    /// assert!(nb.sources().is_empty() && nb.destinations().is_empty());
    /// ```
    pub fn empty() -> Self {
        Neighbourhood {
            sources: Vec::new(),
            destinations: Vec::new(),
            comm: None,
            #[cfg(feature = "mpi")]
            graph: None,
        }
    }
    /// The neighbourhood of `comm` in which this rank receives from `sources` and sends to
    /// `destinations`. Collective; every rank's lists must be the transpose of the others'.
    /// With `comm.raw()` = `Some` it creates the graph communicator by
    /// `MPI_Dist_graph_create_adjacent` (unweighted, `reorder = 0`); otherwise it keeps the rank
    /// lists. It holds `comm.duplicate()` for the exchanges (ADR-0026).
    ///
    /// # Errors
    ///
    /// Agreed over the ranks (ADR-0026, item 15): `IndexOutOfRange` if a rank is not below
    /// `comm.size()` or is this rank (so `Serial` accepts only empty lists), `LengthMismatch`
    /// if a list repeats a rank (expected: the distinct count); `RemoteFailure` on the other
    /// ranks.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{GridError, Neighbourhood, Serial};
    ///
    /// let nb = Neighbourhood::new(&Serial, vec![], vec![])?;
    /// assert!(nb.sources().is_empty());
    /// let err = Neighbourhood::new(&Serial, vec![], vec![0]);
    /// assert!(matches!(err, Err(GridError::IndexOutOfRange { index: 0, len: 1 })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn new(comm: &dyn CommDyn, sources: Vec<usize>, destinations: Vec<usize>) -> Result<Self> {
        let check = |ranks: &[usize]| -> Result<()> {
            if let Some(&r) = ranks
                .iter()
                .find(|&&r| r >= comm.size() || r == comm.rank())
            {
                return Err(GridError::IndexOutOfRange {
                    index: r,
                    len: comm.size(),
                });
            }
            let mut sorted = ranks.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            if sorted.len() != ranks.len() {
                return Err(GridError::LengthMismatch {
                    expected: sorted.len(),
                    found: ranks.len(),
                });
            }
            Ok(())
        };
        let local = check(&sources).and_then(|()| check(&destinations));
        agree(comm, local)?;
        #[cfg(feature = "mpi")]
        let graph = match comm.raw() {
            Some(raw) => Some(Rc::new(crate::graph::GraphComm::create(
                raw,
                &sources,
                &destinations,
            )?)),
            None => None,
        };
        Ok(Neighbourhood {
            sources,
            destinations,
            comm: Some(comm.duplicate()),
            #[cfg(feature = "mpi")]
            graph,
        })
    }
    /// The distributed-graph communicator; `None` unless made on an MPI communicator.
    #[cfg(feature = "mpi")]
    pub(crate) fn graph(&self) -> Option<&crate::graph::GraphComm> {
        self.graph.as_deref()
    }
    /// The communicator the neighbourhood was made on; `None` for [`Neighbourhood::empty`].
    pub(crate) fn comm(&self) -> Option<&Rc<dyn CommDyn>> {
        self.comm.as_ref()
    }
    /// A neighbourhood with the given ranks, for tests of the empty-only rule.
    #[cfg(test)]
    pub(crate) fn from_ranks(sources: Vec<usize>, destinations: Vec<usize>) -> Self {
        Neighbourhood {
            sources,
            destinations,
            comm: None,
            #[cfg(feature = "mpi")]
            graph: None,
        }
    }
    /// The ranks this rank receives from.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::Neighbourhood;
    ///
    /// let nb = Neighbourhood::empty();
    /// assert!(nb.sources().is_empty());
    /// ```
    pub fn sources(&self) -> &[usize] {
        &self.sources
    }
    /// The ranks this rank sends to.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::Neighbourhood;
    ///
    /// let nb = Neighbourhood::empty();
    /// assert!(nb.destinations().is_empty());
    /// ```
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
