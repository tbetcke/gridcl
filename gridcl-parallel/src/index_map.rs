//! `IndexMap` without ghosts (M2); M6 adds ghosts, `scatterer` and the device field.

use std::rc::Rc;

use gridcl_device::Client;

use crate::comm::agree;
use crate::{CommDyn, CommExt, DType, GridError, Neighbourhood, ReduceOp, Result, Scatterer};

/// Ownership of a distributed index set in the DOLFINx sense: `[owned | ghost]` local indices.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{IndexMap, Serial};
///
/// let map = IndexMap::new(&Serial, 10)?;
/// assert_eq!(map.owned_range(), (0, 10));
/// assert_eq!(map.size_global(), 10);
/// assert_eq!((map.num_owned(), map.num_ghosts(), map.num_local()), (10, 0, 10));
/// assert!(map.ghosts().is_empty() && map.ghost_owners().is_empty());
/// # Ok::<(), gridcl_parallel::GridError>(())
/// ```
pub struct IndexMap {
    owned_range: (i64, i64),
    ghosts: Vec<i64>,
    ghost_owners: Vec<i32>,
    size_global: i64,
    /// The neighbourhoods and host index lists, shared by clones.
    exchange: Rc<Exchange>,
}

/// The communication pattern of an [`IndexMap`] (ADR-0026, item 5).
///
/// The forward direction (owner → ghosts) sends `send_counts[i]` values, the owned local
/// indices `send_idx` in order, to `forward.destinations()[i]`, ordered within each
/// destination by the ghost position on the receiver, and receives `recv_counts[i]` values
/// from `forward.sources()[i]` into the ghost local indices `recv_idx`. The reverse
/// neighbourhood is the transpose.
pub(crate) struct Exchange {
    pub(crate) forward: Neighbourhood,
    #[allow(dead_code)] // The scatterer's reverse direction (M6 T7).
    pub(crate) reverse: Neighbourhood,
    pub(crate) send_idx: Vec<u32>,
    pub(crate) send_counts: Vec<usize>,
    pub(crate) recv_idx: Vec<u32>,
    pub(crate) recv_counts: Vec<usize>,
}

impl Exchange {
    /// No neighbours.
    fn empty() -> Self {
        Exchange {
            forward: Neighbourhood::empty(),
            reverse: Neighbourhood::empty(),
            send_idx: Vec::new(),
            send_counts: Vec::new(),
            recv_idx: Vec::new(),
            recv_counts: Vec::new(),
        }
    }
}

/// Splits `n` into its high and low 32-bit halves, each held in a u64.
fn halves(n: u64) -> [u64; 2] {
    [n >> 32, n & 0xFFFF_FFFF]
}

/// Recombines summed halves; exact while fewer than 2^32 ranks contribute.
fn combine([hi, lo]: [u64; 2]) -> u128 {
    (u128::from(hi) << 32) + u128::from(lo)
}

impl IndexMap {
    /// Without ghosts: owned range from `exscan_sum` of `num_owned`, size from `sum`; an i64
    /// overflow gives `TooLarge`.
    ///
    /// Collective: every rank of `comm` must call it. The sum and the exclusive scan run on
    /// the 32-bit halves of `num_owned` as u64, so they cannot wrap (for fewer than 2^32
    /// ranks), and every rank sees the same global size and so returns `TooLarge` alike
    /// when it exceeds `i64::MAX`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{GridError, IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3)?;
    /// assert_eq!(map.owned_range(), (0, 3));
    ///
    /// assert!(matches!(IndexMap::new(&Serial, usize::MAX), Err(GridError::TooLarge)));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn new(comm: &dyn CommDyn, num_owned: usize) -> Result<IndexMap> {
        let n = u64::try_from(num_owned).map_err(|_| GridError::TooLarge)?;
        let mut total = halves(n);
        let mut before = halves(n);
        comm.all_reduce_bytes(
            bytemuck::bytes_of_mut(&mut total),
            DType::U64,
            ReduceOp::Sum,
        );
        comm.exscan_bytes(
            bytemuck::bytes_of_mut(&mut before),
            DType::U64,
            ReduceOp::Sum,
        );

        // begin + n <= total, so all three fit in i64 once total does.
        let size_global = i64::try_from(combine(total)).map_err(|_| GridError::TooLarge)?;
        let begin = i64::try_from(combine(before)).map_err(|_| GridError::TooLarge)?;
        let end = i64::try_from(n)
            .ok()
            .and_then(|n| begin.checked_add(n))
            .ok_or(GridError::TooLarge)?;
        Ok(IndexMap {
            owned_range: (begin, end),
            ghosts: Vec::new(),
            ghost_owners: Vec::new(),
            size_global,
            exchange: Rc::new(Exchange::empty()),
        })
    }
    /// With ghosts: `ghosts[k]` is the global index of ghost k and `ghost_owners[k]` its owning
    /// rank. Collective. Ghosts request their indices from their owners, which range-check them
    /// (`IndexOutOfRange`, agreed); destinations by `all_to_all_v`; forward and transposed
    /// reverse neighbourhoods; send lists ordered by (neighbour rank, ghost position on the
    /// receiver) (ADR-0026).
    ///
    /// The forward neighbourhood receives from the owners of this rank's ghosts and sends to
    /// the ranks that ghost its owned indices, both in ascending rank order; the reverse
    /// neighbourhood is its transpose.
    ///
    /// # Errors
    ///
    /// Agreed over the ranks (ADR-0026, item 15); ranks without a local error return
    /// `RemoteFailure` with the lowest failing rank.
    /// - `LengthMismatch` if `ghost_owners` and `ghosts` differ in length;
    /// - `IndexOutOfRange { index: owner, len: size }` for an owner that is not another rank;
    /// - `IndexOutOfRange { index: g, len: size_global }` on the owner, for a requested global
    ///   index g it does not own;
    /// - `TooLarge` if the local count does not fit u32, or as [`IndexMap::new`].
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{GridError, IndexMap, Serial};
    ///
    /// // One rank has no other rank to take ghosts from.
    /// let map = IndexMap::with_ghosts(&Serial, 4, vec![], vec![])?;
    /// assert_eq!((map.owned_range(), map.num_ghosts()), ((0, 4), 0));
    /// let err = IndexMap::with_ghosts(&Serial, 4, vec![2], vec![0]);
    /// assert!(matches!(err, Err(GridError::IndexOutOfRange { index: 0, len: 1 })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn with_ghosts(
        comm: &dyn CommDyn,
        num_owned: usize,
        ghosts: Vec<i64>,
        ghost_owners: Vec<i32>,
    ) -> Result<IndexMap> {
        let base = IndexMap::new(comm, num_owned)?;
        let (p, rank) = (comm.size(), comm.rank());

        // Local checks, agreed before any exchange.
        let local = (|| {
            if ghost_owners.len() != ghosts.len() {
                return Err(GridError::LengthMismatch {
                    expected: ghosts.len(),
                    found: ghost_owners.len(),
                });
            }
            if let Some(&o) = ghost_owners
                .iter()
                .find(|&&o| usize::try_from(o).map_or(true, |o| o >= p || o == rank))
            {
                return Err(GridError::IndexOutOfRange {
                    index: usize::try_from(o).unwrap_or(usize::MAX),
                    len: p,
                });
            }
            let local = num_owned
                .checked_add(ghosts.len())
                .ok_or(GridError::TooLarge)?;
            u32::try_from(local).map_err(|_| GridError::TooLarge)?;
            Ok(())
        })();
        agree(comm, local)?;

        // Requests to the owners: ghosts grouped by owner, in ghost order within each owner.
        let mut order: Vec<usize> = (0..ghosts.len()).collect();
        order.sort_by_key(|&k| ghost_owners[k]);
        let mut request_counts = vec![0_usize; p];
        for &o in &ghost_owners {
            request_counts[o as usize] += 1;
        }
        let request: Vec<i64> = order.iter().map(|&k| ghosts[k]).collect();
        let (requested, requested_counts) = comm.all_to_all_v(&request, &request_counts)?;

        // Owners range-check the requests.
        let (begin, end) = base.owned_range;
        let local = match requested.iter().find(|g| !(begin..end).contains(g)) {
            Some(&g) => Err(GridError::IndexOutOfRange {
                index: usize::try_from(g).unwrap_or(usize::MAX),
                len: usize::try_from(base.size_global).unwrap_or(usize::MAX),
            }),
            None => Ok(()),
        };
        agree(comm, local)?;

        // In range and below u32 (checked above), so the casts are exact.
        let send_idx: Vec<u32> = requested.iter().map(|&g| (g - begin) as u32).collect();
        let destinations: Vec<usize> = (0..p).filter(|&r| requested_counts[r] > 0).collect();
        let send_counts: Vec<usize> = destinations.iter().map(|&r| requested_counts[r]).collect();
        let sources: Vec<usize> = (0..p).filter(|&r| request_counts[r] > 0).collect();
        let recv_counts: Vec<usize> = sources.iter().map(|&r| request_counts[r]).collect();
        let recv_idx: Vec<u32> = order.iter().map(|&k| (num_owned + k) as u32).collect();
        let forward = Neighbourhood::new(comm, sources.clone(), destinations.clone())?;
        let reverse = Neighbourhood::new(comm, destinations, sources)?;
        Ok(IndexMap {
            ghosts,
            ghost_owners,
            exchange: Rc::new(Exchange {
                forward,
                reverse,
                send_idx,
                send_counts,
                recv_idx,
                recv_counts,
            }),
            ..base
        })
    }
    /// The scatterer of this map for `block_size` values per index on `client`.
    pub fn scatterer(&self, client: &Client, block_size: usize) -> Result<Scatterer> {
        let _ = (client, block_size);
        todo!("M6 T7: IndexMap::scatterer")
    }
    /// The forward send lists: (destination ranks, count per destination, owned local
    /// indices in send order). For tests of the exchange pattern; not part of the stable API.
    #[doc(hidden)]
    pub fn send_lists(&self) -> (&[usize], &[usize], &[u32]) {
        let e = &self.exchange;
        (e.forward.destinations(), &e.send_counts, &e.send_idx)
    }
    /// The forward receive lists: (source ranks, count per source, ghost local indices in
    /// receive order). For tests of the exchange pattern; not part of the stable API.
    #[doc(hidden)]
    pub fn recv_lists(&self) -> (&[usize], &[usize], &[u32]) {
        let e = &self.exchange;
        (e.forward.sources(), &e.recv_counts, &e.recv_idx)
    }
    /// The communication pattern, for the scatterer.
    #[allow(dead_code)] // M6 T7.
    pub(crate) fn exchange(&self) -> &Exchange {
        &self.exchange
    }
    /// Global indices `[begin, end)` owned by this rank.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// assert_eq!(map.owned_range(), (0, 3));
    /// ```
    pub fn owned_range(&self) -> (i64, i64) {
        self.owned_range
    }
    /// Global index of each ghost.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// // Serial maps have no ghosts.
    /// assert!(map.ghosts().is_empty());
    /// ```
    pub fn ghosts(&self) -> &[i64] {
        &self.ghosts
    }
    /// Owning rank of each ghost.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// assert!(map.ghost_owners().is_empty());
    /// ```
    pub fn ghost_owners(&self) -> &[i32] {
        &self.ghost_owners
    }
    /// Owned indices on this rank.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// assert_eq!(map.num_owned(), 3);
    /// ```
    pub fn num_owned(&self) -> usize {
        // Fits: it was a usize when the map was built.
        (self.owned_range.1 - self.owned_range.0) as usize
    }
    /// Ghost indices on this rank.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// assert_eq!(map.num_ghosts(), 0);
    /// ```
    pub fn num_ghosts(&self) -> usize {
        self.ghosts.len()
    }
    /// Owned plus ghosts.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// assert_eq!(map.num_local(), 3);
    /// ```
    pub fn num_local(&self) -> usize {
        self.num_owned() + self.num_ghosts()
    }
    /// Size of the global index set.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_parallel::{IndexMap, Serial};
    ///
    /// let map = IndexMap::new(&Serial, 3).unwrap();
    /// // Serial: one rank owns everything.
    /// assert_eq!(map.size_global(), 3);
    /// ```
    pub fn size_global(&self) -> i64 {
        self.size_global
    }
}

/// Shares the neighbourhoods; the device cache is not copied (ADR-0026).
///
/// # Example
///
/// ```
/// use gridcl_parallel::{IndexMap, Serial};
///
/// let map = IndexMap::new(&Serial, 5).unwrap();
/// let copy = map.clone();
/// assert_eq!((copy.owned_range(), copy.size_global()), (map.owned_range(), map.size_global()));
/// ```
impl Clone for IndexMap {
    fn clone(&self) -> Self {
        IndexMap {
            owned_range: self.owned_range,
            ghosts: self.ghosts.clone(),
            ghost_owners: self.ghost_owners.clone(),
            size_global: self.size_global,
            exchange: Rc::clone(&self.exchange),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Serial;
    use proptest::prelude::*;

    /// Pretends to be rank `rank` of `size` ranks that all pass the same buffer: the sum is
    /// `size` times the value and the exclusive scan `rank` times. Only u64 sums.
    #[derive(Clone)]
    struct Uniform {
        rank: usize,
        size: usize,
    }

    impl Uniform {
        fn scale(buf: &mut [u8], dtype: DType, op: ReduceOp, factor: usize) {
            assert!(matches!((dtype, op), (DType::U64, ReduceOp::Sum)));
            let words: &mut [u64] = bytemuck::cast_slice_mut(buf);
            for w in words {
                *w = w
                    .checked_mul(factor as u64)
                    .expect("the test values do not wrap");
            }
        }
    }

    impl CommDyn for Uniform {
        fn rank(&self) -> usize {
            self.rank
        }
        fn size(&self) -> usize {
            self.size
        }
        fn barrier(&self) {}
        fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
            Self::scale(buf, dtype, op, self.size);
        }
        fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
            Self::scale(buf, dtype, op, self.rank);
        }
        fn all_to_all_v_bytes(
            &self,
            _: &[u8],
            _: &[usize],
            _: &mut [u8],
            _: &[usize],
            _: DType,
        ) -> Result<()> {
            unimplemented!()
        }
        fn neighbour_all_to_all_v_bytes(
            &self,
            _: &Neighbourhood,
            _: &[u8],
            _: &[usize],
            _: &mut [u8],
            _: &[usize],
            _: DType,
        ) -> Result<()> {
            unimplemented!()
        }
        #[cfg(feature = "mpi")]
        fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
            None
        }
        fn duplicate(&self) -> std::rc::Rc<dyn CommDyn> {
            std::rc::Rc::new(self.clone())
        }
    }

    #[test]
    fn empty_map() {
        let map = IndexMap::new(&Serial, 0).unwrap();
        assert_eq!(map.owned_range(), (0, 0));
        assert_eq!(
            (map.num_owned(), map.num_local(), map.size_global()),
            (0, 0, 0)
        );
    }

    #[test]
    fn largest_serial_map_fits_and_one_more_is_too_large() {
        let max = i64::MAX as usize;
        let map = IndexMap::new(&Serial, max).unwrap();
        assert_eq!(map.owned_range(), (0, i64::MAX));
        assert_eq!(map.num_owned(), max);
        assert!(matches!(
            IndexMap::new(&Serial, max + 1),
            Err(GridError::TooLarge)
        ));
        assert!(matches!(
            IndexMap::new(&Serial, usize::MAX),
            Err(GridError::TooLarge)
        ));
    }

    #[test]
    fn ranges_on_several_ranks() {
        for rank in 0..4 {
            let comm = Uniform { rank, size: 4 };
            let map = IndexMap::new(&comm, 5).unwrap();
            let begin = 5 * rank as i64;
            assert_eq!(map.owned_range(), (begin, begin + 5));
            assert_eq!(map.size_global(), 20);
            assert_eq!(map.num_ghosts(), 0);
        }
    }

    #[test]
    fn global_overflow_is_too_large_on_every_rank() {
        // 3 * 2^61 fits in i64; 4 * 2^61 = 2^63 does not.
        let n = 1_usize << 61;
        let last = IndexMap::new(&Uniform { rank: 2, size: 3 }, n).unwrap();
        assert_eq!(last.owned_range(), (1 << 62, 3 << 61));
        assert_eq!(last.size_global(), 3 << 61);
        for rank in 0..4 {
            let r = IndexMap::new(&Uniform { rank, size: 4 }, n);
            assert!(matches!(r, Err(GridError::TooLarge)), "rank {rank}");
        }
        // The sum of the halves does not wrap even when the total exceeds u64.
        for rank in 0..4 {
            let r = IndexMap::new(&Uniform { rank, size: 4 }, 1 << 62);
            assert!(matches!(r, Err(GridError::TooLarge)), "rank {rank}");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn prop_serial_map(n in 0..=(i64::MAX as usize)) {
            let map = IndexMap::new(&Serial, n).unwrap();
            prop_assert_eq!(map.owned_range(), (0, n as i64));
            prop_assert_eq!(map.size_global(), n as i64);
            prop_assert_eq!(map.num_owned(), n);
            prop_assert_eq!(map.num_ghosts(), 0);
            prop_assert_eq!(map.num_local(), n);
            prop_assert!(map.ghosts().is_empty());
            prop_assert!(map.ghost_owners().is_empty());
        }

        #[test]
        fn prop_uniform_ranks(size in 1..64_usize, rank_seed in any::<usize>(), n in 0..(1_usize << 40)) {
            let rank = rank_seed % size;
            let map = IndexMap::new(&Uniform { rank, size }, n).unwrap();
            let begin = (rank * n) as i64;
            prop_assert_eq!(map.owned_range(), (begin, begin + n as i64));
            prop_assert_eq!(map.size_global(), (size * n) as i64);
            prop_assert_eq!(map.num_local(), n);
        }

        #[test]
        fn prop_too_large_serial(n in (i64::MAX as usize + 1)..=usize::MAX) {
            prop_assert!(matches!(IndexMap::new(&Serial, n), Err(GridError::TooLarge)));
        }
    }
}
