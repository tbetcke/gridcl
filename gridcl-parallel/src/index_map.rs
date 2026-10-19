//! `IndexMap` without ghosts (M2); M6 adds ghosts, `scatterer` and the device field.

use crate::{CommDyn, DType, GridError, Neighbourhood, ReduceOp, Result};

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
    #[allow(dead_code)] // M6 reads it for the scatterer.
    neighbourhood: Neighbourhood,
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
            neighbourhood: Neighbourhood::empty(),
        })
    }
    /// Global indices `[begin, end)` owned by this rank.
    pub fn owned_range(&self) -> (i64, i64) {
        self.owned_range
    }
    /// Global index of each ghost.
    pub fn ghosts(&self) -> &[i64] {
        &self.ghosts
    }
    /// Owning rank of each ghost.
    pub fn ghost_owners(&self) -> &[i32] {
        &self.ghost_owners
    }
    /// Owned indices on this rank.
    pub fn num_owned(&self) -> usize {
        // Fits: it was a usize when the map was built.
        (self.owned_range.1 - self.owned_range.0) as usize
    }
    /// Ghost indices on this rank.
    pub fn num_ghosts(&self) -> usize {
        self.ghosts.len()
    }
    /// Owned plus ghosts.
    pub fn num_local(&self) -> usize {
        self.num_owned() + self.num_ghosts()
    }
    /// Size of the global index set.
    pub fn size_global(&self) -> i64 {
        self.size_global
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
