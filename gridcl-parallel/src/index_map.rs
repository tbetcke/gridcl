//! `IndexMap` without ghosts (M2); M6 adds ghosts, `scatterer` and the device field.

use crate::{CommDyn, Neighbourhood, Result};

/// Ownership of a distributed index set in the DOLFINx sense: `[owned | ghost]` local indices.
pub struct IndexMap {
    owned_range: (i64, i64),
    ghosts: Vec<i64>,
    ghost_owners: Vec<i32>,
    size_global: i64,
    #[allow(dead_code)] // M6 reads it for the scatterer.
    neighbourhood: Neighbourhood,
}

impl IndexMap {
    /// Without ghosts: owned range from `exscan_sum` of `num_owned`, size from `sum`; an i64
    /// overflow gives `TooLarge`.
    pub fn new(_comm: &dyn CommDyn, _num_owned: usize) -> Result<IndexMap> {
        todo!()
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
        todo!()
    }
    /// Ghost indices on this rank.
    pub fn num_ghosts(&self) -> usize {
        todo!()
    }
    /// Owned plus ghosts.
    pub fn num_local(&self) -> usize {
        todo!()
    }
    /// Size of the global index set.
    pub fn size_global(&self) -> i64 {
        self.size_global
    }
}
