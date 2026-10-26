//! Matching of entity keys against the entities of one dimension (ADR-0022).

use gridcl_device::{DeviceBuffer, Result, SortKeys};
use gridcl_dims::TopoDim;

use crate::Topology;

impl<TD: TopoDim> Topology<TD> {
    /// The flat index of the entity of dimension `dim` whose sorted local vertex indices are
    /// each key of `keys`, or `u32::MAX` where no entity matches. The word count of `keys`
    /// selects the entity type (1: points, 2: edges, 3: triangles, 4: quadrilaterals).
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` if the entities of `dim` have not been created; allocation
    /// and device errors.
    pub fn match_entities(&self, dim: usize, keys: &SortKeys) -> Result<DeviceBuffer<u32>> {
        let _ = (dim, keys);
        todo!("M4 T9")
    }
}
