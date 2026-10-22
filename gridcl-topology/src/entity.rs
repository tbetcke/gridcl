//! Entity creation, steps 3–5: sort, unique and canonical vertex lists; `EntityBlock` and
//! `Connectivity` (Listing T1).

use gridcl_device::{DeviceCsr, DeviceStrided, Result};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, IndexMap};

/// The entities of one type and dimension (Listing T1).
pub struct EntityBlock {
    /// The entity type, e.g. `Triangle`.
    pub cell: CellType,
    /// The [owned | ghost] entities of this block.
    pub index_map: IndexMap,
    /// Local vertex indices in canonical orientation (A.6), one row per entity.
    pub vertices: DeviceStrided<u32>,
    /// The first flat index of this block in its dimension.
    pub offset: u32,
}

impl EntityBlock {
    /// A deep copy of every device array; the index map is rebuilt on `comm` (collective).
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone` and `IndexMap::new`.
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let _ = comm;
        todo!()
    }
}

/// A stored connectivity (Listing T1).
#[allow(clippy::large_enum_variant)] // Listing T1's layout; one value per stored key
pub enum Connectivity {
    /// One strided array per cell block (cell -> edges, cell -> faces, ...), flat indices.
    Downward(Vec<DeviceStrided<u32>>),
    /// Variable-length rows (entity -> cells, vertex -> cells).
    Upward(DeviceCsr),
}

impl Connectivity {
    /// A deep copy of every device array.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        todo!()
    }
}
