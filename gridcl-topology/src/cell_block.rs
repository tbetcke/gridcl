//! `CellBlock` (Listing B3).

use gridcl_device::{DeviceBuffer, DeviceStrided, Result};
use gridcl_dims::TopoDim;
use gridcl_parallel::IndexMap;

/// The cells of one cell type (Listing B3).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceStrided};
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_parallel::{IndexMap, Serial};
/// use gridcl_topology::{CellBlock, encode_origin_ids};
///
/// let client = gridcl_testing::test_client();
/// let vertices = DeviceBuffer::from_host(&client, &[0u32, 1, 2]).unwrap();
/// let block = CellBlock::<D2>::new(
///     Cell2::Triangle,
///     IndexMap::new(&Serial, 1).unwrap(),
///     DeviceStrided { data: vertices, stride: 3 },
///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[7])).unwrap(),
///     DeviceBuffer::from_host(&client, &[0u32]).unwrap(),
/// );
/// assert_eq!(block.cell, Cell2::Triangle);
/// assert_eq!(block.index_map.num_local(), 1);
/// ```
pub struct CellBlock<TD: TopoDim> {
    /// The cell type, e.g. `Cell3::Prism`.
    pub cell: TD::Cell,
    /// The [owned | ghost] cells of this block.
    pub index_map: IndexMap,
    /// Local vertex indices in reference-cell order, one row per cell.
    pub vertices: DeviceStrided<u32>,
    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each cell.
    pub origin_ids: DeviceBuffer<u32>,
    /// Bit 0: manifold orientation flipped (codimension 1).
    pub flags: DeviceBuffer<u32>,
}

impl<TD: TopoDim> CellBlock<TD> {
    /// A cell block from its parts; see the fields for their layout.
    pub fn new(
        cell: TD::Cell,
        index_map: IndexMap,
        vertices: DeviceStrided<u32>,
        origin_ids: DeviceBuffer<u32>,
        flags: DeviceBuffer<u32>,
    ) -> Self {
        let _ = (cell, index_map, vertices, origin_ids, flags);
        todo!()
    }

    /// A deep copy of every device array.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        todo!()
    }
}
