//! `CellBlock` (Listing B3).

use tetraloma_device::{DeviceBuffer, DeviceStrided, Result};
use tetraloma_dims::TopoDim;
use tetraloma_parallel::{CommDyn, IndexMap};

// Design: Listing B3.
/// The cells of one cell type: the unit in which a mesh stores its cells.
///
/// A block's arrays have fixed row lengths, so a device kernel treats all its cells alike. See
/// [Cell blocks] and the tutorial [Blocks].
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceStrided};
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_parallel::{IndexMap, Serial};
/// use tetraloma_topology::{CellBlock, encode_origin_ids};
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
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
///
/// [Cell blocks]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cell-blocks
/// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
pub struct CellBlock<TD: TopoDim> {
    /// The cell type, e.g. `Cell3::Prism`.
    pub cell: TD::Cell,
    /// The [owned | ghost] cells of this block.
    pub index_map: IndexMap,
    /// Local vertex indices in reference-cell order, one row per cell.
    pub vertices: DeviceStrided<u32>,
    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each cell
    /// ([`encode_origin_ids`](crate::encode_origin_ids)).
    pub origin_ids: DeviceBuffer<u32>,
    /// Bit 0: manifold orientation flipped (codimension 1).
    pub flags: DeviceBuffer<u32>,
}

impl<TD: TopoDim> CellBlock<TD> {
    /// A cell block from its parts; see the fields for their layout.
    ///
    /// It checks nothing: [`Topology::new`](crate::Topology::new) checks the lengths. See
    /// [Cell blocks].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_parallel::{IndexMap, Serial};
    /// use tetraloma_topology::{CellBlock, encode_origin_ids};
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // One tetrahedron with origin id 42 on the vertices 0–3.
    /// let block = CellBlock::<D3>::new(
    ///     Cell3::Tetrahedron,
    ///     IndexMap::new(&Serial, 1)?,
    ///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3])?, stride: 4 },
    ///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[42]))?,
    ///     DeviceBuffer::from_host(&client, &[0u32])?,
    /// );
    /// assert_eq!(block.vertices.stride, 4);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Cell blocks]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cell-blocks
    pub fn new(
        cell: TD::Cell,
        index_map: IndexMap,
        vertices: DeviceStrided<u32>,
        origin_ids: DeviceBuffer<u32>,
        flags: DeviceBuffer<u32>,
    ) -> Self {
        CellBlock {
            cell,
            index_map,
            vertices,
            origin_ids,
            flags,
        }
    }

    /// A deep copy of every device array. The index map, ghosts included, is cloned and
    /// shares its neighbourhoods with the original; `comm` is unused.
    ///
    /// See [Cell blocks].
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    ///
    /// # Example
    ///
    /// ```
    /// // topology: two triangles (0, 1, 2) and (3, 2, 1) on four vertices with origin ids
    /// // 10–13, built as in the example of `Topology::new`.
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell2, D2};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// # use tetraloma_topology::{CellBlock, Topology, encode_origin_ids};
    /// # let client = tetraloma_testing::test_client();
    /// # let block = CellBlock::<D2>::new(
    /// #     Cell2::Triangle,
    /// #     IndexMap::new(&Serial, 2)?,
    /// #     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1])?, stride: 3 },
    /// #     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1]))?,
    /// #     DeviceBuffer::from_host(&client, &[0u32, 0])?,
    /// # );
    /// # let topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// let copy = topology.cell_blocks()[0].try_clone(&Serial)?;
    /// assert_eq!(copy.vertices.data.to_host()?, [0, 1, 2, 3, 2, 1]);
    /// assert_eq!(copy.index_map.num_local(), 2);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Cell blocks]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cell-blocks
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let _ = comm;
        Ok(CellBlock {
            cell: self.cell,
            index_map: self.index_map.clone(),
            vertices: crate::topology::clone_strided(&self.vertices)?,
            origin_ids: self.origin_ids.deep_clone()?,
            flags: self.flags.deep_clone()?,
        })
    }
}
