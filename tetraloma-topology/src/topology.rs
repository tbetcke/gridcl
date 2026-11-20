//! `Topology<TD>` (ADR-0012): cell blocks, vertex `IndexMap` and origin ids, entity blocks,
//! connectivity and permutation info.

use std::collections::HashMap;

use cubecl::prelude::Client;
use tetraloma_device::{ConnKey, DeviceBuffer, DeviceStrided, GridError, Result};
use tetraloma_dims::{CellType, TopoDim};
use tetraloma_parallel::{CommDyn, IndexMap};

use crate::entity::Cells;
use crate::ownership::Sharing;
use crate::{
    CellBlock, Connectivity, EntityBlock, VertexSharers, encode_origin_ids, origin_ids_fit_u32,
};

/// A deep copy of a strided array.
pub(crate) fn clone_strided(s: &DeviceStrided<u32>) -> Result<DeviceStrided<u32>> {
    Ok(DeviceStrided {
        data: s.data.deep_clone()?,
        stride: s.stride,
    })
}

// Design: "Topology construction on the device"; ADR-0012.
/// The topology of a mesh: its cell blocks, vertices, entities, connectivity and permutation
/// bits.
///
/// Entities of dimension 0 ≤ d < tdim are created on demand; cells stay in
/// [`cell_blocks`](Topology::cell_blocks). Stored connectivities are (tdim, d), downward and filled
/// by entity creation, for 1 ≤ d < tdim, and (d, tdim), upward, for 0 ≤ d < tdim; (tdim, 0) is
/// `CellBlock::vertices` and d → 0 is `EntityBlock::vertices`. A `tetraloma::Mesh` owns one and
/// offers its operations as mesh methods. See [Meshes and topology].
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceStrided};
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_parallel::{IndexMap, Serial};
/// use tetraloma_topology::{CellBlock, Topology, encode_origin_ids};
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // Two triangles sharing the edge (1, 2).
/// let block = CellBlock::<D2>::new(
///     Cell2::Triangle,
///     IndexMap::new(&Serial, 2).unwrap(),
///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(), stride: 3 },
///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1])).unwrap(),
///     DeviceBuffer::from_host(&client, &[0u32, 0]).unwrap(),
/// );
/// let topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4).unwrap(), &[10, 11, 12, 13]).unwrap();
/// assert!(topology.ids_fit_u32());
/// // The second triangle's edges (3, 2), (3, 1) and (2, 1) all run from a higher to a lower id.
/// let bits = topology.permutation_info(0).unwrap().to_host().unwrap();
/// assert_eq!(bits, [0b000, 0b111]);
/// ```
///
/// [Meshes and topology]: https://tetracl.github.io/tetraloma/concepts/meshes.html
pub struct Topology<TD: TopoDim> {
    client: Client,
    cell_blocks: Vec<CellBlock<TD>>,
    vertex_index_map: IndexMap,
    vertex_origin_ids: DeviceBuffer<u32>,
    ids_fit_u32: bool,
    permutation_info: Vec<Option<DeviceBuffer<u32>>>,
    /// Entity blocks of dimension d < tdim, once created.
    entities: Vec<Option<Vec<EntityBlock>>>,
    connectivity: HashMap<ConnKey, Connectivity>,
    /// The sharer sets of the local vertices; empty rows until `set_vertex_sharers`.
    vertex_sharers: VertexSharers,
    /// One word per local vertex, 1 if another rank holds it; `None` without sharers.
    vertex_shared: Option<DeviceBuffer<u32>>,
    /// The sharing neighbourhood: the ranks, ascending, of every shared vertex's row (this
    /// rank included, which entity creation drops).
    sharing_ranks: Vec<i32>,
}

impl<TD: TopoDim> Topology<TD> {
    /// The topology of `cell_blocks`, whose vertex rows index the `vertex_index_map.num_local()`
    /// vertices with origin ids `vertex_origin_ids`. Uploads the encoded vertex origin ids and
    /// computes the permutation info of every block of dimension ≥ 2.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `vertex_origin_ids` does not have one id per local vertex, or a
    /// block's vertex rows are not its cell's vertex count, its `origin_ids` not two words
    /// per cell or its `flags` not one word per cell (cells: its index map's `num_local`);
    /// `TooLarge` if a length overflows; device errors.
    ///
    /// The vertex indices themselves stay on the device and are not checked. Out-of-range indices
    /// are not an error but give a meaningless topology: every kernel that reads through a vertex
    /// index clamps an index past the local vertices to the last vertex, so no read leaves its
    /// buffer, and entities, keys and permutation info are computed as if the index were the last
    /// vertex. The (tdim, 0) connectivity, [`CellBlock::vertices`](crate::CellBlock), is the input
    /// unchanged and keeps the raw indices. See [Cell blocks] and [Origin ids].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided, GridError};
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_parallel::{IndexMap, Serial};
    /// use tetraloma_topology::{CellBlock, Topology, encode_origin_ids};
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let block = || -> Result<CellBlock<D2>, GridError> {
    ///     Ok(CellBlock::new(
    ///         Cell2::Triangle,
    ///         IndexMap::new(&Serial, 2)?,
    ///         DeviceStrided {
    ///             data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1])?,
    ///             stride: 3,
    ///         },
    ///         DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1]))?,
    ///         DeviceBuffer::from_host(&client, &[0u32, 0])?,
    ///     ))
    /// };
    /// // Four vertices with origin ids 10–13.
    /// let topology = Topology::new(&client, vec![block()?], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// assert_eq!(topology.cell_blocks().len(), 1);
    ///
    /// // One origin id per local vertex, or `LengthMismatch`.
    /// let short = Topology::new(&client, vec![block()?], IndexMap::new(&Serial, 4)?, &[10, 11]);
    /// assert!(matches!(short, Err(GridError::LengthMismatch { expected: 4, found: 2 })));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Cell blocks]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cell-blocks
    /// [Origin ids]: https://tetracl.github.io/tetraloma/concepts/meshes.html#origin-ids
    pub fn new(
        client: &Client,
        cell_blocks: Vec<CellBlock<TD>>,
        vertex_index_map: IndexMap,
        vertex_origin_ids: &[i64],
    ) -> Result<Self> {
        let num_vertices = vertex_index_map.num_local();
        if vertex_origin_ids.len() != num_vertices {
            return Err(GridError::LengthMismatch {
                expected: num_vertices,
                found: vertex_origin_ids.len(),
            });
        }
        check_cell_blocks(&cell_blocks)?;
        let ids_fit_u32 = origin_ids_fit_u32(vertex_origin_ids);
        let ids = DeviceBuffer::from_host(client, &encode_origin_ids(vertex_origin_ids))?;
        Self::assemble(client, cell_blocks, vertex_index_map, ids, ids_fit_u32)
    }

    /// The topology of checked cell blocks and encoded vertex origin ids, with the
    /// permutation info of every block of dimension ≥ 2.
    fn assemble(
        client: &Client,
        cell_blocks: Vec<CellBlock<TD>>,
        vertex_index_map: IndexMap,
        ids: DeviceBuffer<u32>,
        ids_fit_u32: bool,
    ) -> Result<Self> {
        let permutation_info = cell_blocks
            .iter()
            .map(|b| {
                crate::permutation::permutation_info(b.cell.into(), &b.vertices, &ids, ids_fit_u32)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Topology {
            client: client.clone(),
            cell_blocks,
            vertex_index_map,
            vertex_origin_ids: ids,
            ids_fit_u32,
            permutation_info,
            entities: (0..TD::N).map(|_| None).collect(),
            connectivity: HashMap::new(),
            vertex_sharers: VertexSharers::default(),
            vertex_shared: None,
            sharing_ranks: Vec::new(),
        })
    }

    /// [`new`](Topology::new) with the vertex origin ids already on the device, as interleaved
    /// (hi ^ 0x8000_0000, lo) words ([`encode_origin_ids`]): submesh creation keeps the
    /// parent's ids without a host round trip (ADR-0039, item 17). Computes the permutation
    /// info as `new` does. The ids are not read back, so [`ids_fit_u32`](Topology::ids_fit_u32)
    /// is false and the kernels take the general two-word path, with the same results. For
    /// tetraloma's submeshes; not part of the stable API (test: `from_device_ids_equals_new`).
    ///
    /// # Errors
    ///
    /// As [`new`](Topology::new), with `LengthMismatch` if `vertex_origin_ids` does not have
    /// two words per local vertex.
    #[doc(hidden)]
    pub fn from_device_ids(
        client: &Client,
        cell_blocks: Vec<CellBlock<TD>>,
        vertex_index_map: IndexMap,
        vertex_origin_ids: DeviceBuffer<u32>,
    ) -> Result<Self> {
        let words = vertex_index_map
            .num_local()
            .checked_mul(2)
            .ok_or(GridError::TooLarge)?;
        if vertex_origin_ids.len() != words {
            return Err(GridError::LengthMismatch {
                expected: words,
                found: vertex_origin_ids.len(),
            });
        }
        check_cell_blocks(&cell_blocks)?;
        Self::assemble(
            client,
            cell_blocks,
            vertex_index_map,
            vertex_origin_ids,
            false,
        )
    }

    /// The client the arrays live on.
    ///
    /// See [Compute devices].
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
    /// // A new buffer on the topology's device.
    /// let flags = DeviceBuffer::from_host(topology.client(), &[0u32, 1])?;
    /// assert_eq!(flags.to_host()?, [0, 1]);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Compute devices]: https://tetracl.github.io/tetraloma/devices.html
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The cell blocks, in creation order.
    ///
    /// See [Blocks].
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
    /// let blocks = topology.cell_blocks();
    /// assert_eq!(blocks.len(), 1);
    /// assert_eq!(blocks[0].cell, Cell2::Triangle);
    /// assert_eq!(blocks[0].index_map.num_local(), 2);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
    pub fn cell_blocks(&self) -> &[CellBlock<TD>] {
        &self.cell_blocks
    }

    /// The [owned | ghost] vertices.
    ///
    /// See [Ownership and ghosts].
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
    /// // A serial topology has no ghosts.
    /// let vertices = topology.vertex_index_map();
    /// assert_eq!((vertices.num_owned(), vertices.num_ghosts()), (4, 0));
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Ownership and ghosts]: https://tetracl.github.io/tetraloma/concepts/meshes.html#ownership-and-ghosts
    pub fn vertex_index_map(&self) -> &IndexMap {
        &self.vertex_index_map
    }

    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each local vertex
    /// ([`encode_origin_ids`]).
    ///
    /// See [Origin ids].
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
    /// // A host read, for inspection only.
    /// let words = topology.vertex_origin_ids().to_host()?;
    /// assert_eq!(tetraloma_topology::decode_origin_ids(&words), [10, 11, 12, 13]);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Origin ids]: https://tetracl.github.io/tetraloma/concepts/meshes.html#origin-ids
    pub fn vertex_origin_ids(&self) -> &DeviceBuffer<u32> {
        &self.vertex_origin_ids
    }

    /// Whether every vertex origin id lies in [0, 2³²): then the kernels compare one word per
    /// id instead of two ([`origin_ids_fit_u32`]).
    ///
    /// See [Origin ids].
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
    /// // The origin ids 10–13 fit in 32 bits.
    /// assert!(topology.ids_fit_u32());
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Origin ids]: https://tetracl.github.io/tetraloma/concepts/meshes.html#origin-ids
    pub fn ids_fit_u32(&self) -> bool {
        self.ids_fit_u32
    }

    // Design: Appendix A, A.6.
    /// The orientation word of each cell of block `b`; `None` for interval cells.
    ///
    /// The chapter [Orientation bits] gives the bit layout of every cell.
    ///
    /// # Panics
    ///
    /// If `b` is not a block index.
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
    /// // Edge bits, one per edge: set when the edge runs from a higher origin id to a lower.
    /// // The second triangle's edges (3, 2), (3, 1) and (2, 1) all do.
    /// let bits = topology.permutation_info(0).expect("triangles have edge bits").to_host()?;
    /// assert_eq!(bits, [0b000, 0b111]);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Orientation bits]: https://tetracl.github.io/tetraloma/elements/orientation-bits.html
    pub fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>> {
        self.permutation_info[b].as_ref()
    }

    /// The entity blocks of dimension `dim`, one per entity type.
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` until `create_entities(dim)` has run, and always for
    /// `dim` ≥ tdim.
    ///
    /// See [Entities].
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
    /// # let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// # use tetraloma_device::GridError;
    /// assert!(matches!(topology.entity_blocks(1), Err(GridError::EntitiesNotCreated(1))));
    /// topology.create_entities(&Serial, 1)?;
    /// // One block of edges: five edges, numbered from 0.
    /// let edges = &topology.entity_blocks(1)?[0];
    /// assert_eq!((edges.index_map.num_local(), edges.offset), (5, 0));
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Entities]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#entities
    pub fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]> {
        self.entities
            .get(dim)
            .and_then(Option::as_deref)
            .ok_or(GridError::EntitiesNotCreated(dim))
    }

    /// The cell type and vertex rows of each cell block.
    pub(crate) fn cell_rows(&self) -> Vec<(CellType, &DeviceStrided<u32>)> {
        self.cell_blocks
            .iter()
            .map(|b| (b.cell.into(), &b.vertices))
            .collect()
    }

    /// What entity creation reads: the cell rows, owned counts, origin ids and sharing.
    pub(crate) fn entity_cells(&self) -> Cells<'_> {
        Cells {
            rows: self.cell_rows(),
            owned: self
                .cell_blocks
                .iter()
                .map(|b| b.index_map.num_owned())
                .collect(),
            ids: &self.vertex_origin_ids,
            fast: self.ids_fit_u32,
            sharing: self.vertex_shared.as_ref().map(|flags| Sharing {
                flags,
                sharers: &self.vertex_sharers,
                ranks: &self.sharing_ranks,
            }),
        }
    }

    /// Whether tetraloma stores the connectivity `key`.
    fn supported(key: ConnKey) -> bool {
        let (from, to, tdim) = (usize::from(key.from), usize::from(key.to), TD::N);
        (from == tdim && (1..tdim).contains(&to)) || (to == tdim && from < tdim)
    }

    /// The connectivity `key`.
    ///
    /// # Errors
    ///
    /// `ConnectivityNotCreated(key)` until it has been created; `UnsupportedConnectivity(key)`
    /// for pairs other than (tdim, d) with 1 ≤ d < tdim and (d, tdim) with 0 ≤ d < tdim.
    ///
    /// See [Connectivity].
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
    /// # let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// # use tetraloma_device::{ConnKey, GridError};
    /// # use tetraloma_topology::Connectivity;
    /// let key = ConnKey { from: 1, to: 2 };
    /// assert!(matches!(topology.connectivity(key), Err(GridError::ConnectivityNotCreated(_))));
    /// // Cell → vertex is the cell blocks' `vertices`, not a stored connectivity.
    /// let cell_vertex = ConnKey { from: 2, to: 0 };
    /// assert!(matches!(topology.connectivity(cell_vertex), Err(GridError::UnsupportedConnectivity(_))));
    ///
    /// topology.create_connectivity(&Serial, 1, 2)?;
    /// let Connectivity::Upward(csr) = topology.connectivity(key)? else {
    ///     unreachable!("edge -> cell is upward")
    /// };
    /// // Five edges; only the shared edge (1, 2) has two cells.
    /// assert_eq!((csr.offsets.len(), csr.indices.len()), (6, 6));
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Connectivity]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#connectivity
    pub fn connectivity(&self, key: ConnKey) -> Result<&Connectivity> {
        if !Self::supported(key) {
            return Err(GridError::UnsupportedConnectivity(key));
        }
        self.connectivity
            .get(&key)
            .ok_or(GridError::ConnectivityNotCreated(key))
    }

    // ADR-0026; submesh creation: ADR-0039, item 16.
    /// Stores the sharer sets of the local vertices, uploads one shared flag per vertex and derives
    /// the sharing neighbourhood: every rank that shares a vertex with this one. Without sharers
    /// every vertex is unshared. Mesh creation and submesh creation call it. See
    /// [Ownership and ghosts].
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `s` does not have one row per local vertex, its offsets do not
    /// start at 0 and ascend, or `ranks` and `owned_cell` do not match the offsets;
    /// `IndexOutOfRange { index: v, len }` for the first vertex v whose row is not strictly
    /// ascending, holds a negative rank, or lacks this rank (the one rank common to every
    /// row); device errors of the upload. On error nothing is stored.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided, GridError};
    /// use tetraloma_dims::{Cell1, D1};
    /// use tetraloma_parallel::{IndexMap, Serial};
    /// use tetraloma_topology::{CellBlock, Topology, VertexSharers, encode_origin_ids};
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let block = CellBlock::<D1>::new(
    ///     Cell1::Interval,
    ///     IndexMap::new(&Serial, 1).unwrap(),
    ///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 },
    ///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0])).unwrap(),
    ///     DeviceBuffer::from_host(&client, &[0u32]).unwrap(),
    /// );
    /// let mut t = Topology::new(&client, vec![block], IndexMap::new(&Serial, 2).unwrap(), &[0, 1]).unwrap();
    /// // Rank 1 holds both vertices; vertex 1 is shared with rank 2.
    /// let ok = VertexSharers { offsets: vec![0, 1, 3], ranks: vec![1, 1, 2], owned_cell: vec![true; 3] };
    /// // Vertex 1's row lacks rank 1, the rank of vertex 0's row.
    /// let bad = VertexSharers { offsets: vec![0, 1, 2], ranks: vec![1, 2], owned_cell: vec![true; 2] };
    /// assert!(matches!(t.set_vertex_sharers(bad), Err(GridError::IndexOutOfRange { index: 1, len: 2 })));
    /// t.set_vertex_sharers(ok.clone()).unwrap();
    /// assert_eq!(t.vertex_sharers(), Some(&ok));
    /// ```
    ///
    /// [Ownership and ghosts]: https://tetracl.github.io/tetraloma/concepts/meshes.html#ownership-and-ghosts
    pub fn set_vertex_sharers(&mut self, s: VertexSharers) -> Result<()> {
        let n = self.vertex_index_map.num_local();
        check_sharers(&s, n)?;
        let mut flags = Vec::with_capacity(n);
        let mut ranks = Vec::new();
        for row in s.offsets.windows(2) {
            let row = &s.ranks[row[0]..row[1]];
            let shared = row.len() > 1;
            flags.push(u32::from(shared));
            if shared {
                ranks.extend_from_slice(row);
            }
        }
        ranks.sort_unstable();
        ranks.dedup();
        // The vertex shared flags at mesh and submesh creation (CLAUDE.md, ADR-0026, ADR-0039).
        let flags = DeviceBuffer::from_host(&self.client, &flags)?;
        self.vertex_sharers = s;
        self.vertex_shared = Some(flags);
        self.sharing_ranks = ranks;
        Ok(())
    }

    /// The sharer sets stored by [`set_vertex_sharers`](Topology::set_vertex_sharers);
    /// `None` before. For tetraloma's submeshes, which restrict the parent's rows (ADR-0039,
    /// items 6 and 17); not part of the stable API (test: `vertex_sharers_round_trip`).
    #[doc(hidden)]
    pub fn vertex_sharers(&self) -> Option<&VertexSharers> {
        self.vertex_shared.as_ref().map(|_| &self.vertex_sharers)
    }

    /// Replaces the flags of cell block `b`, one word per local cell: bit 0 is the manifold
    /// orientation (`orient`, submesh creation; ADR-0039, item 17). Not part of the stable
    /// API (test: `set_cell_flags_checks_its_input`).
    ///
    /// # Errors
    ///
    /// `IndexOutOfRange` if `b` is not a block index; `LengthMismatch` if `flags` does not
    /// have one word per local cell of block `b`.
    #[doc(hidden)]
    pub fn set_cell_flags(&mut self, b: usize, flags: DeviceBuffer<u32>) -> Result<()> {
        let len = self.cell_blocks.len();
        let block = self
            .cell_blocks
            .get_mut(b)
            .ok_or(GridError::IndexOutOfRange { index: b, len })?;
        let rows = block.index_map.num_local();
        if flags.len() != rows {
            return Err(GridError::LengthMismatch {
                expected: rows,
                found: flags.len(),
            });
        }
        block.flags = flags;
        Ok(())
    }

    // Design: "Ownership of vertices and entities"; ADR-0026, item 15.
    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and, for dim ≥ 1, the downward
    /// connectivity (tdim, dim). Does nothing if they exist.
    ///
    /// Collective on more than one rank: every rank of `comm` calls it with the same `dim`.
    /// Entities whose vertices are all shared
    /// ([`set_vertex_sharers`](Topology::set_vertex_sharers)) get their owners by an exchange over
    /// the sharing neighbourhood, and every rank returns an error if one rank fails. Each
    /// entity block is numbered [owned | ghost]: owned entities by their smallest touching cell,
    /// ghosts by (owner rank, global index), with global numbers from an exclusive scan of the
    /// owned counts and, for ghosts, from their owners. The dimension-0 `Point` block shares the
    /// vertex index map. See [Cells and entities] and [Ownership and ghosts].
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim; `TooLarge` if the padded emission exceeds
    /// 2^32 keys or the cell blocks hold 2^31 cells or more; device and communication errors;
    /// `RemoteFailure` if another rank failed.
    ///
    /// # Panics
    ///
    /// If an entity whose vertices are all shared has no holder in an owned cell, or a ghost
    /// entity receives no global number, which happens only if the vertex sharers are not
    /// consistent across ranks.
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
    /// # let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// topology.create_entities(&Serial, 1)?;
    /// // The five edges, each stored from its lower origin id to its higher one (a host read,
    /// // for inspection only), and the downward connectivity cell → edge.
    /// let words = topology.entity_blocks(1)?[0].vertices.data.to_host()?;
    /// let mut edges: Vec<[u32; 2]> = words.chunks(2).map(|e| [e[0], e[1]]).collect();
    /// assert!(edges.iter().all(|e| e[0] < e[1]));
    /// edges.sort();
    /// assert_eq!(edges, [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]);
    /// assert!(topology.connectivity(tetraloma_device::ConnKey { from: 2, to: 1 }).is_ok());
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Cells and entities]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cells-and-entities
    /// [Ownership and ghosts]: https://tetracl.github.io/tetraloma/concepts/meshes.html#ownership-and-ghosts
    pub fn create_entities(&mut self, comm: &dyn CommDyn, dim: usize) -> Result<()> {
        self.create_entities_with(comm, dim, false)
    }

    /// [`create_entities`](Topology::create_entities) that gives the empty pages of the device
    /// pools back after the sort and after the grouping (`Client::memory_cleanup`), so their
    /// scratch does not stay reserved: tetraloma's mesh creation calls it (the M8 memory bound,
    /// ADR-0036). Each cleanup waits for the device, so other callers use `create_entities`.
    /// Not part of the stable API.
    ///
    /// # Errors
    ///
    /// As [`create_entities`](Topology::create_entities).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_parallel::{IndexMap, Serial};
    /// use tetraloma_topology::{CellBlock, Topology, encode_origin_ids};
    ///
    /// let client = tetraloma_testing::test_client();
    /// let block = CellBlock::<D2>::new(
    ///     Cell2::Triangle,
    ///     IndexMap::new(&Serial, 2).unwrap(),
    ///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(), stride: 3 },
    ///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1])).unwrap(),
    ///     DeviceBuffer::from_host(&client, &[0u32, 0]).unwrap(),
    /// );
    /// let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4).unwrap(), &[10, 11, 12, 13]).unwrap();
    /// topology.create_entities_releasing_pools(&Serial, 1).unwrap();
    /// assert_eq!(topology.entity_blocks(1).unwrap()[0].index_map.num_local(), 5);
    /// ```
    #[doc(hidden)]
    pub fn create_entities_releasing_pools(
        &mut self,
        comm: &dyn CommDyn,
        dim: usize,
    ) -> Result<()> {
        self.create_entities_with(comm, dim, true)
    }

    fn create_entities_with(
        &mut self,
        comm: &dyn CommDyn,
        dim: usize,
        release: bool,
    ) -> Result<()> {
        if dim >= TD::N {
            return Err(GridError::EntitiesNotCreated(dim));
        }
        if self.entities[dim].is_some() {
            return Ok(());
        }
        if dim == 0 {
            // One Point block of the vertices, with the vertex index map.
            let n = self.vertex_index_map.num_local();
            let block = EntityBlock {
                cell: CellType::Point,
                index_map: self.vertex_index_map.clone(),
                vertices: DeviceStrided {
                    data: crate::renumber::iota(&self.client, n)?,
                    stride: 1,
                },
                offset: 0,
            };
            self.entities[0] = Some(vec![block]);
            return Ok(());
        }
        let crate::entity::Dimension { blocks, downward } = crate::entity::create_dimension(
            &self.client,
            comm,
            &self.entity_cells(),
            dim,
            release,
        )?;
        self.entities[dim] = Some(blocks);
        self.connectivity.insert(
            ConnKey {
                from: TD::N as u8,
                to: dim as u8,
            },
            Connectivity::Downward(downward),
        );
        Ok(())
    }

    /// Creates the connectivity `from` → `to`, creating the entities it needs. Does nothing if
    /// it exists. Collective on more than one rank, as
    /// [`create_entities`](Topology::create_entities).
    ///
    /// # Errors
    ///
    /// `UnsupportedConnectivity` for pairs that are not stored; errors of
    /// [`create_entities`](Topology::create_entities).
    ///
    /// See [Connectivity].
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
    /// # let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// # use tetraloma_device::{ConnKey, GridError};
    /// # use tetraloma_topology::Connectivity;
    /// // Vertex → cell: the cells around each vertex, ascending.
    /// topology.create_connectivity(&Serial, 0, 2)?;
    /// let Connectivity::Upward(csr) = topology.connectivity(ConnKey { from: 0, to: 2 })? else {
    ///     unreachable!("vertex -> cell is upward")
    /// };
    /// assert_eq!(csr.offsets.to_host()?, [0, 1, 3, 5, 6]);
    /// assert_eq!(csr.indices.to_host()?, [0, 0, 1, 0, 1, 1]);
    ///
    /// // Edge → vertex is the edge blocks' `vertices`, not a stored connectivity.
    /// let err = topology.create_connectivity(&Serial, 1, 0);
    /// assert!(matches!(err, Err(GridError::UnsupportedConnectivity(_))));
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Connectivity]: https://tetracl.github.io/tetraloma/concepts/meshes.html#connectivity
    pub fn create_connectivity(
        &mut self,
        comm: &dyn CommDyn,
        from: usize,
        to: usize,
    ) -> Result<()> {
        let tdim = TD::N;
        let key = ConnKey {
            from: u8::try_from(from).map_err(|_| GridError::TooLarge)?,
            to: u8::try_from(to).map_err(|_| GridError::TooLarge)?,
        };
        if !Self::supported(key) {
            return Err(GridError::UnsupportedConnectivity(key));
        }
        if self.connectivity.contains_key(&key) {
            return Ok(());
        }
        if from == tdim {
            // Downward: filled by entity creation.
            return self.create_entities(comm, to);
        }
        self.create_entities(comm, from)?;
        let num_entities = self
            .entity_blocks(from)?
            .iter()
            .map(|b| b.index_map.num_local())
            .sum();
        let csr = if from == 0 {
            let rows: Vec<&DeviceStrided<u32>> =
                self.cell_blocks.iter().map(|b| &b.vertices).collect();
            crate::connectivity::upward(&self.client, &rows, num_entities)?
        } else {
            let down = ConnKey {
                from: tdim as u8,
                to: from as u8,
            };
            let Some(Connectivity::Downward(rows)) = self.connectivity.get(&down) else {
                unreachable!("entity creation stores the downward connectivity");
            };
            let rows: Vec<&DeviceStrided<u32>> = rows.iter().collect();
            crate::connectivity::upward(&self.client, &rows, num_entities)?
        };
        self.connectivity.insert(key, Connectivity::Upward(csr));
        Ok(())
    }

    /// The allocated bytes of every device array (cell blocks, vertex origin ids, permutation
    /// info, entities, connectivity, vertex shared flags) plus the
    /// [`footprint_bytes`](IndexMap::footprint_bytes) of every index map. The host sharer sets
    /// are not counted. For memory accounting in tests (`Mesh::footprint_bytes`, ADR-0035);
    /// not part of the stable API.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_parallel::{IndexMap, Serial};
    /// use tetraloma_topology::{CellBlock, Topology, encode_origin_ids};
    ///
    /// let client = tetraloma_testing::test_client();
    /// let block = CellBlock::<D2>::new(
    ///     Cell2::Triangle,
    ///     IndexMap::new(&Serial, 2).unwrap(),
    ///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(), stride: 3 },
    ///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1])).unwrap(),
    ///     DeviceBuffer::from_host(&client, &[0u32, 0]).unwrap(),
    /// );
    /// let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4).unwrap(), &[10, 11, 12, 13]).unwrap();
    /// let cells = topology.cell_blocks()[0].vertices.data.footprint_bytes();
    /// let before = topology.footprint_bytes();
    /// assert!(before > cells);
    /// // The edges add their vertices.
    /// topology.create_entities(&Serial, 1).unwrap();
    /// assert!(topology.footprint_bytes() > before);
    /// ```
    #[doc(hidden)]
    pub fn footprint_bytes(&self) -> usize {
        let cells: usize = self
            .cell_blocks
            .iter()
            .map(|b| {
                b.index_map.footprint_bytes()
                    + b.vertices.data.footprint_bytes()
                    + b.origin_ids.footprint_bytes()
                    + b.flags.footprint_bytes()
            })
            .sum();
        let entities: usize = self
            .entities
            .iter()
            .flatten()
            .flatten()
            .map(|e| e.index_map.footprint_bytes() + e.vertices.data.footprint_bytes())
            .sum();
        let connectivity: usize = self
            .connectivity
            .values()
            .map(|c| match c {
                Connectivity::Downward(blocks) => {
                    blocks.iter().map(|s| s.data.footprint_bytes()).sum()
                }
                Connectivity::Upward(csr) => {
                    csr.offsets.footprint_bytes() + csr.indices.footprint_bytes()
                }
            })
            .sum();
        let buffers = self
            .permutation_info
            .iter()
            .chain([&self.vertex_shared])
            .flatten()
            .map(DeviceBuffer::footprint_bytes)
            .sum::<usize>();
        cells
            + entities
            + connectivity
            + buffers
            + self.vertex_index_map.footprint_bytes()
            + self.vertex_origin_ids.footprint_bytes()
    }

    // `comm` stays in the signature: ADR-0012.
    /// A deep copy of every device array. Index maps, ghosts included, are cloned and share
    /// their neighbourhoods with the original, so the copy is not collective and `comm` is
    /// unused.
    ///
    /// See [Meshes and topology].
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
    /// # let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4)?, &[10, 11, 12, 13])?;
    /// topology.create_entities(&Serial, 1)?;
    /// let copy = topology.try_clone(&Serial)?;
    /// assert_eq!(copy.entity_blocks(1)?[0].index_map.num_local(), 5);
    /// assert_eq!(copy.permutation_info(0).unwrap().to_host()?, [0b000, 0b111]);
    /// # Ok::<(), tetraloma_device::GridError>(())
    /// ```
    ///
    /// [Meshes and topology]: https://tetracl.github.io/tetraloma/concepts/meshes.html
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let clone_opt =
            |b: &Option<DeviceBuffer<u32>>| b.as_ref().map(DeviceBuffer::deep_clone).transpose();
        Ok(Topology {
            client: self.client.clone(),
            cell_blocks: self
                .cell_blocks
                .iter()
                .map(|b| b.try_clone(comm))
                .collect::<Result<_>>()?,
            vertex_index_map: self.vertex_index_map.clone(),
            vertex_origin_ids: self.vertex_origin_ids.deep_clone()?,
            ids_fit_u32: self.ids_fit_u32,
            permutation_info: self
                .permutation_info
                .iter()
                .map(clone_opt)
                .collect::<Result<_>>()?,
            entities: self
                .entities
                .iter()
                .map(|e| {
                    e.as_ref()
                        .map(|blocks| {
                            blocks
                                .iter()
                                .map(|b| b.try_clone(comm))
                                .collect::<Result<Vec<_>>>()
                        })
                        .transpose()
                })
                .collect::<Result<_>>()?,
            connectivity: self
                .connectivity
                .iter()
                .map(|(k, c)| Ok((*k, c.try_clone()?)))
                .collect::<Result<_>>()?,
            vertex_sharers: self.vertex_sharers.clone(),
            vertex_shared: clone_opt(&self.vertex_shared)?,
            sharing_ranks: self.sharing_ranks.clone(),
        })
    }
}

/// The length checks of [`Topology::new`] on the cell blocks.
fn check_cell_blocks<TD: TopoDim>(cell_blocks: &[CellBlock<TD>]) -> Result<()> {
    for block in cell_blocks {
        let cell: CellType = block.cell.into();
        let nv = tetraloma_refcell::vertices(cell).len();
        let rows = block.index_map.num_local();
        let entries = rows.checked_mul(nv).ok_or(GridError::TooLarge)?;
        if block.vertices.stride as usize != nv || block.vertices.data.len() != entries {
            return Err(GridError::LengthMismatch {
                expected: entries,
                found: block.vertices.data.len(),
            });
        }
        // Interleaved (hi, lo) words: two per cell (crate::origin).
        let words = rows.checked_mul(2).ok_or(GridError::TooLarge)?;
        if block.origin_ids.len() != words {
            return Err(GridError::LengthMismatch {
                expected: words,
                found: block.origin_ids.len(),
            });
        }
        if block.flags.len() != rows {
            return Err(GridError::LengthMismatch {
                expected: rows,
                found: block.flags.len(),
            });
        }
    }
    Ok(())
}

/// The checks of [`Topology::set_vertex_sharers`] on `s` for `n` local vertices.
fn check_sharers(s: &VertexSharers, n: usize) -> Result<()> {
    let mismatch = |expected, found| GridError::LengthMismatch { expected, found };
    if s.offsets.len() != n + 1 {
        return Err(mismatch(n + 1, s.offsets.len()));
    }
    if s.offsets[0] != 0 {
        return Err(mismatch(0, s.offsets[0]));
    }
    if let Some(w) = s.offsets.windows(2).find(|w| w[1] < w[0]) {
        return Err(mismatch(w[0], w[1]));
    }
    if s.ranks.len() != s.offsets[n] {
        return Err(mismatch(s.offsets[n], s.ranks.len()));
    }
    if s.owned_cell.len() != s.ranks.len() {
        return Err(mismatch(s.ranks.len(), s.owned_cell.len()));
    }
    let bad = |v| GridError::IndexOutOfRange { index: v, len: n };
    let rows = || s.offsets.windows(2).map(|w| &s.ranks[w[0]..w[1]]);
    for (v, row) in rows().enumerate() {
        if row.first().is_some_and(|&r| r < 0) || !row.windows(2).all(|w| w[0] < w[1]) {
            return Err(bad(v));
        }
    }
    // This rank is in every row: the candidates are the first row's ranks.
    let Some(first) = rows().next() else {
        return Ok(());
    };
    let mut candidates = first.to_vec();
    for (v, row) in rows().enumerate() {
        candidates.retain(|r| row.binary_search(r).is_ok());
        if candidates.is_empty() {
            return Err(bad(v));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tetraloma_dims::{Cell2, D2};
    use tetraloma_parallel::Serial;

    /// Two triangles sharing the edge (1, 2).
    fn two_triangles() -> Topology<D2> {
        let client = tetraloma_testing::test_client();
        let block = CellBlock::<D2>::new(
            Cell2::Triangle,
            IndexMap::new(&Serial, 2).unwrap(),
            DeviceStrided {
                data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(),
                stride: 3,
            },
            DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1])).unwrap(),
            DeviceBuffer::zeros(&client, 2).unwrap(),
        );
        let map = IndexMap::new(&Serial, 4).unwrap();
        Topology::new(&client, vec![block], map, &[10, 11, 12, 13]).unwrap()
    }

    /// Rows with more than one rank are shared; the neighbourhood is the union of their ranks;
    /// `try_clone` keeps both.
    #[test]
    fn sharers_give_flags_and_neighbourhood() {
        let mut t = two_triangles();
        assert!(t.vertex_shared.is_none() && t.sharing_ranks.is_empty());
        let s = VertexSharers {
            offsets: vec![0, 1, 3, 6, 7],
            ranks: vec![1, 1, 4, 0, 1, 2, 1],
            owned_cell: vec![true, true, false, true, true, false, true],
        };
        t.set_vertex_sharers(s.clone()).unwrap();
        let flags = t.vertex_shared.as_ref().unwrap().to_host().unwrap();
        assert_eq!(flags, [0, 1, 1, 0]);
        assert_eq!(t.sharing_ranks, [0, 1, 2, 4]);
        assert_eq!(t.vertex_sharers, s);
        let c = t.try_clone(&Serial).unwrap();
        assert_eq!(c.vertex_shared.unwrap().to_host().unwrap(), flags);
        assert_eq!((c.sharing_ranks, c.vertex_sharers), (t.sharing_ranks, s));
    }
}
