//! `Topology<TD>` (ADR-0012): cell blocks, vertex `IndexMap` and origin ids, entity blocks,
//! connectivity and permutation info.

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, Result};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, IndexMap};

use crate::{CellBlock, Connectivity, EntityBlock};

/// The topology of a mesh (ADR-0012).
///
/// Entities of dimension 0 ≤ d < tdim are created on demand; cells stay in
/// [`cell_blocks`](Topology::cell_blocks). Stored connectivities are (tdim, d), downward and
/// filled by entity creation, and (d, tdim), upward, for 0 ≤ d < tdim, except (tdim, 0), which
/// is `CellBlock::vertices`; d → 0 is `EntityBlock::vertices`.
pub struct Topology<TD: TopoDim> {
    cell_blocks: Vec<CellBlock<TD>>,
    _todo: core::marker::PhantomData<TD>,
}

impl<TD: TopoDim> Topology<TD> {
    /// The topology of `cell_blocks`, whose vertex rows index the `vertex_index_map.num_local()`
    /// vertices with origin ids `vertex_origin_ids`. Uploads the encoded vertex origin ids and
    /// computes the permutation info of every block of dimension ≥ 2.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `vertex_origin_ids` does not have one id per local vertex; device
    /// errors.
    pub fn new(
        client: &Client,
        cell_blocks: Vec<CellBlock<TD>>,
        vertex_index_map: IndexMap,
        vertex_origin_ids: &[i64],
    ) -> Result<Self> {
        let _ = (client, cell_blocks, vertex_index_map, vertex_origin_ids);
        todo!()
    }

    /// The client the arrays live on.
    pub fn client(&self) -> &Client {
        todo!()
    }

    /// The cell blocks, in creation order.
    pub fn cell_blocks(&self) -> &[CellBlock<TD>] {
        &self.cell_blocks
    }

    /// The [owned | ghost] vertices.
    pub fn vertex_index_map(&self) -> &IndexMap {
        todo!()
    }

    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each local vertex.
    pub fn vertex_origin_ids(&self) -> &DeviceBuffer<u32> {
        todo!()
    }

    /// Whether every vertex origin id lies in [0, 2^32), the comptime fast path of the kernels.
    pub fn ids_fit_u32(&self) -> bool {
        todo!()
    }

    /// The orientation word (A.6) of each cell of block `b`; `None` for interval cells.
    pub fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>> {
        let _ = b;
        todo!()
    }

    /// The entity blocks of dimension `dim`, one per entity type.
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` until `create_entities(dim)` has run, and always for
    /// `dim` = tdim.
    pub fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]> {
        let _ = dim;
        todo!()
    }

    /// The connectivity `key`.
    ///
    /// # Errors
    ///
    /// `ConnectivityNotCreated(key)` until it has been created; `UnsupportedConnectivity(key)`
    /// for pairs other than (tdim, d) with 1 ≤ d < tdim and (d, tdim) with 0 ≤ d < tdim.
    pub fn connectivity(&self, key: ConnKey) -> Result<&Connectivity> {
        let _ = key;
        todo!()
    }

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and, for dim ≥ 1, the downward
    /// connectivity (tdim, dim). Does nothing if they exist.
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim; `TooLarge` if the padded emission exceeds
    /// 2^32 keys; device errors.
    pub fn create_entities(&mut self, comm: &dyn CommDyn, dim: usize) -> Result<()> {
        let _ = (comm, dim);
        todo!()
    }

    /// Creates the connectivity `from` → `to`, creating the entities it needs. Does nothing if
    /// it exists.
    ///
    /// # Errors
    ///
    /// `UnsupportedConnectivity` for pairs that are not stored; errors of
    /// [`create_entities`](Topology::create_entities).
    pub fn create_connectivity(
        &mut self,
        comm: &dyn CommDyn,
        from: usize,
        to: usize,
    ) -> Result<()> {
        let _ = (comm, from, to);
        todo!()
    }

    /// A deep copy of every device array; index maps are rebuilt on `comm` (collective).
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone` and `IndexMap::new`.
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let _ = comm;
        todo!()
    }
}
