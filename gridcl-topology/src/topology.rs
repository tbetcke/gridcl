//! `Topology<TD>` (ADR-0012): cell blocks, vertex `IndexMap` and origin ids, entity blocks,
//! connectivity and permutation info.

use std::collections::HashMap;

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, DeviceStrided, GridError, Result};
use gridcl_dims::{CellType, TopoDim};
use gridcl_parallel::{CommDyn, IndexMap};

use crate::{CellBlock, Connectivity, EntityBlock, encode_origin_ids, origin_ids_fit_u32};

/// A ghost-free index map rebuilt on `comm` with the same owned count (collective).
pub(crate) fn clone_map(map: &IndexMap, comm: &dyn CommDyn) -> Result<IndexMap> {
    assert_eq!(
        map.num_ghosts(),
        0,
        "try_clone: index maps with ghosts arrive in M6"
    );
    IndexMap::new(comm, map.num_owned())
}

/// A deep copy of a strided array.
pub(crate) fn clone_strided(s: &DeviceStrided<u32>) -> Result<DeviceStrided<u32>> {
    Ok(DeviceStrided {
        data: s.data.deep_clone()?,
        stride: s.stride,
    })
}

/// The topology of a mesh (ADR-0012).
///
/// Entities of dimension 0 ≤ d < tdim are created on demand; cells stay in
/// [`cell_blocks`](Topology::cell_blocks). Stored connectivities are (tdim, d), downward and
/// filled by entity creation, for 1 ≤ d < tdim, and (d, tdim), upward, for 0 ≤ d < tdim;
/// (tdim, 0) is `CellBlock::vertices` and d → 0 is `EntityBlock::vertices`.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceStrided};
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_parallel::{IndexMap, Serial};
/// use gridcl_topology::{CellBlock, Topology, encode_origin_ids};
///
/// let client = gridcl_testing::test_client();
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
    /// The vertex indices themselves stay on the device and are not checked. Out-of-range
    /// indices are not an error but give a meaningless topology: every kernel that reads
    /// through a vertex index clamps an index past the local vertices to the last vertex, so
    /// no read leaves its buffer, and entities, keys and permutation info are computed as if
    /// the index were the last vertex. The (tdim, 0) connectivity,
    /// [`CellBlock::vertices`](crate::CellBlock), is the input unchanged and keeps the raw
    /// indices.
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
        for block in &cell_blocks {
            let cell: CellType = block.cell.into();
            let nv = gridcl_refcell::vertices(cell).len();
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
        let ids_fit_u32 = origin_ids_fit_u32(vertex_origin_ids);
        let ids = DeviceBuffer::from_host(client, &encode_origin_ids(vertex_origin_ids))?;
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
        })
    }

    /// The client the arrays live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The cell blocks, in creation order.
    pub fn cell_blocks(&self) -> &[CellBlock<TD>] {
        &self.cell_blocks
    }

    /// The [owned | ghost] vertices.
    pub fn vertex_index_map(&self) -> &IndexMap {
        &self.vertex_index_map
    }

    /// The interleaved (hi ^ 0x8000_0000, lo) origin id of each local vertex.
    pub fn vertex_origin_ids(&self) -> &DeviceBuffer<u32> {
        &self.vertex_origin_ids
    }

    /// Whether every vertex origin id lies in [0, 2^32), the comptime fast path of the kernels.
    pub fn ids_fit_u32(&self) -> bool {
        self.ids_fit_u32
    }

    /// The orientation word (A.6) of each cell of block `b`; `None` for interval cells.
    ///
    /// # Panics
    ///
    /// If `b` is not a block index.
    pub fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>> {
        self.permutation_info[b].as_ref()
    }

    /// The entity blocks of dimension `dim`, one per entity type.
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` until `create_entities(dim)` has run, and always for
    /// `dim` ≥ tdim.
    pub fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]> {
        self.entities
            .get(dim)
            .and_then(Option::as_deref)
            .ok_or(GridError::EntitiesNotCreated(dim))
    }

    /// Whether gridcl stores the connectivity `key`.
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
    pub fn connectivity(&self, key: ConnKey) -> Result<&Connectivity> {
        if !Self::supported(key) {
            return Err(GridError::UnsupportedConnectivity(key));
        }
        self.connectivity
            .get(&key)
            .ok_or(GridError::ConnectivityNotCreated(key))
    }

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and, for dim ≥ 1, the downward
    /// connectivity (tdim, dim). Does nothing if they exist.
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim; `TooLarge` if the padded emission exceeds
    /// 2^32 keys; device errors.
    pub fn create_entities(&mut self, comm: &dyn CommDyn, dim: usize) -> Result<()> {
        if dim >= TD::N {
            return Err(GridError::EntitiesNotCreated(dim));
        }
        if self.entities[dim].is_some() {
            return Ok(());
        }
        if dim == 0 {
            // One Point block of the vertices.
            let n = self.vertex_index_map.num_local();
            let block = EntityBlock {
                cell: CellType::Point,
                index_map: IndexMap::new(comm, n)?,
                vertices: DeviceStrided {
                    data: crate::renumber::iota(&self.client, n)?,
                    stride: 1,
                },
                offset: 0,
            };
            self.entities[0] = Some(vec![block]);
            return Ok(());
        }
        let cells: Vec<(CellType, &DeviceStrided<u32>)> = self
            .cell_blocks
            .iter()
            .map(|b| (b.cell.into(), &b.vertices))
            .collect();
        let (blocks, downward) = crate::entity::create_dimension(
            &self.client,
            comm,
            &cells,
            &self.vertex_origin_ids,
            self.ids_fit_u32,
            dim,
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

    /// A deep copy of every device array; index maps are rebuilt on `comm` (collective).
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone` and `IndexMap::new`.
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
            vertex_index_map: clone_map(&self.vertex_index_map, comm)?,
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
        })
    }
}
