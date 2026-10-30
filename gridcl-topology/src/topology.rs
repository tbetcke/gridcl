//! `Topology<TD>` (ADR-0012): cell blocks, vertex `IndexMap` and origin ids, entity blocks,
//! connectivity and permutation info.

use std::collections::HashMap;

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, DeviceStrided, GridError, Result};
use gridcl_dims::{CellType, TopoDim};
use gridcl_parallel::{CommDyn, IndexMap};

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
            vertex_sharers: VertexSharers::default(),
            vertex_shared: None,
            sharing_ranks: Vec::new(),
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

    /// Stores the sharer sets of the local vertices, uploads one shared flag per vertex and
    /// derives the sharing neighbourhood: every rank that shares a vertex with this one
    /// (ADR-0026). Without sharers every vertex is unshared. Pipeline stage 9 calls it at mesh
    /// creation.
    ///
    /// # Panics
    ///
    /// If `s` does not have one row per local vertex, or `ranks` and `owned_cell` do not
    /// match the offsets; if the upload of the flags fails.
    pub fn set_vertex_sharers(&mut self, s: VertexSharers) {
        let n = self.vertex_index_map.num_local();
        assert!(
            s.offsets.len() == n + 1
                && s.offsets[0] == 0
                && s.offsets.is_sorted()
                && s.ranks.len() == s.offsets[n]
                && s.owned_cell.len() == s.ranks.len(),
            "VertexSharers must have one row per local vertex ({n})"
        );
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
        // The vertex shared flags at mesh creation (CLAUDE.md, ADR-0026).
        let flags = DeviceBuffer::from_host(&self.client, &flags)
            .expect("the upload of the vertex shared flags");
        self.vertex_sharers = s;
        self.vertex_shared = Some(flags);
        self.sharing_ranks = ranks;
    }

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and, for dim ≥ 1, the downward
    /// connectivity (tdim, dim). Does nothing if they exist.
    ///
    /// Collective on more than one rank: every rank of `comm` calls it with the same `dim`.
    /// Entities whose vertices are all shared ([`set_vertex_sharers`](Topology::set_vertex_sharers))
    /// get their owners by an exchange over the sharing neighbourhood (docs/design.md,
    /// "Ownership of vertices and entities"), and errors are agreed (ADR-0026, item 15).
    /// Each entity block is numbered [owned | ghost]: owned entities by their smallest touching
    /// cell, ghosts by (owner rank, global index), with global numbers from an exclusive scan
    /// of the owned counts and, for ghosts, from their owners. The dimension-0 `Point` block
    /// shares the vertex index map.
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
    pub fn create_entities(&mut self, comm: &dyn CommDyn, dim: usize) -> Result<()> {
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
        let crate::entity::Dimension { blocks, downward } =
            crate::entity::create_dimension(&self.client, comm, &self.entity_cells(), dim)?;
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

    /// A deep copy of every device array. Index maps, ghosts included, are cloned and share
    /// their neighbourhoods with the original, so the copy is not collective and `comm` is
    /// unused (it stays in the signature, ADR-0012).
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_dims::{Cell2, D2};
    use gridcl_parallel::Serial;

    /// Two triangles sharing the edge (1, 2).
    fn two_triangles() -> Topology<D2> {
        let client = gridcl_testing::test_client();
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
        t.set_vertex_sharers(s.clone());
        let flags = t.vertex_shared.as_ref().unwrap().to_host().unwrap();
        assert_eq!(flags, [0, 1, 1, 0]);
        assert_eq!(t.sharing_ranks, [0, 1, 2, 4]);
        assert_eq!(t.vertex_sharers, s);
        let c = t.try_clone(&Serial).unwrap();
        assert_eq!(c.vertex_shared.unwrap().to_host().unwrap(), flags);
        assert_eq!((c.sharing_ranks, c.vertex_sharers), (t.sharing_ranks, s));
    }

    #[test]
    #[should_panic(expected = "one row per local vertex")]
    fn sharers_need_one_row_per_vertex() {
        two_triangles().set_vertex_sharers(VertexSharers {
            offsets: vec![0, 1],
            ranks: vec![0],
            owned_cell: vec![true],
        });
    }
}
