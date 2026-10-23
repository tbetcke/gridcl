//! Entity creation, steps 3–5: sort, unique and canonical vertex lists; `EntityBlock` and
//! `Connectivity` (Listing T1).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceCsr, DeviceStrided, GridError, KernelKind, LaunchPlan, Result,
    debug_assert_bindings, gather, radix_sort, scatter, unique_by_key,
};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, IndexMap};

use crate::emission::{EmissionPlan, emission_plans, sub_entity_vertices};
use crate::keys::emit_keys;

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
        Ok(EntityBlock {
            cell: self.cell,
            index_map: crate::topology::clone_map(&self.index_map, comm)?,
            vertices: crate::topology::clone_strided(&self.vertices)?,
            offset: self.offset,
        })
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
        Ok(match self {
            Connectivity::Downward(blocks) => Connectivity::Downward(
                blocks
                    .iter()
                    .map(crate::topology::clone_strided)
                    .collect::<Result<_>>()?,
            ),
            Connectivity::Upward(csr) => Connectivity::Upward(DeviceCsr {
                offsets: csr.offsets.deep_clone()?,
                indices: csr.indices.deep_clone()?,
            }),
        })
    }
}

/// The comptime tables of the head-position kernel for one cell block: the cell's vertex count,
/// the local vertex lists of its sub-entities of the entity type, the entity's vertex count, and
/// the id fast path.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct HeadTables {
    nv: usize,
    entities: Vec<Vec<usize>>,
    words: usize,
    fast: bool,
}

/// Whether vertex `a` has a lower origin id than vertex `b` ((hi, lo) pairs, or lo words only
/// on the fast path).
#[cube]
fn id_less(ids: &[u32], a: u32, b: u32, #[comptime] fast: bool) -> bool {
    let (a, b) = (2 * a as usize, 2 * b as usize);
    let lo = ids[a + 1] < ids[b + 1];
    let mut r = lo;
    if comptime!(!fast) {
        r = ids[a] < ids[b] || (ids[a] == ids[b] && lo);
    }
    r
}

/// Entity → vertex in the A.6 storage slots, from the head occurrence of each entity: group
/// g < count whose first emission index lies in this block's range [start, end) reads its
/// sub-entity from the cell's vertices and writes row `old2new[g]` if it is below `count`.
/// Vertex indices are clamped to `last_vertex` as they are read (as in the TopologyKeys
/// kernel), so the origin-id reads stay inside `ids`. Scalar: one entity per item.
///   first     emission index of each group's first key
///   vertices  [cell][nv] local vertex indices of the block
///   ids       interleaved (hi, lo) vertex origin ids
///   old2new   the renumbering of the groups
///   out       [entity][words]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn head_kernel(
    first: &[u32],
    vertices: &[u32],
    ids: &[u32],
    old2new: &[u32],
    out: &mut [u32],
    count: u32,
    last_vertex: u32,
    start: u32,
    end: u32,
    padded: u32,
    items_per_unit: u32,
    #[comptime] t: HeadTables,
) {
    let words = t.words;
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let g = begin + it;
        if g < count as usize {
            let e = first[g];
            if e >= start && e < end {
                let r = e - start;
                let sub = r / padded;
                let c = (r % padded) as usize;
                let mut v = Array::<u32>::new(comptime!(t.words));
                #[unroll]
                for k in 0..comptime!(t.entities.len()) {
                    if sub == comptime!(k as u32) {
                        #[unroll]
                        for j in 0..words {
                            v[j] = min(
                                vertices[c * t.nv + comptime!(t.entities[k][j])],
                                last_vertex,
                            );
                        }
                    }
                }
                if comptime!(t.words == 4) {
                    // Quadrilateral in tensor slots: the lowest vertex, its lower neighbour on
                    // the cycle 0-1-3-2 (neighbours s ^ 1 and s ^ 2), the other one, the
                    // opposite vertex s ^ 3.
                    let mut s0 = 0u32;
                    #[unroll]
                    for j in 1..4u32 {
                        if id_less(ids, v[j as usize], v[s0 as usize], t.fast) {
                            s0 = j;
                        }
                    }
                    let a = s0 ^ 1u32;
                    let b = s0 ^ 2u32;
                    let mut lower = b;
                    let mut other = a;
                    if id_less(ids, v[a as usize], v[b as usize], t.fast) {
                        lower = a;
                        other = b;
                    }
                    let (w0, w1, w2, w3) = (
                        v[s0 as usize],
                        v[lower as usize],
                        v[other as usize],
                        v[(s0 ^ 3u32) as usize],
                    );
                    v[0] = w0;
                    v[1] = w1;
                    v[2] = w2;
                    v[3] = w3;
                } else {
                    // Edges and triangles: ascending ids, by a sorting network.
                    #[unroll]
                    for pass in 0..comptime!(t.words - 1) {
                        #[unroll]
                        for j in 0..comptime!(t.words - 1 - pass) {
                            let (x, y) = (v[j], v[j + 1]);
                            if id_less(ids, y, x, t.fast) {
                                v[j] = y;
                                v[j + 1] = x;
                            }
                        }
                    }
                }
                let new = old2new[g];
                if new < count {
                    let row = new as usize * words;
                    #[unroll]
                    for j in 0..words {
                        out[row + j] = v[j];
                    }
                }
            }
        }
    }
}

/// The comptime table of the downward kernel: for each local sub-entity of the dimension, the
/// slot of its entity type (0 or 1) and its index among the cell's sub-entities of that type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DownTables {
    slots: Vec<(usize, usize)>,
}

/// cell → entity of one block: `out[c · K + i] = offset_s + flat_s[start_s + j · padded + c]`
/// for local sub-entity i of type slot s, its index j within that type. Scalar: one cell per
/// item.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn downward_kernel(
    flat0: &[u32],
    flat1: &[u32],
    out: &mut [u32],
    num_cells: u32,
    padded: u32,
    start0: u32,
    start1: u32,
    offset0: u32,
    offset1: u32,
    items_per_unit: u32,
    #[comptime] t: DownTables,
) {
    let k = comptime!(t.slots.len());
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = begin + it;
        if c < num_cells as usize {
            #[unroll]
            for i in 0..k {
                let j = comptime!(t.slots[i].1 as u32);
                if comptime!(t.slots[i].0 == 0) {
                    out[c * k + i] = offset0 + flat0[(start0 + j * padded) as usize + c];
                } else {
                    out[c * k + i] = offset1 + flat1[(start1 + j * padded) as usize + c];
                }
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The entities of one type: the entity block and, per emission index, the entity of the
/// emitted sub-entity (padding positions hold arbitrary values).
struct TypeEntities {
    block: EntityBlock,
    flat: DeviceBuffer<u32>,
}

/// What every entity type of one dimension shares.
struct DimContext<'a> {
    client: &'a Client,
    comm: &'a dyn CommDyn,
    cells: &'a [(CellType, &'a DeviceStrided<u32>)],
    ids: &'a DeviceBuffer<u32>,
    fast: bool,
    dim: usize,
    /// The largest vertex index the kernels use: ids.len() / 2 − 1, or 0 when there are no
    /// vertices (then the head kernel reads the zeroed padding words 0 and 1 of `ids`). Below
    /// u32::MAX, since the vertex count must fit in u32, so no clamped key is the all-ones
    /// sentinel.
    last_vertex: u32,
    /// The first flat cell index of each cell block.
    cell_offsets: &'a [usize],
}

/// Steps 2–5 and 7 for one entity type of `plan`, with the block offset `offset`.
fn create_type(cx: &DimContext<'_>, plan: &EmissionPlan, offset: usize) -> Result<TypeEntities> {
    let DimContext {
        client,
        comm,
        cells,
        ids,
        fast,
        dim,
        ..
    } = *cx;
    let words = gridcl_refcell::vertices(plan.entity).len();
    let empty = |count: usize| -> Result<TypeEntities> {
        Ok(TypeEntities {
            block: EntityBlock {
                cell: plan.entity,
                index_map: IndexMap::new(comm, count)?,
                vertices: DeviceStrided {
                    data: DeviceBuffer::uninit(client, 0)?,
                    stride: fit(words)?,
                },
                offset: fit(offset)?,
            },
            flat: DeviceBuffer::uninit(client, plan.total)?,
        })
    };
    if plan.total == 0 {
        return empty(0);
    }
    let (keys, payload) = emit_keys(client, cells, dim, plan, cx.last_vertex)?;
    let (sorted, payload) = radix_sort(&keys, &payload)?;
    let unique = unique_by_key(&sorted, &payload)?; // the count read
    let runs = unique.keys.len();
    let count = runs - usize::from(plan.has_sentinels());
    let old2new =
        crate::renumber::renumber(client, &sorted, &payload, plan, cx.cell_offsets, runs)?;

    // Entity → vertex in canonical orientation, one launch per block.
    let vertices_len = count.checked_mul(words).ok_or(GridError::TooLarge)?;
    fit(vertices_len)?;
    let vertices = DeviceBuffer::<u32>::uninit(client, vertices_len)?;
    for range in plan.ranges.iter().filter(|r| r.cells > 0 && count > 0) {
        let (cell, rows) = cells[range.block];
        let tables = HeadTables {
            nv: gridcl_refcell::vertices(cell).len(),
            entities: sub_entity_vertices(cell, dim, &range.local),
            words,
            fast,
        };
        let launch = LaunchPlan::elementwise(client, KernelKind::Gather, count, 4, 1, 1)?;
        debug_assert_bindings(client, 5);
        // SAFETY: groups g < count read first[g] (count ≤ runs entries); a head in this block's
        // range [start, end) decodes to cell c < padded with sub-entity index < local.len().
        // The keys hold vertex indices clamped to last_vertex < u32::MAX, so no real key equals
        // the all-ones sentinel: the sentinel group holds exactly the padding positions, and
        // the head of every other group has c < cells, so the kernel reads `rows` at
        // c · nv + j within its cells · nv entries. It reads `ids` at 2 · v and 2 · v + 1 with
        // v = min(vertex, last_vertex): at most ids.len() − 1 when there are vertices, and
        // words 0 and 1 of the allocation of at least VECTOR_PAD words when there are none.
        // It reads old2new[g] for g < count ≤ runs, and writes row old2new[g] only when it is
        // below count, within `vertices`' count · words entries.
        unsafe {
            head_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                unique.first.arg(),
                rows.data.arg(),
                ids.arg(),
                old2new.arg(),
                vertices.arg(),
                fit(count)?,
                cx.last_vertex,
                fit(range.start)?,
                fit(range.end())?,
                fit(range.padded)?,
                launch.items_per_unit,
                tables,
            );
        }
    }

    // Provisional cell → entity by emission index, then renumbered.
    let mut flat = DeviceBuffer::<u32>::uninit(client, plan.total)?;
    scatter(&unique.group, &payload, &mut flat)?;
    let flat = gather(&old2new, &flat)?;
    Ok(TypeEntities {
        block: EntityBlock {
            cell: plan.entity,
            index_map: IndexMap::new(comm, count)?,
            vertices: DeviceStrided {
                data: vertices,
                stride: fit(words)?,
            },
            offset: fit(offset)?,
        },
        flat,
    })
}

/// The entity blocks of dimension `dim` ≥ 1 and the downward connectivity (tdim, dim) of the
/// cell blocks `cells` (cell type and vertex rows), over vertices with origin ids `ids`.
pub(crate) fn create_dimension(
    client: &Client,
    comm: &dyn CommDyn,
    cells: &[(CellType, &DeviceStrided<u32>)],
    ids: &DeviceBuffer<u32>,
    fast: bool,
    dim: usize,
) -> Result<(Vec<EntityBlock>, Vec<DeviceStrided<u32>>)> {
    let counts: Vec<(CellType, usize)> = cells
        .iter()
        .map(|(c, rows)| (*c, rows.data.len() / rows.stride.max(1) as usize))
        .collect();
    let plans = emission_plans(&counts, dim)?;
    let mut types = Vec::with_capacity(plans.len());
    let cell_offsets: Vec<usize> = counts
        .iter()
        .scan(0usize, |acc, &(_, n)| {
            let first = *acc;
            *acc += n;
            Some(first)
        })
        .collect();
    let cx = DimContext {
        client,
        comm,
        cells,
        ids,
        fast,
        dim,
        last_vertex: fit(ids.len() / 2)?.saturating_sub(1),
        cell_offsets: &cell_offsets,
    };
    let mut offset = 0usize;
    for plan in &plans {
        let t = create_type(&cx, plan, offset)?;
        offset = offset
            .checked_add(t.block.index_map.num_local())
            .ok_or(GridError::TooLarge)?;
        types.push(t);
    }
    fit(offset)?;

    // cell → entity of each block, with flat indices.
    let mut downward = Vec::with_capacity(cells.len());
    for (b, &(cell, _)) in cells.iter().enumerate() {
        let local_types = gridcl_refcell::sub_entity_types(cell, dim);
        let k = local_types.len();
        let n = counts[b].1;
        // The plans of the block's entity types, in plan order: at most two (faces of prisms
        // and pyramids).
        let used: Vec<usize> = (0..plans.len())
            .filter(|&p| plans[p].ranges.iter().any(|r| r.block == b))
            .collect();
        assert!(
            used.len() <= 2,
            "a cell has at most two sub-entity types per dimension"
        );
        let slots: Vec<(usize, usize)> = local_types
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let s = used
                    .iter()
                    .position(|&p| plans[p].entity == *ty)
                    .expect("planned type");
                let j = local_types[..i].iter().filter(|t| *t == ty).count();
                (s, j)
            })
            .collect();
        let out_len = n.checked_mul(k).ok_or(GridError::TooLarge)?;
        fit(out_len)?;
        let out = DeviceBuffer::<u32>::uninit(client, out_len)?;
        if n > 0 {
            let range = |s: usize| {
                plans[used[s]]
                    .ranges
                    .iter()
                    .find(|r| r.block == b)
                    .expect("block range")
            };
            let p0 = used[0];
            let dummy;
            let flat1 = match used.get(1) {
                Some(&p1) => &types[p1].flat,
                None => {
                    dummy = DeviceBuffer::<u32>::uninit(client, 0)?;
                    &dummy
                }
            };
            let offset1 = used.get(1).map_or(0, |&p1| types[p1].block.offset);
            let launch = LaunchPlan::elementwise(client, KernelKind::Gather, n, 4, 1, 1)?;
            debug_assert_bindings(client, 3);
            // SAFETY: cells c < n write out[c · k + i] < n · k (checked to fit u32 above), the
            // length of `out`; each read flat_s[start + j · padded + c] lies in the block's
            // emission range of type s (j < its local count, c < n ≤ padded), within the plan's
            // total, the length of flat_s. A block with one entity type never reads `flat1`
            // (comptime slots).
            unsafe {
                downward_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    types[p0].flat.arg(),
                    flat1.arg(),
                    out.arg(),
                    fit(n)?,
                    fit(range(0).padded)?,
                    fit(range(0).start)?,
                    fit(range(used.len() - 1).start)?,
                    types[p0].block.offset,
                    offset1,
                    launch.items_per_unit,
                    DownTables { slots },
                );
            }
        }
        downward.push(DeviceStrided {
            data: out,
            stride: fit(k)?,
        });
    }
    Ok((types.into_iter().map(|t| t.block).collect(), downward))
}
