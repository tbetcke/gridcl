//! Entity creation, steps 3–8: sort, unique, ownership (`crate::ownership`), renumbering,
//! canonical vertex lists and global numbers; `EntityBlock` and `Connectivity` (Listing T1).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceCsr, DeviceStrided, GridError, KernelKind, LaunchPlan, Result, Unique,
    debug_assert_bindings, gather, radix_sort, scatter, unique_by_key,
};
use gridcl_dims::CellType;
use gridcl_parallel::{CommDyn, CommExt, IndexMap};

use crate::emission::{EmissionPlan, emission_plans, sub_entity_vertices};
use crate::keys::emit_keys;
use crate::ownership::{Candidates, GHOST_CELL, Ownership, Sharing, SharingExchange, agree};

/// The entities of one type and dimension (Listing T1).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mut mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
/// mesh.create_entities(1)?;
/// let edges = &mesh.entity_blocks(1)?[0];
/// assert_eq!(edges.cell, CellType::Interval);
/// // The square's four sides and its diagonal, two vertices each.
/// assert_eq!(edges.index_map.num_local(), 5);
/// assert_eq!((edges.vertices.stride, edges.vertices.data.len()), (2, 10));
/// assert_eq!(edges.offset, 0);
/// # Ok::<(), GridError>(())
/// ```
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
    /// A deep copy of every device array. The index map, ghosts included, is cloned and
    /// shares its neighbourhoods with the original; `comm` is unused.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// mesh.create_entities(1)?;
    /// let edges = &mesh.entity_blocks(1)?[0];
    /// let copy = edges.try_clone(mesh.comm())?;
    /// assert_eq!(copy.vertices.data.to_host()?, edges.vertices.data.to_host()?);
    /// assert_eq!(copy.index_map.num_local(), 5);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let _ = comm;
        Ok(EntityBlock {
            cell: self.cell,
            index_map: self.index_map.clone(),
            vertices: crate::topology::clone_strided(&self.vertices)?,
            offset: self.offset,
        })
    }
}

/// A stored connectivity (Listing T1).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mut mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
/// mesh.create_connectivity(2, 1)?;
/// let Connectivity::Downward(blocks) = mesh.connectivity(ConnKey { from: 2, to: 1 })? else {
///     unreachable!("cell -> edge is downward")
/// };
/// // Three edges per triangle.
/// assert_eq!((blocks[0].stride, blocks[0].data.len()), (3, 6));
/// mesh.create_connectivity(1, 2)?;
/// let Connectivity::Upward(csr) = mesh.connectivity(ConnKey { from: 1, to: 2 })? else {
///     unreachable!("edge -> cell is upward")
/// };
/// // Five edges; only the diagonal has two cells.
/// assert_eq!((csr.offsets.len(), csr.indices.len()), (6, 6));
/// # Ok::<(), GridError>(())
/// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// mesh.create_connectivity(2, 1)?;
    /// let copy = mesh.connectivity(ConnKey { from: 2, to: 1 })?.try_clone()?;
    /// let Connectivity::Downward(blocks) = copy else { unreachable!("cell -> edge is downward") };
    /// assert_eq!(blocks[0].data.len(), 6);
    /// # Ok::<(), GridError>(())
    /// ```
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

/// The local arrays of one entity type: entity → vertex in canonical orientation and, per
/// emission index, the entity of the emitted sub-entity (padding positions hold arbitrary
/// values).
struct TypeArrays {
    vertices: DeviceBuffer<u32>,
    flat: DeviceBuffer<u32>,
}

/// What every entity type of one dimension shares.
struct DimContext<'a> {
    client: &'a Client,
    cells: &'a [(CellType, &'a DeviceStrided<u32>)],
    /// The owned cell count of each cell block.
    owned: &'a [usize],
    /// The vertex sharing; `None` on one rank or without sharers.
    sharing: Option<Sharing<'a>>,
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

/// One entity type after steps 2–5 on the device, before ownership.
struct Provisional {
    payload: DeviceBuffer<u32>,
    unique: Unique,
    runs: usize,
    /// The entity count: the runs without the sentinel group.
    count: usize,
    /// The smallest touching-cell value of each run (`touching_cells`).
    mins: DeviceBuffer<u32>,
}

/// Steps 2–5 for one entity type of `plan`, and its sharing candidates; `None` without
/// emission. The keys hold vertex indices clamped to last_vertex < u32::MAX, so no real key
/// equals the all-ones sentinel, and the sentinel group, if any, is the last run.
fn provisional(
    cx: &DimContext<'_>,
    plan: &EmissionPlan,
) -> Result<(Option<Provisional>, Candidates)> {
    let words = gridcl_refcell::vertices(plan.entity).len();
    if plan.total == 0 {
        return Ok((None, Candidates::none(words)));
    }
    let (keys, payload) = emit_keys(cx.client, cx.cells, cx.dim, plan, cx.last_vertex)?;
    let (sorted, payload) = radix_sort(&keys, &payload)?;
    let unique = unique_by_key(&sorted, &payload)?; // the count read
    let runs = unique.keys.len();
    let count = runs - usize::from(plan.has_sentinels());
    let mins = crate::renumber::touching_cells(
        cx.client,
        &sorted,
        &payload,
        plan,
        cx.cell_offsets,
        cx.owned,
    )?;
    let candidates = match cx.sharing {
        Some(s) => crate::ownership::candidates(
            &unique.keys,
            &mins,
            count,
            cx.ids,
            s.flags,
            cx.last_vertex,
        )?,
        None => Candidates::none(words),
    };
    let p = Provisional {
        payload,
        unique,
        runs,
        count,
        mins,
    };
    Ok((Some(p), candidates))
}

/// One entity type after the first pass of step 7.
struct Staged {
    payload: DeviceBuffer<u32>,
    unique: Unique,
    count: usize,
    old2new: DeviceBuffer<u32>,
    num_owned: usize,
    /// The local indices of the candidates this rank owns, in candidate order.
    owned_local: Vec<u32>,
}

/// The first pass of step 7 for one entity type: owned entities first, ghosts after them
/// (`crate::renumber::renumber`), and the local indices of the owned candidates.
fn stage(cx: &DimContext<'_>, p: Provisional, o: &Ownership, me: i32) -> Result<Staged> {
    let Provisional {
        payload,
        unique,
        runs,
        count,
        mins,
    } = p;
    let ghosts = o.ghosts(me);
    let old2new = crate::renumber::renumber(cx.client, mins, runs, &ghosts)?;
    let owned: Vec<u32> = o
        .candidates
        .entities
        .iter()
        .zip(&o.owners)
        .filter(|&(_, &q)| q == me)
        .map(|(&g, _)| g)
        .collect();
    let owned_local = crate::renumber::local_indices(&old2new, &owned)?;
    Ok(Staged {
        payload,
        unique,
        count,
        old2new,
        num_owned: count - ghosts.len(),
        owned_local,
    })
}

/// The ghosts of one entity type ordered by (owner rank, global index): their provisional ids,
/// global numbers and owners, from the owners and global numbers of the candidates.
fn ghost_order(o: &Ownership, globals: &[i64], me: i32) -> (Vec<u32>, Vec<i64>, Vec<i32>) {
    let mut ghosts: Vec<(i32, i64, u32)> = (0..o.candidates.len())
        .filter(|&i| o.owners[i] != me)
        .map(|i| (o.owners[i], globals[i], o.candidates.entities[i]))
        .collect();
    ghosts.sort_unstable();
    let ids = ghosts.iter().map(|g| g.2).collect();
    let globals = ghosts.iter().map(|g| g.1).collect();
    let owners = ghosts.iter().map(|g| g.0).collect();
    (ids, globals, owners)
}

/// Entity → vertex and the renumbered cell → entity by emission index of one entity type of
/// `plan`, after step 7.
fn finish_type(cx: &DimContext<'_>, plan: &EmissionPlan, s: Option<Staged>) -> Result<TypeArrays> {
    let DimContext {
        client,
        cells,
        ids,
        fast,
        dim,
        ..
    } = *cx;
    let words = gridcl_refcell::vertices(plan.entity).len();
    let Some(Staged {
        payload,
        unique,
        count,
        old2new,
        ..
    }) = s
    else {
        return Ok(TypeArrays {
            vertices: DeviceBuffer::uninit(client, 0)?,
            flat: DeviceBuffer::uninit(client, plan.total)?,
        });
    };

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
        let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, count, 4, 1, 1)?;
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
    Ok(TypeArrays { vertices, flat })
}

/// The cell type and cell count of each cell block.
pub(crate) fn cell_counts(cells: &[(CellType, &DeviceStrided<u32>)]) -> Vec<(CellType, usize)> {
    cells
        .iter()
        .map(|(c, rows)| (*c, rows.data.len() / rows.stride.max(1) as usize))
        .collect()
}

/// The largest vertex index the kernels read for the origin ids `ids`: ids.len() / 2 − 1, or 0
/// when there are no vertices.
pub(crate) fn last_vertex(ids: &DeviceBuffer<u32>) -> Result<u32> {
    Ok(fit(ids.len() / 2)?.saturating_sub(1))
}

/// The cells of a topology that entity creation reads.
pub(crate) struct Cells<'a> {
    /// The cell type and vertex rows of each cell block.
    pub(crate) rows: Vec<(CellType, &'a DeviceStrided<u32>)>,
    /// The owned cell count of each cell block.
    pub(crate) owned: Vec<usize>,
    /// The interleaved vertex origin ids.
    pub(crate) ids: &'a DeviceBuffer<u32>,
    /// Whether every vertex origin id fits u32 (the kernels' fast path).
    pub(crate) fast: bool,
    /// The vertex sharing; `None` without sharers.
    pub(crate) sharing: Option<Sharing<'a>>,
}

/// The entities of one dimension.
pub(crate) struct Dimension {
    /// One entity block per entity type, with [owned | ghost] index maps.
    pub(crate) blocks: Vec<EntityBlock>,
    /// The downward connectivity (tdim, dim), one strided array per cell block.
    pub(crate) downward: Vec<DeviceStrided<u32>>,
}

/// The entity blocks of dimension `dim` ≥ 1 and the downward connectivity (tdim, dim) of
/// `cells`, with owners (step 6) and global numbers (step 8) on more than one rank. Collective
/// on more than one rank, with an agreed status (ADR-0026, item 15).
pub(crate) fn create_dimension(
    client: &Client,
    comm: &dyn CommDyn,
    cells: &Cells<'_>,
    dim: usize,
) -> Result<Dimension> {
    let parallel = comm.size() > 1;
    let counts = cell_counts(&cells.rows);
    let cell_offsets: Vec<usize> = counts
        .iter()
        .scan(0usize, |acc, &(_, n)| {
            let first = *acc;
            *acc += n;
            Some(first)
        })
        .collect();
    let total_cells: usize = counts.iter().map(|&(_, n)| n).sum();
    let plans = agree(
        comm,
        if total_cells >= GHOST_CELL as usize {
            Err(GridError::TooLarge)
        } else {
            emission_plans(&counts, dim)
        },
    )?;
    let cx = DimContext {
        client,
        cells: &cells.rows,
        owned: &cells.owned,
        sharing: if parallel { cells.sharing } else { None },
        ids: cells.ids,
        fast: cells.fast,
        dim,
        last_vertex: last_vertex(cells.ids)?,
        cell_offsets: &cell_offsets,
    };

    // Steps 2–5 and the candidates of every type, then their owners (step 6).
    let prepared = agree(
        comm,
        plans
            .iter()
            .map(|plan| provisional(&cx, plan))
            .collect::<Result<Vec<_>>>(),
    )?;
    let (provisionals, candidates): (Vec<_>, Vec<_>) = prepared.into_iter().unzip();
    let me = comm.rank() as i32;
    let exchange = if parallel {
        Some(SharingExchange::new(comm, cx.sharing)?)
    } else {
        None
    };
    let ownership = match &exchange {
        Some(ex) => crate::ownership::owners(comm, ex, cx.sharing, candidates)?,
        None => candidates
            .iter()
            .map(|c| Ownership::none(c.words))
            .collect(),
    };

    // Step 7, first pass: owned entities numbered, ghosts after them.
    let staged = agree(
        comm,
        provisionals
            .into_iter()
            .zip(&ownership)
            .map(|(p, o)| p.map(|p| stage(&cx, p, o, me)).transpose())
            .collect::<Result<Vec<_>>>(),
    )?;

    // Step 8: owned ranges by an exclusive scan, and the ghosts' global numbers from their
    // owners.
    let num_owned: Vec<usize> = staged
        .iter()
        .map(|s| s.as_ref().map_or(0, |s| s.num_owned))
        .collect();
    let owned_globals: Vec<Vec<i64>> = staged
        .iter()
        .zip(&ownership)
        .zip(&num_owned)
        .map(|((s, o), &n)| {
            let begin = comm.exscan_sum(n as u64) as i64;
            let mut local = s.iter().flat_map(|s| s.owned_local.iter());
            o.owners
                .iter()
                .map(|&q| {
                    if q == me {
                        begin + i64::from(*local.next().expect("one index per owned candidate"))
                    } else {
                        -1
                    }
                })
                .collect()
        })
        .collect();
    let globals = match &exchange {
        Some(ex) => crate::ownership::global_numbers(comm, ex, &ownership, &owned_globals)?,
        None => owned_globals,
    };

    // Step 7, second pass: the ghosts by (owner rank, global index); the local arrays.
    let ghosts: Vec<_> = ownership
        .iter()
        .zip(&globals)
        .map(|(o, g)| ghost_order(o, g, me))
        .collect();
    let arrays = agree(
        comm,
        plans
            .iter()
            .zip(staged)
            .zip(&ghosts)
            .zip(&num_owned)
            .map(|(((plan, mut s), (ids, ..)), &n)| {
                if let Some(s) = &mut s {
                    let new: Vec<u32> =
                        (0..ids.len()).map(|k| fit(n + k)).collect::<Result<_>>()?;
                    crate::renumber::place_ghosts(&mut s.old2new, ids, &new)?;
                }
                finish_type(&cx, plan, s)
            })
            .collect::<Result<Vec<_>>>(),
    )?;

    // The index maps (collective) and the blocks' flat offsets.
    let mut types = Vec::with_capacity(plans.len());
    for ((plan, a), ((_, ghost_globals, ghost_owners), &n)) in plans
        .iter()
        .zip(arrays)
        .zip(ghosts.into_iter().zip(&num_owned))
    {
        let index_map = if parallel {
            IndexMap::with_ghosts(comm, n, ghost_globals, ghost_owners)?
        } else {
            IndexMap::new(comm, n)?
        };
        let words = gridcl_refcell::vertices(plan.entity).len();
        types.push((plan.entity, index_map, a, words));
    }
    let offsets = agree(
        comm,
        (|| {
            let mut offsets = Vec::with_capacity(types.len());
            let mut next = 0usize;
            for (_, map, ..) in &types {
                offsets.push(fit(next)?);
                next = next
                    .checked_add(map.num_local())
                    .ok_or(GridError::TooLarge)?;
            }
            fit(next)?;
            Ok(offsets)
        })(),
    )?;
    let types: Vec<(EntityBlock, DeviceBuffer<u32>)> = types
        .into_iter()
        .zip(offsets)
        .map(|((cell, index_map, a, words), offset)| {
            Ok((
                EntityBlock {
                    cell,
                    index_map,
                    vertices: DeviceStrided {
                        data: a.vertices,
                        stride: fit(words)?,
                    },
                    offset,
                },
                a.flat,
            ))
        })
        .collect::<Result<_>>()?;
    let cells = &cells.rows;

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
                Some(&p1) => &types[p1].1,
                None => {
                    dummy = DeviceBuffer::<u32>::uninit(client, 0)?;
                    &dummy
                }
            };
            let offset1 = used.get(1).map_or(0, |&p1| types[p1].0.offset);
            let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, n, 4, 1, 1)?;
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
                    types[p0].1.arg(),
                    flat1.arg(),
                    out.arg(),
                    fit(n)?,
                    fit(range(0).padded)?,
                    fit(range(0).start)?,
                    fit(range(used.len() - 1).start)?,
                    types[p0].0.offset,
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
    Ok(Dimension {
        blocks: types.into_iter().map(|t| t.0).collect(),
        downward,
    })
}
