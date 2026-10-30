//! Entity creation, step 6, and the exchange of step 8: the owners of the entities whose
//! vertices are all shared, and the global numbers of the ghosts among them (docs/design.md,
//! "Ownership of vertices and entities"; ADR-0026, items 8, 9 and 14).
//!
//! A scalar kernel marks the sharing candidates, which are compacted on the device; their
//! local vertices, origin-id keys and owned-cell flags go to the host with one count read. The
//! keys are sent to every rank in the intersection of their vertices' sharer sets, which reply
//! whether they hold the entity, and in an owned cell, and the owner is the holder at position
//! `owner_hash(key) mod n` among the owned-cell holders. Once the owners have numbered their
//! entities, each sends the global numbers to the other holders (forward direction).

use std::collections::HashMap;

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, ReduceOp, Result, SortKeys,
    debug_assert_bindings, exclusive_scan, reduce,
};
use gridcl_parallel::{CommDyn, CommExt, Neighbourhood, owner_hash};

use crate::VertexSharers;
use crate::origin::decode_origin_ids;

/// The bit of a touching-cell value that marks a ghost cell: an entity whose smallest value is
/// below it lies in an owned cell of this rank.
pub(crate) const GHOST_CELL: u32 = 1 << 31;

/// The vertex sharing of a topology: the shared flag of each local vertex, the sharer sets,
/// and the ranks that share a vertex with this one (this rank included).
#[derive(Clone, Copy)]
pub(crate) struct Sharing<'a> {
    pub(crate) flags: &'a DeviceBuffer<u32>,
    pub(crate) sharers: &'a VertexSharers,
    pub(crate) ranks: &'a [i32],
}

/// The sharing candidates of one entity type, on the host: the entities whose vertices are all
/// shared.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Candidates {
    /// Vertices per entity.
    pub(crate) words: usize,
    /// The provisional id of each candidate, ascending.
    pub(crate) entities: Vec<u32>,
    /// Whether this rank holds the candidate in an owned cell.
    pub(crate) in_owned: Vec<bool>,
    /// The local vertices of each candidate, `words` per candidate, ascending.
    pub(crate) vertices: Vec<u32>,
    /// The key of each candidate: its vertices' origin ids, ascending, `words` per candidate.
    pub(crate) keys: Vec<i64>,
}

impl Candidates {
    /// No candidates of entities with `words` vertices.
    pub(crate) fn none(words: usize) -> Self {
        Candidates {
            words,
            ..Candidates::default()
        }
    }

    /// The number of candidates.
    pub(crate) fn len(&self) -> usize {
        self.entities.len()
    }

    /// The key of candidate `i`.
    pub(crate) fn key(&self, i: usize) -> &[i64] {
        &self.keys[i * self.words..(i + 1) * self.words]
    }
}

/// The owner rank of each candidate of one entity type; every other entity of the type is
/// owned by this rank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Ownership {
    pub(crate) candidates: Candidates,
    pub(crate) owners: Vec<i32>,
    /// The other ranks that hold each candidate, in any cell, ascending.
    pub(crate) holders: Vec<Vec<i32>>,
}

impl Ownership {
    /// No candidates of entities with `words` vertices: every entity is owned here.
    pub(crate) fn none(words: usize) -> Self {
        Ownership {
            candidates: Candidates::none(words),
            ..Ownership::default()
        }
    }

    /// The provisional ids of the candidates owned by another rank, ascending.
    pub(crate) fn ghosts(&self, me: i32) -> Vec<u32> {
        self.candidates
            .entities
            .iter()
            .zip(&self.owners)
            .filter(|&(_, &o)| o != me)
            .map(|(&g, _)| g)
            .collect()
    }
}

/// The sharing neighbourhood of a topology: the ranks, other than this one, that share a
/// vertex with it, ascending, and the neighbourhood over them in both directions.
pub(crate) struct SharingExchange {
    nb: Neighbourhood,
    neighbours: Vec<usize>,
    slot: HashMap<usize, usize>,
}

impl SharingExchange {
    /// The exchange over the sharing ranks of `sharing` (none without sharers). Collective.
    ///
    /// # Errors
    ///
    /// Errors of `Neighbourhood::new`.
    pub(crate) fn new(comm: &dyn CommDyn, sharing: Option<Sharing<'_>>) -> Result<Self> {
        let me = comm.rank() as i32;
        let neighbours: Vec<usize> = sharing
            .map(|s| s.ranks)
            .unwrap_or_default()
            .iter()
            .filter(|&&r| r != me)
            .map(|&r| r as usize)
            .collect();
        let nb = Neighbourhood::new(comm, neighbours.clone(), neighbours.clone())?;
        let slot = neighbours
            .iter()
            .enumerate()
            .map(|(i, &r)| (r, i))
            .collect();
        Ok(SharingExchange {
            nb,
            neighbours,
            slot,
        })
    }
}

/// `flags[g] = 1` if every vertex of provisional entity g < count is shared, else 0. The
/// vertices are the words of the unique key g, clamped to `last_vertex` as in the head kernel.
/// Scalar: one entity per item.
///   keys    planar unique keys: word j of key g at j · plane + g
///   shared  one flag per local vertex
#[cube(launch_unchecked)]
fn candidate_flags_kernel(
    keys: &[u32],
    shared: &[u32],
    flags: &mut [u32],
    count: u32,
    plane: u32,
    last_vertex: u32,
    items_per_unit: u32,
    #[comptime] words: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let g = begin + it;
        if g < count as usize {
            let mut all = 1u32;
            #[unroll]
            for j in 0..words {
                let v = min(keys[j * plane as usize + g], last_vertex);
                all &= shared[v as usize];
            }
            flags[g] = all;
        }
    }
}

/// Row `pos[g]` of `out` for every candidate g < count (`flags[g] = 1`): g, whether the
/// smallest touching-cell value `mins[g]` is an owned cell, the `words` local vertices and
/// their interleaved (hi, lo) origin ids. Scalar: one entity per item.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn candidate_rows_kernel(
    keys: &[u32],
    flags: &[u32],
    pos: &[u32],
    mins: &[u32],
    ids: &[u32],
    out: &mut [u32],
    count: u32,
    plane: u32,
    last_vertex: u32,
    items_per_unit: u32,
    #[comptime] words: usize,
) {
    let row_len = comptime!(2 + 3 * words);
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let g = begin + it;
        if g < count as usize && flags[g] == 1 {
            let row = pos[g] as usize * row_len;
            out[row] = g as u32;
            out[row + 1] = u32::cast_from(mins[g] < 2147483648u32);
            #[unroll]
            for j in 0..words {
                let v = min(keys[j * plane as usize + g], last_vertex);
                out[row + 2 + j] = v;
                out[row + 2 + words + 2 * j] = ids[2 * v as usize];
                out[row + 3 + words + 2 * j] = ids[2 * v as usize + 1];
            }
        }
    }
}

/// The agreed result of a collective step (ADR-0026, item 15): every rank learns the lowest
/// failing rank, which returns its own error, as does every other failing rank; the others
/// return `RemoteFailure { rank }`. Collective. (gridcl-parallel's helper is crate-private.)
pub(crate) fn agree<T>(comm: &dyn CommDyn, local: Result<T>) -> Result<T> {
    let lowest = comm.min(if local.is_err() {
        comm.rank() as u64
    } else {
        u64::MAX
    });
    match local {
        Ok(_) if lowest != u64::MAX => Err(GridError::RemoteFailure {
            rank: lowest as usize,
        }),
        r => r,
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The sharing candidates among the provisional entities g < `count` of one type, with
/// `words` vertices each: `keys` are the unique keys (local vertices, ascending), `mins` the
/// smallest touching-cell value of each entity (`GHOST_CELL` set for a ghost cell), `ids` the
/// interleaved vertex origin ids and `flags` the shared flags of the vertices. One count read;
/// the candidate rows go to the host (ADR-0026, item 14).
pub(crate) fn candidates(
    keys: &SortKeys,
    mins: &DeviceBuffer<u32>,
    count: usize,
    ids: &DeviceBuffer<u32>,
    flags: &DeviceBuffer<u32>,
    last_vertex: u32,
) -> Result<Candidates> {
    let words = keys.num_words();
    if count == 0 {
        return Ok(Candidates::none(words));
    }
    let client = mins.client();
    let plane = fit(keys.words().len() / words)?;
    let is_candidate = DeviceBuffer::<u32>::uninit(client, count)?;
    let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, count, 4, 1, 1)?;
    debug_assert_bindings(client, 3);
    // SAFETY: entities g < count ≤ keys.len() read words j · plane + g < words · plane, the
    // planar length of `keys`, and write is_candidate[g] within its count entries. Vertex
    // indices are clamped to last_vertex: below flags.len() when there are vertices, and index
    // 0 of the padded allocation otherwise.
    unsafe {
        candidate_flags_kernel::launch_unchecked(
            client,
            launch.cube_count.clone(),
            launch.cube_dim,
            keys.words().arg(),
            flags.arg(),
            is_candidate.arg(),
            fit(count)?,
            plane,
            last_vertex,
            launch.items_per_unit,
            words,
        );
    }
    let pos = exclusive_scan(&is_candidate)?;
    // The candidate count to the host.
    let n = reduce(&is_candidate, ReduceOp::Sum)?.to_host()?[0] as usize;
    if n == 0 {
        return Ok(Candidates::none(words));
    }
    let row_len = 2 + 3 * words;
    let out =
        DeviceBuffer::<u32>::uninit(client, n.checked_mul(row_len).ok_or(GridError::TooLarge)?)?;
    fit(out.len())?;
    debug_assert_bindings(client, 6);
    // SAFETY: as above for `keys` and the vertex reads; flags, pos and mins hold at least count
    // entries. Candidate g writes row pos[g] < n, the n · row_len words of `out`. Origin ids
    // are read at 2 · v and 2 · v + 1 with v ≤ last_vertex, within `ids` (two words per
    // vertex) or its padded allocation when there are no vertices.
    unsafe {
        candidate_rows_kernel::launch_unchecked(
            client,
            launch.cube_count,
            launch.cube_dim,
            keys.words().arg(),
            is_candidate.arg(),
            pos.arg(),
            mins.arg(),
            ids.arg(),
            out.arg(),
            fit(count)?,
            plane,
            last_vertex,
            launch.items_per_unit,
            words,
        );
    }
    // The candidate keys to the host.
    let rows = out.to_host()?;
    let mut c = Candidates::none(words);
    for row in rows.chunks_exact(row_len) {
        c.entities.push(row[0]);
        c.in_owned.push(row[1] == 1);
        c.vertices.extend_from_slice(&row[2..2 + words]);
        let mut key = decode_origin_ids(&row[2 + words..]);
        key.sort_unstable();
        c.keys.extend_from_slice(&key);
    }
    Ok(c)
}

/// The ranks of row `v` of `s`.
fn row(s: &VertexSharers, v: u32) -> &[i32] {
    &s.ranks[s.offsets[v as usize]..s.offsets[v as usize + 1]]
}

/// A holder's reply to a key: it does not hold the entity.
const NOT_HELD: i32 = 0;
/// It holds the entity in ghost cells only.
const HELD: i32 = 1;
/// It holds the entity in an owned cell.
const HELD_IN_OWNED: i32 = 2;

/// The owners of the candidates of every entity type of one dimension. Collective: every rank
/// calls it once per dimension, with its candidates of every type (in any type order).
///
/// Each candidate's key goes to every other rank in the intersection of its vertices' sharer
/// sets, over the sharing neighbourhood `ex`; the reply says whether that rank holds the
/// entity, and whether in an owned cell. Keys are prefixed with their length, which
/// identifies the entity type within a dimension.
///
/// # Errors
///
/// The communicator's errors.
///
/// # Panics
///
/// If a candidate has no holder in an owned cell, which happens only if the vertex sharers are
/// not consistent across ranks.
pub(crate) fn owners(
    comm: &dyn CommDyn,
    ex: &SharingExchange,
    sharing: Option<Sharing<'_>>,
    types: Vec<Candidates>,
) -> Result<Vec<Ownership>> {
    let me = comm.rank() as i32;
    let SharingExchange {
        nb,
        neighbours,
        slot,
    } = ex;

    // Queries: per neighbour, the keys and the (type, candidate) they ask about.
    let mut queries: Vec<Vec<i64>> = vec![Vec::new(); neighbours.len()];
    let mut asked: Vec<Vec<(usize, usize)>> = vec![Vec::new(); neighbours.len()];
    if let Some(s) = sharing {
        for (t, c) in types.iter().enumerate() {
            for i in 0..c.len() {
                let verts = &c.vertices[i * c.words..(i + 1) * c.words];
                let (first, rest) = verts.split_first().expect("an entity has vertices");
                for &r in row(s.sharers, *first) {
                    let shared = rest
                        .iter()
                        .all(|&v| row(s.sharers, v).binary_search(&r).is_ok());
                    if r != me && shared {
                        let k = slot[&(r as usize)];
                        queries[k].push(c.words as i64);
                        queries[k].extend_from_slice(c.key(i));
                        asked[k].push((t, i));
                    }
                }
            }
        }
    }
    let counts: Vec<usize> = queries.iter().map(Vec::len).collect();
    let (received, received_counts) =
        comm.neighbour_all_to_all_v(nb, &queries.concat(), &counts)?;

    // Replies, in the order of the queries of each source.
    let mut lookup: HashMap<&[i64], (usize, usize)> = HashMap::new();
    for (t, c) in types.iter().enumerate() {
        for i in 0..c.len() {
            lookup.insert(c.key(i), (t, i));
        }
    }
    let mut replies = Vec::new();
    let mut reply_counts = Vec::with_capacity(received_counts.len());
    let mut at = 0;
    for &n in &received_counts {
        let before = replies.len();
        let end = at + n;
        while at < end {
            let words = received[at] as usize;
            let key = &received[at + 1..at + 1 + words];
            replies.push(match lookup.get(key) {
                None => NOT_HELD,
                Some(&(t, i)) if types[t].in_owned[i] => HELD_IN_OWNED,
                Some(_) => HELD,
            });
            at += 1 + words;
        }
        reply_counts.push(replies.len() - before);
    }
    let (answers, answer_counts) = comm.neighbour_all_to_all_v(nb, &replies, &reply_counts)?;

    // The owned-cell holders of each candidate, ascending, and the owner among them; the
    // other holders, ascending (neighbours are ascending).
    let mut others: Vec<Vec<Vec<i32>>> = types.iter().map(|c| vec![Vec::new(); c.len()]).collect();
    let mut holders: Vec<Vec<Vec<i32>>> = types
        .iter()
        .map(|c| {
            (0..c.len())
                .map(|i| if c.in_owned[i] { vec![me] } else { Vec::new() })
                .collect()
        })
        .collect();
    let mut at = 0;
    for (k, &n) in answer_counts.iter().enumerate() {
        assert_eq!(n, asked[k].len(), "one reply per query");
        for (&(t, i), &a) in asked[k].iter().zip(&answers[at..at + n]) {
            if a == HELD_IN_OWNED {
                holders[t][i].push(neighbours[k] as i32);
            }
            if a != NOT_HELD {
                others[t][i].push(neighbours[k] as i32);
            }
        }
        at += n;
    }
    Ok(types
        .into_iter()
        .zip(holders)
        .zip(others)
        .map(|((c, holders), others)| {
            let owners = holders
                .into_iter()
                .enumerate()
                .map(|(i, mut h)| {
                    assert!(
                        !h.is_empty(),
                        "entity {:?} has no holder in an owned cell: inconsistent vertex sharers",
                        c.key(i)
                    );
                    h.sort_unstable();
                    h[(owner_hash(c.key(i)) % h.len() as u64) as usize]
                })
                .collect();
            Ownership {
                candidates: c,
                owners,
                holders: others,
            }
        })
        .collect())
}

/// The global number of every candidate of every entity type, given those of the candidates
/// this rank owns (`owned[t][i]`; other entries are ignored). Collective, like [`owners`]:
/// each owner sends the length-prefixed key and global number of its candidates to their other
/// holders, over the sharing neighbourhood `ex`.
///
/// # Errors
///
/// The communicator's errors.
///
/// # Panics
///
/// If a ghost candidate receives no number, which happens only if the vertex sharers are not
/// consistent across ranks.
pub(crate) fn global_numbers(
    comm: &dyn CommDyn,
    ex: &SharingExchange,
    types: &[Ownership],
    owned: &[Vec<i64>],
) -> Result<Vec<Vec<i64>>> {
    let me = comm.rank() as i32;
    let mut sends: Vec<Vec<i64>> = vec![Vec::new(); ex.neighbours.len()];
    for (o, globals) in types.iter().zip(owned) {
        let c = &o.candidates;
        for i in (0..c.len()).filter(|&i| o.owners[i] == me) {
            for r in &o.holders[i] {
                let send = &mut sends[ex.slot[&(*r as usize)]];
                send.push(c.words as i64);
                send.extend_from_slice(c.key(i));
                send.push(globals[i]);
            }
        }
    }
    let counts: Vec<usize> = sends.iter().map(Vec::len).collect();
    let (received, _) = comm.neighbour_all_to_all_v(&ex.nb, &sends.concat(), &counts)?;

    let mut lookup: HashMap<&[i64], (usize, usize)> = HashMap::new();
    for (t, o) in types.iter().enumerate() {
        for i in 0..o.candidates.len() {
            lookup.insert(o.candidates.key(i), (t, i));
        }
    }
    let mut out: Vec<Vec<i64>> = types
        .iter()
        .zip(owned)
        .map(|(o, globals)| {
            (0..o.candidates.len())
                .map(|i| if o.owners[i] == me { globals[i] } else { -1 })
                .collect()
        })
        .collect();
    let mut at = 0;
    while at < received.len() {
        let words = received[at] as usize;
        let key = &received[at + 1..at + 1 + words];
        // Only the other holders receive, so the key is a candidate here.
        let &(t, i) = lookup
            .get(key)
            .expect("a received key is a local candidate");
        out[t][i] = received[at + 1 + words];
        at += 2 + words;
    }
    for (o, globals) in types.iter().zip(&out) {
        if let Some(i) = globals.iter().position(|&g| g < 0) {
            panic!(
                "ghost entity {:?} received no global number: inconsistent vertex sharers",
                o.candidates.key(i)
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use super::*;
    use crate::{CellBlock, Topology, encode_origin_ids};
    use gridcl_device::DeviceStrided;
    use gridcl_dims::{Cell2, Cell3, CellType, D2, D3, TopoDim};
    use gridcl_parallel::IndexMap;
    use gridcl_testing::{HostMesh, ThreadComm};

    /// A host mesh of P1 cells spread over `size` ranks: global cell k (blocks in order) is
    /// owned by rank k · size / n, and with `ghosts` every rank also holds the cells that share
    /// a vertex with one of its owned cells.
    struct Spread {
        /// Block, cell type and vertex origin ids of each global cell.
        cells: Vec<(usize, CellType, Vec<i64>)>,
        num_blocks: usize,
        owner: Vec<usize>,
        /// The global cells of each rank: per block, owned then ghost, ascending.
        local: Vec<Vec<usize>>,
    }

    impl Spread {
        fn new(h: &HostMesh, size: usize, ghosts: bool) -> Spread {
            let mut cells = Vec::new();
            for (b, block) in h.blocks.iter().enumerate() {
                let nv = gridcl_refcell::vertices(block.cell).len();
                for row in block.cells.chunks_exact(block.nodes_per_cell()) {
                    let ids: Vec<i64> = row[..nv]
                        .iter()
                        .map(|&n| {
                            h.node_ids
                                .as_ref()
                                .map_or(i64::from(n), |ids| ids[n as usize])
                        })
                        .collect();
                    cells.push((b, block.cell, ids));
                }
            }
            let n = cells.len();
            let owner: Vec<usize> = (0..n).map(|k| k * size / n).collect();
            let local = (0..size)
                .map(|r| {
                    let owned_vertices: BTreeSet<i64> = (0..n)
                        .filter(|&k| owner[k] == r)
                        .flat_map(|k| cells[k].2.iter().copied())
                        .collect();
                    let mut out = Vec::new();
                    for b in 0..h.blocks.len() {
                        let in_block = (0..n).filter(|&k| cells[k].0 == b);
                        out.extend(in_block.clone().filter(|&k| owner[k] == r));
                        out.extend(in_block.filter(|&k| {
                            ghosts
                                && owner[k] != r
                                && cells[k].2.iter().any(|v| owned_vertices.contains(v))
                        }));
                    }
                    out
                })
                .collect();
            Spread {
                cells,
                num_blocks: h.blocks.len(),
                owner,
                local,
            }
        }

        /// The ranks holding each vertex, with whether they hold it in an owned cell.
        fn sharers(&self) -> BTreeMap<i64, BTreeMap<usize, bool>> {
            let mut out: BTreeMap<i64, BTreeMap<usize, bool>> = BTreeMap::new();
            for (r, local) in self.local.iter().enumerate() {
                for &k in local {
                    for &v in &self.cells[k].2 {
                        *out.entry(v).or_default().entry(r).or_default() |= self.owner[k] == r;
                    }
                }
            }
            out
        }

        /// The topology of rank `comm.rank()`, with its vertex sharers. Collective.
        fn topology<TD: TopoDim>(&self, comm: &dyn CommDyn) -> (Topology<TD>, Vec<i64>) {
            let client = gridcl_testing::test_client();
            let r = comm.rank();
            let mut vertex_ids = Vec::new();
            let mut index: HashMap<i64, u32> = HashMap::new();
            let mut blocks = Vec::new();
            for b in 0..self.num_blocks {
                let cells: Vec<usize> = self.local[r]
                    .iter()
                    .copied()
                    .filter(|&k| self.cells[k].0 == b)
                    .collect();
                // Block-local global indices: owners never decrease along a block.
                let first = self.cells.iter().position(|c| c.0 == b).unwrap();
                let cell = self.cells[first].1;
                let mut rows = Vec::new();
                for &k in &cells {
                    for &v in &self.cells[k].2 {
                        let next = vertex_ids.len() as u32;
                        rows.push(*index.entry(v).or_insert_with(|| {
                            vertex_ids.push(v);
                            next
                        }));
                    }
                }
                let owned = cells.iter().filter(|&&k| self.owner[k] == r).count();
                let ghosts = &cells[owned..];
                let map = IndexMap::with_ghosts(
                    comm,
                    owned,
                    ghosts.iter().map(|&k| (k - first) as i64).collect(),
                    ghosts.iter().map(|&k| self.owner[k] as i32).collect(),
                )
                .unwrap();
                let nv = gridcl_refcell::vertices(cell).len() as u32;
                let ids: Vec<i64> = cells.iter().map(|&k| k as i64).collect();
                blocks.push(CellBlock::<TD>::new(
                    TD::Cell::try_from(cell).unwrap(),
                    map,
                    DeviceStrided {
                        data: DeviceBuffer::from_host(&client, &rows).unwrap(),
                        stride: nv,
                    },
                    DeviceBuffer::from_host(&client, &encode_origin_ids(&ids)).unwrap(),
                    DeviceBuffer::zeros(&client, cells.len()).unwrap(),
                ));
            }
            let map = IndexMap::new(comm, vertex_ids.len()).unwrap();
            let mut t = Topology::new(&client, blocks, map, &vertex_ids).unwrap();
            let all = self.sharers();
            let mut s = VertexSharers {
                offsets: vec![0],
                ..VertexSharers::default()
            };
            for v in &vertex_ids {
                for (&q, &owned) in &all[v] {
                    s.ranks.push(q as i32);
                    s.owned_cell.push(owned);
                }
                s.offsets.push(s.ranks.len());
            }
            t.set_vertex_sharers(s);
            (t, vertex_ids)
        }

        /// The owner of every entity of dimension `dim` by the rules: the owned-cell holder
        /// at position `owner_hash(key) mod n`.
        fn owners(&self, dim: usize) -> BTreeMap<Vec<i64>, usize> {
            let mut holders: BTreeMap<Vec<i64>, BTreeSet<usize>> = BTreeMap::new();
            for (k, (_, cell, ids)) in self.cells.iter().enumerate() {
                for sub in gridcl_refcell::sub_entities(*cell, dim) {
                    let mut key: Vec<i64> = sub.iter().map(|&j| ids[j]).collect();
                    key.sort_unstable();
                    holders.entry(key).or_default().insert(self.owner[k]);
                }
            }
            holders
                .into_iter()
                .map(|(key, h)| {
                    let h: Vec<usize> = h.into_iter().collect();
                    let i = (owner_hash(&key) % h.len() as u64) as usize;
                    (key, h[i])
                })
                .collect()
        }
    }

    /// Creates the entities of every dimension 1 ≤ d < tdim on this rank and checks each
    /// local entity's owner, from its index map, against the rules; that the owned entities of
    /// all ranks are the global entities, each once, numbered 0, 1, … globally; that every
    /// ghost has its owner's global number; and that ghosts are ordered by (owner, global).
    fn check<TD: TopoDim>(comm: &ThreadComm, spread: &Spread, at: &str) {
        let me = comm.rank();
        let (t, vertex_ids) = spread.topology::<TD>(comm);
        for dim in 1..TD::N {
            let want = spread.owners(dim);
            let crate::entity::Dimension { blocks, .. } =
                crate::entity::create_dimension(t.client(), comm, &t.entity_cells(), dim).unwrap();
            let mut ghosts = 0i64;
            let mut global = 0i64;
            for block in &blocks {
                let map = &block.index_map;
                let stride = block.vertices.stride as usize;
                let rows = block.vertices.data.to_host().unwrap();
                assert_eq!(rows.len(), map.num_local() * stride, "{at}, dim {dim}");
                let order: Vec<(i32, i64)> = map
                    .ghost_owners()
                    .iter()
                    .copied()
                    .zip(map.ghosts().iter().copied())
                    .collect();
                assert!(order.is_sorted(), "{at}, rank {me}, dim {dim}: ghost order");
                // Per entity: global number, owned flag, key.
                let mut flat = Vec::new();
                for (e, row) in rows.chunks_exact(stride).enumerate() {
                    let mut key: Vec<i64> = row.iter().map(|&v| vertex_ids[v as usize]).collect();
                    key.sort_unstable();
                    let (got, g) = match e.checked_sub(map.num_owned()) {
                        None => (me, map.owned_range().0 + e as i64),
                        Some(k) => (map.ghost_owners()[k] as usize, map.ghosts()[k]),
                    };
                    assert_eq!(
                        got, want[&key],
                        "{at}, rank {me}, dim {dim}: owner of {key:?}"
                    );
                    ghosts += i64::from(got != me);
                    flat.push(g);
                    flat.push(i64::from(got == me));
                    flat.extend_from_slice(&key);
                }
                let all = gridcl_testing::parallel::gather_to_root(comm, &flat).unwrap();
                if me == 0 {
                    let mut numbers: BTreeMap<Vec<i64>, i64> = BTreeMap::new();
                    for row in all.chunks_exact(2 + stride).filter(|r| r[1] == 1) {
                        assert!(
                            numbers.insert(row[2..].to_vec(), row[0]).is_none(),
                            "{at}, dim {dim}: {:?} owned twice",
                            &row[2..]
                        );
                    }
                    let mut owned: Vec<i64> = numbers.values().copied().collect();
                    owned.sort_unstable();
                    assert!(
                        owned.iter().copied().eq(0..map.size_global()),
                        "{at}, dim {dim}: global numbers"
                    );
                    for row in all.chunks_exact(2 + stride).filter(|r| r[1] == 0) {
                        assert_eq!(
                            numbers[&row[2..]],
                            row[0],
                            "{at}, dim {dim}: global number of ghost {:?}",
                            &row[2..]
                        );
                    }
                }
                global += map.size_global();
            }
            // Every multi-rank case shares entities, and some are ghosts somewhere.
            if comm.size() > 1 {
                assert!(comm.sum(ghosts) > 0, "{at}, dim {dim}: no ghosts");
            }
            assert_eq!(global, want.len() as i64, "{at}, dim {dim}: entity count");
        }
    }

    /// The test meshes: the 2D and 3D boxes of every cell type and the mixed box, P1.
    fn meshes() -> (Vec<HostMesh>, Vec<HostMesh>) {
        let d2 = vec![
            gridcl_testing::square_box(Cell2::Triangle, 4, 1),
            gridcl_testing::square_box(Cell2::Quadrilateral, 3, 1).with_shuffled_ids(5),
        ];
        let d3 = vec![
            gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1),
            gridcl_testing::cube_box(Cell3::Hexahedron, 2, 1),
            gridcl_testing::cube_box(Cell3::Prism, 2, 1),
            gridcl_testing::cube_box(Cell3::Pyramid, 1, 1),
            gridcl_testing::mixed_box(2, 1),
        ];
        (d2, d3)
    }

    /// Owners follow the rules on 1–4 ranks, with and without ghost cells, and every entity
    /// has exactly one owner.
    #[test]
    fn owners_follow_the_rules() {
        let (d2, d3) = meshes();
        for size in 1..=4 {
            ThreadComm::run(size, |comm| {
                for ghosts in [false, true] {
                    for (m, h) in d2.iter().enumerate() {
                        let at = format!("2D mesh {m}, {size} ranks, ghosts {ghosts}");
                        check::<D2>(&comm, &Spread::new(h, size, ghosts), &at);
                    }
                    for (m, h) in d3.iter().enumerate() {
                        let at = format!("3D mesh {m}, {size} ranks, ghosts {ghosts}");
                        check::<D3>(&comm, &Spread::new(h, size, ghosts), &at);
                    }
                }
            });
        }
    }

    /// More ranks than cells: empty ranks take part in the exchange.
    #[test]
    fn empty_ranks_take_part() {
        let h = gridcl_testing::square_box(Cell2::Triangle, 1, 1);
        ThreadComm::run(4, |comm| {
            check::<D2>(&comm, &Spread::new(&h, 4, true), "two triangles on 4 ranks");
        });
    }

    /// On one rank there are no candidates, even with sharers: every entity is owned.
    #[test]
    fn no_candidates_on_one_rank() {
        let (_, d3) = meshes();
        ThreadComm::run(1, |comm| {
            let (t, _) = Spread::new(&d3[0], 1, true).topology::<D3>(&comm);
            assert!(t.entity_cells().sharing.is_some());
            for dim in 1..3 {
                let blocks =
                    crate::entity::create_dimension(t.client(), &comm, &t.entity_cells(), dim)
                        .unwrap()
                        .blocks;
                assert!(blocks.iter().all(|b| b.index_map.num_ghosts() == 0));
            }
        });
    }
}
