//! Tags after stage 9: the physical and elementary tags and the name table of the built mesh
//! (docs/design.md, "Tags, regions and submeshes"; ADR-0026, items 3 and 10; ADR-0035,
//! item 8).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, GridError, Result, SortKeys};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, CommExt};
use gridcl_topology::Topology;

use super::input::{LowerTags, TagKind};
use super::number::NumberedBlock;
use super::rendezvous::{Outbox, rendezvous_rank};
use super::{agree, release_pools};
use crate::{MeshTags, TagRegistry};

/// (flat local entity index, value) entries of one dimension.
type Entries = Vec<(u32, i32)>;

/// (source rank, kind word, value, global input position) of each entry a holder received.
type Received = Vec<(usize, i64, i32, i64)>;

/// The words of a lower-dimensional entry before its key: dimension, kind, value, global input
/// position, error id.
const HEAD: usize = 5;

/// The wire word of a tag kind.
fn kind_word(kind: TagKind) -> i64 {
    match kind {
        TagKind::Physical => 0,
        TagKind::Elementary => 1,
    }
}

/// The tag inputs of one rank after stage 8.
pub(crate) struct TagInputs<'a> {
    /// The cell tags: (flat local cell index, value), from [`cell_tags`].
    pub(crate) cells: Entries,
    /// The elementary tags of the cells, from [`cell_elementary`].
    pub(crate) cell_elementary: Entries,
    /// The tags of dimension below the cells' (stage 1).
    pub(crate) lower: &'a [LowerTags],
    /// This rank's name table (name, dimension, value).
    pub(crate) names: &'a [(String, usize, i32)],
    /// This rank's part of the stage-8 directory: the holders of each node id.
    pub(crate) directory: &'a HashMap<i64, Vec<i32>>,
    /// The local index of each local vertex, by origin id.
    pub(crate) vertex_local: &'a HashMap<i64, u32>,
}

/// The cell tags of the stage-8 blocks: cells are numbered flat over the blocks in block
/// order, owned and ghost cells alike.
pub(crate) fn cell_tags(blocks: &[NumberedBlock]) -> Result<Entries> {
    let mut entries = Vec::new();
    let mut first = 0usize;
    for b in blocks {
        for i in 0..b.cells.len() {
            let index = u32::try_from(first + i).map_err(|_| GridError::TooLarge)?;
            entries.extend(b.cells.tags_of(i).iter().map(|&v| (index, v)));
        }
        first += b.cells.len();
    }
    Ok(entries)
}

/// The elementary tags of the stage-8 blocks' cells (none if the cells have none), numbered as
/// [`cell_tags`].
pub(crate) fn cell_elementary(blocks: &[NumberedBlock]) -> Result<Entries> {
    let mut entries = Vec::new();
    let mut first = 0usize;
    for b in blocks {
        for (i, &v) in b.cells.elementary.iter().enumerate() {
            let index = u32::try_from(first + i).map_err(|_| GridError::TooLarge)?;
            entries.push((index, v));
        }
        first += b.cells.len();
    }
    Ok(entries)
}

/// The tag registry of a mesh on `comm` whose topology the stages 1–9 built. Collective.
///
/// - **Names.** The name tables of every rank, agreed: rank 0's, then each further rank's
///   entries that no lower rank gave.
/// - **Cells.** The cell tags, physical and elementary, travelled with their cells (stages 1,
///   4 and 5).
/// - **Lower dimensions.** For every dimension with a tag on any rank, the entities are created
///   (`Topology::create_entities`). Each entry goes to the rendezvous rank of its sorted key,
///   which asks the stage-8 directory for the holders of each corner vertex and forwards the
///   entry to every rank that holds all of them; each of those ranks matches it with
///   `Topology::match_entities` and reports a match back. Owned and ghost holders thus get the
///   same values (ADR-0026, item 10).
///   Elementary entries take the same path.
/// - **Registry.** Entries are sorted by (index, value) without repeats; group sizes of the
///   physical tags are local counts, ghosts included.
///
/// # Errors
///
/// `UnmatchedTagElement` on every rank for the entry of lowest global input position that no
/// rank matched (with its dimension and error id); agreed allocation, device and communication
/// errors, and those of `create_entities`.
pub(crate) fn registry<TD: TopoDim>(
    client: &Client,
    comm: &dyn CommDyn,
    topology: &mut Topology<TD>,
    mut inputs: TagInputs<'_>,
) -> Result<TagRegistry> {
    let names = agree_names(comm, inputs.names)?;
    let mut physical: BTreeMap<usize, Entries> = BTreeMap::new();
    let mut elementary: BTreeMap<usize, Entries> = BTreeMap::new();
    physical.insert(TD::N, std::mem::take(&mut inputs.cells));
    elementary.insert(TD::N, std::mem::take(&mut inputs.cell_elementary));
    match_lower(
        client,
        comm,
        topology,
        &inputs,
        [&mut physical, &mut elementary],
    )?;
    let local = (|| {
        let mut registry = TagRegistry::new(client, TD::N, names);
        let upload = |dim: usize, mut e: Entries| -> Result<(MeshTags, BTreeMap<i32, usize>)> {
            e.sort_unstable();
            e.dedup();
            let mut sizes: BTreeMap<i32, usize> = BTreeMap::new();
            for &(_, v) in &e {
                *sizes.entry(v).or_default() += 1;
            }
            let (indices, values): (Vec<u32>, Vec<i32>) = e.into_iter().unzip();
            let tags = MeshTags {
                dim,
                indices: DeviceBuffer::from_host(client, &indices)?,
                values: DeviceBuffer::from_host(client, &values)?,
            };
            Ok((tags, sizes))
        };
        for (dim, e) in physical.into_iter().filter(|(_, e)| !e.is_empty()) {
            let (tags, sizes) = upload(dim, e)?;
            registry.set_physical(tags, sizes);
        }
        for (dim, e) in elementary.into_iter().filter(|(_, e)| !e.is_empty()) {
            registry.set_elementary(upload(dim, e)?.0);
        }
        Ok(registry)
    })();
    agree(comm, local)
}

/// The agreed name table: rank 0's entries, then those of each further rank that no lower rank
/// gave. Collective.
fn agree_names(
    comm: &dyn CommDyn,
    names: &[(String, usize, i32)],
) -> Result<Vec<(String, usize, i32)>> {
    let p = comm.size();
    let mut out = Outbox::new(p);
    for dest in 0..p {
        for (name, dim, value) in names {
            let bytes: Vec<i64> = name.bytes().map(i64::from).collect();
            out.push(dest, &[&[*dim as i64, i64::from(*value)], &bytes]);
        }
    }
    let inbox = out.exchange(comm)?;
    let mut union: Vec<(String, usize, i32)> = Vec::new();
    let mut source = 0;
    let mut before = 0;
    for (s, r) in inbox.records() {
        if s != source {
            source = s;
            before = union.len();
        }
        let bytes: Vec<u8> = r[2..].iter().map(|&b| b as u8).collect();
        let name = String::from_utf8(bytes).expect("names travel as UTF-8 bytes");
        let entry = (name, r[0] as usize, r[1] as i32);
        if !union[..before].contains(&entry) {
            union.push(entry);
        }
    }
    Ok(union)
}

/// Matches the tags of dimension below the cells' and adds their entries to `entries`, the
/// physical and the elementary entries by dimension. Collective.
fn match_lower<TD: TopoDim>(
    client: &Client,
    comm: &dyn CommDyn,
    topology: &mut Topology<TD>,
    inputs: &TagInputs<'_>,
    entries: [&mut BTreeMap<usize, Entries>; 2],
) -> Result<()> {
    let p = comm.size();
    let tagged = inputs
        .lower
        .iter()
        .filter(|t| !t.tags.values.is_empty())
        .fold(0u32, |m, t| m | 1 << gridcl_refcell::tdim(t.tags.entity));
    let tagged = comm.all_gather(tagged).into_iter().fold(0, |a, b| a | b);
    if tagged == 0 {
        return Ok(());
    }

    // Each entry [dim, kind, value, position, error id, sorted key…] to the rendezvous rank of
    // its key.
    let mut out = Outbox::new(p);
    for t in inputs.lower {
        let nv = gridcl_refcell::vertices(t.tags.entity).len();
        let dim = gridcl_refcell::tdim(t.tags.entity) as i64;
        let kind = kind_word(t.tags.kind);
        let rows = t.tags.corners.chunks_exact(nv).zip(&t.tags.values);
        for (j, (corners, &value)) in rows.enumerate() {
            let mut key = corners.to_vec();
            key.sort_unstable();
            let at = t.first + j as i64;
            let head = [dim, kind, i64::from(value), at, t.tags.error_id(j, at)];
            out.push(rendezvous_rank(&key, p), &[&head, &key]);
        }
    }
    let at_key = out.exchange(comm)?;

    // The holders of each corner, from the stage-8 directory (none for an unknown id).
    let corners: BTreeSet<i64> = at_key
        .records()
        .flat_map(|(_, r)| r[HEAD..].iter().copied())
        .collect();
    let mut queries = Outbox::new(p);
    for &c in &corners {
        queries.push(rendezvous_rank(&[c], p), &[&[c]]);
    }
    let mut answers = Outbox::new(p);
    for (source, r) in queries.exchange(comm)?.records() {
        let ranks: Vec<i64> = inputs
            .directory
            .get(&r[0])
            .map_or_else(Vec::new, |h| h.iter().map(|&x| i64::from(x)).collect());
        answers.push(source, &[&[r[0]], &ranks]);
    }
    let holders: HashMap<i64, Vec<i64>> = answers
        .exchange(comm)?
        .records()
        .map(|(_, r)| (r[0], r[1..].to_vec()))
        .collect();

    // Each entry to every rank that holds all its corners (holder lists are ascending).
    let mut forward = Outbox::new(p);
    for (_, r) in at_key.records() {
        let key = &r[HEAD..];
        let mut common = holders[&key[0]].clone();
        for c in &key[1..] {
            let h = &holders[c];
            common.retain(|x| h.binary_search(x).is_ok());
        }
        for &rank in &common {
            forward.push(rank as usize, &[r]);
        }
    }
    let at_holder = forward.exchange(comm)?;

    for dim in (0..TD::N).filter(|d| tagged & 1 << d != 0) {
        topology.create_entities(comm, dim)?;
        release_pools(client, &format!("stage 10 (entities of dimension {dim})"));
    }

    // Match by (dimension, corner count): entries whose corners are all local vertices.
    let local = (|| {
        let mut groups: BTreeMap<(usize, usize), Received> = BTreeMap::new();
        let mut keys: BTreeMap<(usize, usize), Vec<u32>> = BTreeMap::new();
        for (source, r) in at_holder.records() {
            let key = &r[HEAD..];
            let Some(mut local) = key
                .iter()
                .map(|id| inputs.vertex_local.get(id).copied())
                .collect::<Option<Vec<u32>>>()
            else {
                continue;
            };
            local.sort_unstable();
            let group = (r[0] as usize, key.len());
            groups
                .entry(group)
                .or_default()
                .push((source, r[1], r[2] as i32, r[3]));
            keys.entry(group).or_default().extend(local);
        }
        let mut matched = Vec::new();
        for (&(dim, nv), group) in &groups {
            let keys = SortKeys::from_host(client, &keys[&(dim, nv)], nv)?;
            // An I/O copy (ADR-0026, item 14): tag matching at mesh creation.
            let found = topology.match_entities(dim, &keys)?.to_host()?;
            release_pools(client, &format!("stage 10 (matches of dimension {dim})"));
            for (&(source, kind, value, position), &f) in group.iter().zip(&found) {
                if f != u32::MAX {
                    matched.push((source, dim, kind, f, value, position));
                }
            }
        }
        Ok(matched)
    })();
    let matched = agree(comm, local)?;

    // Matches back to the rendezvous rank, which names the first entry nobody matched.
    let mut replies = Outbox::new(p);
    let [physical, elementary] = entries;
    for &(source, dim, kind, f, value, position) in &matched {
        let entries = if kind == kind_word(TagKind::Elementary) {
            &mut *elementary
        } else {
            &mut *physical
        };
        entries.entry(dim).or_default().push((f, value));
        replies.push(source, &[&[position]]);
    }
    let found: HashSet<i64> = replies
        .exchange(comm)?
        .records()
        .map(|(_, r)| r[0])
        .collect();
    // (position, dimension) as one word: positions stay below 2^61.
    let unmatched = at_key
        .records()
        .filter(|(_, r)| !found.contains(&r[3]))
        .map(|(_, r)| (r[3] as u64) << 2 | r[0] as u64)
        .min()
        .unwrap_or(u64::MAX);
    let lowest = comm.min(unmatched);
    if lowest == u64::MAX {
        return Ok(());
    }
    // The error id of that entry, from its rendezvous rank.
    let position = (lowest >> 2) as i64;
    let tag = at_key
        .records()
        .find(|(_, r)| r[3] == position)
        .map_or(i64::MAX, |(_, r)| r[4]);
    Err(GridError::UnmatchedTagElement {
        dim: (lowest & 3) as usize,
        tag: comm.min(tag),
    })
}
