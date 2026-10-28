//! The checks of the M6 acceptance tests, generic over the communicator: gridcl-mpi-tests runs
//! them on `Mpi`, and gridcl/tests/parallel_threads.rs on [`ThreadComm`](crate::ThreadComm)
//! (docs/plans/M6.md, T0).
//!
//! A check panics with the case, the rank and what differs; under MPI the panic aborts the job.
//!
//! - [`invariance`]: entity counts, total volume, tag counts and names, canonical entity vertex
//!   lists and permutation bits equal a one-rank build ([`Serial`], on every rank).
//! - [`scatter`]: a forward scatter of global ids gives the ghosts' global ids, and a reverse
//!   `Add` of ones gives the number of ranks holding each index, for the cell, vertex, node and
//!   entity maps at block sizes 1 and 3.
//! - [`determinism`]: two builds, with entities and scatters, give bitwise equal arrays.
//! - [`locality`]: Hilbert ordering at least halves the mean node-index spread of the cells of a
//!   shuffled box.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use cubecl::prelude::Client;
use gridcl::{
    Cell2, Cell3, CellOrdering, Comm, CommDyn, CommExt, CommScalar, ConnKey, Connectivity, D1, D2,
    D3, DeviceBuffer, DeviceGrid, Dim, DistGraph, EmbeddedIn, GhostMode, IndexMap, Mesh,
    Partitioner, Real, Result, ReverseOp, Serial, TopoDim,
};
use gridcl_dims::CellType;

use crate::{HostChunk, HostMesh, TOL_F32, TOL_F64, supports};

/// A test mesh family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// `cube_box` of one cell type.
    Cube(Cell3),
    /// `mixed_box`.
    Mixed,
    /// `square_box` of one cell type.
    Square(Cell2),
    /// `interval_box`.
    Interval,
    /// `sphere`, one refinement.
    Sphere,
}

/// Every family: `cube_box` of every `Cell3`, `mixed_box`, `square_box` of both `Cell2`,
/// `interval_box` and `sphere`.
pub const FAMILIES: [Family; 9] = [
    Family::Cube(Cell3::Tetrahedron),
    Family::Cube(Cell3::Hexahedron),
    Family::Cube(Cell3::Prism),
    Family::Cube(Cell3::Pyramid),
    Family::Mixed,
    Family::Square(Cell2::Triangle),
    Family::Square(Cell2::Quadrilateral),
    Family::Interval,
    Family::Sphere,
];

impl Family {
    /// The host mesh at degree `p`.
    pub fn host(self, p: u32) -> HostMesh {
        match self {
            Family::Cube(Cell3::Hexahedron) => crate::cube_box(Cell3::Hexahedron, 3, p),
            Family::Cube(cell) => crate::cube_box(cell, 2, p),
            Family::Mixed => crate::mixed_box(2, p),
            Family::Square(cell) => crate::square_box(cell, 3, p),
            Family::Interval => crate::interval_box(12, p),
            Family::Sphere => crate::sphere(1, p),
        }
    }
}

/// How the input is spread over the ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Distribution {
    /// Even chunks ([`HostMesh::chunk`]) and the `Block` partitioner.
    Even,
    /// Everything on rank 0.
    RankZero,
    /// Even chunks and [`RoundRobin`], which forces migration.
    RoundRobin,
}

const DISTRIBUTIONS: [Distribution; 3] = [
    Distribution::Even,
    Distribution::RankZero,
    Distribution::RoundRobin,
];

/// The ghost modes of the tests.
pub const GHOST_MODES: [GhostMode; 4] = [
    GhostMode::None,
    GhostMode::SharedFacet,
    GhostMode::SharedVertex { layers: 1 },
    GhostMode::SharedVertex { layers: 2 },
];

/// One test case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Case {
    /// The mesh family.
    pub family: Family,
    /// The degree.
    pub degree: u32,
    /// The ghost mode.
    pub ghosts: GhostMode,
    /// The input distribution.
    pub distribution: Distribution,
    /// f64 if the device supports it, else f32.
    pub double: bool,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} p={} {:?} {:?} {}",
            self.family,
            self.degree,
            self.ghosts,
            self.distribution,
            if self.double { "f64" } else { "f32" }
        )
    }
}

/// How many cases to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every family at p = 1 and 2 in every ghost mode (the MPI tests).
    Full,
    /// Every family once, degrees and ghost modes alternating (the ThreadComm tests).
    Quick,
}

/// The cases of [`invariance`]: distributions cycle over the cases, and every other ghost mode
/// runs in f32.
pub fn cases(coverage: Coverage) -> Vec<Case> {
    let mut out = Vec::new();
    for (f, &family) in FAMILIES.iter().enumerate() {
        for degree in 1..=2 {
            for (g, &ghosts) in GHOST_MODES.iter().enumerate() {
                let quick = degree == 1 + (f % 2) as u32 && g == f % GHOST_MODES.len();
                if coverage == Coverage::Full || quick {
                    out.push(Case {
                        family,
                        degree,
                        ghosts,
                        distribution: DISTRIBUTIONS[(f + degree as usize + g) % 3],
                        double: g % 2 == 0,
                    });
                }
            }
        }
    }
    out
}

/// The cases of [`scatter`]: every family at p = 1 and 2, in f32, ghost modes and
/// distributions cycling (ghosts first).
pub fn scatter_cases() -> Vec<Case> {
    const MODES: [GhostMode; 4] = [
        GhostMode::SharedFacet,
        GhostMode::SharedVertex { layers: 1 },
        GhostMode::SharedVertex { layers: 2 },
        GhostMode::None,
    ];
    let mut out = Vec::new();
    for (f, &family) in FAMILIES.iter().enumerate() {
        for degree in 1..=2 {
            let k = f + degree as usize;
            out.push(Case {
                family,
                degree,
                ghosts: MODES[k % 4],
                distribution: DISTRIBUTIONS[k % 3],
                double: false,
            });
        }
    }
    out
}

/// The cases of [`determinism`].
pub fn determinism_cases() -> Vec<Case> {
    let case = |family, degree, ghosts, distribution, double| Case {
        family,
        degree,
        ghosts,
        distribution,
        double,
    };
    vec![
        case(
            Family::Cube(Cell3::Tetrahedron),
            2,
            GhostMode::SharedVertex { layers: 1 },
            Distribution::Even,
            true,
        ),
        case(
            Family::Mixed,
            1,
            GhostMode::SharedFacet,
            Distribution::RoundRobin,
            false,
        ),
        case(
            Family::Sphere,
            2,
            GhostMode::SharedFacet,
            Distribution::Even,
            false,
        ),
        case(
            Family::Square(Cell2::Quadrilateral),
            1,
            GhostMode::SharedVertex { layers: 2 },
            Distribution::RankZero,
            true,
        ),
    ]
}

/// A test-only partitioner that sends global graph vertex i to rank i mod nparts, so almost
/// every cell migrates.
#[derive(Clone, Copy, Debug, Default)]
pub struct RoundRobin;

impl Partitioner for RoundRobin {
    fn name(&self) -> &str {
        "round-robin"
    }
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let first = g.vtxdist[comm.rank()];
        let n = g.xadj.len().saturating_sub(1);
        Ok((0..n)
            .map(|i| ((first + i as i64) % nparts as i64) as i32)
            .collect())
    }
}

/// `data` of every rank, concatenated in rank order on rank 0; empty elsewhere.
///
/// # Errors
///
/// Errors of `CommExt::all_to_all_v`.
pub fn gather_to_root<T: CommScalar>(comm: &dyn CommDyn, data: &[T]) -> Result<Vec<T>> {
    let mut counts = vec![0; comm.size()];
    counts[0] = data.len();
    Ok(comm.all_to_all_v(data, &counts)?.0)
}

/// The ids of interleaved (hi ^ 0x8000_0000, lo) device words (gridcl-topology's layout).
fn decode_ids(words: &[u32]) -> Vec<i64> {
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[hi, lo]| ((u64::from(hi ^ 0x8000_0000) << 32) | u64::from(*lo)) as i64)
        .collect()
}

/// The chunk of `host` this rank passes under `distribution`.
fn chunk_of(host: &HostMesh, distribution: Distribution, comm: &dyn CommDyn) -> HostChunk {
    match distribution {
        Distribution::RankZero if comm.rank() == 0 => host.chunk(0, 1),
        Distribution::RankZero => HostChunk::empty(host.gdim),
        Distribution::Even | Distribution::RoundRobin => host.chunk(comm.rank(), comm.size()),
    }
}

/// The mesh of `host` on `comm` under `case` and `ordering`, with the entities of every
/// dimension created.
///
/// # Errors
///
/// Errors of `build_on` and `create_entities`.
pub fn build<F, TD, GD, C>(
    comm: C,
    client: &Client,
    host: &HostMesh,
    case: &Case,
    ordering: CellOrdering,
) -> Result<Mesh<F, TD, GD, C>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let chunk = chunk_of(host, case.distribution, &comm);
    let mut builder = chunk
        .builder::<F, TD, GD>(client)?
        .ghosts(case.ghosts)
        .ordering(ordering);
    if case.distribution == Distribution::RoundRobin {
        builder = builder.partitioner(RoundRobin);
    }
    let mut mesh = builder.build_on(comm)?;
    for d in 0..TD::N {
        mesh.create_entities(d)?;
    }
    Ok(mesh)
}

/// A check that runs on a mesh of fixed types; [`dispatch`] picks the types of a case.
pub trait MeshCheck {
    /// Runs the check for `case` on `host`, in scalar `F` and dimensions `TD`, `GD`.
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim;
}

/// Runs `check` on `case` with the dimensions of its family, in f64 if `case.double` and the
/// device supports f64, else in f32.
pub fn dispatch<K: MeshCheck>(client: &Client, case: &Case, check: &mut K) {
    let host = case.family.host(case.degree);
    let double = case.double && supports::<f64>(client);
    macro_rules! run {
        ($td:ty, $gd:ty) => {
            if double {
                check.check::<f64, $td, $gd>(case, &host)
            } else {
                check.check::<f32, $td, $gd>(case, &host)
            }
        };
    }
    match case.family {
        Family::Cube(_) | Family::Mixed => run!(D3, D3),
        Family::Square(_) => run!(D2, D2),
        Family::Interval => run!(D1, D1),
        Family::Sphere => run!(D2, D3),
    }
}

/// The value of `r`, or a panic naming `context`.
fn ok<T>(r: Result<T>, context: &str) -> T {
    r.unwrap_or_else(|e| panic!("{context}: {e}"))
}

/// The relative tolerance of `F`.
fn tol<F: Real>() -> f64 {
    if size_of::<F>() == 8 {
        TOL_F64
    } else {
        TOL_F32
    }
}

/// Global and local quantities of a mesh that a partition must not change.
#[derive(Debug, Default)]
pub struct Summary {
    /// Global counts: (dimension, cell type, count) of every cell block and entity block.
    pub counts: Vec<(usize, CellType, i64)>,
    /// The total volume.
    pub volume: f64,
    /// The tag name table.
    pub names: Vec<(String, usize, i32)>,
    /// Rank 0: the owned entities of each physical tag, by (dimension, value).
    pub tag_counts: BTreeMap<(usize, i32), i64>,
    /// Rank 0: the sorted canonical vertex lists (origin ids) of the owned entities, per
    /// dimension below the cells.
    pub entities: Vec<Vec<Vec<i64>>>,
    /// Rank 0: the sorted origin ids of the owned cells.
    pub cell_ids: Vec<i64>,
    /// This rank: the canonical vertex lists of every local entity, owned or ghost.
    pub local_entities: Vec<Vec<Vec<i64>>>,
    /// This rank: (cell origin id, permutation bits) of every local cell (tdim ≥ 2).
    pub local_bits: Vec<(i64, u32)>,
}

/// Rows of `nv` values flattened as `[nv, row…]` per row, for gathering ragged rows.
fn push_row(out: &mut Vec<i64>, row: &[i64]) {
    out.push(row.len() as i64);
    out.extend_from_slice(row);
}

/// The rows of [`push_row`]'s layout.
fn split_rows(flat: &[i64]) -> Vec<Vec<i64>> {
    let mut rows = Vec::new();
    let mut i = 0;
    while i < flat.len() {
        let n = flat[i] as usize;
        rows.push(flat[i + 1..i + 1 + n].to_vec());
        i += 1 + n;
    }
    rows
}

/// The [`Summary`] of `mesh`, whose entities exist; `quad_degree` is the degree of `measure`.
///
/// # Errors
///
/// Device and communication errors.
pub fn summarise<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>, quad_degree: u32) -> Result<Summary>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let comm = mesh.comm();
    let tdim = TD::N;
    let mut s = Summary {
        volume: mesh.measure(quad_degree)?,
        names: mesh
            .tags()
            .names()
            .map(|(n, d, v)| (n.to_string(), d, v))
            .collect(),
        ..Summary::default()
    };
    let vertex_ids = decode_ids(&mesh.vertex_origin_ids().to_host()?);

    // Cells: counts, owned ids, permutation bits; flat offsets for the cell tags.
    let mut owned_cells = Vec::new();
    let mut cell_offsets = Vec::new();
    let mut offset = 0;
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let cell: CellType = block.cell.into();
        let map = &block.index_map;
        s.counts.push((tdim, cell, map.size_global()));
        let ids = decode_ids(&block.origin_ids.to_host()?);
        owned_cells.extend_from_slice(&ids[..map.num_owned()]);
        if let Some(bits) = mesh.permutation_info(b) {
            s.local_bits
                .extend(ids.iter().copied().zip(bits.to_host()?));
        }
        cell_offsets.push((offset, map.num_local(), map.num_owned()));
        offset += map.num_local();
    }

    // Entities: counts and canonical vertex lists.
    // Per dimension, then the cells: (first flat index, local count, owned count) per block.
    let mut offsets = vec![Vec::new(); tdim];
    s.local_entities = vec![Vec::new(); tdim];
    let mut owned_rows = vec![Vec::new(); tdim];
    for d in 0..tdim {
        for block in mesh.entity_blocks(d)? {
            let map = &block.index_map;
            s.counts.push((d, block.cell, map.size_global()));
            let stride = block.vertices.stride as usize;
            let verts = block.vertices.data.to_host()?;
            for (e, row) in verts.chunks_exact(stride).enumerate() {
                let ids: Vec<i64> = row.iter().map(|&v| vertex_ids[v as usize]).collect();
                if e < map.num_owned() {
                    push_row(&mut owned_rows[d], &ids);
                }
                s.local_entities[d].push(ids);
            }
            offsets[d].push((block.offset as usize, map.num_local(), map.num_owned()));
        }
    }

    // Owned tag counts.
    let mut counts: BTreeMap<(usize, i32), i64> = BTreeMap::new();
    offsets.push(cell_offsets);
    for (d, offsets) in offsets.iter().enumerate() {
        let Some(tags) = mesh.tags().physical(d) else {
            continue;
        };
        let (indices, values) = (tags.indices.to_host()?, tags.values.to_host()?);
        for (&i, &v) in indices.iter().zip(&values) {
            let i = i as usize;
            let &(first, _, owned) = offsets
                .iter()
                .find(|&&(first, len, _)| (first..first + len).contains(&i))
                .unwrap_or_else(|| panic!("tag index {i} of dimension {d} is in no block"));
            if i - first < owned {
                *counts.entry((d, v)).or_default() += 1;
            }
        }
    }
    let flat: Vec<i64> = counts
        .iter()
        .flat_map(|(&(d, v), &n)| [d as i64, i64::from(v), n])
        .collect();
    for &[d, v, n] in gather_to_root(comm, &flat)?.as_chunks::<3>().0 {
        *s.tag_counts.entry((d as usize, v as i32)).or_default() += n;
    }

    // Gathered entity lists and cell ids.
    for rows in &owned_rows {
        let mut all = split_rows(&gather_to_root(comm, rows)?);
        all.sort_unstable();
        s.entities.push(all);
    }
    s.cell_ids = gather_to_root(comm, &owned_cells)?;
    s.cell_ids.sort_unstable();
    Ok(s)
}

/// The parallel-invariance check of one case on `comm`.
pub struct Invariance<'a, C: Comm> {
    /// The communicator of the parallel build.
    pub comm: &'a C,
    /// The client of this rank.
    pub client: &'a Client,
}

impl<C: Comm> MeshCheck for Invariance<'_, C> {
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let rank = self.comm.rank();
        let at = format!("{case}, rank {rank}");
        let quad = 2 * case.degree;
        let serial = ok(
            build::<F, TD, GD, _>(Serial, self.client, host, case, CellOrdering::Input),
            &at,
        );
        let want = ok(summarise(&serial, quad), &at);
        let mesh = ok(
            build::<F, TD, GD, _>(
                self.comm.clone(),
                self.client,
                host,
                case,
                CellOrdering::Input,
            ),
            &at,
        );
        let got = ok(summarise(&mesh, quad), &at);

        assert_eq!(
            got.counts, want.counts,
            "{case}, rank {rank}: entity counts"
        );
        let err = (got.volume - want.volume).abs() / want.volume.abs();
        assert!(
            err <= tol::<F>(),
            "{case}, rank {rank}: volume {} vs {} (relative error {err:e})",
            got.volume,
            want.volume
        );
        assert_eq!(got.names, want.names, "{case}, rank {rank}: tag names");
        if rank == 0 {
            assert_eq!(got.tag_counts, want.tag_counts, "{case}: tag counts");
            assert_eq!(got.cell_ids, want.cell_ids, "{case}: owned cells");
            for (d, (g, w)) in got.entities.iter().zip(&want.entities).enumerate() {
                assert!(
                    g == w,
                    "{case}: canonical entity lists of dimension {d} differ"
                );
            }
        }
        for (d, local) in got.local_entities.iter().enumerate() {
            let known: HashSet<&Vec<i64>> = want.entities[d].iter().collect();
            if let Some(e) = local.iter().find(|e| !known.contains(e)) {
                panic!("{case}, rank {rank}: entity {e:?} of dimension {d} is not canonical");
            }
        }
        let bits: HashMap<i64, u32> = want.local_bits.iter().copied().collect();
        for &(id, b) in &got.local_bits {
            assert_eq!(
                Some(&b),
                bits.get(&id),
                "{case}, rank {rank}: permutation bits of cell {id}"
            );
        }
    }
}

/// Runs the parallel-invariance check of `case` on `comm`.
pub fn invariance<C: Comm>(comm: &C, client: &Client, case: &Case) {
    dispatch(client, case, &mut Invariance { comm, client });
}

/// Every index map of `mesh`: cell blocks, vertices, nodes and entity blocks, with names.
fn index_maps<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<Vec<(String, &IndexMap)>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut maps: Vec<(String, &IndexMap)> = mesh
        .cell_blocks()
        .iter()
        .map(|b| (format!("cells {:?}", b.cell.into()), &b.index_map))
        .collect();
    maps.push(("vertices".into(), mesh.vertex_index_map()));
    maps.push(("nodes".into(), mesh.node_index_map()));
    for d in 0..TD::N {
        for b in mesh.entity_blocks(d)? {
            maps.push((format!("entities {d} {:?}", b.cell), &b.index_map));
        }
    }
    Ok(maps)
}

/// For every owned index of `map`, the number of ranks holding it: 1 plus the ghosts of it on
/// other ranks, counted on the host by an all-to-all of the ghosts to their owners.
///
/// # Errors
///
/// Errors of `CommExt::all_to_all_v`.
pub fn holders(comm: &dyn CommDyn, map: &IndexMap) -> Result<Vec<i32>> {
    let mut by_owner: Vec<Vec<i64>> = vec![Vec::new(); comm.size()];
    for (&g, &owner) in map.ghosts().iter().zip(map.ghost_owners()) {
        by_owner[owner as usize].push(g);
    }
    let counts: Vec<usize> = by_owner.iter().map(Vec::len).collect();
    let send: Vec<i64> = by_owner.concat();
    let (received, _) = comm.all_to_all_v(&send, &counts)?;
    let begin = map.owned_range().0;
    let mut out = vec![1; map.num_owned()];
    for g in received {
        out[(g - begin) as usize] += 1;
    }
    Ok(out)
}

/// The forward scatter of the global ids of `map` at `block` values per index; returns the
/// scattered array.
fn forward_global_ids(client: &Client, map: &IndexMap, block: usize) -> Result<Vec<i32>> {
    let owned = map.num_owned();
    let begin = map.owned_range().0;
    let mut host = vec![-1_i32; map.num_local() * block];
    for i in 0..owned {
        for j in 0..block {
            host[i * block + j] = ((begin + i as i64) * block as i64 + j as i64) as i32;
        }
    }
    let mut x = DeviceBuffer::from_host(client, &host)?;
    map.scatterer(client, block)?.forward(&mut x)?;
    x.to_host()
}

/// The reverse `Add` of ones over `map` at `block` values per index; returns the result.
fn reverse_add_ones(client: &Client, map: &IndexMap, block: usize) -> Result<Vec<i32>> {
    let mut x = DeviceBuffer::from_host(client, &vec![1_i32; map.num_local() * block])?;
    map.scatterer(client, block)?
        .reverse(&mut x, ReverseOp::Add)?;
    x.to_host()
}

/// The scatter acceptance check of one case on `comm`.
pub struct ScatterCheck<'a, C: Comm> {
    /// The communicator of the build.
    pub comm: &'a C,
    /// The client of this rank.
    pub client: &'a Client,
}

impl<C: Comm> MeshCheck for ScatterCheck<'_, C> {
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let rank = self.comm.rank();
        let at = format!("{case}, rank {rank}");
        let mesh = ok(
            build::<F, TD, GD, _>(
                self.comm.clone(),
                self.client,
                host,
                case,
                CellOrdering::Input,
            ),
            &at,
        );
        for (name, map) in ok(index_maps(&mesh), &at) {
            let owned = map.num_owned();
            let want_holders = ok(holders(self.comm, map), &at);
            for block in [1, 3] {
                let x = ok(forward_global_ids(self.client, map, block), &at);
                for (k, &g) in map.ghosts().iter().enumerate() {
                    for j in 0..block {
                        assert_eq!(
                            i64::from(x[(owned + k) * block + j]),
                            g * block as i64 + j as i64,
                            "{case}, rank {rank}: {name}: forward of ghost {k}, block {block}"
                        );
                    }
                }
                let y = ok(reverse_add_ones(self.client, map, block), &at);
                for i in 0..map.num_local() {
                    let want = if i < owned { want_holders[i] } else { 1 };
                    for j in 0..block {
                        assert_eq!(
                            y[i * block + j],
                            want,
                            "{case}, rank {rank}: {name}: reverse Add at index {i}, block {block}"
                        );
                    }
                }
            }
        }
    }
}

/// Runs the scatter check of `case` on `comm`.
pub fn scatter<C: Comm>(comm: &C, client: &Client, case: &Case) {
    dispatch(client, case, &mut ScatterCheck { comm, client });
}

/// Every device array of `mesh`, and the results of a forward scatter of global ids and a
/// reverse `Add` of ones over every index map, as named words.
///
/// # Errors
///
/// Device and communication errors.
pub fn fingerprint<F, TD, GD, C>(
    client: &Client,
    mesh: &Mesh<F, TD, GD, C>,
) -> Result<Vec<(String, Vec<u64>)>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let words = |v: Vec<u32>| v.into_iter().map(u64::from).collect::<Vec<_>>();
    let ints = |v: Vec<i32>| v.into_iter().map(|x| x as u32 as u64).collect::<Vec<_>>();
    let mut out = Vec::new();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        out.push((format!("cells {b}"), words(block.vertices.data.to_host()?)));
        out.push((format!("cell ids {b}"), words(block.origin_ids.to_host()?)));
        out.push((format!("cell flags {b}"), words(block.flags.to_host()?)));
        if let Some(bits) = mesh.permutation_info(b) {
            out.push((format!("permutation {b}"), words(bits.to_host()?)));
        }
        out.push((
            format!("dofmap {b}"),
            words(mesh.geometry_block(b).dofmap.data.to_host()?),
        ));
    }
    out.push((
        "nodes".into(),
        mesh.nodes()
            .to_host()?
            .into_iter()
            .map(|x| x.widen().to_bits())
            .collect(),
    ));
    out.push((
        "origin".into(),
        mesh.origin().as_ref().iter().map(|x| x.to_bits()).collect(),
    ));
    out.push((
        "vertex ids".into(),
        words(mesh.vertex_origin_ids().to_host()?),
    ));
    for d in 0..TD::N {
        for (b, block) in mesh.entity_blocks(d)?.iter().enumerate() {
            out.push((
                format!("entities {d} {b}"),
                words(block.vertices.data.to_host()?),
            ));
        }
        if d > 0 {
            let key = ConnKey {
                from: TD::N as u8,
                to: d as u8,
            };
            if let Connectivity::Downward(blocks) = mesh.connectivity(key)? {
                for (b, s) in blocks.iter().enumerate() {
                    out.push((format!("{key} {b}"), words(s.data.to_host()?)));
                }
            }
        }
    }
    for d in 0..=TD::N {
        if let Some(tags) = mesh.tags().physical(d) {
            out.push((format!("tag indices {d}"), words(tags.indices.to_host()?)));
            out.push((format!("tag values {d}"), ints(tags.values.to_host()?)));
        }
    }
    for (name, map) in index_maps(mesh)? {
        let (begin, end) = map.owned_range();
        let mut m = vec![begin as u64, end as u64, map.size_global() as u64];
        m.extend(map.ghosts().iter().map(|&g| g as u64));
        m.extend(map.ghost_owners().iter().map(|&r| r as u64));
        out.push((format!("{name} map"), m));
        out.push((
            format!("{name} forward"),
            ints(forward_global_ids(client, map, 1)?),
        ));
        out.push((
            format!("{name} reverse"),
            ints(reverse_add_ones(client, map, 1)?),
        ));
    }
    Ok(out)
}

/// The determinism check of one case on `comm`.
pub struct Determinism<'a, C: Comm> {
    /// The communicator of the builds.
    pub comm: &'a C,
    /// The client of this rank.
    pub client: &'a Client,
}

impl<C: Comm> MeshCheck for Determinism<'_, C> {
    fn check<F, TD, GD>(&mut self, case: &Case, host: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let rank = self.comm.rank();
        let at = format!("{case}, rank {rank}");
        let run = || {
            let mesh = build::<F, TD, GD, _>(
                self.comm.clone(),
                self.client,
                host,
                case,
                CellOrdering::Hilbert,
            )?;
            fingerprint(self.client, &mesh)
        };
        let first = ok(run(), &at);
        let second = ok(run(), &at);
        assert_eq!(
            first.len(),
            second.len(),
            "{case}, rank {rank}: array count"
        );
        for ((name, a), (_, b)) in first.iter().zip(&second) {
            assert!(a == b, "{case}, rank {rank}: {name} differs between runs");
        }
    }
}

/// Runs the determinism check of `case` on `comm`.
pub fn determinism<C: Comm>(comm: &C, client: &Client, case: &Case) {
    dispatch(client, case, &mut Determinism { comm, client });
}

/// The mean over owned cells of all ranks of (max − min local node index in the cell).
///
/// # Errors
///
/// Device errors.
pub fn mean_spread<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<f64>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let (mut sum, mut cells) = (0_u64, 0_u64);
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let dofmap = &mesh.geometry_block(b).dofmap;
        let rows = dofmap.data.to_host()?;
        for row in rows
            .chunks_exact(dofmap.stride as usize)
            .take(block.index_map.num_owned())
        {
            let (lo, hi) = row
                .iter()
                .fold((u32::MAX, 0), |(lo, hi), &n| (lo.min(n), hi.max(n)));
            sum += u64::from(hi - lo);
            cells += 1;
        }
    }
    let comm = mesh.comm();
    Ok(comm.sum(sum) as f64 / comm.sum(cells).max(1) as f64)
}

/// The locality check on `comm`: on a box of hexahedra or tetrahedra at p = 1 and 2, with
/// shuffled cells and nodes and at least 64 cells per rank, Hilbert ordering at least halves
/// the mean spread of [`mean_spread`] against input ordering.
pub fn locality<C: Comm>(comm: &C, client: &Client) {
    let rank = comm.rank();
    let size = comm.size();
    for cell in [Cell3::Hexahedron, Cell3::Tetrahedron] {
        let per_cube = if cell == Cell3::Hexahedron { 1 } else { 6 };
        let n = (1..)
            .find(|&n: &usize| per_cube * n * n * n >= 64 * size)
            .expect("a box size");
        for p in 1..=2 {
            let host = crate::cube_box(cell, n, p)
                .with_shuffled_cells(11)
                .with_shuffled_nodes(13);
            let case = Case {
                family: Family::Cube(cell),
                degree: p,
                ghosts: GhostMode::None,
                distribution: Distribution::Even,
                double: false,
            };
            let spread = |ordering| {
                let mesh = build::<f32, D3, D3, _>(comm.clone(), client, &host, &case, ordering)?;
                mean_spread(&mesh)
            };
            let at = format!("locality {cell:?} p={p} n={n}, rank {rank}");
            let input = ok(spread(CellOrdering::Input), &at);
            let hilbert = ok(spread(CellOrdering::Hilbert), &at);
            assert!(
                hilbert <= 0.5 * input,
                "locality {cell:?} p={p} n={n}, rank {rank}: Hilbert spread {hilbert} is more \
                 than half the input spread {input}"
            );
        }
    }
}
