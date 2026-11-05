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
//! - [`submesh`] (M9): boundary, facet, cell and edge submeshes equal a one-rank build's, with a
//!   consistent `EntityMap`, outward normals and agreed errors.
//! - [`orient`] (M9): `orient` gives the flags of a one-rank build, the requested direction and
//!   agreed errors.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use cubecl::prelude::Client;
use gridcl::{
    Cell2, Cell3, CellOrdering, Codim1, Comm, CommDyn, CommExt, CommScalar, ConnKey, Connectivity,
    D1, D2, D3, DeviceBuffer, DeviceGrid, Dim, DistGraph, EmbeddedIn, EntityMap, GhostMode,
    GridError, IndexMap, Mesh, MeshTags, OrientationRule, Partitioner, QuadRule, Real, Result,
    ReverseOp, Serial, TopoDim,
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
    /// `mixed_surface`, one refinement (M9).
    MixedSurface,
    /// `circle` of 12 intervals (M9).
    Circle,
    /// `hemisphere`, one refinement: an open surface (M9).
    Hemisphere,
    /// `mobius_strip` of 12 squares, P1 at every degree: non-orientable (M9).
    Mobius,
    /// `book(3)`, P1 at every degree: non-manifold (M9).
    Book,
    /// `y_junction`: a non-manifold curve (M9).
    YJunction,
    /// `ball`, one refinement: a curved volume centred at the origin (M9).
    Ball,
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
            Family::MixedSurface => crate::mixed_surface(1, p),
            Family::Circle => crate::circle(12, p),
            Family::Hemisphere => crate::hemisphere(1, p),
            Family::Mobius => crate::mobius_strip(12),
            Family::Book => crate::book(3),
            Family::YJunction => crate::y_junction(p),
            Family::Ball => crate::ball(1, p),
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

/// The cases of the M9 submesh checks: every family with facets of a lower `TopoDim` (the
/// boxes, the mixed box, both squares, `sphere`, whose boundary is empty, and `ball`), at p = 1
/// and 2. `Full` runs every ghost mode; `Quick` one per case, cycling, with distributions
/// cycling and every other case in f32.
pub fn submesh_cases(coverage: Coverage) -> Vec<Case> {
    const SUBMESH_FAMILIES: [Family; 9] = [
        Family::Cube(Cell3::Tetrahedron),
        Family::Cube(Cell3::Hexahedron),
        Family::Cube(Cell3::Prism),
        Family::Cube(Cell3::Pyramid),
        Family::Mixed,
        Family::Square(Cell2::Triangle),
        Family::Square(Cell2::Quadrilateral),
        Family::Sphere,
        Family::Ball,
    ];
    m9_cases(&SUBMESH_FAMILIES, coverage)
}

/// The cases of the M9 `orient` checks: the orientable closed families (`sphere`,
/// `mixed_surface`, `circle`; the checks flip a random subset of their cells), the open
/// `hemisphere`, and the error families `mobius_strip`, `book` and `y_junction`, at p = 1 and 2
/// (as [`submesh_cases`]).
pub fn orient_cases(coverage: Coverage) -> Vec<Case> {
    const ORIENT_FAMILIES: [Family; 7] = [
        Family::Sphere,
        Family::MixedSurface,
        Family::Circle,
        Family::Hemisphere,
        Family::Mobius,
        Family::Book,
        Family::YJunction,
    ];
    m9_cases(&ORIENT_FAMILIES, coverage)
}

/// The cases of `families` at p = 1 and 2: every ghost mode (`Full`) or one, cycling (`Quick`);
/// distributions cycle, and every other case runs in f32.
fn m9_cases(families: &[Family], coverage: Coverage) -> Vec<Case> {
    let mut out = Vec::new();
    for (f, &family) in families.iter().enumerate() {
        for degree in 1..=2 {
            for (g, &ghosts) in GHOST_MODES.iter().enumerate() {
                let k = f + degree as usize;
                if coverage == Coverage::Full || g == k % GHOST_MODES.len() {
                    out.push(Case {
                        family,
                        degree,
                        ghosts,
                        distribution: DISTRIBUTIONS[(k + g) % 3],
                        double: (k + g).is_multiple_of(2),
                    });
                }
            }
        }
    }
    out
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
pub(crate) fn decode_ids(words: &[u32]) -> Vec<i64> {
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
        Family::Cube(_) | Family::Mixed | Family::Ball => run!(D3, D3),
        Family::Square(_) => run!(D2, D2),
        Family::Interval => run!(D1, D1),
        Family::Circle | Family::YJunction => run!(D1, D2),
        Family::Sphere
        | Family::MixedSurface
        | Family::Hemisphere
        | Family::Mobius
        | Family::Book => run!(D2, D3),
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

/// The mean over owned cells of all ranks of (max − min local index of the cell's owned
/// nodes); ghost nodes, whose order (owner rank, global index) the cell ordering does not
/// control, are excluded, and a cell with fewer than two owned nodes counts 0 (ADR-0027). On
/// one rank every node is owned.
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
    let owned_nodes = u32::try_from(mesh.node_index_map().num_owned()).expect("u32 nodes");
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
                .filter(|&&n| n < owned_nodes)
                .fold((u32::MAX, 0), |(lo, hi), &n| (lo.min(n), hi.max(n)));
            sum += u64::from(hi.saturating_sub(lo));
            cells += 1;
        }
    }
    let comm = mesh.comm();
    Ok(comm.sum(sum) as f64 / comm.sum(cells).max(1) as f64)
}

/// The locality check on `comm`: on a box of hexahedra or tetrahedra at p = 1 and 2, with
/// shuffled cells and nodes, at least 8 cells per edge and at least 64 cells per rank, Hilbert
/// ordering at least halves the mean spread of [`mean_spread`] against input ordering
/// (ADR-0027).
///
/// The edge bound matters: the Hilbert spread grows like n², the shuffled one like n³, and
/// below 5 cells per edge the bound of 1/2 fails even on one rank (hexahedra, n = 4: 0.55).
pub fn locality<C: Comm>(comm: &C, client: &Client) {
    let rank = comm.rank();
    let size = comm.size();
    for cell in [Cell3::Hexahedron, Cell3::Tetrahedron] {
        let per_cube = if cell == Cell3::Hexahedron { 1 } else { 6 };
        let n = (8..)
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

// ---------------------------------------------------------------------------------------------
// M9: submeshes and orientation (docs/plans/M9.md, T0b; ADR-0039).

/// The sorted vertex origin ids of every local entity of dimension `dim` of `m`, by flat index:
/// the cells over the cell blocks for dim = tdim, else the entity blocks.
///
/// # Errors
///
/// `EntitiesNotCreated(dim)`; device errors.
pub fn entity_rows<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, dim: usize) -> Result<Vec<Vec<i64>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let ids = decode_ids(&m.vertex_origin_ids().to_host()?);
    let rows = |data: &DeviceBuffer<u32>, stride: u32| -> Result<Vec<Vec<i64>>> {
        Ok(data
            .to_host()?
            .chunks_exact(stride as usize)
            .map(|r| {
                let mut t: Vec<i64> = r.iter().map(|&v| ids[v as usize]).collect();
                t.sort_unstable();
                t
            })
            .collect())
    };
    let mut out = Vec::new();
    if dim == TD::N {
        for b in m.cell_blocks() {
            out.extend(rows(&b.vertices.data, b.vertices.stride)?);
        }
    } else {
        for b in m.entity_blocks(dim)? {
            out.extend(rows(&b.vertices.data, b.vertices.stride)?);
        }
    }
    Ok(out)
}

/// The origin id of every local cell of `m`, flat over the cell blocks.
fn cell_ids<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Result<Vec<i64>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut out = Vec::new();
    for b in m.cell_blocks() {
        out.extend(decode_ids(&b.origin_ids.to_host()?));
    }
    Ok(out)
}

/// Whether each local cell of `m`, flat over the cell blocks, is owned.
fn owned_cells<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Vec<bool>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    m.cell_blocks()
        .iter()
        .flat_map(|b| (0..b.index_map.num_local()).map(|c| c < b.index_map.num_owned()))
        .collect()
}

/// The physical tag values of every local entity of dimension `dim`, by flat index.
fn tag_values<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, dim: usize) -> Result<HashMap<u32, Vec<i32>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut out: HashMap<u32, Vec<i32>> = HashMap::new();
    if let Some(t) = m.tags().physical(dim) {
        for (i, v) in t.indices.to_host()?.into_iter().zip(t.values.to_host()?) {
            out.entry(i).or_default().push(v);
        }
    }
    Ok(out)
}

/// The vertex centroid of every local cell of `m` (absolute coordinates), flat over the
/// cell blocks.
///
/// # Errors
///
/// Device errors.
pub fn cell_centroids<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Result<Vec<Vec<f64>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let nodes = m.nodes().to_host()?;
    let origin = m.origin();
    let mut out = Vec::new();
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let nv = gridcl_refcell::vertices(block.cell.into()).len();
        let dofmap = &m.geometry_block(b).dofmap;
        for row in dofmap.data.to_host()?.chunks_exact(dofmap.stride as usize) {
            out.push(
                (0..gd)
                    .map(|g| {
                        row[..nv]
                            .iter()
                            .map(|&n| nodes[n as usize * gd + g].widen())
                            .sum::<f64>()
                            / nv as f64
                            + origin.as_ref()[g]
                    })
                    .collect(),
            );
        }
    }
    Ok(out)
}

/// Device tags of dimension `dim` with value 1 on the local entities `indices` (ascending).
fn tags_of(client: &Client, dim: usize, indices: &[u32]) -> Result<MeshTags> {
    Ok(MeshTags {
        dim,
        indices: DeviceBuffer::from_host(client, indices)?,
        values: DeviceBuffer::from_host(client, &vec![1_i32; indices.len()])?,
    })
}

/// What a partition must not change on a submesh.
#[derive(Debug, Default, PartialEq)]
pub struct ChildSummary {
    /// (cell type, global count) of every child cell block, in order.
    pub blocks: Vec<(CellType, i64)>,
    /// The child's measure.
    pub measure: f64,
    /// Rank 0: (origin id, sorted vertex origin ids) of the owned child cells of all ranks,
    /// sorted.
    pub cells: Vec<Vec<i64>>,
    /// This rank: (origin id, sorted vertex origin ids) of every local child cell.
    pub local: Vec<Vec<i64>>,
}

/// The [`ChildSummary`] of `child`, with Gauss degree `degree` for the measure.
///
/// # Errors
///
/// Device and communication errors.
pub fn child_summary<F, TD, GD, C>(child: &Mesh<F, TD, GD, C>, degree: u32) -> Result<ChildSummary>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let rows = entity_rows(child, TD::N)?;
    let ids = cell_ids(child)?;
    let owned = owned_cells(child);
    let mut s = ChildSummary {
        blocks: child
            .cell_blocks()
            .iter()
            .map(|b| (b.cell.into(), b.index_map.size_global()))
            .collect(),
        measure: child.measure(degree)?,
        ..ChildSummary::default()
    };
    let mut flat = Vec::new();
    for ((id, row), owned) in ids.iter().zip(&rows).zip(owned) {
        let entry: Vec<i64> = std::iter::once(*id).chain(row.iter().copied()).collect();
        if owned {
            push_row(&mut flat, &entry);
        }
        s.local.push(entry);
    }
    s.cells = split_rows(&gather_to_root(child.comm(), &flat)?);
    s.cells.sort_unstable();
    Ok(s)
}

/// Asserts that the parallel child summary `got` equals the one-rank `want`.
fn assert_same_child(got: &ChildSummary, want: &ChildSummary, rank: usize, tol: f64, at: &str) {
    assert_eq!(got.blocks, want.blocks, "{at}: child blocks");
    let scale = want.measure.abs().max(1.0);
    assert!(
        (got.measure - want.measure).abs() <= tol * scale,
        "{at}: child measure {} vs {}",
        got.measure,
        want.measure
    );
    if rank == 0 {
        assert!(got.cells == want.cells, "{at}: owned child cells differ");
    }
    let known: HashSet<&Vec<i64>> = want.local.iter().collect();
    if let Some(c) = got.local.iter().find(|c| !known.contains(c)) {
        panic!("{at}: child cell {c:?} (origin id, vertices) is not the one-rank build's");
    }
}

/// Asserts that `map` is consistent between `parent` and `child`: child cells have the vertices
/// of their parent entities, the sparse inverse inverts `child_to_parent`, child vertices and
/// nodes are their parents', and the child keeps the parent's origin.
///
/// # Errors
///
/// Device errors.
pub fn check_entity_map<F, TD, GD, CTD, C>(
    parent: &Mesh<F, TD, GD, C>,
    child: &Mesh<F, CTD, GD, C>,
    map: &EntityMap,
    at: &str,
) -> Result<()>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    CTD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let parent_rows = entity_rows(parent, map.parent_dim())?;
    let child_rows = entity_rows(child, CTD::N)?;
    let c2p = map.child_to_parent().to_host()?;
    assert_eq!(c2p.len(), child_rows.len(), "{at}: child_to_parent length");
    for (c, &p) in c2p.iter().enumerate() {
        assert_eq!(
            child_rows[c], parent_rows[p as usize],
            "{at}: child cell {c} and parent entity {p}"
        );
    }
    let (parents, children) = map.parent_to_child();
    let (parents, children) = (parents.to_host()?, children.to_host()?);
    assert_eq!(parents.len(), c2p.len(), "{at}: parent_to_child length");
    assert!(
        parents.windows(2).all(|w| w[0] < w[1]),
        "{at}: parent_to_child indices not ascending"
    );
    for (&p, &c) in parents.iter().zip(&children) {
        assert_eq!(c2p[c as usize], p, "{at}: parent_to_child of {p}");
    }
    let parent_ids = decode_ids(&parent.vertex_origin_ids().to_host()?);
    let child_ids = decode_ids(&child.vertex_origin_ids().to_host()?);
    let v2p = map.vertex_to_parent().to_host()?;
    assert_eq!(v2p.len(), child_ids.len(), "{at}: vertex_to_parent length");
    for (v, &p) in v2p.iter().enumerate() {
        assert_eq!(
            child_ids[v], parent_ids[p as usize],
            "{at}: child vertex {v}"
        );
    }
    let gd = GD::N;
    let parent_nodes = parent.nodes().to_host()?;
    let child_nodes = child.nodes().to_host()?;
    let n2p = map.node_to_parent().to_host()?;
    assert_eq!(
        n2p.len() * gd,
        child_nodes.len(),
        "{at}: node_to_parent length"
    );
    for (n, &p) in n2p.iter().enumerate() {
        for g in 0..gd {
            assert!(
                child_nodes[n * gd + g].widen().to_bits()
                    == parent_nodes[p as usize * gd + g].widen().to_bits(),
                "{at}: child node {n} is not bitwise parent node {p}"
            );
        }
    }
    let bits = |o: GD::Array<f64>| o.as_ref().iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(child.origin()),
        bits(parent.origin()),
        "{at}: child origin"
    );
    Ok(())
}

/// The smallest n·(x − centre) over every local cell and Gauss point of degree `degree` of the
/// codimension-1 mesh `m`, and ∮ x·n over its owned cells on all ranks, finished in f64.
///
/// # Errors
///
/// Device and communication errors.
pub fn normals_and_flux<F, TD, GD, C>(
    m: &Mesh<F, TD, GD, C>,
    centre: &[f64],
    degree: u32,
) -> Result<(f64, f64)>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let rule = QuadRule::for_grid(m, degree)?;
    let (x, n, detj) = (m.physical_points(&rule)?, m.normals(&rule)?, m.detj(&rule)?);
    let (mut min, mut flux) = (f64::INFINITY, 0.0);
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let weights = rule.get(block.cell)?.weights();
        let nq = weights.len();
        let x = x.blocks[b].to_host_cell_major()?;
        let n = n.blocks[b].to_host_cell_major()?;
        let detj = detj.blocks[b].to_host_cell_major()?;
        for c in 0..block.index_map.num_local() {
            for (q, w) in weights.iter().enumerate() {
                let i = c * nq + q;
                let (xi, ni) = (x[i].as_ref(), n[i].as_ref());
                let mut away = 0.0;
                let mut xn = 0.0;
                for g in 0..GD::N {
                    away += (xi[g].widen() - centre[g]) * ni[g].widen();
                    xn += xi[g].widen() * ni[g].widen();
                }
                min = min.min(away);
                if c < block.index_map.num_owned() {
                    flux += w * detj[i].widen() * xn;
                }
            }
        }
    }
    Ok((min, m.comm().sum(flux)))
}

/// The (TD, GD) pairs of the submesh families, with what only some of them have: normals on
/// the codimension-1 children of codimension-0 meshes, and edge submeshes of volume meshes.
pub trait SubmeshDims<GD: Dim>: TopoDim + EmbeddedIn<GD> {
    /// [`normals_and_flux`] of `child` if it has normals.
    ///
    /// # Errors
    ///
    /// As [`normals_and_flux`].
    fn child_normals<F: Real, C: Comm>(
        child: &Mesh<F, Self::Lower, GD, C>,
        centre: &[f64],
        degree: u32,
    ) -> Result<Option<(f64, f64)>>
    where
        Self::Lower: TopoDim + EmbeddedIn<GD>;

    /// On a volume mesh: creates the (1, 3) connectivity and returns the edge submesh of the
    /// edges whose smallest vertex origin id is even, checked against its `EntityMap`, as a
    /// [`ChildSummary`]; `None` otherwise.
    ///
    /// # Errors
    ///
    /// Errors of `create_connectivity`, `edge_submesh` and [`child_summary`].
    fn edges<F: Real, C: Comm>(
        mesh: &mut Mesh<F, Self, GD, C>,
        degree: u32,
        at: &str,
    ) -> Result<Option<ChildSummary>>;
}

impl SubmeshDims<D3> for D3 {
    fn child_normals<F: Real, C: Comm>(
        child: &Mesh<F, D2, D3, C>,
        centre: &[f64],
        degree: u32,
    ) -> Result<Option<(f64, f64)>> {
        normals_and_flux(child, centre, degree).map(Some)
    }

    fn edges<F: Real, C: Comm>(
        mesh: &mut Mesh<F, D3, D3, C>,
        degree: u32,
        at: &str,
    ) -> Result<Option<ChildSummary>> {
        mesh.create_connectivity(1, 3)?;
        let even: Vec<u32> = entity_rows(mesh, 1)?
            .iter()
            .enumerate()
            .filter(|(_, row)| row[0] % 2 == 0)
            .map(|(e, _)| e as u32)
            .collect();
        let (child, map) = mesh.edge_submesh(&tags_of(mesh.client(), 1, &even)?)?;
        check_entity_map(mesh, &child, &map, &format!("{at}: edge submesh"))?;
        assert!(
            child
                .cell_blocks()
                .iter()
                .all(|b| b.flags.to_host().is_ok_and(|f| f.iter().all(|&f| f == 0))),
            "{at}: edge submesh flags are not 0"
        );
        child_summary(&child, degree).map(Some)
    }
}

impl SubmeshDims<D2> for D2 {
    fn child_normals<F: Real, C: Comm>(
        child: &Mesh<F, D1, D2, C>,
        centre: &[f64],
        degree: u32,
    ) -> Result<Option<(f64, f64)>> {
        normals_and_flux(child, centre, degree).map(Some)
    }

    fn edges<F: Real, C: Comm>(
        _mesh: &mut Mesh<F, D2, D2, C>,
        _degree: u32,
        _at: &str,
    ) -> Result<Option<ChildSummary>> {
        Ok(None)
    }
}

impl SubmeshDims<D3> for D2 {
    fn child_normals<F: Real, C: Comm>(
        _child: &Mesh<F, D1, D3, C>,
        _centre: &[f64],
        _degree: u32,
    ) -> Result<Option<(f64, f64)>> {
        Ok(None)
    }

    fn edges<F: Real, C: Comm>(
        _mesh: &mut Mesh<F, D2, D3, C>,
        _degree: u32,
        _at: &str,
    ) -> Result<Option<ChildSummary>> {
        Ok(None)
    }
}

/// The midpoint of the bounding box of `host`'s nodes.
fn bbox_centre(host: &HostMesh) -> Vec<f64> {
    (0..host.gdim)
        .map(|g| {
            let (lo, hi) = host
                .nodes
                .iter()
                .skip(g)
                .step_by(host.gdim)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                    (lo.min(x), hi.max(x))
                });
            0.5 * (lo + hi)
        })
        .collect()
}

/// The submeshes of one mesh: boundary, the facets tagged 100, the cells left of the centre,
/// and (volume meshes) the edges of [`SubmeshDims::edges`].
struct Submeshes {
    boundary: ChildSummary,
    normals: Option<(f64, f64)>,
    tagged: ChildSummary,
    cells: ChildSummary,
    edges: Option<ChildSummary>,
}

/// The submeshes of `mesh` with every check that needs no one-rank reference: the `EntityMap`s,
/// child tags, flags of codimension-2 children, a forward scatter of global ids over every child
/// map, entity creation on the boundary child, and `InteriorFacet` (agreed) when every facet is
/// tagged.
fn submeshes<F, TD, GD, C>(
    client: &Client,
    mesh: &mut Mesh<F, TD, GD, C>,
    centre: &[f64],
    degree: u32,
    at: &str,
) -> Result<Submeshes>
where
    F: Real,
    TD: SubmeshDims<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let fdim = TD::N - 1;

    // boundary(): the EntityMap, tags carried as cell tags, flags 0 in codimension 2.
    let (mut child, map) = mesh.boundary()?;
    check_entity_map(mesh, &child, &map, &format!("{at}: boundary"))?;
    let parent_tags = tag_values(mesh, fdim)?;
    let child_tags = tag_values(&child, fdim)?;
    for (c, p) in map.child_to_parent().to_host()?.into_iter().enumerate() {
        assert_eq!(
            child_tags.get(&(c as u32)),
            parent_tags.get(&p),
            "{at}: boundary: tags of child cell {c} (parent facet {p})"
        );
    }
    if GD::N - TD::N == 1 {
        for b in child.cell_blocks() {
            assert!(
                b.flags.to_host()?.iter().all(|&f| f == 0),
                "{at}: boundary of a codimension-1 mesh has flags"
            );
        }
    }
    let normals = TD::child_normals(&child, centre, 3 * degree)?;
    let boundary = child_summary(&child, 3 * degree)?;
    for d in 0..TD::N - 1 {
        child.create_entities(d)?;
    }
    for (name, map) in index_maps(&child)? {
        let x = forward_global_ids(client, map, 1)?;
        let owned = map.num_owned();
        for (k, &g) in map.ghosts().iter().enumerate() {
            assert_eq!(
                i64::from(x[owned + k]),
                g,
                "{at}: boundary child {name}: forward of ghost {k}"
            );
        }
    }

    // facet_submesh of the facets tagged 100: every child cell carries 100.
    let (child, map) = mesh.facet_submesh(&mesh.tags().group(fdim, 100)?)?;
    check_entity_map(mesh, &child, &map, &format!("{at}: facet_submesh"))?;
    let child_tags = tag_values(&child, fdim)?;
    for c in 0..map.child_to_parent().len() {
        assert!(
            child_tags
                .get(&(c as u32))
                .is_some_and(|v| v.contains(&100)),
            "{at}: facet_submesh: child cell {c} lacks tag 100"
        );
    }
    let tagged = child_summary(&child, 3 * degree)?;

    // Every facet tagged: InteriorFacet on every rank.
    let num_facets: usize = mesh
        .entity_blocks(fdim)?
        .iter()
        .map(|b| b.index_map.num_local())
        .sum();
    let all: Vec<u32> = (0..num_facets as u32).collect();
    match mesh.facet_submesh(&tags_of(client, fdim, &all)?) {
        Err(GridError::InteriorFacet) => {}
        Err(e) => panic!("{at}: every facet tagged: {e}, not InteriorFacet"),
        Ok(_) => panic!("{at}: every facet tagged: no error, not InteriorFacet"),
    }

    // cell_submesh of the cells whose centroid lies left of the centre.
    let left: Vec<u32> = cell_centroids(mesh)?
        .iter()
        .enumerate()
        .filter(|(_, x)| x[0] < centre[0])
        .map(|(c, _)| c as u32)
        .collect();
    let (child, map) = mesh.cell_submesh(&tags_of(client, TD::N, &left)?)?;
    check_entity_map(mesh, &child, &map, &format!("{at}: cell_submesh"))?;
    let cells = child_summary(&child, 3 * degree)?;

    let edges = TD::edges(mesh, 3 * degree, at)?;
    Ok(Submeshes {
        boundary,
        normals,
        tagged,
        cells,
        edges,
    })
}

/// The submesh check of one case on `comm`, in scalar `F`.
fn submesh_case<F, TD, GD, C>(comm: &C, client: &Client, case: &Case, host: &HostMesh)
where
    F: Real,
    TD: SubmeshDims<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let rank = comm.rank();
    let at = format!("{case}, rank {rank}");
    let centre = bbox_centre(host);
    let tol = if size_of::<F>() == 8 { 1e-10 } else { TOL_F32 };
    let mut serial = ok(
        build::<F, TD, GD, _>(Serial, client, host, case, CellOrdering::Input),
        &at,
    );
    let want = ok(
        submeshes(
            client,
            &mut serial,
            &centre,
            case.degree,
            &format!("{at} (one rank)"),
        ),
        &at,
    );
    let mut mesh = ok(
        build::<F, TD, GD, _>(comm.clone(), client, host, case, CellOrdering::Input),
        &at,
    );
    let got = ok(submeshes(client, &mut mesh, &centre, case.degree, &at), &at);

    assert_same_child(
        &got.boundary,
        &want.boundary,
        rank,
        tol,
        &format!("{at}: boundary"),
    );
    assert_same_child(
        &got.tagged,
        &want.tagged,
        rank,
        tol,
        &format!("{at}: facet_submesh"),
    );
    assert_same_child(
        &got.cells,
        &want.cells,
        rank,
        tol,
        &format!("{at}: cell_submesh"),
    );
    match (&got.edges, &want.edges) {
        (Some(g), Some(w)) => assert_same_child(g, w, rank, tol, &format!("{at}: edge_submesh")),
        (None, None) => {}
        _ => panic!("{at}: edge submeshes on one side only"),
    }
    if let Some((min, flux)) = got.normals {
        assert!(
            min > 0.0,
            "{at}: a boundary normal points inward (n·(x − c) = {min:e})"
        );
        let volume = ok(mesh.measure(3 * case.degree), &at);
        let want = GD::N as f64 * volume;
        assert!(
            (flux - want).abs() <= tol * want.abs().max(1.0),
            "{at}: ∮ x·n = {flux} vs gdim · V = {want}"
        );
    }
}

/// The owned vertex origin ids of `m` on all ranks, sorted, on rank 0 (empty elsewhere).
fn owned_vertex_ids<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Result<Vec<i64>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let ids = decode_ids(&m.vertex_origin_ids().to_host()?);
    let owned = &ids[..m.vertex_index_map().num_owned()];
    let mut all = gather_to_root(m.comm(), owned)?;
    all.sort_unstable();
    Ok(all)
}

/// The topology of a child on `comm` against the one-rank child `want`: the cell summary, the
/// vertex count, each vertex owned once, ghosts that receive their global ids by a forward
/// scatter over every map, vertex sharer rows that list the holders with an owned-cell holder
/// as owner, and the entities of every lower dimension.
#[allow(clippy::too_many_arguments)]
fn child_topology_matches<F, TD, GD, C>(
    client: &Client,
    child: &mut Mesh<F, TD, GD, C>,
    want: &mut Mesh<F, TD, GD, Serial>,
    degree: u32,
    rank: usize,
    tol: f64,
    at: &str,
) -> Result<()>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    assert_same_child(
        &child_summary(child, degree)?,
        &child_summary(want, degree)?,
        rank,
        tol,
        at,
    );
    let map = child.vertex_index_map();
    assert_eq!(
        map.size_global() as usize,
        want.vertex_index_map().num_local(),
        "{at}: child vertex count"
    );
    let owned = owned_vertex_ids(child)?;
    if rank == 0 {
        let mut all = decode_ids(&want.vertex_origin_ids().to_host()?);
        all.sort_unstable();
        assert!(
            owned == all,
            "{at}: owned child vertices are not each vertex once"
        );
    }
    let me = rank as i32;
    let held = holders(child.comm(), map)?;
    let s = child
        .topology()
        .vertex_sharers()
        .expect("a child has vertex sharers");
    for v in 0..map.num_local() {
        let row = &s.ranks[s.offsets[v]..s.offsets[v + 1]];
        let owner = if v < map.num_owned() {
            me
        } else {
            map.ghost_owners()[v - map.num_owned()]
        };
        let k = row.iter().position(|&r| r == owner);
        assert!(
            k.is_some_and(|k| s.owned_cell[s.offsets[v] + k]),
            "{at}: child vertex {v}: owner {owner} is not an owned-cell sharer in {row:?}"
        );
        assert!(
            row.contains(&me),
            "{at}: child vertex {v}: row {row:?} lacks this rank"
        );
        if let Some(&h) = held.get(v) {
            assert_eq!(
                row.len(),
                h as usize,
                "{at}: child vertex {v}: sharer row {row:?} vs holders"
            );
        }
    }
    let sizes =
        |b: &[gridcl::EntityBlock]| b.iter().map(|b| b.index_map.size_global()).sum::<i64>();
    for d in 0..TD::N {
        child.create_entities(d)?;
        want.create_entities(d)?;
        assert_eq!(
            sizes(child.entity_blocks(d)?),
            sizes(want.entity_blocks(d)?),
            "{at}: child entities of dimension {d}"
        );
    }
    for (name, map) in index_maps(child)? {
        let x = forward_global_ids(client, map, 1)?;
        let owned = map.num_owned();
        for (k, &g) in map.ghosts().iter().enumerate() {
            assert_eq!(
                i64::from(x[owned + k]),
                g,
                "{at}: {name}: forward of ghost {k}"
            );
        }
    }
    Ok(())
}

/// The child topology check of one case on `comm`, in scalar `F`: `boundary()` and the
/// `facet_submesh` of the facets tagged 100 against a one-rank build's.
fn submesh_topology_case<F, TD, GD, C>(comm: &C, client: &Client, case: &Case, host: &HostMesh)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let rank = comm.rank();
    let at = format!("{case}, rank {rank}");
    let tol = if size_of::<F>() == 8 { 1e-10 } else { TOL_F32 };
    let fdim = TD::N - 1;
    let serial = ok(
        build::<F, TD, GD, _>(Serial, client, host, case, CellOrdering::Input),
        &at,
    );
    let mesh = ok(
        build::<F, TD, GD, _>(comm.clone(), client, host, case, CellOrdering::Input),
        &at,
    );
    let degree = 3 * case.degree;
    let (mut got, _) = ok(mesh.boundary(), &at);
    let (mut want, _) = ok(serial.boundary(), &at);
    ok(
        child_topology_matches(
            client,
            &mut got,
            &mut want,
            degree,
            rank,
            tol,
            &format!("{at}: boundary"),
        ),
        &at,
    );
    let tagged = |m: &Mesh<F, TD, GD, _>| m.tags().group(fdim, 100);
    let (mut got, _) = ok(mesh.facet_submesh(&ok(tagged(&mesh), &at)), &at);
    let (mut want, _) = ok(
        serial.facet_submesh(&ok(serial.tags().group(fdim, 100), &at)),
        &at,
    );
    ok(
        child_topology_matches(
            client,
            &mut got,
            &mut want,
            degree,
            rank,
            tol,
            &format!("{at}: facet_submesh"),
        ),
        &at,
    );
}

/// Runs the child topology check of `case` on `comm` (M9 T5a): the cells, vertices, ownership
/// and sharers of `boundary()` and `facet_submesh` equal a one-rank build's. [`submesh_cases`]
/// families only.
///
/// # Panics
///
/// If a check fails, or `case` is not a submesh family.
pub fn submesh_topology<C: Comm>(comm: &C, client: &Client, case: &Case) {
    let host = case.family.host(case.degree);
    let double = case.double && supports::<f64>(client);
    macro_rules! run {
        ($td:ty, $gd:ty) => {
            if double {
                submesh_topology_case::<f64, $td, $gd, C>(comm, client, case, &host)
            } else {
                submesh_topology_case::<f32, $td, $gd, C>(comm, client, case, &host)
            }
        };
    }
    match case.family {
        Family::Cube(_) | Family::Mixed | Family::Ball => run!(D3, D3),
        Family::Square(_) => run!(D2, D2),
        Family::Sphere => run!(D2, D3),
        other => panic!("{other:?} is not a submesh family"),
    }
}

/// Runs the M9 submesh check of `case` on `comm`: [`submesh_cases`] families only.
///
/// # Panics
///
/// If a check fails, or `case` is not a submesh family.
pub fn submesh<C: Comm>(comm: &C, client: &Client, case: &Case) {
    let host = case.family.host(case.degree);
    let double = case.double && supports::<f64>(client);
    macro_rules! run {
        ($td:ty, $gd:ty) => {
            if double {
                submesh_case::<f64, $td, $gd, C>(comm, client, case, &host)
            } else {
                submesh_case::<f32, $td, $gd, C>(comm, client, case, &host)
            }
        };
    }
    match case.family {
        Family::Cube(_) | Family::Mixed | Family::Ball => run!(D3, D3),
        Family::Square(_) => run!(D2, D2),
        Family::Sphere => run!(D2, D3),
        other => panic!("{other:?} is not a submesh family"),
    }
}

/// What `orient` must give for a family: per rule, success or the agreed error.
fn orient_expectations(family: Family) -> Vec<(OrientationRule, Option<GridError>)> {
    use OrientationRule::{AwayFrom, Outward, Towards};
    let origin = [0.0; 3];
    match family {
        Family::Sphere | Family::MixedSurface | Family::Circle => vec![(Outward, None)],
        Family::Hemisphere => vec![
            (Outward, Some(GridError::NeedsReferencePoint)),
            (AwayFrom(origin), None),
            (Towards(origin), None),
        ],
        Family::Mobius => vec![
            (Outward, Some(GridError::NonOrientable)),
            (AwayFrom(origin), Some(GridError::NonOrientable)),
        ],
        Family::Book | Family::YJunction => vec![(Outward, Some(GridError::NonManifold))],
        other => panic!("{other:?} is not an orient family"),
    }
}

/// `orient(rule)` on `mesh`; on success, (cell origin id, flag bit 0) of every local cell and
/// the smallest s · n·x over every local cell and Gauss point of degree 2p, with s = −1 for
/// `Towards`.
fn oriented<F, TD, GD, C>(
    mesh: &mut Mesh<F, TD, GD, C>,
    rule: OrientationRule,
    degree: u32,
) -> Result<(HashMap<i64, u32>, f64)>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    mesh.orient(rule)?;
    let mut flags = HashMap::new();
    for b in mesh.cell_blocks() {
        let ids = decode_ids(&b.origin_ids.to_host()?);
        flags.extend(
            ids.into_iter()
                .zip(b.flags.to_host()?.into_iter().map(|f| f & 1)),
        );
    }
    let sign = if matches!(rule, OrientationRule::Towards(_)) {
        -1.0
    } else {
        1.0
    };
    let rule = QuadRule::for_grid(mesh, 2 * degree)?;
    let (x, n) = (mesh.physical_points(&rule)?, mesh.normals(&rule)?);
    let mut min = f64::INFINITY;
    for (x, n) in x.blocks.iter().zip(&n.blocks) {
        for (x, n) in x.to_host_cell_major()?.iter().zip(n.to_host_cell_major()?) {
            let xn: f64 = (0..GD::N)
                .map(|g| x.as_ref()[g].widen() * n.as_ref()[g].widen())
                .sum();
            min = min.min(sign * xn);
        }
    }
    Ok((flags, min))
}

/// The orient check of one case on `comm`, in scalar `F`.
fn orient_case<F, TD, GD, C>(comm: &C, client: &Client, case: &Case, host: &HostMesh)
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let rank = comm.rank();
    let host = host.with_flipped_cells(17 + u64::from(case.degree), 0.5);
    for (rule, error) in orient_expectations(case.family) {
        let at = format!("{case}, {rule:?}, rank {rank}");
        let mut serial = ok(
            build::<F, TD, GD, _>(Serial, client, &host, case, CellOrdering::Input),
            &at,
        );
        let mut mesh = ok(
            build::<F, TD, GD, _>(comm.clone(), client, &host, case, CellOrdering::Input),
            &at,
        );
        let want = oriented(&mut serial, rule, case.degree);
        let got = oriented(&mut mesh, rule, case.degree);
        match (error, want, got) {
            (None, Ok((want, _)), Ok((got, min))) => {
                for (id, flag) in &got {
                    assert_eq!(Some(flag), want.get(id), "{at}: flag of cell {id}");
                }
                assert!(
                    min > 0.0,
                    "{at}: a normal has the wrong direction ({min:e})"
                );
            }
            (Some(e), Err(w), Err(g)) => {
                let same = |x: &GridError| std::mem::discriminant(x) == std::mem::discriminant(&e);
                assert!(same(&w), "{at}: one rank: {w}, not {e}");
                assert!(same(&g), "{at}: {g}, not {e}");
            }
            (e, w, g) => panic!(
                "{at}: expected {e:?}; one rank gave {:?}, this build {:?}",
                w.err(),
                g.err()
            ),
        }
    }
}

/// Runs the M9 orient check of `case` on `comm`: random cells of the host mesh are flipped,
/// `orient` gives the flags of a one-rank build on every local cell and the requested
/// direction, or the agreed error ([`orient_cases`] families only).
///
/// # Panics
///
/// If a check fails, or `case` is not an orient family.
pub fn orient<C: Comm>(comm: &C, client: &Client, case: &Case) {
    let host = case.family.host(case.degree);
    let double = case.double && supports::<f64>(client);
    macro_rules! run {
        ($td:ty, $gd:ty) => {
            if double {
                orient_case::<f64, $td, $gd, C>(comm, client, case, &host)
            } else {
                orient_case::<f32, $td, $gd, C>(comm, client, case, &host)
            }
        };
    }
    match case.family {
        Family::Circle | Family::YJunction => run!(D1, D2),
        Family::Sphere
        | Family::MixedSurface
        | Family::Hemisphere
        | Family::Mobius
        | Family::Book => run!(D2, D3),
        other => panic!("{other:?} is not an orient family"),
    }
}
