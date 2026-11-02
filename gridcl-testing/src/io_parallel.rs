//! The checks of the M8 acceptance tests (docs/plans/M8.md, T0b; ADR-0035), generic over the
//! communicator: gridcl/tests/io_threads.rs runs them on [`ThreadComm`](crate::ThreadComm) and
//! gridcl-mpi-tests on `Mpi`.
//!
//! A check panics with the case, the rank and what differs; under MPI the panic aborts the job.
//! Files go to a directory every rank sees ([`shared_dir`]). The Python oracle runs on rank 0
//! only, unless `GRIDCL_SKIP_PYTHON=1`.
//!
//! - [`global_mesh_identical`] (bullet 1, rule 11): the mesh read on the communicator, gathered
//!   to rank 0 by cell origin id, equals the one-rank read.
//! - [`block_honours_partitions`] (rule 5): `Block` keeps the partitions of a partitioned file
//!   when the rank count matches, and the even chunks otherwise.
//! - [`partitioned_round_trip`] (rule 7): `.partitioned(true)` parts read serially add up to the
//!   mesh; the gathered file equals the one-rank file after sorting by origin id.
//! - [`ghosts_once`] (bullet 4, rule 8): after VTK removes the ghost cells of a PVTU or VTKHDF
//!   file, every cell appears exactly once.
//! - [`vtu_gathered_equals_serial`] (rule 8): the gathered VTU equals the one-rank VTU after
//!   sorting by origin id.
//! - [`vtkhdf_equals_pvtu`] (bullet 3, rule 12): VTK reads the VTKHDF file as the PVTU.
//! - [`read_memory`] (bullet 2, rule 10): peak resident memory against the mesh footprint.
//! - [`collective_write_errors`] (rules 7–8): a write that fails on one rank fails on all.
//! - [`write_box_msh`]: the binary box file of the memory test (`gen-gmsh-box`).

#[path = "../../gridcl-io/tests/common/mod.rs"]
pub mod fixtures;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};

use cubecl::prelude::Client;
use gridcl::prelude::*;

use crate::oracle::{Oracle, repo};
use crate::parallel::{decode_ids, gather_to_root};
use crate::{HostMesh, supports};

// ---------------------------------------------------------------------------------------------
// Cases.

/// A partitioner of a read.
#[derive(Clone, Copy)]
pub enum Choice {
    /// No `GmshOptions::partitioner` call: the default of rule 6.
    Default,
    /// A named partitioner.
    Named(&'static str, fn() -> Box<dyn Partitioner>),
}

impl fmt::Debug for Choice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Choice::Default => write!(f, "default"),
            Choice::Named(name, _) => write!(f, "{name}"),
        }
    }
}

/// `Block`, explicitly.
pub const BLOCK: Choice = Choice::Named("block", || Box::new(Block));

/// A boxed partitioner as a `Partitioner + 'static`.
pub struct BoxPartitioner(pub Box<dyn Partitioner>);

impl Partitioner for BoxPartitioner {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        self.0.partition(comm, g, nparts)
    }
}

/// The file of a read case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// A file of the Gmsh fixture corpus.
    Fixture(String),
    /// `mixed_box(2, 1)` (hexahedra, prisms, pyramids, tetrahedra), written as a binary file
    /// by gridcl on rank 0: the generated multi-block box.
    MixedBox,
}

/// One read of [`global_mesh_identical`].
#[derive(Clone, Debug)]
pub struct ReadCase {
    /// The file.
    pub source: Source,
    /// The ghost mode.
    pub ghosts: GhostMode,
    /// The partitioner.
    pub partitioner: Choice,
    /// f64 if the device supports it, else f32.
    pub double: bool,
}

impl fmt::Display for ReadCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = match &self.source {
            Source::Fixture(name) => name.as_str(),
            Source::MixedBox => "mixed box",
        };
        write!(
            f,
            "{file} {:?} {:?} {}",
            self.ghosts,
            self.partitioner,
            if self.double { "f64" } else { "f32" }
        )
    }
}

/// Which files [`read_cases`] covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// Every mesh of the corpus and the generated box (1–4 and 8 ranks).
    Full,
    /// One file per cell type, the partitioned files, a serendipity file and the generated box
    /// (5–7 ranks).
    Subset,
}

/// The files of [`Coverage::Subset`].
pub const SUBSET: [&str; 11] = [
    "interval_p2-binary.msh",
    "triangle_p2.msh",
    "quadrilateral_p1-binary.msh",
    "tetrahedron_p2-binary.msh",
    "hexahedron_p1.msh",
    "prism_p2-binary.msh",
    "hybrid_p1-binary.msh",
    "box_surface_p2-binary.msh",
    "hybrid_s2-binary.msh",
    "partitioned_p1.msh",
    "partitioned_ghosts_p1-binary.msh",
];

/// The ghost modes of the read cases.
pub const READ_GHOSTS: [GhostMode; 2] = [GhostMode::None, GhostMode::SharedFacet];

/// The read cases of `coverage` with each of `partitioners`, both ghost modes, alternating f32
/// and f64 over the files.
pub fn read_cases(coverage: Coverage, partitioners: &[Choice]) -> Vec<ReadCase> {
    let mut sources: Vec<Source> = match coverage {
        Coverage::Full => fixtures::manifest()
            .meshes()
            .map(|f| Source::Fixture(f.file.clone()))
            .collect(),
        Coverage::Subset => SUBSET
            .iter()
            .map(|f| Source::Fixture((*f).to_owned()))
            .collect(),
    };
    sources.push(Source::MixedBox);
    let mut out = Vec::new();
    for (i, source) in sources.into_iter().enumerate() {
        for &ghosts in &READ_GHOSTS {
            for &partitioner in partitioners {
                out.push(ReadCase {
                    source: source.clone(),
                    ghosts,
                    partitioner,
                    double: i % 2 == 0,
                });
            }
        }
    }
    out
}

/// The coverage of `ranks` ranks: [`Coverage::Full`] at 1–4 and 8 ranks, else
/// [`Coverage::Subset`] (rule 11).
pub fn coverage_of(ranks: usize) -> Coverage {
    if ranks <= 4 || ranks == 8 {
        Coverage::Full
    } else {
        Coverage::Subset
    }
}

// ---------------------------------------------------------------------------------------------
// Shared helpers.

/// The value of `r`, or a panic naming `context`.
fn ok<T>(r: Result<T>, context: &dyn fmt::Display) -> T {
    r.unwrap_or_else(|e| panic!("{context}: {e}"))
}

/// Whether `double` is honoured on `client`.
fn use_f64(client: &Client, double: bool) -> bool {
    double && supports::<f64>(client)
}

/// A directory that every rank of `comm` sees, empty, named after `name`, the rank count and
/// rank 0's process id: `$GRIDCL_M8_TMP`, else `<repo>/target/m8-tmp` (on a shared file system,
/// unlike the node-local `/tmp` of a cluster).
pub fn shared_dir(comm: &dyn CommDyn, name: &str) -> PathBuf {
    let base = std::env::var_os("GRIDCL_M8_TMP")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| repo().join("target/m8-tmp"));
    let pid = comm.max(if comm.rank() == 0 {
        u64::from(std::process::id())
    } else {
        0
    });
    let dir = base.join(format!("{name}-{}-{pid}", comm.size()));
    if comm.rank() == 0 {
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    }
    comm.barrier();
    dir
}

/// Removes a [`shared_dir`] after every rank is done with it.
pub fn remove_shared_dir(comm: &dyn CommDyn, dir: &Path) {
    comm.barrier();
    if comm.rank() == 0 {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The path of `source` (written on rank 0 into `dir` for the generated box).
fn materialise(comm: &dyn CommDyn, client: &Client, source: &Source, dir: &Path) -> PathBuf {
    match source {
        Source::Fixture(name) => fixtures::fixtures_dir().join(name),
        Source::MixedBox => {
            let path = dir.join("mixed_box.msh");
            if comm.rank() == 0 && !path.exists() {
                let mesh: VolumeMesh<f64> = ok(crate::mixed_box(2, 1).build(client), &"mixed box");
                ok(
                    gmsh::GmshWriter::new(&path).binary(true).write(&mesh),
                    &"write the mixed box",
                );
            }
            comm.barrier();
            path
        }
    }
}

/// Reads `path` on `comm` with the options of `ghosts` and `partitioner`.
fn read_on<F: Real, C: Comm>(
    client: &Client,
    comm: C,
    path: &Path,
    ghosts: GhostMode,
    partitioner: Choice,
) -> Result<AnyMesh<F, C>> {
    let mut options = GmshOptions::default().ghosts(ghosts);
    if let Choice::Named(_, make) = partitioner {
        options = options.partitioner(BoxPartitioner(make()));
    }
    gmsh::read::<F, C>(client, comm, path, options)
}

// ---------------------------------------------------------------------------------------------
// Snapshots keyed by origin ids.

/// A stable code of a cell type.
fn cell_code(cell: CellType) -> i64 {
    match cell {
        CellType::Point => 0,
        CellType::Interval => 1,
        CellType::Triangle => 2,
        CellType::Quadrilateral => 3,
        CellType::Tetrahedron => 4,
        CellType::Hexahedron => 5,
        CellType::Prism => 6,
        CellType::Pyramid => 7,
    }
}

/// A mesh on the host, keyed by origin ids, in a form that gathers as `i64`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Snapshot {
    /// Cell origin id → [cell code, degree, vertex origin ids in cell order..., node coordinates
    /// in internal order as f64 bits...].
    pub cells: BTreeMap<i64, Vec<i64>>,
    /// [kind (0 physical, 1 elementary), dimension, sorted vertex origin ids..., value].
    pub tags: BTreeSet<Vec<i64>>,
    /// (name, dimension, tag).
    pub names: BTreeSet<(String, usize, i32)>,
}

/// The sorted vertex origin ids of every entity of dimension `dim`, by flat index.
fn entity_vertex_ids<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, dim: usize) -> Result<Vec<Vec<i64>>>
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

/// The snapshot of `m`: its owned cells (all cells if `!owned_only`), every tag entry of the
/// rank and the name table.
pub fn snapshot<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, owned_only: bool) -> Result<Snapshot>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let ids = decode_ids(&m.vertex_origin_ids().to_host()?);
    let nodes = m.nodes().to_host()?;
    let origin = m.origin();
    let origin = origin.as_ref();
    let mut out = Snapshot::default();
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let geom = m.geometry_block(b);
        let cell: CellType = block.cell.into();
        let k = geom.dofmap.stride as usize;
        let nv = block.vertices.stride as usize;
        let verts = block.vertices.data.to_host()?;
        let dofmap = geom.dofmap.data.to_host()?;
        let cell_ids = decode_ids(&block.origin_ids.to_host()?);
        let n = if owned_only {
            block.index_map.num_owned()
        } else {
            block.index_map.num_local()
        };
        for (c, &id) in cell_ids.iter().enumerate().take(n) {
            let mut record = vec![cell_code(cell), i64::from(geom.element.degree)];
            record.extend(verts[c * nv..(c + 1) * nv].iter().map(|&v| ids[v as usize]));
            for &node in &dofmap[c * k..(c + 1) * k] {
                for g in 0..gd {
                    let x = nodes[node as usize * gd + g].widen() + origin[g];
                    record.push(x.to_bits() as i64);
                }
            }
            if out.cells.insert(id, record).is_some() {
                return Err(GridError::DuplicateOriginId(id));
            }
        }
    }
    let tags = m.tags();
    for d in 0..=TD::N {
        for (kind, t) in [(0, tags.physical(d)), (1, tags.elementary(d))] {
            let Some(t) = t else { continue };
            let entities = entity_vertex_ids(m, d)?;
            let indices = t.indices.to_host()?;
            let values = t.values.to_host()?;
            for (i, v) in indices.into_iter().zip(values) {
                let mut key = vec![kind, d as i64];
                key.extend(&entities[i as usize]);
                key.push(i64::from(v));
                out.tags.insert(key);
            }
        }
    }
    out.names = tags.names().map(|(n, d, v)| (n.to_owned(), d, v)).collect();
    Ok(out)
}

/// The snapshot of an erased mesh.
pub fn snapshot_any<F: Real, C: Comm>(any: &AnyMesh<F, C>, owned_only: bool) -> Result<Snapshot> {
    with_mesh!(any.as_ref(), m => snapshot(m, owned_only))
}

/// A stable 64-bit hash of the name table (FNV-1a).
fn names_hash(names: &BTreeSet<(String, usize, i32)>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (n, d, v) in names {
        for b in format!("{n}\0{d}\0{v}\n").bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    h
}

/// The snapshots of every rank, merged on rank 0 (cells must not repeat; tags are united);
/// `None` elsewhere. Every rank's name table must equal rank 0's.
pub fn gather_snapshot(comm: &dyn CommDyn, local: &Snapshot) -> Result<Option<Snapshot>> {
    let mut cells = Vec::new();
    for (&id, record) in &local.cells {
        cells.push(id);
        cells.push(record.len() as i64);
        cells.extend(record);
    }
    let mut tags = Vec::new();
    for key in &local.tags {
        tags.push(key.len() as i64);
        tags.extend(key);
    }
    let cells = gather_to_root(comm, &cells)?;
    let tags = gather_to_root(comm, &tags)?;
    let hashes = gather_to_root(comm, &[names_hash(&local.names)])?;
    if comm.rank() != 0 {
        return Ok(None);
    }
    let mut out = Snapshot {
        names: local.names.clone(),
        ..Snapshot::default()
    };
    for (r, &h) in hashes.iter().enumerate() {
        assert_eq!(
            h, hashes[0],
            "rank {r}: the name table differs from rank 0's"
        );
    }
    let mut i = 0;
    while i < cells.len() {
        let (id, n) = (cells[i], cells[i + 1] as usize);
        let record = cells[i + 2..i + 2 + n].to_vec();
        assert!(
            out.cells.insert(id, record).is_none(),
            "cell {id} is owned by more than one rank"
        );
        i += 2 + n;
    }
    let mut i = 0;
    while i < tags.len() {
        let n = tags[i] as usize;
        out.tags.insert(tags[i + 1..i + 1 + n].to_vec());
        i += 1 + n;
    }
    Ok(Some(out))
}

/// Asserts `got == want`, naming the first difference.
pub fn assert_same_snapshot(got: &Snapshot, want: &Snapshot, what: &dyn fmt::Display) {
    let (gk, wk): (BTreeSet<_>, BTreeSet<_>) =
        (got.cells.keys().collect(), want.cells.keys().collect());
    if gk != wk {
        let missing: Vec<_> = wk.difference(&gk).take(8).collect();
        let extra: Vec<_> = gk.difference(&wk).take(8).collect();
        panic!(
            "{what}: cell origin ids differ ({} vs {} cells; missing {missing:?}, extra {extra:?})",
            got.cells.len(),
            want.cells.len()
        );
    }
    for (id, w) in &want.cells {
        let g = &got.cells[id];
        assert_eq!(
            g, w,
            "{what}: cell {id} (type, degree, vertex ids, coordinate bits)"
        );
    }
    if got.tags != want.tags {
        let missing: Vec<_> = want.tags.difference(&got.tags).take(4).collect();
        let extra: Vec<_> = got.tags.difference(&want.tags).take(4).collect();
        panic!("{what}: tags differ: missing {missing:?}, extra {extra:?}");
    }
    assert_eq!(got.names, want.names, "{what}: names");
}

// ---------------------------------------------------------------------------------------------
// Bullet 1: the global mesh.

/// Bullet 1 (rule 11): `case` read on `comm`, gathered to rank 0 by cell origin id, equals the
/// one-rank read (cell types and degrees, vertex origin ids in cell order, node coordinates in
/// internal order bitwise, physical and elementary tags keyed by sorted vertex origin ids, the
/// name table). `dir` is a [`shared_dir`] for the generated box.
pub fn global_mesh_identical<C: Comm>(comm: &C, client: &Client, case: &ReadCase, dir: &Path) {
    let path = materialise(comm, client, &case.source, dir);
    if use_f64(client, case.double) {
        global_mesh_identical_in::<f64, C>(comm, client, case, &path);
    } else {
        global_mesh_identical_in::<f32, C>(comm, client, case, &path);
    }
}

fn global_mesh_identical_in<F: Real, C: Comm>(
    comm: &C,
    client: &Client,
    case: &ReadCase,
    path: &Path,
) {
    let rank = comm.rank();
    let at = format!("{case}, {} ranks, rank {rank}", comm.size());
    let any = ok(
        read_on::<F, C>(client, comm.clone(), path, case.ghosts, case.partitioner),
        &at,
    );
    let local = ok(snapshot_any(&any, true), &at);
    let Some(global) = ok(gather_snapshot(comm, &local), &at) else {
        return;
    };
    let serial = ok(
        gmsh::read::<F, Serial>(client, Serial, path, GmshOptions::default()),
        &format!("{at}: one-rank read"),
    );
    let want = ok(snapshot_any(&serial, true), &at);
    assert_same_snapshot(&global, &want, &at);
}

// ---------------------------------------------------------------------------------------------
// Rule 5: partitioned files.

/// The cell origin ids this rank owns.
fn owned_ids<F: Real, C: Comm>(any: &AnyMesh<F, C>) -> Result<BTreeSet<i64>> {
    Ok(snapshot_any(any, true)?.cells.into_keys().collect())
}

/// Rule 5: every partitioned fixture read with `Block` on `comm`. With as many ranks as
/// partitions, rank r owns the D_TOP elements of partition r + 1 (Gmsh's own view, from the
/// manifest); otherwise the partition data is ignored and rank r owns the r-th even chunk
/// [rN/P, (r+1)N/P) of the D_TOP elements. Both ghost modes; ghost entities are ignored.
pub fn block_honours_partitions<C: Comm>(comm: &C, client: &Client) {
    let (rank, size) = (comm.rank(), comm.size());
    for f in fixtures::manifest()
        .files
        .iter()
        .filter(|f| f.num_partitions > 0)
    {
        for &ghosts in &READ_GHOSTS {
            let at = format!("{} {ghosts:?}, {size} ranks, rank {rank}", f.file);
            let any = ok(
                read_on::<f32, C>(client, comm.clone(), &f.path(), ghosts, BLOCK),
                &at,
            );
            let owned = ok(owned_ids(&any), &at);
            let total: usize = f.partition_elements.iter().map(Vec::len).sum();
            if size == f.num_partitions {
                let want: BTreeSet<i64> = f.partition_elements[rank].iter().copied().collect();
                assert_eq!(owned, want, "{at}: the cells of partition {}", rank + 1);
            } else {
                let want = (rank + 1) * total / size - rank * total / size;
                assert_eq!(owned.len(), want, "{at}: an even chunk of {total} cells");
            }
            assert_eq!(
                comm.sum(owned.len() as u64),
                total as u64,
                "{at}: all cells"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Rule 7: partitioned and gathered Gmsh files.

/// The fixtures of [`partitioned_round_trip`]: tags of every dimension, P2, mixed blocks.
pub const ROUND_TRIP: [&str; 3] = [
    "two_regions_p1.msh",
    "tetrahedron_p2-binary.msh",
    "hybrid_p1.msh",
];

/// Rule 7: each fixture of [`ROUND_TRIP`] read on `comm` with shared-facet ghosts and written
/// with `.partitioned(true)` (more than one rank) and gathered. On rank 0: each part, read
/// serially, holds `$PartitionedEntities` with P partitions; the parts' cells are disjoint and
/// together, with their tags, equal the one-rank read; the gathered file, read serially, equals
/// the file written from the one-rank read (sorting by origin id).
pub fn partitioned_round_trip<C: Comm>(comm: &C, client: &Client, dir: &Path) {
    for (i, name) in ROUND_TRIP.iter().enumerate() {
        if use_f64(client, i % 2 == 0) {
            round_trip_in::<f64, C>(comm, client, name, dir);
        } else {
            round_trip_in::<f32, C>(comm, client, name, dir);
        }
    }
}

fn round_trip_in<F: Real, C: Comm>(comm: &C, client: &Client, name: &str, dir: &Path) {
    let (rank, size) = (comm.rank(), comm.size());
    let at = format!("{name}, {size} ranks, rank {rank}");
    let source = fixtures::fixtures_dir().join(name);
    let any = ok(
        read_on::<F, C>(
            client,
            comm.clone(),
            &source,
            GhostMode::SharedFacet,
            Choice::Default,
        ),
        &at,
    );
    let stem = dir.join(name.trim_end_matches(".msh"));
    let gathered = PathBuf::from(format!("{}-gathered.msh", stem.display()));
    let parts = |r: usize| PathBuf::from(format!("{}_{}.msh", stem.display(), r + 1));
    if size > 1 {
        ok(
            gmsh::GmshWriter::new(stem.with_extension("msh"))
                .partitioned(true)
                .write_any(any.as_ref()),
            &format!("{at}: partitioned write"),
        );
    }
    ok(
        gmsh::GmshWriter::new(&gathered).write_any(any.as_ref()),
        &format!("{at}: gathered write"),
    );
    comm.barrier();
    if rank != 0 {
        return;
    }
    let read_serial = |path: &Path| -> Snapshot {
        let what = format!("{at}: read {}", path.display());
        let any = ok(
            gmsh::read::<F, Serial>(client, Serial, path, GmshOptions::default()),
            &what,
        );
        ok(snapshot_any(&any, true), &what)
    };
    let one_rank = ok(
        gmsh::read::<F, Serial>(client, Serial, &source, GmshOptions::default()),
        &at,
    );
    let want = ok(snapshot_any(&one_rank, true), &at);

    if size > 1 {
        let mut union = Snapshot {
            names: want.names.clone(),
            ..Snapshot::default()
        };
        for r in 0..size {
            let path = parts(r);
            let text = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let marker = format!("$PartitionedEntities\n{size}\n");
            assert!(
                text.windows(marker.len()).any(|w| w == marker.as_bytes()),
                "{at}: part {} lacks $PartitionedEntities with {size} partitions",
                r + 1
            );
            let part = read_serial(&path);
            assert_eq!(part.names, want.names, "{at}: part {}: names", r + 1);
            for (id, record) in part.cells {
                assert!(
                    union.cells.insert(id, record).is_none(),
                    "{at}: cell {id} is in more than one part"
                );
            }
            union.tags.extend(part.tags);
        }
        assert_same_snapshot(&union, &want, &format!("{at}: the union of the parts"));
    }

    let serial_file = PathBuf::from(format!("{}-serial.msh", stem.display()));
    ok(
        with_mesh!(one_rank.as_ref(), m => gmsh::GmshWriter::new(&serial_file).write(m)),
        &at,
    );
    let text = std::fs::read(&gathered).unwrap_or_else(|e| panic!("{}: {e}", gathered.display()));
    assert!(
        !text.windows(20).any(|w| w == b"$PartitionedEntities"),
        "{at}: the gathered file is not partitioned"
    );
    assert_same_snapshot(
        &read_serial(&gathered),
        &read_serial(&serial_file),
        &format!("{at}: the gathered file against the one-rank file"),
    );
}

// ---------------------------------------------------------------------------------------------
// VTK: ghosts, gathered VTU, VTKHDF.

/// The VTK output of [`ghosts_once`].
#[derive(Clone, Copy)]
pub enum VtkFormat {
    /// `VtkWriter::pvtu`.
    Pvtu,
    /// `VtkWriter::vtkhdf` (feature `hdf5`), passed in by the caller.
    Vtkhdf(fn(&Path) -> VtkWriter<'static>),
}

/// The meshes of the VTK checks: (name, host mesh, D_TOP).
fn vtk_meshes() -> [(&'static str, HostMesh, usize); 2] {
    [
        (
            "tetrahedra-p2",
            crate::cube_box(Cell3::Tetrahedron, 2, 2),
            3,
        ),
        ("mixed-p1", crate::mixed_box(2, 1), 3),
    ]
}

/// `h` built on `comm` from even chunks with `ghosts`.
fn build_volume<F: Real, C: Comm>(
    comm: &C,
    client: &Client,
    h: &HostMesh,
    ghosts: GhostMode,
) -> Result<VolumeMesh<F, C>> {
    h.chunk(comm.rank(), comm.size())
        .builder::<F, D3, D3>(client)?
        .ghosts(ghosts)
        .build_on(comm.clone())
}

/// The sorted cell origin ids of `h`.
fn all_cell_ids(h: &HostMesh) -> Vec<i64> {
    let mut ids: Vec<i64> = h.cell_origin_ids().concat();
    ids.sort_unstable();
    ids
}

/// The oracle on rank 0, unless `GRIDCL_SKIP_PYTHON=1`.
fn oracle_on_root(comm: &dyn CommDyn, test: &str) -> Option<Oracle> {
    if comm.rank() == 0 {
        Oracle::find(test)
    } else {
        None
    }
}

/// The `"equal"` result of a `compare`, with the differences on failure.
fn assert_equal(result: &serde_json::Value, what: &dyn fmt::Display) {
    assert_eq!(
        result["equal"], true,
        "{what}: VTK reads the files differently: {}",
        result
    );
}

/// Bullet 4 (rule 8): each mesh of the VTK checks built on `comm` with `ghosts`, written as
/// `format` with origin ids; after VTK removes the ghost cells (`vtkRemoveGhosts`), every cell
/// origin id appears exactly once. The PVTU header has `GhostLevel` 1 exactly when some rank
/// has ghost cells, and `vtkGhostType` exists exactly on more than one rank.
pub fn ghosts_once<C: Comm>(
    comm: &C,
    client: &Client,
    ghosts: GhostMode,
    format: VtkFormat,
    dir: &Path,
) {
    let (rank, size) = (comm.rank(), comm.size());
    let oracle = oracle_on_root(comm, "ghosts_once");
    for (name, h, _) in vtk_meshes() {
        let at = format!("{name} {ghosts:?}, {size} ranks, rank {rank}");
        let mesh = ok(build_volume::<f32, C>(comm, client, &h, ghosts), &at);
        let has_ghosts = mesh
            .cell_blocks()
            .iter()
            .any(|b| b.index_map.num_local() > b.index_map.num_owned());
        let any_ghosts = comm.max(u32::from(has_ghosts)) == 1;
        let stem = dir.join(format!("{name}-{ghosts:?}").replace([' ', '{', '}', ':'], ""));
        let (writer, file) = match format {
            VtkFormat::Pvtu => (VtkWriter::pvtu(&stem), stem.with_extension("pvtu")),
            VtkFormat::Vtkhdf(make) => (make(&stem), stem.with_extension("vtkhdf")),
        };
        ok(
            writer.origin_ids(true).write(&mesh),
            &format!("{at}: write"),
        );
        comm.barrier();
        if rank != 0 {
            continue;
        }
        if matches!(format, VtkFormat::Pvtu) {
            let header = std::fs::read_to_string(&file)
                .unwrap_or_else(|e| panic!("{}: {e}", file.display()));
            let level = if any_ghosts { 1 } else { 0 };
            assert!(
                header.contains(&format!("GhostLevel=\"{level}\"")),
                "{at}: GhostLevel {level} expected in\n{header}"
            );
        }
        let Some(oracle) = &oracle else { continue };
        let read = oracle.run_files("read", std::slice::from_ref(&file));
        let names: Vec<&str> = read[0]["cell_data"]
            .as_array()
            .expect("cell data list")
            .iter()
            .map(|a| a["name"].as_str().expect("array name"))
            .collect();
        assert_eq!(
            names.contains(&"vtkGhostType"),
            size > 1,
            "{at}: vtkGhostType on more than one rank only ({names:?})"
        );
        assert!(names.contains(&"OriginId"), "{at}: OriginId ({names:?})");
        let removed = oracle.run_files("remove-ghosts", std::slice::from_ref(&file));
        let mut ids: Vec<i64> = removed[0]["origin_ids"]
            .as_array()
            .unwrap_or_else(|| panic!("{at}: no OriginId after ghost removal: {}", removed[0]))
            .iter()
            .map(|v| v.as_i64().expect("origin id"))
            .collect();
        ids.sort_unstable();
        assert_eq!(ids, all_cell_ids(&h), "{at}: every cell exactly once");
    }
}

/// Point data from the node coordinates (x + 2y + 3z) and cell data from the cell origin ids,
/// for the writers: the same values on any partition.
fn data_of<C: Comm>(mesh: &VolumeMesh<f64, C>) -> Result<(DeviceBuffer<f64>, DeviceBuffer<i32>)> {
    let client = mesh.client();
    let nodes = mesh.nodes().to_host()?;
    let origin = mesh.origin();
    let point: Vec<f64> = nodes
        .as_chunks::<3>()
        .0
        .iter()
        .map(|x| (x[0] + origin[0]) + 2.0 * (x[1] + origin[1]) + 3.0 * (x[2] + origin[2]))
        .collect();
    let mut cell = Vec::new();
    for b in mesh.cell_blocks() {
        cell.extend(
            decode_ids(&b.origin_ids.to_host()?)
                .iter()
                .map(|&i| i as i32),
        );
    }
    Ok((
        DeviceBuffer::from_host(client, &point)?,
        DeviceBuffer::from_host(client, &cell)?,
    ))
}

/// Rule 8: each mesh of the VTK checks built on `comm` with shared-facet ghosts and written as
/// a gathered VTU with origin ids, point data and cell data equals, after sorting by origin id,
/// the VTU of the one-rank build (oracle `compare --by-origin-id`); it has no `vtkGhostType`.
pub fn vtu_gathered_equals_serial<C: Comm>(comm: &C, client: &Client, dir: &Path) {
    if !supports::<f64>(client) {
        eprintln!("vtu_gathered_equals_serial: f64 is not supported on this device; skipped");
        return;
    }
    let (rank, size) = (comm.rank(), comm.size());
    let oracle = oracle_on_root(comm, "vtu_gathered_equals_serial");
    for (name, h, _) in vtk_meshes() {
        let at = format!("{name}, {size} ranks, rank {rank}");
        let mesh = ok(
            build_volume::<f64, C>(comm, client, &h, GhostMode::SharedFacet),
            &at,
        );
        let (point, cell) = ok(data_of(&mesh), &at);
        let gathered = dir.join(format!("{name}-gathered.vtu"));
        ok(
            VtkWriter::vtu(&gathered)
                .origin_ids(true)
                .point_data("f", &point, 1)
                .cell_data("id", &cell, 1)
                .write(&mesh),
            &format!("{at}: gathered write"),
        );
        comm.barrier();
        if rank != 0 {
            continue;
        }
        let serial: VolumeMesh<f64> = ok(h.build(client), &at);
        let (point, cell) = ok(data_of(&serial), &at);
        let one = dir.join(format!("{name}-serial.vtu"));
        ok(
            VtkWriter::vtu(&one)
                .origin_ids(true)
                .point_data("f", &point, 1)
                .cell_data("id", &cell, 1)
                .write(&serial),
            &at,
        );
        if let Some(oracle) = &oracle {
            let r = oracle.run(
                "compare",
                &["--by-origin-id".into(), gathered.into(), one.into()],
            );
            assert_equal(&r, &format!("{at}: gathered VTU against the one-rank VTU"));
        }
    }
}

/// Bullet 3 (rule 12): each mesh of the VTK checks built on `comm` with shared-facet ghosts and
/// written as PVTU and as VTKHDF (`vtkhdf`, the caller's `VtkWriter::vtkhdf`), with origin ids,
/// point data and cell data: VTK's `vtkHDFReader` and `vtkXMLPUnstructuredGridReader` (ghost
/// level 1) give equal arrays, bitwise, and equal per-piece counts.
pub fn vtkhdf_equals_pvtu<C: Comm>(
    comm: &C,
    client: &Client,
    vtkhdf: fn(&Path) -> VtkWriter<'static>,
    dir: &Path,
) {
    if !supports::<f64>(client) {
        eprintln!("vtkhdf_equals_pvtu: f64 is not supported on this device; skipped");
        return;
    }
    let (rank, size) = (comm.rank(), comm.size());
    let oracle = oracle_on_root(comm, "vtkhdf_equals_pvtu");
    for (name, h, _) in vtk_meshes() {
        let at = format!("{name}, {size} ranks, rank {rank}");
        let mesh = ok(
            build_volume::<f64, C>(comm, client, &h, GhostMode::SharedFacet),
            &at,
        );
        let (point, cell) = ok(data_of(&mesh), &at);
        let stem = dir.join(name);
        for writer in [VtkWriter::pvtu(&stem), vtkhdf(&stem)] {
            ok(
                writer
                    .origin_ids(true)
                    .point_data("f", &point, 1)
                    .cell_data("id", &cell, 1)
                    .write(&mesh),
                &format!("{at}: write"),
            );
        }
        comm.barrier();
        if let Some(oracle) = &oracle {
            let r = oracle.run(
                "compare",
                &[
                    stem.with_extension("vtkhdf").into(),
                    stem.with_extension("pvtu").into(),
                ],
            );
            assert_equal(&r, &format!("{at}: VTKHDF against PVTU"));
            assert_eq!(r["pieces"], size, "{at}: one piece per rank");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Bullet 2: memory.

/// 256 MiB.
pub const MEMORY_SLACK: u64 = 256 << 20;

/// Bullet 2 (rule 10): reads `path` on `comm` with `Block` and no ghosts in f64, then compares
/// the peak resident memory of this process (`peak_rss`, in bytes) with twice the mesh's
/// `footprint_bytes` plus 256 MiB. Prints both on every rank.
pub fn read_memory<C: Comm>(comm: &C, client: &Client, path: &Path, peak_rss: &dyn Fn() -> u64) {
    let (rank, size) = (comm.rank(), comm.size());
    let at = format!("{}, {size} ranks, rank {rank}", path.display());
    let any = ok(
        read_on::<f64, C>(client, comm.clone(), path, GhostMode::None, BLOCK),
        &at,
    );
    let footprint = with_mesh!(any.as_ref(), m => m.footprint_bytes()) as u64;
    let cells = with_mesh!(any.as_ref(), m => m
        .cell_blocks()
        .iter()
        .map(|b| b.index_map.num_owned())
        .sum::<usize>());
    let peak = peak_rss();
    let bound = 2 * footprint + MEMORY_SLACK;
    println!(
        "read_memory: rank {rank}/{size}: {cells} owned cells, footprint {} MiB, peak RSS {} MiB, \
         bound {} MiB",
        footprint >> 20,
        peak >> 20,
        bound >> 20
    );
    comm.barrier();
    assert!(
        peak < bound,
        "{at}: peak RSS {peak} B >= 2 x footprint {footprint} B + 256 MiB"
    );
}

// ---------------------------------------------------------------------------------------------
// Collective writes.

/// Rules 7–8: writes that fail on one rank fail on every rank with `Io`: a gathered Gmsh file
/// and a VTU in a missing directory (rank 0 fails), and, where a directory stands in the way of
/// the last rank's file, a partitioned Gmsh write (more than one rank) and a PVTU.
pub fn collective_write_errors<C: Comm>(comm: &C, client: &Client, dir: &Path) {
    let (rank, size) = (comm.rank(), comm.size());
    let at = format!("{size} ranks, rank {rank}");
    let h = crate::cube_box(Cell3::Hexahedron, 2, 1);
    let mesh = ok(
        build_volume::<f32, C>(comm, client, &h, GhostMode::SharedFacet),
        &at,
    );
    let missing = dir.join("missing");
    if rank == 0 {
        std::fs::create_dir_all(dir.join(format!("blocked_{size}.msh"))).expect("blocker");
        std::fs::create_dir_all(dir.join(format!("blocked_{}.vtu", size - 1))).expect("blocker");
    }
    comm.barrier();
    let mut writes: Vec<(&str, Result<()>)> = vec![
        (
            "gathered Gmsh in a missing directory",
            gmsh::GmshWriter::new(missing.join("m.msh")).write(&mesh),
        ),
        (
            "VTU in a missing directory",
            VtkWriter::vtu(missing.join("m.vtu")).write(&mesh),
        ),
        (
            "PVTU whose last piece is blocked",
            VtkWriter::pvtu(dir.join("blocked")).write(&mesh),
        ),
    ];
    if size > 1 {
        writes.push((
            "partitioned Gmsh whose last part is blocked",
            gmsh::GmshWriter::new(dir.join("blocked.msh"))
                .partitioned(true)
                .write(&mesh),
        ));
    }
    for (what, r) in writes {
        assert!(
            matches!(r, Err(GridError::Io(_))),
            "{at}: {what}: expected Io on every rank, got {r:?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// The box file of the memory test.

/// The physical names of [`write_box_msh`]: the volume (dimension 3, tag 1) and the sides
/// x = 0, x = 1, y = 0, y = 1, z = 0, z = 1 (dimension 2, tags 1–6).
pub const BOX_GROUPS: [(usize, i32, &str); 7] = [
    (3, 1, "volume"),
    (2, 1, "x0"),
    (2, 2, "x1"),
    (2, 3, "y0"),
    (2, 4, "y1"),
    (2, 5, "z0"),
    (2, 6, "z1"),
];

/// Writes `cube_box(Tetrahedron, n, 1)` as a binary MSH 4.1 file (little-endian, 8-byte
/// `size_t`): one volume entity with physical group "volume" holding every node and
/// tetrahedron, and six surface entities, one per side, with the boundary triangles and the
/// physical groups of [`BOX_GROUPS`]. Node tags are node index + 1, tetrahedron tags cell index
/// + 1, triangle tags follow. Streams the file; the host mesh is the largest allocation.
///
/// # Errors
///
/// I/O errors of the file.
pub fn write_box_msh(n: usize, path: &Path) -> std::io::Result<()> {
    let h = crate::cube_box(Cell3::Tetrahedron, n, 1);
    let block = &h.blocks[0];
    assert_eq!(block.cell, CellType::Tetrahedron);
    let num_nodes = h.nodes.len() / 3;
    let num_tets = block.num_cells();
    // Boundary triangles per side: a face of a tetrahedron whose three nodes lie on a side.
    let mut sides: [Vec<[u32; 3]>; 6] = Default::default();
    const FACES: [[usize; 3]; 4] = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
    for tet in block.cells.as_chunks::<4>().0 {
        for face in FACES {
            let f = face.map(|i| tet[i]);
            for (s, side) in sides.iter_mut().enumerate() {
                let (axis, value) = (s / 2, (s % 2) as f64);
                if f.iter().all(|&v| h.nodes[v as usize * 3 + axis] == value) {
                    side.push(f);
                }
            }
        }
    }
    let num_triangles: usize = sides.iter().map(Vec::len).sum();

    let mut w = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(path)?);
    let size = |w: &mut dyn Write, x: usize| w.write_all(&(x as u64).to_le_bytes());
    let int = |w: &mut dyn Write, x: i32| w.write_all(&x.to_le_bytes());
    let real = |w: &mut dyn Write, x: f64| w.write_all(&x.to_le_bytes());
    w.write_all(b"$MeshFormat\n4.1 1 8\n")?;
    int(&mut w, 1)?;
    w.write_all(b"\n$EndMeshFormat\n$PhysicalNames\n")?;
    writeln!(w, "{}", BOX_GROUPS.len())?;
    for (dim, tag, name) in BOX_GROUPS {
        writeln!(w, "{dim} {tag} \"{name}\"")?;
    }
    w.write_all(b"$EndPhysicalNames\n$Entities\n")?;
    for count in [0, 0, 6, 1] {
        size(&mut w, count)?;
    }
    for s in 0..6 {
        int(&mut w, s + 1)?;
        let (axis, value) = (s / 2, (s % 2) as f64);
        for bound in [0.0, 1.0] {
            for a in 0..3 {
                real(&mut w, if a == axis { value } else { bound })?;
            }
        }
        size(&mut w, 1)?;
        int(&mut w, s + 1)?;
        size(&mut w, 0)?;
    }
    int(&mut w, 1)?;
    for x in [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] {
        real(&mut w, x)?;
    }
    size(&mut w, 1)?;
    int(&mut w, 1)?;
    size(&mut w, 6)?;
    for s in 1..=6 {
        int(&mut w, s)?;
    }
    w.write_all(b"\n$EndEntities\n$Nodes\n")?;
    for x in [1, num_nodes, 1, num_nodes] {
        size(&mut w, x)?;
    }
    int(&mut w, 3)?;
    int(&mut w, 1)?;
    int(&mut w, 0)?;
    size(&mut w, num_nodes)?;
    for i in 0..num_nodes {
        size(&mut w, i + 1)?;
    }
    for x in &h.nodes {
        real(&mut w, *x)?;
    }
    w.write_all(b"\n$EndNodes\n$Elements\n")?;
    let num_elements = num_tets + num_triangles;
    for x in [7, num_elements, 1, num_elements] {
        size(&mut w, x)?;
    }
    int(&mut w, 3)?;
    int(&mut w, 1)?;
    int(&mut w, 4)?;
    size(&mut w, num_tets)?;
    for (c, tet) in block.cells.as_chunks::<4>().0.iter().enumerate() {
        size(&mut w, c + 1)?;
        for &v in tet {
            size(&mut w, v as usize + 1)?;
        }
    }
    let mut tag = num_tets;
    for (s, side) in sides.iter().enumerate() {
        int(&mut w, 2)?;
        int(&mut w, s as i32 + 1)?;
        int(&mut w, 2)?;
        size(&mut w, side.len())?;
        for f in side {
            tag += 1;
            size(&mut w, tag)?;
            for &v in f {
                size(&mut w, v as usize + 1)?;
            }
        }
    }
    w.write_all(b"\n$EndElements\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The box file reads into the cube: 6n³ tetrahedra of volume 1, the seven groups, 2n²
    /// triangles per side.
    #[test]
    fn box_file_reads_as_the_cube() {
        let client = crate::test_client();
        let path = std::env::temp_dir().join(format!("gridcl-m8-box-{}.msh", std::process::id()));
        write_box_msh(3, &path).expect("write the box");
        let any = gmsh::read::<f32, Serial>(&client, Serial, &path, GmshOptions::default())
            .expect("read the box");
        std::fs::remove_file(&path).ok();
        let mesh = VolumeMesh::<f32>::try_from(any).expect("a volume mesh");
        assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 6 * 27);
        assert!((mesh.measure(3).expect("measure") - 1.0).abs() < 1e-5);
        let tags = mesh.tags();
        let names: BTreeSet<_> = tags.names().map(|(n, d, v)| (n.to_owned(), d, v)).collect();
        let want: BTreeSet<_> = BOX_GROUPS
            .iter()
            .map(|&(d, v, n)| (n.to_owned(), d, v))
            .collect();
        assert_eq!(names, want);
        for side in ["x0", "x1", "y0", "y1", "z0", "z1"] {
            let t = tags.named(Codim::Bnd, side).expect(side);
            assert_eq!(t.indices.len(), 2 * 9, "{side}");
        }
    }
}
