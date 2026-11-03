//! [`read`] on any number of ranks (ADR-0022; ADR-0035, rules 1–6).
//!
//! Every rank ends up with a [`Chunk`]: its even share of the nodes, of the D_TOP elements and
//! of the lower-dimensional elements of each dimension, converted by the shared steps of
//! `gridcl_io::gmsh` (`Layout`). Every rank scans the file; if it is binary without serendipity
//! elements, each rank reads and converts its own chunk through the scan's [`MshIndex`]
//! (rule 1). Otherwise rank 0 reads the file, upgrades serendipity cells, converts every
//! rank's chunk and scatters them (rule 2); a file whose scan fails takes this path too, so
//! its error is the serial one. The chunks feed stage 1 of the pipeline through the reader's
//! builder hooks; the builder finds duplicate tags, missing nodes and unmatched tag elements
//! (rule 3). With `Block`, a partitioned file whose partition count is the rank count fixes
//! the destination of every cell (rule 5).

use std::ops::Range;
use std::path::Path;

use cubecl::prelude::Client;
use gridcl_device::{GridError, Real, Result, check_precision};
use gridcl_dims::{CellType, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, NodeFamily};
use gridcl_io::gmsh::{
    BlockHeader, CellChunk, ElementChunk, GmshTagGroup, Layout, MshFile, MshIndex, NodeChunk,
    PhysicalName, complete, element_range, element_type, geometric_dim, lagrange_element_type,
    node_range,
};
use gridcl_parallel::{Comm, CommDyn, CommExt};

use super::options::{GmshOptions, Shared};
use crate::pipeline::input::TagKind;
use crate::pipeline::{agree, log_memory};
use crate::{AnyMesh, Mesh, MeshBuilder};

/// One rank's part of a file, converted: what stage 1 needs.
#[derive(Clone, Debug, Default, PartialEq)]
struct Chunk {
    /// D_TOP.
    tdim: usize,
    /// `numPartitions` of `$PartitionedEntities`; 0 without that section.
    num_partitions: usize,
    /// The rank's nodes.
    nodes: NodeChunk,
    /// The rank's D_TOP elements: one chunk per cell block of the file, in `CellType` order
    /// (empty chunks included).
    cells: Vec<CellChunk>,
    /// The rank's lower-dimensional elements.
    groups: Vec<GmshTagGroup>,
}

/// The even share [rN/P, (r+1)N/P) of `n` items on rank `r` of `p` (rule 1).
fn share(r: usize, p: usize, n: usize) -> Result<Range<usize>> {
    let at = |r: usize| -> Result<usize> {
        let x = (r as u128) * (n as u128) / (p as u128);
        usize::try_from(x).map_err(|_| GridError::TooLarge)
    };
    Ok(at(r)?..at(r + 1)?)
}

/// Where chunks come from: a file in memory (rank 0, rule 2) or a binary file on disk, read
/// in byte ranges through its index (rule 1). Both give the same chunks.
#[derive(Clone, Copy)]
enum Source<'a> {
    Memory(&'a MshFile),
    Disk(&'a MshIndex),
}

impl Source<'_> {
    fn num_nodes(self) -> usize {
        match self {
            Source::Memory(file) => file.node_blocks.iter().map(|b| b.tags.len()).sum(),
            Source::Disk(index) => index.num_nodes(),
        }
    }

    fn num_elements(self, dim: usize) -> usize {
        match self {
            Source::Memory(file) => file
                .element_blocks
                .iter()
                .filter(|b| b.dim == dim)
                .map(|b| b.tags.len())
                .sum(),
            Source::Disk(index) => index.num_elements(dim),
        }
    }

    fn nodes(self, range: Range<usize>) -> Result<NodeChunk> {
        match self {
            Source::Memory(file) => node_range(file, range),
            Source::Disk(index) => index.read_nodes(range),
        }
    }

    fn elements(self, dim: usize, range: Range<usize>) -> Result<ElementChunk> {
        match self {
            Source::Memory(file) => element_range(file, dim, range),
            Source::Disk(index) => index.read_elements(dim, range),
        }
    }
}

/// The chunk of rank `r` of `p` of the file `source`, whose layout is `layout`. The D_TOP
/// elements are converted before the rest is read.
fn chunk_of(source: Source<'_>, layout: &Layout, r: usize, p: usize) -> Result<Chunk> {
    let tdim = layout.tdim;
    let top = source.elements(tdim, share(r, p, source.num_elements(tdim))?)?;
    let cells = layout.cells(&top.blocks)?;
    drop(top);
    let mut lower = Vec::new();
    for dim in 0..tdim {
        let range = share(r, p, source.num_elements(dim))?;
        lower.extend(source.elements(dim, range)?.blocks);
    }
    let groups = layout.tag_groups(&lower)?;
    drop(lower);
    Ok(Chunk {
        tdim,
        num_partitions: layout.num_partitions,
        nodes: source.nodes(share(r, p, source.num_nodes())?)?,
        cells,
        groups,
    })
}

/// What rank 0 reads: the chunk of every rank, the physical names, and the coordinates of the
/// file's own nodes (before the serendipity upgrade), from which D_GEO is inferred.
struct Prepared {
    chunks: Vec<Chunk>,
    names: Vec<PhysicalName>,
    coords: Vec<[f64; 3]>,
}

/// Rank 0's read of the file at `path` for `p` ranks (rule 2).
fn prepare(path: &Path, p: usize) -> Result<Prepared> {
    let file = gridcl_io::gmsh::read_file(path)?;
    let headers: Vec<BlockHeader> = file.element_blocks.iter().map(BlockHeader::of).collect();
    let layout = Layout::new(&file, &headers)?;
    let coords = file
        .node_blocks
        .iter()
        .flat_map(|b| b.coords.iter().copied())
        .collect();
    let completed = complete(&file, &layout)?;
    let whole = completed.as_ref().unwrap_or(&file);
    let chunks = (0..p)
        .map(|r| chunk_of(Source::Memory(whole), &layout, r, p))
        .collect::<Result<_>>()?;
    Ok(Prepared {
        chunks,
        names: file.physical_names.clone(),
        coords,
    })
}

/// The Gmsh type id of the Lagrange element (`cell`, `degree`): the wire code of a block.
fn type_id(cell: CellType, degree: u32) -> Result<i64> {
    Ok(i64::from(lagrange_element_type(cell, degree)?.id))
}

/// `len` as a wire word.
fn word(len: usize) -> i64 {
    i64::try_from(len).expect("lengths fit in i64")
}

/// `chunk` as i64 words and f64 coordinates.
fn encode(chunk: &Chunk) -> Result<(Vec<i64>, Vec<f64>)> {
    let mut w = vec![word(chunk.tdim), word(chunk.num_partitions)];
    w.push(word(chunk.nodes.tags.len()));
    w.extend(&chunk.nodes.tags);
    w.push(word(chunk.cells.len()));
    for c in &chunk.cells {
        w.extend([type_id(c.cell, c.degree)?, word(c.tags.len())]);
        w.extend(&c.nodes);
        w.extend(&c.tags);
        w.extend(c.elementary.iter().map(|&e| i64::from(e)));
        w.push(word(c.physical.len()));
        w.extend(
            c.physical
                .iter()
                .flat_map(|&(i, v)| [i64::from(i), i64::from(v)]),
        );
        w.push(word(c.partitions.len()));
        w.extend(c.partitions.iter().map(|&q| i64::from(q)));
    }
    w.push(word(chunk.groups.len()));
    for g in &chunk.groups {
        w.extend([word(g.dim), type_id(g.cell, 1)?, word(g.tags.len())]);
        w.extend(&g.corners);
        w.extend(&g.tags);
        w.extend(g.elementary.iter().map(|&e| i64::from(e)));
        w.push(word(g.physical.len()));
        w.extend(
            g.physical
                .iter()
                .flat_map(|&(i, v)| [i64::from(i), i64::from(v)]),
        );
    }
    let x = chunk.nodes.coords.iter().flatten().copied().collect();
    Ok((w, x))
}

/// Reads the words of [`encode`] back.
struct Words<'a>(&'a [i64]);

impl Words<'_> {
    fn take(&mut self, n: usize) -> &[i64] {
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        head
    }

    fn len(&mut self) -> usize {
        self.take(1)[0] as usize
    }

    fn i32s(&mut self, n: usize) -> Vec<i32> {
        self.take(n).iter().map(|&v| v as i32).collect()
    }

    fn pairs(&mut self) -> Vec<(u32, i32)> {
        let n = self.len();
        self.take(2 * n)
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[i, v]| (i as u32, v as i32))
            .collect()
    }
}

/// The chunk of [`encode`]'s `words` and `coords`.
fn decode(words: &[i64], coords: &[f64]) -> Result<Chunk> {
    let mut w = Words(words);
    let tdim = w.len();
    let num_partitions = w.len();
    let n = w.len();
    let nodes = NodeChunk {
        tags: w.take(n).to_vec(),
        coords: coords.as_chunks::<3>().0.to_vec(),
    };
    let mut cells = Vec::new();
    for _ in 0..w.len() {
        let ty = element_type(w.take(1)[0] as i32)?;
        let n = w.len();
        let k = gridcl_refcell::num_lagrange_nodes(ty.cell, ty.degree);
        let nodes = w.take(n * k).to_vec();
        let tags = w.take(n).to_vec();
        let elementary = w.i32s(n);
        let physical = w.pairs();
        let np = w.len();
        cells.push(CellChunk {
            cell: ty.cell,
            degree: ty.degree,
            nodes,
            tags,
            elementary,
            physical,
            partitions: w.i32s(np),
        });
    }
    let mut groups = Vec::new();
    for _ in 0..w.len() {
        let dim = w.len();
        let cell = element_type(w.take(1)[0] as i32)?.cell;
        let n = w.len();
        let nv = gridcl_refcell::vertices(cell).len();
        groups.push(GmshTagGroup {
            dim,
            cell,
            corners: w.take(n * nv).to_vec(),
            tags: w.take(n).to_vec(),
            elementary: w.i32s(n),
            physical: w.pairs(),
        });
    }
    Ok(Chunk {
        tdim,
        num_partitions,
        nodes,
        cells,
        groups,
    })
}

/// Rank 0's chunks sent to their ranks: every rank's own chunk, and on rank 0 the names and
/// the file's coordinates (empty elsewhere). Collective.
fn scatter(
    comm: &dyn CommDyn,
    prepared: Option<Prepared>,
) -> Result<(Chunk, Vec<PhysicalName>, Vec<[f64; 3]>)> {
    let p = comm.size();
    let mut words = Vec::new();
    let mut coords = Vec::new();
    let mut word_counts = vec![0; p];
    let mut coord_counts = vec![0; p];
    let mut own = None;
    let mut rest = (Vec::new(), Vec::new());
    if let Some(prepared) = prepared {
        let encoded = (|| {
            let mut chunks = prepared.chunks.into_iter();
            own = chunks.next();
            for (r, chunk) in chunks.enumerate() {
                let (w, x) = encode(&chunk)?;
                word_counts[r + 1] = w.len();
                coord_counts[r + 1] = x.len();
                words.extend(w);
                coords.extend(x);
            }
            Ok(())
        })();
        agree(comm, encoded)?;
        rest = (prepared.names, prepared.coords);
    } else {
        agree(comm, Ok(()))?;
    }
    let (w, _) = comm.all_to_all_v(&words, &word_counts)?;
    drop(words);
    let (x, _) = comm.all_to_all_v(&coords, &coord_counts)?;
    drop(coords);
    let chunk = match own {
        Some(chunk) => chunk,
        None => decode(&w, &x)?,
    };
    Ok((chunk, rest.0, rest.1))
}

/// The index and layout of the file at `path` if it can be read in chunks: binary, without
/// serendipity elements, and scanned without error.
fn scan(path: &Path) -> Option<(MshIndex, Layout)> {
    let index = MshIndex::scan(path).ok().filter(MshIndex::is_binary)?;
    let layout = Layout::new(index.sections(), &index.block_headers()).ok()?;
    (!layout.serendipity).then_some((index, layout))
}

/// This rank's chunk, the physical names (on the rank-0 path, empty except on rank 0) and
/// D_GEO, `gdim` if given, of the file at `path`. Collective.
fn local(
    comm: &dyn CommDyn,
    path: &Path,
    gdim: Option<usize>,
) -> Result<(Chunk, Vec<PhysicalName>, usize)> {
    let (r, p) = (comm.rank(), comm.size());
    let scanned = scan(path);
    if comm.min(u64::from(scanned.is_some())) == 1 {
        let (index, layout) = scanned.expect("every rank scanned the file");
        let chunk = agree(comm, chunk_of(Source::Disk(&index), &layout, r, p))?;
        let gdim = geometric_dim(layout.tdim, &chunk.nodes.coords, comm, gdim)?;
        return Ok((chunk, index.sections().physical_names.clone(), gdim));
    }
    drop(scanned);
    let prepared = if r == 0 {
        prepare(path, p).map(Some)
    } else {
        Ok(None)
    };
    let prepared = agree(comm, prepared)?;
    let (chunk, names, file_coords) = scatter(comm, prepared)?;
    let gdim = geometric_dim(chunk.tdim, &file_coords, comm, gdim)?;
    Ok((chunk, names, gdim))
}

/// `Parse` for stage 6's `IndexOutOfRange`: a node of a D_TOP element that `$Nodes` lacks
/// (rule 3).
fn missing_node(e: GridError) -> GridError {
    match e {
        GridError::IndexOutOfRange { index, .. } => GridError::Parse {
            format: "Gmsh",
            offset: 0,
            message: format!(
                "an element references node {index}, which is not in $Nodes \
                 (no byte offset: the file's content)"
            ),
        },
        other => other,
    }
}

/// The destination rank of each of this rank's cells, `partition − 1` in block order, if the
/// partitioner is `Block` (recognised by its name), the file has `comm.size()` partitions and
/// every cell lies in one of them (rule 5); `None` on every rank otherwise. Collective.
fn file_destinations(comm: &dyn CommDyn, chunk: &Chunk, partitioner: &Shared) -> Option<Vec<i32>> {
    let honour = partitioner.0.name() == "block" && chunk.num_partitions == comm.size();
    let dest: Option<Vec<i32>> = honour
        .then(|| {
            chunk
                .cells
                .iter()
                .flat_map(|c| &c.partitions)
                .map(|&q| q - 1)
                .collect()
        })
        .filter(|d: &Vec<i32>| d.iter().all(|&d| d >= 0));
    // `honour` is the same on every rank; the partitions of every rank's cells are not.
    (comm.min(u64::from(dest.is_some())) == 1).then_some(dest?)
}

/// The mesh of `chunk` on `comm`: nodes by tag, cells with their tag values, the
/// lower-dimensional elements as physical and elementary tag inputs whose error ids and
/// element ids are their element tags, and the names `names`; cells go to the ranks of their
/// file partitions under rule 5. Collective.
fn build<F, TD, GD, C>(
    client: &Client,
    comm: C,
    chunk: Chunk,
    names: &[PhysicalName],
    options: &GmshOptions,
    partitioner: Shared,
) -> Result<Mesh<F, TD, GD, C>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Cell: TryFrom<CellType>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let destinations = file_destinations(&comm, &chunk, &partitioner);
    let Chunk {
        nodes,
        cells,
        groups,
        ..
    } = chunk;
    let coords: Vec<f64> = nodes
        .coords
        .iter()
        .flat_map(|x| x[..gd].iter().copied())
        .collect();
    let mut builder = MeshBuilder::<F, TD, GD>::new(client).nodes(&coords, Some(&nodes.tags));
    drop((coords, nodes));
    for c in cells {
        let Ok(cell) = TD::Cell::try_from(c.cell) else {
            unreachable!("cell blocks have dimension D_TOP");
        };
        let element = CoordinateElement::<TD> {
            cell,
            degree: c.degree,
            family: NodeFamily::Equispaced,
        };
        builder = builder.tagged_cell_block(element, c.nodes, c.tags, c.physical, c.elementary);
    }
    let mut element_ids = Vec::new();
    for g in groups {
        let nv = gridcl_refcell::vertices(g.cell).len();
        let mut corners = Vec::with_capacity(g.physical.len() * nv);
        let mut values = Vec::with_capacity(g.physical.len());
        let mut errors = Vec::with_capacity(g.physical.len());
        for &(e, v) in &g.physical {
            let e = e as usize;
            corners.extend_from_slice(&g.corners[e * nv..(e + 1) * nv]);
            values.push(v);
            errors.push(g.tags[e]);
        }
        builder = builder
            .tag_entries(
                TagKind::Elementary,
                g.cell,
                g.corners,
                g.elementary,
                g.tags.clone(),
            )
            .tag_entries(TagKind::Physical, g.cell, corners, values, errors);
        element_ids.extend(g.tags);
    }
    let names: Vec<(&str, usize, i32)> = names
        .iter()
        .map(|n| (n.name.as_str(), n.dim, n.tag))
        .collect();
    if let Some(dest) = destinations {
        builder = builder.destinations(dest);
    }
    builder
        .element_ids(element_ids)
        .tag_names(&names)
        .ghosts(options.ghosts)
        .ordering(options.ordering)
        .partitioner(partitioner)
        .build_on(comm)
        .map_err(missing_node)
}

/// Reads the Gmsh MSH 4.1 file at `path` into a mesh on `comm` whose dimensions the file
/// decides (docs/design.md, "Gmsh MSH 4.1"; ADR-0022, ADR-0035). Collective.
///
/// - D_TOP is the highest element dimension; its elements form the cell blocks, in `CellType`
///   order with cells in file order. D_GEO is inferred from the nodes, or fixed by
///   [`GmshOptions::gdim`].
/// - Node tags are the vertex origin ids and element tags the cell origin ids; serendipity
///   cells are upgraded to complete P2.
/// - Physical groups and elementary tags become the mesh's [`TagRegistry`](crate::TagRegistry):
///   cells carry theirs, lower-dimensional elements are matched against the mesh's entities
///   (created for each tagged dimension).
/// - Each rank takes an even chunk of the nodes and of the elements of each dimension and
///   builds the mesh with the others through the parallel creation pipeline, with the
///   partitioner, ghost mode and cell order of `options`. A binary file without serendipity
///   elements is read in chunks by every rank; any other file is read by rank 0, which
///   scatters the chunks (ADR-0035, rules 1, 2 and 6).
/// - With [`Block`](crate::Block), a partitioned file whose partition count equals the rank
///   count keeps its partitions: each cell goes to rank `partition − 1` (rule 5). Ghost
///   entities of the file are ignored.
/// - Reading into an f32 mesh narrows the coordinates; rank 0 logs one warning per file, and
///   one if the default partitioner falls back to `Block`.
///
/// # Errors
///
/// Agreed over the ranks: a failing rank returns its error, the others `RemoteFailure`.
///
/// - `UnsupportedPrecision` if the device lacks `F`;
/// - `Io`, `Parse`, `UnsupportedElementType`, `DuplicateOriginId`, `DuplicateCellBlock`,
///   `DimensionMismatch` as in `gridcl_io::gmsh::to_gmsh_mesh`;
/// - `UnmatchedTagElement` for a lower-dimensional element that is not an entity of the mesh,
///   with its element tag (on several ranks, the element of lowest input position);
/// - errors of [`MeshBuilder::build_on`] and of the partitioner; allocation and device errors.
///
/// # Examples
///
/// ```
/// use gridcl::prelude::*;
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/two_regions_p1.msh");
/// let client = gridcl_testing::test_client();
/// let any: AnyMesh<f32> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
/// let mesh = PlanarMesh::<f32>::try_from(any)?;
/// assert!((mesh.measure(2)? - 2.0).abs() < 1e-5);
/// // The interface between the two regions, by name.
/// let interface = mesh.tags().named(Codim::Bnd, "interface")?;
/// assert_eq!(interface.indices.len(), 3);
/// # Ok::<(), GridError>(())
/// ```
///
/// On three threads, with ghost cells across facets:
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl_testing::ThreadComm;
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/hybrid_p1.msh");
/// let owned = ThreadComm::run(3, |comm| {
///     let client = gridcl_testing::test_client();
///     let options = GmshOptions::default().ghosts(GhostMode::SharedFacet);
///     let mesh = VolumeMesh::<f32, ThreadComm>::try_from(gmsh::read(&client, comm.clone(), path, options)?)?;
///     let owned: usize = mesh.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
///     Ok::<_, GridError>(comm.sum(owned as u64))
/// });
/// let serial = VolumeMesh::<f32>::try_from(gmsh::read(&gridcl_testing::test_client(), Serial, path, GmshOptions::default())?)?;
/// let cells: usize = serial.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
/// for total in owned {
///     assert_eq!(total?, cells as u64);
/// }
/// # Ok::<(), GridError>(())
/// ```
pub fn read<F: Real, C: Comm>(
    client: &Client,
    comm: C,
    path: impl AsRef<Path>,
    options: GmshOptions,
) -> Result<AnyMesh<F, C>> {
    agree(&comm, check_precision::<F>(client))?;
    let path = path.as_ref();
    log_memory(client, "read (start)");
    let (chunk, names, gdim) = local(&comm, path, options.gdim)?;
    log_memory(client, "read (chunk)");
    let tdim = chunk.tdim;
    let (partitioner, fallback) = options.partitioner_on(&comm);
    if comm.rank() == 0 {
        if size_of::<F>() < size_of::<f64>() {
            log::warn!(
                "{}: reading into an f32 mesh narrows the coordinates to single precision",
                path.display()
            );
        }
        if fallback && comm.size() > 1 {
            log::warn!(
                "{}: no partitioner set and neither `ptscotch` (with an MPI communicator) nor \
                 `metis` available; partitioning with Block",
                path.display()
            );
        }
    }
    let p = partitioner;
    let o = &options;
    Ok(match (tdim, gdim) {
        (1, 1) => AnyMesh::Curve1(build::<F, D1, D1, C>(client, comm, chunk, &names, o, p)?),
        (1, 2) => AnyMesh::Curve2(build::<F, D1, D2, C>(client, comm, chunk, &names, o, p)?),
        (1, 3) => AnyMesh::Curve3(build::<F, D1, D3, C>(client, comm, chunk, &names, o, p)?),
        (2, 2) => AnyMesh::Planar(build::<F, D2, D2, C>(client, comm, chunk, &names, o, p)?),
        (2, 3) => AnyMesh::Surface(build::<F, D2, D3, C>(client, comm, chunk, &names, o, p)?),
        (3, 3) => AnyMesh::Volume(build::<F, D3, D3, C>(client, comm, chunk, &names, o, p)?),
        (tdim, gdim) => {
            unreachable!("geometric_dim gives 1 <= tdim <= gdim <= 3, not ({tdim}, {gdim})")
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gridcl-io/tests/fixtures/gmsh")
            .join(name)
    }

    /// The shares of 1–8 ranks cover 0..n in order, without gaps.
    #[test]
    fn shares_cover_in_order() {
        for n in [0, 1, 5, 17] {
            for p in 1..=8 {
                let ranges: Vec<_> = (0..p).map(|r| share(r, p, n).unwrap()).collect();
                assert_eq!(ranges[0].start, 0);
                assert_eq!(ranges[p - 1].end, n);
                for w in ranges.windows(2) {
                    assert_eq!(w[0].end, w[1].start);
                }
            }
        }
    }

    /// On the chunk path, every rank of 1–8 reads only its share of the node and element data
    /// (Gmsh writes 8-byte `size_t`: 8 + 3 × 8 bytes per node, 8 per tag of an element record),
    /// and its chunk equals the one rank 0 converts in memory.
    #[test]
    fn ranks_read_only_their_ranges() {
        for name in [
            "two_regions_p1-binary.msh",
            "hybrid_p2-binary.msh",
            "interval_p2-binary.msh",
            "partitioned_ghosts_p1-binary.msh",
        ] {
            let path = fixture(name);
            let file = gridcl_io::gmsh::read_file(&path).unwrap();
            let memory = Source::Memory(&file);
            for p in 1..=8 {
                for r in 0..p {
                    let (index, layout) = scan(&path).expect("a binary file");
                    let chunk = chunk_of(Source::Disk(&index), &layout, r, p).unwrap();
                    assert_eq!(chunk, chunk_of(memory, &layout, r, p).unwrap(), "{name}");
                    let nodes = share(r, p, memory.num_nodes()).unwrap().len();
                    let mut bytes = nodes * (8 + 3 * 8);
                    for dim in 0..=layout.tdim {
                        let range = share(r, p, memory.num_elements(dim)).unwrap();
                        let chunk = memory.elements(dim, range).unwrap();
                        bytes += chunk
                            .blocks
                            .iter()
                            .map(|b| 8 * (b.tags.len() + b.nodes.len()))
                            .sum::<usize>();
                    }
                    assert_eq!(index.bytes_read(), bytes as u64, "{name}, rank {r} of {p}");
                }
            }
        }
    }

    /// ASCII files, files with serendipity elements and files that do not scan take the
    /// rank-0 path.
    #[test]
    fn chunk_path_needs_binary_without_serendipity() {
        for name in [
            "two_regions_p1.msh",
            "hybrid_s2-binary.msh",
            "quadrilateral_s2-binary.msh",
            "missing.msh",
        ] {
            assert!(scan(&fixture(name)).is_none(), "{name}");
        }
    }

    /// Every rank's chunk of a file with physical groups in every dimension, partitions and
    /// serendipity cells survives the wire.
    #[test]
    fn chunks_round_trip_through_the_wire() {
        for name in [
            "two_regions_p1.msh",
            "hybrid_s2-binary.msh",
            "partitioned_ghosts_p1.msh",
            "interval_p2.msh",
        ] {
            let prepared = prepare(&fixture(name), 3).unwrap();
            for chunk in &prepared.chunks {
                let (w, x) = encode(chunk).unwrap();
                assert_eq!(&decode(&w, &x).unwrap(), chunk, "{name}");
            }
        }
    }
}
