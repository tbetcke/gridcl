//! Gmsh MSH 4.1 meshes: [`read`] and [`GmshWriter`] (docs/design.md, "Import and export /
//! Gmsh MSH 4.1"; ADR-0022). The file model and the conversion to a host mesh description are
//! in `gridcl_io::gmsh`.

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, GridError, Real, Result, SortKeys, check_precision};
use gridcl_dims::{CellType, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, DeviceGrid, NodeFamily};
use gridcl_io::gmsh::{Encoding, GmshCellBlock, GmshMesh, GmshTagGroup, PhysicalName};
use gridcl_parallel::Comm;
use gridcl_topology::decode_origin_ids;

use crate::{AnyMesh, AnyMeshRef, Mesh, MeshBuilder, MeshTags, TagRegistry};

/// Options of [`read`]. `Default` infers D_GEO.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// // Read a planar mesh into three dimensions.
/// let options = GmshOptions::default().gdim(3);
/// # let _ = options;
/// ```
#[derive(Clone, Debug, Default)]
pub struct GmshOptions {
    gdim: Option<usize>,
}

impl GmshOptions {
    /// Fixes D_GEO: 1–3, at least D_TOP and at least the inferred D_GEO.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/triangle_p1.msh");
    /// let client = gridcl_testing::test_client();
    /// let any = gmsh::read::<f32, _>(&client, Serial, path, GmshOptions::default().gdim(3))?;
    /// assert_eq!((any.tdim(), any.gdim()), (2, 3));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn gdim(self, gdim: usize) -> Self {
        GmshOptions { gdim: Some(gdim) }
    }
}

/// (flat entity index, value) entries of one dimension, sorted and without repeats.
type Entries = Vec<(u32, i32)>;

/// The physical and elementary entries of each dimension.
#[derive(Default)]
struct HostTags {
    physical: BTreeMap<usize, Entries>,
    elementary: BTreeMap<usize, Entries>,
}

/// The mesh of `host` on `client` and `comm`, with its tags.
fn build<F, TD, GD, C>(client: &Client, comm: C, host: &GmshMesh) -> Result<Mesh<F, TD, GD, C>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Cell: TryFrom<CellType>,
    GD: Dim,
    C: Comm,
{
    let mut builder =
        MeshBuilder::<F, TD, GD>::new(client).nodes(&host.nodes, Some(&host.node_tags));
    for b in &host.blocks {
        let Ok(cell) = TD::Cell::try_from(b.cell) else {
            unreachable!("cell blocks have dimension D_TOP");
        };
        let element = CoordinateElement::<TD> {
            cell,
            degree: b.degree,
            family: NodeFamily::Equispaced,
        };
        builder = builder.cell_block(element, &b.nodes, Some(&b.tags));
    }
    let (mut mesh, vertex_of_node) = builder.build_with_comm(comm)?;
    let tdim = host.tdim;
    let mut tags = HostTags::default();

    // Cells: flat numbering over the blocks in block order.
    let mut first = 0u32;
    for b in &host.blocks {
        let physical = tags.physical.entry(tdim).or_default();
        physical.extend(b.physical.iter().map(|&(c, p)| (first + c, p)));
        let elementary = tags.elementary.entry(tdim).or_default();
        elementary.extend(
            b.elementary
                .iter()
                .enumerate()
                .map(|(c, &e)| (first + c as u32, e)),
        );
        first += u32::try_from(b.tags.len()).map_err(|_| GridError::TooLarge)?;
    }

    // Tag elements: corner tags -> local vertex indices on the host, then device matching.
    let index_of: std::collections::HashMap<i64, usize> = host
        .node_tags
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();
    for g in &host.tag_groups {
        mesh.create_entities(g.dim)?;
        let nv = gridcl_refcell::vertices(g.cell).len();
        let mut keys = Vec::with_capacity(g.corners.len());
        for (e, corners) in g.corners.chunks_exact(nv).enumerate() {
            let mut key = Vec::with_capacity(nv);
            for t in corners {
                let v = index_of
                    .get(t)
                    .map(|&i| vertex_of_node[i])
                    .filter(|&v| v != u32::MAX)
                    .ok_or(GridError::UnmatchedTagElement {
                        dim: g.dim,
                        tag: g.tags[e],
                    })?;
                key.push(v);
            }
            key.sort_unstable();
            keys.extend(key);
        }
        let keys = SortKeys::from_host(client, &keys, nv)?;
        // An I/O copy (ADR-0022, item 15): it names the first unmatched element.
        let found = mesh.topology().match_entities(g.dim, &keys)?.to_host()?;
        if let Some(e) = found.iter().position(|&f| f == u32::MAX) {
            return Err(GridError::UnmatchedTagElement {
                dim: g.dim,
                tag: g.tags[e],
            });
        }
        let physical = tags.physical.entry(g.dim).or_default();
        physical.extend(g.physical.iter().map(|&(e, p)| (found[e as usize], p)));
        let elementary = tags.elementary.entry(g.dim).or_default();
        elementary.extend(g.elementary.iter().zip(&found).map(|(&v, &f)| (f, v)));
    }

    let names = host
        .names
        .iter()
        .map(|n| (n.name.clone(), n.dim, n.tag))
        .collect();
    let mut registry = TagRegistry::new(client, tdim, names);
    let upload = |dim: usize, entries: &mut Entries| -> Result<MeshTags> {
        entries.sort_unstable();
        entries.dedup();
        let (indices, values): (Vec<u32>, Vec<i32>) = entries.iter().copied().unzip();
        Ok(MeshTags {
            dim,
            indices: DeviceBuffer::from_host(client, &indices)?,
            values: DeviceBuffer::from_host(client, &values)?,
        })
    };
    for (dim, mut entries) in tags.physical {
        if entries.is_empty() {
            continue;
        }
        let t = upload(dim, &mut entries)?;
        let mut sizes: BTreeMap<i32, usize> = BTreeMap::new();
        for &(_, v) in &entries {
            *sizes.entry(v).or_default() += 1;
        }
        registry.set_physical(t, sizes);
    }
    for (dim, mut entries) in tags.elementary {
        if entries.is_empty() {
            continue;
        }
        registry.set_elementary(upload(dim, &mut entries)?);
    }
    mesh.set_tags(registry);
    Ok(mesh)
}

/// Reads the Gmsh MSH 4.1 file at `path` into a mesh whose dimensions the file decides
/// (docs/design.md, "Gmsh MSH 4.1"; ADR-0022).
///
/// - D_TOP is the highest element dimension; its elements form the cell blocks, in `CellType`
///   order with cells in file order. D_GEO is inferred from the nodes, or fixed by
///   [`GmshOptions::gdim`].
/// - Node tags are the vertex origin ids and element tags the cell origin ids; serendipity
///   cells are upgraded to complete P2.
/// - Physical groups and elementary tags become the mesh's [`TagRegistry`]: lower-dimensional
///   elements are matched against the mesh's entities (created for each tagged dimension) on
///   the device, cells are tagged directly.
/// - Reading into an f32 mesh narrows the coordinates and logs one warning per file.
///
/// # Errors
///
/// - `NotYetImplemented` for more than one rank (until M8);
/// - `UnsupportedPrecision` if the device lacks `F`;
/// - `Io`, `Parse`, `UnsupportedElementType`, `DuplicateOriginId`, `DuplicateCellBlock`,
///   `DimensionMismatch` as in `gridcl_io::gmsh::to_gmsh_mesh`;
/// - `UnmatchedTagElement` for a lower-dimensional element that is not an entity of the mesh;
/// - errors of [`MeshBuilder::build`]; allocation and device errors.
///
/// # Example
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
pub fn read<F: Real, C: Comm>(
    client: &Client,
    comm: C,
    path: impl AsRef<Path>,
    options: GmshOptions,
) -> Result<AnyMesh<F, C>> {
    if comm.size() > 1 {
        return Err(GridError::NotYetImplemented("parallel Gmsh read (M8)"));
    }
    check_precision::<F>(client)?;
    let path = path.as_ref();
    let file = gridcl_io::gmsh::read_file(path)?;
    let host = gridcl_io::gmsh::to_gmsh_mesh(&file, &comm, options.gdim)?;
    if size_of::<F>() < size_of::<f64>() {
        log::warn!(
            "{}: reading into an f32 mesh narrows the coordinates to single precision",
            path.display()
        );
    }
    Ok(match (host.tdim, host.gdim) {
        (1, 1) => AnyMesh::Curve1(build::<F, D1, D1, C>(client, comm, &host)?),
        (1, 2) => AnyMesh::Curve2(build::<F, D1, D2, C>(client, comm, &host)?),
        (1, 3) => AnyMesh::Curve3(build::<F, D1, D3, C>(client, comm, &host)?),
        (2, 2) => AnyMesh::Planar(build::<F, D2, D2, C>(client, comm, &host)?),
        (2, 3) => AnyMesh::Surface(build::<F, D2, D3, C>(client, comm, &host)?),
        (3, 3) => AnyMesh::Volume(build::<F, D3, D3, C>(client, comm, &host)?),
        (tdim, gdim) => {
            unreachable!("to_gmsh_mesh gives 1 <= tdim <= gdim <= 3, not ({tdim}, {gdim})")
        }
    })
}

/// Writes meshes as Gmsh MSH 4.1 files, ASCII by default (ADR-0022, item 10).
///
/// - Node tags are the vertex origin ids; the other nodes are numbered after the largest vertex
///   id. Element tags are the cell origin ids; tag elements are numbered after the largest cell
///   id. Tags below 1 are shifted by one constant per kind so that the smallest is 1.
/// - Tagged entities of dimension < D_TOP are written as linear tag elements from
///   `EntityBlock::vertices` (canonical A.6 orientation); cells carry their physical and
///   elementary tags. A mesh without tags gets one entity with tag 1.
/// - `$Entities` is rebuilt from the elementary tags: one entity per (elementary tag, set of
///   physical tags).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
/// let path = std::env::temp_dir().join(format!("gridcl-doc-{}.msh", std::process::id()));
/// gmsh::GmshWriter::new(&path).binary(true).write(&mesh)?;
/// let back = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, &path, GmshOptions::default())?)?;
/// std::fs::remove_file(&path).ok();
/// assert_eq!(back.cell_blocks()[0].index_map.num_local(), 4);
/// assert!((back.measure(1)? - 1.0).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
#[derive(Clone, Debug)]
pub struct GmshWriter {
    path: PathBuf,
    binary: bool,
}

/// The (entity index → values) map of `tags`, downloaded.
fn values_by_index(tags: Option<&MeshTags>) -> Result<BTreeMap<u32, Vec<i32>>> {
    let mut out: BTreeMap<u32, Vec<i32>> = BTreeMap::new();
    if let Some(t) = tags {
        let indices = t.indices.to_host()?;
        let values = t.values.to_host()?;
        for (i, v) in indices.into_iter().zip(values) {
            out.entry(i).or_default().push(v);
        }
    }
    Ok(out)
}

/// The elementary tag of entity `i`: its first elementary value, or 1 without one.
fn elementary_of(elementary: &BTreeMap<u32, Vec<i32>>, i: u32) -> i32 {
    elementary.get(&i).map_or(1, |v| v[0])
}

/// The host description of `mesh` (one rank): nodes converted to equispaced nodes if a block is
/// GLL and with the origin added back (ADR-0022, item 9; ADR-0024, item 7), vertex origin
/// ids as node tags (other nodes' tags are renumbered by `from_gmsh_mesh`), cell blocks with
/// their tags, and one tag group per entity type of each tagged dimension below D_TOP.
fn download<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<GmshMesh>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let td = TD::N;
    let nodes = mesh.equispaced_host_nodes()?;
    let vertex_ids = decode_origin_ids(&mesh.vertex_origin_ids().to_host()?);
    let mut node_tags = vec![0i64; nodes.len() / gd];
    let tags = mesh.tags();
    let elementary = values_by_index(tags.elementary(td))?;
    let physical = values_by_index(tags.physical(td))?;
    let mut blocks = Vec::new();
    let mut first = 0u32;
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let geometry = mesh.geometry_block(b);
        let cell: CellType = block.cell.into();
        let dofmap = geometry.dofmap.data.to_host()?;
        let vertices = block.vertices.data.to_host()?;
        let k = geometry.dofmap.stride as usize;
        let nv = block.vertices.stride as usize;
        let cells = block.index_map.num_local();
        for c in 0..cells {
            for j in 0..nv {
                node_tags[dofmap[c * k + j] as usize] = vertex_ids[vertices[c * nv + j] as usize];
            }
        }
        let mut out = GmshCellBlock {
            cell,
            degree: geometry.element.degree,
            nodes: dofmap,
            tags: decode_origin_ids(&block.origin_ids.to_host()?),
            elementary: Vec::with_capacity(cells),
            physical: Vec::new(),
        };
        for c in 0..cells {
            let flat = first + u32::try_from(c).map_err(|_| GridError::TooLarge)?;
            out.elementary.push(elementary_of(&elementary, flat));
            if let Some(values) = physical.get(&flat) {
                out.physical.extend(values.iter().map(|&v| (c as u32, v)));
            }
        }
        first += u32::try_from(cells).map_err(|_| GridError::TooLarge)?;
        blocks.push(out);
    }

    let mut tag_groups = Vec::new();
    for dim in 0..td {
        let elementary = values_by_index(tags.elementary(dim))?;
        let physical = values_by_index(tags.physical(dim))?;
        if elementary.is_empty() && physical.is_empty() {
            continue;
        }
        for block in mesh.entity_blocks(dim)? {
            let n = u32::try_from(block.index_map.num_local()).map_err(|_| GridError::TooLarge)?;
            let range = block.offset..block.offset + n;
            let tagged: Vec<u32> = elementary
                .keys()
                .chain(physical.keys())
                .copied()
                .filter(|i| range.contains(i))
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            if tagged.is_empty() {
                continue;
            }
            let rows = block.vertices.data.to_host()?;
            let nv = block.vertices.stride as usize;
            let mut group = GmshTagGroup {
                dim,
                cell: block.cell,
                corners: Vec::with_capacity(tagged.len() * nv),
                tags: Vec::with_capacity(tagged.len()),
                elementary: Vec::with_capacity(tagged.len()),
                physical: Vec::new(),
            };
            for (e, &i) in tagged.iter().enumerate() {
                let local = (i - block.offset) as usize;
                group.corners.extend(
                    rows[local * nv..(local + 1) * nv]
                        .iter()
                        .map(|&v| vertex_ids[v as usize]),
                );
                group.tags.push(i64::from(i));
                group.elementary.push(elementary_of(&elementary, i));
                if let Some(values) = physical.get(&i) {
                    group.physical.extend(values.iter().map(|&v| (e as u32, v)));
                }
            }
            tag_groups.push(group);
        }
    }
    Ok(GmshMesh {
        tdim: td,
        gdim: gd,
        nodes,
        node_tags,
        blocks,
        tag_groups,
        names: tags
            .names()
            .map(|(name, dim, tag)| PhysicalName {
                dim,
                tag,
                name: name.to_owned(),
            })
            .collect(),
    })
}

impl GmshWriter {
    /// A writer of the file at `path`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::gmsh::GmshWriter;
    ///
    /// let writer = GmshWriter::new("mesh.msh").binary(true);
    /// # let _ = writer;
    /// ```
    pub fn new(path: impl AsRef<Path>) -> Self {
        GmshWriter {
            path: path.as_ref().to_path_buf(),
            binary: false,
        }
    }

    /// Binary (little-endian, 8-byte `size_t`) instead of ASCII.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::gmsh::GmshWriter;
    ///
    /// let ascii = GmshWriter::new("mesh.msh").binary(false);
    /// # let _ = ascii;
    /// ```
    pub fn binary(self, binary: bool) -> Self {
        GmshWriter { binary, ..self }
    }

    /// Writes `mesh`, copying it to the host.
    ///
    /// # Errors
    ///
    /// `NotYetImplemented` for more than one rank (until M8); `EntitiesNotCreated` for tags of a
    /// dimension without entities; `Io`; `TooLarge` if a tag overflows; device errors.
    ///
    /// GLL geometry is written as its equispaced interpolant, converted on the device (ADR-0022,
    /// item 9).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: VolumeMesh<f32> = gridcl_testing::cube_box(Cell3::Tetrahedron, 1, 2).build(&client)?;
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-write-{}.msh", std::process::id()));
    /// gmsh::GmshWriter::new(&path).write(&mesh)?;
    /// let text = std::fs::read_to_string(&path).unwrap();
    /// std::fs::remove_file(&path).ok();
    /// assert!(text.starts_with("$MeshFormat\n4.1 0 8\n"));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn write<F, TD, GD, C>(&self, mesh: &Mesh<F, TD, GD, C>) -> Result<()>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        if mesh.comm().size() > 1 {
            return Err(GridError::NotYetImplemented("parallel Gmsh write (M8)"));
        }
        let file = gridcl_io::gmsh::from_gmsh_mesh(&download(mesh)?)?;
        let io = |e: std::io::Error| GridError::Io(format!("{}: {e}", self.path.display()));
        let mut out = std::io::BufWriter::new(std::fs::File::create(&self.path).map_err(io)?);
        let encoding = if self.binary {
            Encoding::Binary
        } else {
            Encoding::Ascii
        };
        file.write_to(&mut out, encoding).map_err(|e| match e {
            GridError::Io(m) => GridError::Io(format!("{}: {m}", self.path.display())),
            other => other,
        })?;
        out.flush().map_err(io)
    }

    /// Writes the mesh behind `mesh`.
    ///
    /// # Errors
    ///
    /// As [`write`](GmshWriter::write).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let any = AnyMesh::Curve1(gridcl_testing::interval_box(4, 1).build::<f32, D1, D1>(&client)?);
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-any-{}.msh", std::process::id()));
    /// gmsh::GmshWriter::new(&path).write_any(any.as_ref())?;
    /// std::fs::remove_file(&path).ok();
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn write_any<F: Real, C: Comm>(&self, mesh: AnyMeshRef<'_, F, C>) -> Result<()> {
        crate::with_mesh!(mesh, m => self.write(m))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use cubecl::prelude::CubeElement;
    use gridcl_parallel::Serial;

    use super::*;
    use crate::test_comm::TwoRanks;
    use crate::{Codim, D2, PlanarMesh};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gridcl-io/tests/fixtures/gmsh")
            .join(name)
    }

    /// Warnings by thread, so that tests running in parallel see only their own.
    static RECORDS: Mutex<Vec<(ThreadId, String)>> = Mutex::new(Vec::new());

    struct Capture;

    impl log::Log for Capture {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Warn
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                let mut records = RECORDS.lock().unwrap();
                records.push((std::thread::current().id(), record.args().to_string()));
            }
        }
        fn flush(&self) {}
    }

    static LOGGER: Capture = Capture;

    fn my_warnings() -> Vec<String> {
        let me = std::thread::current().id();
        let records = RECORDS.lock().unwrap();
        records
            .iter()
            .filter(|r| r.0 == me)
            .map(|r| r.1.clone())
            .collect()
    }

    /// One `log::warn!` per f32 read, naming the file; none for f64.
    #[test]
    fn gmsh_read_warns_once_per_f32_file() {
        // Another test binary thread may have set it already; either way it is ours.
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Warn);
        let client = gridcl_testing::test_client();
        let path = fixture("triangle_p2.msh");
        read::<f32, _>(&client, Serial, &path, GmshOptions::default()).unwrap();
        let warnings = my_warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("triangle_p2.msh"), "{}", warnings[0]);
        if gridcl_testing::supports::<f64>(&client) {
            read::<f64, _>(&client, Serial, &path, GmshOptions::default()).unwrap();
            assert_eq!(my_warnings().len(), 1);
        }
        read::<f32, _>(
            &client,
            Serial,
            fixture("hybrid_p1.msh"),
            GmshOptions::default(),
        )
        .unwrap();
        assert_eq!(my_warnings().len(), 2);
    }

    /// A device without f64 gives `UnsupportedPrecision` before the file is read.
    #[test]
    fn gmsh_read_f64_needs_device_support() {
        let client = gridcl_testing::test_client();
        let mut props = client.properties().clone();
        props.features.types.elem.remove(&f64::cube_type());
        gridcl_device::testing::with_properties(props, || {
            let r = read::<f64, _>(
                &client,
                Serial,
                fixture("triangle_p1.msh"),
                GmshOptions::default(),
            );
            assert!(matches!(
                r,
                Err(GridError::UnsupportedPrecision { scalar: "f64", .. })
            ));
        });
    }

    #[test]
    fn gmsh_read_needs_one_rank() {
        let client = gridcl_testing::test_client();
        let r = read::<f32, _>(
            &client,
            TwoRanks,
            fixture("triangle_p1.msh"),
            GmshOptions::default(),
        );
        assert!(matches!(
            r,
            Err(GridError::NotYetImplemented("parallel Gmsh read (M8)"))
        ));
    }

    /// A tag element whose corners are vertices but not an edge of the mesh (the other diagonal
    /// of a square cut into two triangles); a missing file.
    #[test]
    fn gmsh_read_unmatched_tag_element_and_io() {
        let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
            $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
            $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 1 1 1\n3 2 4\n$EndElements\n";
        let path = std::env::temp_dir().join(format!("gridcl-t11-{}.msh", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let client = gridcl_testing::test_client();
        let r = read::<f32, _>(&client, Serial, &path, GmshOptions::default());
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(r, Err(GridError::UnmatchedTagElement { dim: 1, tag: 3 })),
            "{:?}",
            r.err()
        );
        let r = read::<f32, _>(
            &client,
            Serial,
            fixture("no-such-file.msh"),
            GmshOptions::default(),
        );
        assert!(matches!(r, Err(GridError::Io(_))));
    }

    /// Physical names without elements, in D_TOP and below: empty groups of size 0.
    #[test]
    fn gmsh_read_named_groups_without_elements() {
        let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
            $PhysicalNames\n2\n2 1 \"domain\"\n1 5 \"empty\"\n$EndPhysicalNames\n\
            $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
            $Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
        let path = std::env::temp_dir().join(format!("gridcl-m4r-{}.msh", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let client = gridcl_testing::test_client();
        let r = read::<f32, _>(&client, Serial, &path, GmshOptions::default());
        let _ = std::fs::remove_file(&path);
        let mesh = PlanarMesh::<f32>::try_from(r.unwrap()).unwrap();
        let tags = mesh.tags();
        for (codim, name, dim, value) in [(Codim::Vol, "domain", 2, 1), (Codim::Bnd, "empty", 1, 5)]
        {
            assert_eq!(tags.group_size(dim, value), Some(0), "{name}");
            let t = tags.named(codim, name).unwrap();
            assert_eq!(
                (t.dim, t.indices.len(), t.values.len()),
                (dim, 0, 0),
                "{name}"
            );
        }
    }

    /// Cells are tagged in flat numbering over the blocks; tag elements through the entities.
    #[test]
    fn gmsh_read_tags_cells_and_entities() {
        let client = gridcl_testing::test_client();
        let any = read::<f32, _>(
            &client,
            Serial,
            fixture("two_regions_p1.msh"),
            GmshOptions::default(),
        )
        .unwrap();
        let mesh = PlanarMesh::<f32>::try_from(any).unwrap();
        let tags = mesh.tags();
        // 26 triangles in block 0, 9 quadrilaterals in block 1.
        let quads = tags.named(Codim::Vol, "quadrilaterals").unwrap();
        assert_eq!(
            quads.indices.to_host().unwrap(),
            (26..35).collect::<Vec<u32>>()
        );
        assert_eq!(tags.named(Codim::Vol, "all").unwrap().indices.len(), 35);
        // "boundary" in dimensions 1 and 0.
        assert_eq!(
            tags.named(Codim::Bnd, "boundary").unwrap().indices.len(),
            18
        );
        assert_eq!(
            tags.named(Codim::BBnd, "boundary").unwrap().indices.len(),
            4
        );
        assert_eq!(tags.lookup(1, "interface"), Some(11));
        // Every cell and every tag element has an elementary tag.
        assert_eq!(tags.elementary(2).unwrap().indices.len(), 35);
        assert_eq!(tags.elementary(1).unwrap().indices.len(), 21);
        assert_eq!(tags.elementary(0).unwrap().indices.len(), 6);
    }

    /// A builder mesh of `family`: a 2 × 2 square of quadrilaterals of degree `p`, ids from 0.
    fn builder_mesh<F: Real>(family: NodeFamily, p: u32) -> Result<PlanarMesh<F>> {
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::square_box(crate::Cell2::Quadrilateral, 2, p);
        let b = &h.blocks[0];
        let element = CoordinateElement::<D2> {
            cell: crate::Cell2::Quadrilateral,
            degree: b.degree,
            family,
        };
        MeshBuilder::new(&client)
            .nodes(&h.nodes, None)
            .cell_block(element, &b.cells, None)
            .build()
    }

    /// Without tags and with origin ids from 0: written with ids shifted by one, read back with
    /// the same cells, coordinates and one elementary tag 1.
    #[test]
    fn gmsh_write_untagged_builder_mesh() {
        let client = gridcl_testing::test_client();
        let mesh = builder_mesh::<f32>(NodeFamily::Equispaced, 2).unwrap();
        let path = std::env::temp_dir().join(format!("gridcl-t13-{}.msh", std::process::id()));
        GmshWriter::new(&path).write(&mesh).unwrap();
        let back = PlanarMesh::<f32>::try_from(
            read::<f32, _>(&client, Serial, &path, GmshOptions::default()).unwrap(),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let ids =
            |m: &PlanarMesh<f32>| decode_origin_ids(&m.vertex_origin_ids().to_host().unwrap());
        let cells = |m: &PlanarMesh<f32>| {
            decode_origin_ids(&m.cell_blocks()[0].origin_ids.to_host().unwrap())
        };
        let shifted: Vec<i64> = ids(&mesh).iter().map(|i| i + 1).collect();
        assert_eq!(ids(&back), shifted);
        assert_eq!(
            cells(&back),
            cells(&mesh).iter().map(|i| i + 1).collect::<Vec<_>>()
        );
        assert_eq!(
            mesh.nodes().to_host().unwrap(),
            back.nodes().to_host().unwrap()
        );
        let elementary = back.tags().elementary(2).unwrap();
        assert!(elementary.values.to_host().unwrap().iter().all(|&v| v == 1));
        assert!(back.tags().physical(2).is_none() && back.tags().names().next().is_none());
        // One entity, tag 1, holding the node block.
        assert!(text.contains("$Entities\n0 0 1 0\n1 "), "{text}");
    }

    /// A P3 GLL mesh (the box's equispaced nodes taken as GLL nodes) is written as its
    /// equispaced interpolant: read back, every node equals `equispaced_nodes` with the origin
    /// added, to the precision's tolerance (ADR-0022, item 9).
    fn writes_gll_as_equispaced<F: Real>(name: &str, tol: f64) {
        let client = gridcl_testing::test_client();
        let mesh = builder_mesh::<F>(NodeFamily::Gll, 3).unwrap();
        let path =
            std::env::temp_dir().join(format!("gridcl-m5-gll-{name}-{}.msh", std::process::id()));
        GmshWriter::new(&path).write(&mesh).unwrap();
        let back = PlanarMesh::<F>::try_from(
            read::<F, _>(&client, Serial, &path, GmshOptions::default()).unwrap(),
        )
        .unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            back.geometry_block(0).element.family,
            NodeFamily::Equispaced,
            "{name}"
        );
        let absolute = |m: &PlanarMesh<F>, x: Vec<F>| -> Vec<f64> {
            let origin = m.origin();
            x.iter()
                .enumerate()
                .map(|(i, v)| v.widen() + origin.as_ref()[i % 2])
                .collect()
        };
        let want = absolute(
            &mesh,
            gridcl_geometry::equispaced_nodes(&mesh)
                .unwrap()
                .to_host()
                .unwrap(),
        );
        let got = absolute(&back, back.nodes().to_host().unwrap());
        let raw = absolute(&mesh, mesh.nodes().to_host().unwrap());
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= tol * b.abs().max(1.0),
                "{name}: coordinate {i}: {a} vs {b}"
            );
        }
        assert!(
            raw.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-3),
            "{name}: the conversion moves nodes"
        );
    }

    #[test]
    fn gmsh_write_converts_gll() {
        writes_gll_as_equispaced::<f32>("f32", 1e-5);
        if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
            writes_gll_as_equispaced::<f64>("f64", 1e-12);
        }
    }
}
