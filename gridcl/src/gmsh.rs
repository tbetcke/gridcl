//! Gmsh MSH 4.1 meshes: [`read`] and [`GmshWriter`] (docs/design.md, "Import and export /
//! Gmsh MSH 4.1"; ADR-0022). The file model and the conversion to a host mesh description are
//! in `gridcl_io::gmsh`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, GridError, Real, Result, SortKeys, check_precision};
use gridcl_dims::{CellType, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, NodeFamily};
use gridcl_io::gmsh::GmshMesh;
use gridcl_parallel::Comm;

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
    let mut registry = TagRegistry::new(tdim, names);
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

/// Writes meshes as Gmsh MSH 4.1 files, ASCII by default.
#[derive(Clone, Debug)]
pub struct GmshWriter {
    path: PathBuf,
    binary: bool,
}

impl GmshWriter {
    /// A writer of the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        GmshWriter {
            path: path.as_ref().to_path_buf(),
            binary: false,
        }
    }

    /// Binary (little-endian, 8-byte `size_t`) instead of ASCII.
    pub fn binary(self, binary: bool) -> Self {
        GmshWriter { binary, ..self }
    }

    /// Writes `mesh`.
    ///
    /// # Errors
    ///
    /// `NotYetImplemented` for more than one rank (until M8); `UnsupportedNodeFamily` for GLL
    /// geometry (until M5); `Io`; device errors.
    pub fn write<F, TD, GD, C>(&self, mesh: &Mesh<F, TD, GD, C>) -> Result<()>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let _ = (mesh, &self.path, self.binary);
        todo!("M4 T13")
    }

    /// Writes the mesh behind `mesh`.
    ///
    /// # Errors
    ///
    /// As [`write`](GmshWriter::write).
    pub fn write_any<F: Real, C: Comm>(&self, mesh: AnyMeshRef<'_, F, C>) -> Result<()> {
        crate::with_mesh!(mesh, m => self.write(m))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use cubecl::prelude::CubeElement;
    use gridcl_parallel::{CommDyn, DType, Neighbourhood, ReduceOp, Serial};

    use super::*;
    use crate::{Codim, PlanarMesh};

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

    /// A communicator of two ranks; reading on it is refused before any collective.
    #[derive(Clone)]
    struct TwoRanks;

    impl CommDyn for TwoRanks {
        fn rank(&self) -> usize {
            0
        }
        fn size(&self) -> usize {
            2
        }
        fn barrier(&self) {
            unreachable!()
        }
        fn all_reduce_bytes(&self, _: &mut [u8], _: DType, _: ReduceOp) {
            unreachable!()
        }
        fn exscan_bytes(&self, _: &mut [u8], _: DType, _: ReduceOp) {
            unreachable!()
        }
        fn all_to_all_v_bytes(
            &self,
            _: &[u8],
            _: &[usize],
            _: &mut [u8],
            _: &[usize],
            _: DType,
        ) -> Result<()> {
            unreachable!()
        }
        fn neighbour_all_to_all_v_bytes(
            &self,
            _: &Neighbourhood,
            _: &[u8],
            _: &[usize],
            _: &mut [u8],
            _: &[usize],
            _: DType,
        ) -> Result<()> {
            unreachable!()
        }
        #[cfg(feature = "mpi")]
        fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
            None
        }
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
}
