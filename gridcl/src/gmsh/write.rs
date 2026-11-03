//! [`GmshWriter`]: one file gathered on rank 0, or one partitioned file per rank (ADR-0022,
//! item 10; ADR-0035, item 11).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use gridcl_device::{GridError, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::DeviceGrid;
use gridcl_io::gmsh::{
    Encoding, FileNumbering, GmshCellBlock, GmshMesh, PhysicalName, from_numbered_gmsh_mesh,
};
use gridcl_parallel::Comm;

use crate::io_download::{self, GmshNumbering, agree_write};
use crate::{AnyMeshRef, Mesh, MeshTags};

/// Writes meshes as Gmsh MSH 4.1 files, ASCII by default (ADR-0022, item 10; ADR-0035, item
/// 11).
///
/// - Node tags are the vertex origin ids; the other nodes are numbered after the largest vertex
///   id, in global node order. Element tags are the cell origin ids; tag elements are numbered
///   after the largest cell id. Tags below 1 are shifted by one constant per kind so that the
///   smallest is 1. The numbering is global, so one rank gives the serial numbering and every
///   file of a distributed mesh agrees with the gathered file.
/// - Tagged entities of dimension < D_TOP are written as linear tag elements from
///   `EntityBlock::vertices` (canonical A.6 orientation); cells carry their physical and
///   elementary tags. A mesh without tags gets one entity with tag 1.
/// - `$Entities` is rebuilt from the elementary tags: one entity per (elementary tag, set of
///   physical tags).
/// - On more than one rank the owned cells, their nodes and the owned tag entities are
///   gathered to rank 0, which writes the file: cells by (rank, owned order), nodes in global
///   node order. [`partitioned`](GmshWriter::partitioned) writes one file per rank instead.
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
    partitioned: bool,
}

/// The (entity index → values) map of `tags`, downloaded.
pub(crate) fn values_by_index(tags: Option<&MeshTags>) -> Result<BTreeMap<u32, Vec<i32>>> {
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
pub(crate) fn elementary_of(elementary: &BTreeMap<u32, Vec<i32>>, i: u32) -> i32 {
    elementary.get(&i).map_or(1, |v| v[0])
}

/// The tags of every local cell in flat order: the elementary tag, and the number of physical
/// tags followed by all of them.
struct CellTags {
    elementary: Vec<i32>,
    counts: Vec<u32>,
    physical: Vec<i32>,
}

fn cell_tags<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<CellTags>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let tags = mesh.tags();
    let elementary = values_by_index(tags.elementary(TD::N))?;
    let physical = values_by_index(tags.physical(TD::N))?;
    let cells = mesh
        .cell_blocks()
        .iter()
        .try_fold(0usize, |n, b| n.checked_add(b.index_map.num_local()))
        .ok_or(GridError::TooLarge)?;
    let cells = u32::try_from(cells).map_err(|_| GridError::TooLarge)?;
    let mut out = CellTags {
        elementary: Vec::with_capacity(cells as usize),
        counts: Vec::with_capacity(cells as usize),
        physical: Vec::new(),
    };
    for c in 0..cells {
        out.elementary.push(elementary_of(&elementary, c));
        let values = physical.get(&c).map_or(&[][..], |v| &v[..]);
        out.counts.push(values.len() as u32);
        out.physical.extend_from_slice(values);
    }
    Ok(out)
}

/// The constant that makes `min` 1, or 0 if it is not below 1.
fn shift_to_one(min: Option<i64>) -> Result<i64> {
    match min {
        Some(m) if m < 1 => 1i64.checked_sub(m).ok_or(GridError::TooLarge),
        _ => Ok(0),
    }
}

/// `blocks` (cell type, degree, node indices, origin ids) with their tags: `elementary` one
/// per cell and `counts` physical values per cell from `physical`, all in flat cell order.
fn tagged_blocks(
    blocks: Vec<io_download::DownloadBlock>,
    elementary: &[i32],
    counts: &[u32],
    physical: &[i32],
) -> Vec<GmshCellBlock> {
    let (mut first, mut next) = (0usize, 0usize);
    blocks
        .into_iter()
        .map(|b| {
            let n = b.origin_ids.len();
            let mut physical_pairs = Vec::new();
            for (c, &k) in counts[first..first + n].iter().enumerate() {
                let k = k as usize;
                physical_pairs.extend(physical[next..next + k].iter().map(|&v| (c as u32, v)));
                next += k;
            }
            let out = GmshCellBlock {
                cell: b.cell,
                degree: b.degree,
                nodes: b.nodes,
                tags: b.origin_ids,
                elementary: elementary[first..first + n].to_vec(),
                physical: physical_pairs,
            };
            first += n;
            out
        })
        .collect()
}

/// The names of `mesh`'s physical groups.
fn names<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Vec<PhysicalName>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    mesh.tags()
        .names()
        .map(|(name, dim, tag)| PhysicalName {
            dim,
            tag,
            name: name.to_owned(),
        })
        .collect()
}

/// The host description of the whole mesh on rank 0, gathered (ADR-0035, item 11); on the
/// other ranks one without cells. Node tags and tag-element tags are `numbering`'s.
/// Collective, with an agreed status.
pub(crate) fn gathered<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    numbering: &GmshNumbering,
) -> Result<GmshMesh>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let comm = mesh.comm();
    let tags = crate::pipeline::agree(comm, cell_tags(mesh))?;
    let host = io_download::gather(mesh)?;
    let elementary = io_download::gather_cell_data(mesh, &tags.elementary, 1)?;
    let (counts, physical) = io_download::gather_cell_lists(mesh, &tags.counts, &tags.physical)?;
    Ok(GmshMesh {
        tdim: TD::N,
        gdim: GD::N,
        nodes: host.nodes,
        node_tags: io_download::gather_point_data(mesh, &numbering.node_tags, 1)?,
        blocks: tagged_blocks(host.blocks, &elementary, &counts, &physical),
        tag_groups: io_download::gather_tag_groups(comm, &numbering.tag_groups)?,
        names: names(mesh),
    })
}

/// The host description of this rank's part (ADR-0035, item 11): its owned cells, the nodes
/// they use in local order, and its owned tag entities.
fn part<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>, numbering: &GmshNumbering) -> Result<GmshMesh>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let tags = cell_tags(mesh)?;
    let mut host = io_download::piece(mesh)?;
    let num_nodes = numbering.node_tags.len();
    let mut used = vec![false; num_nodes];
    let (mut elementary, mut counts, mut physical) = (Vec::new(), Vec::new(), Vec::new());
    let (mut first, mut next) = (0usize, 0usize);
    for b in &mut host.blocks {
        let local = b.origin_ids.len();
        let k = gridcl_refcell::num_lagrange_nodes(b.cell, b.degree);
        b.nodes.truncate(b.num_owned * k);
        b.origin_ids.truncate(b.num_owned);
        for &n in &b.nodes {
            used[n as usize] = true;
        }
        let sum = |cells: usize| -> usize {
            tags.counts[first..first + cells]
                .iter()
                .map(|&k| k as usize)
                .sum()
        };
        let owned = sum(b.num_owned);
        elementary.extend_from_slice(&tags.elementary[first..first + b.num_owned]);
        counts.extend_from_slice(&tags.counts[first..first + b.num_owned]);
        physical.extend_from_slice(&tags.physical[next..next + owned]);
        next += sum(local);
        first += local;
    }
    let mut index = vec![u32::MAX; num_nodes];
    let mut nodes = Vec::new();
    let mut node_tags = Vec::new();
    for i in (0..num_nodes).filter(|&i| used[i]) {
        index[i] = u32::try_from(node_tags.len()).map_err(|_| GridError::TooLarge)?;
        nodes.extend_from_slice(&host.nodes[i * GD::N..(i + 1) * GD::N]);
        node_tags.push(numbering.node_tags[i]);
    }
    for b in &mut host.blocks {
        for n in &mut b.nodes {
            *n = index[*n as usize];
        }
    }
    Ok(GmshMesh {
        tdim: TD::N,
        gdim: GD::N,
        nodes,
        node_tags,
        blocks: tagged_blocks(host.blocks, &elementary, &counts, &physical),
        tag_groups: numbering.tag_groups.clone(),
        names: names(mesh),
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
            partitioned: false,
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

    /// On more than one rank, one file `<stem>_<r+1>.msh` per rank r instead of one file
    /// gathered on rank 0 (the default; ADR-0035, item 11). The stem is the path without a
    /// `.msh` extension. Each file holds the rank's owned cells, the nodes they use and its owned
    /// tag entities, `$Entities` and `$PartitionedEntities` (P partitions, one partitioned
    /// entity per entity, in partition r + 1, no ghost entities), and reads serially as that
    /// rank's part. On one rank the writer writes `path` as without this option.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::gmsh::GmshWriter;
    ///
    /// // On 4 ranks: out/mesh_1.msh, …, out/mesh_4.msh.
    /// let writer = GmshWriter::new("out/mesh.msh").partitioned(true);
    /// # let _ = writer;
    /// ```
    pub fn partitioned(self, partitioned: bool) -> Self {
        GmshWriter {
            partitioned,
            ..self
        }
    }

    /// The file of rank `rank` of `size` ranks.
    fn file_path(&self, rank: usize, size: usize) -> PathBuf {
        if !self.partitioned || size == 1 {
            return self.path.clone();
        }
        let stem = if self.path.extension().is_some_and(|e| e == "msh") {
            self.path.with_extension("")
        } else {
            self.path.clone()
        };
        let mut name = stem.into_os_string();
        name.push(format!("_{}.msh", rank + 1));
        PathBuf::from(name)
    }

    /// Writes `mesh`, copying it to the host. Collective: on more than one rank every rank
    /// calls it, and every rank returns an error if any rank fails (ADR-0035, item 11).
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated` for tags of a dimension without entities; `Io` on every rank if
    /// a file cannot be written (on another rank, naming it); `TooLarge` if a tag overflows;
    /// device errors, as `RemoteFailure` on the other ranks.
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
    ///
    /// A distributed mesh (feature `mpi`, run under `mpirun`), written gathered and as one file
    /// per rank:
    ///
    /// ```no_run
    /// # #[cfg(feature = "mpi")] {
    /// use gridcl::prelude::*;
    ///
    /// let universe = mpi::initialize().expect("MPI is not initialised yet");
    /// let comm = Mpi::world(&universe);
    /// let client = gridcl_testing::test_client();
    /// let chunk = gridcl_testing::cube_box(Cell3::Tetrahedron, 4, 1).chunk(comm.rank(), comm.size());
    /// let mesh: VolumeMesh<f32, Mpi> = chunk.builder(&client)?.build_on(comm.clone())?;
    /// gmsh::GmshWriter::new("box.msh").write(&mesh)?; // box.msh, written by rank 0
    /// gmsh::GmshWriter::new("box.msh").partitioned(true).write(&mesh)?; // box_1.msh, …
    /// # }
    /// # Ok::<(), gridcl::GridError>(())
    /// ```
    pub fn write<F, TD, GD, C>(&self, mesh: &Mesh<F, TD, GD, C>) -> Result<()>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let comm = mesh.comm();
        let (rank, size) = (comm.rank(), comm.size());
        let partitioned = self.partitioned && size > 1;
        let path = self.file_path(rank, size);
        let numbering = io_download::gmsh_numbering(mesh)?;
        let shifts = shift_to_one(numbering.node_min)
            .and_then(|n| Ok((n, shift_to_one(numbering.cell_min)?)));
        let (node_shift, element_shift) = crate::pipeline::agree(comm, shifts)?;
        let host = if partitioned {
            part(mesh, &numbering)
        } else {
            gathered(mesh, &numbering)
        };
        let local = host.and_then(|host| {
            if !partitioned && rank != 0 {
                return Ok(());
            }
            let numbering = FileNumbering {
                node_shift,
                element_shift,
                partition: partitioned.then(|| (size, rank as i32 + 1)),
            };
            let file = from_numbered_gmsh_mesh(&host, &numbering)?;
            self.write_file(&path, &file)
        });
        agree_write(comm, &self.path, local)
    }

    /// Writes `file` to `path`; I/O errors name the path.
    fn write_file(&self, path: &Path, file: &gridcl_io::gmsh::MshFile) -> Result<()> {
        let io = |e: std::io::Error| GridError::Io(format!("{}: {e}", path.display()));
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
        let encoding = if self.binary {
            Encoding::Binary
        } else {
            Encoding::Ascii
        };
        file.write_to(&mut out, encoding).map_err(|e| match e {
            GridError::Io(m) => GridError::Io(format!("{}: {m}", path.display())),
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
