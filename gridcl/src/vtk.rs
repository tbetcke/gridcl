//! VTU and PVTU output: [`VtkWriter`] (docs/design.md, "Import and export / VTK", Listing B13;
//! ADR-0024). The file model, the cell-type table, the writer and the parser are in
//! `gridcl_io::vtk`.

use std::io::Write;
use std::path::{Path, PathBuf};

use gridcl_device::{DeviceBuffer, GridError, Real, Result};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::DeviceGrid;
use gridcl_io::vtk::{
    Binary, DataArray, DataValues, VtkCellBlock, VtkMesh, VtuFile, to_vtu, write_pvtu,
};
use gridcl_parallel::Comm;

use crate::{AnyMeshRef, Mesh};

mod sealed {
    use gridcl_device::DeviceBuffer;

    /// A borrowed data buffer of one of the [`VtkScalar`](super::VtkScalar) types.
    #[derive(Clone, Copy)]
    pub enum DataRef<'a> {
        F32(&'a DeviceBuffer<f32>),
        F64(&'a DeviceBuffer<f64>),
        I32(&'a DeviceBuffer<i32>),
        U32(&'a DeviceBuffer<u32>),
    }

    pub trait Sealed: cubecl::prelude::CubeElement {
        /// `data`, erased.
        fn data_ref(data: &DeviceBuffer<Self>) -> DataRef<'_>;
    }

    impl Sealed for f32 {
        fn data_ref(data: &DeviceBuffer<f32>) -> DataRef<'_> {
            DataRef::F32(data)
        }
    }

    impl Sealed for f64 {
        fn data_ref(data: &DeviceBuffer<f64>) -> DataRef<'_> {
            DataRef::F64(data)
        }
    }

    impl Sealed for i32 {
        fn data_ref(data: &DeviceBuffer<i32>) -> DataRef<'_> {
            DataRef::I32(data)
        }
    }

    impl Sealed for u32 {
        fn data_ref(data: &DeviceBuffer<u32>) -> DataRef<'_> {
            DataRef::U32(data)
        }
    }
}

/// The scalars of point and cell data: f32, f64, i32 and u32 (sealed; ADR-0024, item 3).
pub trait VtkScalar: sealed::Sealed {}

impl VtkScalar for f32 {}
impl VtkScalar for f64 {}
impl VtkScalar for i32 {}
impl VtkScalar for u32 {}

/// One attached data array: name, buffer and component count.
type Attached<'a> = (String, sealed::DataRef<'a>, usize);

/// Writes meshes as VTU files, or as a PVTU header with one VTU piece per rank (ADR-0024).
///
/// - Cells of degree p ≥ 2 are VTK Lagrange cells (68–73) unless `.lagrange(false)`; pyramids
///   of degree p ≥ 2 are written from their vertices (type 14), with one warning per file.
/// - Points are all geometry nodes, converted to equispaced nodes and with the origin added:
///   Float64 for f64 meshes, Float32 for f32 meshes.
/// - Data is appended raw unless `.base64(true)`, and zlib-compressed with `.zlib(true)`.
/// - Point data is indexed by geometry node, cell data by flat cell index in block order.
pub struct VtkWriter<'a> {
    path: PathBuf,
    pvtu: bool,
    lagrange: bool,
    base64: bool,
    zlib: bool,
    point_data: Vec<Attached<'a>>,
    cell_data: Vec<Attached<'a>>,
}

/// The host description of `mesh` (one rank): every geometry node, converted to equispaced
/// nodes on the device if a block is GLL and with the origin added in f64, and the local cells
/// of every block in A.5 node order.
fn download<F, TD, GD, C>(mesh: &Mesh<F, TD, GD, C>) -> Result<VtkMesh>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let blocks = mesh.cell_blocks();
    let mut out = Vec::with_capacity(blocks.len());
    for (b, block) in blocks.iter().enumerate() {
        let geometry = mesh.geometry_block(b);
        let mut dofmap = geometry.dofmap.data.to_host()?;
        let len = block
            .index_map
            .num_local()
            .checked_mul(geometry.dofmap.stride as usize)
            .ok_or(GridError::TooLarge)?;
        dofmap.truncate(len);
        out.push(VtkCellBlock {
            cell: block.cell.into(),
            degree: geometry.element.degree,
            nodes: dofmap,
        });
    }
    Ok(VtkMesh {
        gdim: GD::N,
        nodes: mesh.equispaced_host_nodes()?,
        blocks: out,
    })
}

/// Creates `path` and writes it through `f`; I/O errors name the path.
fn write_file(path: &Path, f: impl FnOnce(&mut dyn Write) -> Result<()>) -> Result<()> {
    let io = |e: std::io::Error| GridError::Io(format!("{}: {e}", path.display()));
    let mut out = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
    f(&mut out).map_err(|e| match e {
        GridError::Io(m) => GridError::Io(format!("{}: {m}", path.display())),
        other => other,
    })?;
    out.flush().map_err(io)
}

/// The number of values of `data`.
fn data_len(data: sealed::DataRef<'_>) -> usize {
    match data {
        sealed::DataRef::F32(b) => b.len(),
        sealed::DataRef::F64(b) => b.len(),
        sealed::DataRef::I32(b) => b.len(),
        sealed::DataRef::U32(b) => b.len(),
    }
}

/// `LengthMismatch` unless every array of `arrays` has `entries · components` values and at
/// least one component.
fn check_lengths(arrays: &[Attached<'_>], entries: usize) -> Result<()> {
    for (_, data, components) in arrays {
        if *components == 0 {
            return Err(GridError::LengthMismatch {
                expected: 1,
                found: 0,
            });
        }
        let expected = entries
            .checked_mul(*components)
            .ok_or(GridError::TooLarge)?;
        let found = data_len(*data);
        if found != expected {
            return Err(GridError::LengthMismatch { expected, found });
        }
    }
    Ok(())
}

/// The arrays of `arrays`, downloaded.
fn download_data(arrays: &[Attached<'_>]) -> Result<Vec<DataArray>> {
    arrays
        .iter()
        .map(|(name, data, components)| {
            let values = match data {
                sealed::DataRef::F32(b) => DataValues::F32(b.to_host()?),
                sealed::DataRef::F64(b) => DataValues::F64(b.to_host()?),
                sealed::DataRef::I32(b) => DataValues::I32(b.to_host()?),
                sealed::DataRef::U32(b) => DataValues::U32(b.to_host()?),
            };
            Ok(DataArray {
                name: name.clone(),
                components: *components,
                values,
            })
        })
        .collect()
}

/// `arrays` with `(name, data, components)` in place of an array of the same name, or added
/// at the end.
fn attach<'a>(
    mut arrays: Vec<Attached<'a>>,
    name: &str,
    data: sealed::DataRef<'a>,
    components: usize,
) -> Vec<Attached<'a>> {
    let entry = (name.to_owned(), data, components);
    match arrays.iter_mut().find(|a| a.0 == name) {
        Some(a) => *a = entry,
        None => arrays.push(entry),
    }
    arrays
}

/// The warning of `pyramids` pyramids written from their vertices, naming their degrees.
fn warn_pyramids(path: &Path, mesh: &VtkMesh, pyramids: usize) {
    let degrees: Vec<String> = mesh
        .blocks
        .iter()
        .filter(|b| b.cell == CellType::Pyramid && b.degree >= 2)
        .map(|b| b.degree.to_string())
        .collect();
    log::warn!(
        "{}: {pyramids} pyramids of degree {} written as linear cells (VTK type 14) from \
         their vertices; VTK has no Lagrange pyramid",
        path.display(),
        degrees.join(", ")
    );
}

impl<'a> VtkWriter<'a> {
    /// A writer of the VTU file at `path`, with `.vtu` appended if `path` has no extension.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// // Writes out/scalp.vtu.
    /// let writer = VtkWriter::vtu("out/scalp");
    /// # let _ = writer;
    /// ```
    pub fn vtu(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let path = if path.extension().is_none() {
            let mut p = path.as_os_str().to_owned();
            p.push(".vtu");
            PathBuf::from(p)
        } else {
            path.to_path_buf()
        };
        VtkWriter {
            path,
            pvtu: false,
            lagrange: true,
            base64: false,
            zlib: false,
            point_data: Vec::new(),
            cell_data: Vec::new(),
        }
    }

    /// A writer of `<stem>.pvtu` and its pieces `<stem>_<rank>.vtu` beside it. The stem is
    /// `path` without a `.pvtu` extension.
    ///
    /// The PVTU declares the points and the data arrays of the pieces; ghost arrays come with
    /// M8, and so does more than one rank.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// let rank = DeviceBuffer::from_host(&client, &[0i32; 4])?;
    /// let dir = std::env::temp_dir().join(format!("gridcl-doc-pvtu-{}", std::process::id()));
    /// std::fs::create_dir_all(&dir).unwrap();
    /// VtkWriter::pvtu(dir.join("square")).cell_data("rank", &rank, 1).write(&mesh)?;
    /// let header = std::fs::read_to_string(dir.join("square.pvtu")).unwrap();
    /// assert!(header.contains("<Piece Source=\"square_0.vtu\"/>"));
    /// assert!(header.contains("Name=\"rank\""));
    /// let piece = gridcl_io::vtk::parse_vtu(&std::fs::read(dir.join("square_0.vtu")).unwrap())?;
    /// std::fs::remove_dir_all(&dir).ok();
    /// assert_eq!(piece.types, [9; 4]);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn pvtu(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        let stem = if path.extension().is_some_and(|e| e == "pvtu") {
            path.with_extension("")
        } else {
            path.to_path_buf()
        };
        VtkWriter {
            path: stem,
            pvtu: true,
            ..VtkWriter::vtu("")
        }
    }

    /// Lagrange cells for p ≥ 2 (the default), or linear cells from the corner nodes.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Triangle, 1, 2).build(&client)?;
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-lagrange-{}.vtu", std::process::id()));
    /// VtkWriter::vtu(&path).lagrange(false).write(&mesh)?;
    /// let file = gridcl_io::vtk::parse_vtu(&std::fs::read(&path).unwrap())?;
    /// std::fs::remove_file(&path).ok();
    /// // Two linear triangles (VTK type 5); the edge nodes stay as unreferenced points.
    /// assert_eq!(file.types, [5, 5]);
    /// assert_eq!(file.connectivity.len(), 6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn lagrange(self, lagrange: bool) -> Self {
        VtkWriter { lagrange, ..self }
    }

    /// Inline base64 data instead of appended raw data (the default).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let writer = VtkWriter::vtu("out/scalp").base64(true);
    /// # let _ = writer;
    /// ```
    pub fn base64(self, base64: bool) -> Self {
        VtkWriter { base64, ..self }
    }

    /// zlib compression of every data array (default off).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let writer = VtkWriter::vtu("out/scalp").base64(true).zlib(true);
    /// # let _ = writer;
    /// ```
    pub fn zlib(self, zlib: bool) -> Self {
        VtkWriter { zlib, ..self }
    }

    /// Attaches point data `name`: `components` values per geometry node. A second array of
    /// the same name replaces the first.
    ///
    /// Geometry nodes are the points of the file, also for linear output, which leaves the
    /// non-vertex nodes unreferenced.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build(&client)?;
    /// // One vector per node: the node's coordinates, relative to the origin.
    /// let xy = mesh.nodes().to_host()?;
    /// let vectors: Vec<f32> = xy.chunks_exact(2).flat_map(|x| [x[0], x[1], 0.0]).collect();
    /// let vectors = DeviceBuffer::from_host(&client, &vectors)?;
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-point-data-{}.vtu", std::process::id()));
    /// VtkWriter::vtu(&path).point_data("position", &vectors, 3).write(&mesh)?;
    /// let file = gridcl_io::vtk::parse_vtu(&std::fs::read(&path).unwrap())?;
    /// std::fs::remove_file(&path).ok();
    /// assert_eq!((file.point_data[0].name.as_str(), file.point_data[0].components), ("position", 3));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn point_data<T: VtkScalar>(
        self,
        name: &str,
        data: &'a DeviceBuffer<T>,
        components: usize,
    ) -> Self {
        VtkWriter {
            point_data: attach(self.point_data, name, T::data_ref(data), components),
            ..self
        }
    }

    /// Attaches cell data `name`: `components` values per cell, in flat cell order over the
    /// blocks. A second array of the same name replaces the first.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// let first = DeviceBuffer::from_host(&client, &[0u32; 4])?;
    /// let index = DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3])?;
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-cell-data-{}.vtu", std::process::id()));
    /// // The second "index" replaces the first.
    /// VtkWriter::vtu(&path).cell_data("index", &first, 1).cell_data("index", &index, 1).write(&mesh)?;
    /// let file = gridcl_io::vtk::parse_vtu(&std::fs::read(&path).unwrap())?;
    /// std::fs::remove_file(&path).ok();
    /// assert_eq!(file.cell_data.len(), 1);
    /// assert_eq!(file.cell_data[0].values, gridcl_io::vtk::DataValues::U32(vec![0, 1, 2, 3]));
    ///
    /// // A wrong length is refused.
    /// let short = DeviceBuffer::from_host(&client, &[0u32; 3])?;
    /// let r = VtkWriter::vtu(&path).cell_data("index", &short, 1).write(&mesh);
    /// assert!(matches!(r, Err(GridError::LengthMismatch { expected: 4, found: 3 })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn cell_data<T: VtkScalar>(
        self,
        name: &str,
        data: &'a DeviceBuffer<T>,
        components: usize,
    ) -> Self {
        VtkWriter {
            cell_data: attach(self.cell_data, name, T::data_ref(data), components),
            ..self
        }
    }

    /// Writes `mesh`, copying it to the host: the VTU file, or the PVTU header and its one
    /// piece.
    ///
    /// Pyramids of degree p ≥ 2 are written from their five vertices (VTK type 14), with one
    /// `log::warn!` per file, unless `.lagrange(false)` asks for linear cells anyway.
    ///
    /// # Errors
    ///
    /// `NotYetImplemented` for more than one rank (until M8); `LengthMismatch` for a data
    /// array of the wrong length or without components, before anything is written; `Io`
    /// (also for a PVTU stem without a UTF-8 file name); `TooLarge`; allocation and device
    /// errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let scalp: SurfaceMesh<f32> = gridcl_testing::sphere(0, 2).build(&client)?;
    /// let dir = std::env::temp_dir().join(format!("gridcl-doc-vtk-{}", std::process::id()));
    /// std::fs::create_dir_all(&dir).unwrap();
    /// VtkWriter::vtu(dir.join("scalp")).lagrange(true).write(&scalp)?; // generic over TD, GD
    /// VtkWriter::vtu(dir.join("scalp2")).write_any(AnyMeshRef::Surface(&scalp))?; // erased path, no clone
    /// let file = gridcl_io::vtk::parse_vtu(&std::fs::read(dir.join("scalp.vtu")).unwrap())?;
    /// std::fs::remove_dir_all(&dir).ok();
    /// // P2 triangles are VTK Lagrange triangles (69).
    /// assert!(file.types.iter().all(|&t| t == 69));
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
            return Err(GridError::NotYetImplemented(if self.pvtu {
                "parallel PVTU write (M8)"
            } else {
                "parallel VTU write (M8)"
            }));
        }
        let num_points = mesh.nodes().len() / GD::N;
        let num_cells = mesh
            .cell_blocks()
            .iter()
            .try_fold(0usize, |n, b| n.checked_add(b.index_map.num_local()))
            .ok_or(GridError::TooLarge)?;
        check_lengths(&self.point_data, num_points)?;
        check_lengths(&self.cell_data, num_cells)?;
        let piece = self.piece(mesh.comm().rank())?;

        let host = download(mesh)?;
        let f32_points = size_of::<F>() < size_of::<f64>();
        let (mut file, pyramids) = to_vtu(&host, self.lagrange, f32_points)?;
        file.point_data = download_data(&self.point_data)?;
        file.cell_data = download_data(&self.cell_data)?;
        let path = match &piece {
            Some((path, _)) => path,
            None => &self.path,
        };
        self.write_vtu(path, &file)?;
        if pyramids > 0 {
            warn_pyramids(path, &host, pyramids);
        }
        if let Some((_, source)) = piece {
            let mut header = self.path.as_os_str().to_owned();
            header.push(".pvtu");
            write_file(Path::new(&header), |w| write_pvtu(w, &[source], &file))?;
        }
        Ok(())
    }

    /// For a PVTU, the path of the piece of `rank` and its name relative to the header.
    fn piece(&self, rank: usize) -> Result<Option<(PathBuf, String)>> {
        if !self.pvtu {
            return Ok(None);
        }
        let stem = self
            .path
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                GridError::Io(format!(
                    "{}: a PVTU needs a UTF-8 file name",
                    self.path.display()
                ))
            })?;
        let source = format!("{stem}_{rank}.vtu");
        Ok(Some((self.path.with_file_name(&source), source)))
    }

    /// Writes `file` to `path` in the chosen encoding.
    fn write_vtu(&self, path: &Path, file: &VtuFile) -> Result<()> {
        let binary = if self.base64 {
            Binary::Base64
        } else {
            Binary::Appended
        };
        write_file(path, |w| file.write_to(w, binary, self.zlib))
    }

    /// Writes the mesh behind `mesh`.
    ///
    /// # Errors
    ///
    /// As [`write`](VtkWriter::write).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let any = AnyMesh::Curve1(gridcl_testing::interval_box(4, 3).build::<f32, D1, D1>(&client)?);
    /// let path = std::env::temp_dir().join(format!("gridcl-doc-vtk-any-{}.vtu", std::process::id()));
    /// VtkWriter::vtu(&path).base64(true).zlib(true).write_any(any.as_ref())?;
    /// let file = gridcl_io::vtk::parse_vtu(&std::fs::read(&path).unwrap())?;
    /// std::fs::remove_file(&path).ok();
    /// assert_eq!(file.types, [68; 4]);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn write_any<F: Real, C: Comm>(&self, mesh: AnyMeshRef<'_, F, C>) -> Result<()> {
        crate::with_mesh!(mesh, m => self.write(m))
    }
}

#[cfg(test)]
mod tests {
    use gridcl_io::vtk::{DataValues, parse_vtu};

    use gridcl_geometry::{NodeFamily, equispaced_nodes};

    use super::*;
    use crate::test_comm::TwoRanks;
    use crate::{Cell2, CoordinateElement, D2, MeshBuilder, PlanarMesh};

    /// `.vtu` is appended only to a path without an extension.
    #[test]
    fn vtu_appends_the_extension_if_missing() {
        for (given, path) in [
            ("out/scalp", "out/scalp.vtu"),
            ("out/scalp.vtu", "out/scalp.vtu"),
            ("out/scalp.v2", "out/scalp.v2"),
        ] {
            assert_eq!(VtkWriter::vtu(given).path, Path::new(path), "{given}");
        }
    }

    /// A 2 × 2 square of P3 quadrilaterals whose equispaced box nodes are taken as GLL nodes,
    /// so the equispaced interpolant moves the non-vertex nodes.
    fn gll_mesh<F: Real>() -> Result<PlanarMesh<F>> {
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3);
        let b = &h.blocks[0];
        let element = CoordinateElement::<D2> {
            cell: Cell2::Quadrilateral,
            degree: b.degree,
            family: NodeFamily::Gll,
        };
        MeshBuilder::new(&client)
            .nodes(&h.nodes, None)
            .cell_block(element, &b.cells, None)
            .build()
    }

    /// A GLL mesh writes its equispaced interpolant: every point is `equispaced_nodes` with the
    /// origin added in f64, in the mesh's precision.
    fn writes_equispaced_interpolant<F: Real>(name: &str) {
        let mesh = gll_mesh::<F>().unwrap();
        let path =
            std::env::temp_dir().join(format!("gridcl-m5-gll-{name}-{}.vtu", std::process::id()));
        VtkWriter::vtu(&path).write(&mesh).unwrap();
        let file = parse_vtu(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let origin = mesh.origin();
        let want: Vec<f64> = equispaced_nodes(&mesh)
            .unwrap()
            .to_host()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|x| {
                let o = origin.as_ref();
                [
                    F::narrow(x[0].widen() + o[0]).widen(),
                    F::narrow(x[1].widen() + o[1]).widen(),
                    0.0,
                ]
            })
            .collect();
        let got: Vec<f64> = match file.points {
            DataValues::F32(p) if size_of::<F>() == 4 => p.iter().map(|&x| x.into()).collect(),
            DataValues::F64(p) if size_of::<F>() == 8 => p,
            other => panic!("{name}: points {other:?}"),
        };
        assert_eq!(got.len(), want.len(), "{name}");
        assert!(
            got.iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: points are the equispaced interpolant"
        );
        // The interpolant differs from the GLL nodes themselves.
        let raw: Vec<f64> = mesh
            .nodes()
            .to_host()
            .unwrap()
            .iter()
            .map(|x| x.widen())
            .collect();
        let moved = equispaced_nodes(&mesh)
            .unwrap()
            .to_host()
            .unwrap()
            .iter()
            .zip(&raw)
            .any(|(a, b)| (a.widen() - b).abs() > 1e-3);
        assert!(moved, "{name}: the conversion moves nodes");
    }

    #[test]
    fn vtk_write_converts_gll() {
        writes_equispaced_interpolant::<f32>("f32");
        if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
            writes_equispaced_interpolant::<f64>("f64");
        }
    }

    /// A wrong length, or no components, is refused before anything is written; a repeated
    /// name replaces the earlier array in its place.
    #[test]
    fn vtk_data_lengths_and_repeated_names() {
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::square_box(Cell2::Triangle, 1, 2);
        let element = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 2,
            family: NodeFamily::Equispaced,
        };
        let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
            .nodes(&h.nodes, None)
            .cell_block(element, &h.blocks[0].cells, None)
            .build()
            .unwrap();
        // 9 nodes, 2 cells.
        let nine = DeviceBuffer::from_host(&client, &[1.0f32; 9]).unwrap();
        let eighteen = DeviceBuffer::from_host(&client, &[2i32; 18]).unwrap();
        let two = DeviceBuffer::from_host(&client, &[3u32; 2]).unwrap();
        let path =
            std::env::temp_dir().join(format!("gridcl-m5-lengths-{}.vtu", std::process::id()));
        for (w, expected, found) in [
            (VtkWriter::vtu(&path).point_data("a", &nine, 2), 18, 9),
            (VtkWriter::vtu(&path).point_data("a", &nine, 0), 1, 0),
            (VtkWriter::vtu(&path).cell_data("a", &nine, 1), 2, 9),
            (VtkWriter::vtu(&path).cell_data("a", &two, 3), 6, 2),
        ] {
            let r = w.write(&mesh);
            assert!(
                matches!(r, Err(GridError::LengthMismatch { expected: e, found: f }) if (e, f) == (expected, found)),
                "{r:?}"
            );
            assert!(!path.exists(), "nothing is written");
        }

        VtkWriter::vtu(&path)
            .point_data("a", &nine, 1)
            .point_data("b", &two, 1)
            .point_data("c", &nine, 1)
            .point_data("b", &eighteen, 2)
            .cell_data("b", &two, 1)
            .write(&mesh)
            .unwrap();
        let file = parse_vtu(&std::fs::read(&path).unwrap()).unwrap();
        let _ = std::fs::remove_file(&path);
        let names: Vec<(&str, usize)> = file
            .point_data
            .iter()
            .map(|a| (a.name.as_str(), a.components))
            .collect();
        assert_eq!(names, [("a", 1), ("b", 2), ("c", 1)]);
        assert_eq!(file.point_data[1].values, DataValues::I32(vec![2; 18]));
        assert_eq!(file.cell_data[0].values, DataValues::U32(vec![3; 2]));
    }

    /// The PVTU stem drops a `.pvtu` extension only; the piece sits beside the header.
    #[test]
    fn pvtu_stem_and_piece() {
        for (given, stem) in [
            ("out/mesh", "out/mesh"),
            ("out/mesh.pvtu", "out/mesh"),
            ("out/mesh.v2", "out/mesh.v2"),
        ] {
            let w = VtkWriter::pvtu(given);
            assert_eq!(w.path, Path::new(stem), "{given}");
            let (piece, source) = w.piece(3).unwrap().unwrap();
            let name = format!(
                "{}_3.vtu",
                Path::new(stem).file_name().unwrap().to_str().unwrap()
            );
            assert_eq!(piece, Path::new(stem).with_file_name(&name), "{given}");
            assert_eq!(source, name, "{given}");
        }
        assert!(VtkWriter::vtu("out/mesh").piece(0).unwrap().is_none());
    }

    /// More than one rank is refused before anything is written (until M8).
    #[test]
    fn vtk_write_needs_one_rank() {
        let mesh = gll_mesh::<f32>().unwrap().with_comm(TwoRanks);
        let path =
            std::env::temp_dir().join(format!("gridcl-m5-two-ranks-{}.vtu", std::process::id()));
        let r = VtkWriter::vtu(&path).write(&mesh);
        assert!(
            matches!(
                r,
                Err(GridError::NotYetImplemented("parallel VTU write (M8)"))
            ),
            "{r:?}"
        );
        assert!(!path.exists(), "nothing is written");

        let stem = std::env::temp_dir().join(format!("gridcl-m5-two-ranks-{}", std::process::id()));
        let r = VtkWriter::pvtu(&stem).write(&mesh);
        assert!(
            matches!(
                r,
                Err(GridError::NotYetImplemented("parallel PVTU write (M8)"))
            ),
            "{r:?}"
        );
        assert!(!stem.with_extension("pvtu").exists(), "nothing is written");
    }
}
