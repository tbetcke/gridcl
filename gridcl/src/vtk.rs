//! VTU and PVTU output: [`VtkWriter`] (docs/design.md, "Import and export / VTK", Listing B13;
//! ADR-0024). The file model, the cell-type table, the writer and the parser are in
//! `gridcl_io::vtk`.

use std::io::Write;
use std::path::{Path, PathBuf};

use gridcl_device::{DeviceBuffer, GridError, Real, Result};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{DeviceGrid, NodeFamily, equispaced_nodes};
use gridcl_io::vtk::{Binary, VtkCellBlock, VtkMesh, VtuFile, to_vtu};
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
    let gd = GD::N;
    let blocks = mesh.cell_blocks();
    let gll = (0..blocks.len()).any(|b| mesh.geometry_block(b).element.family == NodeFamily::Gll);
    let nodes = if gll {
        equispaced_nodes(mesh)?.to_host()?
    } else {
        mesh.nodes().to_host()?
    };
    let origin = mesh.origin();
    let nodes = nodes
        .iter()
        .enumerate()
        .map(|(i, x)| x.widen() + origin.as_ref()[i % gd])
        .collect();
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
        gdim: gd,
        nodes,
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

    /// A writer of `<stem>.pvtu` and its pieces `<stem>_<rank>.vtu` beside it.
    pub fn pvtu(path: impl AsRef<Path>) -> Self {
        let _ = path;
        todo!()
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
    pub fn point_data<T: VtkScalar>(
        self,
        name: &str,
        data: &'a DeviceBuffer<T>,
        components: usize,
    ) -> Self {
        let _ = (name, T::data_ref(data), components);
        todo!()
    }

    /// Attaches cell data `name`: `components` values per cell, in flat cell order over the
    /// blocks. A second array of the same name replaces the first.
    pub fn cell_data<T: VtkScalar>(
        self,
        name: &str,
        data: &'a DeviceBuffer<T>,
        components: usize,
    ) -> Self {
        let _ = (name, T::data_ref(data), components);
        todo!()
    }

    /// Writes `mesh`, copying it to the host.
    ///
    /// Pyramids of degree p ≥ 2 are written from their five vertices (VTK type 14), with one
    /// `log::warn!` per file, unless `.lagrange(false)` asks for linear cells anyway.
    ///
    /// # Errors
    ///
    /// `NotYetImplemented` for more than one rank (until M8); `LengthMismatch` for a data
    /// array of the wrong length; `Io`; allocation and device errors.
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
            return Err(GridError::NotYetImplemented("parallel VTU write (M8)"));
        }
        // Point and cell data and the PVTU come with T7.
        let _ = (self.pvtu, &self.point_data, &self.cell_data);
        let host = download(mesh)?;
        let f32_points = size_of::<F>() < size_of::<f64>();
        let (file, pyramids) = to_vtu(&host, self.lagrange, f32_points)?;
        self.write_vtu(&self.path, &file)?;
        if pyramids > 0 {
            warn_pyramids(&self.path, &host, pyramids);
        }
        Ok(())
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
    }
}
