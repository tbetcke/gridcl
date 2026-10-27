//! VTU and PVTU output: [`VtkWriter`] (docs/design.md, "Import and export / VTK", Listing B13;
//! ADR-0024). The file model, the cell-type table, the writer and the parser are in
//! `gridcl_io::vtk`.

use std::path::{Path, PathBuf};

use gridcl_device::{DeviceBuffer, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
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

impl<'a> VtkWriter<'a> {
    /// A writer of the VTU file at `path`, with `.vtu` appended if `path` has no extension.
    pub fn vtu(path: impl AsRef<Path>) -> Self {
        let _ = path;
        todo!()
    }

    /// A writer of `<stem>.pvtu` and its pieces `<stem>_<rank>.vtu` beside it.
    pub fn pvtu(path: impl AsRef<Path>) -> Self {
        let _ = path;
        todo!()
    }

    /// Lagrange cells for p ≥ 2 (the default), or linear cells from the corner nodes.
    pub fn lagrange(self, lagrange: bool) -> Self {
        let _ = lagrange;
        todo!()
    }

    /// Inline base64 data instead of appended raw data (the default).
    pub fn base64(self, base64: bool) -> Self {
        let _ = base64;
        todo!()
    }

    /// zlib compression of every data array (default off).
    pub fn zlib(self, zlib: bool) -> Self {
        let _ = zlib;
        todo!()
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
    /// # Errors
    ///
    /// `NotYetImplemented` for more than one rank (until M8); `LengthMismatch` for a data
    /// array of the wrong length; `Io`; allocation and device errors.
    pub fn write<F, TD, GD, C>(&self, mesh: &Mesh<F, TD, GD, C>) -> Result<()>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let _ = (
            mesh,
            &self.path,
            self.pvtu,
            self.lagrange,
            self.base64,
            self.zlib,
            &self.point_data,
            &self.cell_data,
        );
        todo!()
    }

    /// Writes the mesh behind `mesh`.
    ///
    /// # Errors
    ///
    /// As [`write`](VtkWriter::write).
    pub fn write_any<F: Real, C: Comm>(&self, mesh: AnyMeshRef<'_, F, C>) -> Result<()> {
        crate::with_mesh!(mesh, m => self.write(m))
    }
}
