//! Gmsh MSH 4.1 meshes: [`read`] and [`GmshWriter`] (docs/design.md, "Import and export /
//! Gmsh MSH 4.1"; ADR-0022). The file model and the conversion to a host mesh description are
//! in `gridcl_io::gmsh`.

use std::path::{Path, PathBuf};

use cubecl::prelude::Client;
use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::Comm;

use crate::{AnyMesh, AnyMeshRef, Mesh};

/// Options of [`read`]. `Default` infers D_GEO.
#[derive(Clone, Debug, Default)]
pub struct GmshOptions {
    gdim: Option<usize>,
}

impl GmshOptions {
    /// Fixes D_GEO: 1–3, at least D_TOP and at least the inferred D_GEO.
    pub fn gdim(self, gdim: usize) -> Self {
        GmshOptions { gdim: Some(gdim) }
    }
}

/// Reads the Gmsh MSH 4.1 file at `path` into a mesh whose dimensions the file decides.
///
/// D_TOP is the highest element dimension; its elements form the cell blocks. Physical groups
/// and elementary tags become the mesh's [`TagRegistry`](crate::TagRegistry). Node tags are
/// the vertex origin ids, element tags the cell origin ids. Reading into an f32 mesh logs one
/// warning per file.
///
/// # Errors
///
/// `NotYetImplemented` for more than one rank (until M8); `UnsupportedPrecision` if the device
/// lacks `F`; `Io`, `Parse`, `UnsupportedElementType`, `DuplicateOriginId`,
/// `DuplicateCellBlock`, `DimensionMismatch` and `UnmatchedTagElement` as in
/// `gridcl_io::gmsh`; allocation and device errors.
pub fn read<F: Real, C: Comm>(
    client: &Client,
    comm: C,
    path: impl AsRef<Path>,
    options: GmshOptions,
) -> Result<AnyMesh<F, C>> {
    let _ = (client, comm, path.as_ref(), options.gdim);
    todo!("M4 T11")
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
