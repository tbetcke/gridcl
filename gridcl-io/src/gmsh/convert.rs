//! The conversion between a file and the host mesh description.

use gridcl_device::Result;
use gridcl_dims::CellType;
use gridcl_parallel::CommDyn;

use super::{MshFile, PhysicalName};

/// The cells of one cell type and degree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GmshCellBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree of the complete Lagrange element (serendipity cells are upgraded to 2).
    pub degree: u32,
    /// Indices into [`GmshMesh::nodes`], `[cell][nodes per cell]` in A.5 order.
    pub nodes: Vec<u32>,
    /// The element tag of each cell (its origin id).
    pub tags: Vec<i64>,
    /// The elementary (entity) tag of each cell.
    pub elementary: Vec<i32>,
    /// (cell index in the block, physical tag), sorted.
    pub physical: Vec<(u32, i32)>,
}

/// The lower-dimensional elements of one cell type, which become tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GmshTagGroup {
    /// The dimension of the elements.
    pub dim: usize,
    /// The element's cell type.
    pub cell: CellType,
    /// The corner node tags of each element in internal vertex order, `[element][vertices]`.
    pub corners: Vec<i64>,
    /// The element tag of each element.
    pub tags: Vec<i64>,
    /// The elementary (entity) tag of each element.
    pub elementary: Vec<i32>,
    /// (element index in the group, physical tag), sorted.
    pub physical: Vec<(u32, i32)>,
}

/// The mesh a file describes: D_TOP and D_GEO, nodes with tags, cell blocks of the D_TOP
/// elements in `CellType` order, the lower-dimensional elements as tag groups, and the
/// physical names.
#[derive(Clone, Debug, PartialEq)]
pub struct GmshMesh {
    /// D_TOP: the highest element dimension.
    pub tdim: usize,
    /// D_GEO: inferred, or fixed by the caller.
    pub gdim: usize,
    /// The node coordinates, `[node][gdim]`.
    pub nodes: Vec<f64>,
    /// The node tags (origin ids).
    pub node_tags: Vec<i64>,
    /// The cell blocks, in `CellType` order.
    pub blocks: Vec<GmshCellBlock>,
    /// The tag groups.
    pub tag_groups: Vec<GmshTagGroup>,
    /// The physical names.
    pub names: Vec<PhysicalName>,
}

/// The mesh `file` describes. `gdim` fixes D_GEO; otherwise it is inferred over all nodes of
/// the file, reduced over `comm`.
///
/// # Errors
///
/// `Parse` for a file without elements of dimension ≥ 1 or with an element node that is not in
/// `$Nodes`; `UnsupportedElementType`; `DuplicateOriginId` for repeated node or element tags;
/// `DuplicateCellBlock` for one cell type in two degrees; `DimensionMismatch` for a `gdim`
/// outside 1–3, below D_TOP or below the inferred D_GEO; `UnmatchedTagElement` for a tag
/// element whose corner is not a vertex of a cell.
pub fn to_gmsh_mesh(file: &MshFile, comm: &dyn CommDyn, gdim: Option<usize>) -> Result<GmshMesh> {
    let _ = (file, comm, gdim);
    todo!("M4 T8")
}

/// The file of `mesh`: node and element tags as in ADR-0022 (shifted to start at 1 where
/// needed), linear tag elements, rebuilt `$Entities` and `$PhysicalNames`.
///
/// # Errors
///
/// `TooLarge` if a tag overflows; `UnsupportedElementType` for a cell Gmsh has no type for.
pub fn from_gmsh_mesh(mesh: &GmshMesh) -> Result<MshFile> {
    let _ = mesh;
    todo!("M4 T12")
}
