//! The Gmsh element table (A.7).

use gridcl_device::Result;
use gridcl_dims::CellType;

/// A Gmsh element type of A.7.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElementType {
    /// The Gmsh type id.
    pub id: i32,
    /// The cell type.
    pub cell: CellType,
    /// The degree (2 for the serendipity types).
    pub degree: u32,
    /// The number of nodes per element.
    pub num_nodes: usize,
    /// Whether this is a serendipity type (16–19), upgraded to complete Lagrange on read.
    pub serendipity: bool,
}

/// The element type of Gmsh type id `id`.
///
/// # Errors
///
/// `UnsupportedElementType` for every id not in A.7.
pub fn element_type(id: i32) -> Result<ElementType> {
    let _ = id;
    todo!("M4 T2")
}

/// The complete Lagrange element type of `cell` and `degree`.
///
/// # Errors
///
/// `UnsupportedDegree` if Gmsh has no complete Lagrange type of this cell and degree.
pub fn lagrange_element_type(cell: CellType, degree: u32) -> Result<ElementType> {
    let _ = (cell, degree);
    todo!("M4 T2")
}
