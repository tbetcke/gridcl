//! Gmsh's serendipity elements (types 16–19) and their upgrade to complete P2 (ADR-0022).

use gridcl_dims::CellType;

use crate::RefCellError;

/// The weights that give the missing nodes of a complete-P2 element from the nodes of Gmsh's
/// serendipity element of the same cell: 8 → 9 (quadrilateral), 20 → 27 (hexahedron), 15 → 18
/// (prism), 13 → 14 (pyramid).
///
/// Both node lists are in Gmsh order. The serendipity nodes are the first
/// [`num_file_nodes`](Self::num_file_nodes) nodes of Gmsh's complete-P2 element; missing node
/// m (Gmsh P2 index `num_file_nodes + m`) is Σⱼ `weights[m * num_file_nodes + j]` · node j.
#[derive(Clone, Debug, PartialEq)]
pub struct SerendipityUpgrade {
    /// The node count of the serendipity element: 8, 20, 15 or 13.
    pub num_file_nodes: usize,
    /// The node count of the complete-P2 element: 9, 27, 18 or 14.
    pub num_nodes: usize,
    /// `[missing node][file node]`, row-major.
    pub weights: Vec<f64>,
}

/// The upgrade of Gmsh's serendipity element on `cell` (host, f64): its serendipity map
/// evaluated at the missing complete-P2 nodes.
///
/// # Errors
///
/// `NotInFormat` for cells without a Gmsh serendipity type (all but quadrilaterals,
/// hexahedra, prisms and pyramids).
pub fn serendipity_upgrade(cell: CellType) -> Result<SerendipityUpgrade, RefCellError> {
    let _ = cell;
    todo!("M4 T3")
}
