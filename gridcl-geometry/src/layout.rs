//! `GeomLayout` (Listing B5).

use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};

use crate::CoordinateElement;

/// The comptime layout of a factor kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeomLayout {
    /// The topological dimension.
    pub tdim: usize,
    /// The geometric dimension.
    pub gdim: usize,
    /// Nodes of the coordinate element.
    pub nodes_per_cell: usize,
    /// Points per cell.
    pub num_points: usize,
    /// Cells per `QuadField` tile: 1, 2, 4, 8 or 16.
    pub tile: usize,
}

impl GeomLayout {
    /// The layout of element `e` at `points` points per cell in tiles of `tile` cells.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, D2, D3};
    /// use gridcl_geometry::{CoordinateElement, GeomLayout, NodeFamily};
    ///
    /// let e = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 2, family: NodeFamily::Equispaced };
    /// let l = GeomLayout::new::<D2, D3>(&e, 7, 16);
    /// assert_eq!((l.tdim, l.gdim, l.nodes_per_cell, l.num_points, l.tile), (2, 3, 6, 7, 16));
    /// ```
    pub fn new<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
        e: &CoordinateElement<TD>,
        points: usize,
        tile: usize,
    ) -> Self {
        let cell: CellType = e.cell.into();
        GeomLayout {
            tdim: TD::N,
            gdim: GD::N,
            nodes_per_cell: gridcl_refcell::num_lagrange_nodes(cell, e.degree),
            num_points: points,
            tile,
        }
    }
}
