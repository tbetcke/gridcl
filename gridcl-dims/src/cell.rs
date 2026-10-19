//! Cell families per topological dimension (docs/design.md, Listing B2).

use core::{fmt::Debug, hash::Hash};

use crate::dim::{D1, D2, D3, Dim};

/// Every reference cell type, of any topological dimension.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CellType {
    Point,
    Interval,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// The cell types of topological dimension 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell1 {
    Interval,
}

/// The cell types of topological dimension 2.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell2 {
    Triangle,
    Quadrilateral,
}

/// The cell types of topological dimension 3.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell3 {
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
}

/// A cell type used where a different topological dimension is required.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrongCellDim {
    pub found: CellType,
    pub expected_tdim: usize,
}

/// A topological dimension of a mesh: [`D1`], [`D2`] or [`D3`], with its cell family.
pub trait TopoDim: Dim {
    /// The admissible cell types of this dimension.
    type Cell: Copy
        + Eq
        + Hash
        + Debug
        + Send
        + Sync
        + 'static
        + Into<CellType>
        + TryFrom<CellType, Error = WrongCellDim>;
}
impl TopoDim for D1 {
    type Cell = Cell1;
}
impl TopoDim for D2 {
    type Cell = Cell2;
}
impl TopoDim for D3 {
    type Cell = Cell3;
}

impl From<Cell1> for CellType {
    fn from(_cell: Cell1) -> Self {
        todo!()
    }
}
impl TryFrom<CellType> for Cell1 {
    type Error = WrongCellDim;
    fn try_from(_cell: CellType) -> Result<Self, WrongCellDim> {
        todo!()
    }
}
impl From<Cell2> for CellType {
    fn from(_cell: Cell2) -> Self {
        todo!()
    }
}
impl TryFrom<CellType> for Cell2 {
    type Error = WrongCellDim;
    fn try_from(_cell: CellType) -> Result<Self, WrongCellDim> {
        todo!()
    }
}
impl From<Cell3> for CellType {
    fn from(_cell: Cell3) -> Self {
        todo!()
    }
}
impl TryFrom<CellType> for Cell3 {
    type Error = WrongCellDim;
    fn try_from(_cell: CellType) -> Result<Self, WrongCellDim> {
        todo!()
    }
}
