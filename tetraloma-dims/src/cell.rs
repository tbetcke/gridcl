//! Cell families per topological dimension (docs/design.md, Listing B2).

use core::{fmt::Debug, hash::Hash};

use crate::dim::{D1, D2, D3, Dim};

/// Every reference cell type, of any topological dimension.
///
/// The cells of one dimension convert to and from their family [`Cell1`], [`Cell2`] or [`Cell3`].
/// The book defines each reference cell, with its vertices and sub-entity numbering, in
/// [Reference cell definitions].
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell3, CellType};
///
/// let cell_type = CellType::from(Cell3::Prism);
/// assert_eq!(cell_type, CellType::Prism);
/// assert_eq!(Cell3::try_from(cell_type), Ok(Cell3::Prism));
/// ```
///
/// [Reference cell definitions]: https://tetracl.github.io/tetraloma/elements/reference-cells.html
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CellType {
    /// A point (dimension 0); in no cell family, so every conversion to one fails.
    Point,
    /// An interval (dimension 1).
    Interval,
    /// A triangle (dimension 2).
    Triangle,
    /// A quadrilateral (dimension 2).
    Quadrilateral,
    /// A tetrahedron (dimension 3).
    Tetrahedron,
    /// A hexahedron (dimension 3).
    Hexahedron,
    /// A triangular prism (dimension 3).
    Prism,
    /// A square-based pyramid (dimension 3).
    Pyramid,
}

/// The cell types of topological dimension 1: the cell family of [`D1`] meshes.
///
/// See [Typed dimensions] for the cell family of each dimension.
///
/// [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell1 {
    /// An interval.
    Interval,
}

/// The cell types of topological dimension 2: the cell family of [`D2`] meshes.
///
/// See [Typed dimensions] for the cell family of each dimension.
///
/// [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell2 {
    /// A triangle.
    Triangle,
    /// A quadrilateral.
    Quadrilateral,
}

/// The cell types of topological dimension 3: the cell family of [`D3`] meshes.
///
/// See [Typed dimensions] for the cell family of each dimension.
///
/// [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell3 {
    /// A tetrahedron.
    Tetrahedron,
    /// A hexahedron.
    Hexahedron,
    /// A triangular prism.
    Prism,
    /// A square-based pyramid.
    Pyramid,
}

/// The error of a conversion from [`CellType`] to a cell family of another topological
/// dimension.
///
/// See [Typed dimensions].
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, CellType, WrongCellDim};
///
/// assert_eq!(
///     Cell2::try_from(CellType::Tetrahedron),
///     Err(WrongCellDim { found: CellType::Tetrahedron, expected_tdim: 2 }),
/// );
/// ```
///
/// [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrongCellDim {
    /// The cell type that was given.
    pub found: CellType,
    /// The topological dimension the target cell family requires.
    pub expected_tdim: usize,
}

/// A topological dimension of a mesh: [`D1`], [`D2`] or [`D3`], with its cell family.
///
/// [`D0`](crate::D0) is not a topological dimension, so a curve mesh has no boundary mesh. See
/// [Typed dimensions].
///
/// [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
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

/// `From<$cell> for CellType` and `TryFrom<CellType> for $cell` for the cell family of the
/// topological dimension `$td`. The `From` match is exhaustive, so every variant of `$cell` must
/// be listed.
macro_rules! cell_family {
    ($cell:ident, $td:ty: $($variant:ident),+) => {
        impl From<$cell> for CellType {
            fn from(cell: $cell) -> Self {
                match cell {
                    $($cell::$variant => CellType::$variant,)+
                }
            }
        }

        impl TryFrom<CellType> for $cell {
            type Error = WrongCellDim;

            fn try_from(cell: CellType) -> Result<Self, WrongCellDim> {
                match cell {
                    $(CellType::$variant => Ok($cell::$variant),)+
                    found => Err(WrongCellDim { found, expected_tdim: <$td as Dim>::N }),
                }
            }
        }
    };
}

cell_family!(Cell1, D1: Interval);
cell_family!(Cell2, D2: Triangle, Quadrilateral);
cell_family!(Cell3, D3: Tetrahedron, Hexahedron, Prism, Pyramid);
