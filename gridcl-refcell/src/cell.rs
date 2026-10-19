//! Reference-cell tables A.1–A.3 (docs/design.md, Appendix A).

// T0 stubs: removed by the task that fills in the bodies (docs/plans/M1b.md).
#![allow(unused_variables)]

use gridcl_dims::CellType;

/// The topological dimension of `cell`: 0 for a point, up to 3 for volume cells.
pub fn tdim(cell: CellType) -> usize {
    todo!()
}

/// The vertex coordinates of the reference cell (A.1), in local vertex order.
///
/// Each vertex has [`tdim`] coordinates; the point cell has one vertex with none.
pub fn vertices(cell: CellType) -> &'static [&'static [f64]] {
    todo!()
}

/// The local vertex lists of the sub-entities of dimension `dim` (A.2, A.3), numbered by
/// DefElement Convention 2: sorted vertex lists in lexicographic order.
///
/// Dimension 0 gives the vertices as one-element lists, dimension [`tdim`] the cell itself.
/// Quadrilateral faces list their vertices in the face's tensor order. Empty if `dim > tdim`.
pub fn sub_entities(cell: CellType, dim: usize) -> &'static [&'static [usize]] {
    todo!()
}

/// The cell types of the sub-entities of dimension `dim`, in the order of [`sub_entities`].
pub fn sub_entity_types(cell: CellType, dim: usize) -> &'static [CellType] {
    todo!()
}

/// The number of sub-entities of dimension `dim`; 0 if `dim > tdim`.
pub fn num_sub_entities(cell: CellType, dim: usize) -> usize {
    todo!()
}
