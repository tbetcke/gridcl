//! `CellType` <-> `CellN` conversions (docs/design.md, Listing B2; M1a acceptance).

use gridcl_dims::{Cell1, Cell2, Cell3, CellType, D1, D2, D3, TopoDim, WrongCellDim};

const CELL1: [(Cell1, CellType); 1] = [(Cell1::Interval, CellType::Interval)];
const CELL2: [(Cell2, CellType); 2] = [
    (Cell2::Triangle, CellType::Triangle),
    (Cell2::Quadrilateral, CellType::Quadrilateral),
];
const CELL3: [(Cell3, CellType); 4] = [
    (Cell3::Tetrahedron, CellType::Tetrahedron),
    (Cell3::Hexahedron, CellType::Hexahedron),
    (Cell3::Prism, CellType::Prism),
    (Cell3::Pyramid, CellType::Pyramid),
];
const CELL_TYPES: [CellType; 8] = [
    CellType::Point,
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

// Exhaustive matches: a new variant breaks the build here, and the `lists_are_complete` test
// fails until the variant is also added to its list above.

fn cell1_index(cell: Cell1) -> usize {
    match cell {
        Cell1::Interval => 0,
    }
}

fn cell2_index(cell: Cell2) -> usize {
    match cell {
        Cell2::Triangle => 0,
        Cell2::Quadrilateral => 1,
    }
}

fn cell3_index(cell: Cell3) -> usize {
    match cell {
        Cell3::Tetrahedron => 0,
        Cell3::Hexahedron => 1,
        Cell3::Prism => 2,
        Cell3::Pyramid => 3,
    }
}

/// Index in `CELL_TYPES` and topological dimension.
fn cell_type_index_and_tdim(cell: CellType) -> (usize, usize) {
    match cell {
        CellType::Point => (0, 0),
        CellType::Interval => (1, 1),
        CellType::Triangle => (2, 2),
        CellType::Quadrilateral => (3, 2),
        CellType::Tetrahedron => (4, 3),
        CellType::Hexahedron => (5, 3),
        CellType::Prism => (6, 3),
        CellType::Pyramid => (7, 3),
    }
}

fn tdim(cell: CellType) -> usize {
    cell_type_index_and_tdim(cell).1
}

#[test]
fn lists_are_complete() {
    for (i, (cell, _)) in CELL1.into_iter().enumerate() {
        assert_eq!(cell1_index(cell), i);
    }
    for (i, (cell, _)) in CELL2.into_iter().enumerate() {
        assert_eq!(cell2_index(cell), i);
    }
    for (i, (cell, _)) in CELL3.into_iter().enumerate() {
        assert_eq!(cell3_index(cell), i);
    }
    for (i, cell) in CELL_TYPES.into_iter().enumerate() {
        assert_eq!(cell_type_index_and_tdim(cell).0, i);
    }
}

fn check_cell_n_round_trips<TD: TopoDim>(cells: &[(TD::Cell, CellType)]) {
    for &(cell, cell_type) in cells {
        assert_eq!(cell.into(), cell_type, "{cell:?} -> CellType");
        assert_eq!(
            tdim(cell_type),
            TD::N,
            "{cell_type:?} has dimension {}",
            TD::N
        );
        assert_eq!(
            TD::Cell::try_from(cell_type),
            Ok(cell),
            "{cell_type:?} -> {cell:?}"
        );
    }
}

#[test]
fn cell_n_round_trips() {
    check_cell_n_round_trips::<D1>(&CELL1);
    check_cell_n_round_trips::<D2>(&CELL2);
    check_cell_n_round_trips::<D3>(&CELL3);
}

fn check_cell_type_round_trip<TD: TopoDim>(cell_type: CellType) {
    let cell = TD::Cell::try_from(cell_type)
        .unwrap_or_else(|e| panic!("{cell_type:?} -> cell of dimension {}: {e:?}", TD::N));
    assert_eq!(
        cell.into(),
        cell_type,
        "{cell_type:?} -> {cell:?} -> CellType"
    );
}

#[test]
fn cell_type_round_trips() {
    for cell_type in CELL_TYPES {
        match tdim(cell_type) {
            0 => {}
            1 => check_cell_type_round_trip::<D1>(cell_type),
            2 => check_cell_type_round_trip::<D2>(cell_type),
            3 => check_cell_type_round_trip::<D3>(cell_type),
            d => unreachable!("dimension {d}"),
        }
    }
}

fn check_wrong_dimension<TD: TopoDim>() {
    for found in CELL_TYPES.into_iter().filter(|&c| tdim(c) != TD::N) {
        assert_eq!(
            TD::Cell::try_from(found),
            Err(WrongCellDim {
                found,
                expected_tdim: TD::N
            }),
            "{found:?} as a cell of dimension {}",
            TD::N
        );
    }
}

#[test]
fn wrong_dimension_returns_wrong_cell_dim() {
    // Point (dimension 0) is among the rejected types of all three.
    check_wrong_dimension::<D1>();
    check_wrong_dimension::<D2>();
    check_wrong_dimension::<D3>();
}
