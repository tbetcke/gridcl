//! Static names of cell types and node families in `GridError`s, and the mapping of
//! `RefCellError` onto `GridError` (ADR-0012).

use gridcl_device::GridError;
use gridcl_dims::CellType;
use gridcl_refcell::{NodeFamily, RefCellError};

/// The name of a cell type in errors: its `CellType` variant name.
pub(crate) fn cell_name(cell: CellType) -> &'static str {
    match cell {
        CellType::Point => "Point",
        CellType::Interval => "Interval",
        CellType::Triangle => "Triangle",
        CellType::Quadrilateral => "Quadrilateral",
        CellType::Tetrahedron => "Tetrahedron",
        CellType::Hexahedron => "Hexahedron",
        CellType::Prism => "Prism",
        CellType::Pyramid => "Pyramid",
    }
}

/// The name of a node family in errors.
fn family_name(family: NodeFamily) -> &'static str {
    match family {
        NodeFamily::Equispaced => "Equispaced",
        NodeFamily::Gll => "Gll",
    }
}

/// `GridError` of an element gridcl-refcell does not offer.
///
/// # Panics
///
/// On the other `RefCellError`s, which a valid element and a well-shaped point array never
/// produce.
pub(crate) fn element_error(e: RefCellError) -> GridError {
    match e {
        RefCellError::UnsupportedNodeFamily { cell, family } => GridError::UnsupportedNodeFamily {
            cell: cell_name(cell),
            family: family_name(family),
        },
        RefCellError::UnsupportedDegree { cell, degree } => GridError::UnsupportedDegree {
            cell: cell_name(cell),
            degree,
        },
        other => unreachable!("gridcl-refcell returned {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refcell_errors_map_to_static_names() {
        let e = element_error(RefCellError::UnsupportedNodeFamily {
            cell: CellType::Triangle,
            family: NodeFamily::Gll,
        });
        assert!(matches!(
            e,
            GridError::UnsupportedNodeFamily {
                cell: "Triangle",
                family: "Gll"
            }
        ));
        let e = element_error(RefCellError::UnsupportedDegree {
            cell: CellType::Prism,
            degree: 0,
        });
        assert!(matches!(
            e,
            GridError::UnsupportedDegree {
                cell: "Prism",
                degree: 0
            }
        ));
    }
}
