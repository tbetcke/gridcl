//! M1b T9 conformance (not an acceptance test): gridcl's VTK node maps equal ndelement's
//! `vtk_ordering` for the six cells ndelement supports, p ≤ 4.
//!
//! Same direction: ndelement's VTK-ordered values are `[dof_values[perm[0]], ...]`, so
//! `perm[k]` is the internal index of VTK node k, as in `vtk_to_internal` (A.9).

use gridcl_conformance::{CELLS, reference_cell_type};
use gridcl_dims::CellType;
use gridcl_refcell::{num_lagrange_nodes, vtk_to_internal};
use ndelement::reorderings::vtk_ordering;

#[test]
fn vtk_matches_ndelement() {
    let supported = CELLS
        .into_iter()
        .filter(|c| !matches!(c, CellType::Prism | CellType::Pyramid));
    for cell in supported {
        for degree in 1..=4 {
            let npts = num_lagrange_nodes(cell, degree);
            assert_eq!(
                vtk_to_internal(cell, degree).unwrap(),
                vtk_ordering(reference_cell_type(cell), npts),
                "{cell:?}, p = {degree}"
            );
        }
    }
}
