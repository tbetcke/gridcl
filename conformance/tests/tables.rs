//! Sub-entity tables against ndelement.
//!
//! Until gridcl-refcell has its tables (M1b), gridcl's side is Appendix A.2 of docs/design.md,
//! written out here (docs/plans/M0.md, question 3).

use ndelement::reference_cell;
use ndelement::types::ReferenceCellType;

/// Triangle edges of A.2: edge i is opposite vertex 2 − i.
const TRIANGLE_EDGES: [[usize; 2]; 3] = [[0, 1], [0, 2], [1, 2]];

#[test]
fn triangle_edges_match_ndelement() {
    let edges = reference_cell::edges(ReferenceCellType::Triangle);
    let edges: Vec<Vec<usize>> = edges.into_iter().map(|e| e.to_vec()).collect();
    let want: Vec<Vec<usize>> = TRIANGLE_EDGES.iter().map(|e| e.to_vec()).collect();
    assert_eq!(edges, want, "ndelement's triangle edges differ from A.2");
}
