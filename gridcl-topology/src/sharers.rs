//! `VertexSharers`: the per-vertex sharer sets of pipeline stage 8 (ADR-0026, item 8).

/// A host CSR over local vertices: row v, `ranks[offsets[v]..offsets[v + 1]]`, lists every rank
/// that holds vertex v, this one included, in ascending order; `owned_cell` says, per entry,
/// whether that rank holds v in an owned cell.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexSharers {
    /// Row offsets; one more than the local vertex count.
    pub offsets: Vec<usize>,
    /// The ranks of every row, ascending.
    pub ranks: Vec<i32>,
    /// Whether the rank of each entry holds the vertex in an owned cell.
    pub owned_cell: Vec<bool>,
}
