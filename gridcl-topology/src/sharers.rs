//! `VertexSharers`: the per-vertex sharer sets of pipeline stage 8 (ADR-0026, item 8).

/// A host CSR over local vertices: row v, `ranks[offsets[v]..offsets[v + 1]]`, lists every rank
/// that holds vertex v, this one included, in ascending order; `owned_cell` says, per entry,
/// whether that rank holds v in an owned cell.
///
/// Pipeline stage 8 fills it; a topology without sharers treats every vertex as unshared.
///
/// # Example
///
/// ```
/// use gridcl_topology::VertexSharers;
///
/// // Vertex 0 only on this rank (1); vertex 1 also on rank 3, which holds it in a ghost cell.
/// let s = VertexSharers {
///     offsets: vec![0, 1, 3],
///     ranks: vec![1, 1, 3],
///     owned_cell: vec![true, true, false],
/// };
/// let row = s.offsets[1]..s.offsets[2];
/// assert_eq!((&s.ranks[row.clone()], &s.owned_cell[row]), (&[1, 3][..], &[true, false][..]));
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VertexSharers {
    /// Row offsets; one more than the local vertex count.
    pub offsets: Vec<usize>,
    /// The ranks of every row, ascending.
    pub ranks: Vec<i32>,
    /// Whether the rank of each entry holds the vertex in an owned cell.
    pub owned_cell: Vec<bool>,
}
