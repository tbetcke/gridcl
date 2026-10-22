//! `EVector` and the EVector kernel (ADR-0012), generic over the vector size.

use gridcl_device::Real;

use crate::QuadChunk;

/// The cached E-vector of one geometry block (ADR-0012): chunks of whole tiles,
/// [k][g][cell] with cells fastest, the cell count padded to `VECTOR_PAD` with copies of the
/// block's last cell. An empty block has no chunks. Built on first use by
/// `DeviceGrid::evector`.
pub struct EVector<F: Real> {
    chunks: Vec<QuadChunk<F>>,
    num_cells: usize,
}

impl<F: Real> EVector<F> {
    /// The number of cells, padding excluded.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// The chunks, in cell order. Row (k, g) of a chunk starts at (k · gdim + g) · stride, with
    /// stride the chunk's padded cell count.
    pub fn chunks(&self) -> &[QuadChunk<F>] {
        &self.chunks
    }
}
