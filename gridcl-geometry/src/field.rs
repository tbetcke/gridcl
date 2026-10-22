//! `QuadField`, `QuadChunk` and `BlockFields` (Listing B4b; ADR-0012).

use core::marker::PhantomData;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, Real, Result};

use crate::FieldShape;

/// Values at quadrature points of one cell block, in tiles of `tile` cells (1, 2, 4, 8 or 16;
/// `VectorPolicy::quad_tile`, 16 by default). Component k at point q of cell c is stored at
///
/// ```text
/// ((c / tile) * num_points * LEN + q * LEN + k) * tile + c % tile
/// ```
///
/// counted from the first cell of its chunk, so the lanes of a kernel over consecutive cells
/// read and write contiguous memory. `tile` = 1 is the plain [cell][point][component] layout.
/// Stored in chunks of whole tiles so that no chunk exceeds the device's largest allocation
/// (`device_properties(client).memory.max_page_size`); the cell count is padded to whole tiles.
pub struct QuadField<F: Real, S: FieldShape> {
    chunks: Vec<QuadChunk<F>>,
    num_cells: usize,
    num_points: usize,
    tile: usize,
    _s: PhantomData<S>,
}

/// One chunk of a [`QuadField`] or [`EVector`](crate::EVector): whole tiles of cells.
pub struct QuadChunk<F: Real> {
    /// The first cell of the chunk.
    pub first_cell: usize,
    /// The cells of the chunk, padding excluded.
    pub num_cells: usize,
    /// The values.
    pub data: DeviceBuffer<F>,
}

impl<F: Real, S: FieldShape> QuadField<F, S> {
    /// An uninitialised field of `num_cells` cells with `num_points` points each, in tiles of
    /// `tile` cells, chunked by the device's `max_page_size`.
    ///
    /// # Errors
    ///
    /// `AllocationTooLarge` if one tile does not fit one allocation; `TooLarge` on overflow.
    ///
    /// # Panics
    ///
    /// If `tile` is not 1, 2, 4, 8 or 16.
    pub fn uninit(
        client: &Client,
        num_cells: usize,
        num_points: usize,
        tile: usize,
    ) -> Result<Self> {
        let _ = (client, num_cells, num_points, tile);
        let _: (Vec<QuadChunk<F>>, PhantomData<S>) = (Vec::new(), PhantomData);
        todo!()
    }

    /// Cells per tile.
    pub fn tile(&self) -> usize {
        self.tile
    }

    /// The number of cells, padding excluded.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// Points per cell.
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// The chunks, in cell order.
    pub fn chunks(&self) -> &[QuadChunk<F>] {
        &self.chunks
    }

    /// Host copy in plain [cell][point] order, whatever the tile width.
    ///
    /// # Errors
    ///
    /// Device read errors.
    pub fn to_host_cell_major(&self) -> Result<Vec<S::Host<F>>> {
        todo!()
    }
}

/// One `QuadField` per cell block, indexed like `cell_blocks()`.
pub struct BlockFields<F: Real, S: FieldShape> {
    /// The fields, one per cell block.
    pub blocks: Vec<QuadField<F, S>>,
}
