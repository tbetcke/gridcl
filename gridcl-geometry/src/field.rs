//! `QuadField`, `QuadChunk` and `BlockFields` (Listing B4b; ADR-0012).

use core::marker::PhantomData;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, GridError, Real, Result, VECTOR_PAD, device_properties};

use crate::{EVector, FieldShape};

/// Cells per chunk: the most whole multiples of `align` cells of `per_cell` elements of `F`
/// whose padded allocation fits the device's `max_page_size`.
pub(crate) fn chunk_cells<F: Real>(
    client: &Client,
    per_cell: usize,
    align: usize,
) -> Result<usize> {
    let max = device_properties(client).memory.max_page_size;
    // Elements in one allocation, rounded down to whole VECTOR_PAD blocks, so the padding of
    // `DeviceBuffer` never pushes a chunk over the limit.
    let elems = usize::try_from(max / size_of::<F>() as u64).unwrap_or(usize::MAX) / VECTOR_PAD
        * VECTOR_PAD;
    if per_cell == 0 {
        return Ok(usize::MAX / align * align);
    }
    let cells = elems / per_cell / align * align;
    if cells == 0 {
        let bytes = align
            .checked_mul(per_cell)
            .and_then(|e| e.checked_mul(size_of::<F>()))
            .ok_or(GridError::TooLarge)?;
        return Err(GridError::AllocationTooLarge { bytes, max });
    }
    Ok(cells)
}

/// Chunks of `num_cells` cells of `per_cell` elements, `per_chunk` cells each, every chunk
/// padded to whole multiples of `align` cells.
pub(crate) fn alloc_chunks<F: Real>(
    client: &Client,
    num_cells: usize,
    per_cell: usize,
    align: usize,
    per_chunk: usize,
) -> Result<Vec<QuadChunk<F>>> {
    let mut chunks = Vec::new();
    let mut first = 0;
    while first < num_cells {
        let n = per_chunk.min(num_cells - first);
        let padded = n
            .checked_next_multiple_of(align)
            .ok_or(GridError::TooLarge)?;
        let len = padded.checked_mul(per_cell).ok_or(GridError::TooLarge)?;
        chunks.push(QuadChunk {
            first_cell: first,
            num_cells: n,
            data: DeviceBuffer::uninit(client, len)?,
        });
        first += n;
    }
    Ok(chunks)
}

/// A tile-aligned cell range that lies in one E-vector chunk and one field chunk (Listing B5).
#[allow(dead_code)] // the factor kernels of T13 use it
pub(crate) struct ChunkPair<'a, F: Real> {
    /// The E-vector chunk.
    pub x: &'a DeviceBuffer<F>,
    /// The range's first cell within the E-vector chunk.
    pub x_first_cell: usize,
    /// The E-vector chunk's row stride in cells (its padded cell count).
    pub x_stride_cells: usize,
    /// The field chunk.
    pub out: &'a DeviceBuffer<F>,
    /// The range's first cell within the field chunk; a multiple of the tile.
    pub out_first_cell: usize,
    /// The global index of the range's first cell.
    pub first_cell: usize,
    /// The range's cells rounded up to whole tiles.
    pub padded_cells: usize,
}

/// The tile-aligned cell ranges that each lie in one E-vector chunk and one field chunk, in
/// cell order. E-vector chunks start on multiples of `VECTOR_PAD` and field chunks on multiples
/// of the tile, which divides `VECTOR_PAD`, so every range starts on a tile boundary.
#[allow(dead_code)] // the factor kernels of T13 use it
pub(crate) fn chunk_pairs<'a, F: Real, S: FieldShape>(
    xe: &'a EVector<F>,
    out: &'a QuadField<F, S>,
) -> Vec<ChunkPair<'a, F>> {
    let tile = out.tile();
    let mut pairs = Vec::new();
    let (mut i, mut j) = (0, 0);
    let (xs, os) = (xe.chunks(), out.chunks());
    while i < xs.len() && j < os.len() {
        let (x, o) = (&xs[i], &os[j]);
        let start = x.first_cell.max(o.first_cell);
        let x_end = x.first_cell + x.num_cells;
        let o_end = o.first_cell + o.num_cells;
        let end = x_end.min(o_end);
        if start < end {
            pairs.push(ChunkPair {
                x: &x.data,
                x_first_cell: start - x.first_cell,
                x_stride_cells: x.num_cells.next_multiple_of(VECTOR_PAD),
                out: &o.data,
                out_first_cell: start - o.first_cell,
                first_cell: start,
                padded_cells: (end - start).next_multiple_of(tile),
            });
        }
        if x_end <= o_end {
            i += 1;
        }
        if o_end <= x_end {
            j += 1;
        }
    }
    pairs
}

/// Values at quadrature points of one cell block, in tiles of `tile` cells (1, 2, 4, 8 or 16;
/// `VectorPolicy::quad_tile`, 16 by default). Component k at point q of cell c is stored at
///
/// ```text
/// ((c / tile) * num_points * LEN + q * LEN + k) * tile + c % tile
/// ```
///
/// counted from the first cell of its chunk, so the lanes of a kernel over consecutive cells
/// read and write contiguous memory. `tile` = 1 is the plain `[cell][point][component]` layout.
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
        assert!(
            tile.is_power_of_two() && tile <= VECTOR_PAD,
            "QuadField::uninit: tile {tile} is not 1, 2, 4, 8 or 16"
        );
        let per_cell = num_points.checked_mul(S::LEN).ok_or(GridError::TooLarge)?;
        let per_chunk = chunk_cells::<F>(client, per_cell, tile)?;
        Ok(QuadField {
            chunks: alloc_chunks(client, num_cells, per_cell, tile, per_chunk)?,
            num_cells,
            num_points,
            tile,
            _s: PhantomData,
        })
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

    /// Host copy in plain `[cell][point]` order, whatever the tile width.
    ///
    /// # Errors
    ///
    /// Device read errors.
    pub fn to_host_cell_major(&self) -> Result<Vec<S::Host<F>>> {
        let len = S::LEN;
        assert_eq!(
            (size_of::<S::Host<F>>(), align_of::<S::Host<F>>()),
            (len * size_of::<F>(), align_of::<F>()),
            "FieldShape::Host<T> must be LEN values of T"
        );
        let (q_len, tile) = (self.num_points, self.tile);
        let mut out = Vec::with_capacity(self.num_cells * q_len);
        for chunk in &self.chunks {
            let data = chunk.data.to_host()?;
            for c in 0..chunk.num_cells {
                for q in 0..q_len {
                    let mut value = S::Host::<F>::default();
                    // SAFETY: Host<F> is LEN values of F in an array (FieldShape's contract;
                    // size and alignment checked above), so it is LEN initialised, aligned
                    // values of F, any of which may be overwritten.
                    let parts = unsafe {
                        core::slice::from_raw_parts_mut((&raw mut value).cast::<F>(), len)
                    };
                    for (k, part) in parts.iter_mut().enumerate() {
                        *part = data[((c / tile) * q_len * len + q * len + k) * tile + c % tile];
                    }
                    out.push(value);
                }
            }
        }
        Ok(out)
    }
}

/// One `QuadField` per cell block, indexed like `cell_blocks()`.
pub struct BlockFields<F: Real, S: FieldShape> {
    /// The fields, one per cell block.
    pub blocks: Vec<QuadField<F, S>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScalarShape;
    use gridcl_device::testing::with_properties;

    /// The pairs cover every cell once, start on tile boundaries and stay inside both chunks.
    #[test]
    fn chunk_pairs_tile_both_chunkings() {
        let client = gridcl_testing::test_client();
        let num_cells = 203;
        let nodes = DeviceBuffer::from_host(&client, &[0.0f32; 2]).unwrap();
        let dofmap = DeviceBuffer::from_host(&client, &vec![0u32; num_cells]).unwrap();
        for (x_limit, f_limit) in [
            (None, None),
            (Some(64 * 4), Some(36 * 4)),
            (Some(32 * 4), Some(100 * 4)),
        ] {
            for tile in [1, 2, 4, 8, 16] {
                let mut props = client.properties().clone();
                if let Some(l) = x_limit {
                    props.memory.max_page_size = l;
                }
                let xe = with_properties(props, || {
                    crate::evector::build(&nodes, &dofmap, num_cells, 1, 1, nodes.share())
                })
                .unwrap();
                let mut props = client.properties().clone();
                if let Some(l) = f_limit {
                    props.memory.max_page_size = l;
                }
                let out = with_properties(props, || {
                    QuadField::<f32, ScalarShape>::uninit(&client, num_cells, 1, tile)
                })
                .unwrap();
                let pairs = chunk_pairs(&xe, &out);
                let mut next = 0;
                for p in &pairs {
                    assert_eq!(p.first_cell, next, "contiguous");
                    assert_eq!(p.out_first_cell % tile, 0);
                    assert_eq!(p.first_cell % tile, 0);
                    assert_eq!(p.x_stride_cells % VECTOR_PAD, 0);
                    assert!(p.x_first_cell + p.padded_cells <= p.x_stride_cells);
                    assert!(p.out_first_cell + p.padded_cells <= p.out.len());
                    next = (p.first_cell + p.padded_cells).min(num_cells);
                }
                assert_eq!(next, num_cells, "tile {tile}: cells covered");
            }
        }
    }
}
