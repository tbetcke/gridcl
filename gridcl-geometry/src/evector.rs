//! `EVector` and the EVector kernel (ADR-0012), generic over the vector size.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Real, Result, VECTOR_PAD,
    debug_assert_bindings,
};

use crate::QuadChunk;
use crate::field::{alloc_chunks, chunk_cells};

/// The cached E-vector of one geometry block (ADR-0012): chunks of whole tiles,
/// [k][g][cell] with cells fastest, the cell count padded to `VECTOR_PAD` with copies of the
/// block's last cell. An empty block has no chunks. Built on first use by
/// `DeviceGrid::evector`.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Triangle, 3, 1).build::<f32, D2, D2>(&client)?;
/// let xe = mesh.evector(0)?;
/// assert_eq!(xe.num_cells(), 18);
/// // One chunk of 3 nodes × 2 coordinates × 32 cells (18 padded to VECTOR_PAD).
/// assert_eq!(xe.chunks().len(), 1);
/// assert_eq!(xe.chunks()[0].data.len(), 3 * 2 * 32);
/// # Ok::<(), GridError>(())
/// ```
pub struct EVector<F: Real> {
    chunks: Vec<QuadChunk<F>>,
    num_cells: usize,
    /// The grid's origin narrowed to `F`, [gdim]; the physical-points kernel adds it back.
    origin: DeviceBuffer<F>,
}

impl<F: Real> EVector<F> {
    /// The number of cells, padding excluded.
    pub fn num_cells(&self) -> usize {
        self.num_cells
    }

    /// The chunks, in cell order. Row (k, g) of a chunk starts at (k · gdim + g) · stride, with
    /// stride the chunk's cell count rounded up to `VECTOR_PAD`.
    pub fn chunks(&self) -> &[QuadChunk<F>] {
        &self.chunks
    }

    /// The grid's origin narrowed to `F`.
    pub(crate) fn origin(&self) -> &DeviceBuffer<F> {
        &self.origin
    }
}

/// One E-vector chunk: for every cell vector `v < num_vectors` (Listing B5 loop) of the chunk,
/// row (k, g) gets the coordinate g of node k of each lane's cell, gathered lane by lane from
/// the L-vector; only the stores are vectors. Lanes past the block's last cell copy the last
/// cell.
///   nodes   L-vector [node][gdim]
///   dofmap  [cell][k] node indices of the block
///   out     [k][g][cell] with row stride `stride_vectors` vectors
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn evector_kernel<F: Float, N: Size>(
    nodes: &[F],
    dofmap: &[u32],
    out: &mut [Vector<F, N>],
    first_cell: u32,
    num_cells: u32,
    stride_vectors: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] nodes_per_cell: usize,
    #[comptime] gdim: usize,
) {
    let lanes = N::value();
    let last = num_cells - 1;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            #[unroll]
            for k in 0..nodes_per_cell {
                let mut node = Vector::<u32, N>::broadcast(0u32);
                #[unroll]
                for lane in 0..lanes {
                    let c = min(first_cell + (v * lanes + lane) as u32, last) as usize;
                    node.insert(lane, dofmap[c * nodes_per_cell + k]);
                }
                #[unroll]
                for g in 0..gdim {
                    let mut x = Vector::<F, N>::broadcast(F::from_int(0));
                    #[unroll]
                    for lane in 0..lanes {
                        x.insert(lane, nodes[node.extract(lane) as usize * gdim + g]);
                    }
                    out[(k * gdim + g) * stride_vectors as usize + v] = x;
                }
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The E-vector of `num_cells` cells with `nodes_per_cell` nodes each from the L-vector `nodes`
/// ([node][gdim]) and the block's `dofmap`, chunked by the device's `max_page_size`.
///
/// # Errors
///
/// `AllocationTooLarge` if `VECTOR_PAD` cells do not fit one allocation; device errors.
pub(crate) fn build<F: Real>(
    nodes: &DeviceBuffer<F>,
    dofmap: &DeviceBuffer<u32>,
    num_cells: usize,
    nodes_per_cell: usize,
    gdim: usize,
    origin: DeviceBuffer<F>,
) -> Result<EVector<F>> {
    let client = nodes.client();
    let per_cell = nodes_per_cell
        .checked_mul(gdim)
        .ok_or(GridError::TooLarge)?;
    let per_chunk = chunk_cells::<F>(client, per_cell, VECTOR_PAD)?;
    let chunks = alloc_chunks::<F>(client, num_cells, per_cell, VECTOR_PAD, per_chunk)?;
    for chunk in &chunks {
        let padded = chunk.num_cells.next_multiple_of(VECTOR_PAD);
        let plan = LaunchPlan::elementwise(
            client,
            KernelKind::EVector,
            padded,
            size_of::<F>(),
            VECTOR_PAD,
            2,
        )?;
        let n = plan.vector_size;
        debug_assert_bindings(client, 3);
        // SAFETY: the chunk holds padded · per_cell elements, padded a multiple of VECTOR_PAD
        // and so of n; the kernel writes vectors (k · gdim + g) · padded / n + v with
        // v < num_vectors = padded / n, k < nodes_per_cell and g < gdim, all below
        // per_cell · padded / n. It reads `dofmap` at c · nodes_per_cell + k with
        // c ≤ num_cells − 1, within the block's dofmap, and `nodes` at node · gdim + g for the
        // dofmap's node indices, within the L-vector.
        unsafe {
            evector_kernel::launch_unchecked::<F>(
                client,
                plan.cube_count,
                plan.cube_dim,
                n,
                nodes.arg(),
                dofmap.arg(),
                chunk.data.vector_arg(n),
                fit(chunk.first_cell)?,
                fit(num_cells)?,
                fit(padded / n)?,
                plan.num_vectors,
                plan.items_per_unit,
                nodes_per_cell,
                gdim,
            );
        }
    }
    Ok(EVector {
        chunks,
        num_cells,
        origin,
    })
}
