//! Inclusive and exclusive scan, as reduce-then-scan with a vector-generic tile phase
//! (KernelKind::ScanTile).
//!
//! One partition, from the ScanTile plan; a tile is a cube. Phase 1 writes the sum of each tile
//! (the reduce kernel under the same plan), phase 2 scans the tile sums recursively with a fresh
//! plan, and phase 3 scans within each tile: the in-vector prefix (lane by lane, comptime
//! unrolled), a sequential pass over each unit's block, a shared-memory scan of the unit totals,
//! and the tile offset added last. The phases are separate launches, so no cube waits for
//! another, and every sum is taken in a fixed order.

use cubecl::prelude::*;

use super::reduce::launch_reduce;
use crate::plan::debug_assert_bindings;
use crate::{
    DeviceBuffer, DeviceScalar, GridError, KernelKind, LaunchPlan, ReduceOp, Result, VECTOR_PAD,
    fill,
};

/// The vector of `x` at `v` with lanes at index >= `len` set to 0. A vector that starts at or
/// past `len` is not read.
#[cube]
fn load_masked<T: Numeric, N: Size>(x: &[Vector<T, N>], v: usize, len: usize) -> Vector<T, N> {
    let lanes = N::value();
    let start = v * lanes;
    let mut value = Vector::<T, N>::broadcast(T::from_int(0));
    if start + lanes <= len {
        value = x[v];
    } else if start < len {
        let full = x[v];
        #[unroll]
        for l in 0..lanes {
            if start + l < len {
                value.insert(l, full.extract(l));
            }
        }
    }
    value
}

/// The lanes of `value` summed in lane order, as [`lane_prefix`] sums them.
#[cube]
fn lane_sum<T: Numeric, N: Size>(value: Vector<T, N>) -> T {
    let mut acc = T::from_int(0);
    #[unroll]
    for l in 0..N::value() {
        acc += value.extract(l);
    }
    acc
}

/// The fixed-order in-vector prefix of `value` (ADR-0009): lane l holds the sum of lanes 0..=l,
/// or of lanes 0..l if `exclusive`.
#[cube]
fn lane_prefix<T: Numeric, N: Size>(
    value: Vector<T, N>,
    #[comptime] exclusive: bool,
) -> Vector<T, N> {
    let mut prefix = Vector::<T, N>::broadcast(T::from_int(0));
    let mut acc = T::from_int(0);
    #[unroll]
    for l in 0..N::value() {
        if exclusive {
            prefix.insert(l, acc);
            acc += value.extract(l);
        } else {
            acc += value.extract(l);
            prefix.insert(l, acc);
        }
    }
    prefix
}

/// Phase 3: `out[v]` for every vector `v < num_vectors` of the cube's tile (Listing B5 loop).
/// Lanes at index >= `len` count as 0, so in the exclusive scan the lane at index `len` holds
/// the total. `tile_offsets[CUBE_POS]` is read only if `tiled`.
#[cube(launch_unchecked)]
fn scan_tile_kernel<T: Numeric, N: Size>(
    x: &[Vector<T, N>],
    tile_offsets: &[T],
    out: &mut [Vector<T, N>],
    len: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] cube_dim: u32,
    #[comptime] exclusive: bool,
    #[comptime] tiled: bool,
) {
    let len = len as usize;
    let num_vectors = num_vectors as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    let zero = T::from_int(0);

    // The unit's total, summed as the second pass sums its block.
    let mut total = zero;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors {
            total += lane_sum::<T, N>(load_masked::<T, N>(x, v, len));
        }
    }

    // Inclusive scan of the unit totals in shared memory (Hillis-Steele, fixed order).
    let u = UNIT_POS as usize;
    let mut shared = Shared::<[T]>::new_slice(cube_dim as usize);
    shared[u] = total;
    sync_cube();
    let mut stride = 1usize;
    while stride < cube_dim as usize {
        let mut add = zero;
        if u >= stride {
            add = shared[u - stride];
        }
        sync_cube();
        if u >= stride {
            shared[u] = add + shared[u];
        }
        sync_cube();
        stride *= 2;
    }
    let mut unit_offset = zero;
    if u > 0 {
        unit_offset = shared[u - 1];
    }
    let mut tile_offset = zero;
    if tiled {
        tile_offset = tile_offsets[CUBE_POS];
    }
    let unit_offset = Vector::<T, N>::broadcast(unit_offset);
    let tile_offset = Vector::<T, N>::broadcast(tile_offset);

    // The block, sequentially; the unit offset, then the tile offset, added last.
    let mut block = zero;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors {
            let value = load_masked::<T, N>(x, v, len);
            let local = lane_prefix::<T, N>(value, exclusive) + Vector::<T, N>::broadcast(block);
            out[v] = tile_offset + (unit_offset + local);
            block += lane_sum::<T, N>(value);
        }
    }
}

/// Inclusive prefix sums: `out[i] = x[0] + … + x[i]`, with `x.len()` elements.
///
/// The result is bitwise identical across runs on one device. Float sums depend on the vector
/// size and the launch shape, which the vector policy fixes per device class; integer sums wrap.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, inclusive_scan};
///
/// let client = gridcl_testing::test_client();
/// let x = DeviceBuffer::from_host(&client, &[3u32, 1, 4, 1, 5]).unwrap();
/// assert_eq!(inclusive_scan(&x).unwrap().to_host().unwrap(), vec![3, 4, 8, 9, 14]);
/// ```
pub fn inclusive_scan<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>> {
    if x.len() == 0 {
        return DeviceBuffer::uninit(x.client(), 0);
    }
    scan(x, false)
}

/// Exclusive prefix sums with **n + 1** elements, the last being the total, so the result can
/// serve as CSR offsets. This differs from the usual length-n exclusive scan:
/// `out[i] = x[0] + … + x[i - 1]` for `i` in `0..=n`, and `out[0] = 0`. An empty input gives
/// `[0]`.
///
/// Determinism as for [`inclusive_scan`].
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, exclusive_scan};
///
/// let client = gridcl_testing::test_client();
/// let x = DeviceBuffer::from_host(&client, &[3u32, 1, 4, 1, 5]).unwrap();
/// let offsets = exclusive_scan(&x).unwrap();
/// assert_eq!(offsets.len(), 6);
/// assert_eq!(offsets.to_host().unwrap(), vec![0, 3, 4, 8, 9, 14]);
/// ```
pub fn exclusive_scan<T: DeviceScalar>(x: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>> {
    if x.len() == 0 {
        let mut out = DeviceBuffer::uninit(x.client(), 1)?;
        fill(&mut out, T::from_int(0))?;
        return Ok(out);
    }
    scan(x, true)
}

/// The scan of a non-empty `x`. The exclusive scan plans over `len + 1` items, so its last
/// element is the masked lane (or vector) after the data and receives the total.
fn scan<T: DeviceScalar>(x: &DeviceBuffer<T>, exclusive: bool) -> Result<DeviceBuffer<T>> {
    let client = x.client();
    let items = if exclusive {
        x.len().checked_add(1).ok_or(GridError::TooLarge)?
    } else {
        x.len()
    };
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::ScanTile,
        items,
        size_of::<T>(),
        VECTOR_PAD,
        3,
    )?;
    let cubes = plan.num_cubes();
    let tiled = cubes > 1;
    let tile_offsets = if tiled {
        if cubes + 1 >= items {
            // The recursive scan over cubes + 1 items would not shrink (only possible with
            // injected limits: one unit and one item per cube).
            return Err(GridError::TooLarge);
        }
        let sums = DeviceBuffer::<T>::uninit(client, cubes)?;
        launch_reduce(x, &sums, &plan, ReduceOp::Sum)?;
        scan(&sums, true)?
    } else {
        // Not read: a single tile has offset 0.
        DeviceBuffer::<T>::uninit(client, 1)?
    };
    let out = DeviceBuffer::<T>::uninit(client, items)?;
    launch_scan_tile(x, &tile_offsets, &out, &plan, exclusive, tiled)?;
    Ok(out)
}

/// Phase 3 under `plan`.
fn launch_scan_tile<T: DeviceScalar>(
    x: &DeviceBuffer<T>,
    tile_offsets: &DeviceBuffer<T>,
    out: &DeviceBuffer<T>,
    plan: &LaunchPlan,
    exclusive: bool,
    tiled: bool,
) -> Result<()> {
    let client = x.client();
    let n = plan.vector_size;
    let len = u32::try_from(x.len()).map_err(|_| GridError::TooLarge)?;
    let cube_dim = plan.cube_dim.num_elems();
    debug_assert_bindings(client, 3);
    // SAFETY: the kernel reads vector v of `x` only if v · n < len; `x` holds `padded_len()`
    // elements, a multiple of n that is at least len, so v + 1 vectors fit. `out` holds
    // pad_len(items) elements and the plan's num_vectors is pad_len(items) / n; the kernel
    // writes only vectors below num_vectors. `tile_offsets` has num_cubes + 1 elements when
    // `tiled`, and each cube reads only tile_offsets[CUBE_POS]; otherwise it is not read. The
    // shared array has cube_dim entries, indexed by UNIT_POS < cube_dim.
    unsafe {
        scan_tile_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            n,
            x.vector_arg(n),
            tile_offsets.arg(),
            out.vector_arg(n),
            len,
            plan.num_vectors,
            plan.items_per_unit,
            cube_dim,
            exclusive,
            tiled,
        );
    }
    Ok(())
}
