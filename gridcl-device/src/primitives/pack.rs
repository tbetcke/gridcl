//! Pack and unpack (KernelKind::Pack), generic over the vector size: the device side of the
//! host-staged halo exchange (ADR-0026, item 13).
//!
//! Both move u32 words. With w = size_of::<T>() / 4 words per element, an index addresses a
//! block of `block · w` words; word o of the packed buffer belongs to index k = o / (block · w)
//! at offset o mod (block · w). Lanes run over packed words: pack writes one vector of packed
//! words per step, unpack reads one.

use cubecl::prelude::*;

use crate::plan::debug_assert_bindings;
use crate::primitives::launch_fill;
use crate::{DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, VECTOR_PAD};

/// `out[v]` lane by lane: packed word o = v·N + l is word o mod `block_words` of source block
/// `idx[o / block_words]`. Lanes past `total` and source blocks past `src_blocks` give 0
/// through a select; read addresses are clamped, so no lane branches on its index.
#[cube(launch_unchecked)]
fn pack_kernel<N: Size>(
    src: &[u32],
    idx: &[u32],
    out: &mut [Vector<u32, N>],
    total: u32,
    block_words: u32,
    num_idx: u32,
    src_blocks: u32,
    num_vectors: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let mut value = Vector::<u32, N>::broadcast(0u32);
            #[unroll]
            for l in 0..N::value() {
                let o = (v * N::value() + l) as u32;
                let k = min(o / block_words, num_idx - 1);
                let r = o % block_words;
                let block = idx[k as usize];
                let s = min(block, src_blocks - 1) * block_words + r;
                let x = src[s as usize];
                let x = select(block < src_blocks, x, 0u32);
                value.insert(l, select(o < total, x, 0u32));
            }
            out[v] = value;
        }
    }
}

/// Writes the lanes of `packed[v]`: packed word o = v·N + l goes to word o mod `block_words`
/// of destination block `idx[o / block_words]`, clamped to the last block. Lanes past `total`
/// write nothing (the guard depends on the lane position only, not on data).
#[cube(launch_unchecked)]
fn unpack_kernel<N: Size>(
    packed: &[Vector<u32, N>],
    idx: &[u32],
    dst: &mut [u32],
    total: u32,
    block_words: u32,
    dst_blocks: u32,
    num_vectors: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let value = packed[v];
            #[unroll]
            for l in 0..N::value() {
                let o = (v * N::value() + l) as u32;
                if o < total {
                    let k = o / block_words;
                    let r = o % block_words;
                    let d = min(idx[k as usize], dst_blocks - 1) * block_words + r;
                    dst[d as usize] = value.extract(l);
                }
            }
        }
    }
}

/// The u32 words of `n` elements of `T`, as u32; `TooLarge` past u32.
fn words<T>(n: usize) -> Result<u32> {
    n.checked_mul(size_of::<T>() / 4)
        .and_then(|w| u32::try_from(w).ok())
        .ok_or(GridError::TooLarge)
}

/// `LengthMismatch` unless `len` is a whole number of blocks of `block`; the block count.
fn blocks_of(len: usize, block: usize) -> Result<usize> {
    if !len.is_multiple_of(block) {
        return Err(GridError::LengthMismatch {
            expected: len.next_multiple_of(block),
            found: len,
        });
    }
    Ok(len / block)
}

/// `LengthMismatch` unless `packed` holds `block` elements per index.
fn check_packed(packed: usize, num_idx: usize, block: usize) -> Result<()> {
    let expected = num_idx.checked_mul(block).ok_or(GridError::TooLarge)?;
    if packed != expected {
        return Err(GridError::LengthMismatch {
            expected,
            found: packed,
        });
    }
    Ok(())
}

/// Packs blocks of `src` in the order of `idx`:
/// `out[k·block .. (k+1)·block] = src[idx[k]·block .. (idx[k]+1)·block]`, moved as u32 words,
/// so `size_of::<T>()` must be a multiple of 4 (checked at compile time). An index past the
/// last block of `src` packs zeros. An empty `idx` launches nothing.
///
/// # Errors
///
/// `LengthMismatch` if `block` is 0, if `src.len()` is not a multiple of `block`, or if
/// `out.len()` is not `idx.len() · block`; `TooLarge` if the packed or source words do not fit
/// u32.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, pack};
///
/// let client = gridcl_testing::test_client();
/// // Three blocks of two values.
/// let src = DeviceBuffer::from_host(&client, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let idx = DeviceBuffer::from_host(&client, &[2u32, 0]).unwrap();
/// let mut out = DeviceBuffer::<f32>::uninit(&client, 4).unwrap();
/// pack(&src, &idx, 2, &mut out).unwrap();
/// assert_eq!(out.to_host().unwrap(), vec![5.0, 6.0, 1.0, 2.0]);
/// ```
pub fn pack<T: CubeElement>(
    src: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
    block: usize,
    out: &mut DeviceBuffer<T>,
) -> Result<()> {
    const { assert!(size_of::<T>().is_multiple_of(4), "pack moves u32 words") };
    if block == 0 {
        return Err(GridError::LengthMismatch {
            expected: 1,
            found: 0,
        });
    }
    let src_blocks = blocks_of(src.len(), block)?;
    check_packed(out.len(), idx.len(), block)?;
    if idx.len() == 0 {
        return Ok(());
    }
    let client = src.client();
    let total = words::<T>(out.len())?;
    if src_blocks == 0 {
        // Every index is past the end: zeros, padding included.
        let padded = words::<T>(out.padded_len())?;
        return launch_fill::<u32>(client, out.handle(), padded as usize, 0);
    }
    let block_words = words::<T>(block)?;
    let src_blocks = u32::try_from(src_blocks).map_err(|_| GridError::TooLarge)?;
    words::<T>(src.len())?;
    let num_idx = u32::try_from(idx.len()).map_err(|_| GridError::TooLarge)?;
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::Pack,
        total as usize,
        size_of::<u32>(),
        VECTOR_PAD,
        1,
    )?;
    let n = plan.vector_size;
    // pad_len(out.len()) elements are a multiple of VECTOR_PAD and at least `total` words, so
    // the output handle holds the plan's num_vectors · n words.
    let out_words = out.padded_len() * (size_of::<T>() / 4);
    debug_assert!(plan.num_vectors as usize * n <= out_words);
    debug_assert_bindings(client, 3);
    // SAFETY: the kernel writes `out` only at vectors below num_vectors, which lie inside its
    // padded words (above); it reads `idx` at k <= num_idx - 1 < idx.len(), and `src` at
    // min(block, src_blocks - 1) · block_words + r with r < block_words, below
    // src_blocks · block_words = the words of `src`, which is not empty.
    unsafe {
        pack_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            BufferArg::from_raw_parts(
                src.handle().clone(),
                src.padded_len() * (size_of::<T>() / 4),
            ),
            idx.arg(),
            BufferArg::from_raw_parts(out.handle().clone(), out_words / n),
            total,
            block_words,
            num_idx,
            src_blocks,
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    Ok(())
}

/// Unpacks blocks into `dst` at the positions of `idx`:
/// `dst[idx[k]·block .. (idx[k]+1)·block] = packed[k·block .. (k+1)·block]`, moved as u32
/// words, so `size_of::<T>()` must be a multiple of 4 (checked at compile time). Blocks of
/// `dst` not named in `idx` are unchanged. The indices must be distinct and below
/// `dst.len() / block`; in release builds an index past the end writes the last block. An
/// empty `idx` launches nothing.
///
/// # Errors
///
/// `LengthMismatch` if `block` is 0, if `dst.len()` is not a multiple of `block`, or if
/// `packed.len()` is not `idx.len() · block`; `TooLarge` if the packed or destination words do
/// not fit u32.
///
/// # Panics
///
/// In debug builds, on an index past the end or a repeated index (ADR-0009): the checks read
/// back the maximum index and the maximum of `histogram(idx, dst.len() / block)`.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, unpack};
///
/// let client = gridcl_testing::test_client();
/// let packed = DeviceBuffer::from_host(&client, &[7i32, 8, 9, 10]).unwrap();
/// let idx = DeviceBuffer::from_host(&client, &[2u32, 0]).unwrap();
/// let mut dst = DeviceBuffer::from_host(&client, &[0i32; 6]).unwrap();
/// unpack(&packed, &idx, 2, &mut dst).unwrap();
/// assert_eq!(dst.to_host().unwrap(), vec![9, 10, 0, 0, 7, 8]);
/// ```
pub fn unpack<T: CubeElement>(
    packed: &DeviceBuffer<T>,
    idx: &DeviceBuffer<u32>,
    block: usize,
    dst: &mut DeviceBuffer<T>,
) -> Result<()> {
    const { assert!(size_of::<T>().is_multiple_of(4), "unpack moves u32 words") };
    if block == 0 {
        return Err(GridError::LengthMismatch {
            expected: 1,
            found: 0,
        });
    }
    let dst_blocks = blocks_of(dst.len(), block)?;
    check_packed(packed.len(), idx.len(), block)?;
    if idx.len() == 0 {
        return Ok(());
    }
    let dst_blocks = u32::try_from(dst_blocks).map_err(|_| GridError::TooLarge)?;
    #[cfg(debug_assertions)]
    {
        let max = super::reduce(idx, crate::ReduceOp::Max)?.to_host()?[0];
        assert!(
            max < dst_blocks,
            "unpack: index out of range: {max} >= {dst_blocks} blocks"
        );
        let counts = super::histogram(idx, dst_blocks)?;
        let max = super::reduce(&counts, crate::ReduceOp::Max)?.to_host()?[0];
        assert!(
            max <= 1,
            "unpack: duplicate index ({max} writes to one block)"
        );
    }
    if dst_blocks == 0 {
        return Ok(());
    }
    let client = packed.client();
    let total = words::<T>(packed.len())?;
    let block_words = words::<T>(block)?;
    words::<T>(dst.len())?;
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::Pack,
        total as usize,
        size_of::<u32>(),
        VECTOR_PAD,
        1,
    )?;
    let n = plan.vector_size;
    let packed_words = packed.padded_len() * (size_of::<T>() / 4);
    debug_assert!(plan.num_vectors as usize * n <= packed_words);
    debug_assert_bindings(client, 3);
    // SAFETY: the kernel reads `packed` only at vectors below num_vectors, inside its padded
    // words (as for pack); it reads `idx` at k = o / block_words < num_idx for o < total, and
    // writes `dst` at min(idx, dst_blocks - 1) · block_words + r < dst_blocks · block_words, the
    // words of `dst`, which is not empty.
    unsafe {
        unpack_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            BufferArg::from_raw_parts(packed.handle().clone(), packed_words / n),
            idx.arg(),
            BufferArg::from_raw_parts(
                dst.handle().clone(),
                dst.padded_len() * (size_of::<T>() / 4),
            ),
            total,
            block_words,
            dst_blocks,
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    Ok(())
}
