//! Stable LSD radix sort with 8-bit digits; SortDigits is generic over the vector size.
//!
//! One partition, from the SortDigits plan: a unit owns `items_per_unit · vector_size`
//! consecutive keys and a cube `cube_dim` times as many. Each pass sorts by one 8-bit digit,
//! from the last word's lowest digit to word 0's highest, in three launches:
//!
//! 1. **SortDigits** counts the digits of each cube in shared memory and writes the counts
//!    digit-major, `counts[digit · cubes + cube]`.
//! 2. The exclusive scan of the counts gives each (digit, cube) its first output position.
//! 3. A stable scatter ranks the keys of each cube with the same digit in input order and
//!    writes each key and its payload to its offset plus its rank. The ranking scheme is
//!    chosen from the plan:
//!    - **Blocks** (`items_per_unit > 1`, CPUs): each unit counts the digits of its block in
//!      shared memory, the counts are scanned across the units of the cube, and each unit
//!      walks its block again. The digits are processed in chunks that fit in shared memory.
//!    - **Split** (`items_per_unit = 1`, GPUs): the cube sorts its digits stably in shared
//!      memory by eight 1-bit splits; a key's rank is its position minus the first position
//!      of its digit.
//!
//! The counts are integers and every rank follows the input order, so the result is bitwise
//! deterministic. The passes ping-pong between two buffers; no cube waits for another.

use cubecl::prelude::*;

use crate::buffer::pad_len;
use crate::plan::debug_assert_bindings;
use crate::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, SortKeys, VECTOR_PAD, exclusive_scan,
    testing,
};

/// Digits per pass: 8 bits.
const RADIX: usize = 256;
/// Bits of a packed split slot that hold the key's position in its cube; the digit is above.
const SLOT_BITS: u32 = 16;

/// Digit counts of the cube's keys, written to `counts[digit · cubes + CUBE_POS]`. Reads the
/// plane of the pass's word, which starts at vector `plane`; lanes at index >= `len` are not
/// counted.
#[cube(launch_unchecked)]
fn sort_digits_kernel<N: Size>(
    words: &[Vector<u32, N>],
    counts: &mut [u32],
    plane: u32,
    shift: u32,
    len: u32,
    num_vectors: u32,
    items_per_unit: u32,
    cubes: u32,
    #[comptime] cube_dim: u32,
) {
    let lanes = N::value();
    let hist = Shared::<[Atomic<u32>]>::new_slice(RADIX);
    let mut b = UNIT_POS as usize;
    while b < RADIX {
        hist[b].store(0u32);
        b += cube_dim as usize;
    }
    sync_cube();
    let len = len as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let start = v * lanes;
            let digits = (words[plane as usize + v] >> Vector::<u32, N>::broadcast(shift))
                & Vector::<u32, N>::broadcast(255u32);
            if start + lanes <= len {
                #[unroll]
                for l in 0..lanes {
                    hist[digits.extract(l) as usize].fetch_add(1u32);
                }
            } else if start < len {
                // The one vector that holds the end.
                #[unroll]
                for l in 0..lanes {
                    if start + l < len {
                        hist[digits.extract(l) as usize].fetch_add(1u32);
                    }
                }
            }
        }
    }
    sync_cube();
    let mut b = UNIT_POS as usize;
    while b < RADIX {
        counts[b * cubes as usize + CUBE_POS] = hist[b].load();
        b += cube_dim as usize;
    }
}

/// Writes key `j` (every word) and its payload to position `dest`.
#[cube]
fn move_key(
    keys_in: &[u32],
    payload_in: &[u32],
    keys_out: &mut [u32],
    payload_out: &mut [u32],
    j: usize,
    dest: usize,
    pad: usize,
    num_words: u32,
) {
    for w in 0..num_words as usize {
        keys_out[w * pad + dest] = keys_in[w * pad + j];
    }
    payload_out[dest] = payload_in[j];
}

/// The blocks scheme (`items_per_unit > 1`): each unit owns `span` consecutive keys. Per digit
/// chunk, each unit counts its block's digits in its column of `counts`, the columns are
/// scanned across the units in order, and each unit walks its block again, taking ranks from
/// its column.
#[cube(launch_unchecked)]
fn scatter_blocks_kernel(
    keys_in: &[u32],
    payload_in: &[u32],
    offsets: &[u32],
    keys_out: &mut [u32],
    payload_out: &mut [u32],
    len: u32,
    pad: u32,
    num_words: u32,
    plane: u32,
    shift: u32,
    span: u32,
    cubes: u32,
    #[comptime] cube_dim: u32,
    #[comptime] chunk_digits: u32,
) {
    let u = UNIT_POS as usize;
    let units = cube_dim as usize;
    let mut counts = Shared::<[u32]>::new_slice(chunk_digits as usize * units);
    let len = len as usize;
    let pad = pad as usize;
    let plane = plane as usize;
    let first = ABSOLUTE_POS * span as usize;
    let mut d0 = 0u32;
    while d0 < RADIX as u32 {
        for d in 0..chunk_digits as usize {
            counts[d * units + u] = 0u32;
        }
        for i in 0..span as usize {
            let j = first + i;
            if j < len {
                let d = ((keys_in[plane + j] >> shift) & 255u32) - d0;
                if d < chunk_digits {
                    counts[d as usize * units + u] += 1u32;
                }
            }
        }
        sync_cube();
        // Exclusive scan of each digit's row across the units, in unit order.
        let mut d = u;
        while d < chunk_digits as usize {
            let mut acc = 0u32;
            for v in 0..units {
                let c = counts[d * units + v];
                counts[d * units + v] = acc;
                acc += c;
            }
            d += units;
        }
        sync_cube();
        for i in 0..span as usize {
            let j = first + i;
            if j < len {
                let digit = (keys_in[plane + j] >> shift) & 255u32;
                let d = digit - d0;
                if d < chunk_digits {
                    let slot = d as usize * units + u;
                    let rank = counts[slot];
                    counts[slot] = rank + 1u32;
                    let dest = offsets[digit as usize * cubes as usize + CUBE_POS] + rank;
                    move_key(
                        keys_in,
                        payload_in,
                        keys_out,
                        payload_out,
                        j,
                        dest as usize,
                        pad,
                        num_words,
                    );
                }
            }
        }
        sync_cube();
        d0 += chunk_digits;
    }
}

/// The split scheme (`items_per_unit = 1`): each unit owns `N` consecutive keys. The cube's
/// `num_slots = cube_dim · N` slots hold `digit << SLOT_BITS | position`; eight stable 1-bit splits sort them by digit,
/// and a key's rank is its sorted slot minus the first slot of its digit. Keys at index
/// >= `len` take digit 255 and sort after every real key, so they change no rank.
#[cube(launch_unchecked)]
fn scatter_split_kernel<N: Size>(
    keys_in: &[u32],
    payload_in: &[u32],
    offsets: &[u32],
    keys_out: &mut [u32],
    payload_out: &mut [u32],
    len: u32,
    pad: u32,
    num_words: u32,
    plane: u32,
    shift: u32,
    cubes: u32,
    #[comptime] cube_dim: u32,
    #[comptime] num_slots: u32,
) {
    let lanes = N::value();
    let units = cube_dim as usize;
    let mut slots = Shared::<[u32]>::new_slice(num_slots as usize);
    let mut totals = Shared::<[u32]>::new_slice(cube_dim as usize);
    let mut starts = Shared::<[u32]>::new_slice(RADIX);
    let len = len as usize;
    let pad = pad as usize;
    let u = UNIT_POS as usize;
    let base = CUBE_POS * units * lanes;
    let mine = u * lanes;

    #[unroll]
    for l in 0..lanes {
        let j = base + mine + l;
        let mut digit = 255u32;
        if j < len {
            digit = (keys_in[plane as usize + j] >> shift) & 255u32;
        }
        slots[mine + l] = (digit << SLOT_BITS) | (mine + l) as u32;
    }
    sync_cube();

    for b in 0..8u32 {
        let bit_shift = SLOT_BITS + b;
        let mut values = Vector::<u32, N>::broadcast(0u32);
        let mut zeros = 0u32;
        #[unroll]
        for l in 0..lanes {
            let x = slots[mine + l];
            values.insert(l, x);
            zeros += 1u32 - ((x >> bit_shift) & 1u32);
        }
        // Inclusive scan of the units' zero counts (Hillis-Steele, fixed order).
        totals[u] = zeros;
        sync_cube();
        let mut stride = 1usize;
        while stride < units {
            let mut add = 0u32;
            if u >= stride {
                add = totals[u - stride];
            }
            sync_cube();
            totals[u] += add;
            sync_cube();
            stride *= 2;
        }
        let all_zeros = totals[units - 1];
        // Zeros before the slot: a zero goes there, a one after all zeros.
        let mut z = totals[u] - zeros;
        #[unroll]
        for l in 0..lanes {
            let x = values.extract(l);
            let bit = (x >> bit_shift) & 1u32;
            let p = (mine + l) as u32;
            let dest = (1u32 - bit) * z + bit * (all_zeros + p - z);
            slots[dest as usize] = x;
            z += 1u32 - bit;
        }
        sync_cube();
    }

    #[unroll]
    for l in 0..lanes {
        let p = mine + l;
        let digit = slots[p] >> SLOT_BITS;
        let mut prev = 256u32;
        if p > 0 {
            prev = slots[p - 1] >> SLOT_BITS;
        }
        if digit != prev {
            starts[digit as usize] = p as u32;
        }
    }
    sync_cube();
    #[unroll]
    for l in 0..lanes {
        let p = mine + l;
        let x = slots[p];
        let digit = x >> SLOT_BITS;
        let j = base + (x & ((1u32 << SLOT_BITS) - 1u32)) as usize;
        if j < len {
            let rank = p as u32 - starts[digit as usize];
            let dest = offsets[digit as usize * cubes as usize + CUBE_POS] + rank;
            move_key(
                keys_in,
                payload_in,
                keys_out,
                payload_out,
                j,
                dest as usize,
                pad,
                num_words,
            );
        }
    }
}

/// How the scatter ranks the keys of a cube.
#[derive(Clone, Copy, Debug)]
enum Scheme {
    /// Per-unit counters in shared memory, `chunk_digits` digits at a time.
    Blocks { chunk_digits: u32 },
    /// Eight 1-bit splits in shared memory.
    Split,
}

/// Stable LSD sort with 8-bit digits: the keys in ascending order (word 0 most significant),
/// and the payload permuted with them. Equal keys keep their input order.
///
/// The result is bitwise identical across runs.
///
/// # Errors
///
/// `LengthMismatch { expected: keys.len(), found: payload.len() }` if the lengths differ;
/// `TooLarge` if there are 2^32 or more keys, if the planar words or the digit counts
/// overflow u32 indices, if the device's shared memory cannot hold a cube's ranking, or if the
/// scan of the digit counts returns it (see [`exclusive_scan`](crate::exclusive_scan)).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, SortKeys, radix_sort};
///
/// let client = gridcl_testing::test_client();
/// // Keys of two words, key-major: (1, 5), (0, 9), (1, 2), (0, 9).
/// let keys = SortKeys::from_host(&client, &[1, 5, 0, 9, 1, 2, 0, 9], 2).unwrap();
/// let payload = DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3]).unwrap();
/// let (sorted, perm) = radix_sort(&keys, &payload).unwrap();
/// assert_eq!(sorted.to_host().unwrap(), vec![0, 9, 0, 9, 1, 2, 1, 5]);
/// assert_eq!(perm.to_host().unwrap(), vec![1, 3, 2, 0]); // stable: 1 before 3
/// ```
pub fn radix_sort(
    keys: &SortKeys,
    payload: &DeviceBuffer<u32>,
) -> Result<(SortKeys, DeviceBuffer<u32>)> {
    let len = keys.len();
    if payload.len() != len {
        return Err(GridError::LengthMismatch {
            expected: len,
            found: payload.len(),
        });
    }
    let client = keys.words().client();
    let num_words = keys.num_words();
    if len == 0 {
        return Ok((
            SortKeys::uninit(client, 0, num_words)?,
            DeviceBuffer::uninit(client, 0)?,
        ));
    }
    let to_u32 = |v: usize| u32::try_from(v).map_err(|_| GridError::TooLarge);
    let len_u32 = to_u32(len)?;
    let pad = pad_len(len)?;
    to_u32(keys.words().padded_len())?;

    let plan = LaunchPlan::elementwise(client, KernelKind::SortDigits, len, 4, VECTOR_PAD, 2)?;
    let n = plan.vector_size;
    let cube_dim = plan.cube_dim.num_elems();
    let cubes = plan.num_cubes();
    let num_counts = cubes.checked_mul(RADIX).ok_or(GridError::TooLarge)?;
    to_u32(num_counts + 1)?;
    let shared = testing::properties(client).hardware.max_shared_memory_size;
    let scheme = scheme(&plan, shared)?;

    let mut out = Vec::with_capacity(2);
    for _ in 0..2 {
        out.push((
            SortKeys::uninit(client, len, num_words)?,
            DeviceBuffer::<u32>::uninit(client, len)?,
        ));
    }
    let passes = 4 * num_words;
    for pass in 0..passes {
        let (src_keys, src_payload) = if pass == 0 {
            (keys, payload)
        } else {
            let (k, p) = &out[(pass - 1) % 2];
            (k, p)
        };
        let (dst_keys, dst_payload) = &out[pass % 2];
        let word = num_words - 1 - pass / 4;
        let plane = to_u32(word * pad)?;
        let shift = 8 * (pass % 4) as u32;

        let counts = DeviceBuffer::<u32>::uninit(client, num_counts)?;
        debug_assert_bindings(client, 2);
        // SAFETY: the words buffer holds num_words · pad elements, pad / n whole vectors per
        // plane; the kernel reads vector plane / n + v only for v < num_vectors = pad / n, so
        // within the plane of `word`. `counts` holds 256 · cubes elements, and each cube
        // writes counts[b · cubes + CUBE_POS] for b < 256 and CUBE_POS < cubes. The shared
        // array has 256 entries, indexed by digits (< 256) and by b < 256.
        unsafe {
            sort_digits_kernel::launch_unchecked(
                client,
                plan.cube_count.clone(),
                plan.cube_dim,
                n,
                src_keys.words().vector_arg(n),
                counts.arg(),
                plane / n as u32,
                shift,
                len_u32,
                plan.num_vectors,
                plan.items_per_unit,
                cubes as u32,
                cube_dim,
            );
        }
        let offsets = exclusive_scan(&counts)?;
        debug_assert_bindings(client, 5);
        // SAFETY (both schemes): the kernels read key j and payload j only for j < len, at
        // w · pad + j < num_words · pad for w < num_words, and write them at dest = offsets of
        // (digit, cube) plus the key's rank. The rank counts the cube's earlier keys with the
        // same digit, and the offsets are the exclusive scan of those counts, so dest < len
        // and every dest is written once. `offsets` holds 256 · cubes + 1 elements and is read
        // at digit · cubes + CUBE_POS. Blocks: the shared array has chunk_digits · cube_dim
        // entries, indexed by d < chunk_digits and units < cube_dim. Split: the slots hold
        // cube_dim · n entries, indexed below that by positions and by the destinations of
        // the split (a permutation of the slots); `totals` has cube_dim entries and `starts`
        // 256, indexed by digits.
        unsafe {
            match scheme {
                Scheme::Blocks { chunk_digits } => scatter_blocks_kernel::launch_unchecked(
                    client,
                    plan.cube_count.clone(),
                    plan.cube_dim,
                    src_keys.words().arg(),
                    src_payload.arg(),
                    offsets.arg(),
                    dst_keys.words().arg(),
                    dst_payload.arg(),
                    len_u32,
                    pad as u32,
                    num_words as u32,
                    plane,
                    shift,
                    plan.items_per_unit * n as u32,
                    cubes as u32,
                    cube_dim,
                    chunk_digits,
                ),
                Scheme::Split => scatter_split_kernel::launch_unchecked(
                    client,
                    plan.cube_count.clone(),
                    plan.cube_dim,
                    n,
                    src_keys.words().arg(),
                    src_payload.arg(),
                    offsets.arg(),
                    dst_keys.words().arg(),
                    dst_payload.arg(),
                    len_u32,
                    pad as u32,
                    num_words as u32,
                    plane,
                    shift,
                    cubes as u32,
                    cube_dim,
                    cube_dim * n as u32,
                ),
            }
        }
    }
    Ok(out.swap_remove((passes - 1) % 2))
}

/// The ranking scheme of `plan` on a device with `shared` bytes of shared memory per cube.
fn scheme(plan: &LaunchPlan, shared: usize) -> Result<Scheme> {
    let units = plan.cube_dim.num_elems() as usize;
    if plan.items_per_unit > 1 {
        // The largest power-of-two chunk of digits whose per-unit counters fit.
        let fit = shared / (units * size_of::<u32>());
        if fit == 0 {
            return Err(GridError::TooLarge);
        }
        let chunk_digits = (1usize << fit.ilog2()).min(RADIX);
        Ok(Scheme::Blocks {
            chunk_digits: chunk_digits as u32,
        })
    } else {
        let slots = units * plan.vector_size;
        let bytes = (slots + units + RADIX) * size_of::<u32>();
        if slots > 1 << SLOT_BITS || bytes > shared {
            return Err(GridError::TooLarge);
        }
        Ok(Scheme::Split)
    }
}
