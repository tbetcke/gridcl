//! Run-length encoding and unique-by-key of sorted keys (scalar).
//!
//! One partition, from the plan of the first phase. A flag marks each key that differs from
//! its predecessor (the first key always), the exclusive scan of the flags numbers the runs,
//! and a compaction writes each run's first key to its number. The run count, the last element
//! of the scan, is read back once (a blocking 4-byte read, ADR-0009), so the outputs are
//! exactly sized. Integer scans are exact, so the result is bitwise deterministic.

use cubecl::prelude::*;

use crate::buffer::pad_len;
use crate::plan::debug_assert_bindings;
use crate::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, SortKeys, exclusive_scan, fill,
};

/// `flags[i]` = 1 if key `i` differs from key `i - 1` in any word, or `i` = 0; else 0.
#[cube(launch_unchecked)]
fn flags_kernel(
    words: &[u32],
    flags: &mut [u32],
    len: u32,
    pad: u32,
    num_words: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    let pad = pad as usize;
    for k in 0..items_per_unit as usize {
        let i = first + k;
        if i < len as usize {
            // Key 0 compares with itself, then the select makes it a run start.
            let prev = i - min(i, 1usize);
            let mut diff = 0u32;
            for w in 0..num_words as usize {
                diff |= words[w * pad + i] ^ words[w * pad + prev];
            }
            flags[i] = select(i == 0 || diff != 0u32, 1u32, 0u32);
        }
    }
}

/// Writes key `i` (every word) to run `r` of the output keys.
#[cube]
fn write_key(
    words: &[u32],
    keys_out: &mut [u32],
    i: usize,
    r: usize,
    pad: usize,
    pad_out: usize,
    num_words: u32,
) {
    for w in 0..num_words as usize {
        keys_out[w * pad_out + r] = words[w * pad + i];
    }
}

/// Run-length compaction: run `r = starts[i]` starts at every `i` with
/// `starts[i + 1] != starts[i]`; its key goes to `keys_out` and `i` to `offsets[r]`. The last
/// key also writes `offsets[runs] = len`.
#[cube(launch_unchecked)]
fn runs_kernel(
    words: &[u32],
    starts: &[u32],
    keys_out: &mut [u32],
    offsets: &mut [u32],
    len: u32,
    pad: u32,
    pad_out: u32,
    num_words: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for k in 0..items_per_unit as usize {
        let i = first + k;
        if i < len as usize {
            let r = starts[i];
            let next = starts[i + 1];
            if next != r {
                write_key(
                    words,
                    keys_out,
                    i,
                    r as usize,
                    pad as usize,
                    pad_out as usize,
                    num_words,
                );
                offsets[r as usize] = i as u32;
            }
            if i + 1 == len as usize {
                offsets[next as usize] = len;
            }
        }
    }
}

/// Unique-by-key compaction: as [`runs_kernel`], with `first[r] = payload[i]` for each run
/// start and `group[i] = starts[i + 1] - 1` for every key.
#[cube(launch_unchecked)]
fn unique_kernel(
    words: &[u32],
    starts: &[u32],
    payload: &[u32],
    keys_out: &mut [u32],
    first_out: &mut [u32],
    group: &mut [u32],
    len: u32,
    pad: u32,
    pad_out: u32,
    num_words: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for k in 0..items_per_unit as usize {
        let i = first + k;
        if i < len as usize {
            let r = starts[i];
            let next = starts[i + 1];
            if next != r {
                write_key(
                    words,
                    keys_out,
                    i,
                    r as usize,
                    pad as usize,
                    pad_out as usize,
                    num_words,
                );
                first_out[r as usize] = payload[i];
            }
            group[i] = next - 1u32;
        }
    }
}

/// Runs of consecutive equal keys; `offsets` has runs + 1 entries.
///
/// # Example
///
/// ```
/// use gridcl_device::{Runs, SortKeys, run_length};
///
/// let client = gridcl_testing::test_client();
/// let Runs { keys, offsets } = run_length(&SortKeys::from_host(&client, &[7, 7, 9], 1).unwrap()).unwrap();
/// assert_eq!(keys.to_host().unwrap(), vec![7, 9]);
/// assert_eq!(offsets.to_host().unwrap(), vec![0, 2, 3]);
/// ```
pub struct Runs {
    /// The key of each run.
    pub keys: SortKeys,
    /// Run r covers the input keys `offsets[r]..offsets[r + 1]`; the last entry is the number
    /// of keys.
    pub offsets: DeviceBuffer<u32>,
}

/// The runs of consecutive equal keys (the keys need not be sorted). Reads the run count back
/// once (a blocking 4-byte read), so the result is exactly sized. Empty keys give no runs and
/// `offsets = [0]`.
///
/// # Errors
///
/// `TooLarge` if there are 2^32 or more keys.
///
/// # Example
///
/// ```
/// use gridcl_device::{SortKeys, run_length};
///
/// let client = gridcl_testing::test_client();
/// let keys = SortKeys::from_host(&client, &[4, 4, 4, 1, 4, 4], 1).unwrap();
/// let runs = run_length(&keys).unwrap();
/// assert_eq!(runs.keys.to_host().unwrap(), vec![4, 1, 4]);
/// assert_eq!(runs.offsets.to_host().unwrap(), vec![0, 3, 4, 6]);
/// ```
pub fn run_length(keys: &SortKeys) -> Result<Runs> {
    let client = keys.words().client();
    let num_words = keys.num_words();
    let len = keys.len();
    if len == 0 {
        let mut offsets = DeviceBuffer::uninit(client, 1)?;
        fill(&mut offsets, 0u32)?;
        return Ok(Runs {
            keys: SortKeys::uninit(client, 0, num_words)?,
            offsets,
        });
    }
    let (plan, starts, runs) = number_runs(keys)?;
    let out_keys = SortKeys::uninit(client, runs, num_words)?;
    let offsets = DeviceBuffer::<u32>::uninit(client, runs + 1)?;
    debug_assert_bindings(client, 4);
    // SAFETY: the kernel touches key i only for i < len, reading words w · pad + i below
    // num_words · pad, and starts[i] and starts[i + 1] with i + 1 <= len, within the len + 1
    // elements of `starts`. starts is the exclusive scan of 0/1 flags, so every run number r < runs and
    // starts[len] = runs: keys are written at w · pad_len(runs) + r within the output planes,
    // and offsets at r <= runs, within its runs + 1 elements.
    unsafe {
        runs_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            keys.words().arg(),
            starts.arg(),
            out_keys.words().arg(),
            offsets.arg(),
            to_u32(len)?,
            to_u32(pad_len(len)?)?,
            to_u32(pad_len(runs)?)?,
            to_u32(num_words)?,
            plan.items_per_unit,
        );
    }
    Ok(Runs {
        keys: out_keys,
        offsets,
    })
}

/// Groups of equal neighbours of sorted keys.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, SortKeys, Unique, unique_by_key};
///
/// let client = gridcl_testing::test_client();
/// let sorted = SortKeys::from_host(&client, &[1, 1, 5, 5, 5, 8], 1).unwrap();
/// let payload = DeviceBuffer::from_host(&client, &[10u32, 11, 12, 13, 14, 15]).unwrap();
/// let Unique { keys, first, group } = unique_by_key(&sorted, &payload).unwrap();
/// assert_eq!(keys.to_host().unwrap(), vec![1, 5, 8]);
/// assert_eq!(first.to_host().unwrap(), vec![10, 12, 15]);
/// assert_eq!(group.to_host().unwrap(), vec![0, 0, 1, 1, 1, 2]);
/// ```
pub struct Unique {
    /// The key of each group.
    pub keys: SortKeys,
    /// The payload of each group's first key.
    pub first: DeviceBuffer<u32>,
    /// The group of each input position.
    pub group: DeviceBuffer<u32>,
}

/// Groups equal neighbours of `sorted`. Reads the group count back once (a blocking 4-byte
/// read), so the result is exactly sized.
///
/// # Errors
///
/// `LengthMismatch { expected: sorted.len(), found: payload.len() }` if the lengths differ;
/// `TooLarge` if there are 2^32 or more keys.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, SortKeys, unique_by_key};
///
/// let client = gridcl_testing::test_client();
/// let keys = SortKeys::from_host(&client, &[2, 2, 5, 7, 7, 7], 1).unwrap();
/// let payload = DeviceBuffer::from_host(&client, &[10u32, 11, 12, 13, 14, 15]).unwrap();
/// let u = unique_by_key(&keys, &payload).unwrap();
/// assert_eq!(u.keys.to_host().unwrap(), vec![2, 5, 7]);
/// assert_eq!(u.first.to_host().unwrap(), vec![10, 12, 13]);
/// assert_eq!(u.group.to_host().unwrap(), vec![0, 0, 1, 2, 2, 2]);
/// ```
pub fn unique_by_key(sorted: &SortKeys, payload: &DeviceBuffer<u32>) -> Result<Unique> {
    let len = sorted.len();
    if payload.len() != len {
        return Err(GridError::LengthMismatch {
            expected: len,
            found: payload.len(),
        });
    }
    let client = sorted.words().client();
    let num_words = sorted.num_words();
    if len == 0 {
        return Ok(Unique {
            keys: SortKeys::uninit(client, 0, num_words)?,
            first: DeviceBuffer::uninit(client, 0)?,
            group: DeviceBuffer::uninit(client, 0)?,
        });
    }
    let (plan, starts, runs) = number_runs(sorted)?;
    let keys = SortKeys::uninit(client, runs, num_words)?;
    let first = DeviceBuffer::<u32>::uninit(client, runs)?;
    let group = DeviceBuffer::<u32>::uninit(client, len)?;
    debug_assert_bindings(client, 6);
    // SAFETY: as for `run_length`; in addition payload[i] and group[i] are touched only for
    // i < len = payload.len() = group.len(), and first[r] only for r < runs = first.len().
    unsafe {
        unique_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            sorted.words().arg(),
            starts.arg(),
            payload.arg(),
            keys.words().arg(),
            first.arg(),
            group.arg(),
            to_u32(len)?,
            to_u32(pad_len(len)?)?,
            to_u32(pad_len(runs)?)?,
            to_u32(num_words)?,
            plan.items_per_unit,
        );
    }
    Ok(Unique { keys, first, group })
}

/// The partition of the primitive, the run numbers (the exclusive scan of the run-start flags,
/// `len + 1` entries) and the run count, read back from the scan's last entry. `keys` is not
/// empty.
fn number_runs(keys: &SortKeys) -> Result<(LaunchPlan, DeviceBuffer<u32>, usize)> {
    let client = keys.words().client();
    let len = keys.len();
    let len_u32 = to_u32(len)?;
    let pad = to_u32(pad_len(len)?)?;
    to_u32(keys.words().padded_len())?;
    let plan = LaunchPlan::elementwise(client, KernelKind::Gather, len, 4, 1, 1)?;
    let flags = DeviceBuffer::<u32>::uninit(client, len)?;
    debug_assert_bindings(client, 2);
    // SAFETY: the kernel touches key i only for i < len: words w · pad + i and w · pad + i - 1
    // (or i for i = 0) below num_words · pad, and flags[i] within its len elements.
    unsafe {
        flags_kernel::launch_unchecked(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            keys.words().arg(),
            flags.arg(),
            len_u32,
            pad,
            to_u32(keys.num_words())?,
            plan.items_per_unit,
        );
    }
    let starts = exclusive_scan(&flags)?;
    let runs = read_u32(&starts, len)? as usize;
    Ok((plan, starts, runs))
}

/// `v` as a u32 index or length; `TooLarge` if it does not fit.
fn to_u32(v: usize) -> Result<u32> {
    u32::try_from(v).map_err(|_| GridError::TooLarge)
}

/// Element `i` of `x`: an explicit, blocking 4-byte device-to-host read (ADR-0009).
fn read_u32(x: &DeviceBuffer<u32>, i: usize) -> Result<u32> {
    debug_assert!(i < x.len(), "read_u32: index {i} out of range");
    let bytes_of = |n: usize| {
        n.checked_mul(size_of::<u32>())
            .and_then(|b| u64::try_from(b).ok())
            .ok_or(GridError::TooLarge)
    };
    let size = bytes_of(x.padded_len())?;
    let start = bytes_of(i)?;
    let handle = x.handle().clone().offset_start(start).offset_end(
        start
            .checked_add(size_of::<u32>() as u64)
            .and_then(|end| size.checked_sub(end))
            .expect("read_u32: the index is within the padded length"),
    );
    let bytes = x.client().read_one(handle)?;
    let word: [u8; 4] = bytes[..4].try_into().expect("a 4-byte read");
    Ok(u32::from_ne_bytes(word))
}
