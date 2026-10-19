//! Run-length encoding and unique-by-key of sorted keys.

use crate::{DeviceBuffer, Result, SortKeys};

/// Runs of consecutive equal keys; `offsets` has runs + 1 entries.
pub struct Runs {
    pub keys: SortKeys,
    pub offsets: DeviceBuffer<u32>,
}

/// The runs of consecutive equal keys. Reads the run count back once (a blocking 4-byte
/// read), so the result is exactly sized.
pub fn run_length(_keys: &SortKeys) -> Result<Runs> {
    todo!()
}

/// `first`: payload of each group's first key; `group`: the group of each input position.
pub struct Unique {
    pub keys: SortKeys,
    pub first: DeviceBuffer<u32>,
    pub group: DeviceBuffer<u32>,
}

/// Groups equal neighbours of `sorted`. Reads the group count back once (a blocking 4-byte
/// read), so the result is exactly sized.
pub fn unique_by_key(_sorted: &SortKeys, _payload: &DeviceBuffer<u32>) -> Result<Unique> {
    todo!()
}
