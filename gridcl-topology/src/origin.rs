//! The device layout of origin ids (ADR-0012): interleaved (hi ^ 0x8000_0000, lo) u32 pairs.

/// The device words of `ids`: `(hi ^ 0x8000_0000, lo)` per id, interleaved, where hi and lo are
/// the high and low words of the i64. Unsigned lexicographic pair order equals i64 order.
///
/// # Example
///
/// ```
/// use gridcl_topology::encode_origin_ids;
///
/// assert_eq!(encode_origin_ids(&[5, -1]), vec![0x8000_0000, 5, 0x7fff_ffff, 0xffff_ffff]);
/// ```
pub fn encode_origin_ids(ids: &[i64]) -> Vec<u32> {
    let _ = ids;
    todo!()
}

/// The ids of interleaved device words; the inverse of [`encode_origin_ids`].
///
/// # Example
///
/// ```
/// use gridcl_topology::{decode_origin_ids, encode_origin_ids};
///
/// let ids = [i64::MIN, -3, 0, 1 << 40, i64::MAX];
/// assert_eq!(decode_origin_ids(&encode_origin_ids(&ids)), ids);
/// ```
///
/// # Panics
///
/// If `words.len()` is odd.
pub fn decode_origin_ids(words: &[u32]) -> Vec<i64> {
    let _ = words;
    todo!()
}

/// Whether every id lies in [0, 2^32): then every stored hi word is 0x8000_0000, and kernels
/// compare the lo words only (the comptime fast path).
///
/// # Example
///
/// ```
/// use gridcl_topology::origin_ids_fit_u32;
///
/// assert!(origin_ids_fit_u32(&[0, 7, u32::MAX as i64]));
/// assert!(!origin_ids_fit_u32(&[-1]));
/// assert!(!origin_ids_fit_u32(&[1 << 32]));
/// ```
pub fn origin_ids_fit_u32(ids: &[i64]) -> bool {
    let _ = ids;
    todo!()
}
