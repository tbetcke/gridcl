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
    let mut words = Vec::with_capacity(2 * ids.len());
    for &id in ids {
        let bits = id as u64;
        words.push(((bits >> 32) as u32) ^ SIGN);
        words.push(bits as u32);
    }
    words
}

/// The flipped sign bit of the hi word.
const SIGN: u32 = 0x8000_0000;

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
    assert!(
        words.len().is_multiple_of(2),
        "decode_origin_ids: {} words are not whole (hi, lo) pairs",
        words.len()
    );
    words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[hi, lo]| ((u64::from(hi ^ SIGN) << 32) | u64::from(*lo)) as i64)
        .collect()
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
    ids.iter()
        .all(|&id| (0..=i64::from(u32::MAX)).contains(&id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_order_is_id_order() {
        let mut ids = vec![
            i64::MIN,
            -(1 << 40),
            -1,
            0,
            1,
            i64::from(u32::MAX),
            1 << 32,
            i64::MAX,
        ];
        let words = encode_origin_ids(&ids);
        let pairs: Vec<(u32, u32)> = words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[h, l]| (h, l))
            .collect();
        assert!(pairs.windows(2).all(|p| p[0] < p[1]), "{pairs:x?}");
        assert_eq!(decode_origin_ids(&words), ids);
        ids.retain(|&id| (0..1 << 32).contains(&id));
        assert!(
            encode_origin_ids(&ids)
                .as_chunks::<2>()
                .0
                .iter()
                .all(|w| w[0] == SIGN)
        );
    }
}
