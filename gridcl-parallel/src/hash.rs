//! `owner_hash`: the owner and rendezvous hash of ADR-0026, item 9.

/// The seed of the fold: "gridcl" and the milestone that introduced it.
const SEED: u64 = 0x6772_6964_636C_0006;

/// SplitMix64's finaliser, with wrapping arithmetic.
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The owner and rendezvous hash of `key` (docs/design.md, "Ownership of vertices and
/// entities"; ADR-0026): starting at h = 0x6772_6964_636C_0006, h = mix(h ^ (w as u64)) for
/// each word w, where mix is SplitMix64's finaliser.
///
/// - The owner of a vertex or node is the candidate at position `owner_hash(&[origin id]) mod n`
///   among the n candidates, sorted by rank.
/// - The owner of an entity is the candidate at `owner_hash(sorted origin ids) mod n`.
/// - The rendezvous rank of any key is `owner_hash(key) mod P`.
///
/// The value is the same on every platform, and later milestones use the same function.
///
/// # Example
///
/// ```
/// use gridcl_parallel::owner_hash;
///
/// assert_eq!(owner_hash(&[]), 0x6772_6964_636C_0006);
/// assert_eq!(owner_hash(&[1, 2, 3]), 0x2a25_edf4_f515_fb72);
/// // The order of the words matters: callers sort keys first.
/// assert_ne!(owner_hash(&[1, 2, 3]), owner_hash(&[3, 2, 1]));
/// let rendezvous = owner_hash(&[42]) % 4;
/// assert_eq!(rendezvous, 3);
/// ```
pub fn owner_hash(key: &[i64]) -> u64 {
    key.iter().fold(SEED, |h, &w| mix(h ^ w as u64))
}

#[cfg(test)]
mod tests {
    use super::owner_hash;

    /// Golden values, computed independently of this implementation (Python, 64-bit masks).
    #[test]
    fn golden_values() {
        let cases: [(&[i64], u64); 8] = [
            (&[], 0x6772_6964_636c_0006),
            (&[0], 0xcfaf_ebec_f252_c357),
            (&[1], 0xa33f_5902_208b_da4f),
            (&[-1], 0x5292_7ea5_e764_469e),
            (&[42], 0x1106_70b4_b8ff_0813),
            (&[1, 2, 3], 0x2a25_edf4_f515_fb72),
            (&[3, 2, 1], 0xe580_12de_da18_50da),
            (&[-(1 << 40), 7, 1 << 40], 0x9143_b0cf_547c_77fa),
        ];
        for (key, want) in cases {
            assert_eq!(owner_hash(key), want, "{key:?}");
        }
    }
}
