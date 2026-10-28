//! `owner_hash`: the owner and rendezvous hash of ADR-0026, item 9.

/// The owner and rendezvous hash of `key`: a SplitMix64 fold over its words with a fixed seed
/// (docs/design.md, "Ownership of vertices and entities"; ADR-0026).
pub fn owner_hash(key: &[i64]) -> u64 {
    let _ = key;
    todo!("M6 T1: owner_hash")
}
