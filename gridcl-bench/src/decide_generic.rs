//! The generic decision rule of "Benchmarks" (ADR-0058 item 4), which turns the M10 records
//! into the four sets of `vector-policy.toml` and, with M9h's records between p = 1, 4 and 8
//! (ADR-0060 item 3), of `degree-policy.toml`.

use crate::record::{HighOrderData, M10Data};

/// The generic vector policy: `[Cpu]` and `[Gpu]`, each for 4- and 8-byte scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorDecision {}

impl VectorDecision {
    /// The text of `gridcl-device/vector-policy.toml`.
    pub fn to_toml(&self) -> String {
        todo!("M10 T2")
    }
}

/// The generic vector policy from the M10 candidate records.
pub fn decide_vector(_data: &M10Data) -> VectorDecision {
    todo!("M10 T2")
}

/// The generic degree policy: `[Cpu.<family>]` and `[Gpu.<family>]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeDecision {}

impl DegreeDecision {
    /// The text of `gridcl-geometry/degree-policy.toml`.
    pub fn to_toml(&self) -> String {
        todo!("M10 T2")
    }
}

/// The generic degree policy from M10's records at p = 1, 4 and 8 and M9h's in between.
pub fn decide_degree(_m9h: &HighOrderData, _m10: &M10Data) -> DegreeDecision {
    todo!("M10 T2")
}
