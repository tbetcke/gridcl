//! Strategies of the degree-dependent kernel families and the degree policy that chooses one
//! (docs/design.md, "Geometric factors"; ADR-0042, item 3).
//!
//! The types are `pub` in this private module and reach users only through the hidden
//! [`testing`](crate::testing) module (feature `testing`). The policy itself
//! (`degree-policy.toml`, `GRIDCL_GEOM_STRATEGY`, the scoped hook) is crate-private.

// The launchers dispatch on these from T4 (docs/plans/M9h.md); until then only the hidden
// testing module names them.
#![cfg_attr(not(feature = "testing"), allow(dead_code))]

/// How a kernel family evaluates a block, from the simplest to the most involved.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Strategy {
    /// M9's kernels: the degree at comptime, node loops fully unrolled.
    Direct,
    /// Runtime loops over nodes (and basis functions): one kernel for every degree.
    Loops,
    /// Sum factorisation on tensor cells (quadrilaterals, hexahedra, prisms) and tensor rules.
    SumFactorised,
    /// Tabulations or coefficients staged in shared memory.
    Shared,
    /// A group of units per cell (`LaunchPlan::per_group`).
    Cooperative,
}

impl Strategy {
    /// Every strategy, simplest first (the tie order of the decision rule; ADR-0042, item 12).
    pub const ALL: [Strategy; 5] = [
        Strategy::Direct,
        Strategy::Loops,
        Strategy::SumFactorised,
        Strategy::Shared,
        Strategy::Cooperative,
    ];

    /// The name in `GRIDCL_GEOM_STRATEGY` and `degree-policy.toml`: `direct`, `loops`,
    /// `sumfact`, `shared` or `cooperative`.
    pub fn name(self) -> &'static str {
        match self {
            Strategy::Direct => "direct",
            Strategy::Loops => "loops",
            Strategy::SumFactorised => "sumfact",
            Strategy::Shared => "shared",
            Strategy::Cooperative => "cooperative",
        }
    }
}

/// The kernel families the degree policy chooses a strategy for. `Jacobian` covers J, det J and
/// the pseudo-inverse; `PullBack` covers the device basis evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StrategyFamily {
    EVector,
    Points,
    Jacobian,
    Normals,
    FacetNormals,
    PullBack,
}

impl StrategyFamily {
    /// Every family, in the order of `degree-policy.toml`.
    pub const ALL: [StrategyFamily; 6] = [
        StrategyFamily::EVector,
        StrategyFamily::Points,
        StrategyFamily::Jacobian,
        StrategyFamily::Normals,
        StrategyFamily::FacetNormals,
        StrategyFamily::PullBack,
    ];

    /// The key in `degree-policy.toml` (`EVector`, `Points`, ...); `GRIDCL_GEOM_STRATEGY`
    /// accepts it in any case.
    pub fn name(self) -> &'static str {
        match self {
            StrategyFamily::EVector => "EVector",
            StrategyFamily::Points => "Points",
            StrategyFamily::Jacobian => "Jacobian",
            StrategyFamily::Normals => "Normals",
            StrategyFamily::FacetNormals => "FacetNormals",
            StrategyFamily::PullBack => "PullBack",
        }
    }
}

/// The value of `GRIDCL_GEOM_STRATEGY` for one family, and of the scoped hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StrategyOverride {
    /// The degree policy of `degree-policy.toml`.
    Policy,
    /// This strategy, or `Loops` where it does not apply (ADR-0042, item 3).
    Fixed(Strategy),
}
