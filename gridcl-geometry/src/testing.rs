//! Test hooks of the degree policy (ADR-0042, item 3). Only with the Cargo feature `testing`,
//! which is off by default and not part of the stable API; gridcl-testing turns it on for every
//! workspace test.
//!
//! The overrides are scoped and thread-local, because gridcl runs one host thread per mesh:
//! thread-rank tests set them inside each rank thread. Precedence: the scoped hook, then
//! `GRIDCL_GEOM_STRATEGY`, then `degree-policy.toml`.

pub use crate::strategy::{Strategy, StrategyFamily, StrategyOverride};

/// Runs `f` with the strategy of `family` overridden by `s`, for launches inside `f` on this
/// thread. A forced strategy that does not apply runs `Loops`; [`record_strategies`] shows what
/// ran.
pub fn with_strategy<R>(family: StrategyFamily, s: StrategyOverride, f: impl FnOnce() -> R) -> R {
    let _ = (family, s, f);
    todo!("gridcl_geometry::testing::with_strategy (docs/plans/M9h.md, T4)")
}

/// Runs `f` and returns, with its result, the (family, strategy) of every launch of a
/// degree-dependent family inside `f` on this thread, in order.
pub fn record_strategies<R>(f: impl FnOnce() -> R) -> (R, Vec<(StrategyFamily, Strategy)>) {
    let _ = f;
    todo!("gridcl_geometry::testing::record_strategies (docs/plans/M9h.md, T4)")
}
