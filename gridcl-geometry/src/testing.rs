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
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl_geometry::testing::{
///     Strategy, StrategyFamily, StrategyOverride, record_strategies, with_strategy,
/// };
///
/// let client = gridcl_testing::test_client();
/// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 2).build::<f32, D2, D2>(&client)?;
/// let rule = QuadRule::for_grid(&mesh, 2)?;
/// let fixed = StrategyOverride::Fixed(Strategy::Direct);
/// let (x, ran) = record_strategies(|| with_strategy(StrategyFamily::Points, fixed, || {
///     mesh.physical_points(&rule)
/// }));
/// x?;
/// assert!(ran.contains(&(StrategyFamily::Points, Strategy::Direct)));
/// # Ok::<(), GridError>(())
/// ```
pub fn with_strategy<R>(family: StrategyFamily, s: StrategyOverride, f: impl FnOnce() -> R) -> R {
    crate::strategy::hook::with(family, s, f)
}

/// Runs `f` and returns, with its result, the (family, strategy) of every launch of a
/// degree-dependent family inside `f` on this thread, in order. An enclosing
/// `record_strategies` sees them too.
pub fn record_strategies<R>(f: impl FnOnce() -> R) -> (R, Vec<(StrategyFamily, Strategy)>) {
    crate::strategy::hook::recording(f)
}
