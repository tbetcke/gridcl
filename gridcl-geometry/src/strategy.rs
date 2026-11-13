//! Strategies of the degree-dependent kernel families and the degree policy that chooses one
//! (docs/design.md, "Geometric factors"; ADR-0042, item 3).
//!
//! The types are `pub` in this private module and reach users only through the hidden
//! [`testing`](crate::testing) module (feature `testing`). The policy itself
//! (`degree-policy.toml`, `GRIDCL_GEOM_STRATEGY`, the scoped hook) is crate-private.
//!
//! Every launcher of a degree-dependent family asks [`choose`] for its strategy: the scoped
//! hook, then `GRIDCL_GEOM_STRATEGY`, then `degree-policy.toml` for the client's device class;
//! a strategy that does not apply to the launch (the cell, the point set, a table beyond the
//! shared-memory limit) falls back to `Loops`.

use std::collections::HashMap;
use std::sync::OnceLock;

use cubecl::prelude::Client;
use gridcl_device::{DeviceClass, GridError, Result};
use gridcl_dims::CellType;

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

    /// The strategy of `name` (see [`name`](Strategy::name)).
    fn named(name: &str) -> Option<Strategy> {
        Strategy::ALL.into_iter().find(|s| s.name() == name)
    }

    /// Whether the strategy applies to a launch of `family` on a block of `cell`, where
    /// `tensor` says whether the launch's points are a tensor point set (a point set that
    /// records its factors; the pull-back, at arbitrary points, passes `true`). Sum
    /// factorisation needs a tensor cell and tensor points; neither it nor `cooperative`
    /// applies to the E-vector, a gather without sums (ADR-0042, item 3). `shared` needs a
    /// table that fits the shared memory, which [`choose`] checks with the launch's table.
    fn applies(self, family: StrategyFamily, cell: CellType, tensor: bool) -> bool {
        match self {
            Strategy::SumFactorised => {
                family != StrategyFamily::EVector
                    && tensor
                    && matches!(
                        cell,
                        CellType::Quadrilateral | CellType::Hexahedron | CellType::Prism
                    )
            }
            Strategy::Cooperative => family != StrategyFamily::EVector,
            _ => true,
        }
    }

    /// `None` when the launchers of `family` run the strategy, else the task that implements it.
    /// `direct` and `loops` run for every family (T6a, T6b); the later strategies arrive family
    /// by family. Sum factorisation runs for the factor families wherever the point set records
    /// its factors (quadrilateral and hexahedron cell rules from T7a, facet rules from T7b,
    /// prisms from T7c), and for the pull-back on every tensor cell (T8); `shared` for every
    /// family with a table (T9); `cooperative` for the factor families (T10a).
    fn not_yet(self, family: StrategyFamily) -> Option<&'static str> {
        match (self, family) {
            (Strategy::Cooperative, StrategyFamily::PullBack) => {
                Some("geometry strategy cooperative (docs/plans/M9h.md, T10b)")
            }
            _ => None,
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

    /// The cell types the family runs on: those of codimension-1 meshes for `Normals`, every
    /// cell type otherwise.
    fn cells(self) -> &'static [CellType] {
        match self {
            StrategyFamily::Normals => &CELLS[..3],
            _ => &CELLS,
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

impl StrategyOverride {
    /// `policy` or a strategy's name.
    fn named(name: &str) -> Option<StrategyOverride> {
        match name {
            "policy" => Some(StrategyOverride::Policy),
            _ => Strategy::named(name).map(StrategyOverride::Fixed),
        }
    }
}

/// The cell types of the policy, codimension-1 cells first.
const CELLS: [CellType; 7] = [
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

/// The device classes of the policy, by their table names.
const CLASSES: [(DeviceClass, &str); 4] = [
    (DeviceClass::CpuSimd128, "CpuSimd128"),
    (DeviceClass::CpuSimd256, "CpuSimd256"),
    (DeviceClass::CpuSimd512, "CpuSimd512"),
    (DeviceClass::Gpu, "Gpu"),
];

/// The scalar sizes of the policy in bytes.
const SCALAR_SIZES: [usize; 2] = [4, 8];

/// The environment variable that overrides the policy (ADR-0042, item 3).
pub(crate) const STRATEGY_VAR: &str = "GRIDCL_GEOM_STRATEGY";

/// The degree policy: per (family, cell type, device class, scalar size), the first degree and
/// strategy of each degree range, ascending from 1; the last range is open-ended.
#[derive(Debug)]
struct DegreePolicy {
    ranges: HashMap<PolicyKey, Vec<(u32, Strategy)>>,
}

/// (family, cell type, device class, scalar size in bytes).
type PolicyKey = (StrategyFamily, CellType, DeviceClass, usize);

/// The policy file, compiled in.
const POLICY_TOML: &str = include_str!("../degree-policy.toml");

impl DegreePolicy {
    /// The compiled-in policy, parsed once.
    fn get() -> &'static DegreePolicy {
        static POLICY: OnceLock<DegreePolicy> = OnceLock::new();
        POLICY.get_or_init(|| {
            DegreePolicy::parse(POLICY_TOML)
                .unwrap_or_else(|e| panic!("gridcl-geometry/degree-policy.toml: {e}"))
        })
    }

    /// The policy in the TOML text `text`: a table per class and family with an entry for every
    /// cell type of the family and every scalar size, and nothing else.
    fn parse(text: &str) -> std::result::Result<DegreePolicy, String> {
        let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
        let mut ranges = HashMap::new();
        for key in table.keys() {
            if !CLASSES.iter().any(|(_, c)| c == key) {
                return Err(format!("unknown device class [{key}]"));
            }
        }
        for (class, c) in CLASSES {
            let classes = table
                .get(c)
                .and_then(|v| v.as_table())
                .ok_or_else(|| format!("no table [{c}]"))?;
            for key in classes.keys() {
                if !StrategyFamily::ALL.iter().any(|f| f.name() == key) {
                    return Err(format!("unknown family [{c}.{key}]"));
                }
            }
            for family in StrategyFamily::ALL {
                let at = format!("[{c}.{}]", family.name());
                let cells = classes
                    .get(family.name())
                    .and_then(|v| v.as_table())
                    .ok_or_else(|| format!("no table {at}"))?;
                for key in cells.keys() {
                    if !family
                        .cells()
                        .iter()
                        .any(|cell| format!("{cell:?}") == *key)
                    {
                        return Err(format!("{at}: unknown cell type {key}"));
                    }
                }
                for &cell in family.cells() {
                    let name = format!("{cell:?}");
                    let sizes = cells
                        .get(&name)
                        .and_then(|v| v.as_table())
                        .ok_or_else(|| format!("{at} has no table {name}"))?;
                    for key in sizes.keys() {
                        if !SCALAR_SIZES.iter().any(|s| s.to_string() == *key) {
                            return Err(format!("{at} {name}: unknown scalar size {key}"));
                        }
                    }
                    for size in SCALAR_SIZES {
                        let at = format!("{at} {name} {size}");
                        let list = sizes
                            .get(&size.to_string())
                            .ok_or_else(|| format!("{at}: missing"))?;
                        let list = parse_ranges(list).map_err(|e| format!("{at}: {e}"))?;
                        ranges.insert((family, cell, class, size), list);
                    }
                }
            }
        }
        Ok(DegreePolicy { ranges })
    }

    /// The strategy of `degree`: that of the last range starting at or below it.
    fn strategy(
        &self,
        family: StrategyFamily,
        cell: CellType,
        class: DeviceClass,
        elem_bytes: usize,
        degree: u32,
    ) -> Option<Strategy> {
        self.ranges
            .get(&(family, cell, class, elem_bytes))?
            .iter()
            .rev()
            .find(|(first, _)| *first <= degree)
            .map(|(_, s)| *s)
    }
}

/// `[[first degree, strategy], ...]`, first degrees ascending from 1.
fn parse_ranges(value: &toml::Value) -> std::result::Result<Vec<(u32, Strategy)>, String> {
    let list = value
        .as_array()
        .ok_or("expected an array of [first degree, strategy] pairs")?;
    let mut ranges: Vec<(u32, Strategy)> = Vec::new();
    for pair in list {
        let (first, name) = match pair.as_array().map(Vec::as_slice) {
            Some([first, name]) => (first.as_integer(), name.as_str()),
            _ => (None, None),
        };
        let (Some(first), Some(name)) = (first, name) else {
            return Err(format!("{pair} is not a [first degree, strategy] pair"));
        };
        let first = u32::try_from(first).map_err(|_| format!("degree {first}"))?;
        let s = Strategy::named(name).ok_or_else(|| format!("unknown strategy {name}"))?;
        match ranges.last() {
            None if first != 1 => return Err(format!("the first range starts at {first}, not 1")),
            Some(&(previous, _)) if first <= previous => {
                return Err(format!("degree {first} after {previous}: not ascending"));
            }
            _ => ranges.push((first, s)),
        }
    }
    if ranges.is_empty() {
        return Err("no degree range".to_owned());
    }
    Ok(ranges)
}

/// The overrides of a `GRIDCL_GEOM_STRATEGY` value: `policy` or a strategy for every family,
/// or a comma list of `<family>=<policy or strategy>` with the family in any case.
fn parse_env(
    value: &str,
) -> std::result::Result<HashMap<StrategyFamily, StrategyOverride>, String> {
    let invalid = || {
        format!(
            "{STRATEGY_VAR}={value}: expected policy, direct, loops, sumfact, shared or \
             cooperative, or a comma list of <family>=<strategy> with the families EVector, \
             Points, Jacobian, Normals, FacetNormals and PullBack"
        )
    };
    if let Some(s) = StrategyOverride::named(value) {
        return Ok(StrategyFamily::ALL.into_iter().map(|f| (f, s)).collect());
    }
    let mut out = HashMap::new();
    for item in value.split(',') {
        let (family, s) = item.split_once('=').ok_or_else(invalid)?;
        let family = StrategyFamily::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(family.trim()))
            .ok_or_else(invalid)?;
        let s = StrategyOverride::named(s.trim()).ok_or_else(invalid)?;
        if out.insert(family, s).is_some() {
            return Err(format!("{}: {} given twice", invalid(), family.name()));
        }
    }
    Ok(out)
}

/// The override of `family` in `GRIDCL_GEOM_STRATEGY`, read once per process; `None` when the
/// variable is unset or empty or does not name the family.
///
/// # Panics
///
/// If the value is invalid.
fn env_override(family: StrategyFamily) -> Option<StrategyOverride> {
    static ENV: OnceLock<HashMap<StrategyFamily, StrategyOverride>> = OnceLock::new();
    ENV.get_or_init(|| match std::env::var(STRATEGY_VAR) {
        Ok(v) if !v.is_empty() => parse_env(&v).unwrap_or_else(|e| panic!("{e}")),
        _ => HashMap::new(),
    })
    .get(&family)
    .copied()
}

/// The strategy a launch of `family` on a block of `cell` at `degree` with `elem_bytes`-byte
/// scalars runs on `client`'s device: the scoped hook, then `GRIDCL_GEOM_STRATEGY`, then
/// `degree-policy.toml`; `Loops` where that strategy does not apply (`tensor`: whether the
/// launch's points are a tensor point set, see `Strategy::applies`). `table` gives the
/// entries of the table the `shared` variant of the launch stages (`None` where it has
/// none, the E-vector and the bench evaluator), and is called only when `shared` is wanted;
/// a table beyond the device's shared memory per cube makes `shared` fall back to `Loops` too
/// (ADR-0042, item 3). Recorded for `testing::record_strategies`.
///
/// # Errors
///
/// `NotYetImplemented` for a strategy the launchers do not run yet; the errors of `table`.
///
/// # Panics
///
/// If `GRIDCL_GEOM_STRATEGY` is invalid, or the policy has no entry for the family, cell type
/// and scalar size (it has one for every family, its cell types and both float sizes).
pub(crate) fn choose(
    client: &Client,
    family: StrategyFamily,
    cell: CellType,
    degree: u32,
    elem_bytes: usize,
    tensor: bool,
    table: impl FnOnce() -> Result<Option<usize>>,
) -> Result<Strategy> {
    let wanted = match hook::get(family)
        .or_else(|| env_override(family))
        .unwrap_or(StrategyOverride::Policy)
    {
        StrategyOverride::Policy => {
            let class = DeviceClass::for_client(client);
            DegreePolicy::get()
                .strategy(family, cell, class, elem_bytes, degree)
                .unwrap_or_else(|| {
                    panic!(
                        "degree-policy.toml has no entry {class:?}.{}.{cell:?} for {elem_bytes}-byte \
                         scalars",
                        family.name()
                    )
                })
        }
        StrategyOverride::Fixed(s) => s,
    };
    let applies = wanted.applies(family, cell, tensor)
        && (wanted != Strategy::Shared
            || table()?.is_some_and(|len| crate::shared::fits(client, len, elem_bytes)));
    let s = if applies { wanted } else { Strategy::Loops };
    if let Some(task) = s.not_yet(family) {
        return Err(GridError::NotYetImplemented(task));
    }
    hook::record(family, s);
    Ok(s)
}

/// Whether a launch of `family` on `cell` at points that are (`tensor`) or are not a tensor
/// point set can be sum-factorised: the variant `shared` stages for it (`crate::shared`).
pub(crate) fn sum_factorises(family: StrategyFamily, cell: CellType, tensor: bool) -> bool {
    Strategy::SumFactorised.applies(family, cell, tensor)
}

/// The scoped hook and the recorder of `testing` (thread-local; absent without the feature).
#[cfg(feature = "testing")]
pub(crate) mod hook {
    use std::cell::RefCell;

    use super::{Strategy, StrategyFamily, StrategyOverride};

    thread_local! {
        /// The overrides in scope, innermost last.
        static HOOKS: RefCell<Vec<(StrategyFamily, StrategyOverride)>> =
            const { RefCell::new(Vec::new()) };
        /// The launches of the innermost `record_strategies`.
        static RECORD: RefCell<Option<Vec<(StrategyFamily, Strategy)>>> =
            const { RefCell::new(None) };
    }

    /// The innermost override of `family`.
    pub(crate) fn get(family: StrategyFamily) -> Option<StrategyOverride> {
        HOOKS.with(|h| {
            h.borrow()
                .iter()
                .rev()
                .find(|(f, _)| *f == family)
                .map(|(_, s)| *s)
        })
    }

    pub(crate) fn record(family: StrategyFamily, s: Strategy) {
        RECORD.with(|r| {
            if let Some(r) = r.borrow_mut().as_mut() {
                r.push((family, s));
            }
        });
    }

    /// Runs `f` with `family` overridden by `s`, also when `f` panics.
    pub(crate) fn with<R>(family: StrategyFamily, s: StrategyOverride, f: impl FnOnce() -> R) -> R {
        struct Pop;
        impl Drop for Pop {
            fn drop(&mut self) {
                HOOKS.with(|h| h.borrow_mut().pop());
            }
        }
        HOOKS.with(|h| h.borrow_mut().push((family, s)));
        let _pop = Pop;
        f()
    }

    /// Runs `f` and returns the launches recorded inside it; an enclosing call sees them too.
    pub(crate) fn recording<R>(f: impl FnOnce() -> R) -> (R, Vec<(StrategyFamily, Strategy)>) {
        struct Restore {
            previous: Option<Vec<(StrategyFamily, Strategy)>>,
        }
        impl Drop for Restore {
            fn drop(&mut self) {
                let mut previous = self.previous.take();
                RECORD.with(|r| {
                    let mut r = r.borrow_mut();
                    // On a panic inside `f` the inner launches still reach the outer scope.
                    if let (Some(outer), Some(inner)) = (previous.as_mut(), r.take()) {
                        outer.extend(inner);
                    }
                    *r = previous;
                });
            }
        }
        let previous = RECORD.with(|r| r.borrow_mut().replace(Vec::new()));
        let restore = Restore { previous };
        let result = f();
        let ran = RECORD.with(|r| r.borrow().clone().unwrap_or_default());
        drop(restore);
        (result, ran)
    }
}

#[cfg(not(feature = "testing"))]
mod hook {
    use super::{Strategy, StrategyFamily, StrategyOverride};

    pub(crate) fn get(_: StrategyFamily) -> Option<StrategyOverride> {
        None
    }

    pub(crate) fn record(_: StrategyFamily, _: Strategy) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_policy_covers_every_family_and_class() {
        let policy = DegreePolicy::parse(POLICY_TOML).unwrap();
        for (class, _) in CLASSES {
            for family in StrategyFamily::ALL {
                for &cell in family.cells() {
                    for size in SCALAR_SIZES {
                        // p = 9–10 resolve through the open-ended last range.
                        for degree in 1..=10 {
                            let s = policy.strategy(family, cell, class, size, degree);
                            assert!(
                                s.is_some(),
                                "{class:?} {family:?} {cell:?} {size} p={degree}"
                            );
                        }
                    }
                }
            }
        }
        // T4: M9's kernels everywhere.
        assert!(
            policy
                .ranges
                .values()
                .all(|r| r.as_slice() == [(1, Strategy::Direct)])
        );
    }

    #[test]
    fn degree_ranges_resolve_by_their_first_degree() {
        let text = POLICY_TOML.replace(
            "[Gpu.Points]\nInterval = { 4 = [[1, \"direct\"]],",
            "[Gpu.Points]\nInterval = { 4 = [[1, \"direct\"], [3, \"loops\"], [6, \"sumfact\"]],",
        );
        let policy = DegreePolicy::parse(&text).unwrap();
        let at = |p| {
            policy
                .strategy(
                    StrategyFamily::Points,
                    CellType::Interval,
                    DeviceClass::Gpu,
                    4,
                    p,
                )
                .unwrap()
        };
        assert_eq!(
            (1..=10).map(at).collect::<Vec<_>>(),
            [
                Strategy::Direct,
                Strategy::Direct,
                Strategy::Loops,
                Strategy::Loops,
                Strategy::Loops,
                Strategy::SumFactorised,
                Strategy::SumFactorised,
                Strategy::SumFactorised,
                Strategy::SumFactorised,
                Strategy::SumFactorised,
            ]
        );
    }

    #[test]
    fn invalid_degree_policies_are_rejected() {
        let entry = "[Gpu.Points]\nInterval = { 4 = [[1, \"direct\"]],";
        let with = |replacement: &str| POLICY_TOML.replace(entry, replacement);
        for (text, want) in [
            ("[Gpu".to_owned(), "TOML parse error"),
            (
                with("[Gpu.Points]\nInterval = { 4 = [[2, \"direct\"]],"),
                "starts at 2",
            ),
            (
                with("[Gpu.Points]\nInterval = { 4 = [[1, \"direct\"], [1, \"loops\"]],"),
                "not ascending",
            ),
            (
                with("[Gpu.Points]\nInterval = { 4 = [[1, \"fast\"]],"),
                "unknown strategy fast",
            ),
            (
                with("[Gpu.Points]\nInterval = { 4 = [],"),
                "no degree range",
            ),
            (
                with("[Gpu.Points]\nInterval = { 4 = [[1]],"),
                "not a [first degree, strategy]",
            ),
            (
                with("[Gpu.Points]\nInterval = { 4 = \"direct\","),
                "expected an array",
            ),
            (
                with("[Gpu.Points]\nInterval = { 2 = [[1, \"direct\"]],"),
                "unknown scalar size 2",
            ),
            (with("[Gpu.Points]\nInterval = {"), "Interval 4: missing"),
            (
                POLICY_TOML.replace("[Gpu.Normals]\nInterval", "[Gpu.Normals]\nTetrahedron"),
                "unknown cell type Tetrahedron",
            ),
            (
                POLICY_TOML.replace("[Gpu.Normals]\nInterval", "[Gpu.Normals]\n# Interval"),
                "no table Interval",
            ),
            (
                POLICY_TOML.replace("[Gpu.Normals]", "[Gpu.Normal]"),
                "unknown family",
            ),
            (
                POLICY_TOML.replace("[Gpu.", "[Tpu."),
                "unknown device class [Tpu]",
            ),
        ] {
            let e = DegreePolicy::parse(&text).unwrap_err();
            assert!(e.contains(want), "{want}: {e}");
        }
    }

    #[test]
    fn environment_values_parse() {
        let all = |s| StrategyFamily::ALL.into_iter().map(|f| (f, s)).collect();
        assert_eq!(parse_env("policy").unwrap(), all(StrategyOverride::Policy));
        for s in Strategy::ALL {
            assert_eq!(
                parse_env(s.name()).unwrap(),
                all(StrategyOverride::Fixed(s))
            );
        }
        assert_eq!(
            parse_env("jacobian=loops, PullBack = policy").unwrap(),
            HashMap::from([
                (
                    StrategyFamily::Jacobian,
                    StrategyOverride::Fixed(Strategy::Loops)
                ),
                (StrategyFamily::PullBack, StrategyOverride::Policy),
            ])
        );
    }

    #[test]
    fn invalid_environment_values_are_rejected() {
        for value in [
            "fast",
            "Direct",
            "loops,",
            "jacobian",
            "jacobian=fast",
            "factors=loops",
            "points=loops,points=direct",
        ] {
            let e = parse_env(value).unwrap_err();
            assert!(
                e.starts_with(&format!("{STRATEGY_VAR}={value}: expected")),
                "{e}"
            );
        }
    }

    #[test]
    fn sum_factorisation_and_cooperation_apply_where_they_sum() {
        let tensor_cells = [
            CellType::Quadrilateral,
            CellType::Hexahedron,
            CellType::Prism,
        ];
        for family in StrategyFamily::ALL {
            for cell in CELLS {
                for tensor in [false, true] {
                    assert_eq!(
                        Strategy::SumFactorised.applies(family, cell, tensor),
                        family != StrategyFamily::EVector && tensor && tensor_cells.contains(&cell),
                        "{family:?} {cell:?} {tensor}"
                    );
                    assert_eq!(
                        Strategy::Cooperative.applies(family, cell, tensor),
                        family != StrategyFamily::EVector,
                        "{family:?} {cell:?} {tensor}"
                    );
                    for s in [Strategy::Direct, Strategy::Loops, Strategy::Shared] {
                        assert!(s.applies(family, cell, tensor));
                    }
                }
            }
        }
    }
}
