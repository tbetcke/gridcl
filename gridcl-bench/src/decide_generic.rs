//! The generic decision rule of "Benchmarks" (ADR-0058 item 4), which turns the M10 records
//! into the four sets of `vector-policy.toml` and, with M9h's records between p = 1, 4 and 8
//! (ADR-0060 item 3), of `degree-policy.toml`.
//!
//! For one policy entry, class and element size, the members are the class's platforms
//! ([`m10_platforms`]) and, for a shared 4-byte entry, each 4-byte type of the entry on each
//! of them (f32 and u32 are separate members). Only candidates that every member measured at
//! every rule size take part. With t(c, m, s) a member's median time:
//!
//! - r(c, m, s) = t(c, m, s) / min over c' of t(c', m, s);
//! - **admission:** a candidate other than the baseline is admitted if, at each rule size, the
//!   geometric mean over the members of t(c, m, s) / t(baseline, m, s) is at most 1/1.10;
//! - **bound:** of the baseline and the admitted candidates, those with r ≤ 1.25 for every
//!   member and size are kept;
//! - **choice:** of those kept, the simplest within 3% of the smallest geometric mean of r over
//!   members and sizes (simplicity: [`ConfigKey`] order for vector configurations; `direct` <
//!   `loops` < `sumfact` < `shared` < `cooperative` for strategies);
//! - **fallback:** if none is kept, the admitted candidate with the smallest worst-case r, which
//!   the report marks with that worst case.
//!
//! The comparisons allow a relative 10^-12 for rounding, as the acceptance tests that recompute
//! the rule do.

use std::collections::{BTreeMap, BTreeSet};

use gridcl_device::{CpuUnits, DeviceClass, KernelKind};

use crate::candidates::{ConfigKey, VectorConfig, candidate_kinds, vector_baseline};
use crate::decide::{BLOCK_UNIT, ELEM_BYTES, units_name};
use crate::decide_degree::{
    POLICY_FAMILIES, SCALARS, Times, cells_of, max_degree, members, sweep_devices,
};
use crate::protocol::CANDIDATE_SIZES;
use crate::record::{
    CubeDimMode, DegreeRecord, HighOrderData, KINDS, LegacyClass, M10Data, STRATEGIES, Scalar,
    SweepRecord,
};

/// A candidate other than the baseline is admitted if the baseline takes at least this factor
/// longer, by the geometric mean over the members, at each rule size.
pub const ADMIT: f64 = 1.10;
/// The bound on every member's r at every rule size.
pub const BOUND: f64 = 1.25;
/// The simplest candidate within this factor of the smallest geometric mean of r is chosen.
pub const WITHIN: f64 = 1.03;
/// Relative slack of every comparison, for rounding.
const SLACK: f64 = 1e-12;

/// A benchmark platform of M10 (ADR-0058 item 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Platform {
    /// The target: `macmini`, `locust`, `kathleen`, or a fallback.
    pub target: &'static str,
    /// The backend: `cpu`, `metal` or `cuda`.
    pub backend: &'static str,
    pub class: DeviceClass,
}

/// The benchmark platforms of M10 (ADR-0058 item 3): the local Mac's CPU runtime and Metal,
/// Locust's Grace CPU and H100, one Kathleen socket. A Myriad D node stands in for Kathleen,
/// and Myriad's A100 or V100 for the H100, only where `present(target, backend)` says the
/// primary target has no record (ADR-0014, ADR-0016); the A4000 workstation does not vote
/// (ADR-0037).
pub fn m10_platforms(present: impl Fn(&str, &str) -> bool) -> Vec<Platform> {
    let pick = |primary: &'static str, backend: &str, fallbacks: &[&'static str]| {
        if present(primary, backend) {
            return primary;
        }
        fallbacks
            .iter()
            .copied()
            .find(|t| present(t, backend))
            .unwrap_or(primary)
    };
    let p = |target, backend, class| Platform {
        target,
        backend,
        class,
    };
    vec![
        p("macmini", "cpu", DeviceClass::Cpu),
        p("macmini", "metal", DeviceClass::Gpu),
        p("locust", "cpu", DeviceClass::Cpu),
        p(
            pick("locust", "cuda", &["myriad-a100", "myriad-v100"]),
            "cuda",
            DeviceClass::Gpu,
        ),
        p(
            pick("kathleen", "cpu", &["myriad-cpu"]),
            "cpu",
            DeviceClass::Cpu,
        ),
    ]
}

/// The scalar types a backend measures: Metal has no f64.
pub fn backend_scalars(backend: &str, scalars: &[Scalar]) -> Vec<Scalar> {
    scalars
        .iter()
        .copied()
        .filter(|&s| !(backend == "metal" && s == Scalar::F64))
        .collect()
}

/// The geometric mean, as `exp(Σ ln v / n)` in the order of `v`.
pub fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// The outcome of the generic rule on one entry.
#[derive(Clone, Debug, PartialEq)]
pub struct RuleOutcome<K> {
    pub choice: K,
    /// No candidate met the bound: the choice has the smallest worst case.
    pub fallback: bool,
    /// The baseline and the admitted candidates, in simplicity order.
    pub admitted: Vec<K>,
    /// The largest r of the choice over the members and sizes.
    pub worst: f64,
    /// The geometric mean of the choice's r over the members and sizes.
    pub geomean: f64,
    /// r of every candidate, `[member][size]`.
    pub r: BTreeMap<K, Vec<Vec<f64>>>,
}

/// The generic rule (ADR-0058 item 4) over `candidates` in simplicity order, the baseline among
/// them, with `t(c, m, s)` the time of candidate c on member m at rule size s, for `members`
/// members and `sizes` sizes; every candidate has a time on every member and size.
///
/// # Panics
///
/// If `candidates` does not contain `baseline`, or there are no members or sizes.
pub fn generic_rule<K: Copy + Ord>(
    baseline: K,
    candidates: &[K],
    members: usize,
    sizes: usize,
    t: impl Fn(K, usize, usize) -> f64,
) -> RuleOutcome<K> {
    assert!(
        candidates.contains(&baseline),
        "the baseline is a candidate"
    );
    assert!(members > 0 && sizes > 0, "the rule needs members and sizes");
    let fastest: Vec<Vec<f64>> = (0..members)
        .map(|m| {
            (0..sizes)
                .map(|s| {
                    candidates
                        .iter()
                        .map(|&c| t(c, m, s))
                        .fold(f64::INFINITY, f64::min)
                })
                .collect()
        })
        .collect();
    let r: BTreeMap<K, Vec<Vec<f64>>> = candidates
        .iter()
        .map(|&c| {
            let rc = (0..members)
                .map(|m| (0..sizes).map(|s| t(c, m, s) / fastest[m][s]).collect())
                .collect();
            (c, rc)
        })
        .collect();
    let flat = |c: &K| -> Vec<f64> { r[c].iter().flatten().copied().collect() };
    let worst = |c: &K| flat(c).into_iter().fold(0.0, f64::max);
    let admitted: Vec<K> = candidates
        .iter()
        .copied()
        .filter(|&c| {
            c == baseline
                || (0..sizes).all(|s| {
                    let ratios: Vec<f64> = (0..members)
                        .map(|m| t(c, m, s) / t(baseline, m, s))
                        .collect();
                    geomean(&ratios) <= 1.0 / ADMIT * (1.0 + SLACK)
                })
        })
        .collect();
    let kept: Vec<K> = admitted
        .iter()
        .copied()
        .filter(|c| worst(c) <= BOUND * (1.0 + SLACK))
        .collect();
    let (choice, fallback) = if kept.is_empty() {
        let mut best = admitted[0];
        for &c in &admitted[1..] {
            if worst(&c) < worst(&best) {
                best = c;
            }
        }
        (best, true)
    } else {
        let g = |c: &K| geomean(&flat(c));
        let best = kept.iter().map(g).fold(f64::INFINITY, f64::min);
        let c = kept
            .iter()
            .copied()
            .find(|c| g(c) <= WITHIN * best * (1.0 + SLACK))
            .expect("the best is within 3% of itself");
        (c, false)
    };
    RuleOutcome {
        worst: worst(&choice),
        geomean: geomean(&flat(&choice)),
        choice,
        fallback,
        admitted,
        r,
    }
}

/// How the generic rule reached an entry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GenericFit {
    /// The simplest kept candidate within 3% of the smallest geometric mean of r.
    Bound,
    /// No candidate met the bound: the smallest worst case, `worst`.
    Fallback { worst: f64 },
    /// No member measured the entry, or not every candidate the rule needs: the baseline at
    /// vector size 1 on all hardware threads.
    Default,
}

/// One platform's view of a vector entry.
#[derive(Clone, Debug, PartialEq)]
pub struct PlatformRatio {
    pub platform: Platform,
    /// The candidate with the smallest geometric mean of r over the platform's members and
    /// sizes (the simplest on ties).
    pub best: VectorConfig,
    /// The largest r of the generic choice over the platform's members and sizes.
    pub r: f64,
}

/// One entry of the generic vector policy.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorEntry {
    pub class: DeviceClass,
    pub kind: KernelKind,
    /// Element size in bytes, 4 or 8.
    pub bytes: usize,
    pub baseline: VectorConfig,
    pub choice: VectorConfig,
    pub fit: GenericFit,
    /// Per platform of the class that measured the entry, in [`m10_platforms`] order.
    pub platforms: Vec<PlatformRatio>,
    /// Why, in one line for the report.
    pub reason: String,
}

/// The generic vector policy: `[Cpu]` and `[Gpu]`, each for 4- and 8-byte scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorDecision {
    /// Every (class, kind, element size), Cpu first, in [`KINDS`] order.
    pub entries: Vec<VectorEntry>,
}

impl VectorDecision {
    /// The entry of (`class`, `kind`, `bytes`).
    pub fn entry(
        &self,
        class: DeviceClass,
        kind: KernelKind,
        bytes: usize,
    ) -> Option<&VectorEntry> {
        self.entries
            .iter()
            .find(|e| e.class == class && e.kind == kind && e.bytes == bytes)
    }

    /// The QuadField tile of `class` and `bytes`: the GeomFactors choice's.
    pub fn quad_tile(&self, class: DeviceClass, bytes: usize) -> usize {
        self.entry(class, KernelKind::GeomFactors, bytes)
            .and_then(|e| e.choice.tile)
            .unwrap_or(crate::decide::DEFAULT_TILE)
    }

    /// The text of `gridcl-device/vector-policy.toml`.
    pub fn to_toml(&self) -> String {
        let mut out = String::from(VECTOR_TOML_HEADER);
        let row = |value: &dyn Fn(usize) -> String| {
            let entries: Vec<String> = ELEM_BYTES
                .iter()
                .map(|&bytes| format!("{bytes} = {}", value(bytes)))
                .collect();
            format!("{{ {} }}", entries.join(", "))
        };
        for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
            let c = format!("{class:?}");
            let choice = |kind, bytes| {
                self.entry(class, kind, bytes)
                    .map(|e| e.choice)
                    .unwrap_or_else(|| default_config(class, kind, bytes))
            };
            out.push_str(&format!(
                "\n[{c}]\nquad_tile = {}\n[{c}.vector_size]\n",
                row(&|bytes| self.quad_tile(class, bytes).to_string())
            ));
            for kind in KINDS {
                let r = row(&|bytes| choice(kind, bytes).vector_size.to_string());
                out.push_str(&format!("{kind:?} = {r}\n"));
            }
            if class == DeviceClass::Cpu {
                out.push_str(&format!("[{c}.cpu_block]\n"));
                for kind in KINDS {
                    let r = row(&|bytes| {
                        let block = choice(kind, bytes)
                            .block_bytes
                            .unwrap_or(crate::decide::DEFAULT_BLOCK_BYTES);
                        (block / BLOCK_UNIT).to_string()
                    });
                    out.push_str(&format!("{kind:?} = {r}\n"));
                }
                out.push_str(&format!("[{c}.cpu_units]\n"));
                for kind in KINDS {
                    let r = row(&|bytes| {
                        let units = choice(kind, bytes).units.unwrap_or(CpuUnits::All);
                        format!("\"{}\"", units_name(units))
                    });
                    out.push_str(&format!("{kind:?} = {r}\n"));
                }
            }
        }
        out
    }
}

const VECTOR_TOML_HEADER: &str = "\
# Vector policy of gridcl-device (docs/design.md, \"Launch plans and the vector policy\"; ADR-0015,
# ADR-0058).
#
# Two device classes, Cpu and Gpu, each with one generic set per scalar size in bytes (4 and 8;
# ADR-0058): the QuadField tile width by scalar size; the vector size per kernel family and
# scalar size; on Cpu, the CPU block per kernel family and scalar size of the widest stream, in
# units of 4 KiB, and the CPU units (all hardware threads or the performance cores; ADR-0017).
# Written by `cargo run -p gridcl-bench --bin policy_report` from the M10 candidate records in
# docs/perf/data/ with the generic decision rule of docs/perf/vectorization.md (ADR-0058 item 4);
# do not edit by hand.
";

/// The configuration of an entry without data: the baseline on all hardware threads.
fn default_config(class: DeviceClass, kind: KernelKind, bytes: usize) -> VectorConfig {
    let base = vector_baseline(class, kind, bytes);
    VectorConfig {
        units: base.units.map(|_| CpuUnits::All),
        ..base
    }
}

/// The units of a CPU configuration in a record: `All` for all hardware threads, `Performance`
/// for the performance-core count; a device without one gives its all-threads time to both
/// (ADR-0017). `None` twice on a GPU.
fn record_units(r: &SweepRecord) -> [Option<CpuUnits>; 2] {
    if r.backend != "cpu" {
        return [None, None];
    }
    match r.cube_dim {
        CubeDimMode::PerformanceCores => [Some(CpuUnits::Performance), None],
        CubeDimMode::Plan if r.cores.performance.is_none() => {
            [Some(CpuUnits::All), Some(CpuUnits::Performance)]
        }
        CubeDimMode::Plan => [Some(CpuUnits::All), None],
    }
}

/// A member of a vector entry: a platform and a scalar type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Member {
    platform: Platform,
    scalar: Scalar,
}

/// The time of every configuration `m` measured for `kind` at `size`; a later record of the
/// same configuration replaces an earlier one (a repeated run).
fn member_times(
    vector: &[SweepRecord],
    m: Member,
    kind: KernelKind,
    size: usize,
) -> BTreeMap<ConfigKey, f64> {
    let mut out = BTreeMap::new();
    for r in vector.iter().filter(|r| {
        r.target == m.platform.target
            && r.backend == m.platform.backend
            && r.family == kind
            && r.scalar == m.scalar
            && r.nominal_n == size
    }) {
        for units in record_units(r) {
            if r.backend == "cpu" && units.is_none() {
                continue;
            }
            let c = VectorConfig {
                vector_size: r.vector_size,
                tile: r.tile,
                block_bytes: r.block_bytes,
                units,
            };
            out.insert(c.key(), r.median_s);
        }
    }
    out
}

/// The generic vector policy from the M10 candidate records.
pub fn decide_vector(data: &M10Data) -> VectorDecision {
    let vector = &data.vector;
    let present = |t: &str, b: &str| vector.iter().any(|r| r.target == t && r.backend == b);
    let platforms = m10_platforms(present);
    let mut entries = Vec::new();
    for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
        for (kind, scalars) in candidate_kinds() {
            for bytes in ELEM_BYTES {
                entries.push(vector_entry(
                    vector, &platforms, class, kind, scalars, bytes,
                ));
            }
        }
    }
    VectorDecision { entries }
}

fn vector_entry(
    vector: &[SweepRecord],
    platforms: &[Platform],
    class: DeviceClass,
    kind: KernelKind,
    scalars: &[Scalar],
    bytes: usize,
) -> VectorEntry {
    let baseline = vector_baseline(class, kind, bytes);
    let default = |reason: String| VectorEntry {
        class,
        kind,
        bytes,
        baseline,
        choice: default_config(class, kind, bytes),
        fit: GenericFit::Default,
        platforms: Vec::new(),
        reason,
    };
    let typed: Vec<Scalar> = scalars
        .iter()
        .copied()
        .filter(|s| s.bytes() == bytes)
        .collect();
    if typed.is_empty() || (kind == KernelKind::Scalar && class == DeviceClass::Gpu) {
        return default(format!("not used: no {bytes}-byte type on {class:?}"));
    }
    let members: Vec<Member> = platforms
        .iter()
        .filter(|p| p.class == class)
        .flat_map(|&platform| {
            backend_scalars(platform.backend, &typed)
                .into_iter()
                .map(move |scalar| Member { platform, scalar })
        })
        .filter(|&m| !member_times(vector, m, kind, CANDIDATE_SIZES[0]).is_empty())
        .collect();
    if members.is_empty() {
        return default("no M10 record".to_owned());
    }
    let times: Vec<Vec<BTreeMap<ConfigKey, f64>>> = members
        .iter()
        .map(|&m| {
            CANDIDATE_SIZES
                .iter()
                .map(|&size| member_times(vector, m, kind, size))
                .collect()
        })
        .collect();
    let common: Vec<ConfigKey> = times[0][0]
        .keys()
        .copied()
        .filter(|k| times.iter().flatten().all(|t| t.contains_key(k)))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if !common.contains(&baseline.key()) {
        return default(format!(
            "the baseline {} is not measured by every member at 10^6 and 10^7",
            describe(&baseline)
        ));
    }
    let outcome = generic_rule(
        baseline.key(),
        &common,
        members.len(),
        CANDIDATE_SIZES.len(),
        |c, m, s| times[m][s][&c],
    );
    let choice = VectorConfig::from_key(class, outcome.choice);
    let mut per_platform: Vec<PlatformRatio> = Vec::new();
    for (j, m) in members.iter().enumerate() {
        let worst_j = outcome.r[&outcome.choice][j]
            .iter()
            .copied()
            .fold(0.0, f64::max);
        match per_platform.iter_mut().find(|p| p.platform == m.platform) {
            Some(p) => p.r = p.r.max(worst_j),
            None => per_platform.push(PlatformRatio {
                platform: m.platform,
                best: baseline,
                r: worst_j,
            }),
        }
    }
    for p in &mut per_platform {
        let of: Vec<usize> = (0..members.len())
            .filter(|&j| members[j].platform == p.platform)
            .collect();
        let g = |c: &ConfigKey| -> f64 {
            let v: Vec<f64> = of
                .iter()
                .flat_map(|&j| outcome.r[c][j].iter().copied())
                .collect();
            geomean(&v)
        };
        let mut best = common[0];
        for c in &common[1..] {
            if g(c) < g(&best) {
                best = *c;
            }
        }
        p.best = VectorConfig::from_key(class, best);
    }
    let (fit, reason) = vector_reason(&outcome, &baseline, &choice, members.len(), &times);
    VectorEntry {
        class,
        kind,
        bytes,
        baseline,
        choice,
        fit,
        platforms: per_platform,
        reason,
    }
}

/// The fit and one-line reason of a vector entry.
fn vector_reason(
    outcome: &RuleOutcome<ConfigKey>,
    baseline: &VectorConfig,
    choice: &VectorConfig,
    members: usize,
    times: &[Vec<BTreeMap<ConfigKey, f64>>],
) -> (GenericFit, String) {
    if outcome.fallback {
        return (
            GenericFit::Fallback {
                worst: outcome.worst,
            },
            format!(
                "worst case r = {:.2}: no admitted candidate within {BOUND} on every platform \
                 and size; the smallest worst case",
                outcome.worst
            ),
        );
    }
    let speedup: Vec<String> = (0..CANDIDATE_SIZES.len())
        .map(|s| {
            let ratios: Vec<f64> = (0..members)
                .map(|m| times[m][s][&baseline.key()] / times[m][s][&choice.key()])
                .collect();
            format!("{:.2}x", geomean(&ratios))
        })
        .collect();
    let text = if choice == baseline {
        if outcome.admitted.len() == 1 {
            "the baseline: no candidate at least 10% faster by the geometric mean over the \
             platforms at 10^6 and 10^7"
                .to_owned()
        } else {
            format!(
                "the baseline: the simplest within 3% of the smallest geometric mean of r \
                 ({:.3}; worst r {:.2})",
                outcome.geomean, outcome.worst
            )
        }
    } else {
        format!(
            "{} over the baseline at 10^6 / 10^7 (geometric mean over the platforms); the \
             simplest within 3% of the smallest geometric mean of r ({:.3}; worst r {:.2})",
            speedup.join(" / "),
            outcome.geomean,
            outcome.worst
        )
    };
    (GenericFit::Bound, text)
}

/// A configuration in one cell: `n = 4, tile 4, 16 KiB, performance`.
pub fn describe(c: &VectorConfig) -> String {
    let mut parts = vec![format!("n = {}", c.vector_size)];
    if let Some(t) = c.tile {
        parts.push(format!("tile {t}"));
    }
    if let Some(b) = c.block_bytes {
        parts.push(format!("{} KiB", b >> 10));
    }
    if let Some(u) = c.units {
        parts.push(units_name(u).to_owned());
    }
    parts.join(", ")
}

// ---------------------------------------------------------------------------------------------
// The degree policy.

/// The degrees M10 measures every degree-policy entry at (ADR-0057 item 3).
pub const M10_DEGREES: [u32; 3] = [1, 4, 8];

/// A time at 10^6 points that moves by more than this fraction from M9h's record sends its range
/// to re-measurement (ADR-0058 item 5, ADR-0060 item 3).
pub const MOVE: f64 = 0.10;

/// The M10 degrees of a cell type: p = 1, 4 and 8, pyramids at p = 1 only.
pub fn m10_degrees(cell: &str) -> Vec<u32> {
    if cell == "Pyramid" {
        vec![1]
    } else {
        M10_DEGREES.to_vec()
    }
}

/// The strategies a class chooses among, in simplicity order: every strategy on Gpu; `direct`,
/// `loops` and `sumfact` on Cpu, where `shared` and `cooperative` reduce to `loops` (ADR-0052).
pub fn degree_strategies(class: DeviceClass) -> &'static [&'static str] {
    match class {
        DeviceClass::Gpu => &STRATEGIES,
        DeviceClass::Cpu => &["direct", "loops", "sumfact"],
    }
}

/// Where a degree's choice comes from (ADR-0058 item 5, ADR-0060 item 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DegreeSource {
    /// M10's records: p = 1, 4 and 8, and the degrees re-measured next to a range's boundaries;
    /// every M10 platform of the class, at 10^6 and 10^7 points.
    M10,
    /// M9h's per-degree records, at their own rule sizes (10^6 points on Cpu; 10^6 and 10^7 on
    /// Gpu), on the platforms that measured them (no Grace).
    M9h,
}

/// One platform's view of a degree choice.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyRatio {
    pub platform: Platform,
    /// The selectable strategy with the smallest geometric mean of r over the rule sizes (the
    /// simplest on ties).
    pub best: &'static str,
    /// The largest r of the generic choice over the rule sizes.
    pub r: f64,
}

/// The generic rule's outcome at one degree of one degree-policy entry.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreePick {
    pub degree: u32,
    pub source: DegreeSource,
    pub strategy: &'static str,
    /// `Default` where neither `direct` nor `loops` is selectable: `loops` is stored.
    pub fit: GenericFit,
    /// The baseline: `direct`, or `loops` where `direct` is not selectable.
    pub base: &'static str,
    /// The selectable strategies, in simplicity order: measured on every platform at every
    /// rule size, and none failed or took over 30 s to compile and load (ADR-0056).
    pub selectable: Vec<&'static str>,
    /// The rule sizes, in quadrature points.
    pub sizes: Vec<usize>,
    /// Per platform that took part, in [`m10_platforms`] order.
    pub platforms: Vec<StrategyRatio>,
    /// Why, in one line for the report.
    pub reason: String,
}

/// An M10 time at 10^6 points against M9h's on the same platform and strategy (ADR-0060
/// item 3).
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeMove {
    pub degree: u32,
    pub platform: Platform,
    pub strategy: &'static str,
    /// M10's time per point over M9h's.
    pub ratio: f64,
}

impl DegreeMove {
    /// The time moved by more than [`MOVE`].
    pub fn moved(&self) -> bool {
        (self.ratio - 1.0).abs() > MOVE * (1.0 + SLACK)
    }
}

/// One entry of the generic degree policy: (class, policy family, cell, scalar) with its choice
/// at every measured degree, its degree ranges and M10's check of them.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeEntry {
    pub class: DeviceClass,
    pub family: &'static str,
    pub cell: &'static str,
    pub scalar: Scalar,
    /// The choice at p = 1, …, the cell's highest measured degree (8, pyramids 3).
    pub picks: Vec<DegreePick>,
    /// Ascending (first degree, strategy), from 1; the last range is open-ended.
    pub ranges: Vec<(u32, &'static str)>,
    /// Every comparison of M10's check at p = 1, 4 and 8 with M9h's records.
    pub moves: Vec<DegreeMove>,
    /// The degrees M10 has not measured that a range of this entry needs re-measured: a range
    /// whose check at p = 1, 4 or 8 misses the bound or moves by more than 10% is re-measured,
    /// with candidates only, at the degrees on both sides of each of its boundaries (ADR-0060
    /// item 3).
    pub remeasure: Vec<u32>,
}

impl DegreeEntry {
    /// The strategy of `degree`.
    pub fn strategy(&self, degree: u32) -> &'static str {
        self.ranges
            .iter()
            .rev()
            .find(|(first, _)| *first <= degree)
            .map_or("loops", |(_, s)| s)
    }

    /// The pick at `degree`, if measured.
    pub fn pick(&self, degree: u32) -> Option<&DegreePick> {
        self.picks.iter().find(|p| p.degree == degree)
    }

    /// Each range as (first degree, last degree or `None` for the open-ended one, strategy).
    pub fn spans(&self) -> Vec<(u32, Option<u32>, &'static str)> {
        self.ranges
            .iter()
            .enumerate()
            .map(|(i, &(first, s))| (first, self.ranges.get(i + 1).map(|r| r.0 - 1), s))
            .collect()
    }
}

/// The generic degree policy: `[Cpu.<family>]` and `[Gpu.<family>]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeDecision {
    /// Every (class, family, cell, scalar), Cpu first, in the order of `degree-policy.toml`.
    pub entries: Vec<DegreeEntry>,
}

impl DegreeDecision {
    /// The entry of (`class`, policy `family`, `cell`, `scalar`).
    pub fn entry(
        &self,
        class: DeviceClass,
        family: &str,
        cell: &str,
        scalar: Scalar,
    ) -> Option<&DegreeEntry> {
        self.entries.iter().find(|e| {
            e.class == class && e.family == family && e.cell == cell && e.scalar == scalar
        })
    }

    /// The text of `gridcl-geometry/degree-policy.toml`.
    pub fn to_toml(&self) -> String {
        let mut out = String::from(DEGREE_TOML_HEADER);
        for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
            for family in POLICY_FAMILIES {
                out.push_str(&format!("\n[{class:?}.{family}]\n"));
                for &cell in cells_of(family) {
                    let sizes: Vec<String> = SCALARS
                        .iter()
                        .map(|&(scalar, bytes)| {
                            let e = self
                                .entry(class, family, cell, scalar)
                                .expect("an entry per class, family, cell and scalar");
                            let ranges: Vec<String> = e
                                .ranges
                                .iter()
                                .map(|(first, s)| format!("[{first}, \"{s}\"]"))
                                .collect();
                            format!("{bytes} = [{}]", ranges.join(", "))
                        })
                        .collect();
                    out.push_str(&format!("{cell} = {{ {} }}\n", sizes.join(", ")));
                }
            }
        }
        out
    }
}

const DEGREE_TOML_HEADER: &str = "\
# Degree policy of gridcl-geometry (docs/design.md, \"Geometric factors\"; ADR-0042, item 3;
# ADR-0058).
#
# Per device class (Cpu or Gpu; ADR-0058) and kernel family, per cell type and scalar size in
# bytes: the strategy of each degree range, as [first degree, strategy] pairs with ascending
# first degrees, starting at 1; the last range is open-ended. Families: EVector, Points,
# Jacobian (J, det J and the pseudo-inverse), Normals (codimension 1: intervals, triangles and
# quadrilaterals), FacetNormals and PullBack (with the device basis evaluator). Strategies:
# direct, loops, sumfact, shared, cooperative. Written by
# `cargo run -p gridcl-bench --bin high_order_report` from M10's records at p = 1, 4 and 8 and
# M9h's in between (docs/perf/data/) with the generic decision rule of docs/perf/high-order.md
# (ADR-0058 item 4, ADR-0060 item 3); do not edit by hand.
";

/// `data` without its records whose check error is not finite (ADR-0060 item 3).
fn finite_checks(data: &HighOrderData) -> HighOrderData {
    let keep = |records: &[DegreeRecord]| -> Vec<DegreeRecord> {
        records
            .iter()
            .filter(|r| r.check_error.is_none_or(f64::is_finite))
            .cloned()
            .collect()
    };
    HighOrderData {
        baseline: data.baseline.clone(),
        degree: keep(&data.degree),
        launch: keep(&data.launch),
        resources: data.resources.clone(),
    }
}

/// The platforms of M9h's sweep in `class` that measure `scalar`: the Mac mini's CPU and one
/// Kathleen socket (a Myriad D node where Kathleen has none) for Cpu, Metal and Locust's GPU
/// for Gpu.
fn m9h_platforms(m9h: &HighOrderData, class: DeviceClass, scalar: Scalar) -> Vec<Platform> {
    sweep_devices(&m9h.degree)
        .into_iter()
        .map(|(target, backend, old)| Platform {
            target,
            backend,
            class: if old == LegacyClass::Gpu {
                DeviceClass::Gpu
            } else {
                DeviceClass::Cpu
            },
        })
        .filter(|p| p.class == class && backend_scalars(p.backend, &[scalar]).contains(&scalar))
        .collect()
}

/// The generic degree policy from M10's records at p = 1, 4 and 8 (and any degrees they
/// re-measured) and M9h's in between (ADR-0058 item 5, ADR-0060 item 3).
///
/// - At each degree of an entry, the M10 records decide if any M10 platform of the class has a
///   record of one of the family's members there; the platforms with such a record take part,
///   at 10^6 and 10^7 points. Otherwise M9h's records decide, on M9h's platforms of the class
///   at their own rule sizes; M9h records whose check error is not finite are left out.
/// - At each degree, the selectable strategies of the class (as M9h's rule: every member
///   measured itself on every platform at every size, none failed or took over 30 s at first
///   use) go through [`generic_rule`] with the baseline `direct`, or `loops` where `direct` is
///   not selectable; with neither, `loops` is stored (`GenericFit::Default`).
/// - Degree ranges are maximal runs of equal choices; a `direct` choice at the highest measured
///   degree is followed by an open-ended `loops` range (ADR-0042 item 3).
/// - At p = 1, 4 and 8, every strategy M10 and M9h both timed at 10^6 points on a platform is
///   compared ([`DegreeEntry::moves`]); a range whose check falls back or moves by more than
///   10% lists the unmeasured degrees on both sides of each of its boundaries in
///   [`DegreeEntry::remeasure`].
pub fn decide_degree(m9h: &HighOrderData, m10: &M10Data) -> DegreeDecision {
    let m9h = finite_checks(m9h);
    let m10_data = HighOrderData {
        degree: m10.degree.clone(),
        ..HighOrderData::default()
    };
    let old = Times::new(&m9h);
    let new = Times::new(&m10_data);
    let present = |t: &str, b: &str| m10.degree.iter().any(|r| r.target == t && r.backend == b);
    let platforms = m10_platforms(present);
    let mut entries = Vec::new();
    for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
        for family in POLICY_FAMILIES {
            for &cell in cells_of(family) {
                for (scalar, _) in SCALARS {
                    let ctx = EntryContext {
                        class,
                        family,
                        cell,
                        scalar,
                        m10_records: &m10.degree,
                        m10_platforms: &platforms,
                        m9h_platforms: m9h_platforms(&m9h, class, scalar),
                        old: &old,
                        new: &new,
                    };
                    entries.push(ctx.entry());
                }
            }
        }
    }
    DegreeDecision { entries }
}

/// What one entry's decision reads.
struct EntryContext<'a> {
    class: DeviceClass,
    family: &'static str,
    cell: &'static str,
    scalar: Scalar,
    m10_records: &'a [DegreeRecord],
    m10_platforms: &'a [Platform],
    m9h_platforms: Vec<Platform>,
    old: &'a Times<'a>,
    new: &'a Times<'a>,
}

impl EntryContext<'_> {
    fn entry(&self) -> DegreeEntry {
        let picks: Vec<DegreePick> = (1..=max_degree(self.cell))
            .map(|p| match self.m10_members(p) {
                ref m if !m.is_empty() => self.pick(self.new, m, &CANDIDATE_SIZES, p, true),
                _ => {
                    let sizes: &[usize] = match self.class {
                        DeviceClass::Gpu => &CANDIDATE_SIZES,
                        DeviceClass::Cpu => &CANDIDATE_SIZES[..1],
                    };
                    self.pick(self.old, &self.m9h_platforms, sizes, p, false)
                }
            })
            .collect();
        let ranges = degree_ranges(&picks);
        let moves: Vec<DegreeMove> = picks
            .iter()
            .filter(|k| k.source == DegreeSource::M10 && m10_degrees(self.cell).contains(&k.degree))
            .flat_map(|k| self.moves(k))
            .collect();
        let mut entry = DegreeEntry {
            class: self.class,
            family: self.family,
            cell: self.cell,
            scalar: self.scalar,
            picks,
            ranges,
            moves,
            remeasure: Vec::new(),
        };
        entry.remeasure = remeasure(&entry);
        entry
    }

    /// The M10 platforms of the class (and scalar type) with a record of one of the family's
    /// members at `degree`.
    fn m10_members(&self, degree: u32) -> Vec<Platform> {
        let members = members(self.family);
        self.m10_platforms
            .iter()
            .copied()
            .filter(|p| {
                p.class == self.class
                    && backend_scalars(p.backend, &[self.scalar]).contains(&self.scalar)
            })
            .filter(|p| {
                self.m10_records.iter().any(|r| {
                    r.target == p.target
                        && r.backend == p.backend
                        && r.cell == self.cell
                        && r.degree == degree
                        && r.scalar == self.scalar
                        && members.iter().any(|&(f, _)| f == r.family)
                })
            })
            .collect()
    }

    /// The generic rule at `degree` on `platforms` with the records indexed in `times`.
    fn pick(
        &self,
        times: &Times,
        platforms: &[Platform],
        sizes: &[usize],
        degree: u32,
        from_m10: bool,
    ) -> DegreePick {
        let source = if from_m10 {
            DegreeSource::M10
        } else {
            DegreeSource::M9h
        };
        let devices: Vec<(&str, &str)> = platforms.iter().map(|p| (p.target, p.backend)).collect();
        let (family, cell, scalar) = (self.family, self.cell, self.scalar);
        let t = |s: &str, m: usize, k: usize| {
            times.family_time(devices[m], family, cell, degree, scalar, sizes[k], s)
        };
        let selectable: Vec<&'static str> = degree_strategies(self.class)
            .iter()
            .copied()
            .filter(|&s| {
                !devices.is_empty()
                    && (0..devices.len()).all(|m| (0..sizes.len()).all(|k| t(s, m, k).is_some()))
                    && times.first_use_ok(&devices, family, cell, degree, scalar, s)
            })
            .collect();
        let base = ["direct", "loops"]
            .into_iter()
            .find(|b| selectable.contains(b));
        let Some(base) = base else {
            return DegreePick {
                degree,
                source,
                strategy: "loops",
                fit: GenericFit::Default,
                base: "loops",
                selectable,
                sizes: sizes.to_vec(),
                platforms: Vec::new(),
                reason: "neither direct nor loops is selectable on every platform: loops"
                    .to_owned(),
            };
        };
        let time = |s: &'static str, m: usize, k: usize| t(s, m, k).expect("selectable");
        let outcome = generic_rule(base, &selectable, devices.len(), sizes.len(), time);
        let per_platform = platforms
            .iter()
            .enumerate()
            .map(|(m, &platform)| {
                let g = |s: &&'static str| geomean(&outcome.r[s][m]);
                let mut best = selectable[0];
                for s in &selectable[1..] {
                    if g(s) < g(&best) {
                        best = s;
                    }
                }
                StrategyRatio {
                    platform,
                    best,
                    r: outcome.r[&outcome.choice][m]
                        .iter()
                        .copied()
                        .fold(0.0, f64::max),
                }
            })
            .collect();
        let (fit, reason) = degree_reason(&outcome, base, sizes, devices.len(), time);
        DegreePick {
            degree,
            source,
            strategy: outcome.choice,
            fit,
            base,
            selectable,
            sizes: sizes.to_vec(),
            platforms: per_platform,
            reason,
        }
    }

    /// M10's check at `pick` (from M10) against M9h's records at 10^6 points, per platform and
    /// strategy both timed.
    fn moves(&self, pick: &DegreePick) -> Vec<DegreeMove> {
        let size = CANDIDATE_SIZES[0];
        let mut out = Vec::new();
        for k in &pick.platforms {
            let p = k.platform;
            if !self.m9h_platforms.contains(&p) {
                continue;
            }
            for &s in &pick.selectable {
                let at = |times: &Times| {
                    times.family_time(
                        (p.target, p.backend),
                        self.family,
                        self.cell,
                        pick.degree,
                        self.scalar,
                        size,
                        s,
                    )
                };
                if let (Some(new), Some(old)) = (at(self.new), at(self.old)) {
                    out.push(DegreeMove {
                        degree: pick.degree,
                        platform: p,
                        strategy: s,
                        ratio: new / old,
                    });
                }
            }
        }
        out
    }
}

/// The fit and one-line reason of a degree pick.
fn degree_reason(
    outcome: &RuleOutcome<&'static str>,
    base: &'static str,
    sizes: &[usize],
    members: usize,
    t: impl Fn(&'static str, usize, usize) -> f64,
) -> (GenericFit, String) {
    if outcome.fallback {
        return (
            GenericFit::Fallback {
                worst: outcome.worst,
            },
            format!(
                "worst case r = {:.2}: no admitted strategy within {BOUND} on every platform \
                 and size; the smallest worst case",
                outcome.worst
            ),
        );
    }
    let at: Vec<String> = sizes.iter().map(|n| format!("10^{}", n.ilog10())).collect();
    let choice = outcome.choice;
    let text = if choice == base {
        if outcome.admitted.len() == 1 {
            format!(
                "{base}: no strategy at least 10% faster by the geometric mean over the \
                 platforms at {}",
                at.join(" and ")
            )
        } else {
            format!(
                "{base}: the simplest within 3% of the smallest geometric mean of r ({:.3}; \
                 worst r {:.2})",
                outcome.geomean, outcome.worst
            )
        }
    } else {
        let speedup: Vec<String> = (0..sizes.len())
            .map(|k| {
                let v: Vec<f64> = (0..members)
                    .map(|m| t(base, m, k) / t(choice, m, k))
                    .collect();
                format!("{:.2}x", geomean(&v))
            })
            .collect();
        format!(
            "{choice} {} over {base} at {} (geometric mean over the platforms); the simplest \
             within 3% of the smallest geometric mean of r ({:.3}; worst r {:.2})",
            speedup.join(" / "),
            at.join(" / "),
            outcome.geomean,
            outcome.worst
        )
    };
    (GenericFit::Bound, text)
}

/// The ranges of the picks at p = 1, 2, …: maximal runs of equal strategies, then `loops` after
/// a `direct` last pick (the open-ended range must compile at p = 9–10; ADR-0042 item 3).
fn degree_ranges(picks: &[DegreePick]) -> Vec<(u32, &'static str)> {
    let mut out: Vec<(u32, &'static str)> = Vec::new();
    for k in picks {
        if out.last().is_none_or(|&(_, s)| s != k.strategy) {
            out.push((k.degree, k.strategy));
        }
    }
    match (out.last(), picks.last()) {
        (None, _) => out.push((1, "loops")),
        (Some(&(_, "direct")), Some(last)) => out.push((last.degree + 1, "loops")),
        _ => {}
    }
    out
}

/// The degrees M10 has not measured on both sides of each boundary of every range whose M10
/// check (p = 1, 4 or 8) falls back or moved by more than 10% from M9h, ascending (ADR-0060
/// item 3).
fn remeasure(e: &DegreeEntry) -> Vec<u32> {
    let top = max_degree(e.cell);
    let flagged = |p: u32| {
        e.pick(p).is_some_and(|k| {
            k.source == DegreeSource::M10 && matches!(k.fit, GenericFit::Fallback { .. })
        }) || e.moves.iter().any(|m| m.degree == p && m.moved())
    };
    let unmeasured = |p: u32| e.pick(p).is_some_and(|k| k.source == DegreeSource::M9h);
    let mut out = BTreeSet::new();
    for (first, last, _) in e.spans() {
        let last = last.unwrap_or(top).min(top);
        let checked = m10_degrees(e.cell)
            .into_iter()
            .any(|p| (first..=last).contains(&p) && flagged(p));
        if first > top || !checked {
            continue;
        }
        let mut sides = Vec::new();
        if first > 1 {
            sides.extend([first - 1, first]);
        }
        if last < top {
            sides.extend([last, last + 1]);
        }
        out.extend(sides.into_iter().filter(|&p| unmeasured(p)));
    }
    out.into_iter().collect()
}

#[cfg(test)]
mod tests;
