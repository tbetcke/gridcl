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
use crate::protocol::CANDIDATE_SIZES;
use crate::record::{CubeDimMode, HighOrderData, KINDS, M10Data, Scalar, SweepRecord};

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

/// The generic degree policy: `[Cpu.<family>]` and `[Gpu.<family>]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeDecision {}

impl DegreeDecision {
    /// The text of `gridcl-geometry/degree-policy.toml`.
    pub fn to_toml(&self) -> String {
        todo!("M10 T2b")
    }
}

/// The generic degree policy from M10's records at p = 1, 4 and 8 and M9h's in between.
pub fn decide_degree(_m9h: &HighOrderData, _m10: &M10Data) -> DegreeDecision {
    todo!("M10 T2b")
}

#[cfg(test)]
mod tests;
