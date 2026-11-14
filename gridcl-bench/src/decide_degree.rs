//! The decision rule of the degree sweep (ADR-0042, item 12; ADR-0050 to ADR-0052): per policy
//! family, cell type, device class, scalar size and degree, the strategy written to
//! `gridcl-geometry/degree-policy.toml`.
//!
//! - A policy family's time is the sum of its bench families' times per point: J + det J + K
//!   for `Jacobian`, the pull-back at 9 Newton iterations for `PullBack`, the family itself
//!   otherwise, on the codimension-0 meshes (codimension 1 for `Normals`) with equispaced nodes.
//! - The rule sizes are 10^6 and 10^7 points on the Gpu class and 10^6 on the CPU classes; the
//!   candidates are every strategy on the Gpu class and `direct`, `loops` and `sumfact` on the
//!   CPU classes (ADR-0052). The measured devices are the sweep's: the Mac mini's CPU
//!   (CpuSimd128), one Kathleen socket (CpuSimd512, a Myriad D node where Kathleen has none),
//!   Metal and Locust's GPU (Gpu); Locust's Grace CPU only keeps the paired check (ADR-0051).
//! - A strategy is selectable if every device of the class has an `ok` record of every member
//!   family at every rule size that ran the strategy itself, not its fallback `loops` (sum
//!   factorisation and cooperation on the E-vector, sum factorisation on simplices, `shared`
//!   beyond the shared-memory limit: `loops` measured again), and on the Gpu class if none of
//!   its Locust CUDA records of the member families (launch-only or timed) failed or took over
//!   30 s to compile and load (rule 7).
//! - Class ratios are geometric means of the per-device ratios (ADR-0015, item 5).
//! - The baseline is `direct`, or `loops` where `direct` is not selectable (as where the sweep
//!   leaves it out, ADR-0050). A strategy is admitted if it is the baseline or at least 10%
//!   faster than the baseline at every rule size. Among the admitted strategies within 3% of the
//!   fastest admitted one at every size, the simplest wins (`direct` < `loops` < `sumfact` <
//!   `shared` < `cooperative`); if none is within 3% at every size, the one with the smallest
//!   worst ratio to the fastest (simplest on ties).
//! - Degree ranges are maximal runs of equal choices. The last range is open-ended and must
//!   compile at p = 9–10, so a `direct` choice at the highest measured degree (8, pyramids 3) is
//!   followed by a `loops` range from the next degree (ADR-0042, item 3).
//! - CpuSimd256 copies the CpuSimd512 table ("Benchmarks", "Where").

use std::collections::{HashMap, HashSet};

use gridcl_device::DeviceClass;

use crate::record::{CLASSES, DegreeRecord, HighOrderData, STRATEGIES, Scalar};

/// A strategy is admitted if its class ratio to the baseline is at most this at every rule size
/// (at least 10% faster).
pub const ADMIT: f64 = 0.9;
/// A strategy qualifies if it takes at most this factor of the fastest admitted one.
pub const WITHIN: f64 = 1.03;
/// The bound of rule 7 on Locust's first-use compile and load time, in seconds.
pub const COMPILE_BOUND_S: f64 = 30.0;

/// The policy families, in the order of `degree-policy.toml`.
pub const POLICY_FAMILIES: [&str; 6] = [
    "EVector",
    "Points",
    "Jacobian",
    "Normals",
    "FacetNormals",
    "PullBack",
];

/// Every cell type, by its `CellType` name.
pub const CELLS: [&str; 7] = [
    "Interval",
    "Triangle",
    "Quadrilateral",
    "Tetrahedron",
    "Hexahedron",
    "Prism",
    "Pyramid",
];

/// The scalar types of the policy, with their sizes in bytes.
pub const SCALARS: [(Scalar, usize); 2] = [(Scalar::F32, 4), (Scalar::F64, 8)];

/// The measured device classes; CpuSimd256 copies CpuSimd512.
pub const MEASURED_CLASSES: [DeviceClass; 3] = [
    DeviceClass::CpuSimd128,
    DeviceClass::CpuSimd512,
    DeviceClass::Gpu,
];

/// The rule sizes of a class, in quadrature points (rule 12, ADR-0052).
pub fn rule_sizes(class: DeviceClass) -> &'static [usize] {
    if class == DeviceClass::Gpu {
        &[1_000_000, 10_000_000]
    } else {
        &[1_000_000]
    }
}

/// The strategies a class chooses among (ADR-0052).
pub fn candidates(class: DeviceClass) -> &'static [&'static str] {
    if class == DeviceClass::Gpu {
        &STRATEGIES
    } else {
        &["direct", "loops", "sumfact"]
    }
}

/// The cell types of a policy or bench family: codimension-1 cells for `Normals`.
pub fn cells_of(family: &str) -> &'static [&'static str] {
    if family == "Normals" {
        &CELLS[..3]
    } else {
        &CELLS
    }
}

/// The highest measured degree of a cell type: 8, and 3 on pyramids (ADR-0041).
pub fn max_degree(cell: &str) -> u32 {
    if cell == "Pyramid" { 3 } else { 8 }
}

/// The topological dimension of a cell type.
pub fn tdim(cell: &str) -> usize {
    match cell {
        "Interval" => 1,
        "Triangle" | "Quadrilateral" => 2,
        _ => 3,
    }
}

/// gdim of a bench family's mesh: tdim, one more for `Normals`.
pub fn gdim_of(family: &str, cell: &str) -> usize {
    tdim(cell) + usize::from(family == "Normals")
}

/// The bench families (and Newton iterations) whose times sum to a policy family's.
pub fn members(family: &str) -> Vec<(&'static str, Option<u32>)> {
    match family {
        "EVector" => vec![("EVector", None)],
        "Points" => vec![("Points", None)],
        "Jacobian" => vec![("Jacobian", None), ("DetJ", None), ("Pinv", None)],
        "Normals" => vec![("Normals", None)],
        "FacetNormals" => vec![("FacetNormals", None)],
        "PullBack" => vec![("PullBack", Some(9))],
        f => panic!("unknown policy family {f}"),
    }
}

/// The policy family of a bench family: J, det J and K share `Jacobian`; the basis evaluator is
/// the pull-back's.
pub fn policy_family(family: &str) -> &str {
    match family {
        "Jacobian" | "DetJ" | "Pinv" => "Jacobian",
        "BasisEval" => "PullBack",
        f => f,
    }
}

/// The (target, backend, class) of the degree sweep's devices: the Mac mini's CPU and Metal,
/// Locust's GPU and one Kathleen socket, or a Myriad D node where Kathleen has no CPU record
/// (ADR-0016); not Locust's Grace CPU (ADR-0051).
pub fn sweep_devices(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, DeviceClass)> {
    let avx512 = if records
        .iter()
        .any(|r| r.target == "kathleen" && r.backend == "cpu")
    {
        "kathleen"
    } else {
        "myriad-cpu"
    };
    vec![
        ("macmini", "cpu", DeviceClass::CpuSimd128),
        ("macmini", "metal", DeviceClass::Gpu),
        ("locust", "cuda", DeviceClass::Gpu),
        (avx512, "cpu", DeviceClass::CpuSimd512),
    ]
}

/// The scalar types a backend measures: Metal has no f64.
pub fn scalars_of(backend: &str) -> &'static [Scalar] {
    if backend == "metal" {
        &[Scalar::F32]
    } else {
        &[Scalar::F32, Scalar::F64]
    }
}

/// The devices of `class` that measure `scalar`, as (target, backend).
pub fn class_devices(
    records: &[DegreeRecord],
    class: DeviceClass,
    scalar: Scalar,
) -> Vec<(&'static str, &'static str)> {
    sweep_devices(records)
        .into_iter()
        .filter(|&(_, backend, c)| c == class && scalars_of(backend).contains(&scalar))
        .map(|(t, b, _)| (t, b))
        .collect()
}

/// The geometric mean.
pub fn geomean(v: &[f64]) -> f64 {
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// A timed configuration of one device: (target, backend, bench family, cell, degree, scalar,
/// size, iterations, strategy).
type TimeKey<'a> = (
    &'a str,
    &'a str,
    &'a str,
    &'a str,
    u32,
    Scalar,
    usize,
    Option<u32>,
    &'a str,
);

/// The timed records of a sweep, indexed: the time per point of the last `ok` record of each
/// configuration on the codimension-0 meshes (codimension 1 for `Normals`) with equispaced
/// nodes that ran its strategy, an interleaved one if there is one (the CPU classes at p ≤ 4,
/// ADR-0055), T3's baseline likewise, and Locust's CUDA compile and launch
/// outcomes.
pub struct Times<'a> {
    times: HashMap<TimeKey<'a>, f64>,
    baseline: HashMap<TimeKey<'a>, f64>,
    /// (bench family, cell, degree, scalar, strategy) → every Locust CUDA record (launch-only or
    /// timed) was `ok` within the compile bound.
    locust: HashMap<(&'a str, &'a str, u32, Scalar, &'a str), bool>,
    /// The largest Locust CUDA compile and load time of the same keys.
    locust_compile: HashMap<(&'a str, &'a str, u32, Scalar, &'a str), f64>,
}

impl<'a> Times<'a> {
    /// The index of `data`'s strategy sweeps and launch records. Rule 11's paired runs (every
    /// run with `policy` records) check the policy and are left out.
    pub fn new(data: &'a HighOrderData) -> Times<'a> {
        let paired: HashSet<&str> = data
            .degree
            .iter()
            .filter(|r| r.strategy == "policy")
            .map(|r| r.run_id.as_str())
            .collect();
        let index = |records: &'a [DegreeRecord]| {
            let mut times: HashMap<TimeKey<'a>, (f64, bool)> = HashMap::new();
            for r in records {
                if paired.contains(r.run_id.as_str())
                    || r.strategy_ran != r.strategy
                    || r.launch_only
                    || r.status != "ok"
                    || r.node_family != "equispaced"
                    || r.gdim != gdim_of(&r.family, &r.cell)
                {
                    continue;
                }
                let key = (
                    r.target.as_str(),
                    r.backend.as_str(),
                    r.family.as_str(),
                    r.cell.as_str(),
                    r.degree,
                    r.scalar,
                    r.nominal_points,
                    r.iterations,
                    r.strategy.as_str(),
                );
                // The last record, interleaved ones before sequential ones (ADR-0055).
                if times
                    .get(&key)
                    .is_none_or(|&(_, interleaved)| r.interleaved || !interleaved)
                {
                    times.insert(key, (r.time_per_point(), r.interleaved));
                }
            }
            times.into_iter().map(|(k, (t, _))| (k, t)).collect()
        };
        let times = index(&data.degree);
        let baseline = index(&data.baseline);
        let mut locust = HashMap::new();
        let mut locust_compile: HashMap<_, f64> = HashMap::new();
        for r in data.launch.iter().chain(&data.degree) {
            if r.target != "locust" || r.backend != "cuda" || paired.contains(r.run_id.as_str()) {
                continue;
            }
            let key = (
                r.family.as_str(),
                r.cell.as_str(),
                r.degree,
                r.scalar,
                r.strategy.as_str(),
            );
            let good = r.status == "ok" && r.compile_s <= COMPILE_BOUND_S;
            *locust.entry(key).or_insert(true) &= good;
            let c = locust_compile.entry(key).or_insert(0.0);
            *c = c.max(r.compile_s);
        }
        Times {
            times,
            baseline,
            locust,
            locust_compile,
        }
    }

    /// The time per point of one device's configuration, if it has an `ok` record.
    #[allow(clippy::too_many_arguments)]
    pub fn get(
        &self,
        device: (&str, &str),
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        size: usize,
        iterations: Option<u32>,
        strategy: &str,
    ) -> Option<f64> {
        let key = (
            device.0, device.1, family, cell, degree, scalar, size, iterations, strategy,
        );
        self.times.get(&key).copied()
    }

    /// The time per point of one device's configuration in T3's baseline (`direct`).
    #[allow(clippy::too_many_arguments)]
    pub fn get_baseline(
        &self,
        device: (&str, &str),
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        size: usize,
        iterations: Option<u32>,
    ) -> Option<f64> {
        let key = (
            device.0, device.1, family, cell, degree, scalar, size, iterations, "direct",
        );
        self.baseline.get(&key).copied()
    }

    /// The summed time per point of a policy family's members on one device in T3's baseline
    /// (`direct` at a88ab2a), if it has `ok` records.
    pub fn baseline_family_time(
        &self,
        device: (&str, &str),
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        size: usize,
    ) -> Option<f64> {
        members(family)
            .into_iter()
            .map(|(f, it)| {
                let key = (
                    device.0, device.1, f, cell, degree, scalar, size, it, "direct",
                );
                self.baseline.get(&key).copied()
            })
            .sum()
    }

    /// The summed time per point of a policy family's members on one device.
    #[allow(clippy::too_many_arguments)]
    pub fn family_time(
        &self,
        device: (&str, &str),
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        size: usize,
        strategy: &str,
    ) -> Option<f64> {
        members(family)
            .into_iter()
            .map(|(f, it)| self.get(device, f, cell, degree, scalar, size, it, strategy))
            .sum()
    }

    /// Whether `strategy` passed rule 7 on Locust for every member family of `family`: no
    /// failed launch and no compile over 30 s. Configurations without a record pass.
    pub fn locust_ok(
        &self,
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        strategy: &str,
    ) -> bool {
        members(family).into_iter().all(|(f, _)| {
            self.locust
                .get(&(f, cell, degree, scalar, strategy))
                .copied()
                .unwrap_or(true)
        })
    }

    /// The largest Locust CUDA compile and load time of a bench family's configuration.
    pub fn locust_compile_s(
        &self,
        family: &str,
        cell: &str,
        degree: u32,
        scalar: Scalar,
        strategy: &str,
    ) -> Option<f64> {
        self.locust_compile
            .get(&(family, cell, degree, scalar, strategy))
            .copied()
    }
}

/// How the rule reached a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Within 3% of the fastest admitted strategy at every rule size.
    Rule,
    /// No admitted strategy is within 3% at every size: the smallest worst ratio.
    Closest,
    /// Neither `direct` nor `loops` is selectable: `loops` is stored, and the case goes to the
    /// maintainer.
    Unresolved,
}

/// The rule's outcome at one degree of one policy entry.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeChoice {
    pub degree: u32,
    pub strategy: &'static str,
    pub fit: Fit,
    /// The baseline: `direct`, or `loops` where `direct` is not selectable.
    pub base: &'static str,
    /// The class time per point of the baseline at each rule size, in seconds.
    pub base_time: Vec<f64>,
    /// Every selectable candidate's class ratio to the baseline at each rule size.
    pub ratios: Vec<(&'static str, Vec<f64>)>,
    /// Where `direct` is not selectable and T3's baseline measured it: its class ratio to the
    /// baseline at each rule size (ADR-0050).
    pub t3_direct: Option<Vec<f64>>,
    /// Why, in one line for the report.
    pub reason: String,
}

impl DegreeChoice {
    /// The class ratios of `strategy` to the baseline, if it is selectable.
    pub fn ratio(&self, strategy: &str) -> Option<&[f64]> {
        self.ratios
            .iter()
            .find(|(s, _)| *s == strategy)
            .map(|(_, r)| r.as_slice())
    }
}

/// One entry of the policy: (class, policy family, cell, scalar size) with its choices at the
/// measured degrees and its degree ranges.
#[derive(Clone, Debug, PartialEq)]
pub struct PolicyEntry {
    pub class: DeviceClass,
    pub family: &'static str,
    pub cell: &'static str,
    pub scalar: Scalar,
    /// The choice at each measured degree; empty for CpuSimd256 (a copy) and where the class
    /// measured nothing in this scalar type.
    pub choices: Vec<DegreeChoice>,
    /// Ascending (first degree, strategy), from 1; the last range is open-ended.
    pub ranges: Vec<(u32, &'static str)>,
}

impl PolicyEntry {
    /// The strategy of `degree`.
    pub fn strategy(&self, degree: u32) -> &'static str {
        self.ranges
            .iter()
            .rev()
            .find(|(first, _)| *first <= degree)
            .map_or("loops", |(_, s)| s)
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

/// Ascending (first degree, strategy) of one policy entry.
type Ranges = Vec<(u32, &'static str)>;

/// The whole degree policy.
#[derive(Clone, Debug, PartialEq)]
pub struct DegreeDecision {
    pub entries: Vec<PolicyEntry>,
}

/// The order of the policy's classes.
pub const POLICY_CLASSES: [DeviceClass; 4] = CLASSES;

/// The decision for `data`.
pub fn decide_degree(data: &HighOrderData) -> DegreeDecision {
    let times = Times::new(data);
    let mut entries = Vec::new();
    for class in POLICY_CLASSES {
        let measured = if class == DeviceClass::CpuSimd256 {
            DeviceClass::CpuSimd512
        } else {
            class
        };
        for family in POLICY_FAMILIES {
            for &cell in cells_of(family) {
                for (scalar, _) in SCALARS {
                    let devices = class_devices(&data.degree, measured, scalar);
                    let choices: Vec<DegreeChoice> = if class == DeviceClass::CpuSimd256
                        || devices.is_empty()
                    {
                        Vec::new()
                    } else {
                        (1..=max_degree(cell))
                            .map(|p| choose(&times, measured, &devices, family, cell, p, scalar))
                            .collect()
                    };
                    entries.push(PolicyEntry {
                        class,
                        family,
                        cell,
                        scalar,
                        ranges: Vec::new(),
                        choices,
                    });
                }
            }
        }
    }
    for e in &mut entries {
        e.ranges = ranges(&e.choices);
    }
    // CpuSimd256 takes CpuSimd512's ranges.
    let copies: HashMap<(&str, &str, Scalar), Ranges> = entries
        .iter()
        .filter(|e| e.class == DeviceClass::CpuSimd512)
        .map(|e| ((e.family, e.cell, e.scalar), e.ranges.clone()))
        .collect();
    for e in &mut entries {
        if e.class == DeviceClass::CpuSimd256 {
            e.ranges = copies[&(e.family, e.cell, e.scalar)].clone();
        }
    }
    DegreeDecision { entries }
}

/// The ranges of the choices at p = 1, 2, …: maximal runs of equal strategies, then `loops`
/// after a `direct` last choice; `loops` everywhere without choices.
fn ranges(choices: &[DegreeChoice]) -> Vec<(u32, &'static str)> {
    let mut out: Vec<(u32, &'static str)> = Vec::new();
    for c in choices {
        if out.last().is_none_or(|&(_, s)| s != c.strategy) {
            out.push((c.degree, c.strategy));
        }
    }
    match (out.last(), choices.last()) {
        (None, _) => out.push((1, "loops")),
        (Some(&(_, "direct")), Some(last)) => out.push((last.degree + 1, "loops")),
        _ => {}
    }
    out
}

/// The rule at one degree.
fn choose(
    times: &Times,
    class: DeviceClass,
    devices: &[(&str, &str)],
    family: &'static str,
    cell: &str,
    degree: u32,
    scalar: Scalar,
) -> DegreeChoice {
    let sizes = rule_sizes(class);
    let class_time = |s: &str, size: usize| -> Option<f64> {
        let t: Option<Vec<f64>> = devices
            .iter()
            .map(|&d| times.family_time(d, family, cell, degree, scalar, size, s))
            .collect();
        t.map(|t| geomean(&t))
    };
    let ratio = |s: &str, base: &str, size: usize| -> Option<f64> {
        let r: Option<Vec<f64>> = devices
            .iter()
            .map(|&d| {
                Some(
                    times.family_time(d, family, cell, degree, scalar, size, s)?
                        / times.family_time(d, family, cell, degree, scalar, size, base)?,
                )
            })
            .collect();
        r.map(|r| geomean(&r))
    };
    let selectable = |s: &str| {
        sizes.iter().all(|&n| class_time(s, n).is_some())
            && (class != DeviceClass::Gpu || times.locust_ok(family, cell, degree, scalar, s))
    };
    let base = if selectable("direct") {
        "direct"
    } else if selectable("loops") {
        "loops"
    } else {
        return DegreeChoice {
            degree,
            strategy: "loops",
            fit: Fit::Unresolved,
            base: "loops",
            base_time: Vec::new(),
            ratios: Vec::new(),
            t3_direct: None,
            reason: "neither direct nor loops ran everywhere".to_owned(),
        };
    };
    let ratios: Vec<(&'static str, Vec<f64>)> = candidates(class)
        .iter()
        .copied()
        .filter(|&s| selectable(s))
        .map(|s| {
            let r = sizes
                .iter()
                .map(|&n| ratio(s, base, n).expect("selectable"))
                .collect();
            (s, r)
        })
        .collect();
    let admitted: Vec<&(&'static str, Vec<f64>)> = ratios
        .iter()
        .filter(|(s, r)| *s == base || r.iter().all(|&x| x <= ADMIT))
        .collect();
    let fastest: Vec<f64> = (0..sizes.len())
        .map(|i| {
            admitted
                .iter()
                .map(|(_, r)| r[i])
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let worst = |r: &[f64]| {
        r.iter()
            .zip(&fastest)
            .map(|(x, f)| x / f)
            .fold(0.0, f64::max)
    };
    let (strategy, fit) = match admitted.iter().find(|(_, r)| worst(r) <= WITHIN) {
        Some((s, _)) => (*s, Fit::Rule),
        None => {
            let (s, _) = admitted
                .iter()
                .min_by(|a, b| worst(&a.1).total_cmp(&worst(&b.1)))
                .expect("the baseline is admitted");
            (*s, Fit::Closest)
        }
    };
    let show = |r: &[f64]| {
        r.iter()
            .map(|x| format!("{x:.2}"))
            .collect::<Vec<_>>()
            .join(" / ")
    };
    let (best, best_ratios) = admitted
        .iter()
        .min_by(|a, b| geomean(&a.1).total_cmp(&geomean(&b.1)))
        .map(|(s, r)| (*s, r.as_slice()))
        .expect("the baseline is admitted");
    let own = &ratios
        .iter()
        .find(|(s, _)| *s == strategy)
        .expect("chosen")
        .1;
    let reason = match fit {
        Fit::Closest => format!(
            "closest: {strategy} {} of {base}, {:.0}% slower than the fastest at one size",
            show(own),
            (worst(own) - 1.0) * 100.0
        ),
        _ if admitted.len() == 1 => format!("no strategy 10% faster than {base}"),
        _ if best == strategy => format!("{strategy} {} of {base}", show(own)),
        _ if strategy == base => {
            format!(
                "{base} within 3% of {best} ({} of {base})",
                show(best_ratios)
            )
        }
        _ => format!(
            "{strategy} {} of {base}, within 3% of {best} ({})",
            show(own),
            show(best_ratios)
        ),
    };
    let t3_direct = (base != "direct")
        .then(|| {
            sizes
                .iter()
                .map(|&n| {
                    let r: Option<Vec<f64>> = devices
                        .iter()
                        .map(|&d| {
                            Some(
                                times.baseline_family_time(d, family, cell, degree, scalar, n)?
                                    / times
                                        .family_time(d, family, cell, degree, scalar, n, base)?,
                            )
                        })
                        .collect();
                    r.map(|r| geomean(&r))
                })
                .collect::<Option<Vec<f64>>>()
        })
        .flatten();
    DegreeChoice {
        degree,
        strategy,
        fit,
        base,
        reason,
        t3_direct,
        base_time: sizes
            .iter()
            .map(|&n| class_time(base, n).expect("selectable"))
            .collect(),
        ratios,
    }
}

impl DegreeDecision {
    /// The entry of (class, policy family, cell, scalar).
    pub fn entry(
        &self,
        class: DeviceClass,
        family: &str,
        cell: &str,
        scalar: Scalar,
    ) -> Option<&PolicyEntry> {
        self.entries.iter().find(|e| {
            e.class == class && e.family == family && e.cell == cell && e.scalar == scalar
        })
    }

    /// The entries the rule could not resolve.
    pub fn unresolved(&self) -> impl Iterator<Item = (&PolicyEntry, &DegreeChoice)> {
        self.entries.iter().flat_map(|e| {
            e.choices
                .iter()
                .filter(|c| c.fit == Fit::Unresolved)
                .map(move |c| (e, c))
        })
    }

    /// The text of `gridcl-geometry/degree-policy.toml`.
    pub fn to_toml(&self) -> String {
        let mut out = String::from(TOML_HEADER);
        for class in POLICY_CLASSES {
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

/// The bound of rule 1 on a ratio of time per point at p = 8 over p = 4: 3 for the geometric
/// factors and the pull-back's set-up, 1.1 · (9/5)^d for a Newton step.
pub fn tensor_bound(what: &str, d: i32) -> f64 {
    if what == "step" {
        1.1 * (9.0f64 / 5.0).powi(d)
    } else {
        3.0
    }
}

/// Rule 1's ratio of one (class, scalar, cell, bench family, part, size) on quadrilaterals and
/// hexahedra: the time per point at p = 8 over p = 4 of the strategy the policy selects at
/// p = 8, the geometric mean over the class's devices.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorRatio {
    pub class: DeviceClass,
    pub scalar: Scalar,
    pub cell: &'static str,
    pub family: &'static str,
    /// `factor`, or the pull-back's `step` or `set-up` (rule 1).
    pub part: &'static str,
    pub strategy: &'static str,
    pub size: usize,
    /// `None` where a device lacks a record.
    pub ratio: Option<f64>,
    pub bound: f64,
    /// The speedup at p = 8 over M9's kernels: T3's `direct` time over the strategy's, the
    /// geometric mean over the devices; `None` where a device has no baseline record (ADR-0053).
    pub speedup_m9: Option<f64>,
}

/// Rule 1's ratios under `decision`: step = (t(9) − t(1)) / 8, set-up = t(1) − step; measured
/// and reported, not enforced (ADR-0053).
pub fn tensor_ratios(data: &HighOrderData, decision: &DegreeDecision) -> Vec<TensorRatio> {
    let times = Times::new(data);
    let mut out = Vec::new();
    for class in MEASURED_CLASSES {
        for (scalar, _) in SCALARS {
            let devices = class_devices(&data.degree, class, scalar);
            if devices.is_empty() {
                continue;
            }
            for (cell, d) in [("Quadrilateral", 2), ("Hexahedron", 3)] {
                let mut families = vec!["Points", "Jacobian", "DetJ", "Pinv", "FacetNormals"];
                if cell == "Quadrilateral" {
                    families.push("Normals");
                }
                families.push("PullBack");
                for family in families {
                    let strategy = decision
                        .entry(class, policy_family(family), cell, scalar)
                        .expect("an entry")
                        .strategy(8);
                    for &size in rule_sizes(class) {
                        // `m9`: T3's baseline instead of the strategy's records.
                        let t = |m9: bool, device: (&str, &str), degree, iterations| {
                            if m9 {
                                times.get_baseline(
                                    device, family, cell, degree, scalar, size, iterations,
                                )
                            } else {
                                times.get(
                                    device, family, cell, degree, scalar, size, iterations,
                                    strategy,
                                )
                            }
                        };
                        let step = |m9, device, p| {
                            Some((t(m9, device, p, Some(9))? - t(m9, device, p, Some(1))?) / 8.0)
                        };
                        let part_time = |m9, part, device, p| match part {
                            "step" => step(m9, device, p),
                            "set-up" => Some(t(m9, device, p, Some(1))? - step(m9, device, p)?),
                            _ => t(m9, device, p, None),
                        };
                        let parts: &[&'static str] = if family == "PullBack" {
                            &["step", "set-up"]
                        } else {
                            &["factor"]
                        };
                        for &part in parts {
                            let r: Option<Vec<f64>> = devices
                                .iter()
                                .map(|&dv| {
                                    Some(
                                        part_time(false, part, dv, 8)?
                                            / part_time(false, part, dv, 4)?,
                                    )
                                })
                                .collect();
                            let speedup: Option<Vec<f64>> = devices
                                .iter()
                                .map(|&dv| {
                                    Some(
                                        part_time(true, part, dv, 8)?
                                            / part_time(false, part, dv, 8)?,
                                    )
                                })
                                .collect();
                            out.push(TensorRatio {
                                class,
                                scalar,
                                cell,
                                family,
                                part,
                                strategy,
                                size,
                                ratio: r.map(|r| geomean(&r)),
                                bound: tensor_bound(part, d),
                                speedup_m9: speedup.map(|r| geomean(&r)),
                            });
                        }
                    }
                }
            }
        }
    }
    out
}

/// The devices of the paired check of rule 11 in `class` measuring `scalar`: the sweep's and
/// Locust's Grace CPU (CpuSimd128, ADR-0051).
pub fn paired_devices(
    records: &[DegreeRecord],
    class: DeviceClass,
    scalar: Scalar,
) -> Vec<(&'static str, &'static str)> {
    let mut out = class_devices(records, class, scalar);
    if class == DeviceClass::CpuSimd128 {
        out.push(("locust", "cpu"));
    }
    out
}

/// The paired ratio of two records of one configuration: the median over the timed runs of
/// `a`'s run i over `b`'s run i (with `degree_sweep --interleave`, adjacent in time); the ratio
/// of the medians where the run counts differ.
pub fn paired_ratio(a: &DegreeRecord, b: &DegreeRecord) -> f64 {
    if a.times_s.len() != b.times_s.len() || a.times_s.is_empty() {
        return a.median_s / b.median_s;
    }
    let r: Vec<f64> = a
        .times_s
        .iter()
        .zip(&b.times_s)
        .map(|(x, y)| x / y)
        .collect();
    crate::protocol::median(&r)
}

/// Rule 11's paired check of one (class, scalar), read as in ADR-0055: per policy family (the
/// members' times summed per run, the pull-back at 9 iterations), cell type, degree p ≤ 4 and
/// rule size where `direct` is selectable for the class, the class ratios over each device's
/// newest paired run of the policy's run i over `direct`'s run i: their median, and their lower
/// quartile, which must not exceed [`PAIRED_BOUND`] (the policy over 5% slower in at least three
/// quarters of the run pairs).
#[derive(Clone, Debug, PartialEq)]
pub struct PairedCheck {
    pub class: DeviceClass,
    pub scalar: Scalar,
    /// (target, backend, run id) of each device with a paired run.
    pub runs: Vec<(String, String, String)>,
    /// Configurations compared, left out because `direct` is not selectable for the class, and
    /// without records on every device.
    pub compared: usize,
    pub excluded: usize,
    pub missing: usize,
    /// The largest median and lower-quartile class ratios, with their configurations.
    pub worst_median: Option<(f64, String)>,
    pub worst_lower: Option<(f64, String)>,
    /// The configurations whose lower-quartile class ratio exceeds [`PAIRED_BOUND`].
    pub failures: Vec<String>,
    /// Reported, not bounded: per bench family the policy family does not decide alone
    /// (BasisEval; J, det J and K; the pull-back at 1 iteration), the largest median class ratio
    /// over every compared configuration.
    pub reported: Vec<(String, f64, String)>,
}

/// Bench families with their Newton iterations (`None` outside the pull-back).
type Members = Vec<(&'static str, Option<u32>)>;

/// Rule 11's bound on the lower quartile of a class's paired ratios (ADR-0055).
pub const PAIRED_BOUND: f64 = 1.05;

/// The lower quartile of `v` (linear interpolation between order statistics).
pub fn lower_quartile(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let i = (v.len() - 1) as f64 * 0.25;
    let lo = i.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (i - lo as f64)
}

/// The per-run ratios of the summed times of `a` over those of `b` (records of the members of
/// one policy family, in the same order).
fn run_ratios(a: &[&DegreeRecord], b: &[&DegreeRecord]) -> Option<Vec<f64>> {
    let n = a.first()?.times_s.len();
    if n == 0 || a.iter().chain(b).any(|r| r.times_s.len() != n) {
        return None;
    }
    Some(
        (0..n)
            .map(|i| {
                a.iter().map(|r| r.times_s[i]).sum::<f64>()
                    / b.iter().map(|r| r.times_s[i]).sum::<f64>()
            })
            .collect(),
    )
}

/// Rule 11's paired checks, per measured class and scalar type (ADR-0055).
pub fn paired_checks(data: &HighOrderData, decision: &DegreeDecision) -> Vec<PairedCheck> {
    let mut out = Vec::new();
    for class in MEASURED_CLASSES {
        for (scalar, _) in SCALARS {
            let devices = paired_devices(&data.degree, class, scalar);
            let runs: Vec<(&str, &str, &str)> = devices
                .iter()
                .filter_map(|&(t, b)| {
                    data.degree
                        .iter()
                        .filter(|r| r.target == t && r.backend == b && r.strategy == "policy")
                        .map(|r| r.run_id.as_str())
                        .max()
                        .map(|run| (t, b, run))
                })
                .collect();
            let mut check = PairedCheck {
                class,
                scalar,
                runs: runs
                    .iter()
                    .map(|&(t, b, r)| (t.to_owned(), b.to_owned(), r.to_owned()))
                    .collect(),
                compared: 0,
                excluded: 0,
                missing: 0,
                worst_median: None,
                worst_lower: None,
                failures: Vec::new(),
                reported: Vec::new(),
            };
            if devices.is_empty() || runs.len() < devices.len() {
                out.push(check);
                continue;
            }
            let mut index: HashMap<_, &DegreeRecord> = HashMap::new();
            for r in &data.degree {
                if runs.contains(&(r.target.as_str(), r.backend.as_str(), r.run_id.as_str()))
                    && r.status == "ok"
                    && !r.launch_only
                    && r.node_family == "equispaced"
                    && r.scalar == scalar
                    && r.gdim == gdim_of(&r.family, &r.cell)
                {
                    let key = (
                        r.target.as_str(),
                        r.backend.as_str(),
                        r.family.as_str(),
                        r.cell.as_str(),
                        r.degree,
                        r.nominal_points,
                        r.iterations,
                        r.strategy.as_str(),
                    );
                    index.insert(key, r);
                }
            }
            // The class's median and lower-quartile ratios of `members` at one configuration.
            let class_ratios = |members: &[(&str, Option<u32>)], cell, degree, size| {
                let mut medians = Vec::new();
                let mut lowers = Vec::new();
                for &(t, b, _) in &runs {
                    let get = |s| -> Option<Vec<&DegreeRecord>> {
                        members
                            .iter()
                            .map(|&(f, it)| {
                                index.get(&(t, b, f, cell, degree, size, it, s)).copied()
                            })
                            .collect()
                    };
                    let r = run_ratios(&get("policy")?, &get("direct")?)?;
                    medians.push(crate::protocol::median(&r));
                    lowers.push(lower_quartile(&r));
                }
                Some((geomean(&medians), geomean(&lowers)))
            };
            let mut reported: Vec<(String, f64, String)> = Vec::new();
            for family in POLICY_FAMILIES {
                let mut parts: Vec<(String, Members)> = Vec::new();
                match family {
                    "Jacobian" => {
                        for f in ["Jacobian", "DetJ", "Pinv"] {
                            parts.push((f.to_owned(), vec![(f, None)]));
                        }
                    }
                    "PullBack" => {
                        parts.push(("PullBack 1 it.".to_owned(), vec![("PullBack", Some(1))]));
                        parts.push(("BasisEval".to_owned(), vec![("BasisEval", None)]));
                    }
                    _ => {}
                }
                for &cell in cells_of(family) {
                    let entry = decision
                        .entry(class, family, cell, scalar)
                        .expect("an entry per class, family, cell and scalar");
                    for degree in 1..=max_degree(cell).min(4) {
                        let at = |size: usize| format!("{family} {cell} p={degree} n={size}");
                        for &size in rule_sizes(class) {
                            for (name, m) in &parts {
                                if let Some((median, _)) = class_ratios(m, cell, degree, size) {
                                    let worse = match reported.iter().position(|r| &r.0 == name) {
                                        Some(i) if reported[i].1 >= median => None,
                                        Some(i) => Some(i),
                                        None => {
                                            reported.push((name.clone(), 0.0, String::new()));
                                            Some(reported.len() - 1)
                                        }
                                    };
                                    if let Some(i) = worse {
                                        reported[i].1 = median;
                                        reported[i].2 = at(size).replacen(family, name, 1);
                                    }
                                }
                            }
                            let selectable = entry
                                .choices
                                .get(degree as usize - 1)
                                .is_some_and(|c| c.base == "direct");
                            if !selectable {
                                check.excluded += 1;
                                continue;
                            }
                            let Some((median, lower)) =
                                class_ratios(&members(family), cell, degree, size)
                            else {
                                check.missing += 1;
                                continue;
                            };
                            check.compared += 1;
                            if check.worst_median.as_ref().is_none_or(|w| median > w.0) {
                                check.worst_median = Some((median, at(size)));
                            }
                            if check.worst_lower.as_ref().is_none_or(|w| lower > w.0) {
                                check.worst_lower = Some((lower, at(size)));
                            }
                            if lower > PAIRED_BOUND {
                                check.failures.push(format!(
                                    "{} (median {median:.3}, lower quartile {lower:.3})",
                                    at(size)
                                ));
                            }
                        }
                    }
                }
            }
            check.reported = reported;
            out.push(check);
        }
    }
    out
}

const TOML_HEADER: &str = "\
# Degree policy of gridcl-geometry (docs/design.md, \"Geometric factors\"; ADR-0042, item 3).
#
# Per device class and kernel family, per cell type and scalar size in bytes: the strategy of
# each degree range, as [first degree, strategy] pairs with ascending first degrees, starting at
# 1; the last range is open-ended. Families: EVector, Points, Jacobian (J, det J and the
# pseudo-inverse), Normals (codimension 1: intervals, triangles and quadrilaterals),
# FacetNormals and PullBack (with the device basis evaluator). Strategies: direct, loops,
# sumfact, shared, cooperative. Written by `cargo run -p gridcl-bench --bin high_order_report`
# from docs/perf/data/ with the decision rule of docs/perf/high-order.md; do not edit by hand.
";

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(degree: u32, strategy: &'static str) -> DegreeChoice {
        DegreeChoice {
            degree,
            strategy,
            fit: Fit::Rule,
            base: "direct",
            base_time: vec![1e-9],
            ratios: vec![("direct", vec![1.0])],
            t3_direct: None,
            reason: String::new(),
        }
    }

    #[test]
    fn ranges_are_maximal_runs_and_end_beyond_direct() {
        let c = |v: &[&'static str]| {
            v.iter()
                .enumerate()
                .map(|(i, s)| choice(i as u32 + 1, s))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            ranges(&c(&["direct", "direct", "loops", "loops", "sumfact"])),
            [(1, "direct"), (3, "loops"), (5, "sumfact")]
        );
        assert_eq!(
            ranges(&c(&["direct", "direct", "direct"])),
            [(1, "direct"), (4, "loops")]
        );
        assert_eq!(
            ranges(&c(&["loops", "direct"])),
            [(1, "loops"), (2, "direct"), (3, "loops")]
        );
        assert_eq!(ranges(&[]), [(1, "loops")]);
    }

    #[test]
    fn without_data_every_entry_is_loops() {
        let decision = decide_degree(&HighOrderData::default());
        assert!(decision.entries.iter().all(|e| e.ranges == [(1, "loops")]));
        let toml = decision.to_toml();
        assert!(toml.contains(
            "[CpuSimd256.Normals]\nInterval = { 4 = [[1, \"loops\"]], 8 = [[1, \"loops\"]] }\n\
             Triangle"
        ));
        assert!(toml.contains("[Gpu.PullBack]\n"));
    }
}
