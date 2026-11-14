//! M9h acceptance (docs/design.md, "M9h High-order performance", bullets 2, 3, 4 and 6, read as
//! in ADR-0042, items 1, 6, 7, 11 and 12): the committed degree-sweep data under
//! `docs/perf/data/m9h-*.json`, the degree policy `gridcl-geometry/degree-policy.toml` and the
//! report `docs/perf/high-order.md`.
//!
//! The checks are written here independently of gridcl-bench's decision code: they read the
//! policy with the `toml` crate and recompute class ratios from the raw records.
//!
//! **Policy layout** (read by gridcl-geometry from T4): one table per device class and family,
//! one key per cell type, and per scalar size in bytes a list of `[first degree, strategy]`
//! pairs with ascending first degrees, starting at 1; each range runs to the next first degree,
//! and the last is open-ended:
//!
//! ```toml
//! [Gpu.Jacobian]
//! Hexahedron = { 4 = [[1, "direct"], [5, "sumfact"]], 8 = [[1, "direct"], [3, "sumfact"]] }
//! ```
//!
//! **Targets** (as M3s, ADR-0034): the local Mac (`macmini`: CPU as CpuSimd128 and Metal as
//! Gpu), Locust (Grace CPU as CpuSimd128, H100 as Gpu) and one Kathleen socket (CpuSimd512; a
//! Myriad D node as the fallback). The degree sweep leaves out the Grace CPU, which keeps T3's
//! baseline and rule 11's paired check (ADR-0051). Times are per quadrature point (Gauss degree
//! 2p) on the equispaced meshes, at 10^6 and 10^7 points on the Gpu class and at 10^6 points on
//! the CPU classes (ADR-0052); classes with several devices use the geometric mean of the
//! per-device ratios (ADR-0015, item 5).

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use gridcl_bench::record::{
    DEGREE_FAMILIES, DegreeRecord, HighOrderData, NodeLoad, STRATEGIES, Scalar, load_high_order_dir,
};
use gridcl_bench::report::{BEGIN_MARKER, END_MARKER};
use gridcl_device::DeviceClass;

/// The baseline's problem sizes, in quadrature points (rule 11).
const BASELINE_SIZES: [usize; 2] = [1_000_000, 10_000_000];

/// The problem sizes of the rules of a class, in quadrature points (rule 12): 10^6 and 10^7 on
/// the Gpu class, 10^6 on the CPU classes (ADR-0052).
fn rule_sizes(class: DeviceClass) -> &'static [usize] {
    if class == DeviceClass::Gpu {
        &[1_000_000, 10_000_000]
    } else {
        &[1_000_000]
    }
}

/// The strategies the policy of a class chooses among: all on the Gpu class; `direct`, `loops`
/// and `sumfact` on the CPU classes, where `shared` and `cooperative` reduce to `loops`
/// (ADR-0052).
fn candidates(class: DeviceClass) -> &'static [&'static str] {
    if class == DeviceClass::Gpu {
        &STRATEGIES
    } else {
        &["direct", "loops", "sumfact"]
    }
}

/// Whether the degree sweep measures `strategy` on `cell` at `degree`: not `direct` on
/// tetrahedra, hexahedra and prisms at p ≥ 5 (ADR-0050).
fn swept(cell: &str, degree: u32, strategy: &str) -> bool {
    !(strategy == "direct" && degree >= 5 && matches!(cell, "Tetrahedron" | "Hexahedron" | "Prism"))
}

/// Every cell type, by its `CellType` name.
const CELLS: [&str; 7] = [
    "Interval",
    "Triangle",
    "Quadrilateral",
    "Tetrahedron",
    "Hexahedron",
    "Prism",
    "Pyramid",
];

/// The classes the policy holds; CpuSimd256 uses the CpuSimd512 table ("Benchmarks").
const POLICY_CLASSES: [DeviceClass; 4] = [
    DeviceClass::CpuSimd128,
    DeviceClass::CpuSimd256,
    DeviceClass::CpuSimd512,
    DeviceClass::Gpu,
];

/// The measured classes.
const MEASURED_CLASSES: [DeviceClass; 3] = [
    DeviceClass::CpuSimd128,
    DeviceClass::CpuSimd512,
    DeviceClass::Gpu,
];

/// The scalar types and their sizes in bytes.
const SCALARS: [(Scalar, usize); 2] = [(Scalar::F32, 4), (Scalar::F64, 8)];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn read(path: &str) -> String {
    let path = root().join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every M9h record of `docs/perf/data/`; panics with "no m9h data" if there is no
/// `m9h-*.json` file.
fn data() -> HighOrderData {
    let dir = root().join("docs/perf/data");
    let any = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .any(|name| name.starts_with("m9h-") && name.ends_with(".json"));
    assert!(any, "no m9h data (m9h-*.json) under docs/perf/data/");
    load_high_order_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

// ---------------------------------------------------------------------------------------------
// Families, cells, targets.

fn tdim(cell: &str) -> usize {
    match cell {
        "Interval" => 1,
        "Triangle" | "Quadrilateral" => 2,
        _ => 3,
    }
}

/// The cells a bench family is measured on: `Normals` on manifolds (intervals, triangles and
/// quadrilaterals one dimension up), every cell type otherwise.
fn cells_of(family: &str) -> &'static [&'static str] {
    if family == "Normals" {
        &CELLS[..3]
    } else {
        &CELLS
    }
}

/// gdim of a bench family's mesh: tdim, one more for `Normals`.
fn gdim_of(family: &str, cell: &str) -> usize {
    tdim(cell) + usize::from(family == "Normals")
}

/// The highest degree measured: 8, and 3 on pyramids (ADR-0041).
fn max_degree(cell: &str) -> u32 {
    if cell == "Pyramid" { 3 } else { 8 }
}

/// The baseline degrees of a (target, backend, family, cell): p = 1–8 (pyramids 1–3), except
/// on Locust for the pull-back families (PullBack, BasisEval) and on the Grace CPU, which sample
/// p ∈ {1, 2, 4, 8} (ADR-0043).
fn baseline_degrees(target: &str, backend: &str, family: &str, cell: &str) -> Vec<u32> {
    let sampled =
        target == "locust" && (backend == "cpu" || matches!(family, "PullBack" | "BasisEval"));
    (1..=max_degree(cell))
        .filter(|p| !sampled || [1, 2, 4, 8].contains(p))
        .collect()
}

/// The policy family of a bench family: J, det J and K share `Jacobian`; the basis evaluator is
/// the pull-back's.
fn policy_family(family: &str) -> &str {
    match family {
        "Jacobian" | "DetJ" | "Pinv" => "Jacobian",
        "BasisEval" => "PullBack",
        f => f,
    }
}

/// The scalar types a backend measures: Metal has no f64.
fn scalars_of(backend: &str) -> Vec<Scalar> {
    if backend == "metal" {
        vec![Scalar::F32]
    } else {
        vec![Scalar::F32, Scalar::F64]
    }
}

/// The required (target, backend, class) of `records`: CpuSimd512 on Kathleen, or on a Myriad
/// D node where Kathleen has no record (ADR-0016). The baseline and the paired runs use these.
fn required(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, DeviceClass)> {
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
        ("locust", "cpu", DeviceClass::CpuSimd128),
        ("locust", "cuda", DeviceClass::Gpu),
        (avx512, "cpu", DeviceClass::CpuSimd512),
    ]
}

/// The (target, backend, class) of the degree sweep: [`required`] without the Grace CPU
/// (ADR-0051).
fn sweep_targets(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, DeviceClass)> {
    required(records)
        .into_iter()
        .filter(|&(target, backend, _)| (target, backend) != ("locust", "cpu"))
        .collect()
}

/// The devices among `targets` of `class` that measure `scalar`.
fn devices(
    targets: Vec<(&'static str, &'static str, DeviceClass)>,
    class: DeviceClass,
    scalar: Scalar,
) -> Vec<(&'static str, &'static str)> {
    targets
        .into_iter()
        .filter(|&(_, backend, c)| c == class && scalars_of(backend).contains(&scalar))
        .map(|(t, b, _)| (t, b))
        .collect()
}

/// The geometric mean.
fn geomean(v: &[f64]) -> f64 {
    assert!(!v.is_empty(), "geometric mean of nothing");
    (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp()
}

/// Both node-load snapshots of a Locust run are `idle` (ADR-0002, ADR-0005).
fn check_idle(r: &DegreeRecord, what: &str) {
    if r.target != "locust" || r.launch_only {
        return;
    }
    let load: &NodeLoad = r
        .node_load
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: a Locust record without node-load snapshots"));
    assert!(
        load.before == "idle" && load.after == "idle",
        "{what}: Locust was not idle (before: {}; after: {})",
        load.before,
        load.after
    );
}

fn describe(r: &DegreeRecord) -> String {
    format!(
        "{} {} {} {} gdim {} p={} {} {} {} (ran {}) n={}{}",
        r.target,
        r.backend,
        r.family,
        r.cell,
        r.gdim,
        r.degree,
        r.node_family,
        r.scalar.name(),
        r.strategy,
        r.strategy_ran,
        r.nominal_points,
        r.iterations
            .map(|i| format!(" iterations {i}"))
            .unwrap_or_default()
    )
}

/// The protocol of every timed record: release builds, ten timed runs (five or ten on CPUs,
/// ADR-0052), idle Locust, a fallback reason on Myriad, one Kathleen socket (ADR-0020).
fn check_protocol(records: &[DegreeRecord]) {
    for r in records {
        let what = describe(r);
        assert_eq!(r.profile, "release", "{what}: record release builds only");
        if r.launch_only || r.status != "ok" {
            continue;
        }
        if r.backend == "cpu" {
            assert!(
                matches!(r.times_s.len(), 5 | 10),
                "{what}: five or ten timed runs"
            );
        } else {
            assert_eq!(r.times_s.len(), 10, "{what}: ten timed runs");
        }
        check_idle(r, &what);
        if r.target == "kathleen" {
            assert_eq!(
                r.cores.threads, 20,
                "{what}: not bound to one socket (ADR-0020)"
            );
        }
        if r.target.starts_with("myriad-") {
            assert!(
                r.fallback_reason.as_deref().is_some_and(|s| !s.is_empty()),
                "{what}: a Myriad record needs a fallback_reason (ADR-0016)"
            );
        }
    }
}

/// A timed configuration: (target, backend, family, cell, gdim, degree, scalar, size,
/// iterations).
#[derive(Clone, Copy, Debug)]
struct Config<'a> {
    target: &'a str,
    backend: &'a str,
    family: &'a str,
    cell: &'a str,
    degree: u32,
    scalar: Scalar,
    size: usize,
    iterations: Option<u32>,
}

impl Config<'_> {
    fn matches(&self, r: &DegreeRecord) -> bool {
        r.target == self.target
            && r.backend == self.backend
            && r.family == self.family
            && r.cell == self.cell
            && r.gdim == gdim_of(self.family, self.cell)
            && r.degree == self.degree
            && r.scalar == self.scalar
            && r.nominal_points == self.size
            && r.iterations == self.iterations
            && r.node_family == "equispaced"
            && !r.launch_only
    }
}

/// The iteration counts of a bench family's records: 1 and 9 for the pull-back (rule 1).
fn iterations_of(family: &str) -> Vec<Option<u32>> {
    if family == "PullBack" {
        vec![Some(1), Some(9)]
    } else {
        vec![None]
    }
}

/// The time per point of the last `ok` record of `c` with `pick`, if any.
fn time(records: &[DegreeRecord], c: Config, pick: impl Fn(&DegreeRecord) -> bool) -> Option<f64> {
    records
        .iter()
        .filter(|r| c.matches(r) && r.status == "ok" && pick(r))
        .last()
        .map(DegreeRecord::time_per_point)
}

// ---------------------------------------------------------------------------------------------
// The degree policy.

/// A policy entry: (class, family, cell, scalar bytes).
type PolicyKey = (String, String, String, usize);

/// `degree-policy.toml`: each entry's ascending (first degree, strategy).
struct Policy(BTreeMap<PolicyKey, Vec<(u32, String)>>);

fn policy() -> Policy {
    let text = read("gridcl-geometry/degree-policy.toml");
    let value: toml::Table = text.parse().expect("degree-policy.toml is TOML");
    let mut out = BTreeMap::new();
    for (class, families) in &value {
        let families = families
            .as_table()
            .unwrap_or_else(|| panic!("[{class}] is not a table"));
        for (family, cells) in families {
            let cells = cells
                .as_table()
                .unwrap_or_else(|| panic!("[{class}.{family}] is not a table"));
            for (cell, sizes) in cells {
                let sizes = sizes
                    .as_table()
                    .unwrap_or_else(|| panic!("{class}.{family}.{cell} is not a table"));
                for (bytes, ranges) in sizes {
                    let at = format!("{class}.{family}.{cell}.{bytes}");
                    let ranges = ranges
                        .as_array()
                        .unwrap_or_else(|| panic!("{at}: not an array"))
                        .iter()
                        .map(|r| {
                            let r = r.as_array().unwrap_or_else(|| panic!("{at}: range {r}"));
                            assert_eq!(r.len(), 2, "{at}: range {r:?}");
                            let first = r[0].as_integer().expect("first degree") as u32;
                            let s = r[1].as_str().expect("strategy").to_owned();
                            assert!(STRATEGIES.contains(&s.as_str()), "{at}: strategy {s}");
                            (first, s)
                        })
                        .collect::<Vec<_>>();
                    let bytes = bytes.parse().unwrap_or_else(|_| panic!("{at}: size"));
                    out.insert((class.clone(), family.clone(), cell.clone(), bytes), ranges);
                }
            }
        }
    }
    Policy(out)
}

impl Policy {
    fn ranges(
        &self,
        class: DeviceClass,
        family: &str,
        cell: &str,
        bytes: usize,
    ) -> &[(u32, String)] {
        let key = (
            format!("{class:?}"),
            family.to_owned(),
            cell.to_owned(),
            bytes,
        );
        self.0
            .get(&key)
            .unwrap_or_else(|| panic!("degree-policy.toml has no entry {key:?}"))
    }

    /// The strategy of `degree`.
    fn strategy(
        &self,
        class: DeviceClass,
        family: &str,
        cell: &str,
        bytes: usize,
        degree: u32,
    ) -> &str {
        self.ranges(class, family, cell, bytes)
            .iter()
            .rev()
            .find(|(first, _)| *first <= degree)
            .map(|(_, s)| s.as_str())
            .unwrap_or_else(|| {
                panic!("{class:?}.{family}.{cell}.{bytes}: no range holds p={degree}")
            })
    }
}

/// The policy families, in the order of `degree-policy.toml`.
const POLICY_FAMILIES: [&str; 6] = [
    "EVector",
    "Points",
    "Jacobian",
    "Normals",
    "FacetNormals",
    "PullBack",
];

// ---------------------------------------------------------------------------------------------
// Tests.

/// T3's drift baseline (rule 11): `direct` at one pinned commit, at p = 1–8 (pyramids 1–3), for
/// every bench family, cell type and scalar type of every target; on Locust sampled for the
/// pull-back families and the Grace CPU (ADR-0043, [`baseline_degrees`]). Failures and "too
/// slow" are records too (not retried).
#[test]
fn baseline_covers_every_target_and_degree() {
    let data = data();
    assert!(
        !data.baseline.is_empty(),
        "no baseline records (m9h-baseline-*.json)"
    );
    check_protocol(&data.baseline);
    let commits: BTreeSet<&str> = data.baseline.iter().map(|r| r.commit.as_str()).collect();
    assert_eq!(
        commits.len(),
        1,
        "baseline records of several commits: {commits:?}"
    );
    for r in &data.baseline {
        assert_eq!(
            r.strategy,
            "direct",
            "{}: the baseline is direct",
            describe(r)
        );
    }
    for (target, backend, class) in required(&data.baseline) {
        for r in data
            .baseline
            .iter()
            .filter(|r| r.target == target && r.backend == backend)
        {
            assert_eq!(r.device_class, class, "{}: device class", describe(r));
        }
        for family in DEGREE_FAMILIES {
            for &cell in cells_of(family) {
                for degree in baseline_degrees(target, backend, family, cell) {
                    for scalar in scalars_of(backend) {
                        for size in BASELINE_SIZES {
                            for iterations in iterations_of(family) {
                                let c = Config {
                                    target,
                                    backend,
                                    family,
                                    cell,
                                    degree,
                                    scalar,
                                    size,
                                    iterations,
                                };
                                assert!(
                                    data.baseline.iter().any(|r| c.matches(r)),
                                    "no baseline record of {c:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Bullet 2 (rule 7): every (family, cell type, degree, precision, strategy) compiles, loads and
/// launches on each backend, at vector size 1 and at the vector policy's size: a
/// `degree_sweep --launch-only` record with status `ok`, which ran the strategy or its fallback
/// `loops`. Not `direct` on tetrahedra, hexahedra and prisms at p ≥ 5 (ADR-0050), nor on the
/// Grace CPU (ADR-0051).
#[test]
fn every_variant_launched() {
    let data = data();
    assert!(
        !data.launch.is_empty(),
        "no launch records (m9h-launch-*.json)"
    );
    check_protocol(&data.launch);
    for (target, backend, _) in sweep_targets(&data.launch) {
        for family in DEGREE_FAMILIES {
            for &cell in cells_of(family) {
                for degree in 1..=max_degree(cell) {
                    for scalar in scalars_of(backend) {
                        for strategy in STRATEGIES {
                            if !swept(cell, degree, strategy) {
                                continue;
                            }
                            let what = format!(
                                "{target} {backend} {family} {cell} p={degree} {} {strategy}",
                                scalar.name()
                            );
                            let records: Vec<&DegreeRecord> = data
                                .launch
                                .iter()
                                .filter(|r| {
                                    r.target == target
                                        && r.backend == backend
                                        && r.family == family
                                        && r.cell == cell
                                        && r.gdim == gdim_of(family, cell)
                                        && r.degree == degree
                                        && r.scalar == scalar
                                        && r.strategy == strategy
                                        && r.launch_only
                                })
                                .collect();
                            assert!(!records.is_empty(), "{what}: never launched");
                            for r in &records {
                                assert_eq!(r.status, "ok", "{}", describe(r));
                                assert!(
                                    r.strategy_ran == strategy || r.strategy_ran == "loops",
                                    "{}: ran {}",
                                    describe(r),
                                    r.strategy_ran
                                );
                            }
                            assert!(
                                records.iter().any(|r| r.vector_size == 1),
                                "{what}: not launched at vector size 1"
                            );
                            assert!(
                                records
                                    .iter()
                                    .any(|r| r.vector_size == r.policy_vector_size),
                                "{what}: not launched at the vector policy's size"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Bullet 2 (rule 7): on Locust, with `CUDA_CACHE_DISABLE=1`, every variant the policy selects
/// for the Gpu class has a record, launched, with a first-use compile and load time of at most
/// 30 s.
#[test]
fn compile_and_load_within_30_s_on_locust() {
    let data = data();
    let policy = policy();
    let locust: Vec<&DegreeRecord> = data
        .launch
        .iter()
        .chain(&data.degree)
        .filter(|r| r.target == "locust" && r.backend == "cuda" && r.strategy != "policy")
        .collect();
    assert!(!locust.is_empty(), "no Locust CUDA records");
    for r in &locust {
        assert_eq!(
            r.kernel_cache,
            "disabled",
            "{}: CUDA_CACHE_DISABLE=1 not set",
            describe(r)
        );
    }
    for family in DEGREE_FAMILIES {
        for &cell in cells_of(family) {
            for degree in 1..=max_degree(cell) {
                for (scalar, bytes) in SCALARS {
                    let s = policy.strategy(
                        DeviceClass::Gpu,
                        policy_family(family),
                        cell,
                        bytes,
                        degree,
                    );
                    let chosen: Vec<&&DegreeRecord> = locust
                        .iter()
                        .filter(|r| {
                            r.family == family
                                && r.cell == cell
                                && r.gdim == gdim_of(family, cell)
                                && r.degree == degree
                                && r.scalar == scalar
                                && r.strategy == s
                        })
                        .collect();
                    let what = format!("{family} {cell} p={degree} {} {s}", scalar.name());
                    assert!(
                        !chosen.is_empty(),
                        "{what}: no Locust record of the Gpu choice"
                    );
                    for r in chosen {
                        assert_eq!(r.status, "ok", "{}: the Gpu choice failed", describe(r));
                        assert!(
                            r.compile_s <= 30.0,
                            "{}: compile and load {:.1} s > 30 s",
                            describe(r),
                            r.compile_s
                        );
                    }
                }
            }
        }
    }
}

/// Bullet 3 (rule 1): on quadrilaterals and hexahedra, the time per point at p = 8 over p = 4
/// of the strategy the policy selects at p = 8, at the class's rule sizes: at most 3
/// for the geometric factors and for the pull-back's set-up per point (where the strategy has
/// one: not `sumfact`), and at most 1.1 · (9/5)^d for a Newton step, with step =
/// (t(9) − t(1)) / 8 and set-up = t(1) − step.
#[test]
fn tensor_cost_ratio_p8_over_p4() {
    let data = data();
    let policy = policy();
    check_protocol(&data.degree);
    let records = &data.degree;
    let factors = ["Points", "Jacobian", "DetJ", "Pinv", "FacetNormals"];
    for class in MEASURED_CLASSES {
        for (scalar, bytes) in SCALARS {
            let devices = devices(sweep_targets(records), class, scalar);
            if devices.is_empty() {
                continue;
            }
            for (cell, d) in [("Quadrilateral", 2), ("Hexahedron", 3)] {
                let mut families: Vec<&str> = factors.to_vec();
                if cell == "Quadrilateral" {
                    families.push("Normals");
                }
                families.push("PullBack");
                for family in families {
                    let s = policy.strategy(class, policy_family(family), cell, bytes, 8);
                    for &size in rule_sizes(class) {
                        let what = format!(
                            "{class:?} {} {family} {cell} {s} at {size} points",
                            scalar.name()
                        );
                        let t = |target, backend, degree, iterations| {
                            let c = Config {
                                target,
                                backend,
                                family,
                                cell,
                                degree,
                                scalar,
                                size,
                                iterations,
                            };
                            time(records, c, |r| r.strategy == s)
                                .unwrap_or_else(|| panic!("{what}: no record of {c:?}"))
                        };
                        if family == "PullBack" {
                            let mut steps = Vec::new();
                            let mut setups = Vec::new();
                            for &(target, backend) in &devices {
                                let step = |p| {
                                    (t(target, backend, p, Some(9))
                                        - t(target, backend, p, Some(1)))
                                        / 8.0
                                };
                                let setup = |p| t(target, backend, p, Some(1)) - step(p);
                                steps.push(step(8) / step(4));
                                if s != "sumfact" {
                                    setups.push(setup(8) / setup(4));
                                }
                            }
                            let bound = 1.1 * (9.0f64 / 5.0).powi(d);
                            let ratio = geomean(&steps);
                            assert!(
                                ratio <= bound,
                                "{what}: Newton step p=8/p=4 = {ratio:.2} > {bound:.2} ({steps:?})"
                            );
                            if !setups.is_empty() {
                                let ratio = geomean(&setups);
                                assert!(
                                    ratio <= 3.0,
                                    "{what}: set-up p=8/p=4 = {ratio:.2} > 3 ({setups:?})"
                                );
                            }
                        } else {
                            let ratios: Vec<f64> = devices
                                .iter()
                                .map(|&(target, backend)| {
                                    t(target, backend, 8, None) / t(target, backend, 4, None)
                                })
                                .collect();
                            let ratio = geomean(&ratios);
                            assert!(
                                ratio <= 3.0,
                                "{what}: p=8/p=4 = {ratio:.2} > 3 ({ratios:?})"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The run of each (target, backend) with the newest paired policy records (rule 11).
fn paired_run<'a>(records: &'a [DegreeRecord], target: &str, backend: &str) -> &'a str {
    records
        .iter()
        .filter(|r| r.target == target && r.backend == backend && r.strategy == "policy")
        .map(|r| r.run_id.as_str())
        .max()
        .unwrap_or_else(|| panic!("{target} {backend}: no paired policy run"))
}

/// Bullet 4 (rule 11): in one job per target, the Grace CPU included (ADR-0051), the policy
/// against `direct` at p ≤ 4, per bench family, cell type, scalar type and the class's rule
/// sizes: the class ratio is at most 1.05.
#[test]
fn no_family_slower_than_m9_at_low_degree() {
    let data = data();
    check_protocol(&data.degree);
    let records = &data.degree;
    for class in MEASURED_CLASSES {
        for (scalar, _) in SCALARS {
            let devices = devices(required(records), class, scalar);
            if devices.is_empty() {
                continue;
            }
            for family in DEGREE_FAMILIES {
                for &cell in cells_of(family) {
                    for degree in 1..=max_degree(cell).min(4) {
                        for &size in rule_sizes(class) {
                            for iterations in iterations_of(family) {
                                let ratios: Vec<f64> = devices
                                    .iter()
                                    .map(|&(target, backend)| {
                                        let run = paired_run(records, target, backend);
                                        let c = Config {
                                            target,
                                            backend,
                                            family,
                                            cell,
                                            degree,
                                            scalar,
                                            size,
                                            iterations,
                                        };
                                        let t = |s: &str| {
                                            time(records, c, |r| r.run_id == run && r.strategy == s)
                                                .unwrap_or_else(|| {
                                                    panic!("no {s} record of {c:?} in run {run}")
                                                })
                                        };
                                        t("policy") / t("direct")
                                    })
                                    .collect();
                                let ratio = geomean(&ratios);
                                assert!(
                                    ratio <= 1.05,
                                    "{class:?} {} {family} {cell} p={degree} n={size} \
                                     {iterations:?}: policy / direct = {ratio:.3} ({ratios:?})",
                                    scalar.name()
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The cells of a Markdown table row, trimmed.
fn row_cells(line: &str) -> Option<Vec<&str>> {
    let line = line.trim();
    (line.starts_with('|') && line.ends_with('|'))
        .then(|| line[1..line.len() - 1].split('|').map(str::trim).collect())
}

/// Bullet 6 (rule 12): `docs/perf/high-order.md` has generated tables, and for every policy
/// entry (family, cell type, class, scalar) and degree range a row that names them, the range
/// (`a–b`, or `a–` for the open-ended last one) and the chosen strategy.
#[test]
fn report_covers_every_family_class_and_range() {
    data(); // The report and the policy come from the data.
    let policy = policy();
    let text = read("docs/perf/high-order.md");
    assert!(
        text.contains(BEGIN_MARKER) && text.contains(END_MARKER),
        "high-order.md has no generated section"
    );
    let rows: Vec<Vec<&str>> = text.lines().filter_map(row_cells).collect();
    for class in POLICY_CLASSES {
        for family in POLICY_FAMILIES {
            for cell in CELLS {
                for (scalar, bytes) in SCALARS {
                    let ranges = policy.ranges(class, family, cell, bytes);
                    for (i, (first, s)) in ranges.iter().enumerate() {
                        let range = match ranges.get(i + 1) {
                            Some((next, _)) => format!("{first}–{}", next - 1),
                            None => format!("{first}–"),
                        };
                        let class = format!("{class:?}");
                        let want = [family, cell, class.as_str(), scalar.name(), &range, s];
                        assert!(
                            rows.iter().any(|r| want.iter().all(|w| r.contains(w))),
                            "high-order.md: no row with {want:?}"
                        );
                    }
                }
            }
        }
    }
}

/// The class ratio of strategy `s` against `base` for one (policy family, cell, degree, scalar)
/// at `size`: the geometric mean over `devices` of the summed member families' times; `None`
/// where a device lacks an `ok` record.
#[allow(clippy::too_many_arguments)]
fn class_ratio(
    records: &[DegreeRecord],
    devices: &[(&str, &str)],
    members: &[(&str, Option<u32>)],
    cell: &str,
    degree: u32,
    scalar: Scalar,
    size: usize,
    s: &str,
    base: &str,
) -> Option<f64> {
    let total = |target, backend, strategy: &str| -> Option<f64> {
        members
            .iter()
            .map(|&(family, iterations)| {
                let c = Config {
                    target,
                    backend,
                    family,
                    cell,
                    degree,
                    scalar,
                    size,
                    iterations,
                };
                time(records, c, |r| r.strategy == strategy)
            })
            .sum()
    };
    let ratios: Option<Vec<f64>> = devices
        .iter()
        .map(|&(t, b)| Some(total(t, b, s)? / total(t, b, base)?))
        .collect();
    ratios.map(|r| geomean(&r))
}

/// Whether strategy `s` is selectable for the Gpu class (rule 7): every Locust CUDA record of
/// its member families at `degree` launched and compiled within 30 s.
fn gpu_selectable(
    data: &HighOrderData,
    members: &[(&str, Option<u32>)],
    cell: &str,
    degree: u32,
    scalar: Scalar,
    s: &str,
) -> bool {
    data.launch.iter().chain(&data.degree).all(|r| {
        !(r.target == "locust"
            && r.backend == "cuda"
            && r.strategy == s
            && r.cell == cell
            && r.degree == degree
            && r.scalar == scalar
            && members.iter().any(|&(f, _)| f == r.family))
            || (r.status == "ok" && r.compile_s <= 30.0)
    })
}

/// Bullet 6 (rule 12): the policy follows the decision rule on the data, per policy family, cell
/// type, measured class, scalar type and degree p = 1–8 (pyramids 1–3):
///
/// - a family's time is the sum of its bench families' (J + det J + K for `Jacobian`; the
///   pull-back at 9 iterations for `PullBack`), on the codimension-0 meshes (codimension 1 for
///   `Normals`);
/// - the sizes are the class's rule sizes, and the strategies its candidates (ADR-0052);
/// - the baseline is `direct`, or `loops` where `direct` is not selectable (a device without an
///   `ok` record at a rule size, as where the sweep leaves `direct` out (ADR-0050); on the Gpu
///   class also a failed launch or a compile over 30 s on Locust);
/// - a strategy is admitted if it is the baseline or selectable and at least 10% faster at every
///   rule size (class ratio ≤ 0.9); among the admitted within 3% of the fastest at every size
///   the simplest wins; if none is, the one with the smallest worst ratio to the fastest;
/// - ranges are maximal runs of equal choices, the last range (p ≥ 9 included) is not `direct`,
///   and CpuSimd256 has CpuSimd512's table.
#[test]
fn degree_policy_follows_the_decision_rule() {
    let data = data();
    let policy = policy();
    let records: Vec<DegreeRecord> = data
        .degree
        .iter()
        .filter(|r| r.strategy != "policy")
        .cloned()
        .collect();
    for ((class, family, cell, bytes), ranges) in &policy.0 {
        let at = format!("{class}.{family}.{cell}.{bytes}");
        assert_eq!(
            ranges.first().map(|r| r.0),
            Some(1),
            "{at}: ranges start at p = 1"
        );
        for w in ranges.windows(2) {
            assert!(w[0].0 < w[1].0, "{at}: ranges not ascending");
            assert!(
                w[0].1 != w[1].1,
                "{at}: two adjacent ranges choose {}",
                w[0].1
            );
        }
        assert_ne!(
            ranges.last().map(|r| r.1.as_str()),
            Some("direct"),
            "{at}: the open-ended last range must compile at p = 9–10 (loops or better)"
        );
        if class == "CpuSimd256" {
            let key = (
                "CpuSimd512".to_owned(),
                family.clone(),
                cell.clone(),
                *bytes,
            );
            assert_eq!(Some(ranges), policy.0.get(&key), "{at}: CpuSimd512's table");
        }
    }
    for class in MEASURED_CLASSES {
        for (scalar, bytes) in SCALARS {
            let devices = devices(sweep_targets(&records), class, scalar);
            let sizes = rule_sizes(class);
            if devices.is_empty() {
                continue;
            }
            for family in POLICY_FAMILIES {
                let members: Vec<(&str, Option<u32>)> = match family {
                    "Jacobian" => vec![("Jacobian", None), ("DetJ", None), ("Pinv", None)],
                    "PullBack" => vec![("PullBack", Some(9))],
                    f => vec![(f, None)],
                };
                for &cell in cells_of(family) {
                    for degree in 1..=max_degree(cell) {
                        let at = format!("{class:?} {family} {cell} p={degree} {}", scalar.name());
                        let ratio = |s: &str, base: &str, size| {
                            class_ratio(
                                &records, &devices, &members, cell, degree, scalar, size, s, base,
                            )
                        };
                        let selectable = |s: &str| {
                            sizes.iter().all(|&n| ratio(s, s, n).is_some())
                                && (class != DeviceClass::Gpu
                                    || gpu_selectable(&data, &members, cell, degree, scalar, s))
                        };
                        let base = if selectable("direct") {
                            "direct"
                        } else {
                            assert!(selectable("loops"), "{at}: neither direct nor loops ran");
                            "loops"
                        };
                        let ratios: Vec<(&str, Vec<f64>)> = candidates(class)
                            .iter()
                            .copied()
                            .filter(|&s| selectable(s))
                            .map(|s| {
                                let r = sizes
                                    .iter()
                                    .map(|&n| ratio(s, base, n).expect("selectable"));
                                (s, r.collect::<Vec<f64>>())
                            })
                            .filter(|(s, r)| *s == base || r.iter().all(|&x| x <= 0.9))
                            .collect();
                        let fastest: Vec<f64> = (0..sizes.len())
                            .map(|i| {
                                ratios
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
                        let want = ratios
                            .iter()
                            .find(|(_, r)| worst(r) <= 1.03)
                            .or_else(|| {
                                ratios
                                    .iter()
                                    .min_by(|a, b| worst(&a.1).total_cmp(&worst(&b.1)))
                            })
                            .map(|(s, _)| *s)
                            .expect("the baseline is admitted");
                        let got = policy.strategy(class, family, cell, bytes, degree);
                        assert_eq!(
                            got, want,
                            "{at}: policy {got}, decision rule {want} ({ratios:?})"
                        );
                    }
                }
            }
        }
    }
}
