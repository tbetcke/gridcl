//! M9h acceptance (docs/design.md, "M9h High-order performance", bullets 2, 3, 4 and 6, read as
//! in ADR-0042, items 1, 6, 7, 11 and 12): the committed degree-sweep data under
//! `docs/perf/data/m9h-*.json`, the M9h degree policy and the report `docs/perf/high-order.md`.
//! From M10 these are history tests (ADR-0058, ADR-0060 items 2 and 4): the M9h policy is
//! frozen in `gridcl-bench/history/degree-policy-pre-m10.toml`, its classes are the
//! [`LegacyClass`]es, and the M10 tests at the end of the file check the live policy.
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

use gridcl_bench::record::LegacyClass;
use gridcl_bench::record::{
    DEGREE_FAMILIES, DegreeRecord, HighOrderData, NodeLoad, STRATEGIES, Scalar, load_high_order_dir,
};
use gridcl_bench::report::{BEGIN_MARKER, END_MARKER};
use gridcl_device::DeviceClass;

/// The baseline's problem sizes, in quadrature points (rule 11).
const BASELINE_SIZES: [usize; 2] = [1_000_000, 10_000_000];

/// The problem sizes of the rules of a class, in quadrature points (rule 12): 10^6 and 10^7 on
/// the Gpu class, 10^6 on the CPU classes (ADR-0052).
fn rule_sizes(class: impl IsGpu) -> &'static [usize] {
    if class.gpu() {
        &[1_000_000, 10_000_000]
    } else {
        &[1_000_000]
    }
}

/// The strategies the policy of a class chooses among: all on the Gpu class; `direct`, `loops`
/// and `sumfact` on the CPU classes, where `shared` and `cooperative` reduce to `loops`
/// (ADR-0052).
fn candidates(class: impl IsGpu) -> &'static [&'static str] {
    if class.gpu() {
        &STRATEGIES
    } else {
        &["direct", "loops", "sumfact"]
    }
}

/// A class of M9h ([`LegacyClass`]) or of M10 ([`DeviceClass`]).
trait IsGpu {
    fn gpu(self) -> bool;
}

impl IsGpu for LegacyClass {
    fn gpu(self) -> bool {
        self == LegacyClass::Gpu
    }
}

impl IsGpu for DeviceClass {
    fn gpu(self) -> bool {
        self == DeviceClass::Gpu
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
const POLICY_CLASSES: [LegacyClass; 4] = [
    LegacyClass::CpuSimd128,
    LegacyClass::CpuSimd256,
    LegacyClass::CpuSimd512,
    LegacyClass::Gpu,
];

/// The measured classes.
const MEASURED_CLASSES: [LegacyClass; 3] = [
    LegacyClass::CpuSimd128,
    LegacyClass::CpuSimd512,
    LegacyClass::Gpu,
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
fn required(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, LegacyClass)> {
    let avx512 = if records
        .iter()
        .any(|r| r.target == "kathleen" && r.backend == "cpu")
    {
        "kathleen"
    } else {
        "myriad-cpu"
    };
    vec![
        ("macmini", "cpu", LegacyClass::CpuSimd128),
        ("macmini", "metal", LegacyClass::Gpu),
        ("locust", "cpu", LegacyClass::CpuSimd128),
        ("locust", "cuda", LegacyClass::Gpu),
        (avx512, "cpu", LegacyClass::CpuSimd512),
    ]
}

/// The (target, backend, class) of the degree sweep: [`required`] without the Grace CPU
/// (ADR-0051).
fn sweep_targets(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, LegacyClass)> {
    required(records)
        .into_iter()
        .filter(|&(target, backend, _)| (target, backend) != ("locust", "cpu"))
        .collect()
}

/// The devices among `targets` of `class` that measure `scalar`.
fn devices(
    targets: Vec<(&'static str, &'static str, LegacyClass)>,
    class: LegacyClass,
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

/// The time per point of the last `ok` record of `c` with `pick`, an interleaved one if there is
/// one (the CPU classes' re-timing at p ≤ 4, ADR-0055), if any.
fn time(records: &[DegreeRecord], c: Config, pick: impl Fn(&DegreeRecord) -> bool) -> Option<f64> {
    let ok: Vec<&DegreeRecord> = records
        .iter()
        .filter(|r| c.matches(r) && r.status == "ok" && pick(r))
        .collect();
    ok.iter()
        .rev()
        .find(|r| r.interleaved)
        .or(ok.last())
        .map(|r| r.time_per_point())
}

/// The strategy sweeps' records: `data.degree` without rule 11's paired runs (the runs with
/// `policy` records), which check the policy and do not decide it.
fn sweep_records(data: &HighOrderData) -> Vec<DegreeRecord> {
    let paired: BTreeSet<&str> = data
        .degree
        .iter()
        .filter(|r| r.strategy == "policy")
        .map(|r| r.run_id.as_str())
        .collect();
    data.degree
        .iter()
        .filter(|r| !paired.contains(r.run_id.as_str()))
        .cloned()
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The degree policy.

/// A policy entry: (class, family, cell, scalar bytes).
type PolicyKey = (String, String, String, usize);

/// `degree-policy.toml`: each entry's ascending (first degree, strategy).
struct Policy(BTreeMap<PolicyKey, Vec<(u32, String)>>);

fn policy() -> Policy {
    policy_at("gridcl-geometry/degree-policy.toml")
}

/// The M9h policy, frozen before M10 (ADR-0060 item 2).
fn history_policy() -> Policy {
    policy_at("gridcl-bench/history/degree-policy-pre-m10.toml")
}

/// The degree policy in the file `path` (relative to the workspace).
fn policy_at(path: &str) -> Policy {
    let text = read(path);
    let value: toml::Table = text
        .parse()
        .unwrap_or_else(|e| panic!("{path} is not TOML: {e}"));
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
        class: LegacyClass,
        family: &str,
        cell: &str,
        bytes: usize,
    ) -> &[(u32, String)] {
        self.ranges_of(&format!("{class:?}"), family, cell, bytes)
    }

    /// The ranges of the class named `class`.
    fn ranges_of(&self, class: &str, family: &str, cell: &str, bytes: usize) -> &[(u32, String)] {
        let key = (class.to_owned(), family.to_owned(), cell.to_owned(), bytes);
        self.0
            .get(&key)
            .unwrap_or_else(|| panic!("degree-policy.toml has no entry {key:?}"))
    }

    /// The strategy of `degree`.
    fn strategy(
        &self,
        class: LegacyClass,
        family: &str,
        cell: &str,
        bytes: usize,
        degree: u32,
    ) -> &str {
        self.strategy_of(&format!("{class:?}"), family, cell, bytes, degree)
    }

    /// The strategy of `degree` in the class named `class`.
    fn strategy_of(
        &self,
        class: &str,
        family: &str,
        cell: &str,
        bytes: usize,
        degree: u32,
    ) -> &str {
        self.ranges_of(class, family, cell, bytes)
            .iter()
            .rev()
            .find(|(first, _)| *first <= degree)
            .map(|(_, s)| s.as_str())
            .unwrap_or_else(|| panic!("{class}.{family}.{cell}.{bytes}: no range holds p={degree}"))
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
            assert_eq!(r.legacy_class, Some(class), "{}: device class", describe(r));
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
    let policy = history_policy();
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
                        LegacyClass::Gpu,
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

/// ADR-0056: no variant the policy selects for a class failed or took over 30 s to compile and
/// load on a sweep device of that class (the Mac mini for CpuSimd128, one Kathleen socket for
/// CpuSimd512, Metal and Locust for Gpu), in any launch-only or timed record of the sweeps.
#[test]
fn policy_choices_first_use_within_30_s_on_every_class() {
    let data = data();
    let policy = history_policy();
    let records = sweep_records(&data);
    for class in MEASURED_CLASSES {
        for (scalar, bytes) in SCALARS {
            let devices = devices(sweep_targets(&records), class, scalar);
            for family in DEGREE_FAMILIES {
                for &cell in cells_of(family) {
                    for degree in 1..=max_degree(cell) {
                        let s = policy.strategy(class, policy_family(family), cell, bytes, degree);
                        let what =
                            format!("{class:?} {family} {cell} p={degree} {} {s}", scalar.name());
                        // The rule as the decision reads it (launch-only and timed records,
                        // the paired runs left out).
                        assert!(
                            first_use_ok(
                                &data,
                                &devices,
                                &[(family, None)],
                                cell,
                                degree,
                                scalar,
                                s
                            ),
                            "{what}: a record of the choice failed or took over 30 s to first use"
                        );
                        // Not vacuous: every device has a record of the choice.
                        for &(target, backend) in &devices {
                            assert!(
                                data.launch.iter().chain(&records).any(|r| {
                                    r.target == target
                                        && r.backend == backend
                                        && r.family == family
                                        && r.cell == cell
                                        && r.degree == degree
                                        && r.scalar == scalar
                                        && r.strategy == s
                                }),
                                "{what}: no record on {target} {backend}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Bullet 3 (rule 1, ADR-0053): on quadrilaterals and hexahedra, the time per point at p = 8
/// over p = 4 of the strategy the policy selects at p = 8, at the class's rule sizes, is reported
/// in `docs/perf/high-order.md` against the bounds of ADR-0042 item 1 (3 for the geometric
/// factors and the pull-back's set-up, every strategy having one, `sumfact`'s sum-factorised,
/// ADR-0049; 1.1 · (9/5)^d for a Newton step), with step = (t(9) − t(1)) / 8 and set-up =
/// t(1) − step. Each ratio, recomputed here from the records, appears in a row with the class,
/// type, cell, family, part, strategy, size and bound; the bound is not enforced in M9h.
#[test]
fn tensor_cost_ratio_reported() {
    let data = data();
    let policy = history_policy();
    check_protocol(&data.degree);
    let records = &sweep_records(&data);
    let text = read("docs/perf/high-order.md");
    let rows: Vec<Vec<&str>> = text.lines().filter_map(row_cells).collect();
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
                            time(records, c, |r| r.strategy == s && r.strategy_ran == s)
                                .unwrap_or_else(|| panic!("{what}: no record of {c:?}"))
                        };
                        let mut parts: Vec<(&str, Vec<f64>, f64)> = Vec::new();
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
                                setups.push(setup(8) / setup(4));
                            }
                            parts.push(("step", steps, 1.1 * (9.0f64 / 5.0).powi(d)));
                            parts.push(("set-up", setups, 3.0));
                        } else {
                            let ratios = devices
                                .iter()
                                .map(|&(target, backend)| {
                                    t(target, backend, 8, None) / t(target, backend, 4, None)
                                })
                                .collect();
                            parts.push(("factor", ratios, 3.0));
                        }
                        for (part, ratios, bound) in parts {
                            let ratio = format!("{:.2}", geomean(&ratios));
                            let bound = format!("{bound:.2}");
                            let class = format!("{class:?}");
                            let points = format!("10^{}", size.ilog10());
                            let want = [
                                class.as_str(),
                                scalar.name(),
                                cell,
                                family,
                                part,
                                s,
                                &points,
                                &ratio,
                                &bound,
                            ];
                            assert!(
                                rows.iter()
                                    .any(|r| r.len() >= want.len() && r[..want.len()] == want),
                                "high-order.md: no row {want:?} ({what}, {ratios:?})"
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

/// The per-run ratios of the summed member times of the policy's record set `a` over
/// `direct`'s `b`: run i of each, which `degree_sweep --interleave` alternates.
fn run_ratios(a: &[&DegreeRecord], b: &[&DegreeRecord]) -> Vec<f64> {
    let n = a[0].times_s.len();
    for r in a.iter().chain(b) {
        assert_eq!(
            r.times_s.len(),
            n,
            "{}: paired run counts differ",
            describe(r)
        );
    }
    (0..n)
        .map(|i| {
            a.iter().map(|r| r.times_s[i]).sum::<f64>()
                / b.iter().map(|r| r.times_s[i]).sum::<f64>()
        })
        .collect()
}

/// The `q`-quantile of `v`, interpolating linearly between order statistics.
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let x = (v.len() - 1) as f64 * q;
    let lo = x.floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    v[lo] + (v[hi] - v[lo]) * (x - lo as f64)
}

/// Whether `direct` is selectable for `class` at one (policy family, cell, degree, scalar), as
/// the decision rule reads it: `ok` records of every member on every sweep device at every rule
/// size, and no failed record or first use over 30 s on the class's sweep devices (rule 7 on
/// Locust; every class, ADR-0056).
fn direct_selectable(
    data: &HighOrderData,
    sweep: &[DegreeRecord],
    class: LegacyClass,
    members: &[(&str, Option<u32>)],
    cell: &str,
    degree: u32,
    scalar: Scalar,
) -> bool {
    let devices = devices(sweep_targets(sweep), class, scalar);
    rule_sizes(class).iter().all(|&n| {
        class_ratio(
            sweep, &devices, members, cell, degree, scalar, n, "direct", "direct",
        )
        .is_some()
    }) && first_use_ok(data, &devices, members, cell, degree, scalar, "direct")
}

/// The members of a policy family: J, det J and K for `Jacobian`, the pull-back at 9 iterations
/// for `PullBack`.
fn members_of(family: &str) -> Vec<(&str, Option<u32>)> {
    match family {
        "Jacobian" => vec![("Jacobian", None), ("DetJ", None), ("Pinv", None)],
        "PullBack" => vec![("PullBack", Some(9))],
        f => vec![(f, None)],
    }
}

/// Bullet 4 (rule 11, read as in ADR-0055): in one paired run per device, the Grace CPU
/// included (ADR-0051), the policy against `direct` at p ≤ 4 per policy family (the members'
/// times summed per run; the pull-back at 9 iterations), cell type, scalar type and rule size,
/// where `direct` is selectable for the class. Each device gives the per-run ratios of the
/// policy's run i over `direct`'s run i; the class's geometric mean of their lower quartiles is at
/// most 1.05 (the policy is not over 5% slower in three quarters of the run pairs).
#[test]
fn no_family_slower_than_m9_at_low_degree() {
    let data = data();
    check_protocol(&data.degree);
    let records = &data.degree;
    let sweep = sweep_records(&data);
    for class in MEASURED_CLASSES {
        for (scalar, _) in SCALARS {
            let devices = devices(required(records), class, scalar);
            if devices.is_empty() {
                continue;
            }
            for family in POLICY_FAMILIES {
                let members = members_of(family);
                for &cell in cells_of(family) {
                    for degree in 1..=max_degree(cell).min(4) {
                        // The whole data: `first_use_ok` leaves the paired runs out itself,
                        // from the launch records too, as the decision rule does.
                        if !direct_selectable(&data, &sweep, class, &members, cell, degree, scalar)
                        {
                            continue;
                        }
                        for &size in rule_sizes(class) {
                            let lowers: Vec<f64> = devices
                                .iter()
                                .map(|&(target, backend)| {
                                    let run = paired_run(records, target, backend);
                                    let get = |s: &str| -> Vec<&DegreeRecord> {
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
                                                records
                                                    .iter()
                                                    .rev()
                                                    .find(|r| {
                                                        c.matches(r)
                                                            && r.status == "ok"
                                                            && r.run_id == run
                                                            && r.strategy == s
                                                    })
                                                    .unwrap_or_else(|| {
                                                        panic!(
                                                            "no {s} record of {c:?} in run {run}"
                                                        )
                                                    })
                                            })
                                            .collect()
                                    };
                                    quantile(&run_ratios(&get("policy"), &get("direct")), 0.25)
                                })
                                .collect();
                            let lower = geomean(&lowers);
                            assert!(
                                lower <= 1.05,
                                "{class:?} {} {family} {cell} p={degree} n={size}: lower quartile \
                                 of policy / direct = {lower:.3} ({lowers:?})",
                                scalar.name()
                            );
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
    let policy = history_policy();
    let text = read("docs/perf/high-order.md");
    assert!(
        text.contains(BEGIN_MARKER) && text.contains(END_MARKER),
        "high-order.md has no generated section"
    );
    let rows: Vec<Vec<&str>> = text.lines().filter_map(row_cells).collect();
    for class in POLICY_CLASSES {
        for family in POLICY_FAMILIES {
            for &cell in cells_of(family) {
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
                time(records, c, |r| {
                    r.strategy == strategy && r.strategy_ran == strategy
                })
            })
            .sum()
    };
    let ratios: Option<Vec<f64>> = devices
        .iter()
        .map(|&(t, b)| Some(total(t, b, s)? / total(t, b, base)?))
        .collect();
    ratios.map(|r| geomean(&r))
}

/// Whether strategy `s` passes the first-use rule on a class's `devices` (rule 7 on Locust;
/// every class's sweep devices, ADR-0056): every record of its member families at `degree`
/// on those devices, launch-only or timed and outside the paired runs, launched and compiled
/// within 30 s.
fn first_use_ok(
    data: &HighOrderData,
    devices: &[(&str, &str)],
    members: &[(&str, Option<u32>)],
    cell: &str,
    degree: u32,
    scalar: Scalar,
    s: &str,
) -> bool {
    let paired: BTreeSet<&str> = data
        .degree
        .iter()
        .filter(|r| r.strategy == "policy")
        .map(|r| r.run_id.as_str())
        .collect();
    data.launch.iter().chain(&data.degree).all(|r| {
        !(devices.contains(&(r.target.as_str(), r.backend.as_str()))
            && !paired.contains(r.run_id.as_str())
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
///   `Normals`), from records that ran the strategy itself, not its fallback `loops`;
/// - the sizes are the class's rule sizes, and the strategies its candidates (ADR-0052);
/// - the baseline is `direct`, or `loops` where `direct` is not selectable (a device without an
///   `ok` record at a rule size, as where the sweep leaves `direct` out (ADR-0050); also a
///   failed record or a compile over 30 s on a sweep device of the class, ADR-0056);
/// - a strategy is admitted if it is the baseline or selectable and at least 10% faster at every
///   rule size (class ratio ≤ 0.9); among the admitted within 3% of the fastest at every size
///   the simplest wins; if none is, the one with the smallest worst ratio to the fastest;
/// - ranges are maximal runs of equal choices, the last range (p ≥ 9 included) is not `direct`,
///   and CpuSimd256 has CpuSimd512's table.
#[test]
fn degree_policy_follows_the_decision_rule() {
    let data = data();
    let policy = history_policy();
    // `first_use_ok` gets the whole data and leaves the paired runs out itself, from the launch
    // records too, as the decision rule does.
    let records = sweep_records(&data);
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
                                && first_use_ok(&data, &devices, &members, cell, degree, scalar, s)
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

// =============================================================================================
// M10 (docs/design.md, Roadmap M10, bullets 2 to 5, read as in ADR-0057, ADR-0058 and ADR-0060):
// the degree check at p = 1, 4 and 8 under `docs/perf/data/m10-degree-*.json`, the generic degree
// policy and M10's section of `docs/perf/high-order.md`.

use gridcl_bench::candidates::degree_candidates;
use gridcl_bench::decide_generic::decide_degree;
use gridcl_bench::record::{M10Data, load_m10_dir};
use gridcl_bench::report::{M10_BEGIN_MARKER, M10_END_MARKER};

/// The degrees M10 measures (ADR-0057, ADR-0058 item 5).
const M10_DEGREES: [u32; 3] = [1, 4, 8];

/// The rule sizes of the candidate protocol, on every platform (ADR-0057 item 3).
const M10_SIZES: [usize; 2] = [1_000_000, 10_000_000];

/// The rule sizes of a (family, cell, scalar, degree): [`M10_SIZES`], except FacetNormals on
/// intervals in f32, at sizes f32 can resolve: 10^5 and 10^6 points at p = 4–6, 10^4 and 10^5
/// at p = 7–8 (ADR-0063).
fn m10_sizes(family: &str, cell: &str, scalar: Scalar, degree: u32) -> [usize; 2] {
    match (family, cell, scalar, degree) {
        ("FacetNormals", "Interval", Scalar::F32, 4..=6) => [100_000, 1_000_000],
        ("FacetNormals", "Interval", Scalar::F32, 7..) => [10_000, 100_000],
        _ => M10_SIZES,
    }
}

/// Timed runs of the candidate protocol.
const M10_RUNS: usize = 5;

fn m10() -> M10Data {
    let dir = root().join("docs/perf/data");
    load_m10_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// The M10 degree records; panics if there are none.
fn m10_degree(data: &M10Data) -> &[DegreeRecord] {
    assert!(
        !data.degree.is_empty(),
        "no M10 degree records (m10-degree-*.json) under docs/perf/data/"
    );
    &data.degree
}

/// The `DeviceClass` named `name` (`Cpu` or `Gpu`).
fn device_class(name: &str) -> DeviceClass {
    gridcl_bench::record::class_named(name).unwrap_or_else(|| panic!("no device class {name}"))
}

/// The name of a record's class, `Cpu` or `Gpu`.
fn class_name(class: &impl std::fmt::Debug) -> String {
    format!("{class:?}").trim_matches('"').to_owned()
}

/// M10's platforms (ADR-0058 item 3) as (target, backend, class), Grace included; a Myriad D
/// node for Kathleen and Myriad's A100 or V100 for the H100 only where the primary target has
/// no record (ADR-0014, ADR-0016).
fn m10_platforms(records: &[DegreeRecord]) -> Vec<(&'static str, &'static str, &'static str)> {
    let present = |t: &str, b: &str| records.iter().any(|r| r.target == t && r.backend == b);
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
    vec![
        ("macmini", "cpu", "Cpu"),
        ("macmini", "metal", "Gpu"),
        ("locust", "cpu", "Cpu"),
        (
            pick("locust", "cuda", &["myriad-a100", "myriad-v100"]),
            "cuda",
            "Gpu",
        ),
        (pick("kathleen", "cpu", &["myriad-cpu"]), "cpu", "Cpu"),
    ]
}

/// The platforms of `class` that measure `scalar`.
fn m10_devices(
    records: &[DegreeRecord],
    class: &str,
    scalar: Scalar,
) -> Vec<(&'static str, &'static str)> {
    m10_platforms(records)
        .into_iter()
        .filter(|&(_, b, c)| c == class && scalars_of(b).contains(&scalar))
        .map(|(t, b, _)| (t, b))
        .collect()
}

/// The M10 degrees of a cell: p = 1, 4 and 8, pyramids at p = 1 only.
fn m10_degrees(cell: &str) -> Vec<u32> {
    if cell == "Pyramid" {
        vec![1]
    } else {
        M10_DEGREES.to_vec()
    }
}

/// The candidate protocol of every M10 degree record: release builds, five timed runs,
/// interleaved on CPUs (ADR-0055), the class of its backend, `load_width` and the core counts
/// (ADR-0060 item 2), idle Locust, one Kathleen socket, a fallback reason on Myriad, no A4000.
fn check_m10_protocol(records: &[DegreeRecord]) {
    for r in records {
        let what = describe(r);
        assert_eq!(r.profile, "release", "{what}: record release builds only");
        let class = if r.backend == "cpu" { "Cpu" } else { "Gpu" };
        assert_eq!(class_name(&r.device_class), class, "{what}: device class");
        assert!(r.load_width.is_some(), "{what}: no load_width");
        assert_ne!(
            r.target, "a4000",
            "{what}: the A4000 does not vote (ADR-0037)"
        );
        if r.target.starts_with("myriad-") {
            assert!(
                r.fallback_reason.as_deref().is_some_and(|s| !s.is_empty()),
                "{what}: a Myriad record needs a fallback_reason (ADR-0016)"
            );
        }
        if r.launch_only || r.status != "ok" {
            continue;
        }
        assert_eq!(r.times_s.len(), M10_RUNS, "{what}: five timed runs");
        if r.backend == "cpu" {
            assert!(r.interleaved, "{what}: CPU runs are interleaved (ADR-0055)");
        }
        check_idle(r, &what);
        if r.target == "kathleen" {
            assert_eq!(
                r.cores.threads, 20,
                "{what}: not bound to one socket (ADR-0020)"
            );
        }
    }
}

/// The old classes whose M9h choices a Cpu or Gpu entry measures (CpuSimd256 had CpuSimd512's
/// table).
fn old_classes(class: &str) -> &'static [&'static str] {
    if class == "Cpu" {
        &["CpuSimd128", "CpuSimd512"]
    } else {
        &["Gpu"]
    }
}

/// The baseline strategy at `degree`: `direct`, or `loops` where the sweep leaves `direct` out
/// (ADR-0050).
fn baseline(cell: &str, degree: u32) -> &'static str {
    if swept(cell, degree, "direct") {
        "direct"
    } else {
        "loops"
    }
}

/// Bullet 4 (ADR-0057 item 3, ADR-0060 items 2 and 3): every degree-dependent bench family, cell
/// type and scalar type is measured on every M10 platform at p = 1, 4 and 8 (pyramids at p = 1)
/// and at 10^6 and 10^7 points (FacetNormals on intervals in f32 at [`m10_sizes`], ADR-0063),
/// with its candidates: the baseline, the M9h choices of the old
/// classes of its class (`history/degree-policy-pre-m10.toml`) and every strategy of
/// `candidates::degree_candidates`. A failed or too slow strategy is a record too.
#[test]
fn m10_degree_check_covers_p1_4_8_on_every_platform() {
    let data = m10();
    let records = m10_degree(&data);
    let history = policy_at("gridcl-bench/history/degree-policy-pre-m10.toml");
    check_m10_protocol(records);
    for (target, backend, class) in m10_platforms(records) {
        for scalar in scalars_of(backend) {
            let bytes = scalar.bytes();
            for family in DEGREE_FAMILIES {
                let pf = policy_family(family);
                for &cell in cells_of(family) {
                    for degree in m10_degrees(cell) {
                        let mut want: Vec<&str> = vec![baseline(cell, degree)];
                        for old in old_classes(class) {
                            want.push(history.strategy_of(old, pf, cell, bytes, degree));
                        }
                        want.extend(degree_candidates(
                            device_class(class),
                            pf,
                            cell,
                            bytes,
                            degree,
                        ));
                        for size in m10_sizes(family, cell, scalar, degree) {
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
                                for &s in &want {
                                    assert!(
                                        records.iter().any(|r| c.matches(r) && r.strategy == s),
                                        "{target} {backend} ({class}): no {s} record of {c:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The time per point of the summed `members` with strategy `s` on one platform at `size`, from
/// `ok` records that ran `s` itself (`strategy_ran`), if every member has one.
#[allow(clippy::too_many_arguments)]
fn m10_time(
    records: &[DegreeRecord],
    (target, backend): (&str, &str),
    members: &[(&str, Option<u32>)],
    cell: &str,
    degree: u32,
    scalar: Scalar,
    size: usize,
    s: &str,
) -> Option<f64> {
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
            time(records, c, |r| r.strategy == s && r.strategy_ran == s)
        })
        .sum()
}

/// The bench families a policy family's time sums: J + det J + K for `Jacobian`, the pull-back
/// at nine iterations for `PullBack`.
fn policy_members(family: &str) -> Vec<(&str, Option<u32>)> {
    match family {
        "Jacobian" => vec![("Jacobian", None), ("DetJ", None), ("Pinv", None)],
        "PullBack" => vec![("PullBack", Some(9))],
        f => vec![(f, None)],
    }
}

/// M10's generated section of `docs/perf/high-order.md`.
fn m10_high_order_section() -> String {
    let text = read("docs/perf/high-order.md");
    let begin = text
        .find(M10_BEGIN_MARKER)
        .expect("docs/perf/high-order.md has no M10 BEGIN marker")
        + M10_BEGIN_MARKER.len();
    let end = text[begin..]
        .find(M10_END_MARKER)
        .expect("docs/perf/high-order.md has no M10 END marker")
        + begin;
    text[begin..end].trim().to_owned()
}

/// Bullet 5 (ADR-0058, ADR-0060 item 3): `degree-policy.toml` holds `[Cpu.<family>]` and
/// `[Gpu.<family>]` only, is what `decide_generic` writes from M10's records at p = 1, 4 and 8
/// and M9h's in between, its ranges are maximal runs starting at p = 1 whose last is not
/// `direct`, and at p = 1, 4 and 8 every entry follows the generic rule, recomputed here from
/// the M10 records: a family's time as in M9h (J + det J + K; the pull-back at nine iterations;
/// records that ran the strategy itself); a strategy that failed or took over 30 s to compile
/// on a platform of the class is not selectable (ADR-0056); admission at 10% over the baseline
/// by the geometric mean over the platforms at each size; the bound r ≤ 1.25 on every platform
/// and size; the smallest geometric mean of r, the simplest within 3%; otherwise the smallest
/// worst case, which the report marks.
#[test]
fn degree_policy_follows_the_generic_rule() {
    let data = m10();
    let records = m10_degree(&data);
    let m9h = self::data();
    let text = read("gridcl-geometry/degree-policy.toml");
    assert_eq!(
        text,
        decide_degree(&m9h, &data).to_toml(),
        "degree-policy.toml differs from the generic rule: run high_order_report"
    );
    let policy = policy();
    let section = m10_high_order_section();
    let classes: BTreeSet<&str> = policy.0.keys().map(|k| k.0.as_str()).collect();
    assert_eq!(
        classes,
        BTreeSet::from(["Cpu", "Gpu"]),
        "degree-policy.toml classes"
    );
    for ((class, family, cell, bytes), ranges) in &policy.0 {
        let at = format!("{class}.{family}.{cell}.{bytes}");
        assert!([4, 8].contains(bytes), "{at}: scalar size");
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
    }
    let geomean = |v: &[f64]| (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp();
    for class in ["Cpu", "Gpu"] {
        for (scalar, bytes) in SCALARS {
            let devices = m10_devices(records, class, scalar);
            if devices.is_empty() {
                continue;
            }
            for family in POLICY_FAMILIES {
                let members = policy_members(family);
                for &cell in cells_of(family) {
                    for degree in m10_degrees(cell) {
                        let at = format!("{class} {family} {cell} p={degree} {}", scalar.name());
                        let sizes = m10_sizes(family, cell, scalar, degree);
                        let t = |d: (&str, &str), size, s: &str| {
                            m10_time(records, d, &members, cell, degree, scalar, size, s)
                        };
                        let first_use = |s: &str| {
                            records.iter().all(|r| {
                                !(devices.contains(&(r.target.as_str(), r.backend.as_str()))
                                    && r.strategy == s
                                    && r.cell == cell
                                    && r.degree == degree
                                    && r.scalar == scalar
                                    && members.iter().any(|&(f, _)| f == r.family))
                                    || (r.status == "ok" && r.compile_s <= 30.0)
                            })
                        };
                        let selectable = |s: &str| {
                            first_use(s)
                                && devices
                                    .iter()
                                    .all(|&d| sizes.iter().all(|&n| t(d, n, s).is_some()))
                        };
                        let cand: Vec<&str> = candidates(device_class(class))
                            .iter()
                            .copied()
                            .filter(|&s| selectable(s))
                            .collect();
                        let base = if cand.contains(&"direct") {
                            "direct"
                        } else {
                            assert!(
                                cand.contains(&"loops"),
                                "{at}: neither direct nor loops ran"
                            );
                            "loops"
                        };
                        // r(s, d, n), in the order of `devices` × `sizes`.
                        let r = |s: &str| -> Vec<f64> {
                            let mut out = Vec::new();
                            for &d in &devices {
                                for &n in &sizes {
                                    let fastest = cand
                                        .iter()
                                        .map(|&c| t(d, n, c).unwrap())
                                        .fold(f64::INFINITY, f64::min);
                                    out.push(t(d, n, s).unwrap() / fastest);
                                }
                            }
                            out
                        };
                        let admitted: Vec<&str> = cand
                            .iter()
                            .copied()
                            .filter(|&s| {
                                s == base
                                    || sizes.iter().all(|&n| {
                                        let v: Vec<f64> = devices
                                            .iter()
                                            .map(|&d| t(d, n, s).unwrap() / t(d, n, base).unwrap())
                                            .collect();
                                        geomean(&v) <= 1.0 / 1.10 * (1.0 + 1e-12)
                                    })
                            })
                            .collect();
                        let worst = |s: &str| r(s).into_iter().fold(0.0, f64::max);
                        let kept: Vec<&str> = admitted
                            .iter()
                            .copied()
                            .filter(|&s| worst(s) <= 1.25 * (1.0 + 1e-12))
                            .collect();
                        let rank = |s: &str| STRATEGIES.iter().position(|&x| x == s).unwrap();
                        let (want, fallback) = if kept.is_empty() {
                            let s = admitted
                                .iter()
                                .copied()
                                .min_by(|&a, &b| {
                                    worst(a).total_cmp(&worst(b)).then(rank(a).cmp(&rank(b)))
                                })
                                .expect("the baseline is admitted");
                            (s, true)
                        } else {
                            let g = |s: &str| geomean(&r(s));
                            let best = kept.iter().map(|&s| g(s)).fold(f64::INFINITY, f64::min);
                            let s = kept
                                .iter()
                                .copied()
                                .filter(|&s| g(s) <= 1.03 * best * (1.0 + 1e-12))
                                .min_by_key(|&s| rank(s))
                                .unwrap();
                            (s, false)
                        };
                        let got = policy.strategy_of(class, family, cell, bytes, degree);
                        assert_eq!(got, want, "{at}: policy {got}, generic rule {want}");
                        if fallback {
                            let prefix = format!("| {class} | {family} | {cell} | {bytes} |");
                            assert!(
                                section
                                    .lines()
                                    .any(|l| l.starts_with(&prefix) && l.contains("worst case")),
                                "{at}: no strategy meets the bound, and high-order.md does not \
                                 mark the entry with its worst case"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Bullet 2 (ADR-0042 item 1 as amended by ADR-0064, ADR-0060 item 5): on quadrilaterals and
/// hexahedra, per platform, scalar type, family and rule size, the time per point at p = 8 over
/// p = 4 under the generic policy (each degree's own choice) is at most its bound (3 for the
/// factors and the pull-back's set-up; 1.1 · (9/5)^d for a Newton step, step = (t(9) − t(1)) /
/// 8, set-up = t(1) − step), or the gap is reported with its cause. Every ratio, recomputed here, appears in
/// M10's section of `docs/perf/high-order.md` as a row `| target backend | scalar | cell |
/// family | part | s4 / s8 | 10^k | ratio | bound | cause |`, with a cause for each miss.
#[test]
fn tensor_cost_ratio_within_bound_or_reported() {
    let data = m10();
    let records = m10_degree(&data);
    let policy = policy();
    let section = m10_high_order_section();
    let rows: Vec<Vec<&str>> = section.lines().filter_map(row_cells).collect();
    let factors = ["Points", "Jacobian", "DetJ", "Pinv", "FacetNormals"];
    for (target, backend, class) in m10_platforms(records) {
        for scalar in scalars_of(backend) {
            let bytes = scalar.bytes();
            for (cell, d) in [("Quadrilateral", 2), ("Hexahedron", 3)] {
                let mut families: Vec<&str> = factors.to_vec();
                if cell == "Quadrilateral" {
                    families.push("Normals");
                }
                families.push("PullBack");
                for family in families {
                    let pf = policy_family(family);
                    let s4 = policy.strategy_of(class, pf, cell, bytes, 4);
                    let s8 = policy.strategy_of(class, pf, cell, bytes, 8);
                    for size in M10_SIZES {
                        let what = format!(
                            "{target} {backend} {} {family} {cell} {s4}/{s8} at {size} points",
                            scalar.name()
                        );
                        let t = |degree, iterations| {
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
                            let s = if degree == 4 { s4 } else { s8 };
                            time(records, c, |r| r.strategy == s)
                                .unwrap_or_else(|| panic!("{what}: no record of {c:?} with {s}"))
                        };
                        let mut parts: Vec<(&str, f64, f64)> = Vec::new();
                        if family == "PullBack" {
                            let step = |p| (t(p, Some(9)) - t(p, Some(1))) / 8.0;
                            let setup = |p| t(p, Some(1)) - step(p);
                            parts.push(("step", step(8) / step(4), 1.1 * (9.0f64 / 5.0).powi(d)));
                            parts.push(("set-up", setup(8) / setup(4), 3.0));
                        } else {
                            parts.push(("factor", t(8, None) / t(4, None), 3.0));
                        }
                        for (part, ratio, bound) in parts {
                            let platform = format!("{target} {backend}");
                            let strategies = format!("{s4} / {s8}");
                            let points = format!("10^{}", size.ilog10());
                            let ratio_s = format!("{ratio:.2}");
                            let bound_s = format!("{bound:.2}");
                            let want = [
                                platform.as_str(),
                                scalar.name(),
                                cell,
                                family,
                                part,
                                &strategies,
                                &points,
                                &ratio_s,
                                &bound_s,
                            ];
                            let row = rows
                                .iter()
                                .find(|r| r.len() >= 10 && r[..9] == want)
                                .unwrap_or_else(|| {
                                    panic!("high-order.md: no M10 row {want:?} ({what})")
                                });
                            if ratio > bound {
                                assert!(
                                    !row[9].is_empty() && row[9] != "–" && row[9] != "-",
                                    "{what}: {part} ratio {ratio:.2} above {bound:.2} without a \
                                     cause"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// ADR-0056 under the generic policy: every choice at p = 1, 4 and 8 (pyramids at p = 1) has
/// records on every platform of its class, all of which ran and took at most 30 s to compile and
/// load at first use.
#[test]
fn m10_policy_first_use_within_30_s() {
    let data = m10();
    let records = m10_degree(&data);
    let policy = policy();
    for class in ["Cpu", "Gpu"] {
        for (scalar, bytes) in SCALARS {
            let devices = m10_devices(records, class, scalar);
            for family in DEGREE_FAMILIES {
                for &cell in cells_of(family) {
                    for degree in m10_degrees(cell) {
                        let s =
                            policy.strategy_of(class, policy_family(family), cell, bytes, degree);
                        let what =
                            format!("{class} {family} {cell} p={degree} {} {s}", scalar.name());
                        for &(target, backend) in &devices {
                            let of: Vec<&DegreeRecord> = records
                                .iter()
                                .filter(|r| {
                                    r.target == target
                                        && r.backend == backend
                                        && r.family == family
                                        && r.cell == cell
                                        && r.degree == degree
                                        && r.scalar == scalar
                                        && r.strategy == s
                                })
                                .collect();
                            assert!(!of.is_empty(), "{what}: no record on {target} {backend}");
                            for r in of {
                                assert!(
                                    r.status == "ok" && r.compile_s <= 30.0,
                                    "{what} on {target} {backend}: status {}, first use {:.1} s",
                                    r.status,
                                    r.compile_s
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
