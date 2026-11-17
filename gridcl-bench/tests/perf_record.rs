//! M3s acceptance (docs/design.md, "M3s SIMD vectorization and benchmarks"; ADR-0015,
//! ADR-0017, ADR-0018): the committed benchmark data, the policy and the throughput records.
//! From M10 these are history tests (ADR-0058, ADR-0060 item 2): the M3s policy is frozen in
//! `gridcl-bench/history/vector-policy-pre-m10.toml`, its classes are the [`LegacyClass`]es,
//! and the M10 tests at the end of the file check the live policy and report.
//!
//! The checks of the decision rule are written here independently of `gridcl_bench::decide`:
//! they read the M3s policy with the `toml` crate and recompute the class times from the
//! raw records.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use gridcl_bench::decide::decide;
use gridcl_bench::record::{
    CubeDimMode, LegacyClass, Measure, PerfData, Scalar, SweepRecord, ThroughputRecord, load_dir,
};
use gridcl_device::{DeviceClass, KernelKind};

/// The decision rule's problem sizes.
const RULE_SIZES: [usize; 2] = [1_000_000, 10_000_000];
/// The protocol's problem sizes.
const SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
/// CPU block targets of the protocol.
const BLOCKS: [usize; 3] = [4 << 10, 16 << 10, 64 << 10];
/// QuadField tiles of the protocol (GeomFactors).
const TILES: [usize; 3] = [1, 4, 16];

/// Every kind in `vector-policy.toml`, with the scalar types the benchmarks measure it in.
/// Pack and PullBack exist only from M6 and M9; `Scalar` is measured for its CPU blocks only
/// (histogram in u32, segmented reduce in f64; ADR-0015, item 7).
fn kinds() -> Vec<(KernelKind, Vec<Scalar>)> {
    use KernelKind as K;
    use Scalar::{F32, F64, U32};
    vec![
        (K::Fill, vec![F32, F64, U32]),
        (K::Copy, vec![U32]),
        (K::Reduce, vec![F32, F64, U32]),
        (K::ScanTile, vec![U32]),
        (K::SortDigits, vec![U32]),
        (K::Gather, vec![F32, F64, U32]),
        (K::Pack, vec![F32, F64]),
        (K::TopologyKeys, vec![U32]),
        (K::PermutationInfo, vec![U32]),
        (K::EVector, vec![F32, F64]),
        (K::GeomFactors, vec![F32, F64]),
        (K::PullBack, vec![F32, F64]),
        (K::Scalar, vec![U32, F64]),
    ]
}

/// The candidate families that exist after M3: every one is measured on every target.
fn measured(kind: KernelKind) -> bool {
    !matches!(kind, KernelKind::Pack | KernelKind::PullBack)
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn data() -> PerfData {
    let dir = root().join("docs/perf/data");
    let data = load_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    assert!(
        !data.sweeps.is_empty(),
        "no M3s sweep data (m3s-sweep-*.json) under docs/perf/data/"
    );
    data
}

fn read(path: &str) -> String {
    let path = root().join(path);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn is_cpu(class: LegacyClass) -> bool {
    class != LegacyClass::Gpu
}

/// Each timed run repeats the call k times, k fixed by the calibration call: ⌈20 ms / t_cal⌉ in
/// whole nanoseconds, at most 10,000 (ADR-0018; docs/plans/M3s.md, T22).
fn check_calls_per_run(calls_per_run: usize, calibration_s: f64, what: &str) {
    assert!(calls_per_run >= 1, "{what}: calls_per_run is 0");
    assert!(calibration_s >= 0.0, "{what}: negative calibration_s");
    let t_ns = ((calibration_s * 1e9).round() as u64).max(1);
    let want = 20_000_000u64.div_ceil(t_ns).min(10_000) as usize;
    assert_eq!(
        calls_per_run, want,
        "{what}: calls_per_run {calls_per_run} for a calibration call of {calibration_s:.3e} s"
    );
}

/// Both node-load snapshots of a Locust run are `idle` (ADR-0002, ADR-0005).
/// Kathleen benchmarks run on one socket, 20 threads (ADR-0020).
fn check_kathleen_socket(target: &str, threads: u32, what: &str) {
    if target == "kathleen" {
        assert_eq!(threads, 20, "{what}: not bound to one socket (ADR-0020)");
    }
}

fn check_idle(target: &str, node_load: &Option<gridcl_bench::record::NodeLoad>, what: &str) {
    if target != "locust" {
        return;
    }
    let load = node_load
        .as_ref()
        .unwrap_or_else(|| panic!("{what}: a Locust record without node-load snapshots"));
    assert!(
        load.before == "idle" && load.after == "idle",
        "{what}: Locust was not idle (before: {}; after: {})",
        load.before,
        load.after
    );
}

#[test]
fn benchmark_data_covers_required_targets() {
    let data = data();
    for r in &data.sweeps {
        let what = format!(
            "{} {} {:?} {}",
            r.target,
            r.backend,
            r.family,
            r.scalar.name()
        );
        check_idle(&r.target, &r.node_load, &what);
        assert_eq!(r.profile, "release", "{what}: record release builds only");
        assert_eq!(r.times_s.len(), 10, "{what}: ten timed runs");
        check_calls_per_run(r.calls_per_run, r.calibration_s, &what);
        check_kathleen_socket(&r.target, r.cores.threads, &what);
        if r.target.starts_with("myriad-") {
            assert!(
                r.fallback_reason.as_deref().is_some_and(|s| !s.is_empty()),
                "{what}: a Myriad record needs a fallback_reason (ADR-0014, ADR-0016)"
            );
        }
    }

    let cuda = if data
        .sweeps
        .iter()
        .any(|r| r.target == "locust" && r.backend == "cuda")
    {
        "locust"
    } else if data
        .sweeps
        .iter()
        .any(|r| r.target == "myriad-a100" && r.backend == "cuda")
    {
        "myriad-a100"
    } else {
        "myriad-v100"
    };
    // CpuSimd512 on a whole Kathleen node; a Myriad D node only as the fallback (ADR-0016).
    let avx512 = if data
        .sweeps
        .iter()
        .any(|r| r.target == "kathleen" && r.backend == "cpu")
    {
        "kathleen"
    } else {
        "myriad-cpu"
    };
    let required = [
        ("m3max", "cpu", LegacyClass::CpuSimd128),
        ("m3max", "metal", LegacyClass::Gpu),
        ("locust", "cpu", LegacyClass::CpuSimd128),
        (cuda, "cuda", LegacyClass::Gpu),
        (avx512, "cpu", LegacyClass::CpuSimd512),
    ];
    for (target, backend, class) in required {
        let records: Vec<&SweepRecord> = data
            .sweeps
            .iter()
            .filter(|r| r.target == target && r.backend == backend)
            .collect();
        assert!(!records.is_empty(), "no sweep data for {target} {backend}");
        for r in &records {
            assert_eq!(
                r.legacy_class,
                Some(class),
                "{target} {backend}: device class"
            );
        }
        // A CPU with a performance-core count is swept on its performance cores too (ADR-0017);
        // the M3 Max has one.
        let has_performance = |r: &&SweepRecord| r.cores.performance.is_some_and(|p| p > 0);
        if (target, backend) == ("m3max", "cpu") {
            assert!(
                records.iter().all(has_performance),
                "m3max cpu: records without the performance-core count"
            );
        }
        let performance = is_cpu(class) && records.iter().any(has_performance);
        for (kind, scalars) in kinds() {
            if !measured(kind) || (kind == KernelKind::Scalar && !is_cpu(class)) {
                continue;
            }
            for scalar in scalars {
                if scalar == Scalar::F64 && backend == "metal" {
                    continue; // Metal has no f64
                }
                let what = format!("{target} {backend} {kind:?} {}", scalar.name());
                let of: Vec<&&SweepRecord> = records
                    .iter()
                    .filter(|r| r.family == kind && r.scalar == scalar)
                    .collect();
                for size in SIZES {
                    let at = |vector: usize| {
                        of.iter()
                            .filter(|r| r.nominal_n == size && r.vector_size == vector)
                            .filter(|r| r.cube_dim == CubeDimMode::Plan)
                            .collect::<Vec<_>>()
                    };
                    let scalar_runs = at(1);
                    assert!(!scalar_runs.is_empty(), "{what}: nothing at {size}");
                    if performance {
                        assert!(
                            of.iter().any(|r| r.nominal_n == size
                                && r.vector_size == 1
                                && r.cube_dim == CubeDimMode::PerformanceCores),
                            "{what} at {size}: no performance-core runs (ADR-0017)"
                        );
                    }
                    if is_cpu(class) {
                        for block in BLOCKS {
                            assert!(
                                scalar_runs.iter().any(|r| r.block_bytes == Some(block)),
                                "{what} at {size}: no {} KiB block",
                                block >> 10
                            );
                        }
                    }
                    if kind == KernelKind::GeomFactors {
                        for tile in TILES {
                            assert!(
                                scalar_runs.iter().any(|r| r.tile == Some(tile)),
                                "{what} at {size}: no tile {tile}"
                            );
                        }
                    }
                    // Every size the device offers: 1, 2, 4, ... up to the widest measured.
                    let sizes: BTreeSet<usize> = of
                        .iter()
                        .filter(|r| r.nominal_n == size)
                        .map(|r| r.vector_size)
                        .collect();
                    let widest = *sizes.last().unwrap();
                    let want: BTreeSet<usize> = (0..=widest.ilog2()).map(|k| 1usize << k).collect();
                    assert_eq!(sizes, want, "{what} at {size}: vector sizes");
                    if kind == KernelKind::Scalar {
                        assert_eq!(widest, 1, "{what}: Scalar kernels run at size 1");
                    }
                }
            }
        }
    }
}

/// A configuration: (requested vector size, tile, CPU block in bytes, CPU units on CPUs:
/// `"all"` or `"performance"`).
type Config = (usize, Option<usize>, Option<usize>, Option<&'static str>);

/// The class time of every configuration of (class, kind, scalar) at `size`: the geometric
/// mean over the class's devices of their medians, for the configurations every device
/// measured (ADR-0015, item 5). Ratios of class times are geometric means of per-device ratios.
/// On CPUs, all-threads records (`cube_dim` `plan`) give the units `all` and performance-core
/// records `performance`; a device without a performance-core count gives both its
/// all-threads time (ADR-0017).
fn class_times(
    data: &PerfData,
    class: LegacyClass,
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
) -> BTreeMap<Config, f64> {
    let cpu = is_cpu(class);
    let mut per_device: BTreeMap<(String, String), BTreeMap<Config, f64>> = BTreeMap::new();
    let mut neutral = Vec::new();
    for r in data.sweeps.iter().filter(|r| {
        r.legacy_class == Some(class)
            && r.family == kind
            && r.scalar == scalar
            && r.nominal_n == size
            && (cpu || r.cube_dim == CubeDimMode::Plan)
    }) {
        let units = match r.cube_dim {
            CubeDimMode::Plan => "all",
            CubeDimMode::PerformanceCores => "performance",
        };
        let config = (r.vector_size, r.tile, r.block_bytes, cpu.then_some(units));
        let device = (r.target.clone(), r.backend.clone());
        if cpu && units == "all" && r.cores.performance.is_none() {
            let same = (r.vector_size, r.tile, r.block_bytes, Some("performance"));
            neutral.push((device.clone(), same, r.median_s));
        }
        let times = per_device.entry(device).or_default();
        assert!(
            times.insert(config, r.median_s).is_none(),
            "{} {} {kind:?} {} at {size}: configuration {config:?} recorded twice",
            r.target,
            r.backend,
            scalar.name()
        );
    }
    for (device, config, t) in neutral {
        per_device
            .entry(device)
            .or_default()
            .entry(config)
            .or_insert(t);
    }
    let devices = per_device.len();
    let mut out = BTreeMap::new();
    if devices == 0 {
        return out;
    }
    let first = per_device.values().next().unwrap();
    for config in first.keys() {
        let logs: Vec<f64> = per_device
            .values()
            .filter_map(|times| times.get(config).map(|t| t.ln()))
            .collect();
        if logs.len() == devices {
            out.insert(*config, (logs.iter().sum::<f64>() / devices as f64).exp());
        }
    }
    out
}

/// The smallest class time of the configurations whose vector size satisfies `keep`.
fn best(times: &BTreeMap<Config, f64>, keep: impl Fn(usize) -> bool) -> Option<f64> {
    times
        .iter()
        .filter(|((n, ..), _)| keep(*n))
        .map(|(_, &t)| t)
        .min_by(f64::total_cmp)
}

fn entry(table: &toml::Table, path: &[&str]) -> Option<usize> {
    let (last, inner) = path.split_last().unwrap();
    let mut t = table;
    for key in inner {
        t = t.get(*key)?.as_table()?;
    }
    t.get(*last)?.as_integer().map(|v| v as usize)
}

#[test]
fn policy_follows_the_decision_rule() {
    let data = data();
    let text = read("gridcl-bench/history/vector-policy-pre-m10.toml");
    assert_eq!(
        text,
        decide(&data).to_toml(),
        "history/vector-policy-pre-m10.toml differs from the M3s decision rule, whose choices \
         are M10's candidates (ADR-0060 item 2)"
    );

    let policy: toml::Table = text.parse().expect("vector-policy-pre-m10.toml parses");
    let report = read("docs/perf/vectorization.md");
    // CpuSimd256 copies the CpuSimd512 table ("Benchmarks", "Where").
    assert_eq!(
        policy.get("CpuSimd256"),
        policy.get("CpuSimd512"),
        "CpuSimd256 copies CpuSimd512"
    );
    for class in [
        LegacyClass::CpuSimd128,
        LegacyClass::CpuSimd512,
        LegacyClass::Gpu,
    ] {
        let c = format!("{class:?}");
        for (kind, scalars) in kinds() {
            let k = format!("{kind:?}");
            for bytes in [4usize, 8] {
                let b = bytes.to_string();
                let what = format!("[{c}] {k} {bytes}");
                let n = entry(&policy, &[&c, "vector_size", &k, &b])
                    .unwrap_or_else(|| panic!("{what}: no vector_size"));
                let block = is_cpu(class).then(|| {
                    entry(&policy, &[&c, "cpu_block", &k, &b])
                        .unwrap_or_else(|| panic!("{what}: no cpu_block"))
                        << 12
                });
                let tile = (kind == KernelKind::GeomFactors).then(|| {
                    entry(&policy, &[&c, "quad_tile", &b])
                        .unwrap_or_else(|| panic!("[{c}] no quad_tile {bytes}"))
                });
                let units = is_cpu(class).then(|| {
                    let v = policy
                        .get(&c)
                        .and_then(|t| t.get("cpu_units"))
                        .and_then(|t| t.get(&k))
                        .and_then(|t| t.get(&b))
                        .and_then(|v| v.as_str())
                        .unwrap_or_else(|| panic!("{what}: no cpu_units"));
                    match v {
                        "all" => "all",
                        "performance" => "performance",
                        _ => panic!("{what}: cpu_units {v:?} is not all or performance"),
                    }
                });
                let chosen: Config = (n, tile, block, units);
                let with_data: Vec<Scalar> = scalars
                    .iter()
                    .copied()
                    .filter(|s| s.bytes() == bytes)
                    .filter(|&s| !class_times(&data, class, kind, s, RULE_SIZES[0]).is_empty())
                    .collect();
                if with_data.is_empty() {
                    assert_eq!(n, 1, "{what}: vector size above 1 without data");
                    assert!(
                        units.is_none_or(|u| u == "all"),
                        "{what}: performance cores without data"
                    );
                    continue;
                }
                // Every 4-byte type measured must admit a shared entry (ADR-0015, item 6).
                let mut views: Vec<TypeView> = with_data
                    .iter()
                    .map(|&scalar| TypeView::new(&data, class, kind, scalar))
                    .collect();
                // A shared entry measures each type against its fastest configuration at a size
                // every type admits (ADR-0019).
                let entry: BTreeSet<usize> = views[0]
                    .admitted
                    .iter()
                    .copied()
                    .filter(|n| views.iter().all(|v| v.admitted.contains(n)))
                    .collect();
                for v in &mut views {
                    v.measure_against(&entry, &what);
                }
                for v in &views {
                    let what = format!("{what} ({})", v.scalar.name());
                    if n > 1 {
                        for (t, size) in v.times.iter().zip(RULE_SIZES) {
                            let s = speedup(t, n).unwrap_or(0.0);
                            assert!(
                                s >= 1.1,
                                "{what}: size {n} is {s:.3}x size 1 at {size}, not 1.1x"
                            );
                        }
                    }
                    for (t, size) in v.times.iter().zip(RULE_SIZES) {
                        assert!(
                            t.contains_key(&chosen),
                            "{what}: the chosen {chosen:?} was not measured at {size}"
                        );
                    }
                }
                let worst = |c: &Config| -> Option<f64> {
                    let mut worst = 0.0f64;
                    for v in &views {
                        for i in 0..2 {
                            worst = worst.max(v.times[i].get(c)? / v.fastest[i]);
                        }
                    }
                    Some(worst)
                };
                let chosen_worst = worst(&chosen).unwrap();
                let candidates: Vec<Config> = views[0].times[0]
                    .keys()
                    .copied()
                    .filter(|c| entry.contains(&c.0))
                    .collect();
                if chosen_worst <= 1.03 {
                    // Ties go to the smallest vector size, tile and block, then `all` (ADR-0015,
                    // ADR-0017): no smaller configuration is within 3% too.
                    for c in candidates.iter().filter(|&&c| c < chosen) {
                        if let Some(w) = worst(c) {
                            assert!(
                                w > 1.03 * (1.0 - 1e-12),
                                "{what}: {c:?} is within 3% ({w:.4}) and precedes the chosen \
                                 {chosen:?} in the tie order"
                            );
                        }
                    }
                    continue;
                }
                // The backstop (ADR-0018): no admitted configuration within 3% for every type at
                // both sizes; the chosen one is the closest, at most 10% slower, and marked. A
                // shared entry with more than one type measured may go up to 25%, marked "closest
                // across types" above 10% (ADR-0021).
                for c in &candidates {
                    if let Some(w) = worst(c) {
                        assert!(
                            w > 1.03,
                            "{what}: the chosen {chosen:?} is {:.1}% above the fastest admitted, \
                             but {c:?} is within 3% at both sizes",
                            (chosen_worst - 1.0) * 100.0
                        );
                        assert!(
                            chosen_worst <= w * (1.0 + 1e-12),
                            "{what}: the backstop chose {chosen:?} ({chosen_worst:.4}), but \
                             {c:?} is closer ({w:.4})"
                        );
                    }
                }
                let (bound, percent, adr) = if views.len() > 1 {
                    (1.25, 25, "ADR-0021")
                } else {
                    (1.1, 10, "ADR-0018")
                };
                assert!(
                    chosen_worst <= bound,
                    "{what}: unresolved: the closest admitted configuration {chosen:?} is \
                     {:.1}% slower than the fastest, above the {percent}% bound ({adr})",
                    (chosen_worst - 1.0) * 100.0
                );
                let mark = if chosen_worst <= 1.1 {
                    "| closest, "
                } else {
                    "| closest across types, "
                };
                for v in &views {
                    let row = if kind == KernelKind::Scalar {
                        format!("| {c} | {} |", v.scalar.name())
                    } else {
                        format!("| {k} | {c} | {} |", v.scalar.name())
                    };
                    let marked = report
                        .lines()
                        .any(|line| line.starts_with(&row) && line.contains(mark));
                    assert!(
                        marked,
                        "{what}: the report does not mark the backstop entry {row} as closest"
                    );
                }
            }
        }
    }
}

/// One scalar type's view of the rule: class times at both rule sizes, the admitted sizes (1,
/// and every size at least 10% faster at both), and the fastest time at each size at a size the
/// whole entry admits (ADR-0019).
struct TypeView {
    scalar: Scalar,
    times: Vec<BTreeMap<Config, f64>>,
    admitted: BTreeSet<usize>,
    fastest: [f64; 2],
}

impl TypeView {
    fn new(data: &PerfData, class: LegacyClass, kind: KernelKind, scalar: Scalar) -> TypeView {
        let times: Vec<BTreeMap<Config, f64>> = RULE_SIZES
            .iter()
            .map(|&size| class_times(data, class, kind, scalar, size))
            .collect();
        let sizes: BTreeSet<usize> = times[0].keys().map(|c| c.0).collect();
        let admitted: BTreeSet<usize> = sizes
            .into_iter()
            .filter(|&v| {
                v == 1
                    || times
                        .iter()
                        .all(|t| speedup(t, v).is_some_and(|s| s >= 1.1))
            })
            .collect();
        TypeView {
            scalar,
            times,
            admitted,
            fastest: [f64::NAN; 2],
        }
    }

    /// Sets `fastest` from the configurations at the sizes in `entry`.
    fn measure_against(&mut self, entry: &BTreeSet<usize>, what: &str) {
        self.fastest = [0, 1].map(|i| {
            best(&self.times[i], |v| entry.contains(&v)).unwrap_or_else(|| {
                panic!(
                    "{what} ({}): nothing admitted at {}",
                    self.scalar.name(),
                    RULE_SIZES[i]
                )
            })
        });
    }
}

/// Size 1's best class time over size `v`'s best.
fn speedup(times: &BTreeMap<Config, f64>, v: usize) -> Option<f64> {
    Some(best(times, |m| m == 1)? / best(times, |m| m == v)?)
}

#[test]
fn throughput_recorded_with_policy_and_scalar() {
    let data = data();
    for r in &data.throughput {
        let what = format!("{} {:?}", r.target, r.measure);
        check_idle(&r.target, &r.node_load, &what);
        check_calls_per_run(r.calls_per_run, r.calibration_s, &what);
        check_kathleen_socket(&r.target, r.cores.threads, &what);
        assert_eq!(r.times_s.len(), 10, "{what}: ten timed runs");
        assert_eq!(
            r.scalar,
            Scalar::F64,
            "{what}: throughput is measured in f64"
        );
    }
    // A Kathleen node; a Myriad D node only as the fallback, with its reason (ADR-0016).
    for r in data
        .throughput
        .iter()
        .filter(|r| r.target.starts_with("myriad-"))
    {
        assert!(
            r.fallback_reason.as_deref().is_some_and(|s| !s.is_empty()),
            "{} {:?}: a Myriad record needs a fallback_reason (ADR-0016)",
            r.target,
            r.measure
        );
    }
    let avx512 = if data.throughput.iter().any(|r| r.target == "kathleen") {
        "kathleen"
    } else {
        "myriad-cpu"
    };
    for target in ["m3max", avx512] {
        for measure in [Measure::EntitiesPerSecond, Measure::FactorsPerSecond] {
            for over in ["policy", "scalar"] {
                let what = format!("{target} cpu {measure:?} with {over}");
                let found: Vec<&ThroughputRecord> = data
                    .throughput
                    .iter()
                    .filter(|r| {
                        r.target == target
                            && r.backend == "cpu"
                            && r.measure == measure
                            && r.vector_size_override == over
                    })
                    .collect();
                assert!(!found.is_empty(), "{what}: no throughput record");
                for r in found {
                    assert_eq!(r.profile, "release", "{what}");
                    assert!(r.items > 0 && r.median_s > 0.0, "{what}");
                    let per_s = r.items as f64 / r.median_s;
                    assert!(
                        (r.per_s - per_s).abs() <= 1e-9 * per_s,
                        "{what}: per_s {} is not items / median_s = {per_s}",
                        r.per_s
                    );
                }
            }
        }
    }
}

// =============================================================================================
// M10 (docs/design.md, Roadmap M10, read as in ADR-0057, ADR-0058, ADR-0059 and ADR-0060): the
// records under `docs/perf/data/m10-*.json`, the four generic sets of `vector-policy.toml` and
// M10's tables of `docs/perf/vectorization.md`.
//
// As above, the rule is recomputed here from the raw records, independently of
// `gridcl_bench::decide_generic`.

use gridcl_bench::candidates::{VectorConfig, vector_candidates};
use gridcl_bench::decide_generic::decide_vector;
use gridcl_bench::record::{Cores, HALO_PARTS, HaloRecord, M10Data, NodeLoad, load_m10_dir};
use gridcl_bench::report::{M10_BEGIN_MARKER, M10_END_MARKER, render_m10_vector};
use gridcl_device::CpuUnits;

/// Timed runs of the candidate protocol (ADR-0057 item 3).
const CANDIDATE_RUNS: usize = 5;

fn m10() -> M10Data {
    let dir = root().join("docs/perf/data");
    load_m10_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

/// The name of a record's class, `Cpu` or `Gpu` (whatever type the record stores it as).
fn class_name(class: &impl std::fmt::Debug) -> String {
    format!("{class:?}").trim_matches('"').to_owned()
}

/// The benchmark platforms of M10 (ADR-0058 item 3), as (target, backend, class): the local
/// Mac's CPU runtime and Metal, Locust's Grace CPU and H100, one Kathleen socket. A Myriad D
/// node stands in for Kathleen, and Myriad's A100 or V100 for the H100, only where the primary
/// target has no record (ADR-0014, ADR-0016); the A4000 workstation does not vote (ADR-0037).
fn m10_platforms(
    present: impl Fn(&str, &str) -> bool,
) -> Vec<(&'static str, &'static str, &'static str)> {
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

/// The `DeviceClass` named `name` (`Cpu` or `Gpu`).
fn device_class(name: &str) -> DeviceClass {
    gridcl_bench::record::class_named(name).unwrap_or_else(|| panic!("no device class {name}"))
}

/// The scalar types a backend measures: Metal has no f64.
fn backend_scalars(backend: &str, scalars: &[Scalar]) -> Vec<Scalar> {
    scalars
        .iter()
        .copied()
        .filter(|&s| !(backend == "metal" && s == Scalar::F64))
        .collect()
}

/// The fields every M10 record shares.
struct Common<'a> {
    target: &'a str,
    backend: &'a str,
    class: String,
    profile: &'a str,
    runs: usize,
    cores: &'a Cores,
    load_width: Option<u32>,
    fallback_reason: Option<&'a str>,
    node_load: &'a Option<NodeLoad>,
}

/// The conditions of every M10 record: release builds, `runs` timed runs, the class of its
/// backend, the hardware (`load_width`, core counts; ADR-0060 item 2), idle Locust, one Kathleen
/// socket, and a fallback reason on Myriad; no A4000 record (ADR-0037).
fn check_m10_common(c: Common, runs: usize, what: &str) {
    assert_eq!(c.profile, "release", "{what}: record release builds only");
    assert_eq!(c.runs, runs, "{what}: {runs} timed runs");
    let class = if c.backend == "cpu" { "Cpu" } else { "Gpu" };
    assert_eq!(c.class, class, "{what}: device class");
    assert!(c.load_width.is_some(), "{what}: no load_width");
    assert!(c.cores.threads > 0, "{what}: no core counts");
    if c.target == "macmini" && c.backend == "cpu" {
        assert!(
            !c.cores.levels.is_empty(),
            "{what}: no performance levels (ADR-0034)"
        );
    }
    check_idle(c.target, c.node_load, what);
    check_kathleen_socket(c.target, c.cores.threads, what);
    if c.target.starts_with("myriad-") {
        assert!(
            c.fallback_reason.is_some_and(|s| !s.is_empty()),
            "{what}: a Myriad record needs a fallback_reason (ADR-0014, ADR-0016)"
        );
    }
    assert_ne!(
        c.target, "a4000",
        "{what}: the A4000 does not vote (ADR-0037)"
    );
}

fn median(times: &[f64]) -> f64 {
    let mut v = times.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Bullet 1 (ADR-0059): the host-staged halo exchange is recorded for halos of 10^3, 10^4 and
/// 10^5 nodes with block 3 and 10^6 owned nodes per rank, on the Mac mini's CPU runtime (f64)
/// and Metal (f32) on two ranks, and on Locust's CUDA (f64) on one and two ranks; one rank
/// times the copies and kernels only, two ranks also the host exchange, `forward` and `reverse`.
/// At 10^5 nodes, on each, the host-to-device copy takes at most 1.5 times the device-to-host
/// copy, and `reverse` at most twice `forward`. (Bitwise results: the `scatter` MPI tests and
/// `scatter_threads`.)
#[test]
fn halo_exchange_recorded() {
    let data = m10();
    let halo = &data.halo;
    assert!(
        !halo.is_empty(),
        "no M10 halo records (m10-halo-*.json) under docs/perf/data/"
    );
    for r in halo {
        let what = format!(
            "{} {} {} ranks {} halo {} {}",
            r.target,
            r.backend,
            r.scalar.name(),
            r.ranks,
            r.halo_nodes,
            r.part
        );
        let c = Common {
            target: &r.target,
            backend: &r.backend,
            class: class_name(&r.device_class),
            profile: &r.profile,
            runs: r.times_s.len(),
            cores: &r.cores,
            load_width: r.load_width,
            fallback_reason: r.fallback_reason.as_deref(),
            node_load: &r.node_load,
        };
        check_m10_common(c, 50, &what);
        assert_eq!(r.warmups, 5, "{what}: five warm-ups");
        assert_eq!(r.block, 3, "{what}: block 3");
        assert_eq!(
            r.owned_nodes, 1_000_000,
            "{what}: 10^6 owned nodes per rank"
        );
        assert!(
            HALO_PARTS.contains(&r.part.as_str()),
            "{what}: unknown part"
        );
        assert!(r.times_s.iter().all(|t| *t > 0.0), "{what}: a time ≤ 0");
        assert_eq!(r.median_s, median(&r.times_s), "{what}: median_s");
    }
    let required = [
        ("macmini", "cpu", Scalar::F64, 2),
        ("macmini", "metal", Scalar::F32, 2),
        ("locust", "cuda", Scalar::F64, 1),
        ("locust", "cuda", Scalar::F64, 2),
    ];
    for (target, backend, scalar, ranks) in required {
        let at = |halo_nodes: usize, part: &str| -> &HaloRecord {
            halo.iter()
                .rev()
                .find(|r| {
                    r.target == target
                        && r.backend == backend
                        && r.scalar == scalar
                        && r.ranks == ranks
                        && r.halo_nodes == halo_nodes
                        && r.part == part
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{target} {backend} {} {ranks} ranks: no {part} record at {halo_nodes} \
                         halo nodes",
                        scalar.name()
                    )
                })
        };
        let parts: &[&str] = if ranks == 1 {
            &["pack", "d2h", "h2d", "unpack"]
        } else {
            &HALO_PARTS
        };
        for halo_nodes in [1_000, 10_000, 100_000] {
            for part in parts {
                at(halo_nodes, part);
            }
        }
        let t = |part| at(100_000, part).median_s;
        let what = format!(
            "{target} {backend} {} {ranks} ranks, 10^5 nodes",
            scalar.name()
        );
        assert!(
            t("h2d") <= 1.5 * t("d2h"),
            "{what}: host-to-device {:.0} µs > 1.5 × device-to-host {:.0} µs",
            t("h2d") * 1e6,
            t("d2h") * 1e6
        );
        if ranks > 1 {
            assert!(
                t("reverse") <= 2.0 * t("forward"),
                "{what}: reverse {:.0} µs > 2 × forward {:.0} µs",
                t("reverse") * 1e6,
                t("forward") * 1e6
            );
        }
    }
}

/// The units of a CPU configuration in a record: `All` for all hardware threads, `Performance`
/// for the performance-core count; a device without one gives its all-threads time for both
/// (ADR-0017).
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

/// Whether record `r` measured configuration `c`.
fn measures(r: &SweepRecord, c: &VectorConfig) -> bool {
    r.vector_size == c.vector_size
        && r.tile == c.tile
        && r.block_bytes == c.block_bytes
        && record_units(r).contains(&c.units)
}

/// The frozen pre-M10 vector policy (ADR-0060 item 2).
fn history_vector_policy() -> toml::Table {
    read("gridcl-bench/history/vector-policy-pre-m10.toml")
        .parse()
        .expect("history/vector-policy-pre-m10.toml parses")
}

/// The configuration of `kind` and `bytes` in the old class `old` of the pre-M10 policy, at
/// vector size `n` if given (the baseline), else the class's own.
fn history_config(
    policy: &toml::Table,
    old: &str,
    kind: KernelKind,
    bytes: usize,
    n: Option<usize>,
) -> VectorConfig {
    let k = format!("{kind:?}");
    let b = bytes.to_string();
    let cpu = old != "Gpu";
    let n = n.unwrap_or_else(|| {
        entry(policy, &[old, "vector_size", &k, &b])
            .unwrap_or_else(|| panic!("history [{old}] {k} {b}: no vector_size"))
    });
    let tile = (kind == KernelKind::GeomFactors)
        .then(|| entry(policy, &[old, "quad_tile", &b]).expect("history quad_tile"));
    let block_bytes = cpu.then(|| entry(policy, &[old, "cpu_block", &k, &b]).expect("block") << 12);
    let units = cpu.then(|| match policy[old]["cpu_units"][&k][&b].as_str() {
        Some("performance") => CpuUnits::Performance,
        Some("all") => CpuUnits::All,
        v => panic!("history [{old}] cpu_units {k} {b}: {v:?}"),
    });
    VectorConfig {
        vector_size: n,
        tile,
        block_bytes,
        units,
    }
}

/// The old classes whose choices a Cpu or Gpu entry measures (CpuSimd256 copied CpuSimd512).
fn old_classes(class: &str) -> &'static [&'static str] {
    if class == "Cpu" {
        &["CpuSimd128", "CpuSimd512"]
    } else {
        &["Gpu"]
    }
}

/// The kinds of a class: `Scalar` (CPU blocks only) on Cpu only.
fn kinds_of(class: &str) -> Vec<(KernelKind, Vec<Scalar>)> {
    kinds()
        .into_iter()
        .filter(|(k, _)| class == "Cpu" || *k != KernelKind::Scalar)
        .collect()
}

/// Bullet 4 (ADR-0057 item 3, ADR-0060 item 2): the candidate protocol ran for every kind,
/// including Pack and PullBack, on every platform, at 10^6 and 10^7 items, five runs each:
/// the baseline (vector size 1 at the old CpuSimd128 tile, block and units on Cpu, at the Gpu
/// tile on Gpu), the M3s choices of every old class of the entry's class, the widest size for
/// Pack and PullBack, and every configuration of `candidates::vector_candidates`.
#[test]
fn m10_candidates_cover_every_kind_and_platform() {
    let data = m10();
    let history = history_vector_policy();
    let vector = &data.vector;
    assert!(
        !vector.is_empty(),
        "no M10 vector records (m10-vector-*.json) under docs/perf/data/"
    );
    for r in vector {
        let what = format!(
            "{} {} {:?} {} n={} at {}",
            r.target,
            r.backend,
            r.family,
            r.scalar.name(),
            r.vector_size,
            r.nominal_n
        );
        let c = Common {
            target: &r.target,
            backend: &r.backend,
            class: class_name(&r.device_class),
            profile: &r.profile,
            runs: r.times_s.len(),
            cores: &r.cores,
            load_width: r.load_width,
            fallback_reason: r.fallback_reason.as_deref(),
            node_load: &r.node_load,
        };
        check_m10_common(c, CANDIDATE_RUNS, &what);
        check_calls_per_run(r.calls_per_run, r.calibration_s, &what);
    }
    let present = |t: &str, b: &str| vector.iter().any(|r| r.target == t && r.backend == b);
    for (target, backend, class) in m10_platforms(present) {
        for (kind, scalars) in kinds_of(class) {
            for scalar in backend_scalars(backend, &scalars) {
                let bytes = scalar.bytes();
                let what = format!("{target} {backend} {class} {kind:?} {}", scalar.name());
                let of: Vec<&SweepRecord> = vector
                    .iter()
                    .filter(|r| r.target == target && r.backend == backend)
                    .filter(|r| r.family == kind && r.scalar == scalar)
                    .collect();
                assert!(!of.is_empty(), "{what}: no records");
                let widest = of.iter().map(|r| r.vector_size).max().unwrap();
                let baseline_class = if class == "Cpu" { "CpuSimd128" } else { "Gpu" };
                let mut want = vec![history_config(
                    &history,
                    baseline_class,
                    kind,
                    bytes,
                    Some(1),
                )];
                if kind != KernelKind::Scalar {
                    for old in old_classes(class) {
                        want.push(history_config(&history, old, kind, bytes, None));
                    }
                }
                want.extend(vector_candidates(device_class(class), kind, scalar, widest));
                if matches!(kind, KernelKind::Pack | KernelKind::PullBack) {
                    assert!(
                        widest > 1,
                        "{what}: Pack and PullBack measure the widest size (ADR-0060 item 2)"
                    );
                }
                for size in RULE_SIZES {
                    for c in &want {
                        assert!(
                            of.iter().any(|r| r.nominal_n == size && measures(r, c)),
                            "{what} at {size}: candidate {c:?} not measured"
                        );
                    }
                }
            }
        }
    }
}

/// A configuration as an ordered key: (vector size, tile, block, performance cores). The order
/// is the generic rule's simplicity order: the smallest vector size, then tile, then block; ties
/// in units go to all hardware threads (ADR-0058 item 4).
type Key = (usize, Option<usize>, Option<usize>, bool);

fn key_of(c: &VectorConfig) -> Key {
    (
        c.vector_size,
        c.tile,
        c.block_bytes,
        c.units == Some(CpuUnits::Performance),
    )
}

/// One member of a generic entry: a platform and, for a shared 4-byte entry, a scalar type
/// (ADR-0058 item 4: f32 and u32 count as separate members).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Member {
    target: &'static str,
    backend: &'static str,
    scalar: Scalar,
}

/// The rule's view of one entry (class, kind, bytes).
struct GenericEntry {
    baseline: Key,
    /// The chosen configuration.
    choice: Key,
    /// Whether no configuration met the bound (the report marks it with its worst case).
    fallback: bool,
    /// Per platform: the largest r of the choice over its members and the rule sizes.
    platform_r: BTreeMap<(&'static str, &'static str), f64>,
}

/// The time of every configuration of `m` at `size` (the last record of each).
fn member_times(
    vector: &[SweepRecord],
    m: Member,
    kind: KernelKind,
    size: usize,
) -> BTreeMap<Key, f64> {
    let mut out = BTreeMap::new();
    for r in vector.iter().filter(|r| {
        r.target == m.target
            && r.backend == m.backend
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
            out.insert(key_of(&c), r.median_s);
        }
    }
    out
}

/// The generic rule (ADR-0058 item 4) on the M10 records of one entry; `None` if no platform of
/// the class measured it.
fn generic_entry(
    data: &M10Data,
    history: &toml::Table,
    class: &str,
    kind: KernelKind,
    scalars: &[Scalar],
    bytes: usize,
) -> Option<GenericEntry> {
    let vector = &data.vector;
    let present = |t: &str, b: &str| vector.iter().any(|r| r.target == t && r.backend == b);
    let members: Vec<Member> = m10_platforms(present)
        .into_iter()
        .filter(|&(_, _, c)| c == class)
        .flat_map(|(target, backend, _)| {
            backend_scalars(backend, scalars)
                .into_iter()
                .filter(move |s| s.bytes() == bytes)
                .map(move |scalar| Member {
                    target,
                    backend,
                    scalar,
                })
        })
        .filter(|&m| !member_times(vector, m, kind, RULE_SIZES[0]).is_empty())
        .collect();
    if members.is_empty() {
        return None;
    }
    let what = format!("[{class}] {kind:?} {bytes}");
    // times[m][s]: member m's times at rule size s.
    let times: Vec<Vec<BTreeMap<Key, f64>>> = members
        .iter()
        .map(|&m| {
            RULE_SIZES
                .iter()
                .map(|&size| member_times(vector, m, kind, size))
                .collect()
        })
        .collect();
    let common: BTreeSet<Key> = times[0][0]
        .keys()
        .copied()
        .filter(|k| times.iter().flatten().all(|t| t.contains_key(k)))
        .collect();
    let baseline_class = if class == "Cpu" { "CpuSimd128" } else { "Gpu" };
    let baseline = key_of(&history_config(
        history,
        baseline_class,
        kind,
        bytes,
        Some(1),
    ));
    assert!(
        common.contains(&baseline),
        "{what}: the baseline {baseline:?} is not measured on every platform and size"
    );
    let geomean = |v: &[f64]| (v.iter().map(|x| x.ln()).sum::<f64>() / v.len() as f64).exp();
    // r(c, m, s): c's time over the fastest candidate's on m at s.
    let r = |c: &Key, m: usize, s: usize| -> f64 {
        let fastest = common
            .iter()
            .map(|k| times[m][s][k])
            .fold(f64::INFINITY, f64::min);
        times[m][s][c] / fastest
    };
    let admitted: Vec<Key> = common
        .iter()
        .copied()
        .filter(|c| {
            *c == baseline
                || (0..RULE_SIZES.len()).all(|s| {
                    let ratios: Vec<f64> = (0..members.len())
                        .map(|m| times[m][s][c] / times[m][s][&baseline])
                        .collect();
                    geomean(&ratios) <= 1.0 / 1.10 * (1.0 + 1e-12)
                })
        })
        .collect();
    let all_r = |c: &Key| -> Vec<f64> {
        (0..members.len())
            .flat_map(|m| (0..RULE_SIZES.len()).map(move |s| (m, s)))
            .map(|(m, s)| r(c, m, s))
            .collect()
    };
    let worst = |c: &Key| all_r(c).into_iter().fold(0.0, f64::max);
    let kept: Vec<Key> = admitted
        .iter()
        .copied()
        .filter(|c| worst(c) <= 1.25 * (1.0 + 1e-12))
        .collect();
    let (choice, fallback) = if kept.is_empty() {
        let c = admitted
            .iter()
            .copied()
            .min_by(|a, b| worst(a).total_cmp(&worst(b)).then(a.cmp(b)))
            .expect("the baseline is admitted");
        (c, true)
    } else {
        let g = |c: &Key| geomean(&all_r(c));
        let best = kept.iter().map(g).fold(f64::INFINITY, f64::min);
        let c = kept
            .iter()
            .copied()
            .filter(|c| g(c) <= 1.03 * best * (1.0 + 1e-12))
            .min()
            .expect("the best is within 3% of itself");
        (c, false)
    };
    let mut platform_r: BTreeMap<(&'static str, &'static str), f64> = BTreeMap::new();
    for (j, m) in members.iter().enumerate() {
        let worst_j = (0..RULE_SIZES.len())
            .map(|s| r(&choice, j, s))
            .fold(0.0, f64::max);
        let at = platform_r.entry((m.target, m.backend)).or_insert(0.0);
        *at = at.max(worst_j);
    }
    Some(GenericEntry {
        baseline,
        choice,
        fallback,
        platform_r,
    })
}

/// The configuration of `vector-policy.toml` for (class, kind, bytes).
fn policy_config(policy: &toml::Table, class: &str, kind: KernelKind, bytes: usize) -> Key {
    let k = format!("{kind:?}");
    let b = bytes.to_string();
    let what = format!("[{class}] {k} {bytes}");
    let n = entry(policy, &[class, "vector_size", &k, &b])
        .unwrap_or_else(|| panic!("{what}: no vector_size"));
    let tile = (kind == KernelKind::GeomFactors).then(|| {
        entry(policy, &[class, "quad_tile", &b]).unwrap_or_else(|| panic!("{what}: no quad_tile"))
    });
    let cpu = class == "Cpu";
    let block = cpu.then(|| {
        entry(policy, &[class, "cpu_block", &k, &b])
            .unwrap_or_else(|| panic!("{what}: no cpu_block"))
            << 12
    });
    let performance = cpu
        && match policy[class]["cpu_units"][&k][&b].as_str() {
            Some("performance") => true,
            Some("all") => false,
            v => panic!("{what}: cpu_units {v:?}"),
        };
    (n, tile, block, performance)
}

/// M10's generated tables in `docs/perf/vectorization.md`.
fn m10_vector_section() -> String {
    let text = read("docs/perf/vectorization.md");
    let begin = text
        .find(M10_BEGIN_MARKER)
        .expect("docs/perf/vectorization.md has no M10 BEGIN marker")
        + M10_BEGIN_MARKER.len();
    let end = text[begin..]
        .find(M10_END_MARKER)
        .expect("docs/perf/vectorization.md has no M10 END marker")
        + begin;
    text[begin..end].trim().to_owned()
}

/// The cells of a table row, trimmed: `| a | b |` gives `["a", "b"]`.
fn cells(line: &str) -> Option<Vec<&str>> {
    let inner = line.trim().strip_prefix('|')?.strip_suffix('|')?;
    Some(inner.split('|').map(str::trim).collect())
}

/// Bullet 5 (ADR-0058): `vector-policy.toml` holds `[Cpu]` and `[Gpu]` with 4- and 8-byte
/// entries only, is what `decide_generic` writes from the M10 records, and every entry follows
/// the generic rule, recomputed here: admission at 10% over the baseline by the geometric mean
/// over the platforms, the bound r ≤ 1.25 on every platform and size, the smallest geometric
/// mean of r with the 3% simplicity order, and otherwise the smallest worst case, which the
/// report marks.
#[test]
fn vector_policy_follows_the_generic_rule() {
    let data = m10();
    let history = history_vector_policy();
    let text = read("gridcl-device/vector-policy.toml");
    assert_eq!(
        text,
        decide_vector(&data).to_toml(),
        "vector-policy.toml differs from the generic rule: run policy_report"
    );
    let policy: toml::Table = text.parse().expect("vector-policy.toml parses");
    let classes: BTreeSet<&str> = policy.keys().map(String::as_str).collect();
    assert_eq!(
        classes,
        BTreeSet::from(["Cpu", "Gpu"]),
        "vector-policy.toml classes"
    );
    let section = m10_vector_section();
    for class in ["Cpu", "Gpu"] {
        for (kind, scalars) in kinds_of(class) {
            for bytes in [4usize, 8] {
                let what = format!("[{class}] {kind:?} {bytes}");
                let got = policy_config(&policy, class, kind, bytes);
                let Some(e) = generic_entry(&data, &history, class, kind, &scalars, bytes) else {
                    assert_eq!(got.0, 1, "{what}: vector size above 1 without data");
                    assert!(!got.3, "{what}: performance cores without data");
                    continue;
                };
                assert_eq!(
                    got, e.choice,
                    "{what}: the policy has {got:?}, the generic rule {:?} (baseline {:?})",
                    e.choice, e.baseline
                );
                if e.fallback {
                    let prefix = format!("| {kind:?} | {class} | {bytes} |");
                    assert!(
                        section
                            .lines()
                            .any(|l| l.starts_with(&prefix) && l.contains("worst case")),
                        "{what}: no configuration meets the bound, and the report does not mark \
                         the entry with its worst case"
                    );
                }
            }
        }
    }
}

/// Bullet 4 and ADR-0058 item 6: M10's tables in `docs/perf/vectorization.md` are what
/// `policy_report` renders, and list for every entry and platform of its class the platform's
/// best configuration, the generic choice, the platform's r for that choice (recomputed here:
/// the largest over its members and the rule sizes) and the reason. The M3s section stays as it
/// was (frozen).
#[test]
fn vectorization_report_lists_platform_ratios() {
    let data = m10();
    let history = history_vector_policy();
    let section = m10_vector_section();
    assert_eq!(
        section,
        render_m10_vector(&data, &decide_vector(&data)).trim(),
        "M10's tables are stale: run policy_report"
    );
    let rows: Vec<Vec<&str>> = section.lines().filter_map(cells).collect();
    for class in ["Cpu", "Gpu"] {
        for (kind, scalars) in kinds_of(class) {
            for bytes in [4usize, 8] {
                let Some(e) = generic_entry(&data, &history, class, kind, &scalars, bytes) else {
                    continue;
                };
                for (&(target, backend), &r) in &e.platform_r {
                    let k = format!("{kind:?}");
                    let b = bytes.to_string();
                    let platform = format!("{target} {backend}");
                    let what = format!("[{class}] {k} {bytes} on {platform}");
                    let row = rows
                        .iter()
                        .find(|row| {
                            row.len() >= 8 && row[..4] == [k.as_str(), class, &b, &platform]
                        })
                        .unwrap_or_else(|| panic!("{what}: no row"));
                    assert!(!row[4].is_empty(), "{what}: no best configuration");
                    assert!(!row[5].is_empty(), "{what}: no choice");
                    assert_eq!(row[6], format!("{r:.2}"), "{what}: r of the choice");
                    assert!(!row[7].is_empty(), "{what}: no reason");
                }
            }
        }
    }
}

/// Bullet 3 (ADR-0041, ADR-0058): on every M10 platform, entities created per second on the M3s
/// P1 tetrahedron and hexahedron boxes, and geometric factors per second at p = 1, 4 and 8 on
/// tetrahedron and hexahedron boxes of about 10^6 points, in f32 and f64 (Metal f32 only),
/// under the policy, five runs each.
#[test]
fn throughput_recorded_on_every_platform() {
    let data = m10();
    let throughput = &data.throughput;
    assert!(
        !throughput.is_empty(),
        "no M10 throughput records (m10-throughput-*.json) under docs/perf/data/"
    );
    for r in throughput {
        let what = format!(
            "{} {} {:?} {} {}",
            r.target,
            r.backend,
            r.measure,
            r.mesh,
            r.scalar.name()
        );
        let c = Common {
            target: &r.target,
            backend: &r.backend,
            class: class_name(&r.device_class),
            profile: &r.profile,
            runs: r.times_s.len(),
            cores: &r.cores,
            load_width: r.load_width,
            fallback_reason: r.fallback_reason.as_deref(),
            node_load: &r.node_load,
        };
        check_m10_common(c, CANDIDATE_RUNS, &what);
        check_calls_per_run(r.calls_per_run, r.calibration_s, &what);
        assert_eq!(r.vector_size_override, "policy", "{what}: under the policy");
        assert!(r.items > 0 && r.median_s > 0.0, "{what}");
        let per_s = r.items as f64 / r.median_s;
        assert!(
            (r.per_s - per_s).abs() <= 1e-9 * per_s,
            "{what}: per_s {} is not items / median_s = {per_s}",
            r.per_s
        );
    }
    let present = |t: &str, b: &str| throughput.iter().any(|r| r.target == t && r.backend == b);
    for (target, backend, _) in m10_platforms(present) {
        for scalar in backend_scalars(backend, &[Scalar::F32, Scalar::F64]) {
            let mut want: Vec<(Measure, String)> = Vec::new();
            for cell in ["tetrahedron", "hexahedron"] {
                want.push((Measure::EntitiesPerSecond, format!("P1 {cell} box")));
                for p in [1, 4, 8] {
                    want.push((Measure::FactorsPerSecond, format!("P{p} {cell} box")));
                }
            }
            for (measure, mesh) in want {
                let what = format!("{target} {backend} {measure:?} {mesh} {}", scalar.name());
                let r = throughput
                    .iter()
                    .find(|r| {
                        r.target == target
                            && r.backend == backend
                            && r.measure == measure
                            && r.mesh == mesh
                            && r.scalar == scalar
                    })
                    .unwrap_or_else(|| panic!("{what}: no throughput record"));
                if measure == Measure::FactorsPerSecond {
                    assert!(
                        (500_000..=2_000_000).contains(&r.items),
                        "{what}: {} points, not about 10^6",
                        r.items
                    );
                }
            }
        }
    }
}
