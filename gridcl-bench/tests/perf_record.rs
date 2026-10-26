//! M3s acceptance (docs/design.md, "M3s SIMD vectorization and benchmarks"; ADR-0015,
//! ADR-0017, ADR-0018): the committed benchmark data, the generated report, the policy and the
//! throughput records.
//!
//! The checks of the decision rule are written here independently of `gridcl_bench::decide`:
//! they read `vector-policy.toml` with the `toml` crate and recompute the class times from the
//! raw records.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use gridcl_bench::decide::decide;
use gridcl_bench::record::{
    CubeDimMode, Measure, PerfData, Scalar, SweepRecord, ThroughputRecord, load_dir,
};
use gridcl_bench::report::{BEGIN_MARKER, END_MARKER, render};
use gridcl_device::{DeviceClass, KernelKind};

/// The decision rule's problem sizes.
const RULE_SIZES: [usize; 2] = [1_000_000, 10_000_000];
/// The protocol's problem sizes.
const SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
/// CPU block targets of the protocol.
const BLOCKS: [usize; 3] = [4 << 10, 16 << 10, 64 << 10];
/// QuadField tiles of the protocol (GeomFactors).
const TILES: [usize; 3] = [1, 4, 16];

const CLASSES: [DeviceClass; 4] = [
    DeviceClass::CpuSimd128,
    DeviceClass::CpuSimd256,
    DeviceClass::CpuSimd512,
    DeviceClass::Gpu,
];

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

fn is_cpu(class: DeviceClass) -> bool {
    class != DeviceClass::Gpu
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
        ("m3max", "cpu", DeviceClass::CpuSimd128),
        ("m3max", "metal", DeviceClass::Gpu),
        ("locust", "cpu", DeviceClass::CpuSimd128),
        (cuda, "cuda", DeviceClass::Gpu),
        (avx512, "cpu", DeviceClass::CpuSimd512),
    ];
    for (target, backend, class) in required {
        let records: Vec<&SweepRecord> = data
            .sweeps
            .iter()
            .filter(|r| r.target == target && r.backend == backend)
            .collect();
        assert!(!records.is_empty(), "no sweep data for {target} {backend}");
        for r in &records {
            assert_eq!(r.device_class, class, "{target} {backend}: device class");
        }
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

#[test]
fn report_lists_every_family_class_and_type() {
    let data = data();
    let decision = decide(&data);
    let text = read("docs/perf/vectorization.md");
    let begin = text
        .find(BEGIN_MARKER)
        .expect("docs/perf/vectorization.md has no BEGIN GENERATED marker")
        + BEGIN_MARKER.len();
    let end = text[begin..]
        .find(END_MARKER)
        .expect("docs/perf/vectorization.md has no END GENERATED marker")
        + begin;
    let generated = text[begin..end].trim();
    assert_eq!(
        generated,
        render(&data, &decision).trim(),
        "the generated tables are stale: run policy_report"
    );
    for (kind, scalars) in kinds() {
        if kind == KernelKind::Scalar {
            continue; // not a candidate family
        }
        for class in CLASSES {
            for scalar in &scalars {
                let row = format!("| {kind:?} | {class:?} | {} |", scalar.name());
                assert!(generated.contains(&row), "no row {row}");
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
    class: DeviceClass,
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
) -> BTreeMap<Config, f64> {
    let cpu = is_cpu(class);
    let mut per_device: BTreeMap<(String, String), BTreeMap<Config, f64>> = BTreeMap::new();
    let mut neutral = Vec::new();
    for r in data.sweeps.iter().filter(|r| {
        r.device_class == class
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
    let text = read("gridcl-device/vector-policy.toml");
    assert_eq!(
        text,
        decide(&data).to_toml(),
        "vector-policy.toml differs from the decision rule: run policy_report"
    );

    let policy: toml::Table = text.parse().expect("vector-policy.toml parses");
    let report = read("docs/perf/vectorization.md");
    // CpuSimd256 copies the CpuSimd512 table ("Benchmarks", "Where").
    assert_eq!(
        policy.get("CpuSimd256"),
        policy.get("CpuSimd512"),
        "CpuSimd256 copies CpuSimd512"
    );
    for class in [
        DeviceClass::CpuSimd128,
        DeviceClass::CpuSimd512,
        DeviceClass::Gpu,
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
                if chosen_worst <= 1.03 {
                    continue;
                }
                // The backstop (ADR-0018): no admitted configuration within 3% for every type at
                // both sizes; the chosen one is the closest, at most 10% slower, and marked. A
                // shared entry with more than one type measured may go up to 25%, marked "closest
                // across types" above 10% (ADR-0021).
                let candidates: Vec<Config> = views[0].times[0]
                    .keys()
                    .copied()
                    .filter(|c| entry.contains(&c.0))
                    .collect();
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
    fn new(data: &PerfData, class: DeviceClass, kind: KernelKind, scalar: Scalar) -> TypeView {
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
