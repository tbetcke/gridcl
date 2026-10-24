//! M3s acceptance (docs/design.md, "M3s SIMD vectorization and benchmarks"; ADR-0015): the
//! committed benchmark data, the generated report, the policy and the throughput records.
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

/// A configuration: (requested vector size, tile, CPU block in bytes).
type Config = (usize, Option<usize>, Option<usize>);

/// The class time of every configuration of (class, kind, scalar) at `size`: the geometric
/// mean over the class's devices of their medians, for the configurations every device
/// measured (ADR-0015, item 5). Ratios of class times are geometric means of per-device ratios.
fn class_times(
    data: &PerfData,
    class: DeviceClass,
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
) -> BTreeMap<Config, f64> {
    let mut per_device: BTreeMap<(String, String), BTreeMap<Config, f64>> = BTreeMap::new();
    for r in data.sweeps.iter().filter(|r| {
        r.device_class == class
            && r.family == kind
            && r.scalar == scalar
            && r.nominal_n == size
            && r.cube_dim == CubeDimMode::Plan
    }) {
        let config = (r.vector_size, r.tile, r.block_bytes);
        let times = per_device
            .entry((r.target.clone(), r.backend.clone()))
            .or_default();
        assert!(
            times.insert(config, r.median_s).is_none(),
            "{} {} {kind:?} {} at {size}: configuration {config:?} recorded twice",
            r.target,
            r.backend,
            scalar.name()
        );
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
        .filter(|((n, _, _), _)| keep(*n))
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
                let chosen: Config = (n, tile, block);
                let with_data: Vec<Scalar> = scalars
                    .iter()
                    .copied()
                    .filter(|s| s.bytes() == bytes)
                    .filter(|&s| !class_times(&data, class, kind, s, RULE_SIZES[0]).is_empty())
                    .collect();
                if with_data.is_empty() {
                    assert_eq!(n, 1, "{what}: vector size above 1 without data");
                    continue;
                }
                // Every 4-byte type measured must admit a shared entry (ADR-0015, item 6).
                for scalar in with_data {
                    let what = format!("{what} ({})", scalar.name());
                    let times: Vec<BTreeMap<Config, f64>> = RULE_SIZES
                        .iter()
                        .map(|&size| class_times(&data, class, kind, scalar, size))
                        .collect();
                    let speedup = |times: &BTreeMap<Config, f64>, v: usize| -> Option<f64> {
                        Some(best(times, |m| m == 1)? / best(times, |m| m == v)?)
                    };
                    // Admitted sizes: 1, and every size at least 10% faster at both sizes.
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
                    if n > 1 {
                        for (t, size) in times.iter().zip(RULE_SIZES) {
                            let s = speedup(t, n).unwrap_or(0.0);
                            assert!(
                                s >= 1.1,
                                "{what}: size {n} is {s:.3}x size 1 at {size}, not 1.1x"
                            );
                        }
                    }
                    for (t, size) in times.iter().zip(RULE_SIZES) {
                        let fastest = best(t, |v| admitted.contains(&v)).unwrap();
                        let got = *t.get(&chosen).unwrap_or_else(|| {
                            panic!("{what}: the chosen {chosen:?} was not measured at {size}")
                        });
                        assert!(
                            got <= 1.03 * fastest,
                            "{what}: the chosen {chosen:?} takes {got:.3e} s at {size}, more \
                             than 3% above the fastest admitted {fastest:.3e} s"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn throughput_recorded_with_policy_and_scalar() {
    let data = data();
    for r in &data.throughput {
        check_idle(
            &r.target,
            &r.node_load,
            &format!("{} {:?}", r.target, r.measure),
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
