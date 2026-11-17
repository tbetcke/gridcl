//! Unit tests of the generic rule on synthetic records (ADR-0058 item 4).

use super::*;
use crate::record::{Cores, SweepRecord};

/// A candidate record: `config` of `kind` in `scalar` on (`target`, `backend`) at `size`,
/// median `t`. CPU records on all hardware threads unless `config.units` is `Performance`;
/// `perf` is the device's performance-core count.
fn rec(
    (target, backend): (&str, &str),
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
    config: VectorConfig,
    t: f64,
    perf: Option<u32>,
) -> SweepRecord {
    let class = if backend == "cpu" {
        DeviceClass::Cpu
    } else {
        DeviceClass::Gpu
    };
    SweepRecord {
        backend: backend.to_owned(),
        device: "test".to_owned(),
        device_class: class,
        legacy_class: crate::record::LegacyClass::of_new(class),
        target: target.to_owned(),
        family: kind,
        scalar,
        nominal_n: size,
        n: size,
        vector_size: config.vector_size,
        vector_size_ran: config.vector_size,
        tile: config.tile,
        block_bytes: config.block_bytes,
        cube_dim: if config.units == Some(CpuUnits::Performance) {
            CubeDimMode::PerformanceCores
        } else {
            CubeDimMode::Plan
        },
        units: 1,
        times_s: vec![t; 5],
        median_s: t,
        calls_per_run: 1,
        calibration_s: 0.02,
        compile_s: 0.0,
        kernel_cache: "off".to_owned(),
        bandwidth_ratio: None,
        commit: "test".to_owned(),
        profile: "release".to_owned(),
        cores: Cores {
            performance: perf,
            efficiency: None,
            threads: 8,
            levels: Vec::new(),
        },
        load_width: Some(128),
        fallback_reason: None,
        node_load: None,
    }
}

const METAL: (&str, &str) = ("macmini", "metal");
const CUDA: (&str, &str) = ("locust", "cuda");

/// The Gpu baseline of `kind` and `bytes` at vector size `n`.
fn gpu(kind: KernelKind, bytes: usize, n: usize) -> VectorConfig {
    VectorConfig {
        vector_size: n,
        ..vector_baseline(DeviceClass::Gpu, kind, bytes)
    }
}

/// Records of `kind` in `scalar` on `platform` at both rule sizes: (vector size, time).
fn gpu_records(
    platform: (&str, &str),
    kind: KernelKind,
    scalar: Scalar,
    times: &[(usize, f64)],
) -> Vec<SweepRecord> {
    CANDIDATE_SIZES
        .iter()
        .flat_map(|&size| {
            times.iter().map(move |&(n, t)| {
                rec(
                    platform,
                    kind,
                    scalar,
                    size,
                    gpu(kind, scalar.bytes(), n),
                    t,
                    None,
                )
            })
        })
        .collect()
}

fn decide_of(records: Vec<SweepRecord>) -> VectorDecision {
    decide_vector(&M10Data {
        vector: records,
        ..M10Data::default()
    })
}

fn entry(d: &VectorDecision, class: DeviceClass, kind: KernelKind, bytes: usize) -> &VectorEntry {
    d.entry(class, kind, bytes).expect("an entry")
}

/// A candidate 8% faster than the baseline is not admitted, even where it would be the
/// simplest within 3% of the fastest; one 10% faster is.
#[test]
fn admission_needs_ten_percent_by_the_geometric_mean() {
    let k = KernelKind::EVector;
    let times = [(1, 1.0), (2, 0.92), (4, 0.90)];
    let mut records = gpu_records(METAL, k, Scalar::F32, &times);
    records.extend(gpu_records(CUDA, k, Scalar::F32, &times));
    let d = decide_of(records);
    let e = entry(&d, DeviceClass::Gpu, k, 4);
    assert_eq!(e.fit, GenericFit::Bound);
    assert_eq!(e.choice.vector_size, 4, "{}", e.reason);
    assert_eq!(e.baseline.vector_size, 1);
}

/// A candidate that is fastest on one platform and above 1.25 on another is not kept.
#[test]
fn the_bound_holds_on_every_platform() {
    let k = KernelKind::EVector;
    let mut records = gpu_records(CUDA, k, Scalar::F32, &[(1, 1.0), (4, 0.5), (8, 0.6)]);
    records.extend(gpu_records(
        METAL,
        k,
        Scalar::F32,
        &[(1, 1.0), (4, 1.3), (8, 1.0)],
    ));
    // Size 2 completes the metal minimum below the baseline: fastest on metal is 0.95.
    records.extend(gpu_records(CUDA, k, Scalar::F32, &[(2, 0.8)]));
    records.extend(gpu_records(METAL, k, Scalar::F32, &[(2, 0.95)]));
    let d = decide_of(records);
    let e = entry(&d, DeviceClass::Gpu, k, 4);
    assert_eq!(
        (e.fit, e.choice.vector_size),
        (GenericFit::Bound, 8),
        "{}",
        e.reason
    );
    let r: Vec<(&str, String)> = e
        .platforms
        .iter()
        .map(|p| (p.platform.backend, format!("{:.3}", p.r)))
        .collect();
    assert_eq!(
        r,
        [("metal", "1.053".to_owned()), ("cuda", "1.200".to_owned())]
    );
    let best: Vec<usize> = e.platforms.iter().map(|p| p.best.vector_size).collect();
    assert_eq!(best, [2, 4], "each platform's best");
}

/// With no candidate within the bound, the admitted candidate with the smallest worst case.
#[test]
fn fallback_takes_the_smallest_worst_case() {
    let k = KernelKind::EVector;
    let mut records = gpu_records(CUDA, k, Scalar::F32, &[(1, 1.0), (4, 0.5)]);
    records.extend(gpu_records(METAL, k, Scalar::F32, &[(1, 1.0), (4, 1.5)]));
    let d = decide_of(records);
    let e = entry(&d, DeviceClass::Gpu, k, 4);
    assert_eq!(e.choice.vector_size, 4);
    match e.fit {
        GenericFit::Fallback { worst } => assert!((worst - 1.5).abs() < 1e-12),
        f => panic!("{f:?}"),
    }
    assert!(e.reason.contains("worst case"), "{}", e.reason);
}

/// Of the kept candidates within 3% of the smallest geometric mean, the smallest vector size.
#[test]
fn the_simplest_within_three_percent_wins() {
    let k = KernelKind::Reduce;
    let times = [(1, 1.0), (2, 0.80), (4, 0.79), (8, 0.70)];
    let d = decide_of(gpu_records(METAL, k, Scalar::F32, &times));
    assert_eq!(entry(&d, DeviceClass::Gpu, k, 4).choice.vector_size, 8);
    let times = [(1, 1.0), (2, 0.80), (4, 0.79), (8, 0.785)];
    let d = decide_of(gpu_records(METAL, k, Scalar::F32, &times));
    assert_eq!(entry(&d, DeviceClass::Gpu, k, 4).choice.vector_size, 2);
}

/// A shared 4-byte entry counts f32 and u32 as members: a size good for f32 alone may miss
/// the bound for u32.
#[test]
fn shared_entries_count_each_four_byte_type() {
    let k = KernelKind::Fill;
    let f32_times = [(1, 1.0), (2, 0.80), (4, 0.70)];
    let alone = decide_of(gpu_records(METAL, k, Scalar::F32, &f32_times));
    assert_eq!(entry(&alone, DeviceClass::Gpu, k, 4).choice.vector_size, 4);
    let mut records = gpu_records(METAL, k, Scalar::F32, &f32_times);
    records.extend(gpu_records(
        METAL,
        k,
        Scalar::U32,
        &[(1, 1.0), (2, 0.85), (4, 1.10)],
    ));
    let shared = decide_of(records);
    let e = entry(&shared, DeviceClass::Gpu, k, 4);
    assert_eq!(
        (e.fit, e.choice.vector_size),
        (GenericFit::Bound, 2),
        "{}",
        e.reason
    );
    assert_eq!(e.platforms.len(), 1, "one platform, two members");
    // The 8-byte entry has no record of its own.
    assert_eq!(
        entry(&shared, DeviceClass::Gpu, k, 8).fit,
        GenericFit::Default
    );
}

/// Equal times on all threads and on the performance cores go to all threads; a device
/// without a performance-core count gives its all-threads time to both.
#[test]
fn unit_ties_go_to_all_threads() {
    let k = KernelKind::Gather;
    let base = vector_baseline(DeviceClass::Cpu, k, 8);
    let with = |units| VectorConfig {
        units: Some(units),
        ..base
    };
    let mut records = Vec::new();
    for size in CANDIDATE_SIZES {
        for units in [CpuUnits::All, CpuUnits::Performance] {
            let t = if units == base.units.unwrap() {
                1.0
            } else {
                0.8
            };
            records.push(rec(
                ("macmini", "cpu"),
                k,
                Scalar::F64,
                size,
                with(units),
                t,
                Some(4),
            ));
        }
        let mac_all = with(CpuUnits::All);
        records.push(rec(
            ("kathleen", "cpu"),
            k,
            Scalar::F64,
            size,
            mac_all,
            1.0,
            None,
        ));
    }
    let d = decide_of(records);
    let e = entry(&d, DeviceClass::Cpu, k, 8);
    // Gather 8's baseline is on the performance cores (CpuSimd128): all threads, 20% faster on
    // the Mac and the same on Kathleen, is admitted (geometric mean 0.894) and chosen.
    assert_eq!(base.units, Some(CpuUnits::Performance));
    assert_eq!(e.choice, with(CpuUnits::All), "{}", e.reason);
    let mut equal = Vec::new();
    for size in CANDIDATE_SIZES {
        for units in [CpuUnits::All, CpuUnits::Performance] {
            equal.push(rec(
                ("macmini", "cpu"),
                k,
                Scalar::F64,
                size,
                with(units),
                1.0,
                Some(4),
            ));
        }
    }
    let d = decide_of(equal);
    assert_eq!(
        entry(&d, DeviceClass::Cpu, k, 8).choice,
        base,
        "the baseline stays"
    );
}

/// A later record of a configuration replaces an earlier one; a record of the A4000 does not
/// vote; Myriad's A100 stands in for an absent Locust H100.
#[test]
fn records_platforms_and_fallbacks() {
    let k = KernelKind::EVector;
    let mut records = gpu_records(METAL, k, Scalar::F32, &[(1, 1.0), (4, 0.5)]);
    records.extend(gpu_records(METAL, k, Scalar::F32, &[(4, 2.0)]));
    records.extend(gpu_records(
        ("a4000", "cuda"),
        k,
        Scalar::F32,
        &[(1, 1.0), (4, 0.1)],
    ));
    let d = decide_of(records.clone());
    let e = entry(&d, DeviceClass::Gpu, k, 4);
    assert_eq!(e.choice.vector_size, 1, "the repeat at 2.0 replaces 0.5");
    assert_eq!(e.platforms.len(), 1, "no A4000 member");
    records.extend(gpu_records(
        ("myriad-a100", "cuda"),
        k,
        Scalar::F32,
        &[(1, 1.0), (4, 0.5)],
    ));
    let d = decide_of(records);
    let targets: Vec<&str> = entry(&d, DeviceClass::Gpu, k, 4)
        .platforms
        .iter()
        .map(|p| p.platform.target)
        .collect();
    assert_eq!(targets, ["macmini", "myriad-a100"]);
}

/// Without records every entry is the default, and the TOML holds exactly the four sets.
#[test]
fn defaults_and_toml_without_records() {
    let d = decide_of(Vec::new());
    assert!(d.entries.iter().all(|e| e.fit == GenericFit::Default));
    assert!(
        d.entries
            .iter()
            .all(|e| e.choice.vector_size == 1 && e.choice.units != Some(CpuUnits::Performance))
    );
    let text = d.to_toml();
    let table: toml::Table = text.parse().unwrap();
    let classes: Vec<&String> = table.keys().collect();
    assert_eq!(classes, ["Cpu", "Gpu"]);
    let cpu = table["Cpu"].as_table().unwrap();
    let keys: BTreeSet<&str> = cpu.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        BTreeSet::from(["quad_tile", "vector_size", "cpu_block", "cpu_units"])
    );
    let gpu = table["Gpu"].as_table().unwrap();
    let keys: BTreeSet<&str> = gpu.keys().map(String::as_str).collect();
    assert_eq!(keys, BTreeSet::from(["quad_tile", "vector_size"]));
    for kind in KINDS {
        let k = format!("{kind:?}");
        for b in ["4", "8"] {
            assert_eq!(cpu["vector_size"][&k][b].as_integer(), Some(1), "{k} {b}");
            assert!(cpu["cpu_block"][&k][b].as_integer().is_some(), "{k} {b}");
            assert_eq!(gpu["vector_size"][&k][b].as_integer(), Some(1), "{k} {b}");
        }
    }
    // The interim policy's tiles: the CpuSimd128 and Gpu baselines.
    assert_eq!(cpu["quad_tile"]["4"].as_integer(), Some(4));
    assert_eq!(gpu["quad_tile"]["8"].as_integer(), Some(16));
}

mod prop {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// The choice does not depend on the order of the members (platforms).
        #[test]
        fn choice_does_not_depend_on_platform_order(
            times in proptest::collection::vec(0.2f64..1.0, 4 * 5 * 2),
            members in 1usize..=4,
            candidates in 1usize..=5,
            rotate in 0usize..4,
        ) {
            let t = |c: usize, m: usize, s: usize| times[(m * 5 + c) * 2 + s];
            let order: Vec<usize> = (0..members).map(|m| (m + rotate) % members).collect();
            let keys: Vec<usize> = (0..candidates).collect();
            let a = generic_rule(0, &keys, members, 2, t);
            let b = generic_rule(0, &keys, members, 2, |c, m, s| t(c, order[m], s));
            prop_assert_eq!(a.choice, b.choice);
            prop_assert_eq!(a.fallback, b.fallback);
            prop_assert_eq!(a.admitted, b.admitted);
            prop_assert!((a.worst - b.worst).abs() <= 1e-12 * a.worst);
        }
    }
}
