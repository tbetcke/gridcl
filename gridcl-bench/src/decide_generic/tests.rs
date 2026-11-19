//! Unit tests of the generic rule on synthetic records (ADR-0058 item 4).

use super::*;
use crate::record::{Cores, DegreeRecord, SweepRecord};

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
        seed: None,
        positions: None,
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
    decide_vector(
        &M10Data {
            vector: records,
            ..M10Data::default()
        },
        GenericRule::M10,
    )
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
        (e.fit.clone(), e.choice.vector_size),
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
    match &e.fit {
        GenericFit::Fallback { worst, .. } => assert!((worst - 1.5).abs() < 1e-12),
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
        (e.fit.clone(), e.choice.vector_size),
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

// ---------------------------------------------------------------------------------------------
// The degree rule.

const MAC: (&str, &str) = ("macmini", "cpu");
const GRACE: (&str, &str) = ("locust", "cpu");
const KATHLEEN: (&str, &str) = ("kathleen", "cpu");

/// A timed degree record of the bench family `family` with `strategy` on `platform`: `t`
/// seconds per point at `size` points, first use `compile` seconds.
#[allow(clippy::too_many_arguments)]
fn drec(
    (target, backend): (&str, &str),
    family: &str,
    cell: &str,
    degree: u32,
    scalar: Scalar,
    size: usize,
    strategy: &str,
    t: f64,
) -> DegreeRecord {
    let class = if backend == "cpu" {
        DeviceClass::Cpu
    } else {
        DeviceClass::Gpu
    };
    DegreeRecord {
        backend: backend.to_owned(),
        device: "test".to_owned(),
        device_class: class,
        legacy_class: crate::record::LegacyClass::of_new(class),
        target: target.to_owned(),
        run_id: "test".to_owned(),
        family: family.to_owned(),
        cell: cell.to_owned(),
        gdim: crate::decide_degree::gdim_of(family, cell),
        degree,
        node_family: "equispaced".to_owned(),
        scalar,
        strategy: strategy.to_owned(),
        strategy_ran: strategy.to_owned(),
        nominal_points: size,
        cells: size / 10,
        points_per_cell: 10,
        iterations: (family == "PullBack").then_some(9),
        vector_size: 1,
        vector_size_ran: 1,
        policy_vector_size: 1,
        tile: 4,
        block_bytes: None,
        units_per_group: 1,
        status: "ok".to_owned(),
        launch_only: false,
        interleaved: backend == "cpu",
        check_error: Some(1e-15),
        check_tolerance: Some(1e-12),
        times_s: vec![t * size as f64; 5],
        median_s: t * size as f64,
        calls_per_run: 1,
        calibration_s: 0.02,
        compile_s: 1.0,
        kernel_cache: "off".to_owned(),
        commit: "test".to_owned(),
        profile: "release".to_owned(),
        cores: Cores {
            performance: None,
            efficiency: None,
            threads: 8,
            levels: Vec::new(),
        },
        load_width: Some(128),
        fallback_reason: None,
        node_load: None,
    }
}

/// Points records on hexahedra in f64 at `degree` on `platform`, at `sizes`: (strategy, time).
fn points(
    platform: (&str, &str),
    degree: u32,
    sizes: &[usize],
    times: &[(&str, f64)],
) -> Vec<DegreeRecord> {
    sizes
        .iter()
        .flat_map(|&n| {
            times.iter().map(move |&(s, t)| {
                drec(
                    platform,
                    "Points",
                    "Hexahedron",
                    degree,
                    Scalar::F64,
                    n,
                    s,
                    t,
                )
            })
        })
        .collect()
}

/// M10 records of Points on hexahedra in f64 at `degree` on the three Cpu platforms.
fn m10_points(degree: u32, times: &[(&str, f64)]) -> Vec<DegreeRecord> {
    [MAC, GRACE, KATHLEEN]
        .into_iter()
        .flat_map(|d| points(d, degree, &CANDIDATE_SIZES, times))
        .collect()
}

/// M9h records of Points on hexahedra in f64 at `degree` on M9h's Cpu platforms (no Grace), at
/// 10^6 points.
fn m9h_points(degree: u32, times: &[(&str, f64)]) -> Vec<DegreeRecord> {
    [MAC, KATHLEEN]
        .into_iter()
        .flat_map(|d| points(d, degree, &CANDIDATE_SIZES[..1], times))
        .collect()
}

fn decide_degree_of(m9h: Vec<DegreeRecord>, m10: Vec<DegreeRecord>) -> DegreeDecision {
    decide_degree(
        &HighOrderData {
            degree: m9h,
            ..HighOrderData::default()
        },
        &M10Data {
            degree: m10,
            ..M10Data::default()
        },
        GenericRule::M10,
    )
}

fn cpu_points(d: &DegreeDecision) -> &DegreeEntry {
    d.entry(DeviceClass::Cpu, "Points", "Hexahedron", Scalar::F64)
        .expect("an entry")
}

/// Every M10 degree and every M9h degree in between, from the given records: sumfact 20%
/// faster than direct everywhere except at `direct_at`, where direct wins.
fn full_points(direct_at: &[u32]) -> (Vec<DegreeRecord>, Vec<DegreeRecord>) {
    let times = |p: u32| -> Vec<(&'static str, f64)> {
        let fast = if direct_at.contains(&p) { 0.5 } else { 1.0 };
        let mut t = vec![("loops", 1.0), ("sumfact", 0.8)];
        if p <= 4 {
            t.push(("direct", fast));
        }
        t
    };
    let mut m9h = Vec::new();
    let mut m10 = Vec::new();
    for p in 1..=8 {
        if M10_DEGREES.contains(&p) {
            m10.extend(m10_points(p, &times(p)));
        } else {
            m9h.extend(m9h_points(p, &times(p)));
        }
    }
    (m9h, m10)
}

/// p = 1, 4 and 8 come from M10 on its three Cpu platforms at 10^6 and 10^7; the degrees in
/// between from M9h on the Mac and Kathleen at 10^6; the ranges are maximal runs.
#[test]
fn degrees_in_between_come_from_m9h() {
    let (m9h, m10) = full_points(&[2, 3]);
    let d = decide_degree_of(m9h, m10);
    let e = cpu_points(&d);
    for k in &e.picks {
        let m10_degree = M10_DEGREES.contains(&k.degree);
        let want = if m10_degree {
            DegreeSource::M10
        } else {
            DegreeSource::M9h
        };
        assert_eq!(k.source, want, "p = {}", k.degree);
        let targets: Vec<&str> = k.platforms.iter().map(|p| p.platform.target).collect();
        if m10_degree {
            assert_eq!(
                targets,
                ["macmini", "locust", "kathleen"],
                "p = {}",
                k.degree
            );
            assert_eq!(k.sizes, CANDIDATE_SIZES);
        } else {
            assert_eq!(targets, ["macmini", "kathleen"], "p = {}", k.degree);
            assert_eq!(k.sizes, [1_000_000]);
        }
    }
    assert_eq!(
        e.ranges,
        [(1, "sumfact"), (2, "direct"), (4, "sumfact")],
        "{:?}",
        e.picks
    );
    assert_eq!(
        e.spans(),
        [
            (1, Some(1), "sumfact"),
            (2, Some(3), "direct"),
            (4, None, "sumfact")
        ]
    );
}

/// A `direct` choice at p = 8 is followed by an open-ended `loops` range; without data an entry
/// is `loops` throughout.
#[test]
fn the_last_range_is_open_ended_and_compiles() {
    let mut m10 = Vec::new();
    for p in M10_DEGREES {
        m10.extend(m10_points(p, &[("direct", 0.5), ("loops", 1.0)]));
    }
    let d = decide_degree_of(Vec::new(), m10);
    let e = cpu_points(&d);
    assert_eq!(e.strategy(8), "direct");
    assert_eq!(e.ranges.last(), Some(&(9, "loops")));
    // p = 2, 3 and 5–7 have no data: loops, which splits the direct runs.
    assert_eq!(e.pick(2).unwrap().fit, GenericFit::Default);
    assert_eq!(
        e.ranges,
        [
            (1, "direct"),
            (2, "loops"),
            (4, "direct"),
            (5, "loops"),
            (8, "direct"),
            (9, "loops")
        ]
    );
    let none = decide_degree_of(Vec::new(), Vec::new());
    assert!(none.entries.iter().all(|e| e.ranges == [(1, "loops")]));
}

/// A strategy that took over 30 s to compile and load on one platform of the class is not
/// selectable, even where it is the fastest; with `direct` out, `loops` is the baseline.
#[test]
fn slow_first_use_is_not_selectable() {
    let mut m10 = m10_points(4, &[("direct", 1.0), ("loops", 0.95), ("sumfact", 0.5)]);
    for r in m10
        .iter_mut()
        .filter(|r| r.target == "locust" && r.strategy == "sumfact")
    {
        r.compile_s = 31.0;
    }
    let d = decide_degree_of(Vec::new(), m10.clone());
    let k = cpu_points(&d).pick(4).unwrap();
    assert_eq!(k.selectable, ["direct", "loops"]);
    assert_eq!((k.base, k.strategy), ("direct", "direct"));
    for r in m10
        .iter_mut()
        .filter(|r| r.target == "macmini" && r.strategy == "direct")
    {
        r.status = "failed: test".to_owned();
    }
    let d = decide_degree_of(Vec::new(), m10);
    let k = cpu_points(&d).pick(4).unwrap();
    assert_eq!(k.selectable, ["loops"]);
    assert_eq!((k.base, k.strategy), ("loops", "loops"));
}

/// M9h records whose check error is not finite are left out (ADR-0060 item 3).
#[test]
fn non_finite_checks_are_left_out() {
    let mut m9h = m9h_points(2, &[("direct", 1.0), ("loops", 1.0), ("sumfact", 0.5)]);
    let d = decide_degree_of(m9h.clone(), Vec::new());
    assert_eq!(cpu_points(&d).strategy(2), "sumfact");
    for r in m9h
        .iter_mut()
        .filter(|r| r.strategy == "sumfact" && r.target == "kathleen")
    {
        r.check_error = Some(f64::NAN);
    }
    let d = decide_degree_of(m9h.clone(), Vec::new());
    let k = cpu_points(&d).pick(2).unwrap();
    assert_eq!(k.source, DegreeSource::M9h);
    assert_eq!(k.selectable, ["direct", "loops"]);
    assert_eq!(k.strategy, "direct");
    for r in m9h.iter_mut().filter(|r| r.strategy == "sumfact") {
        r.check_error = Some(f64::INFINITY);
    }
    let d = decide_degree_of(m9h, Vec::new());
    assert_eq!(
        cpu_points(&d).pick(2).unwrap().selectable,
        ["direct", "loops"]
    );
}

/// The generic rule at one degree: admission by the geometric mean, the bound on every
/// platform, the fallback with the smallest worst case.
#[test]
fn degree_picks_follow_the_generic_rule() {
    // sumfact is 2x faster on the Mac and Kathleen, 1.3x slower on Grace: admitted by the
    // geometric mean (0.5 · 0.5 · 1.3)^(1/3) ≈ 0.69, but r = 1.3 on Grace breaks the bound;
    // direct has r = 2 on two platforms: no strategy meets the bound, and the smallest worst
    // case is sumfact's 1.3.
    let mut m10 = Vec::new();
    for (d, s) in [(MAC, 0.5), (GRACE, 1.3), (KATHLEEN, 0.5)] {
        m10.extend(points(
            d,
            1,
            &CANDIDATE_SIZES,
            &[("direct", 1.0), ("sumfact", s)],
        ));
    }
    let d = decide_degree_of(Vec::new(), m10);
    let k = cpu_points(&d).pick(1).unwrap();
    assert_eq!(k.strategy, "sumfact");
    match &k.fit {
        GenericFit::Fallback { worst, .. } => assert!((worst - 1.3).abs() < 1e-12, "{worst}"),
        f => panic!("{f:?}"),
    }
    assert!(k.reason.contains("worst case r = 1.30"), "{}", k.reason);
    let r: Vec<(&str, &str, f64)> = k
        .platforms
        .iter()
        .map(|p| (p.platform.target, p.best, p.r))
        .collect();
    assert_eq!(r[1].0, "locust");
    assert_eq!(r[1].1, "direct");
    assert!((r[1].2 - 1.3).abs() < 1e-12);
    // 8% faster everywhere: not admitted.
    let d = decide_degree_of(
        Vec::new(),
        m10_points(1, &[("direct", 1.0), ("sumfact", 0.92)]),
    );
    let k = cpu_points(&d).pick(1).unwrap();
    assert_eq!((k.strategy, k.fit.clone()), ("direct", GenericFit::Bound));
    assert!(k.reason.starts_with("direct: no strategy"), "{}", k.reason);
}

/// On Gpu every strategy is a candidate, and Metal measures f32 only.
#[test]
fn gpu_entries_take_every_strategy() {
    let times = [
        ("direct", 1.0),
        ("loops", 0.9),
        ("sumfact", 0.8),
        ("shared", 0.5),
        ("cooperative", 0.51),
    ];
    let mut m10 = Vec::new();
    for d in [METAL, CUDA] {
        for n in CANDIDATE_SIZES {
            for &(s, t) in &times {
                m10.push(drec(d, "Points", "Quadrilateral", 4, Scalar::F32, n, s, t));
                m10.push(drec(
                    d,
                    "Points",
                    "Quadrilateral",
                    4,
                    Scalar::F64,
                    n,
                    s,
                    t * 2.0,
                ));
            }
        }
    }
    let d = decide_degree_of(Vec::new(), m10);
    let f32 = d
        .entry(DeviceClass::Gpu, "Points", "Quadrilateral", Scalar::F32)
        .unwrap()
        .pick(4)
        .unwrap();
    assert_eq!(f32.selectable, STRATEGIES);
    assert_eq!(
        f32.strategy, "shared",
        "cooperative is within 3% but less simple"
    );
    assert_eq!(f32.platforms.len(), 2);
    let f64 = d
        .entry(DeviceClass::Gpu, "Points", "Quadrilateral", Scalar::F64)
        .unwrap()
        .pick(4)
        .unwrap();
    let targets: Vec<&str> = f64.platforms.iter().map(|p| p.platform.target).collect();
    assert_eq!(targets, ["locust"], "Metal has no f64");
}

/// The Jacobian family sums J, det J and K; a member missing on one platform makes the
/// strategy unselectable.
#[test]
fn policy_families_sum_their_members() {
    let mut m10 = Vec::new();
    for d in [MAC, GRACE, KATHLEEN] {
        for n in CANDIDATE_SIZES {
            for f in ["Jacobian", "DetJ", "Pinv"] {
                m10.push(drec(d, f, "Hexahedron", 1, Scalar::F64, n, "direct", 1.0));
                if !(f == "Pinv" && d == GRACE) {
                    m10.push(drec(d, f, "Hexahedron", 1, Scalar::F64, n, "sumfact", 0.5));
                }
            }
        }
    }
    let d = decide_degree_of(Vec::new(), m10);
    let k = d
        .entry(DeviceClass::Cpu, "Jacobian", "Hexahedron", Scalar::F64)
        .unwrap()
        .pick(1)
        .unwrap();
    assert_eq!(k.selectable, ["direct"]);
    assert_eq!(k.strategy, "direct");
}

/// A degree M10 re-measured (ADR-0060 item 3) replaces M9h's choice there, with M10's
/// platforms and sizes.
#[test]
fn remeasured_degrees_replace_m9h() {
    let (mut m9h, mut m10) = full_points(&[]);
    m9h.extend(m9h_points(2, &[("direct", 0.1)]));
    let d = decide_degree_of(m9h.clone(), m10.clone());
    assert_eq!(cpu_points(&d).strategy(2), "direct");
    m10.extend(m10_points(
        2,
        &[("direct", 1.0), ("loops", 1.0), ("sumfact", 0.8)],
    ));
    let d = decide_degree_of(m9h, m10);
    let k = cpu_points(&d).pick(2).unwrap();
    assert_eq!((k.source, k.strategy), (DegreeSource::M10, "sumfact"));
    assert_eq!(cpu_points(&d).ranges, [(1, "sumfact")]);
}

/// A check whose time at 10^6 points moved by more than 10% from M9h's, or that falls back,
/// lists its range's unmeasured first and last degree for re-measurement.
#[test]
fn moves_and_fallbacks_mark_ranges_for_remeasurement() {
    // Ranges: sumfact 1, direct 2–3, sumfact 4–8.
    let (mut m9h, m10) = full_points(&[2, 3]);
    m9h.extend(m9h_points(
        4,
        &[("direct", 1.0), ("loops", 1.0), ("sumfact", 0.8)],
    ));
    let d = decide_degree_of(m9h.clone(), m10.clone());
    let e = cpu_points(&d);
    // M10 and M9h agree at p = 4 on the Mac and Kathleen, for all three strategies; Grace has
    // no M9h record.
    let at4: Vec<&DegreeMove> = e.moves.iter().filter(|m| m.degree == 4).collect();
    assert_eq!(at4.len(), 6);
    assert!(
        at4.iter()
            .all(|m| !m.moved() && m.platform.target != "locust")
    );
    assert!(e.remeasure.is_empty(), "{:?}", e.remeasure);
    // sumfact at p = 4 on Kathleen was 20% slower in M9h: the boundary 3 | 4 of the range 4–8
    // is re-measured at 3 (4 is M10's).
    for r in m9h
        .iter_mut()
        .filter(|r| r.degree == 4 && r.target == "kathleen" && r.strategy == "sumfact")
    {
        r.median_s *= 1.2;
    }
    let d = decide_degree_of(m9h, m10);
    let e = cpu_points(&d);
    let moved: Vec<(&str, &str)> = e
        .moves
        .iter()
        .filter(|m| m.moved())
        .map(|m| (m.platform.target, m.strategy))
        .collect();
    assert_eq!(moved, [("kathleen", "sumfact")]);
    assert_eq!(e.remeasure, [3]);
    // A fallback at p = 1 in the range 1–3: its boundary 3 | 4 is re-measured at 3.
    let (m9h, mut m10) = full_points(&[1, 2, 3]);
    for r in m10
        .iter_mut()
        .filter(|r| r.degree == 1 && r.target == "locust" && r.strategy == "direct")
    {
        r.median_s *= 3.0;
    }
    let d = decide_degree_of(m9h, m10);
    let e = cpu_points(&d);
    assert!(
        matches!(e.pick(1).unwrap().fit, GenericFit::Fallback { .. }),
        "{:?}",
        e.pick(1)
    );
    assert_eq!(e.ranges[0], (1, "direct"));
    assert_eq!(e.remeasure, [3]);
}

/// The TOML holds exactly `[Cpu.<family>]` and `[Gpu.<family>]`, every cell type of each
/// family, 4- and 8-byte ranges, and reads back.
#[test]
fn degree_toml_holds_the_four_sets() {
    let (m9h, m10) = full_points(&[2, 3]);
    let d = decide_degree_of(m9h, m10);
    let text = d.to_toml();
    let table: toml::Table = text.parse().unwrap();
    let classes: Vec<&String> = table.keys().collect();
    assert_eq!(classes, ["Cpu", "Gpu"]);
    for class in ["Cpu", "Gpu"] {
        let families: BTreeSet<&str> = table[class]
            .as_table()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(families, BTreeSet::from(POLICY_FAMILIES));
        for family in POLICY_FAMILIES {
            let cells: BTreeSet<&str> = table[class][family]
                .as_table()
                .unwrap()
                .keys()
                .map(String::as_str)
                .collect();
            assert_eq!(
                cells,
                cells_of(family).iter().copied().collect(),
                "{family}"
            );
        }
    }
    let at = |p| crate::candidates::table_strategy(&text, "Cpu", "Points", "Hexahedron", 8, p);
    assert_eq!(at(1), Ok("sumfact"));
    assert_eq!(at(3), Ok("direct"));
    assert_eq!(at(9), Ok("sumfact"));
    assert_eq!(
        crate::candidates::table_strategy(&text, "Gpu", "PullBack", "Pyramid", 4, 1),
        Ok("loops")
    );
}

// ---------------------------------------------------------------------------------------------
// M10c (ADR-0065): the fallback without the admission gate, missing cells and the grid records
// of two processes.

const CPUS: [(&str, &str); 3] = [("macmini", "cpu"), ("locust", "cpu"), ("kathleen", "cpu")];

/// The Cpu baseline of `kind` and `bytes` at vector size `n`.
fn cpu(kind: KernelKind, bytes: usize, n: usize) -> VectorConfig {
    VectorConfig {
        vector_size: n,
        ..vector_baseline(DeviceClass::Cpu, kind, bytes)
    }
}

/// A grid record of process `seed`: the Mac mini reports performance cores, the others not.
fn grid_rec(
    platform: (&str, &str),
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
    config: VectorConfig,
    t: f64,
    seed: u64,
) -> SweepRecord {
    let perf = (platform.0 == "macmini").then_some(4);
    SweepRecord {
        seed: Some(seed),
        positions: Some(vec![0; 5]),
        ..rec(platform, kind, scalar, size, config, t, perf)
    }
}

fn decide_grid(records: Vec<SweepRecord>) -> VectorDecision {
    decide_vector(
        &M10Data {
            cpu_grid: records,
            ..M10Data::default()
        },
        GenericRule::M10c,
    )
}

/// A platform and its (vector size, time) pairs.
type PlatformTimes<'a> = ((&'a str, &'a str), &'a [(usize, f64)]);

/// Grid records of both processes with the same time, at `sizes`, for (vector size, time) per
/// platform.
fn grid_records(
    kind: KernelKind,
    scalar: Scalar,
    sizes: &[usize],
    per_platform: &[PlatformTimes],
) -> Vec<SweepRecord> {
    let mut out = Vec::new();
    for &(platform, times) in per_platform {
        for &size in sizes {
            for &(n, t) in times {
                for seed in [1, 2] {
                    let c = cpu(kind, scalar.bytes(), n);
                    out.push(grid_rec(platform, kind, scalar, size, c, t, seed));
                }
            }
        }
    }
    out
}

/// ADR-0065, Problem 3 (Pack 4B): a candidate within 1.2 on every platform that is only 1.06
/// times faster than the baseline (geometric mean) is not admitted; the baseline is 1.7 times
/// slower than the best on one platform. M10's rule falls back to the baseline; M10c's applies
/// the bound and the choice to all candidates and takes the candidate.
#[test]
fn m10c_waives_admission_when_no_admitted_candidate_meets_the_bound() {
    let t = |c: usize, m: usize, _s: usize| {
        Some(match (c, m) {
            (0, 0) => 1.7,
            (1, 0) => 1.0,
            (0, _) => 1.0,
            _ => 1.2,
        })
    };
    let m10 = generic_rule(GenericRule::M10, 0, &[0, 1], 3, 2, t);
    assert_eq!((m10.choice, m10.fallback, m10.waived), (0, true, false));
    assert_eq!(m10.admitted, [0], "1.06x is not admitted");
    let m10c = generic_rule(GenericRule::M10c, 0, &[0, 1], 3, 2, t);
    assert_eq!((m10c.choice, m10c.fallback, m10c.waived), (1, false, true));
    assert!((m10c.worst - 1.2).abs() < 1e-12, "{}", m10c.worst);
    assert_eq!(m10c.admitted, [0], "admission itself is unchanged");

    // The same shape through the grid records of Pack 4B on the three Cpu platforms.
    let k = KernelKind::Pack;
    let records = grid_records(
        k,
        Scalar::F32,
        &CANDIDATE_SIZES,
        &[
            (CPUS[0], &[(1, 1.7), (16, 1.0)]),
            (CPUS[1], &[(1, 1.0), (16, 1.2)]),
            (CPUS[2], &[(1, 1.0), (16, 1.2)]),
        ],
    );
    let d = decide_grid(records);
    let e = entry(&d, DeviceClass::Cpu, k, 4);
    assert_eq!(
        (e.fit.clone(), e.choice.vector_size),
        (GenericFit::Bound, 16)
    );
    assert!(e.reason.contains("apply to all candidates"), "{}", e.reason);
}

/// Where no candidate meets the bound even without the gate, the admitted candidate with the
/// smallest worst case, marked with its worst case and a cause naming the conflicting bests.
#[test]
fn m10c_fallback_stays_a_fallback_with_a_cause() {
    let k = KernelKind::ScanTile;
    let records = grid_records(
        k,
        Scalar::U32,
        &CANDIDATE_SIZES,
        &[
            (CPUS[0], &[(1, 2.0), (8, 1.0)]),
            (CPUS[1], &[(1, 1.0), (8, 2.0)]),
            (CPUS[2], &[(1, 1.0), (8, 1.0)]),
        ],
    );
    let d = decide_grid(records);
    let e = entry(&d, DeviceClass::Cpu, k, 4);
    assert_eq!(e.choice.vector_size, 1, "the baseline is the only admitted");
    let GenericFit::Fallback { worst, cause } = &e.fit else {
        panic!("{:?}", e.fit);
    };
    assert!((worst - 2.0).abs() < 1e-12, "{worst}");
    assert!(e.reason.contains("worst case"), "{}", e.reason);
    assert!(
        cause.starts_with("largest r 2.00 on macmini cpu")
            && cause.contains("conflict")
            && cause.contains("locust cpu best n = 1"),
        "{cause}"
    );
}

/// A member without a time at one size (GeomFactors f64 at 10^7 on the Mac mini, ADR-0065
/// item 4) is left out of that size: admission uses the platforms present, r and the bound the
/// cells present. M10's rule needs every cell and takes the default.
#[test]
fn m10c_leaves_out_missing_cells() {
    let k = KernelKind::GeomFactors;
    let mut records = grid_records(
        k,
        Scalar::F64,
        &CANDIDATE_SIZES[..1],
        &[(CPUS[0], &[(1, 1.0), (4, 1.2)])],
    );
    records.extend(grid_records(
        k,
        Scalar::F64,
        &CANDIDATE_SIZES,
        &[
            (CPUS[1], &[(1, 1.0), (4, 0.5)]),
            (CPUS[2], &[(1, 1.0), (4, 0.5)]),
        ],
    ));
    let d = decide_grid(records.clone());
    let e = entry(&d, DeviceClass::Cpu, k, 8);
    assert_eq!(
        (e.fit.clone(), e.choice.vector_size),
        (GenericFit::Bound, 4),
        "{}",
        e.reason
    );
    let r: Vec<(&str, String)> = e
        .platforms
        .iter()
        .map(|p| (p.platform.target, format!("{:.2}", p.r)))
        .collect();
    assert_eq!(
        r,
        [
            ("macmini", "1.20".to_owned()),
            ("locust", "1.00".to_owned()),
            ("kathleen", "1.00".to_owned())
        ]
    );
    // 10^6: geometric mean of 1.2, 0.5 and 0.5; 10^7: of 0.5 and 0.5.
    assert!(e.reason.starts_with("1.49x / 2.00x"), "{}", e.reason);
    let m10 = decide_vector(
        &M10Data {
            vector: records,
            ..M10Data::default()
        },
        GenericRule::M10,
    );
    assert_eq!(entry(&m10, DeviceClass::Cpu, k, 8).fit, GenericFit::Default);
}

/// A configuration's time is the geometric mean of the two processes' medians.
#[test]
fn m10c_combines_two_processes_by_the_geometric_mean() {
    let k = KernelKind::Copy;
    let mut records = Vec::new();
    for size in CANDIDATE_SIZES {
        for (seed, t4) in [(1, 0.5), (2, 0.72)] {
            records.push(grid_rec(
                CPUS[1],
                k,
                Scalar::U32,
                size,
                cpu(k, 4, 1),
                1.0,
                seed,
            ));
            records.push(grid_rec(
                CPUS[1],
                k,
                Scalar::U32,
                size,
                cpu(k, 4, 4),
                t4,
                seed,
            ));
        }
    }
    let d = decide_grid(records);
    let e = entry(&d, DeviceClass::Cpu, k, 4);
    assert_eq!(e.choice.vector_size, 4);
    // sqrt(0.5 · 0.72) = 0.6: 1.67 times faster (the last process alone would give 1.39).
    assert!(e.reason.starts_with("1.67x / 1.67x"), "{}", e.reason);
}

/// Grid records come from `vector_sweep --grid`, which writes the process seed.
#[test]
#[should_panic(expected = "a grid record without a seed")]
fn m10c_rejects_a_grid_record_without_a_seed() {
    let k = KernelKind::Copy;
    let mut r = grid_rec(CPUS[1], k, Scalar::U32, 1_000_000, cpu(k, 4, 1), 1.0, 1);
    r.seed = None;
    decide_grid(vec![r]);
}

/// The degree policy follows the waiver too: an admitted-only fallback becomes the strategy
/// within the bound.
#[test]
fn m10c_degree_picks_waive_admission() {
    // direct is the baseline; loops is within 1.2 everywhere but only 5% faster overall.
    let mut records = Vec::new();
    for (d, direct, loops) in [(MAC, 1.7, 1.0), (GRACE, 1.0, 1.2), (KATHLEEN, 1.0, 1.2)] {
        for n in CANDIDATE_SIZES {
            for (s, t) in [("direct", direct), ("loops", loops)] {
                records.push(drec(d, "Points", "Hexahedron", 1, Scalar::F64, n, s, t));
            }
        }
    }
    let m10 = decide_degree_of(Vec::new(), records.clone());
    let k = cpu_points(&m10).pick(1).unwrap();
    assert_eq!(k.strategy, "direct");
    assert!(matches!(k.fit, GenericFit::Fallback { .. }));
    let m10c = decide_degree(
        &HighOrderData::default(),
        &M10Data {
            degree: records,
            ..M10Data::default()
        },
        GenericRule::M10c,
    );
    let k = cpu_points(&m10c).pick(1).unwrap();
    assert_eq!((k.strategy, k.fit.clone()), ("loops", GenericFit::Bound));
    assert!(k.reason.contains("apply to all"), "{}", k.reason);
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
            for rule in [GenericRule::M10, GenericRule::M10c] {
                let a = generic_rule(rule, 0, &keys, members, 2, |c, m, s| Some(t(c, m, s)));
                let b = generic_rule(rule, 0, &keys, members, 2, |c, m, s| {
                    Some(t(c, order[m], s))
                });
                prop_assert_eq!(a.choice, b.choice);
                prop_assert_eq!(a.fallback, b.fallback);
                prop_assert_eq!(a.waived, b.waived);
                prop_assert_eq!(a.admitted, b.admitted);
                prop_assert!((a.worst - b.worst).abs() <= 1e-12 * a.worst);
            }
        }

        /// Under M10c with cells left out (a member without a size), the choice does not depend
        /// on the order of the members either.
        #[test]
        fn m10c_choice_with_missing_cells_does_not_depend_on_platform_order(
            times in proptest::collection::vec(0.2f64..1.0, 4 * 5 * 2),
            missing in proptest::collection::vec(any::<bool>(), 4 * 2),
            members in 1usize..=4,
            candidates in 1usize..=5,
            rotate in 0usize..4,
        ) {
            // Member 0 keeps both sizes, so the baseline always has a time.
            let t = |c: usize, m: usize, s: usize| {
                (m == 0 || !missing[m * 2 + s]).then(|| times[(m * 5 + c) * 2 + s])
            };
            let order: Vec<usize> = (0..members).map(|m| (m + rotate) % members).collect();
            let keys: Vec<usize> = (0..candidates).collect();
            let a = generic_rule(GenericRule::M10c, 0, &keys, members, 2, t);
            let b = generic_rule(GenericRule::M10c, 0, &keys, members, 2, |c, m, s| {
                t(c, order[m], s)
            });
            prop_assert_eq!(a.choice, b.choice);
            prop_assert_eq!(a.fallback, b.fallback);
            prop_assert_eq!(a.admitted, b.admitted);
            prop_assert!((a.worst - b.worst).abs() <= 1e-12 * a.worst);
        }

        /// The M10c vector choice does not depend on the order of the processes (which seed
        /// measured which times) or of the records.
        #[test]
        fn m10c_choice_does_not_depend_on_process_or_record_order(
            times in proptest::collection::vec(0.2f64..1.0, 3 * 3 * 2 * 2),
            reverse in any::<bool>(),
        ) {
            let k = KernelKind::Copy;
            let records = |swap: bool| -> Vec<SweepRecord> {
                let mut out = Vec::new();
                for (p, &platform) in CPUS.iter().enumerate() {
                    for (si, &size) in CANDIDATE_SIZES.iter().enumerate() {
                        for (c, n) in [1, 4, 8].into_iter().enumerate() {
                            for process in 0..2 {
                                let seed = if swap { 2 - process } else { 1 + process } as u64;
                                let t = times[((p * 3 + c) * 2 + si) * 2 + process];
                                out.push(grid_rec(platform, k, Scalar::U32, size, cpu(k, 4, n), t, seed));
                            }
                        }
                    }
                }
                out
            };
            let a = decide_grid(records(false));
            let mut swapped = records(true);
            if reverse {
                swapped.reverse();
            }
            let b = decide_grid(swapped);
            let (a, b) = (entry(&a, DeviceClass::Cpu, k, 4), entry(&b, DeviceClass::Cpu, k, 4));
            prop_assert_eq!(a.choice, b.choice);
            prop_assert_eq!(&a.fit, &b.fit);
            prop_assert_eq!(&a.reason, &b.reason);
        }

        /// The degree choice does not depend on which platform measured which times: permuting
        /// the platforms' records gives the same strategy, fit and baseline.
        #[test]
        fn degree_choice_does_not_depend_on_platform_order(
            times in proptest::collection::vec(0.2f64..1.0, 3 * 3 * 2),
            rotate in 1usize..3,
        ) {
            const S: [&str; 3] = ["direct", "loops", "sumfact"];
            let platforms = [MAC, GRACE, KATHLEEN];
            let records = |shift: usize| -> Vec<DegreeRecord> {
                let mut out = Vec::new();
                for (m, &d) in platforms.iter().enumerate() {
                    let src = (m + shift) % 3;
                    for (k, &n) in CANDIDATE_SIZES.iter().enumerate() {
                        for (c, s) in S.into_iter().enumerate() {
                            let t = times[(src * 3 + c) * 2 + k];
                            out.push(drec(d, "Points", "Hexahedron", 1, Scalar::F64, n, s, t));
                        }
                    }
                }
                out
            };
            let a = decide_degree_of(Vec::new(), records(0));
            let b = decide_degree_of(Vec::new(), records(rotate));
            let (a, b) = (cpu_points(&a).pick(1).unwrap(), cpu_points(&b).pick(1).unwrap());
            prop_assert_eq!(a.strategy, b.strategy);
            prop_assert_eq!(a.base, b.base);
            let worst = |k: &DegreePick| match &k.fit {
                GenericFit::Fallback { worst, .. } => Some(*worst),
                _ => None,
            };
            prop_assert_eq!(worst(a).is_some(), worst(b).is_some());
            if let (Some(x), Some(y)) = (worst(a), worst(b)) {
                prop_assert!((x - y).abs() <= 1e-12 * x);
            }
        }
    }
}
