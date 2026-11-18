//! The candidate protocol of "Benchmarks" (ADR-0057 item 3; ADR-0060 item 2): the
//! configurations and strategies each policy entry measures in M10 and after a CubeCL pin bump.
//!
//! - **Baseline:** vector size 1 at the old CpuSimd128 tile, block and units (a Cpu entry), or
//!   at the Gpu tile (a Gpu entry); for strategies, `direct`, or `loops` where `direct` is not
//!   selectable.
//! - **History:** the choices of the unchanged M3s rule (`decide`) and M9h rule
//!   (`decide_degree`) on the committed records, for any old device class, as frozen in
//!   `gridcl-bench/history/{vector,degree}-policy-pre-m10.toml`. A Cpu entry takes the
//!   CpuSimd128 and CpuSimd512 choices (CpuSimd256 copied CpuSimd512), a Gpu entry the Gpu ones.
//! - **Own:** the configuration the entry's current policy names (`gridcl-device/vector-policy.toml`
//!   and `gridcl-geometry/degree-policy.toml`, as compiled into this crate).
//! - Pack and PullBack, never measured before M10, also measure the widest size and the M3s
//!   choices of Gather and GeomFactors.
//!
//! The history files are guarded by the M3s and M9h history tests (`perf_record.rs`,
//! `high_order_record.rs`), which check them against the unchanged rules on the committed
//! records.

use gridcl_device::{CpuUnits, DeviceClass, KernelKind};

use crate::record::{LegacyClass, STRATEGIES, Scalar};

/// Every kind of the vector policy with the scalar types the candidate protocol measures it
/// in, in the order of `vector-policy.toml`: the M3s families (`families::families`) with Pack
/// and PullBack, which exist from M6 and M9. Scalar is measured for its CPU blocks only.
pub fn candidate_kinds() -> Vec<(KernelKind, &'static [Scalar])> {
    use Scalar::{F32, F64, U32};
    vec![
        (KernelKind::Fill, &[F32, F64, U32]),
        (KernelKind::Copy, &[U32]),
        (KernelKind::Reduce, &[F32, F64, U32]),
        (KernelKind::ScanTile, &[U32]),
        (KernelKind::SortDigits, &[U32]),
        (KernelKind::Gather, &[F32, F64, U32]),
        (KernelKind::Pack, &[F32, F64]),
        (KernelKind::TopologyKeys, &[U32]),
        (KernelKind::PermutationInfo, &[U32]),
        (KernelKind::EVector, &[F32, F64]),
        (KernelKind::GeomFactors, &[F32, F64]),
        (KernelKind::PullBack, &[F32, F64]),
        (KernelKind::Scalar, &[U32, F64]),
    ]
}

/// The frozen pre-M10 vector policy (ADR-0060 item 2).
pub const HISTORY_VECTOR_TOML: &str = include_str!("../history/vector-policy-pre-m10.toml");
/// The live vector policy, as compiled into this crate.
pub const VECTOR_POLICY_TOML: &str = include_str!("../../gridcl-device/vector-policy.toml");

/// One configuration of a vector-policy entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VectorConfig {
    /// Requested vector size.
    pub vector_size: usize,
    /// QuadField tile width (GeomFactors only).
    pub tile: Option<usize>,
    /// CPU block in bytes (Cpu only).
    pub block_bytes: Option<usize>,
    /// CPU units (Cpu only).
    pub units: Option<CpuUnits>,
}

/// A configuration as an ordered key, in the generic rule's simplicity order (ADR-0058 item 4):
/// the smallest vector size, then tile, then block; ties in units go to all hardware threads.
pub type ConfigKey = (usize, Option<usize>, Option<usize>, bool);

impl VectorConfig {
    /// The configuration's key in the simplicity order.
    pub fn key(&self) -> ConfigKey {
        (
            self.vector_size,
            self.tile,
            self.block_bytes,
            self.units == Some(CpuUnits::Performance),
        )
    }

    /// The configuration of `key` in an entry of `class`: units only on Cpu.
    pub fn from_key(class: DeviceClass, key: ConfigKey) -> VectorConfig {
        VectorConfig {
            vector_size: key.0,
            tile: key.1,
            block_bytes: key.2,
            units: (class == DeviceClass::Cpu).then_some(if key.3 {
                CpuUnits::Performance
            } else {
                CpuUnits::All
            }),
        }
    }
}

/// The configuration of (`kind`, `bytes`) in the table `table` of the vector-policy text
/// `text`, at vector size `n` if given (a baseline), else the table's own. The tile is the
/// table's `quad_tile` for GeomFactors, `None` otherwise; block and units are read where the
/// table has `cpu_block` (CPU tables).
///
/// # Errors
///
/// If the text does not parse, or the table lacks an entry.
pub fn table_config(
    text: &str,
    table: &str,
    kind: KernelKind,
    bytes: usize,
    n: Option<usize>,
) -> Result<VectorConfig, String> {
    let policy: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
    let k = format!("{kind:?}");
    let b = bytes.to_string();
    let what = format!("[{table}] {k} {bytes}");
    let class = policy
        .get(table)
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("no table [{table}]"))?;
    let int = |path: &[&str]| -> Result<usize, String> {
        let mut v: &toml::Value = class
            .get(path[0])
            .ok_or_else(|| format!("{what}: no {}", path[0]))?;
        for p in &path[1..] {
            v = v
                .get(p)
                .ok_or_else(|| format!("{what}: no {}", path.join(".")))?;
        }
        v.as_integer()
            .and_then(|i| usize::try_from(i).ok())
            .ok_or_else(|| format!("{what}: {} is not a size", path.join(".")))
    };
    let vector_size = match n {
        Some(n) => n,
        None => int(&["vector_size", &k, &b])?,
    };
    let tile = if kind == KernelKind::GeomFactors {
        Some(int(&["quad_tile", &b])?)
    } else {
        None
    };
    let cpu = class.contains_key("cpu_block");
    let block_bytes = if cpu {
        Some(int(&["cpu_block", &k, &b])? * crate::decide::BLOCK_UNIT)
    } else {
        None
    };
    let units = if cpu {
        let name = class
            .get("cpu_units")
            .and_then(|u| u.get(&k))
            .and_then(|u| u.get(&b))
            .and_then(toml::Value::as_str);
        Some(match name {
            Some("all") => CpuUnits::All,
            Some("performance") => CpuUnits::Performance,
            v => return Err(format!("{what}: cpu_units {v:?}")),
        })
    } else {
        None
    };
    Ok(VectorConfig {
        vector_size,
        tile,
        block_bytes,
        units,
    })
}

fn history_config(
    old: LegacyClass,
    kind: KernelKind,
    bytes: usize,
    n: Option<usize>,
) -> VectorConfig {
    table_config(HISTORY_VECTOR_TOML, &format!("{old:?}"), kind, bytes, n)
        .unwrap_or_else(|e| panic!("gridcl-bench/history/vector-policy-pre-m10.toml: {e}"))
}

/// The old classes whose M3s and M9h choices an entry of `class` measures: CpuSimd128 and
/// CpuSimd512 for Cpu (CpuSimd256 copied CpuSimd512), Gpu for Gpu.
pub fn old_classes(class: DeviceClass) -> &'static [LegacyClass] {
    match class {
        DeviceClass::Cpu => &[LegacyClass::CpuSimd128, LegacyClass::CpuSimd512],
        DeviceClass::Gpu => &[LegacyClass::Gpu],
    }
}

/// The baseline of the vector-policy entry (`class`, `kind`, `bytes`): vector size 1 at the old
/// CpuSimd128 tile, block and units on Cpu, at the Gpu tile on Gpu (ADR-0060 item 2).
pub fn vector_baseline(class: DeviceClass, kind: KernelKind, bytes: usize) -> VectorConfig {
    let old = match class {
        DeviceClass::Cpu => LegacyClass::CpuSimd128,
        DeviceClass::Gpu => LegacyClass::Gpu,
    };
    history_config(old, kind, bytes, Some(1))
}

/// The configuration the live policy names for (`class`, `kind`, `bytes`).
pub fn policy_vector_config(class: DeviceClass, kind: KernelKind, bytes: usize) -> VectorConfig {
    table_config(VECTOR_POLICY_TOML, &format!("{class:?}"), kind, bytes, None)
        .unwrap_or_else(|e| panic!("gridcl-device/vector-policy.toml: {e}"))
}

/// The M3s choice of the old class `old` for (`kind`, `bytes`), from the frozen policy.
pub fn history_vector_config(old: LegacyClass, kind: KernelKind, bytes: usize) -> VectorConfig {
    history_config(old, kind, bytes, None)
}

/// The candidates of the vector-policy entry (`class`, `kind`, `scalar`'s size) on a device
/// whose widest vector size for `scalar` is `widest`, baseline first, each once:
///
/// - the baseline ([`vector_baseline`]);
/// - the M3s choices of the class's old classes ([`old_classes`]);
/// - the live policy's configuration;
/// - for Pack and PullBack, the M3s choices of Gather and GeomFactors as configurations of
///   the kind (PullBack has no tile of its own), and the baseline at the widest size. The
///   widest size is `widest`, or the widest of the other candidates if that is wider, so that a
///   device's records give back the same set when their widest requested size is passed.
///
/// Scalar kernels plan at size 1 (ADR-0015, item 7): their candidates differ in block and
/// units only.
pub fn vector_candidates(
    class: DeviceClass,
    kind: KernelKind,
    scalar: Scalar,
    widest: usize,
) -> Vec<VectorConfig> {
    let bytes = scalar.bytes();
    let baseline = vector_baseline(class, kind, bytes);
    let mut out = vec![baseline];
    fn push(out: &mut Vec<VectorConfig>, c: VectorConfig) {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    for &old in old_classes(class) {
        push(&mut out, history_vector_config(old, kind, bytes));
    }
    push(&mut out, policy_vector_config(class, kind, bytes));
    let nearest = match kind {
        KernelKind::Pack => Some(KernelKind::Gather),
        KernelKind::PullBack => Some(KernelKind::GeomFactors),
        _ => None,
    };
    if let Some(nearest) = nearest {
        for &old in old_classes(class) {
            let c = history_vector_config(old, nearest, bytes);
            push(&mut out, VectorConfig { tile: None, ..c });
        }
        let wider = out.iter().map(|c| c.vector_size).fold(widest, usize::max);
        push(
            &mut out,
            VectorConfig {
                vector_size: wider,
                ..baseline
            },
        );
    }
    out
}

/// M10's vector policy, frozen as history when M10c re-derived the live one (ADR-0065): the
/// "own policy" of M10's candidate sets.
pub const M10_VECTOR_TOML: &str = include_str!("../history/vector-policy-m10.toml");

/// The vector sizes of the M10c grid of `[Cpu]` vector entries (ADR-0065 item 2).
pub const GRID_VECTOR_SIZES: [usize; 4] = [1, 4, 8, 16];
/// The CPU blocks of the M10c grid, in bytes.
pub const GRID_BLOCKS: [usize; 3] = [4 << 10, 16 << 10, 64 << 10];

/// The cases the M10c grid leaves out, as (target, kind, scalar, size): those that do not fit a
/// platform's memory (ADR-0065 item 4; further cases from the M10c smoke run are named in the
/// plan's Deviations log).
pub const GRID_OMITTED: &[(&str, KernelKind, Scalar, usize)] =
    &[("macmini", KernelKind::GeomFactors, Scalar::F64, 10_000_000)];

/// The M10c grid of the `[Cpu]` vector entry of `kind` in `scalar` (ADR-0065 item 2), in the
/// simplicity order: vector sizes 1, 4, 8 and 16, blocks of 4, 16 and 64 KiB, and both unit
/// settings (a device without performance cores runs all threads, which count for both;
/// ADR-0017). GeomFactors uses tile 4 up to size 4 and tile 16 above; Scalar has blocks and units
/// only. 24 configurations (Scalar 6).
pub fn cpu_grid(kind: KernelKind, scalar: Scalar) -> Vec<VectorConfig> {
    let _ = (kind, scalar);
    todo!("M10c grid (T3)")
}

/// The frozen pre-M10 degree policy (ADR-0060 item 2).
pub const HISTORY_DEGREE_TOML: &str = include_str!("../history/degree-policy-pre-m10.toml");
/// The live degree policy, as compiled into this crate.
pub const DEGREE_POLICY_TOML: &str = include_str!("../../gridcl-geometry/degree-policy.toml");
/// M10's degree policy, frozen as history when M10c re-derived the live one (ADR-0065).
pub const M10_DEGREE_TOML: &str = include_str!("../history/degree-policy-m10.toml");

/// The strategy of (`family`, `cell`, `bytes`) at `degree` in the table `table` of the
/// degree-policy text `text`: the last range whose first degree is at most `degree`.
///
/// # Errors
///
/// If the text does not parse, the table lacks the entry, or a range is malformed or names an
/// unknown strategy.
pub fn table_strategy(
    text: &str,
    table: &str,
    family: &str,
    cell: &str,
    bytes: usize,
    degree: u32,
) -> Result<&'static str, String> {
    let policy: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
    let what = format!("[{table}.{family}] {cell} {bytes}");
    let ranges = policy
        .get(table)
        .and_then(|t| t.get(family))
        .and_then(|f| f.get(cell))
        .and_then(|c| c.get(bytes.to_string()))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| format!("{what}: no entry"))?;
    let mut strategy = None;
    for range in ranges {
        let pair = range
            .as_array()
            .filter(|p| p.len() == 2)
            .ok_or_else(|| format!("{what}: range {range}"))?;
        let first = pair[0]
            .as_integer()
            .ok_or_else(|| format!("{what}: first degree {}", pair[0]))?;
        let name = pair[1]
            .as_str()
            .ok_or_else(|| format!("{what}: strategy {}", pair[1]))?;
        let s = STRATEGIES
            .iter()
            .copied()
            .find(|&s| s == name)
            .ok_or_else(|| format!("{what}: unknown strategy {name}"))?;
        if first <= i64::from(degree) {
            strategy = Some(s);
        }
    }
    strategy.ok_or_else(|| format!("{what}: no range holds p = {degree}"))
}

/// The baseline strategy of a degree-policy entry at `degree` (ADR-0057 item 3): `direct`, or
/// `loops` where the sweep leaves `direct` out (tetrahedra, hexahedra and prisms at p ≥ 5,
/// ADR-0050). Where `direct` is measured but fails or takes over 30 s to compile and load,
/// `degree_sweep --candidates` adds `loops` (ADR-0056).
pub fn degree_baseline(cell: &str, degree: u32) -> &'static str {
    if degree >= 5 && matches!(cell, "Tetrahedron" | "Hexahedron" | "Prism") {
        "loops"
    } else {
        "direct"
    }
}

/// The M9h choice of the old class `old` for (`family`, `cell`, `bytes`) at `degree`, from the
/// frozen policy.
pub fn history_strategy(
    old: LegacyClass,
    family: &str,
    cell: &str,
    bytes: usize,
    degree: u32,
) -> &'static str {
    table_strategy(
        HISTORY_DEGREE_TOML,
        &format!("{old:?}"),
        family,
        cell,
        bytes,
        degree,
    )
    .unwrap_or_else(|e| panic!("gridcl-bench/history/degree-policy-pre-m10.toml: {e}"))
}

/// The strategy the live degree policy names for (`class`, `family`, `cell`, `bytes`) at
/// `degree`.
pub fn policy_strategy(
    class: DeviceClass,
    family: &str,
    cell: &str,
    bytes: usize,
    degree: u32,
) -> &'static str {
    table_strategy(
        DEGREE_POLICY_TOML,
        &format!("{class:?}"),
        family,
        cell,
        bytes,
        degree,
    )
    .unwrap_or_else(|e| panic!("gridcl-geometry/degree-policy.toml: {e}"))
}

/// The candidate strategies of the degree-policy entry (`class`, policy `family`, `cell`,
/// `bytes`) at `degree`, baseline first, each once: the baseline ([`degree_baseline`]), the M9h
/// choices of the class's old classes ([`old_classes`]) and the live policy's choice.
pub fn degree_candidates(
    class: DeviceClass,
    family: &str,
    cell: &str,
    bytes: usize,
    degree: u32,
) -> Vec<&'static str> {
    let mut out = vec![degree_baseline(cell, degree)];
    let history = old_classes(class)
        .iter()
        .map(|&old| history_strategy(old, family, cell, bytes, degree));
    for s in history.chain([policy_strategy(class, family, cell, bytes, degree)]) {
        if !out.contains(&s) {
            out.push(s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::decide::decide;
    use crate::record::{KINDS, load_dir};

    /// The class of the M3s decision rule whose choices a legacy class's table holds.
    fn decided(
        d: &crate::decide::Decision,
        old: LegacyClass,
        kind: KernelKind,
        bytes: usize,
    ) -> VectorConfig {
        let c = d.choice(old, kind, bytes).expect("a choice per entry");
        VectorConfig {
            vector_size: c.vector_size,
            tile: (kind == KernelKind::GeomFactors).then(|| d.quad_tile(old, bytes)),
            block_bytes: c.cpu_block.map(|b| b * crate::decide::BLOCK_UNIT),
            units: c.cpu_units,
        }
    }

    /// The history configurations are the unchanged M3s rule's choices on the committed
    /// records (ADR-0060 item 2).
    #[test]
    fn history_candidates_are_the_m3s_choices_of_the_committed_records() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/perf/data");
        let d = decide(&load_dir(&dir).unwrap());
        for old in LegacyClass::ALL {
            for kind in KINDS {
                for bytes in [4, 8] {
                    assert_eq!(
                        history_vector_config(old, kind, bytes),
                        decided(&d, old, kind, bytes),
                        "{old:?} {kind:?} {bytes}"
                    );
                }
            }
        }
    }

    /// The candidate kinds are every policy kind, and the M3s families with their types.
    #[test]
    fn candidate_kinds_extend_the_m3s_families() {
        let kinds: Vec<KernelKind> = candidate_kinds().into_iter().map(|(k, _)| k).collect();
        assert_eq!(kinds, KINDS);
        for (kind, scalars) in crate::families::families() {
            let (_, ours) = candidate_kinds()
                .into_iter()
                .find(|&(k, _)| k == kind)
                .unwrap();
            assert_eq!(ours, scalars, "{kind:?}");
        }
    }

    #[test]
    fn candidates_start_with_the_baseline_and_are_distinct() {
        for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
            for kind in KINDS {
                for scalar in [Scalar::F32, Scalar::F64] {
                    let c = vector_candidates(class, kind, scalar, 16);
                    assert_eq!(c[0], vector_baseline(class, kind, scalar.bytes()));
                    assert_eq!(c[0].vector_size, 1);
                    for (i, a) in c.iter().enumerate() {
                        assert!(!c[i + 1..].contains(a), "{class:?} {kind:?}: {a:?} twice");
                        assert_eq!(a.units.is_some(), class == DeviceClass::Cpu);
                        assert_eq!(a.block_bytes.is_some(), class == DeviceClass::Cpu);
                        assert_eq!(a.tile.is_some(), kind == KernelKind::GeomFactors);
                    }
                    if kind == KernelKind::Scalar {
                        assert!(c.iter().all(|a| a.vector_size == 1), "{class:?}: {c:?}");
                    }
                }
            }
        }
    }

    /// The baselines of ADR-0060 item 2: size 1 at the CpuSimd128 tile, block and units, or at
    /// the Gpu tile.
    #[test]
    fn baselines_follow_the_old_tables() {
        let cpu = vector_baseline(DeviceClass::Cpu, KernelKind::GeomFactors, 8);
        assert_eq!(
            cpu,
            VectorConfig {
                vector_size: 1,
                tile: Some(4),
                block_bytes: Some(64 << 10),
                units: Some(CpuUnits::All),
            }
        );
        let fill = vector_baseline(DeviceClass::Cpu, KernelKind::Fill, 4);
        assert_eq!(
            (fill.vector_size, fill.block_bytes, fill.units),
            (1, Some(64 << 10), Some(CpuUnits::Performance))
        );
        let gpu = vector_baseline(DeviceClass::Gpu, KernelKind::GeomFactors, 4);
        assert_eq!(
            gpu,
            VectorConfig {
                vector_size: 1,
                tile: Some(16),
                block_bytes: None,
                units: None,
            }
        );
    }

    /// The history strategies are the unchanged M9h rule's choices on the committed records
    /// (ADR-0060 item 2), at every degree of every entry of every old class.
    #[test]
    fn history_strategies_are_the_m9h_choices_of_the_committed_records() {
        use crate::decide_degree::{POLICY_FAMILIES, SCALARS, cells_of, max_degree};
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/perf/data");
        let data = crate::record::load_high_order_dir(&dir).unwrap();
        let d = crate::decide_degree::decide_degree(&data);
        for old in LegacyClass::ALL {
            for family in POLICY_FAMILIES {
                for &cell in cells_of(family) {
                    for (scalar, bytes) in SCALARS {
                        let e = d.entry(old, family, cell, scalar).expect("an entry");
                        for p in 1..=max_degree(cell) + 2 {
                            assert_eq!(
                                history_strategy(old, family, cell, bytes, p),
                                e.strategy(p),
                                "{old:?} {family} {cell} {bytes} p={p}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The expected sets: the baseline first, then the M9h choices of the old classes and the
    /// live policy's, each once.
    #[test]
    fn degree_candidates_are_the_baseline_history_and_policy() {
        use crate::decide_degree::{POLICY_FAMILIES, SCALARS, cells_of, max_degree};
        for class in [DeviceClass::Cpu, DeviceClass::Gpu] {
            for family in POLICY_FAMILIES {
                for &cell in cells_of(family) {
                    for (_, bytes) in SCALARS {
                        for p in 1..=max_degree(cell) {
                            let c = degree_candidates(class, family, cell, bytes, p);
                            let at = format!("{class:?} {family} {cell} {bytes} p={p}");
                            assert_eq!(c[0], degree_baseline(cell, p), "{at}");
                            let mut want: BTreeSet<&str> = old_classes(class)
                                .iter()
                                .map(|&o| history_strategy(o, family, cell, bytes, p))
                                .collect();
                            want.insert(policy_strategy(class, family, cell, bytes, p));
                            want.insert(c[0]);
                            let got: BTreeSet<&str> = c.iter().copied().collect();
                            assert_eq!(got.len(), c.len(), "{at}: {c:?} repeats");
                            assert_eq!(got, want, "{at}");
                            if class == DeviceClass::Cpu {
                                assert!(
                                    c.iter().all(|s| ["direct", "loops", "sumfact"].contains(s)),
                                    "{at}: {c:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // Spot checks against the frozen tables: CpuSimd128 and CpuSimd512 both choose sumfact
        // for Points on hexahedra; Gpu's baseline at p = 8 on hexahedra is loops.
        assert_eq!(
            degree_candidates(DeviceClass::Cpu, "Points", "Hexahedron", 8, 1),
            ["direct", "sumfact"]
        );
        assert_eq!(
            degree_candidates(DeviceClass::Gpu, "EVector", "Hexahedron", 4, 8)[0],
            "loops"
        );
        assert_eq!(degree_baseline("Triangle", 8), "direct");
        assert_eq!(degree_baseline("Prism", 5), "loops");
        assert_eq!(degree_baseline("Prism", 4), "direct");
    }

    #[test]
    fn table_strategy_reads_ranges_and_rejects_bad_tables() {
        let text = "[Cpu.Points]\nHexahedron = { 4 = [[1, \"direct\"], [3, \"sumfact\"]] }\n";
        let at = |p| table_strategy(text, "Cpu", "Points", "Hexahedron", 4, p);
        assert_eq!(at(1), Ok("direct"));
        assert_eq!(at(2), Ok("direct"));
        assert_eq!(at(3), Ok("sumfact"));
        assert_eq!(at(10), Ok("sumfact"));
        assert!(table_strategy(text, "Gpu", "Points", "Hexahedron", 4, 1).is_err());
        assert!(table_strategy(text, "Cpu", "Points", "Hexahedron", 8, 1).is_err());
        let bad = "[Cpu.Points]\nHexahedron = { 4 = [[1, \"fast\"]] }\n";
        assert!(table_strategy(bad, "Cpu", "Points", "Hexahedron", 4, 1).is_err());
        let late = "[Cpu.Points]\nHexahedron = { 4 = [[2, \"loops\"]] }\n";
        assert!(table_strategy(late, "Cpu", "Points", "Hexahedron", 4, 1).is_err());
    }

    /// Pack and PullBack: the baseline, the Gather or GeomFactors choices of the old classes
    /// without a tile, and the baseline at the widest size, widened to the widest candidate.
    #[test]
    fn pack_and_pull_back_measure_the_nearest_kinds_and_the_widest_size() {
        let c = vector_candidates(DeviceClass::Cpu, KernelKind::PullBack, Scalar::F32, 4);
        let base = vector_baseline(DeviceClass::Cpu, KernelKind::PullBack, 4);
        // CpuSimd512's GeomFactors size is 16, wider than the device's 4.
        let gf = history_vector_config(LegacyClass::CpuSimd512, KernelKind::GeomFactors, 4);
        assert_eq!(gf.vector_size, 16);
        assert!(c.contains(&VectorConfig { tile: None, ..gf }), "{c:?}");
        assert!(
            c.contains(&VectorConfig {
                vector_size: 16,
                ..base
            }),
            "{c:?}"
        );
        // The set is closed: passing its widest size gives it back.
        let widest = c.iter().map(|a| a.vector_size).max().unwrap();
        assert_eq!(
            vector_candidates(DeviceClass::Cpu, KernelKind::PullBack, Scalar::F32, widest),
            c
        );
        let p = vector_candidates(DeviceClass::Gpu, KernelKind::Pack, Scalar::F64, 4);
        assert_eq!(
            p.iter().map(|a| a.vector_size).collect::<Vec<_>>(),
            [1, 4],
            "Gpu Pack f64: baseline and widest ({p:?})"
        );
    }
}
