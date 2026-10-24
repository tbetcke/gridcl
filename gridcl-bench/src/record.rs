//! The JSON records of `docs/perf/data/`: one [`SweepRecord`] per measured configuration of the
//! vector-size sweep, and one [`ThroughputRecord`] per end-to-end throughput measurement.
//!
//! Files named `m3s-sweep-*.json` and `m3s-throughput-*.json` each hold a JSON array of records.

use std::path::Path;

use gridcl_device::{DeviceClass, KernelKind};
use serde_json::{Map, Value, json};

/// Scalar types of the benchmarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Scalar {
    F32,
    F64,
    U32,
}

impl Scalar {
    /// The name in records and reports: `f32`, `f64` or `u32`.
    pub fn name(self) -> &'static str {
        match self {
            Scalar::F32 => "f32",
            Scalar::F64 => "f64",
            Scalar::U32 => "u32",
        }
    }

    /// Bytes per element.
    pub fn bytes(self) -> usize {
        match self {
            Scalar::F32 | Scalar::U32 => 4,
            Scalar::F64 => 8,
        }
    }
}

/// How the CPU `cube_dim` was chosen: the plan's (one unit per hardware thread) or the
/// performance-core count (M3 Max only; reported, not part of a configuration).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CubeDimMode {
    Plan,
    PerformanceCores,
}

/// Core counts of the machine: performance and efficiency cores where the OS reports them
/// (`sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`), and hardware threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cores {
    pub performance: Option<u32>,
    pub efficiency: Option<u32>,
    pub threads: u32,
}

/// The runner's node-load snapshots before and after a run on Locust (ADR-0002, ADR-0005):
/// `idle`, `busy: <reasons>` or `not recorded`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeLoad {
    pub before: String,
    pub after: String,
}

/// One measured configuration of the sweep.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepRecord {
    /// CubeCL runtime: `cpu`, `metal`, `cuda`, ...
    pub backend: String,
    /// Device name as the client reports it.
    pub device: String,
    pub device_class: DeviceClass,
    /// Machine: `m3max`, or a target of scripts/hpc/remote-test.sh (`locust`, `kathleen`,
    /// `myriad-cpu`, `myriad-v100`, `myriad-a100`).
    pub target: String,
    pub family: KernelKind,
    pub scalar: Scalar,
    /// Problem size of the protocol: 10^5, 10^6 or 10^7.
    pub nominal_n: usize,
    /// Items actually processed (cells of the nearest box for the mesh families).
    pub n: usize,
    /// Requested vector size (`with_kind_vector_size(Fixed(n))`).
    pub vector_size: usize,
    /// Vector size of the family's plans, from `record_plans`.
    pub vector_size_ran: usize,
    /// QuadField tile width (GeomFactors only).
    pub tile: Option<usize>,
    /// CPU block target in bytes (CPU classes only).
    pub block_bytes: Option<usize>,
    pub cube_dim: CubeDimMode,
    /// Units per cube of the family's plans.
    pub units: u32,
    /// Every timed run, in seconds.
    pub times_s: Vec<f64>,
    pub median_s: f64,
    /// Warm-up time minus the median.
    pub compile_s: f64,
    /// State of CubeCL's persistent kernel cache during the warm-up.
    pub kernel_cache: String,
    /// Effective bandwidth relative to `copy` at the same size on the same device, for the
    /// families with a byte count.
    pub bandwidth_ratio: Option<f64>,
    pub commit: String,
    /// `release` or `debug`.
    pub profile: String,
    pub cores: Cores,
    /// Why a Myriad GPU ran in place of Locust (ADR-0014), or a Myriad D node in place of
    /// Kathleen (ADR-0016); empty elsewhere.
    pub fallback_reason: Option<String>,
    /// Locust's node-load snapshots.
    pub node_load: Option<NodeLoad>,
}

/// What a throughput record measures (docs/plans/M3s.md, T13).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Measure {
    /// Entities created by `create_entities(d)`, 0 < d < tdim, per second.
    EntitiesPerSecond,
    /// (cell, quadrature point) pairs with X, J, det J and K computed, per second.
    FactorsPerSecond,
}

/// One end-to-end throughput measurement.
#[derive(Clone, Debug, PartialEq)]
pub struct ThroughputRecord {
    pub backend: String,
    pub device: String,
    pub device_class: DeviceClass,
    pub target: String,
    pub measure: Measure,
    /// The mesh, for example `P1 tetrahedron box`.
    pub mesh: String,
    pub cells: usize,
    pub scalar: Scalar,
    /// `policy` or `scalar`.
    pub vector_size_override: String,
    /// Entities created, or (cell, point) pairs.
    pub items: usize,
    pub times_s: Vec<f64>,
    pub median_s: f64,
    /// `items / median_s`.
    pub per_s: f64,
    pub commit: String,
    pub profile: String,
    pub cores: Cores,
    /// Why a Myriad D node ran in place of Kathleen (ADR-0016); empty elsewhere.
    pub fallback_reason: Option<String>,
    pub node_load: Option<NodeLoad>,
}

/// Every M3s record of a directory.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PerfData {
    pub sweeps: Vec<SweepRecord>,
    pub throughput: Vec<ThroughputRecord>,
}

/// The records of every `m3s-sweep-*.json` and `m3s-throughput-*.json` file in `dir`.
///
/// # Errors
///
/// A message naming the file, for unreadable files and malformed records.
pub fn load_dir(dir: &Path) -> Result<PerfData, String> {
    let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut names: Vec<String> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.ends_with(".json"))
        .collect();
    names.sort();
    let mut data = PerfData::default();
    for name in names {
        let path = dir.join(&name);
        if name.starts_with("m3s-sweep-") {
            data.sweeps
                .extend(read_records(&path, SweepRecord::from_json)?);
        } else if name.starts_with("m3s-throughput-") {
            data.throughput
                .extend(read_records(&path, ThroughputRecord::from_json)?);
        }
    }
    Ok(data)
}

/// The records of one JSON file, an array of objects.
///
/// # Errors
///
/// A message naming the file and the index of the first malformed record.
pub fn read_records<T>(
    path: &Path,
    parse: impl Fn(&Value) -> Result<T, String>,
) -> Result<Vec<T>, String> {
    let what = path.display();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{what}: {e}"))?;
    let value: Value = serde_json::from_str(&text).map_err(|e| format!("{what}: {e}"))?;
    let Value::Array(items) = value else {
        return Err(format!("{what}: not a JSON array"));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, item)| parse(item).map_err(|e| format!("{what}: record {i}: {e}")))
        .collect()
}

/// Writes `records` as a pretty-printed JSON array, one file per run.
///
/// # Errors
///
/// A message naming the file if it cannot be written.
pub fn write_records(path: &Path, records: &[Value]) -> Result<(), String> {
    let text = serde_json::to_string_pretty(records).expect("JSON of records");
    std::fs::write(path, format!("{text}\n")).map_err(|e| format!("{}: {e}", path.display()))
}

/// Every kind, in the order of `vector-policy.toml`.
pub const KINDS: [KernelKind; 13] = [
    KernelKind::Fill,
    KernelKind::Copy,
    KernelKind::Reduce,
    KernelKind::ScanTile,
    KernelKind::SortDigits,
    KernelKind::Gather,
    KernelKind::Pack,
    KernelKind::TopologyKeys,
    KernelKind::PermutationInfo,
    KernelKind::EVector,
    KernelKind::GeomFactors,
    KernelKind::PullBack,
    KernelKind::Scalar,
];

/// Every device class, in the order of `vector-policy.toml`.
pub const CLASSES: [DeviceClass; 4] = [
    DeviceClass::CpuSimd128,
    DeviceClass::CpuSimd256,
    DeviceClass::CpuSimd512,
    DeviceClass::Gpu,
];

/// The kind named `name` (its policy key, `Fill`, `ScanTile`, ...), ignoring case.
pub fn kind_named(name: &str) -> Option<KernelKind> {
    KINDS
        .into_iter()
        .find(|k| format!("{k:?}").eq_ignore_ascii_case(name))
}

/// The class named `name` (its policy key), ignoring case.
pub fn class_named(name: &str) -> Option<DeviceClass> {
    CLASSES
        .into_iter()
        .find(|c| format!("{c:?}").eq_ignore_ascii_case(name))
}

impl Scalar {
    /// The scalar type named `name` (`f32`, `f64` or `u32`).
    pub fn named(name: &str) -> Option<Scalar> {
        [Scalar::F32, Scalar::F64, Scalar::U32]
            .into_iter()
            .find(|s| s.name() == name)
    }
}

impl CubeDimMode {
    fn name(self) -> &'static str {
        match self {
            CubeDimMode::Plan => "plan",
            CubeDimMode::PerformanceCores => "performance_cores",
        }
    }

    fn named(name: &str) -> Option<CubeDimMode> {
        [CubeDimMode::Plan, CubeDimMode::PerformanceCores]
            .into_iter()
            .find(|m| m.name() == name)
    }
}

impl Measure {
    fn name(self) -> &'static str {
        match self {
            Measure::EntitiesPerSecond => "entities_per_s",
            Measure::FactorsPerSecond => "factors_per_s",
        }
    }

    fn named(name: &str) -> Option<Measure> {
        [Measure::EntitiesPerSecond, Measure::FactorsPerSecond]
            .into_iter()
            .find(|m| m.name() == name)
    }
}

impl Cores {
    fn to_json(&self) -> Value {
        json!({
            "performance": self.performance,
            "efficiency": self.efficiency,
            "threads": self.threads,
        })
    }

    fn from_json(v: &Value) -> Result<Cores, String> {
        let o = Fields::new(v, "cores")?;
        Ok(Cores {
            performance: o.opt_u32("performance")?,
            efficiency: o.opt_u32("efficiency")?,
            threads: o.u32("threads")?,
        })
    }
}

impl NodeLoad {
    /// The snapshots in a runner results directory, `node-load-before.txt` and
    /// `node-load-after.txt` (scripts/hpc/jobs/common.sh): the value of each file's last
    /// `node_load=` line, or `not recorded`.
    pub fn from_results_dir(dir: &Path) -> NodeLoad {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .ok()
                .and_then(|text| {
                    text.lines()
                        .rev()
                        .filter_map(|line| line.strip_prefix("node_load="))
                        .next()
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "not recorded".to_owned())
        };
        NodeLoad {
            before: read("node-load-before.txt"),
            after: read("node-load-after.txt"),
        }
    }

    fn to_json(&self) -> Value {
        json!({ "before": self.before, "after": self.after })
    }

    fn from_json(v: &Value) -> Result<NodeLoad, String> {
        let o = Fields::new(v, "node_load")?;
        Ok(NodeLoad {
            before: o.string("before")?,
            after: o.string("after")?,
        })
    }
}

impl SweepRecord {
    pub fn to_json(&self) -> Value {
        json!({
            "backend": self.backend,
            "device": self.device,
            "device_class": format!("{:?}", self.device_class),
            "target": self.target,
            "family": format!("{:?}", self.family),
            "scalar": self.scalar.name(),
            "nominal_n": self.nominal_n,
            "n": self.n,
            "vector_size": self.vector_size,
            "vector_size_ran": self.vector_size_ran,
            "tile": self.tile,
            "block_bytes": self.block_bytes,
            "cube_dim": self.cube_dim.name(),
            "units": self.units,
            "times_s": self.times_s,
            "median_s": self.median_s,
            "compile_s": self.compile_s,
            "kernel_cache": self.kernel_cache,
            "bandwidth_ratio": self.bandwidth_ratio,
            "commit": self.commit,
            "profile": self.profile,
            "cores": self.cores.to_json(),
            "fallback_reason": self.fallback_reason,
            "node_load": self.node_load.as_ref().map(NodeLoad::to_json),
        })
    }

    pub fn from_json(v: &Value) -> Result<SweepRecord, String> {
        let o = Fields::new(v, "sweep record")?;
        Ok(SweepRecord {
            backend: o.string("backend")?,
            device: o.string("device")?,
            device_class: o.class("device_class")?,
            target: o.string("target")?,
            family: o.kind("family")?,
            scalar: o.scalar("scalar")?,
            nominal_n: o.usize("nominal_n")?,
            n: o.usize("n")?,
            vector_size: o.usize("vector_size")?,
            vector_size_ran: o.usize("vector_size_ran")?,
            tile: o.opt_usize("tile")?,
            block_bytes: o.opt_usize("block_bytes")?,
            cube_dim: o.parsed("cube_dim", CubeDimMode::named)?,
            units: o.u32("units")?,
            times_s: o.times("times_s")?,
            median_s: o.f64("median_s")?,
            compile_s: o.f64("compile_s")?,
            kernel_cache: o.string("kernel_cache")?,
            bandwidth_ratio: o.opt_f64("bandwidth_ratio")?,
            commit: o.string("commit")?,
            profile: o.string("profile")?,
            cores: Cores::from_json(o.get("cores")?)?,
            fallback_reason: o.opt_string("fallback_reason")?,
            node_load: o.opt("node_load").map(NodeLoad::from_json).transpose()?,
        })
    }
}

impl ThroughputRecord {
    pub fn to_json(&self) -> Value {
        json!({
            "backend": self.backend,
            "device": self.device,
            "device_class": format!("{:?}", self.device_class),
            "target": self.target,
            "measure": self.measure.name(),
            "mesh": self.mesh,
            "cells": self.cells,
            "scalar": self.scalar.name(),
            "vector_size_override": self.vector_size_override,
            "items": self.items,
            "times_s": self.times_s,
            "median_s": self.median_s,
            "per_s": self.per_s,
            "commit": self.commit,
            "profile": self.profile,
            "cores": self.cores.to_json(),
            "fallback_reason": self.fallback_reason,
            "node_load": self.node_load.as_ref().map(NodeLoad::to_json),
        })
    }

    pub fn from_json(v: &Value) -> Result<ThroughputRecord, String> {
        let o = Fields::new(v, "throughput record")?;
        Ok(ThroughputRecord {
            backend: o.string("backend")?,
            device: o.string("device")?,
            device_class: o.class("device_class")?,
            target: o.string("target")?,
            measure: o.parsed("measure", Measure::named)?,
            mesh: o.string("mesh")?,
            cells: o.usize("cells")?,
            scalar: o.scalar("scalar")?,
            vector_size_override: o.string("vector_size_override")?,
            items: o.usize("items")?,
            times_s: o.times("times_s")?,
            median_s: o.f64("median_s")?,
            per_s: o.f64("per_s")?,
            commit: o.string("commit")?,
            profile: o.string("profile")?,
            cores: Cores::from_json(o.get("cores")?)?,
            fallback_reason: o.opt_string("fallback_reason")?,
            node_load: o.opt("node_load").map(NodeLoad::from_json).transpose()?,
        })
    }
}

/// Typed access to the fields of a JSON object, with messages naming the field.
struct Fields<'a>(&'a Map<String, Value>);

impl<'a> Fields<'a> {
    fn new(v: &'a Value, what: &str) -> Result<Fields<'a>, String> {
        v.as_object()
            .map(Fields)
            .ok_or_else(|| format!("{what} is not a JSON object"))
    }

    fn get(&self, key: &str) -> Result<&'a Value, String> {
        self.0.get(key).ok_or_else(|| format!("no field {key}"))
    }

    /// The field, or `None` if it is missing or null.
    fn opt(&self, key: &str) -> Option<&'a Value> {
        self.0.get(key).filter(|v| !v.is_null())
    }

    fn typed<T>(
        &self,
        key: &str,
        kind: &str,
        f: impl Fn(&Value) -> Option<T>,
    ) -> Result<T, String> {
        f(self.get(key)?).ok_or_else(|| format!("field {key} is not {kind}"))
    }

    fn opt_typed<T>(
        &self,
        key: &str,
        kind: &str,
        f: impl Fn(&Value) -> Option<T>,
    ) -> Result<Option<T>, String> {
        self.opt(key)
            .map(|v| f(v).ok_or_else(|| format!("field {key} is not {kind}")))
            .transpose()
    }

    fn string(&self, key: &str) -> Result<String, String> {
        self.typed(key, "a string", |v| v.as_str().map(str::to_owned))
    }

    fn opt_string(&self, key: &str) -> Result<Option<String>, String> {
        self.opt_typed(key, "a string", |v| v.as_str().map(str::to_owned))
    }

    fn usize(&self, key: &str) -> Result<usize, String> {
        self.typed(key, "an unsigned integer", as_usize)
    }

    fn opt_usize(&self, key: &str) -> Result<Option<usize>, String> {
        self.opt_typed(key, "an unsigned integer", as_usize)
    }

    fn u32(&self, key: &str) -> Result<u32, String> {
        self.typed(key, "a u32", as_u32)
    }

    fn opt_u32(&self, key: &str) -> Result<Option<u32>, String> {
        self.opt_typed(key, "a u32", as_u32)
    }

    fn f64(&self, key: &str) -> Result<f64, String> {
        self.typed(key, "a number", Value::as_f64)
    }

    fn opt_f64(&self, key: &str) -> Result<Option<f64>, String> {
        self.opt_typed(key, "a number", Value::as_f64)
    }

    fn times(&self, key: &str) -> Result<Vec<f64>, String> {
        self.typed(key, "an array of numbers", |v| {
            v.as_array()?.iter().map(Value::as_f64).collect()
        })
    }

    fn parsed<T>(&self, key: &str, parse: impl Fn(&str) -> Option<T>) -> Result<T, String> {
        let name = self.string(key)?;
        parse(&name).ok_or_else(|| format!("field {key}: unknown value '{name}'"))
    }

    fn class(&self, key: &str) -> Result<DeviceClass, String> {
        self.parsed(key, class_named)
    }

    fn kind(&self, key: &str) -> Result<KernelKind, String> {
        self.parsed(key, kind_named)
    }

    fn scalar(&self, key: &str) -> Result<Scalar, String> {
        self.parsed(key, Scalar::named)
    }
}

fn as_usize(v: &Value) -> Option<usize> {
    usize::try_from(v.as_u64()?).ok()
}

fn as_u32(v: &Value) -> Option<u32> {
    u32::try_from(v.as_u64()?).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cores() -> Cores {
        Cores {
            performance: Some(12),
            efficiency: None,
            threads: 16,
        }
    }

    fn sweep() -> SweepRecord {
        SweepRecord {
            backend: "cpu".to_owned(),
            device: "Apple M3 Max".to_owned(),
            device_class: DeviceClass::CpuSimd128,
            target: "m3max".to_owned(),
            family: KernelKind::ScanTile,
            scalar: Scalar::U32,
            nominal_n: 1_000_000,
            n: 1_000_000,
            vector_size: 4,
            vector_size_ran: 4,
            tile: None,
            block_bytes: Some(16 << 10),
            cube_dim: CubeDimMode::PerformanceCores,
            units: 12,
            times_s: vec![0.25, 0.5, 0.125],
            median_s: 0.25,
            compile_s: 1.5,
            kernel_cache: "off".to_owned(),
            bandwidth_ratio: Some(0.75),
            commit: "abc".to_owned(),
            profile: "release".to_owned(),
            cores: cores(),
            fallback_reason: None,
            node_load: Some(NodeLoad {
                before: "idle".to_owned(),
                after: "busy: gpu".to_owned(),
            }),
        }
    }

    fn throughput() -> ThroughputRecord {
        ThroughputRecord {
            backend: "cpu".to_owned(),
            device: "Xeon".to_owned(),
            device_class: DeviceClass::CpuSimd512,
            target: "myriad-cpu".to_owned(),
            measure: Measure::FactorsPerSecond,
            mesh: "P2 tetrahedron box".to_owned(),
            cells: 1000,
            scalar: Scalar::F64,
            vector_size_override: "scalar".to_owned(),
            items: 24_000,
            times_s: vec![2.0],
            median_s: 2.0,
            per_s: 12_000.0,
            commit: "abc".to_owned(),
            profile: "release".to_owned(),
            cores: cores(),
            fallback_reason: Some("Kathleen drained".to_owned()),
            node_load: None,
        }
    }

    #[test]
    fn records_round_trip_through_json() {
        let r = sweep();
        assert_eq!(SweepRecord::from_json(&r.to_json()).unwrap(), r);
        let r = SweepRecord {
            tile: Some(16),
            block_bytes: None,
            bandwidth_ratio: None,
            node_load: None,
            fallback_reason: Some("Locust busy".to_owned()),
            ..sweep()
        };
        assert_eq!(SweepRecord::from_json(&r.to_json()).unwrap(), r);
        let t = throughput();
        assert_eq!(ThroughputRecord::from_json(&t.to_json()).unwrap(), t);
    }

    #[test]
    fn json_names_are_the_policy_keys() {
        let v = sweep().to_json();
        assert_eq!(v["family"], "ScanTile");
        assert_eq!(v["device_class"], "CpuSimd128");
        assert_eq!(v["scalar"], "u32");
        assert_eq!(v["cube_dim"], "performance_cores");
        assert_eq!(throughput().to_json()["measure"], "factors_per_s");
        for kind in KINDS {
            assert_eq!(kind_named(&format!("{kind:?}")), Some(kind));
        }
        for class in CLASSES {
            assert_eq!(
                class_named(&format!("{class:?}").to_lowercase()),
                Some(class)
            );
        }
    }

    #[test]
    fn malformed_records_name_the_field() {
        let mut v = sweep().to_json();
        v["family"] = json!("Nope");
        assert_eq!(
            SweepRecord::from_json(&v).unwrap_err(),
            "field family: unknown value 'Nope'"
        );
        let mut v = sweep().to_json();
        v.as_object_mut().unwrap().remove("median_s");
        assert_eq!(SweepRecord::from_json(&v).unwrap_err(), "no field median_s");
        let mut v = sweep().to_json();
        v["units"] = json!(-1);
        assert_eq!(
            SweepRecord::from_json(&v).unwrap_err(),
            "field units is not a u32"
        );
    }

    #[test]
    fn load_dir_reads_only_m3s_files() {
        let dir = std::env::temp_dir().join(format!("gridcl-bench-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write_records(&dir.join("m3s-sweep-cpu-x.json"), &[sweep().to_json()]).unwrap();
        write_records(
            &dir.join("m3s-throughput-cpu-x.json"),
            &[throughput().to_json()],
        )
        .unwrap();
        std::fs::write(dir.join("m2-primitives-cpu.json"), "[{\"other\": 1}]").unwrap();
        let data = load_dir(&dir).unwrap();
        assert_eq!(data.sweeps, vec![sweep()]);
        assert_eq!(data.throughput, vec![throughput()]);

        std::fs::write(dir.join("m3s-sweep-bad.json"), "[{}]").unwrap();
        let error = load_dir(&dir).unwrap_err();
        assert!(
            error.contains("m3s-sweep-bad.json: record 0: no field"),
            "{error}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn node_load_reads_the_last_snapshot_line() {
        let dir = std::env::temp_dir().join(format!("gridcl-bench-load-{}-nl", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("node-load-before.txt"),
            "gpu: 0%\nnode_load=busy: x\nnode_load=idle\n",
        )
        .unwrap();
        let load = NodeLoad::from_results_dir(&dir);
        assert_eq!(load.before, "idle");
        assert_eq!(load.after, "not recorded");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
