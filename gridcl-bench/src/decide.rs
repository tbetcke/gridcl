//! The decision rule of "Benchmarks" as ADR-0015 reads it (items 5 and 6): per kernel family,
//! device class and element size, the vector size, CPU block, CPU units and QuadField tile
//! written to `gridcl-device/vector-policy.toml`.
//!
//! - A configuration is (vector size, tile, CPU block, CPU units) (ADR-0017). On CPUs the units
//!   are all hardware threads (`CubeDimMode::Plan` records) or the performance cores
//!   (`PerformanceCores` records). A device without a performance-core count gives both
//!   settings the same time, its all-threads time. On GPUs the units are `None`.
//! - The class time of a configuration is the geometric mean, over the class's devices, of
//!   their medians, for the configurations every device measured; ratios of class times are
//!   then geometric means of per-device ratios (item 5). A device with fewer lanes contributes
//!   the time it ran at its widest size, because the sweep requests every size up to
//!   `VECTOR_PAD` on every device.
//! - A size above 1 is admitted if its best class time over tiles and blocks is at least 10%
//!   below size 1's best, at 10^6 and at 10^7 items. Among the admitted configurations within 3%
//!   of the fastest admitted one at both sizes, the smallest vector size wins, then the smallest
//!   tile, then the smallest block (item 6), then all hardware threads (ADR-0017).
//! - A 4-byte entry shared by f32 and u32 takes a size only if every 4-byte type measured admits
//!   it, and must be within 3% for each of them.
//! - A configuration whose requested size ran narrower on every device is chosen only if no
//!   other configuration qualifies, so the TOML holds sizes that ran (item 6, register budget).
//! - Backstop (ADR-0018): if no admitted configuration is within 3% at both sizes (for every
//!   type of a shared entry), the one with the smallest worst-case ratio is chosen if that ratio
//!   is at most 1.10, and its reason starts "closest, x% slower". Above 1.10 the entry is
//!   unresolved: it still holds the closest configuration, and the case goes to the maintainer.
//! - CpuSimd256 copies the CpuSimd512 table ("Benchmarks", "Where").

use std::collections::{BTreeMap, BTreeSet};

use gridcl_device::{CpuUnits, DeviceClass, KernelKind};

use crate::families;
use crate::record::{CLASSES, CubeDimMode, KINDS, PerfData, Scalar};

/// The problem sizes of the decision rule.
pub const RULE_SIZES: [usize; 2] = [1_000_000, 10_000_000];
/// A size above 1 is admitted if size 1 takes at least this factor longer.
pub const ADOPT: f64 = 1.1;
/// A configuration qualifies if it takes at most this factor of the fastest admitted one.
pub const WITHIN: f64 = 1.03;
/// The backstop takes the closest admitted configuration only up to this worst-case ratio
/// (ADR-0018).
pub const BACKSTOP: f64 = 1.1;
/// The element sizes of the policy, in bytes.
pub const ELEM_BYTES: [usize; 2] = [4, 8];
/// The unit of `cpu_block` entries in `vector-policy.toml`.
pub const BLOCK_UNIT: usize = 4 << 10;
/// The policy's values where there is no data: vector size 1, tile 16, 16 KiB blocks.
pub const DEFAULT_TILE: usize = 16;
pub const DEFAULT_BLOCK_BYTES: usize = 16 << 10;

/// A configuration: (requested vector size, tile, CPU block in bytes, CPU units).
pub type Config = (usize, Option<usize>, Option<usize>, Option<CubeDimMode>);

/// A device of a class: (target, backend).
pub type Device = (String, String);

/// What one device, or a class, measured for one configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    /// The median, or for a class the geometric mean of the devices' medians, in seconds.
    pub time_s: f64,
    /// The vector size that ran; for a class, the widest over its devices.
    pub ran: usize,
}

/// The choice for one (device class, kernel kind, element size) entry of the policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Choice {
    pub class: DeviceClass,
    pub kind: KernelKind,
    pub elem_bytes: usize,
    pub vector_size: usize,
    /// CPU block in units of 4 KiB; `None` on `Gpu`.
    pub cpu_block: Option<usize>,
    /// CPU units; `None` on `Gpu`.
    pub cpu_units: Option<CpuUnits>,
    /// How the rule reached the choice.
    pub fit: Fit,
    /// Why, in one line for the report.
    pub reason: String,
}

/// How the rule reached a choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Within 3% of the fastest admitted configuration at both sizes, or the defaults where
    /// there is no data.
    Rule,
    /// The backstop: no admitted configuration is within 3%, and this one is the closest, at
    /// most 10% slower (ADR-0018).
    Closest,
    /// No admitted configuration within 10%: the closest is stored, and the case goes to the
    /// maintainer.
    Unresolved,
}

/// Every entry of the policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub choices: Vec<Choice>,
    /// (class, element size, tile) for every class and element size.
    pub quad_tiles: Vec<(DeviceClass, usize, usize)>,
}

impl Decision {
    /// The choice of one entry.
    pub fn choice(
        &self,
        class: DeviceClass,
        kind: KernelKind,
        elem_bytes: usize,
    ) -> Option<&Choice> {
        self.choices
            .iter()
            .find(|c| c.class == class && c.kind == kind && c.elem_bytes == elem_bytes)
    }

    /// The entries the rule could not resolve (ADR-0018).
    pub fn unresolved(&self) -> impl Iterator<Item = &Choice> {
        self.choices.iter().filter(|c| c.fit == Fit::Unresolved)
    }

    /// The tile of one class and element size.
    pub fn quad_tile(&self, class: DeviceClass, elem_bytes: usize) -> usize {
        self.quad_tiles
            .iter()
            .find(|&&(c, b, _)| c == class && b == elem_bytes)
            .map_or(DEFAULT_TILE, |&(_, _, t)| t)
    }

    /// The text of `gridcl-device/vector-policy.toml`.
    pub fn to_toml(&self) -> String {
        let mut out = String::from(TOML_HEADER);
        for class in CLASSES {
            let c = format!("{class:?}");
            out.push_str(&format!(
                "\n[{c}]\nquad_tile = {}\n[{c}.vector_size]\n",
                self.row(|bytes| self.quad_tile(class, bytes))
            ));
            for kind in KINDS {
                let row =
                    self.row(|bytes| self.choice(class, kind, bytes).map_or(1, |c| c.vector_size));
                out.push_str(&format!("{kind:?} = {row}\n"));
            }
            if class != DeviceClass::Gpu {
                out.push_str(&format!("[{c}.cpu_block]\n"));
                for kind in KINDS {
                    let row = self.row(|bytes| {
                        self.choice(class, kind, bytes)
                            .and_then(|c| c.cpu_block)
                            .unwrap_or(DEFAULT_BLOCK_BYTES / BLOCK_UNIT)
                    });
                    out.push_str(&format!("{kind:?} = {row}\n"));
                }
                out.push_str(&format!("[{c}.cpu_units]\n"));
                for kind in KINDS {
                    let row = self.row(|bytes| {
                        let units = self
                            .choice(class, kind, bytes)
                            .and_then(|c| c.cpu_units)
                            .unwrap_or(CpuUnits::All);
                        format!("\"{}\"", units_name(units))
                    });
                    out.push_str(&format!("{kind:?} = {row}\n"));
                }
            }
        }
        out
    }

    /// `{ 4 = a, 8 = b }`.
    fn row<T: std::fmt::Display>(&self, value: impl Fn(usize) -> T) -> String {
        let entries: Vec<String> = ELEM_BYTES
            .iter()
            .map(|&bytes| format!("{bytes} = {}", value(bytes)))
            .collect();
        format!("{{ {} }}", entries.join(", "))
    }
}

const TOML_HEADER: &str = "\
# Vector policy of gridcl-device (docs/design.md, \"Launch plans and the vector policy\"; ADR-0015).
#
# Per device class: the QuadField tile width by scalar size in bytes; the vector size per kernel
# family and scalar size; on CPU classes, the CPU block per kernel family and scalar size of the
# widest stream, in units of 4 KiB, and the CPU units (all hardware threads or the performance
# cores; ADR-0017). Written by `cargo run -p gridcl-bench --bin policy_report` from
# docs/perf/data/ with the decision rule of docs/perf/vectorization.md; do not edit by hand.
";

/// The value of a `cpu_units` entry: `all` or `performance`.
pub fn units_name(units: CpuUnits) -> &'static str {
    match units {
        CpuUnits::All => "all",
        CpuUnits::Performance => "performance",
    }
}

/// The CPU units a sweep record's `cube_dim` mode measured.
pub fn units_of(mode: CubeDimMode) -> CpuUnits {
    match mode {
        CubeDimMode::Plan => CpuUnits::All,
        CubeDimMode::PerformanceCores => CpuUnits::Performance,
    }
}

/// The decision for `data`.
///
/// # Panics
///
/// If one device recorded the same configuration twice at one size.
pub fn decide(data: &PerfData) -> Decision {
    let mut choices = Vec::new();
    let mut quad_tiles = Vec::new();
    for class in CLASSES {
        if class == DeviceClass::CpuSimd256 {
            continue; // copied below
        }
        for bytes in ELEM_BYTES {
            let mut tile = DEFAULT_TILE;
            for kind in KINDS {
                let entry = choose(data, class, kind, bytes);
                if kind == KernelKind::GeomFactors
                    && let Some((_, Some(t), ..)) = entry.config
                {
                    tile = t;
                }
                choices.push(Choice {
                    class,
                    kind,
                    elem_bytes: bytes,
                    vector_size: entry.config.map_or(1, |c| c.0),
                    cpu_block: (class != DeviceClass::Gpu).then(|| {
                        entry
                            .config
                            .and_then(|c| c.2)
                            .unwrap_or(DEFAULT_BLOCK_BYTES)
                            / BLOCK_UNIT
                    }),
                    cpu_units: (class != DeviceClass::Gpu).then(|| {
                        entry
                            .config
                            .and_then(|c| c.3)
                            .map_or(CpuUnits::All, units_of)
                    }),
                    fit: entry.fit,
                    reason: entry.reason,
                });
            }
            quad_tiles.push((class, bytes, tile));
        }
    }
    let copies: Vec<Choice> = choices
        .iter()
        .filter(|c| c.class == DeviceClass::CpuSimd512)
        .map(|c| Choice {
            class: DeviceClass::CpuSimd256,
            reason: format!("copies CpuSimd512: {}", c.reason),
            ..c.clone()
        })
        .collect();
    choices.extend(copies);
    let tiles: Vec<_> = quad_tiles
        .iter()
        .filter(|t| t.0 == DeviceClass::CpuSimd512)
        .map(|&(_, b, t)| (DeviceClass::CpuSimd256, b, t))
        .collect();
    quad_tiles.extend(tiles);
    let order = |c: DeviceClass| CLASSES.iter().position(|&x| x == c);
    choices.sort_by_key(|c| {
        (
            order(c.class),
            c.elem_bytes,
            KINDS.iter().position(|&k| k == c.kind),
        )
    });
    quad_tiles.sort_by_key(|&(c, b, _)| (order(c), b));
    Decision {
        choices,
        quad_tiles,
    }
}

/// The scalar types the sweep measures `kind` in; empty for kinds without a kernel yet.
pub fn scalars_of(kind: KernelKind) -> Vec<Scalar> {
    families::families()
        .into_iter()
        .find(|&(k, _)| k == kind)
        .map(|(_, s)| s.to_vec())
        .unwrap_or_default()
}

/// What each device of `class` measured for `kind` in `scalar` at `size`, by configuration. A
/// CPU without a performance-core count gives its all-threads time to both unit settings.
///
/// # Panics
///
/// If one device recorded the same configuration twice.
pub fn device_times(
    data: &PerfData,
    class: DeviceClass,
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
) -> BTreeMap<Device, BTreeMap<Config, Timing>> {
    let cpu = class != DeviceClass::Gpu;
    let mut per_device: BTreeMap<Device, BTreeMap<Config, Timing>> = BTreeMap::new();
    let mut neutral = Vec::new();
    for r in data.sweeps.iter().filter(|r| {
        r.device_class == class
            && r.family == kind
            && r.scalar == scalar
            && r.nominal_n == size
            && (cpu || r.cube_dim == CubeDimMode::Plan)
    }) {
        let config = (
            r.vector_size,
            r.tile,
            r.block_bytes,
            cpu.then_some(r.cube_dim),
        );
        let timing = Timing {
            time_s: r.median_s,
            ran: r.vector_size_ran,
        };
        let device = (r.target.clone(), r.backend.clone());
        if cpu && r.cube_dim == CubeDimMode::Plan && r.cores.performance.is_none() {
            let same = (
                config.0,
                config.1,
                config.2,
                Some(CubeDimMode::PerformanceCores),
            );
            neutral.push((device.clone(), same, timing));
        }
        let previous = per_device.entry(device).or_default().insert(config, timing);
        assert!(
            previous.is_none(),
            "{} {} {kind:?} {} at {size}: configuration {config:?} recorded twice",
            r.target,
            r.backend,
            scalar.name()
        );
    }
    for (device, config, timing) in neutral {
        per_device
            .entry(device)
            .or_default()
            .entry(config)
            .or_insert(timing);
    }
    per_device
}

/// The class time of every configuration that every device of `class` measured.
///
/// # Panics
///
/// As [`device_times`].
pub fn class_times(
    data: &PerfData,
    class: DeviceClass,
    kind: KernelKind,
    scalar: Scalar,
    size: usize,
) -> BTreeMap<Config, Timing> {
    let per_device = device_times(data, class, kind, scalar, size);
    let mut out = BTreeMap::new();
    let Some(first) = per_device.values().next() else {
        return out;
    };
    let devices = per_device.len();
    for config in first.keys() {
        let timings: Vec<Timing> = per_device
            .values()
            .filter_map(|times| times.get(config).copied())
            .collect();
        if timings.len() == devices {
            let logs: Vec<f64> = timings.iter().map(|t| t.time_s.ln()).collect();
            out.insert(
                *config,
                Timing {
                    time_s: (logs.iter().sum::<f64>() / devices as f64).exp(),
                    ran: timings.iter().map(|t| t.ran).max().unwrap_or(1),
                },
            );
        }
    }
    out
}

/// The smallest time of the configurations whose vector size satisfies `keep`.
pub fn best(times: &BTreeMap<Config, Timing>, keep: impl Fn(usize) -> bool) -> Option<f64> {
    times
        .iter()
        .filter(|((n, ..), _)| keep(*n))
        .map(|(_, t)| t.time_s)
        .min_by(f64::total_cmp)
}

/// Size 1's best time over size `v`'s best time.
pub fn speedup(times: &BTreeMap<Config, Timing>, v: usize) -> Option<f64> {
    Some(best(times, |m| m == 1)? / best(times, |m| m == v)?)
}

/// The rule's view of one scalar type: class times at each rule size and the admitted sizes.
struct TypeView {
    scalar: Scalar,
    times: [BTreeMap<Config, Timing>; 2],
    /// Sizes measured at 10^6: 1, and the sizes at least 10% faster than 1 at both rule sizes.
    admitted: Vec<usize>,
    /// The fastest admitted time at each rule size.
    fastest: [f64; 2],
}

impl TypeView {
    fn new(
        data: &PerfData,
        class: DeviceClass,
        kind: KernelKind,
        scalar: Scalar,
    ) -> Option<TypeView> {
        let times = RULE_SIZES.map(|size| class_times(data, class, kind, scalar, size));
        let mut sizes: Vec<usize> = times[0].keys().map(|c| c.0).collect();
        sizes.dedup();
        let admitted: Vec<usize> = sizes
            .into_iter()
            .filter(|&v| {
                v == 1
                    || times
                        .iter()
                        .all(|t| speedup(t, v).is_some_and(|s| s >= ADOPT))
            })
            .collect();
        let fastest = [0, 1].map(|i| best(&times[i], |v| admitted.contains(&v)));
        Some(TypeView {
            scalar,
            fastest: [fastest[0]?, fastest[1]?],
            times,
            admitted,
        })
    }

    /// The configuration's time over the fastest admitted, at each rule size.
    fn ratios(&self, config: &Config) -> Option<[f64; 2]> {
        let t0 = self.times[0].get(config)?.time_s;
        let t1 = self.times[1].get(config)?.time_s;
        Some([t0 / self.fastest[0], t1 / self.fastest[1]])
    }

    fn within(&self, config: &Config) -> bool {
        [0, 1].into_iter().all(|i| {
            self.times[i]
                .get(config)
                .is_some_and(|t| t.time_s <= WITHIN * self.fastest[i])
        })
    }
}

/// The chosen configuration of one entry, `None` for the defaults, with the reason.
struct Entry {
    config: Option<Config>,
    fit: Fit,
    reason: String,
}

fn defaults(reason: impl Into<String>) -> Entry {
    Entry {
        config: None,
        fit: Fit::Rule,
        reason: reason.into(),
    }
}

fn choose(data: &PerfData, class: DeviceClass, kind: KernelKind, bytes: usize) -> Entry {
    let types: Vec<Scalar> = scalars_of(kind)
        .into_iter()
        .filter(|s| s.bytes() == bytes)
        .collect();
    if scalars_of(kind).is_empty() {
        return defaults("no kernel before M6/M9; measured in the M10 re-run");
    }
    if types.is_empty() {
        return defaults(format!("not used: no {bytes}-byte type"));
    }
    let measured: Vec<Scalar> = types
        .iter()
        .copied()
        .filter(|&s| !class_times(data, class, kind, s, RULE_SIZES[0]).is_empty())
        .collect();
    if measured.is_empty() {
        return defaults("no data");
    }
    let mut views = Vec::new();
    for &scalar in &measured {
        match TypeView::new(data, class, kind, scalar) {
            Some(view) => views.push(view),
            None => return defaults(format!("{}: no data at both 10^6 and 10^7", scalar.name())),
        }
    }

    // Configurations every type measured at both sizes, at a size every type admits.
    let candidates: Vec<Config> = views[0].times[0]
        .keys()
        .copied()
        .filter(|c| {
            views.iter().all(|v| {
                v.admitted.contains(&c.0)
                    && v.times[0].contains_key(c)
                    && v.times[1].contains_key(c)
            })
        })
        .collect();
    let narrower = |c: &Config| views.iter().all(|v| v.times[0][c].ran < c.0);
    let within = |c: &Config| views.iter().all(|v| v.within(c));
    let chosen = candidates
        .iter()
        .find(|c| within(c) && !narrower(c))
        .or_else(|| candidates.iter().find(|c| within(c)));
    let Some(&config) = chosen else {
        // No configuration within 3% for every type at both sizes: the backstop (ADR-0018).
        let worst = |c: &Config| {
            views
                .iter()
                .filter_map(|v| v.ratios(c))
                .flatten()
                .fold(1.0f64, f64::max)
        };
        let Some(config) = candidates
            .iter()
            .copied()
            .min_by(|a, b| worst(a).total_cmp(&worst(b)))
        else {
            return Entry {
                fit: Fit::Unresolved,
                ..defaults("unresolved: no configuration measured at both sizes for every type")
            };
        };
        let slower = (worst(&config) - 1.0) * 100.0;
        let (fit, text) = if worst(&config) <= BACKSTOP {
            (
                Fit::Closest,
                format!(
                    "closest, {slower:.1}% slower: no admitted configuration within 3% of the \
                     fastest at both sizes (ADR-0018)"
                ),
            )
        } else {
            (
                Fit::Unresolved,
                format!(
                    "unresolved: the closest admitted configuration is {slower:.1}% slower, \
                     above the 10% bound (ADR-0018)"
                ),
            )
        };
        return Entry {
            config: Some(config),
            fit,
            reason: with_partial(&views, text),
        };
    };
    Entry {
        config: Some(config),
        fit: Fit::Rule,
        reason: with_partial(&views, reason(&views, config, narrower(&config))),
    }
}

/// The reason for `config`, in one line.
///
/// Size 1 is never within 3% of an admitted size above 1, which is at least 10% faster, so a
/// choice of 1 means that no size above 1 is admitted for every type.
fn reason(views: &[TypeView], config: Config, narrower: bool) -> String {
    let n = config.0;
    let mut text = if n > 1 {
        let s = [0, 1].map(|i| {
            views
                .iter()
                .filter_map(|t| speedup(&t.times[i], n))
                .fold(f64::INFINITY, f64::min)
        });
        format!(
            "{:.2}x / {:.2}x over size 1; smallest admitted size within 3% of the fastest",
            s[0], s[1]
        )
    } else {
        "no size at least 10% faster than 1 at 10^6 and 10^7".to_owned()
    };
    if config.3 == Some(CubeDimMode::PerformanceCores) {
        // The best admitted all-threads time over the chosen time, the smallest over the types.
        let gain = [0, 1].map(|i| {
            views
                .iter()
                .filter_map(|v| {
                    let all = v.times[i]
                        .iter()
                        .filter(|(c, _)| v.admitted.contains(&c.0))
                        .filter(|(c, _)| c.3 == Some(CubeDimMode::Plan))
                        .map(|(_, t)| t.time_s)
                        .min_by(f64::total_cmp)?;
                    Some(all / v.times[i].get(&config)?.time_s)
                })
                .fold(f64::INFINITY, f64::min)
        });
        if gain.iter().all(|g| g.is_finite()) {
            text.push_str(&format!(
                "; performance cores, {:.2}x / {:.2}x over all threads",
                gain[0], gain[1]
            ));
        } else {
            text.push_str("; performance cores");
        }
    }
    if narrower {
        let ran = views
            .iter()
            .map(|v| v.times[0][&config].ran)
            .max()
            .unwrap_or(n);
        text.push_str(&format!("; runs at {ran}"));
    }
    text
}

/// `text` with a note after it on the sizes that some 4-byte types admit but not all.
fn with_partial(views: &[TypeView], text: String) -> String {
    let partial: BTreeSet<usize> = views
        .iter()
        .flat_map(|v| v.admitted.iter().copied())
        .filter(|&v| v > 1 && !views.iter().all(|t| t.admitted.contains(&v)))
        .collect();
    if partial.is_empty() {
        return text;
    }
    let sizes: Vec<String> = partial.iter().map(usize::to_string).collect();
    let who: Vec<&str> = views
        .iter()
        .filter(|v| v.admitted.iter().any(|a| partial.contains(a)))
        .map(|v| v.scalar.name())
        .collect();
    let note = format!(
        "size {} admitted for {} only (shared 4-byte entry)",
        sizes.join(", "),
        who.join(", ")
    );
    format!("{text}; {note}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{Cores, SweepRecord};

    /// What a record measured: (requested vector size, tile, CPU block in bytes).
    type Spec = (usize, Option<usize>, Option<usize>);

    /// A record of `target` (backend from the class) with median `t`, on all hardware threads
    /// of a CPU without a performance-core count.
    fn rec(
        class: DeviceClass,
        target: &str,
        kind: KernelKind,
        scalar: Scalar,
        n: usize,
        (v, tile, block): Spec,
        t: f64,
    ) -> SweepRecord {
        SweepRecord {
            backend: if class == DeviceClass::Gpu {
                "gpu"
            } else {
                "cpu"
            }
            .to_owned(),
            device: target.to_owned(),
            device_class: class,
            target: target.to_owned(),
            family: kind,
            scalar,
            nominal_n: n,
            n,
            vector_size: v,
            vector_size_ran: v,
            tile,
            block_bytes: block,
            cube_dim: CubeDimMode::Plan,
            units: 1,
            times_s: vec![t; 10],
            median_s: t,
            calls_per_run: 1,
            calibration_s: t,
            compile_s: 0.0,
            kernel_cache: "off".to_owned(),
            bandwidth_ratio: None,
            commit: "test".to_owned(),
            profile: "release".to_owned(),
            cores: Cores {
                performance: None,
                efficiency: None,
                threads: 1,
            },
            fallback_reason: None,
            node_load: None,
        }
    }

    const CPU: DeviceClass = DeviceClass::CpuSimd128;
    const K16: Option<usize> = Some(16 << 10);

    /// Records at both rule sizes for every (size, time at 10^6); 10^7 takes ten times as long.
    fn family(
        class: DeviceClass,
        target: &str,
        kind: KernelKind,
        scalar: Scalar,
        times: &[(Spec, f64)],
    ) -> Vec<SweepRecord> {
        RULE_SIZES
            .iter()
            .flat_map(|&n| {
                let scale = (n / RULE_SIZES[0]) as f64;
                times
                    .iter()
                    .map(move |&(c, t)| rec(class, target, kind, scalar, n, c, t * scale))
            })
            .collect()
    }

    fn sizes(times: &[(usize, f64)]) -> Vec<(Spec, f64)> {
        times.iter().map(|&(v, t)| ((v, None, K16), t)).collect()
    }

    fn data(sweeps: Vec<SweepRecord>) -> PerfData {
        PerfData {
            sweeps,
            throughput: Vec::new(),
        }
    }

    fn entry(d: &Decision, class: DeviceClass, kind: KernelKind, bytes: usize) -> &Choice {
        d.choice(class, kind, bytes).unwrap()
    }

    #[test]
    fn no_data_gives_the_current_defaults() {
        let d = decide(&PerfData::default());
        let policy: toml::Table = d.to_toml().parse().unwrap();
        for class in CLASSES {
            let c = &policy[&format!("{class:?}")];
            assert_eq!(c["quad_tile"]["8"].as_integer(), Some(16));
            assert_eq!(c["vector_size"]["PullBack"]["4"].as_integer(), Some(1));
            let block = c.get("cpu_block").map(|b| b["Scalar"]["8"].as_integer());
            assert_eq!(
                block,
                (class != DeviceClass::Gpu).then_some(Some(4)),
                "{class:?}"
            );
        }
        for c in &d.choices {
            assert_eq!(c.vector_size, 1, "{c:?}");
            assert_eq!(
                c.cpu_block,
                (c.class != DeviceClass::Gpu).then_some(4),
                "{c:?}"
            );
            assert_eq!(
                c.cpu_units,
                (c.class != DeviceClass::Gpu).then_some(CpuUnits::All),
                "{c:?}"
            );
        }
        for class in CLASSES {
            let units = policy[&format!("{class:?}")]
                .get("cpu_units")
                .map(|u| u["GeomFactors"]["4"].as_str());
            assert_eq!(
                units,
                (class != DeviceClass::Gpu).then_some(Some("all")),
                "{class:?}"
            );
        }
        assert!(d.quad_tiles.iter().all(|&(_, _, t)| t == DEFAULT_TILE));
        assert_eq!(
            d.choices.len(),
            CLASSES.len() * KINDS.len() * ELEM_BYTES.len()
        );
        let pack = entry(&d, CPU, KernelKind::Pack, 4);
        assert!(pack.reason.contains("M10"), "{}", pack.reason);
        assert_eq!(
            entry(&d, CPU, KernelKind::Copy, 8).reason,
            "not used: no 8-byte type"
        );
        assert_eq!(entry(&d, CPU, KernelKind::Fill, 8).reason, "no data");
    }

    #[test]
    fn gains_of_three_to_ten_percent_stay_at_one() {
        let d = decide(&data(family(
            CPU,
            "a",
            KernelKind::Fill,
            Scalar::F64,
            &sizes(&[(1, 1.0), (2, 0.95), (4, 0.92)]),
        )));
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(c.vector_size, 1);
        assert!(c.reason.starts_with("no size at least 10%"), "{}", c.reason);
    }

    #[test]
    fn a_gain_at_only_one_size_stays_at_one() {
        let mut sweeps = family(
            CPU,
            "a",
            KernelKind::Fill,
            Scalar::F64,
            &sizes(&[(1, 1.0), (4, 0.8)]),
        );
        // At 10^7 size 4 is only 5% faster.
        for r in &mut sweeps {
            if r.nominal_n == RULE_SIZES[1] && r.vector_size == 4 {
                r.median_s = 9.5;
            }
        }
        let d = decide(&data(sweeps));
        assert_eq!(entry(&d, CPU, KernelKind::Fill, 8).vector_size, 1);
    }

    #[test]
    fn the_smallest_size_within_three_percent_wins() {
        let d = decide(&data(family(
            CPU,
            "a",
            KernelKind::Reduce,
            Scalar::F64,
            &sizes(&[(1, 1.0), (2, 0.85), (4, 0.70), (8, 0.69), (16, 0.80)]),
        )));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!(c.vector_size, 4, "{}", c.reason);
        assert!(c.reason.starts_with("1.43x / 1.43x"), "{}", c.reason);
        // At size 1 the smallest block within 3% of the fastest wins.
        let d = decide(&data(family(
            CPU,
            "a",
            KernelKind::Reduce,
            Scalar::F64,
            &[
                ((1, None, Some(4 << 10)), 0.92),
                ((1, None, K16), 1.0),
                ((1, None, Some(64 << 10)), 0.90),
                ((2, None, K16), 0.86),
            ],
        )));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!((c.vector_size, c.cpu_block), (1, Some(1)), "{}", c.reason);
        assert_eq!(
            c.reason,
            "no size at least 10% faster than 1 at 10^6 and 10^7"
        );
    }

    #[test]
    fn tile_and_block_ties_take_the_smallest() {
        let k = |b: usize| Some(b << 10);
        let d = decide(&data(family(
            CPU,
            "a",
            KernelKind::GeomFactors,
            Scalar::F32,
            &[
                ((1, Some(1), k(4)), 1.0),
                ((1, Some(4), k(4)), 1.0),
                ((4, Some(4), k(4)), 0.5),
                ((4, Some(4), k(16)), 0.405),
                ((4, Some(4), k(64)), 0.40),
                ((4, Some(16), k(64)), 0.40),
                ((4, Some(16), k(16)), 0.40),
            ],
        )));
        // Tile 4 before 16, then the 16 KiB block before 64 KiB.
        let c = entry(&d, CPU, KernelKind::GeomFactors, 4);
        assert_eq!((c.vector_size, c.cpu_block), (4, Some(4)), "{}", c.reason);
        assert_eq!(d.quad_tile(CPU, 4), 4);
        assert_eq!(d.quad_tile(CPU, 8), DEFAULT_TILE);
        assert!(
            d.to_toml()
                .contains("[CpuSimd128]\nquad_tile = { 4 = 4, 8 = 16 }\n")
        );
    }

    #[test]
    fn devices_of_a_class_combine_by_geometric_means() {
        let gpu = DeviceClass::Gpu;
        let mut sweeps = Vec::new();
        // Device a gains 1.5x at size 4; device b only 1.0x: the mean is 1.22x, admitted.
        // At 8 and 16, a runs at 4 lanes (clamped) and b gains 1.3x: the mean is 1.40x.
        let none = |v| (v, None, None);
        sweeps.extend(family(
            gpu,
            "a",
            KernelKind::Fill,
            Scalar::F32,
            &[
                (none(1), 1.5),
                (none(4), 1.0),
                (none(8), 1.0),
                (none(16), 1.0),
            ],
        ));
        for r in &mut sweeps {
            r.vector_size_ran = r.vector_size.min(4);
        }
        sweeps.extend(family(
            gpu,
            "b",
            KernelKind::Fill,
            Scalar::F32,
            &[
                (none(1), 1.3),
                (none(4), 1.3),
                (none(8), 1.0),
                (none(16), 0.995),
            ],
        ));
        // b alone measures f64 (a has no f64): b's own ratio decides.
        sweeps.extend(family(
            gpu,
            "b",
            KernelKind::Fill,
            Scalar::F64,
            &[(none(1), 1.0), (none(2), 0.5)],
        ));
        let d = decide(&data(sweeps.clone()));
        let c = entry(&d, gpu, KernelKind::Fill, 4);
        assert_eq!((c.vector_size, c.cpu_block), (8, None), "{}", c.reason);
        assert_eq!(entry(&d, gpu, KernelKind::Fill, 8).vector_size, 2);
        let times = class_times(
            &data(sweeps),
            gpu,
            KernelKind::Fill,
            Scalar::F32,
            RULE_SIZES[0],
        );
        assert!((times[&(4, None, None, None)].time_s - 1.3f64.sqrt()).abs() < 1e-12);
        assert_eq!(times[&(8, None, None, None)].ran, 8);
        assert!(!d.to_toml().contains("[Gpu.cpu_block]"));
    }

    #[test]
    fn a_size_that_ran_narrower_everywhere_is_not_chosen() {
        let narrow = |times: &[(usize, f64)]| {
            let mut sweeps = family(CPU, "a", KernelKind::Fill, Scalar::F64, &sizes(times));
            for r in &mut sweeps {
                r.vector_size_ran = if r.vector_size == 8 { 4 } else { r.vector_size };
            }
            decide(&data(sweeps))
        };
        // Size 8 ran at 4; 4 is just outside 3% by noise, 8 and 16 within: 16 is stored.
        let d = narrow(&[(1, 1.0), (4, 0.62), (8, 0.60), (16, 0.605)]);
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(c.vector_size, 16, "{}", c.reason);
        // Only 8 within 3%: stored, and the reason says what runs.
        let d = narrow(&[(1, 1.0), (4, 0.62), (8, 0.60)]);
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(c.vector_size, 8);
        assert!(c.reason.ends_with("; runs at 4"), "{}", c.reason);
    }

    #[test]
    fn a_shared_four_byte_entry_needs_every_type() {
        let mut sweeps = family(
            CPU,
            "a",
            KernelKind::Gather,
            Scalar::F32,
            &sizes(&[(1, 1.0), (4, 0.5)]),
        );
        sweeps.extend(family(
            CPU,
            "a",
            KernelKind::Gather,
            Scalar::U32,
            &sizes(&[(1, 1.0), (4, 0.95)]),
        ));
        let d = decide(&data(sweeps.clone()));
        let c = entry(&d, CPU, KernelKind::Gather, 4);
        // Size 1 is 100% slower than f32's fastest: beyond the backstop's bound (ADR-0018).
        assert_eq!((c.vector_size, c.fit), (1, Fit::Unresolved));
        assert!(
            c.reason.starts_with("unresolved: ")
                && c.reason
                    .ends_with("; size 4 admitted for f32 only (shared 4-byte entry)"),
            "{}",
            c.reason
        );
        // Both admit 4: it is chosen.
        sweeps.extend(family(
            CPU,
            "b",
            KernelKind::Fill,
            Scalar::F32,
            &sizes(&[(1, 1.0), (4, 0.5)]),
        ));
        sweeps.extend(family(
            CPU,
            "b",
            KernelKind::Fill,
            Scalar::U32,
            &sizes(&[(1, 1.0), (4, 0.8)]),
        ));
        let d = decide(&data(sweeps));
        assert_eq!(entry(&d, CPU, KernelKind::Fill, 4).vector_size, 4);
    }

    #[test]
    fn cpu_simd_256_copies_cpu_simd_512() {
        let avx = DeviceClass::CpuSimd512;
        let mut sweeps = family(
            avx,
            "k",
            KernelKind::Copy,
            Scalar::U32,
            &sizes(&[(1, 1.0), (16, 0.5)]),
        );
        sweeps.extend(family(
            avx,
            "k",
            KernelKind::Scalar,
            Scalar::F64,
            &[
                ((1, None, Some(4 << 10)), 1.0),
                ((1, None, Some(64 << 10)), 0.5),
            ],
        ));
        let d = decide(&data(sweeps));
        for class in [avx, DeviceClass::CpuSimd256] {
            assert_eq!(entry(&d, class, KernelKind::Copy, 4).vector_size, 16);
            assert_eq!(entry(&d, class, KernelKind::Scalar, 8).cpu_block, Some(16));
            assert_eq!(entry(&d, class, KernelKind::Scalar, 8).vector_size, 1);
        }
        let policy: toml::Table = d.to_toml().parse().unwrap();
        assert_eq!(policy["CpuSimd256"], policy["CpuSimd512"]);
        assert_eq!(entry(&d, CPU, KernelKind::Copy, 4).vector_size, 1);
    }

    #[test]
    fn no_configuration_within_three_percent_at_both_sizes() {
        let mut sweeps = family(
            CPU,
            "a",
            KernelKind::Fill,
            Scalar::F64,
            &[
                ((1, None, Some(4 << 10)), 1.0),
                ((1, None, Some(16 << 10)), 1.1),
            ],
        );
        // At 10^7 the 16 KiB block is 10% faster.
        for r in &mut sweeps {
            if r.nominal_n == RULE_SIZES[1] {
                r.median_s = if r.block_bytes == Some(4 << 10) {
                    11.0
                } else {
                    10.0
                };
            }
        }
        let d = decide(&data(sweeps));
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(c.vector_size, 1);
        // Both blocks are 10% slower than the fastest at one size: the smaller block, at the
        // bound.
        assert_eq!(c.cpu_block, Some(1));
        assert_eq!(c.fit, Fit::Closest, "{}", c.reason);
        assert!(
            c.reason.starts_with("closest, 10.0% slower"),
            "{}",
            c.reason
        );
        assert_eq!(d.unresolved().count(), 0);
    }

    /// Times at (10^6, 10^7) for a 4 KiB and a 16 KiB block at size 1.
    fn two_blocks(t4: (f64, f64), t16: (f64, f64)) -> PerfData {
        let mut sweeps = Vec::new();
        for (i, &n) in RULE_SIZES.iter().enumerate() {
            for (block, t) in [(4 << 10, t4), (16 << 10, t16)] {
                let t = if i == 0 { t.0 } else { t.1 };
                sweeps.push(rec(
                    CPU,
                    "a",
                    KernelKind::Fill,
                    Scalar::F64,
                    n,
                    (1, None, Some(block)),
                    t,
                ));
            }
        }
        data(sweeps)
    }

    #[test]
    fn the_backstop_takes_the_smallest_worst_ratio() {
        // 4 KiB: 6% slower at 10^7; 16 KiB: 5% slower at 10^6. Neither within 3%.
        let d = decide(&two_blocks((1.0, 10.6), (1.05, 10.0)));
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(
            (c.vector_size, c.cpu_block, c.fit),
            (1, Some(4), Fit::Closest)
        );
        assert!(c.reason.starts_with("closest, 5.0% slower"), "{}", c.reason);
    }

    #[test]
    fn no_backstop_above_ten_percent() {
        let d = decide(&two_blocks((1.0, 11.2), (1.12, 10.0)));
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!(c.fit, Fit::Unresolved, "{}", c.reason);
        assert!(
            c.reason.starts_with("unresolved: the closest"),
            "{}",
            c.reason
        );
        assert!(c.reason.contains("12.0% slower"), "{}", c.reason);
        let unresolved: Vec<_> = d
            .unresolved()
            .map(|c| (c.class, c.kind, c.elem_bytes))
            .collect();
        assert_eq!(unresolved, vec![(CPU, KernelKind::Fill, 8)]);
    }

    #[test]
    fn no_backstop_when_a_configuration_meets_three_percent() {
        // 16 KiB is 2% slower at 10^6 and the fastest at 10^7: within 3% at both sizes.
        let d = decide(&two_blocks((1.0, 10.6), (1.02, 10.0)));
        let c = entry(&d, CPU, KernelKind::Fill, 8);
        assert_eq!((c.cpu_block, c.fit), (Some(4), Fit::Rule), "{}", c.reason);
        assert!(!c.reason.contains("closest"), "{}", c.reason);
    }

    #[test]
    #[should_panic(expected = "recorded twice")]
    fn a_configuration_recorded_twice_panics() {
        let mut sweeps = family(CPU, "a", KernelKind::Fill, Scalar::F64, &sizes(&[(1, 1.0)]));
        sweeps.push(sweeps[0].clone());
        decide(&data(sweeps));
    }

    /// Records of `target` on a CPU with 12 performance cores: all hardware threads at the
    /// given times, and the performance cores at `factor` times them.
    fn split_cores(
        target: &str,
        kind: KernelKind,
        times: &[(Spec, f64)],
        factor: f64,
    ) -> Vec<SweepRecord> {
        let all = family(CPU, target, kind, Scalar::F64, times);
        let perf: Vec<SweepRecord> = all
            .iter()
            .map(|r| SweepRecord {
                cube_dim: CubeDimMode::PerformanceCores,
                median_s: r.median_s * factor,
                units: 12,
                ..r.clone()
            })
            .collect();
        all.into_iter()
            .chain(perf)
            .map(|r| SweepRecord {
                cores: Cores {
                    performance: Some(12),
                    efficiency: Some(4),
                    threads: 16,
                },
                ..r
            })
            .collect()
    }

    /// A gain of the performance cores within 3% stays at all hardware threads (ADR-0017).
    #[test]
    fn a_units_gain_within_three_percent_stays_all() {
        let sweeps = split_cores("a", KernelKind::Reduce, &sizes(&[(1, 1.0), (4, 0.5)]), 0.98);
        let d = decide(&data(sweeps));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!(
            (c.vector_size, c.cpu_units),
            (4, Some(CpuUnits::All)),
            "{}",
            c.reason
        );
        assert!(!c.reason.contains("performance cores"), "{}", c.reason);
    }

    /// A larger gain chooses the performance cores, which the TOML and the reason show.
    #[test]
    fn a_larger_units_gain_chooses_performance() {
        let sweeps = split_cores("a", KernelKind::Reduce, &sizes(&[(1, 1.0), (4, 0.5)]), 0.5);
        let d = decide(&data(sweeps));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!(
            (c.vector_size, c.cpu_units),
            (4, Some(CpuUnits::Performance)),
            "{}",
            c.reason
        );
        assert!(
            c.reason
                .ends_with("; performance cores, 2.00x / 2.00x over all threads"),
            "{}",
            c.reason
        );
        assert_eq!(
            entry(&d, CPU, KernelKind::Reduce, 4).cpu_units,
            Some(CpuUnits::All)
        );
        assert!(
            d.to_toml()
                .contains("[CpuSimd128.cpu_units]\nFill = { 4 = \"all\", 8 = \"all\" }\nCopy = { 4 = \"all\", 8 = \"all\" }\nReduce = { 4 = \"all\", 8 = \"performance\" }\n")
        );
        // The performance cores also count for the size-1 baseline: 4 must beat 1 on them.
        let sweeps = split_cores("a", KernelKind::Fill, &sizes(&[(1, 1.0), (4, 0.95)]), 0.5);
        let c = entry(&decide(&data(sweeps)), CPU, KernelKind::Fill, 8).clone();
        assert_eq!(
            (c.vector_size, c.cpu_units),
            (1, Some(CpuUnits::Performance)),
            "{}",
            c.reason
        );
    }

    /// A device without a performance-core count (Grace) gives both settings its time: alone it
    /// keeps all threads; with a split device it halves that device's gain in the geometric mean.
    #[test]
    fn devices_without_a_core_count_are_neutral() {
        let grace = family(
            CPU,
            "grace",
            KernelKind::Reduce,
            Scalar::F64,
            &sizes(&[(1, 1.0)]),
        );
        let d = decide(&data(grace.clone()));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!(c.cpu_units, Some(CpuUnits::All), "{}", c.reason);
        let times = class_times(
            &data(grace.clone()),
            CPU,
            KernelKind::Reduce,
            Scalar::F64,
            RULE_SIZES[0],
        );
        assert_eq!(
            times[&(1, None, K16, Some(CubeDimMode::Plan))],
            times[&(1, None, K16, Some(CubeDimMode::PerformanceCores))]
        );

        // M3 Max-like device: 1.25x on the performance cores; the class mean is about 1.12x.
        let mut sweeps = split_cores("m3", KernelKind::Reduce, &sizes(&[(1, 1.0)]), 0.8);
        sweeps.extend(grace.clone());
        let d = decide(&data(sweeps));
        let c = entry(&d, CPU, KernelKind::Reduce, 8);
        assert_eq!(c.cpu_units, Some(CpuUnits::Performance), "{}", c.reason);
        assert!(
            c.reason.contains("1.12x / 1.12x over all threads"),
            "{}",
            c.reason
        );
        // At 1.05x on the split device the class mean is within 3%: all threads.
        let mut sweeps = split_cores("m3", KernelKind::Reduce, &sizes(&[(1, 1.0)]), 1.0 / 1.05);
        sweeps.extend(grace);
        let c = entry(&decide(&data(sweeps)), CPU, KernelKind::Reduce, 8).clone();
        assert_eq!(c.cpu_units, Some(CpuUnits::All), "{}", c.reason);
    }

    mod prop {
        use proptest::prelude::*;

        use super::*;

        /// Per configuration a time at 10^6, and noise within 1% at 10^7: some configuration is
        /// then always within 3% of the fastest at both sizes.
        fn times() -> impl Strategy<Value = Vec<(f64, f64)>> {
            proptest::collection::vec((0.2f64..1.0, -0.01f64..0.01), 5 * 3)
        }

        /// Independent check of the acceptance inequalities for one (class, kind, type).
        fn check(
            data: &PerfData,
            d: &Decision,
            class: DeviceClass,
            kind: KernelKind,
            scalar: Scalar,
        ) {
            let c = d.choice(class, kind, scalar.bytes()).unwrap();
            let tile =
                (kind == KernelKind::GeomFactors).then(|| d.quad_tile(class, scalar.bytes()));
            let units = c.cpu_units.map(|u| match u {
                CpuUnits::All => CubeDimMode::Plan,
                CpuUnits::Performance => CubeDimMode::PerformanceCores,
            });
            let chosen = (
                c.vector_size,
                tile,
                c.cpu_block.map(|b| b * BLOCK_UNIT),
                units,
            );
            for size in RULE_SIZES {
                let t = class_times(data, class, kind, scalar, size);
                let s1 = best(&t, |v| v == 1).unwrap();
                if c.vector_size > 1 {
                    assert!(
                        s1 / best(&t, |v| v == c.vector_size).unwrap() >= ADOPT,
                        "{c:?}"
                    );
                }
                let admitted = |v: usize| {
                    v == 1
                        || RULE_SIZES.iter().all(|&n| {
                            speedup(&class_times(data, class, kind, scalar, n), v)
                                .is_some_and(|s| s >= ADOPT)
                        })
                };
                let fastest = best(&t, admitted).unwrap();
                assert!(t[&chosen].time_s <= WITHIN * fastest, "{c:?} at {size}");
            }
        }

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(64))]

            #[test]
            fn decisions_satisfy_both_inequalities(
                a in times(),
                a_perf in times(),
                b in times(),
                tiles in times(),
            ) {
                let configs: Vec<Spec> = [1usize, 2, 4, 8, 16]
                    .into_iter()
                    .flat_map(|v| [4usize, 16, 64].into_iter().map(move |k| (v, None, Some(k << 10))))
                    .collect();
                let mut sweeps = Vec::new();
                for (target, noise) in [("a", &a), ("b", &b)] {
                    for (i, &c) in configs.iter().enumerate() {
                        let (t, e) = noise[i];
                        for (n, scale) in [(RULE_SIZES[0], 1.0), (RULE_SIZES[1], 10.0 * (1.0 + e))] {
                            sweeps.push(rec(CPU, target, KernelKind::Fill, Scalar::F64, n, c, t * scale));
                            // Scalar kernels: size 1 only.
                            if c.0 == 1 {
                                sweeps.push(rec(CPU, target, KernelKind::Scalar, Scalar::U32, n, c, t * scale));
                            }
                        }
                    }
                }
                // GeomFactors on one GPU: tiles 1, 4, 16 at sizes up to the tile.
                for (i, tile) in [1usize, 4, 16].into_iter().enumerate() {
                    for v in [1usize, 2, 4, 8, 16].into_iter().filter(|&v| v <= tile) {
                        let (t, e) = tiles[i * 5 + v.ilog2() as usize];
                        for (n, scale) in [(RULE_SIZES[0], 1.0), (RULE_SIZES[1], 10.0 * (1.0 + e))] {
                            sweeps.push(rec(DeviceClass::Gpu, "g", KernelKind::GeomFactors, Scalar::F64, n, (v, Some(tile), None), t * scale));
                        }
                    }
                }
                // Device a also has 12 performance cores, measured with their own times; b has no
                // count.
                let perf: Vec<SweepRecord> = sweeps
                    .iter()
                    .filter(|r| r.target == "a")
                    .map(|r| {
                        let i = configs
                            .iter()
                            .position(|c| *c == (r.vector_size, r.tile, r.block_bytes))
                            .unwrap();
                        let (t, e) = a_perf[i];
                        let scale = if r.nominal_n == RULE_SIZES[0] { 1.0 } else { 10.0 * (1.0 + e) };
                        SweepRecord {
                            cube_dim: CubeDimMode::PerformanceCores,
                            median_s: t * scale,
                            ..r.clone()
                        }
                    })
                    .collect();
                sweeps.extend(perf);
                for r in sweeps.iter_mut().filter(|r| r.target == "a") {
                    r.cores.performance = Some(12);
                }
                let data = data(sweeps);
                let d = decide(&data);
                check(&data, &d, CPU, KernelKind::Fill, Scalar::F64);
                check(&data, &d, CPU, KernelKind::Scalar, Scalar::U32);
                check(&data, &d, DeviceClass::Gpu, KernelKind::GeomFactors, Scalar::F64);
                for c in &d.choices {
                    prop_assert_eq!(c.fit, Fit::Rule, "{:?}", c);
                }
            }
        }
    }
}
