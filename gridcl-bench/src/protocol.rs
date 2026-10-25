//! The measurement protocol of "Benchmarks" (docs/plans/M3s.md, T8): one warm-up launch per
//! configuration, checked against a host reference, then the median of ten timed runs, each
//! ending with `client.sync()`.
//!
//! A configuration is applied with the scoped overrides of `gridcl_device::testing`
//! (ADR-0015): `with_kind_vector_size` for the family's vector size, `with_cpu_block_bytes` for
//! the CPU block, `with_quad_tile` for the QuadField tile, and `with_properties` for the
//! performance-core `cube_dim`. The size that actually ran comes from `record_plans`.

use std::any::Any;
use std::process::Command;
use std::time::Instant;

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::prelude::Client;
use gridcl_device::testing::{
    record_plans, with_cpu_block_bytes, with_kind_vector_size, with_properties, with_quad_tile,
};
use gridcl_device::{DeviceClass, KernelKind, VectorSizeOverride};

use crate::record::{Cores, CubeDimMode};

/// Timed runs per configuration.
pub const RUNS: usize = 10;

/// The median of `times`; the mean of the middle two for an even count.
///
/// # Panics
///
/// If `times` is empty.
pub fn median(times: &[f64]) -> f64 {
    assert!(!times.is_empty(), "median of no times");
    let mut sorted = times.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        0.5 * (sorted[mid - 1] + sorted[mid])
    }
}

/// Effective bandwidth of `bytes` moved in `median_s`, relative to a copy that moved
/// `copy_bytes` in `copy_median_s` on the same device at the same size.
pub fn bandwidth_ratio(bytes: usize, median_s: f64, copy_bytes: usize, copy_median_s: f64) -> f64 {
    (bytes as f64 / median_s) / (copy_bytes as f64 / copy_median_s)
}

/// The commit the binary was run from: `GRIDCL_BENCH_COMMIT`, else `GITHUB_SHA`, else
/// `git rev-parse HEAD` (with `-dirty` when the tree has changes), else `unknown`.
pub fn commit() -> String {
    for var in ["GRIDCL_BENCH_COMMIT", "GITHUB_SHA"] {
        if let Ok(value) = std::env::var(var)
            && !value.is_empty()
        {
            return value;
        }
    }
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    match git(&["rev-parse", "HEAD"]) {
        Some(head) => match git(&["status", "--porcelain", "--untracked-files=no"]) {
            Some(status) if status.is_empty() => head,
            _ => format!("{head}-dirty"),
        },
        None => "unknown".to_owned(),
    }
}

/// The backend as `GRIDCL_TEST_DEVICE` names it (`cpu` when unset), not the client's name
/// (`wgpu<msl>` on Metal).
pub fn backend() -> String {
    std::env::var(gridcl_testing::TEST_DEVICE_VAR)
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "cpu".to_owned())
}

/// `release` or `debug`.
pub fn profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// The machine's core counts. Performance and efficiency cores come from
/// `sysctl -n hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` where it answers (macOS).
pub fn cores() -> Cores {
    let sysctl = |name: &str| {
        Command::new("sysctl")
            .args(["-n", name])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .and_then(|out| String::from_utf8_lossy(&out.stdout).trim().parse().ok())
    };
    Cores {
        performance: sysctl("hw.perflevel0.physicalcpu"),
        efficiency: sysctl("hw.perflevel1.physicalcpu"),
        threads: std::thread::available_parallelism()
            .map_or(1, |n| u32::try_from(n.get()).unwrap_or(u32::MAX)),
    }
}

/// State of CubeCL's persistent kernel cache (`[compilation] cache` in `cubecl.toml`): `on` or
/// `off`. The default is off, so the warm-up compiles every kernel.
pub fn kernel_cache() -> &'static str {
    if CubeClRuntimeConfig::get().compilation.cache {
        "on"
    } else {
        "off"
    }
}

/// One measured configuration of a family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    pub vector_size: usize,
    /// CPU block target in bytes; `None` keeps the policy's (and on GPUs there is none).
    pub block_bytes: Option<usize>,
    /// QuadField tile width; `None` keeps the policy's.
    pub tile: Option<usize>,
    pub cube_dim: CubeDimMode,
}

/// What one configuration measured.
#[derive(Clone, Debug, PartialEq)]
pub struct Measured {
    /// The family's vector size in its plans; the widest if they differ.
    pub vector_size_ran: usize,
    /// Units per cube of the family's plans.
    pub units: u32,
    pub warmup_s: f64,
    pub times_s: Vec<f64>,
}

impl Measured {
    pub fn median_s(&self) -> f64 {
        median(&self.times_s)
    }

    /// Warm-up time minus the median: the compile time of the configuration's kernels.
    pub fn compile_s(&self) -> f64 {
        (self.warmup_s - self.median_s()).max(0.0)
    }
}

/// Runs `f` under `config`'s overrides for `kind`. `performance_cores` is the count that
/// `CubeDimMode::PerformanceCores` lowers `max_units_per_cube` and `max_cube_dim.0` to.
///
/// # Errors
///
/// If `config` asks for the performance-core `cube_dim` without a count.
pub fn with_config<R>(
    client: &Client,
    kind: KernelKind,
    config: &Config,
    performance_cores: Option<u32>,
    f: impl FnOnce() -> R,
) -> Result<R, String> {
    let f =
        || {
            with_kind_vector_size(kind, VectorSizeOverride::Fixed(config.vector_size), || {
                match config.tile {
                    Some(tile) => with_quad_tile(tile, f),
                    None => f(),
                }
            })
        };
    let f = || match config.block_bytes {
        Some(bytes) => with_cpu_block_bytes(bytes, f),
        None => f(),
    };
    match config.cube_dim {
        CubeDimMode::Plan => Ok(f()),
        CubeDimMode::PerformanceCores => {
            let cores = performance_cores
                .ok_or("cube_dim = performance cores needs the performance-core count")?;
            let mut props = client.properties().clone();
            props.hardware.max_units_per_cube = cores;
            props.hardware.max_cube_dim.0 = cores;
            Ok(with_properties(props, f))
        }
    }
}

/// One family's work at one size and scalar type, on inputs already on the device.
pub trait Workload {
    /// Items processed (elements, or cells for the mesh families).
    fn items(&self) -> usize;

    /// Bytes one call reads and writes, for the bandwidth ratio; `None` if the family has no
    /// meaningful byte count.
    fn bytes(&self) -> Option<usize>;

    /// Puts the outputs in a state the next check can tell from a correct result.
    fn reset(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// Untimed work before every call, warm-up included, such as a fresh copy of a cached
    /// input.
    fn prepare(&mut self) -> Result<(), String> {
        Ok(())
    }

    /// One timed call. Returns what must stay alive until `client.sync()`.
    fn run(&mut self) -> Result<Box<dyn Any>, String>;

    /// Compares the warm-up's output with the host reference.
    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String>;
}

/// The protocol for one configuration: `reset`, a warm-up `run` whose output `check` compares
/// with the host reference, then `runs` timed `run`s, each ending with `client.sync()`. Every
/// `run` follows an untimed `prepare` and sync. Only the plans of `kind` made during the
/// warm-up give the size that ran.
///
/// # Errors
///
/// The error of any call, a failed check, or a warm-up that made no plan of `kind`.
pub fn measure(
    client: &Client,
    kind: KernelKind,
    runs: usize,
    work: &mut dyn Workload,
) -> Result<Measured, String> {
    work.reset()?;
    work.prepare()?;
    sync(client)?;
    let start = Instant::now();
    let (out, plans) = record_plans(|| -> Result<Box<dyn Any>, String> {
        let out = work.run()?;
        sync(client)?;
        Ok(out)
    });
    let warmup_s = start.elapsed().as_secs_f64();
    work.check(out?)?;
    let plans: Vec<_> = plans.into_iter().filter(|(k, _)| *k == kind).collect();
    let first = plans
        .first()
        .ok_or_else(|| format!("the warm-up made no {kind:?} plan"))?;
    let units = first.1.cube_dim.num_elems();
    let vector_size_ran = plans.iter().map(|(_, p)| p.vector_size).max().unwrap_or(1);

    let mut times_s = Vec::with_capacity(runs);
    for _ in 0..runs {
        work.prepare()?;
        sync(client)?;
        let start = Instant::now();
        let out = work.run()?;
        sync(client)?;
        times_s.push(start.elapsed().as_secs_f64());
        drop(out);
    }
    Ok(Measured {
        vector_size_ran,
        units,
        warmup_s,
        times_s,
    })
}

/// Waits for every launch on `client`.
///
/// # Errors
///
/// The runtime's error, as text.
pub fn sync(client: &Client) -> Result<(), String> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("sync: {e:?}"))
}

/// Vector sizes the sweep requests: 1, 2, 4, ... up to `VECTOR_PAD` on every device. A device
/// with fewer lanes (Metal: 4) runs the wider requests at its widest size, which the records
/// show in `vector_size_ran`; so it contributes that time to its class at every size, as
/// ADR-0015 (item 5) requires when the class's devices are combined.
pub fn vector_sizes() -> Vec<usize> {
    (0..=gridcl_device::VECTOR_PAD.ilog2())
        .map(|k| 1 << k)
        .collect()
}

/// Whether `class` is a CPU class, where block sizes apply.
pub fn is_cpu(class: DeviceClass) -> bool {
    class != DeviceClass::Gpu
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_of_odd_and_even_counts() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[5.0]), 5.0);
    }

    #[test]
    #[should_panic(expected = "median of no times")]
    fn median_of_nothing_panics() {
        median(&[]);
    }

    #[test]
    fn bandwidth_ratio_compares_bytes_per_second() {
        // A copy moves 8 bytes per u32 item; a fill of f64 moves 8 bytes per item too.
        let n = 1000;
        assert_eq!(bandwidth_ratio(8 * n, 2.0, 8 * n, 2.0), 1.0);
        // Half the bytes in the same time: half the bandwidth.
        assert_eq!(bandwidth_ratio(4 * n, 2.0, 8 * n, 2.0), 0.5);
        // The same bytes in half the time: twice the bandwidth.
        assert_eq!(bandwidth_ratio(8 * n, 1.0, 8 * n, 2.0), 2.0);
    }

    #[test]
    fn compile_time_is_warmup_minus_median_and_never_negative() {
        let m = |warmup_s| Measured {
            vector_size_ran: 1,
            units: 1,
            warmup_s,
            times_s: vec![1.0, 3.0, 2.0],
        };
        assert_eq!(m(5.0).compile_s(), 3.0);
        assert_eq!(m(1.5).compile_s(), 0.0);
    }
}
