//! The measurement protocol of "Benchmarks" (docs/plans/M3s.md, T8; ADR-0018): one warm-up
//! launch per configuration, checked against a host reference, then one untimed calibration
//! call that fixes the calls per run, k = ⌈20 ms / t_cal⌉ (at most 10,000), and the median of
//! ten timed runs of k calls each, as time per call. Every call ends with `client.sync()`.
//!
//! A configuration is applied with the scoped overrides of `gridcl_device::testing`
//! (ADR-0015): `with_kind_vector_size` for the family's vector size, `with_cpu_block_bytes` for
//! the CPU block, `with_quad_tile` for the QuadField tile, and `with_cpu_units` with
//! `with_performance_cores` for the CPU units (ADR-0017). The size that actually ran comes from
//! `record_plans`.

use std::any::Any;
use std::process::Command;
use std::time::{Duration, Instant};

use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use cubecl::prelude::Client;
use gridcl_device::testing::{
    record_plans, with_cpu_block_bytes, with_cpu_units, with_kind_vector_size,
    with_performance_cores, with_quad_tile,
};
use gridcl_device::{CpuUnits, DeviceClass, KernelKind, VectorSizeOverride};

use crate::record::{Cores, CubeDimMode};

/// Timed runs per configuration.
pub const RUNS: usize = 10;
/// The shortest timed run: a run repeats the call until about this long (ADR-0018).
pub const MIN_RUN: Duration = Duration::from_millis(20);
/// The most calls per timed run.
pub const MAX_CALLS_PER_RUN: usize = 10_000;

/// Calls per timed run for a calibration call of `calibration`: ⌈[`MIN_RUN`] / calibration⌉,
/// at least 1 and at most [`MAX_CALLS_PER_RUN`] (ADR-0018). Computed in whole nanoseconds, so
/// a record's `calibration_s` gives back its `calls_per_run`.
pub fn calls_per_run(calibration: Duration) -> usize {
    let min_run = MIN_RUN.as_nanos();
    let t = calibration.as_nanos().max(1);
    usize::try_from(min_run.div_ceil(t))
        .unwrap_or(MAX_CALLS_PER_RUN)
        .clamp(1, MAX_CALLS_PER_RUN)
}

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
    /// The untimed calibration call that fixed `calls_per_run`.
    pub calibration_s: f64,
    /// Calls per timed run (ADR-0018).
    pub calls_per_run: usize,
    /// Every timed run, as time per call.
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

/// Runs `f` under `config`'s overrides for `kind`. `CubeDimMode::Plan` forces all hardware
/// threads and `CubeDimMode::PerformanceCores` the performance cores, `performance_cores` of
/// them, whatever the policy says (ADR-0017); both have no effect on GPUs.
///
/// # Errors
///
/// If `config` asks for the performance-core `cube_dim` without a count.
pub fn with_config<R>(
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
        CubeDimMode::Plan => Ok(with_cpu_units(CpuUnits::All, f)),
        CubeDimMode::PerformanceCores => {
            let cores = performance_cores
                .ok_or("cube_dim = performance cores needs the performance-core count")?;
            Ok(with_performance_cores(Some(cores), || {
                with_cpu_units(CpuUnits::Performance, f)
            }))
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
/// with the host reference, an untimed calibration `run` that fixes the calls per run
/// ([`calls_per_run`]), then `runs` timed runs of that many `run`s each, recorded as time per
/// call (ADR-0018). Every `run` ends with `client.sync()` and follows an untimed `prepare` and
/// sync. Only the plans of `kind` made during the warm-up give the size that ran.
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

    let calibration = timed_call(client, work)?;
    let calls = calls_per_run(calibration);
    let mut times_s = Vec::with_capacity(runs);
    for _ in 0..runs {
        let mut run = Duration::ZERO;
        for _ in 0..calls {
            run += timed_call(client, work)?;
        }
        times_s.push(run.as_secs_f64() / calls as f64);
    }
    Ok(Measured {
        vector_size_ran,
        units,
        warmup_s,
        calibration_s: calibration.as_secs_f64(),
        calls_per_run: calls,
        times_s,
    })
}

/// One `run` after an untimed `prepare` and sync: the time of the `run` and its sync.
fn timed_call(client: &Client, work: &mut dyn Workload) -> Result<Duration, String> {
    work.prepare()?;
    sync(client)?;
    let start = Instant::now();
    let out = work.run()?;
    sync(client)?;
    let elapsed = start.elapsed();
    drop(out);
    Ok(elapsed)
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
    fn calls_per_run_fills_twenty_milliseconds_up_to_the_cap() {
        let ms = Duration::from_millis;
        assert_eq!(calls_per_run(ms(7)), 3);
        assert_eq!(calls_per_run(ms(4)), 5);
        assert_eq!(calls_per_run(Duration::from_micros(4001)), 5);
        assert_eq!(calls_per_run(Duration::from_micros(3999)), 6);
        // A call of 20 ms or more is timed alone.
        assert_eq!(calls_per_run(ms(20)), 1);
        assert_eq!(calls_per_run(ms(500)), 1);
        // The cap.
        assert_eq!(calls_per_run(Duration::from_micros(2)), MAX_CALLS_PER_RUN);
        assert_eq!(calls_per_run(Duration::from_nanos(1)), MAX_CALLS_PER_RUN);
        assert_eq!(calls_per_run(Duration::ZERO), MAX_CALLS_PER_RUN);
    }

    /// A fill of 1000 f32 that also sleeps `delays[i]` in its `i`-th call (warm-up 0,
    /// calibration 1), and counts its calls.
    struct Delayed {
        inner: Box<dyn Workload>,
        delays: Vec<Duration>,
        calls: usize,
    }

    impl Workload for Delayed {
        fn items(&self) -> usize {
            self.inner.items()
        }
        fn bytes(&self) -> Option<usize> {
            self.inner.bytes()
        }
        fn reset(&mut self) -> Result<(), String> {
            self.inner.reset()
        }
        fn prepare(&mut self) -> Result<(), String> {
            self.inner.prepare()
        }
        fn run(&mut self) -> Result<Box<dyn Any>, String> {
            let out = self.inner.run()?;
            if let Some(&delay) = self.delays.get(self.calls) {
                std::thread::sleep(delay);
            }
            self.calls += 1;
            Ok(out)
        }
        fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
            self.inner.check(out)
        }
    }

    fn measure_delayed(calibration: Duration) -> (Measured, usize) {
        let client = gridcl_testing::test_client();
        let inner =
            crate::families::workload(&client, KernelKind::Fill, crate::record::Scalar::F32, 1000)
                .unwrap();
        let mut work = Delayed {
            inner,
            delays: vec![Duration::ZERO, calibration],
            calls: 0,
        };
        let m = measure(&client, KernelKind::Fill, 3, &mut work).unwrap();
        (m, work.calls)
    }

    #[test]
    fn measure_takes_the_calls_per_run_from_the_calibration_call() {
        // The calibration call takes 8–10 ms, so k = 3, whatever the later calls take.
        let (m, calls) = measure_delayed(Duration::from_millis(8));
        assert_eq!(m.calls_per_run, 3, "{m:?}");
        assert!(m.calibration_s >= 0.008, "{m:?}");
        assert_eq!(calls_per_run(Duration::from_secs_f64(m.calibration_s)), 3);
        assert_eq!(
            calls,
            2 + 3 * 3,
            "warm-up, calibration and three runs of three calls"
        );
        assert_eq!(m.times_s.len(), 3);
        // Time per call: the undelayed fills take far less than the calibration call.
        assert!(m.median_s() < m.calibration_s, "{m:?}");
    }

    #[test]
    fn measure_times_calls_of_twenty_milliseconds_alone() {
        let (m, calls) = measure_delayed(Duration::from_millis(25));
        assert_eq!(m.calls_per_run, 1, "{m:?}");
        assert_eq!(calls, 2 + 3);
    }

    #[test]
    fn compile_time_is_warmup_minus_median_and_never_negative() {
        let m = |warmup_s| Measured {
            vector_size_ran: 1,
            units: 1,
            warmup_s,
            calibration_s: 1.0,
            calls_per_run: 1,
            times_s: vec![1.0, 3.0, 2.0],
        };
        assert_eq!(m(5.0).compile_s(), 3.0);
        assert_eq!(m(1.5).compile_s(), 0.0);
    }
}
