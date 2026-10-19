//! E0 acceptance tests of env-check (docs/plans/E0.md, "Acceptance coverage").
//!
//! The tests run the env-check binary, so they check the exit status and the JSON
//! report exactly as the remote-test runner sees them.

#![cfg(any(feature = "cpu", feature = "metal", feature = "cuda"))]
// The device helpers serve the CPU and Metal tests; CUDA is checked by scripts/hpc/acceptance.sh.
#![cfg_attr(not(any(feature = "cpu", feature = "metal")), allow(dead_code))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use env_check::report::{CheckStatus, DeviceReport, KernelCheck, Report};

const BIN: &str = env!("CARGO_BIN_EXE_env-check");

/// A fresh directory for the reports of one test.
fn out_dir(test: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(test);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the test output directory");
    dir
}

fn describe(output: &Output) -> String {
    format!(
        "status {:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

/// Runs env-check with `args`, writing the report to a file; returns the output and the report.
fn run_env_check(test: &str, args: &[&str]) -> (Output, Report) {
    let report_path = out_dir(test).join("report.json");
    let output = Command::new(BIN)
        .args(args)
        .arg("--out")
        .arg(&report_path)
        .output()
        .expect("start env-check");
    let text = std::fs::read_to_string(&report_path).unwrap_or_else(|e| {
        panic!(
            "no report at {}: {e}\n{}",
            report_path.display(),
            describe(&output)
        )
    });
    let report = Report::from_json(&text)
        .unwrap_or_else(|e| panic!("invalid report: {e}\n{text}\n{}", describe(&output)));
    (output, report)
}

fn devices<'a>(report: &'a Report, runtime: &str) -> Vec<&'a DeviceReport> {
    report
        .devices
        .iter()
        .filter(|d| d.runtime == runtime)
        .collect()
}

fn checks<'a>(device: &'a DeviceReport, kernel: &str, precision: &str) -> Vec<&'a KernelCheck> {
    device
        .checks
        .iter()
        .filter(|c| c.kernel == kernel && c.precision == precision)
        .collect()
}

/// The element-wise add ran at vector size 1 and at the widest IO-optimized size,
/// and every run has `status`.
fn assert_add_runs(
    device: &DeviceReport,
    precision: &str,
    io_sizes: &[usize],
    status: CheckStatus,
) {
    let runs = checks(device, "elementwise_add", precision);
    let sizes: Vec<usize> = runs.iter().map(|c| c.vector_size).collect();
    let widest = io_sizes.first().copied().unwrap_or(1);
    assert!(
        sizes.contains(&1),
        "{precision} add at vector size 1 missing: {runs:#?}"
    );
    assert!(
        sizes.contains(&widest),
        "{precision} add at the widest size {widest} missing: {runs:#?}"
    );
    for run in &runs {
        assert_eq!(run.status, status, "{precision} add: {run:#?}");
    }
}

fn assert_histogram_passes(device: &DeviceReport) {
    let runs = checks(device, "atomic_histogram", "u32");
    assert_eq!(runs.len(), 1, "one u32 histogram run: {:#?}", device.checks);
    assert_eq!(runs[0].status, CheckStatus::Pass, "{:#?}", runs[0]);
}

/// Properties that every runtime must report with sensible values.
fn assert_properties(device: &DeviceReport) {
    assert!(!device.name.is_empty(), "device name");
    assert!(device.max_page_size > 0, "max_page_size");
    assert!(
        device.max_bindings >= 6,
        "max_bindings {}",
        device.max_bindings
    );
    assert!(
        device.max_cube_count.iter().all(|&c| c > 0),
        "max_cube_count"
    );
    assert!(device.plane_size_min >= 1 && device.plane_size_min <= device.plane_size_max);
    assert!(device.load_width > 0, "load_width");
    assert!(device.max_units_per_cube > 0, "max_units_per_cube");
    assert!(
        !device.io_vector_sizes_f32.is_empty(),
        "io_optimized_vector_sizes(4)"
    );
    assert!(
        !device.io_vector_sizes_f64.is_empty(),
        "io_optimized_vector_sizes(8)"
    );
    assert!(device.atomic_u32_add, "u32 atomic add");
    assert!(device.supports_u64, "u64");
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_runtime_passes_f32_f64() {
    let (output, report) = run_env_check("cpu_runtime_passes_f32_f64", &["--runtime", "cpu"]);
    assert!(output.status.success(), "{}", describe(&output));
    assert!(report.passed, "{report:#?}");

    let cpu = devices(&report, "cpu");
    assert_eq!(cpu.len(), 1, "one CPU device: {report:#?}");
    let device = cpu[0];
    assert_properties(device);
    assert!(device.supports_f64, "the CPU runtime supports f64");
    assert_add_runs(
        device,
        "f32",
        &device.io_vector_sizes_f32,
        CheckStatus::Pass,
    );
    assert_add_runs(
        device,
        "f64",
        &device.io_vector_sizes_f64,
        CheckStatus::Pass,
    );
    assert_histogram_passes(device);
    for check in &device.checks {
        assert_eq!(
            check.elements,
            env_check::kernels::DEFAULT_ELEMENTS,
            "{check:#?}"
        );
    }
    assert!(device.passed);
}

#[cfg(feature = "metal")]
#[test]
fn metal_passes_f32_reports_f64_unsupported() {
    let (output, report) = run_env_check(
        "metal_passes_f32_reports_f64_unsupported",
        &["--runtime", "metal"],
    );
    assert!(output.status.success(), "{}", describe(&output));
    assert!(report.passed, "{report:#?}");

    let metal = devices(&report, "metal");
    assert_eq!(metal.len(), 1, "one Metal device: {report:#?}");
    let device = metal[0];
    assert_properties(device);
    assert!(!device.supports_f64, "Metal has no f64");
    assert_add_runs(
        device,
        "f32",
        &device.io_vector_sizes_f32,
        CheckStatus::Pass,
    );
    let f64_runs = checks(device, "elementwise_add", "f64");
    assert!(
        !f64_runs.is_empty(),
        "f64 add recorded: {:#?}",
        device.checks
    );
    for run in f64_runs {
        assert_eq!(run.status, CheckStatus::Unsupported, "{run:#?}");
    }
    assert_histogram_passes(device);
    assert!(device.passed);
}

#[test]
fn report_is_valid_json() {
    // Default runtime selection and default output (stdout).
    let output = Command::new(BIN)
        .args(["--elements", "4096"])
        .output()
        .expect("start env-check");
    assert!(output.status.success(), "{}", describe(&output));
    let text = String::from_utf8(output.stdout.clone()).expect("UTF-8 report");
    let value: serde_json::Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("stdout is not JSON: {e}\n{}", describe(&output)));
    for key in [
        "schema_version",
        "tool_version",
        "host",
        "devices",
        "passed",
    ] {
        assert!(value.get(key).is_some(), "missing key {key}: {text}");
    }

    let report = Report::from_json(&text).expect("report parses");
    assert_eq!(report.schema_version, env_check::report::SCHEMA_VERSION);
    assert!(report.passed);
    assert!(report.errors.is_empty(), "{:?}", report.errors);
    assert!(!report.host.os.is_empty() && !report.host.arch.is_empty());
    assert!(
        report.host.rustc_version.starts_with("rustc 1.99"),
        "{:?}",
        report.host.rustc_version
    );
    assert!(report.host.hardware_threads > 0);

    let compiled: Vec<&str> = [
        (cfg!(feature = "cpu"), "cpu"),
        (cfg!(feature = "metal"), "metal"),
        (cfg!(feature = "cuda"), "cuda"),
    ]
    .into_iter()
    .filter_map(|(on, name)| on.then_some(name))
    .collect();
    for runtime in compiled {
        assert!(
            !devices(&report, runtime).is_empty(),
            "no {runtime} device: {text}"
        );
    }
    for device in &report.devices {
        assert!(
            device.checks.iter().all(|c| c.elements == 4096),
            "{device:#?}"
        );
    }
    // The report round-trips.
    assert_eq!(
        Report::from_json(&report.to_json()).expect("round trip"),
        report
    );
}

#[cfg(feature = "mpi")]
#[test]
fn mpi_allreduce_of_rank_ids() {
    let ranks: usize = std::env::var("GRIDCL_RANKS")
        .ok()
        .map(|r| r.parse().expect("GRIDCL_RANKS is a number"))
        .unwrap_or(4);
    let dir = out_dir("mpi_allreduce_of_rank_ids");
    let pattern = dir.join("rank-{rank}.json");

    let mut command = Command::new("mpirun");
    // Extra launcher flags, for example --oversubscribe for Open MPI on small hosts.
    if let Ok(flags) = std::env::var("GRIDCL_MPIRUN_FLAGS") {
        command.args(flags.split_whitespace());
    }
    let output = command
        .args(["-n", &ranks.to_string(), BIN, "--mpi-only", "--out"])
        .arg(&pattern)
        .output()
        .expect("start mpirun");
    assert!(output.status.success(), "{}", describe(&output));

    let expected_sum = (ranks * (ranks - 1) / 2) as i64;
    let mut used =
        std::collections::BTreeMap::<String, (usize, std::collections::BTreeSet<usize>)>::new();
    for rank in 0..ranks {
        let path = dir.join(format!("rank-{rank}.json"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("no report for rank {rank}: {e}\n{}", describe(&output)));
        let report = Report::from_json(&text).expect("rank report parses");
        assert!(report.passed, "rank {rank}: {report:#?}");
        let mpi = report.mpi.as_ref().expect("MPI record");
        assert_eq!(mpi.rank as usize, rank);
        assert_eq!(mpi.size as usize, ranks);
        assert_eq!(mpi.allreduce_expected, expected_sum);
        assert_eq!(mpi.allreduce_result, expected_sum);
        assert!(mpi.allreduce_passed);
        assert!(!mpi.library_version.is_empty());
        assert!(!mpi.cuda_aware_method.is_empty());
        assert!(!report.devices.is_empty(), "rank {rank} checked no device");
        for device in &report.devices {
            assert!(device.passed, "rank {rank}: {device:#?}");
            assert_eq!(
                device.index,
                rank % device.visible_devices,
                "rank {rank}: {device:#?}"
            );
            let entry = used.entry(device.runtime.clone()).or_default();
            entry.0 = device.visible_devices;
            entry.1.insert(device.index);
        }
    }
    // Every device a rank could be given was checked: with --gpus 2 --ranks 2 both GPUs.
    for (runtime, (visible, indices)) in used {
        assert_eq!(
            indices.len(),
            visible.min(ranks),
            "{runtime}: devices {indices:?} of {visible}"
        );
    }
}
