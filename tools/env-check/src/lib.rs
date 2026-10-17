//! env-check: probe of a gridcl development target.
//!
//! For each runtime compiled in (`cpu`, `metal`, `cuda`) env-check opens the devices,
//! records their properties, runs an element-wise add in f32 and f64 at vector size 1
//! and at the widest IO-optimized size, and a u32 atomic histogram, and writes a JSON
//! report. With the `mpi` feature and `--mpi-only` every rank checks the device
//! `rank % devices` and an allreduce of rank ids. See docs/design.md, milestone E0.

pub mod cuda_info;
pub mod kernels;
pub mod mpi_check;
pub mod probe;
pub mod report;

use std::panic::AssertUnwindSafe;
use std::path::PathBuf;

use kernels::Precision;
use probe::{OpenedDevice, RuntimeKind};
use report::{CheckStatus, DeviceReport, KernelCheck, Report};

/// Usage text of the command line.
pub const USAGE: &str = "usage: env-check [--runtime cpu|metal|cuda|all] [--elements N] \
[--out PATH] [--mpi-only] [--expect-sm NN] [--expect-cuda-major M]

  --runtime            runtime to check (default: all runtimes compiled in)
  --elements N         elements of the sample kernels (default: 10000000)
  --out PATH           write the JSON report to PATH instead of stdout;
                       {rank} in PATH is replaced by the MPI rank
  --mpi-only           run under MPI: allreduce of rank ids and the device rank % devices
  --expect-sm NN       fail unless every CUDA device has compute capability NN
  --expect-cuda-major M  fail unless the CUDA toolkit's major version is M

exit status 0 iff every check passes";

/// Which runtimes to check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeSelection {
    /// Every runtime compiled in.
    All,
    One(RuntimeKind),
}

/// Parsed command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    pub runtime: RuntimeSelection,
    pub elements: usize,
    pub out: Option<PathBuf>,
    pub mpi_only: bool,
    pub expect_sm: Option<u32>,
    pub expect_cuda_major: Option<u32>,
}

impl Options {
    /// Parses the arguments after the program name.
    ///
    /// Each option takes its value as the next argument or after `=` (`--elements=4096`).
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<Options, String> {
        let mut options = Options {
            runtime: RuntimeSelection::All,
            elements: kernels::DEFAULT_ELEMENTS,
            out: None,
            mpi_only: false,
            expect_sm: None,
            expect_cuda_major: None,
        };
        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            let (name, inline) = match arg.split_once('=') {
                Some((name, value)) if name.starts_with("--") => {
                    (name.to_string(), Some(value.to_string()))
                }
                _ => (arg.clone(), None),
            };
            let has_inline = inline.is_some();
            let mut value = |what: &str| -> Result<String, String> {
                match inline.clone() {
                    Some(value) => Ok(value),
                    None => args.next().ok_or_else(|| format!("{name} needs {what}")),
                }
            };
            match name.as_str() {
                "--runtime" => {
                    let v = value("a runtime")?;
                    options.runtime = match v.as_str() {
                        "all" => RuntimeSelection::All,
                        other => RuntimeSelection::One(
                            RuntimeKind::parse(other)
                                .ok_or_else(|| format!("unknown runtime `{other}`"))?,
                        ),
                    };
                }
                "--elements" => {
                    let v = value("a count")?;
                    options.elements = match v.parse::<usize>() {
                        Ok(n) if n > 0 => n,
                        _ => return Err(format!("--elements needs a positive count, not `{v}`")),
                    };
                }
                "--out" => options.out = Some(PathBuf::from(value("a path")?)),
                "--expect-sm" => options.expect_sm = Some(parse_u32(&name, &value("NN")?)?),
                "--expect-cuda-major" => {
                    options.expect_cuda_major = Some(parse_u32(&name, &value("M")?)?)
                }
                "--mpi-only" if !has_inline => options.mpi_only = true,
                _ => return Err(format!("unknown argument `{arg}`")),
            }
        }
        Ok(options)
    }
}

fn parse_u32(name: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{name} needs a number, not `{value}`"))
}

/// The runtimes `selection` names, with an error for each one this build lacks.
fn selected_runtimes(selection: RuntimeSelection, errors: &mut Vec<String>) -> Vec<RuntimeKind> {
    match selection {
        RuntimeSelection::All => {
            let compiled: Vec<RuntimeKind> = RuntimeKind::ALL
                .into_iter()
                .filter(|kind| kind.compiled_in())
                .collect();
            if compiled.is_empty() {
                errors.push(
                    "no runtime compiled in: build with --features cpu, metal or cuda".to_string(),
                );
            }
            compiled
        }
        RuntimeSelection::One(kind) if kind.compiled_in() => vec![kind],
        RuntimeSelection::One(kind) => {
            errors.push(format!(
                "{}: this build does not include the runtime (cargo feature `{}`)",
                kind.name(),
                kind.name()
            ));
            Vec::new()
        }
    }
}

/// The message of a caught panic.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with a non-string payload".to_string()
    }
}

/// Runs `f`, turning a panic (CubeCL panics on some device and compiler errors) into `Err`.
fn catch<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).map_err(|p| panic_message(p.as_ref()))
}

/// Checks the devices of `runtime` and appends their reports.
///
/// With `rank`, only the device `rank % visible devices` is checked (MPI, E0 T2);
/// otherwise every device the runtime opens.
fn check_runtime(
    runtime: RuntimeKind,
    options: &Options,
    rank: Option<usize>,
    report: &mut Report,
) {
    let opened = match catch(|| probe::open_devices(runtime)) {
        Ok(Ok(opened)) => opened,
        Ok(Err(error)) => return report.errors.push(error),
        Err(panic) => {
            return report.errors.push(format!(
                "{}: opening the devices panicked: {panic}",
                runtime.name()
            ));
        }
    };
    if opened.is_empty() {
        return report.errors.push(format!("{}: no device", runtime.name()));
    }
    let selected: Vec<&OpenedDevice> = match rank {
        Some(rank) => vec![&opened[rank % opened.len()]],
        None => opened.iter().collect(),
    };
    for device in selected {
        match catch(|| check_device(device, options.elements)) {
            Ok(device_report) => report.devices.push(device_report),
            Err(panic) => report.errors.push(format!(
                "{} device {}: panicked: {panic}",
                runtime.name(),
                device.index
            )),
        }
    }
    // TODO(E0 T2): for RuntimeKind::Cuda set report.cuda from cuda_info::collect(
    // CudaCompiler::of_build(), the fingerprints of the checked devices, options.expect_sm,
    // options.expect_cuda_major).
}

/// A check that did not run, with its outcome and the reason.
fn not_run(
    kernel: &str,
    precision: &str,
    vector_size: usize,
    elements: usize,
    status: CheckStatus,
    detail: String,
) -> KernelCheck {
    KernelCheck {
        kernel: kernel.to_string(),
        precision: precision.to_string(),
        vector_size,
        elements,
        status,
        millis: None,
        detail: Some(detail),
    }
}

/// Opens the client of one device, records its properties and runs every kernel check.
fn check_device(opened: &OpenedDevice, elements: usize) -> DeviceReport {
    const ADD: &str = "elementwise_add";
    const HISTOGRAM: &str = "atomic_histogram";

    let client = opened.device.client();
    let mut device = probe::device_report(opened, &client);
    let runtime = opened.runtime.name();
    // A panicking check fails on its own; the remaining checks still run.
    let checked = |kernel: &str, precision: &str, size: usize, f: &dyn Fn() -> KernelCheck| {
        catch(f).unwrap_or_else(|panic| {
            not_run(
                kernel,
                precision,
                size,
                elements,
                CheckStatus::Fail,
                format!("panicked: {panic}"),
            )
        })
    };

    for size in kernels::add_vector_sizes(&client, f32::BYTES) {
        device.checks.push(checked(ADD, f32::NAME, size, &|| {
            kernels::check_elementwise_add::<f32>(&client, elements, size)
        }));
    }
    for size in kernels::add_vector_sizes(&client, f64::BYTES) {
        let check = if device.supports_f64 {
            checked(ADD, f64::NAME, size, &|| {
                kernels::check_elementwise_add::<f64>(&client, elements, size)
            })
        } else if opened.runtime == RuntimeKind::Metal {
            // docs/design.md, "Supported backends": Metal has no f64.
            not_run(
                ADD,
                f64::NAME,
                size,
                elements,
                CheckStatus::Unsupported,
                "the device does not support f64 (Metal has no f64)".to_string(),
            )
        } else {
            not_run(
                ADD,
                f64::NAME,
                size,
                elements,
                CheckStatus::Fail,
                format!("the device does not support f64, which the design requires on {runtime}"),
            )
        };
        device.checks.push(check);
    }
    let histogram = if device.atomic_u32_add {
        checked(HISTOGRAM, "u32", 1, &|| {
            kernels::check_atomic_histogram(&client, elements, kernels::HISTOGRAM_BINS)
        })
    } else {
        not_run(
            HISTOGRAM,
            "u32",
            1,
            elements,
            CheckStatus::Fail,
            "the device does not register u32 atomic add".to_string(),
        )
    };
    device.checks.push(histogram);
    device.finish();
    device
}

/// Runs every check the options select and returns the finished report.
pub fn run(options: &Options) -> Report {
    let mut report = Report {
        schema_version: report::SCHEMA_VERSION,
        tool_version: env!("CARGO_PKG_VERSION").to_string(),
        host: probe::host_info(),
        devices: Vec::new(),
        mpi: None,
        cuda: None,
        errors: Vec::new(),
        passed: false,
    };
    if options.mpi_only {
        run_mpi(options, &mut report);
    } else {
        for runtime in selected_runtimes(options.runtime, &mut report.errors) {
            check_runtime(runtime, options, None, &mut report);
        }
    }
    report.finish();
    report
}

/// `--mpi-only`: the allreduce and the device `rank % devices` of each selected runtime.
#[cfg(feature = "mpi")]
fn run_mpi(options: &Options, report: &mut Report) {
    // TODO(E0 T2): initialise mpi_check::MpiSession, call check_runtime(runtime, options,
    // Some(rank), report) for each of selected_runtimes(..), and set report.mpi from
    // session.check(device index).
    let _ = options;
    report
        .errors
        .push("--mpi-only is not yet implemented (E0 T2)".to_string());
}

/// `--mpi-only` without the `mpi` feature is an error.
#[cfg(not(feature = "mpi"))]
fn run_mpi(options: &Options, report: &mut Report) {
    let _ = options;
    report
        .errors
        .push("--mpi-only needs a build with the `mpi` feature".to_string());
}

/// Where to write the report: `--out` with `{rank}` replaced by the MPI rank, if any.
///
/// Without an MPI record the path is returned unchanged.
pub fn output_path(options: &Options, report: &Report) -> Option<PathBuf> {
    let out = options.out.as_ref()?;
    match &report.mpi {
        Some(mpi) => Some(PathBuf::from(
            out.to_string_lossy()
                .replace("{rank}", &mpi.rank.to_string()),
        )),
        None => Some(out.clone()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use report::{CudaAware, MpiInfo};

    fn parse(args: &[&str]) -> Result<Options, String> {
        Options::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults() {
        let options = parse(&[]).expect("no arguments");
        assert_eq!(options.runtime, RuntimeSelection::All);
        assert_eq!(options.elements, kernels::DEFAULT_ELEMENTS);
        assert_eq!(options.out, None);
        assert!(!options.mpi_only);
        assert_eq!(options.expect_sm, None);
        assert_eq!(options.expect_cuda_major, None);
    }

    #[test]
    fn every_option() {
        let options = parse(&[
            "--runtime",
            "metal",
            "--elements=4096",
            "--out",
            "r-{rank}.json",
            "--mpi-only",
            "--expect-sm",
            "70",
            "--expect-cuda-major=12",
        ])
        .expect("valid arguments");
        assert_eq!(options.runtime, RuntimeSelection::One(RuntimeKind::Metal));
        assert_eq!(options.elements, 4096);
        assert_eq!(options.out, Some(PathBuf::from("r-{rank}.json")));
        assert!(options.mpi_only);
        assert_eq!(options.expect_sm, Some(70));
        assert_eq!(options.expect_cuda_major, Some(12));
        assert_eq!(
            parse(&["--runtime", "all"]).expect("all").runtime,
            RuntimeSelection::All
        );
    }

    #[test]
    fn invalid_arguments() {
        for args in [
            &["--runtime", "vulkan"][..],
            &["--runtime"],
            &["--elements", "0"],
            &["--elements", "ten"],
            &["--expect-sm", "sm_70"],
            &["--mpi-only=yes"],
            &["--verbose"],
            &["cpu"],
        ] {
            assert!(parse(args).is_err(), "{args:?} accepted");
        }
    }

    #[test]
    fn output_path_replaces_rank_only_under_mpi() {
        let options = parse(&["--out", "dir/rank-{rank}.json"]).expect("valid");
        let mut report = Report {
            schema_version: report::SCHEMA_VERSION,
            tool_version: String::new(),
            host: probe::host_info(),
            devices: Vec::new(),
            mpi: None,
            cuda: None,
            errors: Vec::new(),
            passed: true,
        };
        assert_eq!(
            output_path(&options, &report),
            Some(PathBuf::from("dir/rank-{rank}.json"))
        );
        assert_eq!(output_path(&parse(&[]).expect("valid"), &report), None);
        report.mpi = Some(MpiInfo {
            rank: 3,
            size: 4,
            library_version: String::new(),
            implementation: "unknown".to_string(),
            allreduce_result: 6,
            allreduce_expected: 6,
            allreduce_passed: true,
            device_index: None,
            cuda_aware: CudaAware::Unknown,
            cuda_aware_method: String::new(),
        });
        assert_eq!(
            output_path(&options, &report),
            Some(PathBuf::from("dir/rank-3.json"))
        );
    }

    #[test]
    fn missing_runtime_is_an_error() {
        let mut errors = Vec::new();
        for kind in RuntimeKind::ALL {
            let selected = selected_runtimes(RuntimeSelection::One(kind), &mut errors);
            assert_eq!(selected.is_empty(), !kind.compiled_in());
        }
        let missing = RuntimeKind::ALL.iter().filter(|k| !k.compiled_in()).count();
        assert_eq!(errors.len(), missing, "{errors:?}");
    }
}
