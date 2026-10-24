//! Applies the decision rule (docs/plans/M3s.md, T12; ADR-0015) to the benchmark data: writes
//! `gridcl-device/vector-policy.toml` and the generated tables of `docs/perf/vectorization.md`.
//!
//! ```text
//! cargo run -p gridcl-bench --bin policy_report -- [--data DIR] [--policy FILE]
//!     [--report FILE] [--check]
//! ```
//!
//! - `--data`: the records, default `docs/perf/data/`;
//! - `--policy`: default `gridcl-device/vector-policy.toml`;
//! - `--report`: default `docs/perf/vectorization.md`. Only the part between the
//!   `BEGIN GENERATED` and `END GENERATED` markers changes; a file without markers gets them
//!   appended, and a missing file is created with a title.
//! - `--check`: writes nothing, and fails if either file differs from what it would write.
//!
//! Default paths are relative to the workspace root.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gridcl_bench::decide::decide;
use gridcl_bench::record::load_dir;
use gridcl_bench::report::{render, splice};

const USAGE: &str = "usage: policy_report [--data DIR] [--policy FILE] [--report FILE] [--check]";

/// The title of a report the binary creates.
const TITLE: &str = "# Vectorization study (M3s)\n";

fn main() -> ExitCode {
    match Args::parse(std::env::args().skip(1)).and_then(|args| run(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("policy_report: {message}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    data: PathBuf,
    policy: PathBuf,
    report: PathBuf,
    check: bool,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut parsed = Args {
            data: root.join("docs/perf/data"),
            policy: root.join("gridcl-device/vector-policy.toml"),
            report: root.join("docs/perf/vectorization.md"),
            check: false,
        };
        while let Some(flag) = args.next() {
            let mut value = || {
                args.next()
                    .map(PathBuf::from)
                    .ok_or_else(|| format!("{flag} needs a value\n{USAGE}"))
            };
            match flag.as_str() {
                "--data" => parsed.data = value()?,
                "--policy" => parsed.policy = value()?,
                "--report" => parsed.report = value()?,
                "--check" => parsed.check = true,
                "-h" | "--help" => return Err(USAGE.to_owned()),
                _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
            }
        }
        Ok(parsed)
    }
}

fn run(args: &Args) -> Result<(), String> {
    let data = load_dir(&args.data)?;
    let decision = decide(&data);
    let policy = decision.to_toml();
    let old_report = match std::fs::read_to_string(&args.report) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => TITLE.to_owned(),
        Err(e) => return Err(format!("{}: {e}", args.report.display())),
    };
    let report = splice(&old_report, &render(&data, &decision))
        .map_err(|e| format!("{}: {e}", args.report.display()))?;

    let mut stale = Vec::new();
    for (path, text) in [(&args.policy, &policy), (&args.report, &report)] {
        let current = std::fs::read_to_string(path).ok();
        if current.as_deref() == Some(text.as_str()) {
            continue;
        }
        if args.check {
            stale.push(path.display().to_string());
        } else {
            std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("policy_report: wrote {}", path.display());
        }
    }
    eprintln!(
        "policy_report: {} sweep and {} throughput records",
        data.sweeps.len(),
        data.throughput.len()
    );
    if stale.is_empty() {
        Ok(())
    } else {
        Err(format!("stale: {}", stale.join(", ")))
    }
}
