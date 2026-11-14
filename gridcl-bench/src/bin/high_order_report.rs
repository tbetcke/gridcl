//! Applies the degree sweep's decision rule (docs/plans/M9h.md, T11b; ADR-0042, item 12) to the
//! M9h records: writes `gridcl-geometry/degree-policy.toml` and the generated tables of
//! `docs/perf/high-order.md`.
//!
//! ```text
//! cargo run -p gridcl-bench --bin high_order_report -- [--data DIR] [--policy FILE]
//!     [--report FILE] [--check]
//! ```
//!
//! - `--data`: the `m9h-*.json` records, default `docs/perf/data/`;
//! - `--policy`: default `gridcl-geometry/degree-policy.toml`;
//! - `--report`: default `docs/perf/high-order.md`. Only the part between the
//!   `BEGIN GENERATED` and `END GENERATED` markers changes; a file without markers gets them
//!   appended, and a missing file is created with a title.
//! - `--check`: writes nothing, and fails if either file differs from what it would write.
//!
//! Entries the rule leaves unresolved (neither `direct` nor `loops` ran on every device of a
//! class) are listed on stderr, and the binary fails after writing.
//!
//! Default paths are relative to the workspace root.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gridcl_bench::decide_degree::decide_degree;
use gridcl_bench::record::load_high_order_dir;
use gridcl_bench::report::{render_high_order, splice};

const USAGE: &str =
    "usage: high_order_report [--data DIR] [--policy FILE] [--report FILE] [--check]";

/// The title of a report the binary creates.
const TITLE: &str = "# High-order geometry kernels on gridcl's devices (M9h)\n";

fn main() -> ExitCode {
    match Args::parse(std::env::args().skip(1)).and_then(|args| run(&args)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("high_order_report: {message}");
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
            policy: root.join("gridcl-geometry/degree-policy.toml"),
            report: root.join("docs/perf/high-order.md"),
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
    let data = load_high_order_dir(&args.data)?;
    let decision = decide_degree(&data);
    let policy = decision.to_toml();
    let old_report = match std::fs::read_to_string(&args.report) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => TITLE.to_owned(),
        Err(e) => return Err(format!("{}: {e}", args.report.display())),
    };
    let report = splice(&old_report, &render_high_order(&data, &decision))
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
            eprintln!("high_order_report: wrote {}", path.display());
        }
    }
    eprintln!(
        "high_order_report: {} baseline, {} degree, {} launch and {} CUDA resource records",
        data.baseline.len(),
        data.degree.len(),
        data.launch.len(),
        data.resources.len()
    );
    let unresolved: Vec<String> = decision
        .unresolved()
        .map(|(e, c)| {
            format!(
                "{:?} {} {} {} p={}: {}",
                e.class,
                e.family,
                e.cell,
                e.scalar.name(),
                c.degree,
                c.reason
            )
        })
        .collect();
    for line in &unresolved {
        eprintln!("high_order_report: {line}");
    }
    if !stale.is_empty() {
        Err(format!("stale: {}", stale.join(", ")))
    } else if !unresolved.is_empty() {
        Err(format!(
            "{} unresolved entries; they go to the maintainer",
            unresolved.len()
        ))
    } else {
        Ok(())
    }
}
