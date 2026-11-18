//! Applies the degree sweep's decision rules to the records: M9h's rule (docs/plans/M9h.md,
//! T11b; ADR-0042, item 12) to the M9h records, for the frozen M9h policy and the M9h tables of
//! `docs/perf/high-order.md`; and the generic rule (ADR-0058 item 4, ADR-0060 item 3) to M10's
//! records at p = 1, 4 and 8 with M9h's in between, for `gridcl-geometry/degree-policy.toml`
//! and the M10 tables.
//!
//! ```text
//! cargo run -p gridcl-bench --bin high_order_report -- [--data DIR] [--policy FILE]
//!     [--degree-policy FILE] [--report FILE] [--check]
//! ```
//!
//! - `--data`: the `m9h-*.json` and `m10-degree-*.json` records, default `docs/perf/data/`;
//! - `--policy`: default `gridcl-bench/history/degree-policy-pre-m10.toml`, the M9h policy
//!   frozen before M10 (ADR-0060 item 2);
//! - `--degree-policy`: default `gridcl-geometry/degree-policy.toml`, M10's generic sets;
//! - `--report`: default `docs/perf/high-order.md`. Only the parts between the
//!   `BEGIN GENERATED` and `END GENERATED` markers (M9h; appended to a file without them, and a
//!   missing file is created with a title) and between the M10 markers (which must exist)
//!   change.
//! - `--check`: writes nothing, and fails if any file differs from what it would write.
//!
//! Entries M9h's rule leaves unresolved (neither `direct` nor `loops` ran on every device of a
//! class) are listed on stderr, and the binary fails after writing. The generic rule's
//! fallbacks, defaults and re-measurements are listed on stderr; they do not fail the run.
//!
//! Default paths are relative to the workspace root.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gridcl_bench::decide_degree::decide_degree;
use gridcl_bench::decide_generic::{self, DegreeSource, GenericFit};
use gridcl_bench::record::{load_high_order_dir, load_m10_dir};
use gridcl_bench::report::{render_high_order, render_m10_high_order, splice, splice_m10};

const USAGE: &str = "usage: high_order_report [--data DIR] [--policy FILE] \
    [--degree-policy FILE] [--report FILE] [--check]";

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
    degree_policy: PathBuf,
    report: PathBuf,
    check: bool,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut parsed = Args {
            data: root.join("docs/perf/data"),
            policy: root.join("gridcl-bench/history/degree-policy-pre-m10.toml"),
            degree_policy: root.join("gridcl-geometry/degree-policy.toml"),
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
                "--degree-policy" => parsed.degree_policy = value()?,
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
    let m10 = load_m10_dir(&args.data)?;
    let generic = decide_generic::decide_degree(&data, &m10, decide_generic::GenericRule::M10);
    let degree_policy = generic.to_toml();
    let report = splice_m10(&report, &render_m10_high_order(&m10, &generic))
        .map_err(|e| format!("{}: {e}", args.report.display()))?;

    let mut stale = Vec::new();
    for (path, text) in [
        (&args.policy, &policy),
        (&args.degree_policy, &degree_policy),
        (&args.report, &report),
    ] {
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
    eprintln!("high_order_report: {} M10 degree records", m10.degree.len());
    for e in &generic.entries {
        let at = format!("{:?} {} {} {}", e.class, e.family, e.cell, e.scalar.name());
        for k in e.picks.iter().filter(|k| k.source == DegreeSource::M10) {
            match &k.fit {
                GenericFit::Fallback { worst, .. } => eprintln!(
                    "high_order_report: M10 fallback {at} p={}: {} (worst r {worst:.2})",
                    k.degree, k.strategy
                ),
                GenericFit::Default => eprintln!(
                    "high_order_report: M10 default {at} p={}: {}",
                    k.degree, k.reason
                ),
                GenericFit::Bound => {}
            }
        }
        if !e.remeasure.is_empty() {
            eprintln!(
                "high_order_report: M10 re-measure {at} at p = {:?}",
                e.remeasure
            );
        }
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
