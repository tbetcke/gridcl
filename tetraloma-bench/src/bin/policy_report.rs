//! Writes the vector policy and the generated tables of `docs/perf/vectorization.md`:
//!
//! - the four generic sets of `tetraloma-device/vector-policy.toml` and M10c's tables, by the
//!   generic rule with M10c's fallback (ADR-0058 item 4, ADR-0065; `decide_generic`): `[Cpu]`
//!   from the `m10c-vector-cpu-*.json` grid records, `[Gpu]` from the `m10-vector-*.json`
//!   records;
//! - M10's tables, from the `m10-vector-*.json` records by M10's rule (history; M10's policy is
//!   frozen in `tetraloma-bench/history/vector-policy-m10.toml`);
//! - the frozen M3s policy and M3s tables, from the M3s records by the M3s decision rule
//!   (docs/plans/M3s.md, T12; ADR-0015, ADR-0060 item 2).
//!
//! ```text
//! cargo run -p tetraloma-bench --bin policy_report -- [--data DIR] [--vector-policy FILE]
//!     [--policy FILE] [--report FILE] [--check]
//! ```
//!
//! - `--data`: the records, default `docs/perf/data/`;
//! - `--vector-policy`: default `tetraloma-device/vector-policy.toml`, the generic sets;
//! - `--policy`: default `tetraloma-bench/history/vector-policy-pre-m10.toml`, the M3s policy
//!   frozen before M10;
//! - `--report`: default `docs/perf/vectorization.md`. Only the parts between the
//!   `BEGIN GENERATED` and `END GENERATED` markers and between the M10 and M10c markers change;
//!   a file without the M3s markers gets them appended, and a missing file is created with a
//!   title. The M10 and M10c markers must be present.
//! - `--check`: writes nothing, and fails if a file differs from what it would write.
//!
//! M3s entries the rule leaves unresolved (no admitted configuration within 10% of the
//! fastest, ADR-0018) are listed on stderr, and the binary fails after writing. Entries that
//! need the generic rule's fallback (no candidate within the bound) under M10c's rule are listed
//! on stderr; the report marks them with their worst case and cause.
//!
//! Default paths are relative to the workspace root.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use tetraloma_bench::decide::decide;
use tetraloma_bench::decide_generic::{GenericFit, GenericRule, decide_vector};
use tetraloma_bench::record::{load_dir, load_m10_dir};
use tetraloma_bench::report::{
    render, render_m10_vector, render_m10c_vector, splice, splice_m10, splice_m10c,
};

const USAGE: &str = "usage: policy_report [--data DIR] [--vector-policy FILE] [--policy FILE] \
                     [--report FILE] [--check]";

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
    vector_policy: PathBuf,
    policy: PathBuf,
    report: PathBuf,
    check: bool,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut parsed = Args {
            data: root.join("docs/perf/data"),
            vector_policy: root.join("tetraloma-device/vector-policy.toml"),
            policy: root.join("tetraloma-bench/history/vector-policy-pre-m10.toml"),
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
                "--vector-policy" => parsed.vector_policy = value()?,
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
    let m10 = load_m10_dir(&args.data)?;
    let history = decide_vector(&m10, GenericRule::M10);
    let report = splice_m10(&report, &render_m10_vector(&m10, &history))
        .map_err(|e| format!("{}: {e}", args.report.display()))?;
    let generic = decide_vector(&m10, GenericRule::M10c);
    let vector_policy = generic.to_toml();
    let report = splice_m10c(&report, &render_m10c_vector(&m10, &generic))
        .map_err(|e| format!("{}: {e}", args.report.display()))?;

    let mut stale = Vec::new();
    for (path, text) in [
        (&args.vector_policy, &vector_policy),
        (&args.policy, &policy),
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
            eprintln!("policy_report: wrote {}", path.display());
        }
    }
    eprintln!(
        "policy_report: {} sweep and {} throughput records (M3s), {} vector records (M10), {} \
         grid records (M10c)",
        data.sweeps.len(),
        data.throughput.len(),
        m10.vector.len(),
        m10.cpu_grid.len()
    );
    for e in &generic.entries {
        if let GenericFit::Fallback { worst, cause } = &e.fit {
            eprintln!(
                "policy_report: M10c {:?} {:?} {}-byte: fallback, worst case r = {worst:.2}; \
                 {cause}",
                e.class, e.kind, e.bytes
            );
        }
    }
    let unresolved: Vec<String> = decision
        .unresolved()
        .map(|c| {
            format!(
                "{:?} {:?} {}-byte: {}",
                c.class, c.kind, c.elem_bytes, c.reason
            )
        })
        .collect();
    for line in &unresolved {
        eprintln!("policy_report: {line}");
    }
    if !stale.is_empty() {
        Err(format!("stale: {}", stale.join(", ")))
    } else if !unresolved.is_empty() {
        Err(format!(
            "{} unresolved entries; they go to the maintainer (ADR-0018)",
            unresolved.len()
        ))
    } else {
        Ok(())
    }
}
