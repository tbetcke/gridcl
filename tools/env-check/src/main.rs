use std::process::ExitCode;

use env_check::report::{CheckStatus, Report};
use env_check::{Options, USAGE};

/// One line per device and check on stderr, so a run is readable without the JSON.
fn print_summary(report: &Report) {
    for device in &report.devices {
        eprintln!(
            "env-check: {} device {} ({}): {}",
            device.runtime,
            device.index,
            device.name,
            if device.passed { "pass" } else { "FAIL" }
        );
        for check in &device.checks {
            let status = match check.status {
                CheckStatus::Pass => "pass",
                CheckStatus::Fail => "FAIL",
                CheckStatus::Unsupported => "unsupported",
            };
            let millis = check
                .millis
                .map(|ms| format!(" {ms:.3} ms"))
                .unwrap_or_default();
            let detail = check
                .detail
                .as_ref()
                .map(|d| format!(": {d}"))
                .unwrap_or_default();
            eprintln!(
                "env-check:   {} {} x{}: {status}{millis}{detail}",
                check.kernel, check.precision, check.vector_size
            );
        }
    }
    for error in &report.errors {
        eprintln!("env-check: error: {error}");
    }
    eprintln!(
        "env-check: {}",
        if report.passed { "passed" } else { "FAILED" }
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let options = match Options::parse(args) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("env-check: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let report = env_check::run(&options);
    print_summary(&report);
    let json = report.to_json();
    match env_check::output_path(&options, &report) {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, json + "\n") {
                eprintln!("env-check: cannot write {}: {e}", path.display());
                return ExitCode::from(2);
            }
        }
        None => println!("{json}"),
    }
    if report.passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
