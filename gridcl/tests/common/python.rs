//! The Python oracles of the M5 tests (ADR-0024, item 11): interpreter discovery,
//! `GRIDCL_SKIP_PYTHON`, and runs of `scripts/oracles/vtk_oracle.py` with JSON on stdout.
//!
//! The interpreter is `GRIDCL_PYTHON`, else `<repo>/.venv/bin/python`. A missing interpreter, or
//! one without the pinned packages, fails the test with the install command, unless
//! `GRIDCL_SKIP_PYTHON=1`: then [`Oracle::find`] prints that the test skipped its oracle checks
//! and returns `None`.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// Names the interpreter.
pub const PYTHON_VAR: &str = "GRIDCL_PYTHON";

/// `1` skips every oracle check.
pub const SKIP_VAR: &str = "GRIDCL_SKIP_PYTHON";

/// How to install the pinned oracles.
pub const INSTALL: &str = "python3 -m venv .venv && .venv/bin/pip install vtk==9.7.1 meshio==5.3.5 (or set GRIDCL_PYTHON; \
     GRIDCL_SKIP_PYTHON=1 skips the oracle checks)";

/// The repository root.
pub fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("gridcl/ has a parent")
        .to_path_buf()
}

/// The oracle script and the interpreter that runs it.
#[derive(Clone, Debug)]
pub struct Oracle {
    python: PathBuf,
    script: PathBuf,
    test: String,
}

impl Oracle {
    /// The oracle of `test`, or `None` (after printing that `test` skips its oracle checks) if
    /// `GRIDCL_SKIP_PYTHON=1`.
    ///
    /// # Panics
    ///
    /// If the interpreter is missing and `GRIDCL_SKIP_PYTHON` is not `1`.
    pub fn find(test: &str) -> Option<Oracle> {
        if std::env::var(SKIP_VAR).is_ok_and(|v| v == "1") {
            eprintln!("{test}: {SKIP_VAR}=1: skipped the Python oracle checks");
            return None;
        }
        let python = std::env::var_os(PYTHON_VAR)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| repo().join(".venv/bin/python"));
        assert!(
            python.exists(),
            "{test}: no Python interpreter at {} ({PYTHON_VAR} or <repo>/.venv/bin/python); \
             install the oracles: {INSTALL}",
            python.display()
        );
        Some(Oracle {
            python,
            script: repo().join("scripts/oracles/vtk_oracle.py"),
            test: test.to_owned(),
        })
    }

    /// Runs `vtk_oracle.py <command> <args...>` and parses its stdout as JSON.
    ///
    /// # Panics
    ///
    /// If the script fails (for example because the pinned versions are not installed) or its
    /// output is not JSON; the message holds its stderr.
    pub fn run(&self, command: &str, args: &[OsString]) -> Value {
        let out = Command::new(&self.python)
            .arg(&self.script)
            .arg(command)
            .args(args)
            .output()
            .unwrap_or_else(|e| panic!("{}: run {}: {e}", self.test, self.python.display()));
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "{}: vtk_oracle.py {command} failed ({}):\n{stderr}\nInstall the pinned oracles: {INSTALL}",
            self.test,
            out.status
        );
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
            panic!(
                "{}: vtk_oracle.py {command}: output is not JSON ({e}):\n{}\n{stderr}",
                self.test,
                String::from_utf8_lossy(&out.stdout)
            )
        })
    }

    /// [`run`](Oracle::run) on paths.
    pub fn run_files(&self, command: &str, files: &[PathBuf]) -> Vec<Value> {
        let args: Vec<OsString> = files.iter().map(|p| p.clone().into_os_string()).collect();
        let out = self.run(command, &args);
        let list = out
            .as_array()
            .unwrap_or_else(|| panic!("{}: {command}: expected a JSON list", self.test))
            .clone();
        assert_eq!(
            list.len(),
            files.len(),
            "{}: {command}: one result per file",
            self.test
        );
        list
    }
}
