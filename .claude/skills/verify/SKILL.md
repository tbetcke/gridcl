---
name: verify
description: Run gridcl's pre-commit checks - formatting, clippy and the tests of the crates touched by the pending change. Use before every commit of code changes.
allowed-tools: Bash(cargo fmt *) Bash(cargo clippy *) Bash(cargo test *) Bash(git diff *) Bash(git status *) Bash(bash -n *) Bash(shellcheck *)
---

# Pre-commit checks

1. Before milestone M0 there is no `Cargo.toml` at the repository root. Then check only what E0
   adds: `tools/env-check/` with the commands of step 3 and `--manifest-path tools/env-check/Cargo.toml`,
   and the shell scripts as in step 3.5. If neither changed, report "nothing to verify yet" and stop.
2. Find the touched crates: list the changed paths with `git status --porcelain` and
   `git diff --name-only HEAD`, and map each path to the workspace crate that contains it.
   Changes only under `docs/`, `.claude/` or to `*.md` files need no checks.
3. Run, stopping at the first failure:
    1. `cargo fmt --all -- --check`. If it fails, run `cargo fmt --all` and include the result in
       the commit.
    2. `cargo clippy -p <crate> --all-targets -- -D warnings` for each touched crate.
    3. `cargo test -p <crate>` for each touched crate, on the default CPU runtime.
    4. If `conformance/` or `gridcl-refcell` changed: `cargo test --manifest-path conformance/Cargo.toml`
       when BLAS/LAPACK is available here; otherwise say it was skipped.
    5. If shell scripts under `scripts/` changed: `bash -n <script>` for each, and `shellcheck <script>`
       when shellcheck is installed.
    6. `tools/env-check/` is outside the workspace: if it changed, run the commands above with
       `--manifest-path tools/env-check/Cargo.toml`.
4. Report in at most ten lines: each command and its result, plus the first error if one failed.
   Do not commit when a check fails; fix it first, or explain why it cannot be fixed in this task.

GPU backends, MPI, remote targets and the full workspace run are not part of this check; they
run through the test-runner subagent and the hpc-test skill at the end of a task that changed
kernels, and at the end of a milestone.
