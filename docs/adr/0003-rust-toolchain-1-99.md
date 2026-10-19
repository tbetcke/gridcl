# ADR-0003: Pin the Rust toolchain at 1.99 instead of 1.95

- **Status:** Accepted
- **Date:** 2026-10-17
- **Raised in:** E0 T6 (maintainer's request)
- **Design sections affected:** "Implementation roadmap / E0 Development environment" (scope); "M0 Workspace and CI" (scope); "Risks and mitigations" (the Rust MSRV row); Listing H1 (CLAUDE.md, toolchain line)

## Context

The design pins Rust 1.95 in four places:

> E0 scope: `rust-toolchain.toml` with Rust 1.95 at the repository root, …
> M0 scope: … `rust-toolchain.toml` with Rust 1.95; …
> Risks: Rust 1.95 MSRV is declared only in CubeCL's workspace, not in its published manifests | Pin the toolchain in `rust-toolchain.toml`; CI images with Rust ≥ 1.95.
> Listing H1: Toolchain: stable Rust 1.95, pinned in rust-toolchain.toml.

E0 T0 created `rust-toolchain.toml` with `channel = "1.95"` (rustfmt, clippy). The remote runner installs the pinned toolchain on each login node from that file (`rustup toolchain install`), so the pin also selects the compiler on Myriad and Locust. 1.95 was the CubeCL 0.11 MSRV at design time, released 2026-04-16. The current stable is **1.99.1** (2026-10-13; 1.99.0 on 2026-10-01), four releases later.

No library crate exists yet (E0 has only the standalone `tools/env-check` and shell scripts), so this is the cheapest moment to move the pin.

## Problem

This is not a defect in the design. The maintainer asks to start the project on the current stable release instead of a six-month-old one. Evidence gathered on the M3 Max (2026-10-17, `cargo +1.99`, separate target directory):

| Check on Rust 1.99.1 | Result |
| --- | --- |
| `cargo check --all-targets`, env-check `--features cpu,metal,mpi` | passes (cubecl 0.11.0, mpi 0.8.2) |
| `cargo check --all-targets`, env-check `--features cuda,cuda-cpp,mpi` | passes |
| `cargo clippy --all-targets -- -D warnings`, both feature sets | no warnings |
| `cargo fmt --check` | clean |
| `cargo test --release --features cpu,mpi` | 28 unit tests pass; `report_is_valid_json` fails only on its assertion `rustc_version.starts_with("rustc 1.95")`, which checks the pin |
| `cargo check -p cubecl-llvm` without features (package in the lock file, not activated) | still panics in cargo's feature resolver, as on 1.95; the runner fix of E0 T6 is needed either way |

Not yet checked: Rust 1.99 on the Myriad login and compute nodes (RHEL 7.9, glibc 2.17) and on Locust. Rust's `*-linux-gnu` baseline has been glibc 2.17 and kernel 3.2 since Rust 1.64, and I found no announcement that raises it up to 1.99, but the runner run is the real check.

## Options

1. **Keep the design as written (1.95).** Nothing to change; E0 T6 continues on 1.95. The project starts on a compiler six months and four releases behind stable, and a later bump would touch many more crates, CI images and the vector-policy benchmarks (compiler changes can move SIMD codegen on the CPU runtime). Not proposed.
2. **Pin 1.99 now.** `rust-toolchain.toml` gets `channel = "1.99"` (resolving to the newest 1.99.x patch, today 1.99.1); the design, CLAUDE.md and the documents that name 1.95 follow. The design rows keep their meaning, only the number changes. Cost: one commit, the T0 acceptance assertion changes from `rustc 1.95` to `rustc 1.99`, the M3 Max clean build time and toolchain row of `docs/dev-environment.md` are re-recorded, and E0 T5's verification is re-run on 1.99.
3. **Track the latest stable (`channel = "stable"`).** Always current, but builds stop being reproducible (the compiler changes under a fixed commit, and remote targets may get a different release than the M3 Max), and the design's rule that toolchain changes are deliberate is lost. Not proposed.

## Proposed decision

Option 2: pin Rust 1.99 in `rust-toolchain.toml` and change every "1.95" that means the pinned toolchain to "1.99". Do it now, inside E0, before any library crate exists. Future bumps stay deliberate: each one needs an ADR or a dedicated task, with the full suite on the M3 Max and the V100.

Conditions before E0 continues on 1.99:

- the E0 T5 verification passes on 1.99 on the M3 Max (`scripts/check-local-tools.sh && cargo test --release --manifest-path tools/env-check/Cargo.toml --features cpu,metal,mpi`, run outside the sandbox for Metal);
- the E0 T6 smoke run (`remote-test.sh myriad-v100 -- cargo run --release … --features cuda`) installs and runs 1.99 on Myriad, which checks glibc 2.17.

## Consequences

- Text to change in docs/design.md:
    - "Implementation roadmap / E0 Development environment", scope: "Rust 1.95" → "Rust 1.99";
    - "M0 Workspace and CI", scope: "Rust 1.95" → "Rust 1.99";
    - "Risks and mitigations", the MSRV row: keep the CubeCL MSRV statement (1.95) as the reason for the pin, and change the mitigation to "Pin the toolchain (Rust 1.99) in `rust-toolchain.toml`; CI images with that toolchain.";
    - Listing H1: "Toolchain: stable Rust 1.99, pinned in rust-toolchain.toml."
- Outside docs/design.md, in the apply commit: `CLAUDE.md` (toolchain line); `rust-toolchain.toml` (`channel = "1.99"`); `IMPLEMENTATION.md` and `docs/dev-environment.md` (rustup rows "installs Rust 1.99"); `tools/env-check/tests/acceptance.rs` (`report_is_valid_json` expects `rustc 1.99`); the fake `cargo` and `rustc` of `scripts/hpc/tests/fake-bin/remote/` (version strings); docs/plans/E0.md (scope summary, T0 text and open question 11 read 1.99, plus a Deviations-log row naming this ADR).
- Public API changes: none.
- Acceptance tests affected: `env-check::report_is_valid_json` (the version assertion). The E0 M3 Max record (toolchain row, clean build time) is re-measured on 1.99.
- Later milestones affected: M0 creates the workspace and CI images with 1.99 instead of 1.95. The M3s vector-size benchmarks run on 1.99; no data exists yet, so nothing is invalidated.

## Decision record

Accepted by the maintainer on 2026-10-17 (option 2), with the conditions under "Proposed decision".
