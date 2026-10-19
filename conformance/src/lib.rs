//! gridcl-conformance: compares gridcl-refcell with ndelement at the pinned commit.
//!
//! gridcl writes its own reference-cell definitions and follows the DefElement convention;
//! ndelement only checks them (docs/design.md, A.10). This crate is outside the gridcl
//! workspace, has its own Cargo.lock and is never published. It is the only crate that depends
//! on ndelement and on BLAS/LAPACK: Accelerate on macOS, OpenBLAS on Linux.
//!
//! # The ndelement pin
//!
//! - ndelement is a git dependency at a fixed `rev`. Never add `version =` to it, and never add
//!   ndelement to a library crate of the workspace.
//! - After a pin bump, run `cargo test --manifest-path conformance/Cargo.toml` and re-export the
//!   fixtures with `cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures`
//!   (from M1b), then commit `gridcl-refcell/tests/fixtures/` with the new pin.
//! - A failure after a bump means the convention moved. It calls for an explicit decision:
//!   follow the change (update gridcl-refcell and the fixtures) or keep the old pin.

// ndelement's own BLAS/LAPACK crates are dev-dependencies only, so the backends are linked here.
extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
