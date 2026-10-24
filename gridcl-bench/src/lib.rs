//! Kernel benchmarks across vector sizes, tile widths and CPU block sizes, end-to-end
//! throughput, and JSON output for `docs/perf/` (docs/design.md, "Vectorization and CPU
//! execution / Benchmarks"; ADR-0015).
//!
//! - [`protocol`]: warm-up, check and timed runs of one configuration;
//! - [`families`]: the workloads of each kernel family and the sweep over configurations;
//! - [`record`]: the JSON records of `docs/perf/data/` and their loader;
//! - [`decide`]: the decision rule, which turns sweep records into `vector-policy.toml`;
//! - [`report`]: the generated tables of `docs/perf/vectorization.md`.

pub mod decide;
pub mod families;
pub mod protocol;
pub mod record;
pub mod report;
