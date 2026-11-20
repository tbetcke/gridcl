//! Kernel benchmarks across vector sizes, tile widths and CPU block sizes, end-to-end
//! throughput, and JSON output for `docs/perf/` (docs/design.md, "Vectorization and CPU
//! execution / Benchmarks"; ADR-0015).
//!
//! - [`protocol`]: warm-up, check and timed runs of one configuration;
//! - [`families`]: the workloads of each kernel family and the sweep over configurations;
//! - [`record`]: the JSON records of `docs/perf/data/` and their loader;
//! - [`decide`]: the decision rule, which turns sweep records into `vector-policy.toml`;
//! - [`decide_degree`]: the degree sweep's decision rule, which writes `degree-policy.toml`;
//! - [`candidates`]: the candidate protocol of M10 (ADR-0057, ADR-0060);
//! - [`decide_generic`]: the generic decision rule of the four sets (ADR-0058);
//! - [`report`]: the generated tables of `docs/perf/vectorization.md` and
//!   `docs/perf/high-order.md`.

pub mod candidates;
pub mod decide;
pub mod decide_degree;
pub mod decide_generic;
pub mod families;
pub mod protocol;
pub mod record;
pub mod report;
