// Design: "Vectorization and CPU execution", "Benchmarks"; ADR-0015. The JSON records and the
// generated tables live in docs/perf/; candidates follow ADR-0057 and ADR-0060, the generic
// decision rule ADR-0058.
//! Kernel benchmarks across vector sizes, tile widths and CPU block sizes, end-to-end
//! throughput, and the decision rules that turn their results into tetraloma's vector and degree
//! policies.
//!
//! This crate lives in the repository and is not published. It is for the developers of tetraloma,
//! who run it to choose the vector sizes and launch settings that the library ships in
//! `vector-policy.toml` and `degree-policy.toml`. As a user you do not need it: the library applies
//! those policies itself, and the chapter [Compute devices] of the [user book] describes the
//! settings you can change.
//!
//! - [`protocol`]: warm-up, check and timed runs of one configuration;
//! - [`families`]: the workloads of each kernel family and the sweep over configurations;
//! - [`record`]: the JSON records of the benchmark runs and their loader;
//! - [`decide`]: the decision rule, which turns sweep records into `vector-policy.toml`;
//! - [`decide_degree`]: the degree sweep's decision rule, which writes `degree-policy.toml`;
//! - [`candidates`]: the candidate protocol, which compares a few candidate settings per
//!   platform;
//! - [`decide_generic`]: the generic decision rule of the four policy sets (CPU and GPU, f32
//!   and f64);
//! - [`report`]: the generated tables of the vectorisation and high-order performance
//!   reports.
//!
//! [Compute devices]: https://tetracl.github.io/tetraloma/devices.html#performance-settings
//! [user book]: https://tetracl.github.io/tetraloma/

pub mod candidates;
pub mod decide;
pub mod decide_degree;
pub mod decide_generic;
pub mod families;
pub mod protocol;
pub mod record;
pub mod report;
