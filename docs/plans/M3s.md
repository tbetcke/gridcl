# M3s SIMD vectorization and benchmarks — plan

- **Status:** approved
- **Branch:** tbetcke/tbetcke-m3s-simd-vectorization-and-benchmarks (question 1)
- **Base commit:** 5882b8c8c8dc9b17ab7d635da43a6ef98bc8cc9a
- **Design sections:**
  - "Implementation roadmap / M3s SIMD vectorization and benchmarks", and M10 for the re-run;
  - "Vectorization and CPU execution": all of it (CubeCL's vector model, how the CPU runtime runs a kernel, Listing S2, the bullets after it, "Padding and tiles", "Where vectors apply", "Determinism across vector sizes", "Benchmarks");
  - "Topology construction on the device": "Entity creation" step 1 (key emission), and "Permutation info";
  - "Geometry": "Geometric factors", and "Coordinate element and storage" (E-vectors, tiled `QuadField`);
  - "Testing and validation" (vector-size invariance, tolerances);
  - "Development environment and test targets" (Kathleen, Myriad CPU D node as the fallback, Locust conditions, remote-test runner; ADR-0016);
  - "Workspace layout": crate table (gridcl-bench), "Cargo features", CI matrix (the local job's vector-size variants).
- **Depends on:** M3 (done, PR #6, merged in 1c03fda; plan `Status: done` in 5882b8c)
- **Accepted ADRs that apply:** ADR-0002 (amended by ADR-0005), ADR-0003, ADR-0004, ADR-0009, ADR-0010, ADR-0011, ADR-0012, ADR-0013, ADR-0014. ADR-0015 (accepted 2026-10-24, applied to docs/design.md). ADR-0016 (accepted 2026-10-24, applied): Kathleen for CPU-only runs and two-node MPI; Myriad D nodes as the fallback.

<!--
Rules for this file (do not delete):
- Status moves draft -> approved (maintainer) -> implemented (/implement-milestone) -> done (maintainer, after merge).
- Every task is one commit. T0 is always the acceptance tests.
- The plan references design sections; it never restates or changes requirements.
- Tick a task ([x]) in the same commit that completes it.
-->

## Scope summary

M3s fills in gridcl-bench with the protocol of "Vectorization and CPU execution / Benchmarks". It
measures every candidate family that exists after M3 (Fill, Copy, Reduce, ScanTile, SortDigits,
Gather, TopologyKeys, PermutationInfo, EVector and GeomFactors) at every vector size, QuadField
tile width and CPU block size. For TopologyKeys and PermutationInfo it adds the cell-fastest
connectivity copy behind the policy and measures it including the transpose. It applies the
decision rule per family, device class and scalar type, and writes `vector-policy.toml` and
`docs/perf/vectorization.md` from the data. Pack and PullBack do not exist until M6 and M9, so
they stay at 1 until the M10 re-run. The M3 review follow-ups that the maintainer scheduled for
M3s come first (M3 plan, "Follow-up schedule"): m9, m10, m11, m3, m2 and the small m8 items in
T1–T4, m8's kinds in T6, and m6 in T14.

## Planned public API and rules (ADR-0015, question 2)

This section follows M3's precedent: it fixes what ADR-0015 must decide before T0. No type of
Listing S2 changes shape except as stated in questions 2, 6 and 14.

- **gridcl-device, `testing` module** (feature `testing`):
  - `pub fn with_kind_vector_size<R>(kind: KernelKind, v: VectorSizeOverride, f: impl FnOnce() -> R) -> R`: a scoped override for one family. For that kind it takes precedence over `with_vector_size` and `GRIDCL_VECTOR_SIZE`.
  - `pub fn with_cpu_block_bytes<R>(bytes: usize, f: impl FnOnce() -> R) -> R`: replaces the CPU block target ("CPUs": about 16 KiB of the widest stream) for plans made inside `f`. The value must be a power of two from 4 KiB to 64 KiB; anything else panics.
  - `cube_dim` = performance-core count is measured with the existing `with_properties`, by lowering `max_units_per_cube` and `max_cube_dim.0`. No new hook (question 8).
- **gridcl-device, `VectorPolicy` and `vector-policy.toml`:**
  - Each CPU class gets `[<Class>.cpu_block]`, laid out like `vector_size`: one entry per `KernelKind` and element size, in units of 4 KiB (1, 4 or 16). Every value then stays a power of two from 1 to 16 (acceptance bullet 4).
  - `pub fn cpu_block_bytes(&self, kind: KernelKind, elem_bytes: usize) -> Option<usize>`, `None` on `Gpu`. `LaunchPlan::elementwise` reads it on CPUs instead of the private 16 KiB constant.
  - The values start at 4 (16 KiB), so T5 changes no launch.
- **gridcl-device, `quad_tile`** (question 14): `pub fn quad_tile(&self, elem_bytes: usize) -> usize`, with `quad_tile = { 4 = 16, 8 = 16 }` per class. The f32 and f64 factor kernels can then choose their tiles separately. `with_quad_tile` and `GRIDCL_QUAD_TILE` override both sizes.
- **gridcl-device, `KernelKind::Scalar`** (question 6): for kernels that are not on the candidate list and plan with `max_vector = 1`. Its `vector_size` entries are fixed at 1, which `parse` enforces; its block entries come from the policy like any other kind.
- **gridcl-topology, feature `bench`** (off by default; only gridcl-bench enables it; question 3):
  - `#[doc(hidden)] pub mod bench` with `emit_keys(topology: &Topology, dim: usize) -> Result<BenchOutput>` and `permutation_info(topology: &Topology, b: usize) -> Result<BenchOutput>`.
  - Each runs exactly the family's launch sequence, the cell-fastest copy included, on inputs already on the device. It returns an opaque owner of the outputs, so the caller syncs, times and drops it.
  - `Topology::new` is not used for timing, because it also uploads origin ids and builds an `IndexMap`.
  - gridcl-bench's crate-table row gains gridcl-topology (feature `bench`) and `toml`. Feature unification turns `bench` on in `--workspace` builds; the module is hidden and only adds functions.
- **Decision rule as applied** (question 4):
  - A configuration is (vector size, tile, block); on the M3 Max the performance-core `cube_dim` is reported but not part of it (question 8).
  - The speedup of size n is the best time over size-1 configurations divided by the best time over size-n configurations.
  - A class measured on several devices (CpuSimd128: the M3 Max and Grace; Gpu: Metal and H100) uses the geometric mean of the per-device time ratios. A device that cannot run size n (Metal above 4 lanes, or without f64) contributes its time at the size it clamps to, or nothing if it lacks the type.
  - The 3% rule picks among the configurations the 10% rule admits: size 1, plus every larger size that is at least 10% faster at 10^6 and 10^7 items. Read literally, the two acceptance bullets conflict when a larger size is 3–10% faster, because size 1 is then not within 3% of the fastest yet may not be replaced.
  - A 4-byte entry shared by f32 and u32 (Fill, Reduce, Gather) must pass for every 4-byte type measured, or it takes the smaller size.
  - CpuSimd256 copies the CpuSimd512 table ("Benchmarks", "Where").
  - Under `Policy` the register budget can clamp a size, so the TOML stores the size that `record_plans` shows actually ran under the chosen policy. Bullet 3 then checks what runs.
- **One timed region per family** (question 13): the smallest call whose launches all take their shape from the family's plan.
  - Fill, Copy, Reduce, Gather: the primitive.
  - ScanTile: `inclusive_scan` and `exclusive_scan`, which plan every phase with ScanTile (ADR-0009, "One partition per primitive").
  - SortDigits: `radix_sort`. All its phases use the SortDigits plan, except the `exclusive_scan` it calls, which stays at its own default through `with_kind_vector_size`.
  - TopologyKeys, PermutationInfo: the `bench` hooks.
  - EVector: `DeviceGrid::evector` on a fresh `try_clone`, the clone outside the timing.
  - GeomFactors: the block functions.
- **Cell-fastest connectivity** (question 5): private to gridcl-topology.
  - At N > 1, TopologyKeys and PermutationInfo build a transient cell-fastest copy, `[local vertex][padded cell]`, of the cell–vertex connectivity (and of the origin-id words for PermutationInfo), read it with vector loads, and drop it.
  - At N = 1 they read the cell-major connectivity, as they do now.
  - The transpose is an indirect copy, so it is scalar and plans as `KernelKind::Scalar`. Its time still counts towards the family, because the `bench` hooks time the whole sequence.

## Tasks

- [x] **T0 Acceptance tests (failing)**
  - Goal: every acceptance bullet of M3s as an automated test, plus stubs for the ADR-0015 items.
  - Crates/files:
    - `gridcl-bench/tests/perf_record.rs` (new);
    - `gridcl-bench/src/{lib.rs, record.rs, decide.rs, report.rs}` (stubs);
    - `gridcl-bench/Cargo.toml` (adds `toml`);
    - `gridcl-device/src/{policy.rs, testing.rs}` and `gridcl-device/vector-policy.toml`;
    - `gridcl-topology/{Cargo.toml, src/bench.rs}` (stubs);
    - `gridcl/tests/kernel_invariance.rs`.
  - Stubs (`todo!()` bodies):
    - `with_kind_vector_size`, `with_cpu_block_bytes`, `VectorPolicy::cpu_block_bytes`;
    - `bench::{emit_keys, permutation_info}`;
    - `record::{SweepRecord, ThroughputRecord, load_dir}`, `decide::{decide, Decision}`, `report::render`.
    - `KernelKind::Scalar` is added for real, with its `Scalar = { 4 = 1, 8 = 1 }` rows in every class, because `parse` requires every kind (policy.rs:325) and a missing row would break every test.
    - `quad_tile(elem_bytes)` keeps returning the single value for now, with every caller updated.
  - Tests:
    - `gridcl-bench/tests/perf_record.rs::benchmark_data_covers_required_targets`: `docs/perf/data/` holds M3s sweep JSON for the M3 Max (`cpu`, `metal`), Locust (`cpu`, `cuda`) and a Myriad D node (`cpu`, CpuSimd512). A Myriad V100 or A100 CUDA record counts only with a non-empty `fallback_reason` (question 7). Every Locust record carries both node-load snapshots, and both are `idle`.
    - `…::report_lists_every_family_class_and_type`: the generated tables of `docs/perf/vectorization.md` (between marker comments) equal `report::render(data, policy)`. `render` lists, for every candidate family, device class and scalar type, the speedup of each vector size over 1, the chosen size and the reason.
    - `…::policy_follows_the_decision_rule`: `gridcl-device/vector-policy.toml` equals `decide::decide(data)`. Independently of `decide`, every size above 1 is at least 10% faster than 1 at 10^6 and 10^7, and every chosen (vector size, tile, block) is within 3% of the fastest, as read in question 4.
    - `…::throughput_recorded_with_policy_and_scalar`: throughput JSON for the M3 Max CPU runtime and a Myriad D node, with `vector_size_override` `policy` and `scalar`, holding entities per second and geometric factors per second.
    - `gridcl-device/src/policy.rs::tests::policy_covers_every_kind_and_class` (unit test, because the list of all kinds is private). It replaces `the_policy_file_parses_with_one_and_sixteen_everywhere`. Every `KernelKind` and `DeviceClass` has `vector_size`, `quad_tile` and (CPU classes) `cpu_block` entries, every value is a power of two from 1 to 16, and `Scalar` is 1. It fails until T5 adds `cpu_block` and the per-size `quad_tile`.
    - `gridcl/tests/kernel_invariance.rs::{topology_keys_sizes, permutation_info_sizes}`: above size 1, `record_plans` must show a `Scalar` transpose plan before the candidate plan. Fails until T10.
  - Verify: `cargo test --workspace --no-run` compiles. The new tests fail with `not yet implemented`, a missing-data message, or the missing policy entries; every other test passes.
  - After: –
  - Parallel with: –
  - Risks / notes: larger than about 600 lines only if the stubs are; data-dependent tests fail until T17.
- [x] **T1 M3 review follow-ups m9, m10, m11 and the small m8 items**
  - Goal: close the follow-ups the maintainer scheduled first.
  - Crates/files:
    - `gridcl-topology/src/topology.rs` (m9): `Topology::new`'s rustdoc states that kernels clamp out-of-range indices and that (tdim, 0) returns the input unchanged.
    - `gridcl-geometry/tests/bounds.rs` (m10): `cells_without_nodes_read_no_node_memory` in f32 and, where supported, f64; skipped f64 cases print why, as `tests/field.rs` does.
    - `gridcl-geometry/src/geometry.rs` (m11): `Geometry::new` returns `LengthMismatch` for zero nodes with a non-empty dofmap.
    - `gridcl-geometry/src/layout.rs` (m8: SAFETY comment wrapping) and `gridcl-geometry/src/point_set.rs` (m8: "Gauss–Jacobi" in the `len` example).
    - `gridcl/Cargo.toml` and `gridcl-geometry/Cargo.toml` (m8): drop the unused `proptest` dev-dependency.
  - Public API added or changed: none. `Geometry::new` rejects one more invalid input, as the M3 review did for `Topology::new`.
  - Tests: new `gridcl-geometry/tests/bounds.rs::empty_nodes_with_cells_are_rejected`; the m10 changes.
  - Verify: `cargo test -p gridcl-topology -p gridcl-geometry -p gridcl`; the same on Metal.
  - After: T0
  - Parallel with: T5, T7
  - Risks / notes: –
- [x] **T2 Property tests at 256 cases (m3)**
  - Goal: `prop_entities_invariant_under_shuffles` and `prop_bits_match_orientation_bits` run 256 cases.
  - Crates/files: `gridcl-topology/tests/{entities.rs, permutation.rs}`. Also `.github/workflows/ci.yml`, only if the tests become release-only: the hosted cpu job has no release step.
  - Public API added or changed: none.
  - Tests: those two property tests.
  - Verify: `cargo test -p gridcl-topology` in debug, timed.
  - After: T0
  - Parallel with: T3, T4, T5, T7
  - Risks / notes: if they add more than about 2 min in debug on the M3 Max, they get `#[cfg_attr(debug_assertions, ignore = "release only")]`, and a release step for them is added to the hosted cpu job. Logged.
- [x] **T3 Rustdoc examples, gridcl-geometry and gridcl-topology (m2, part 1)**
  - Goal: examples for `DeviceGrid`, `EVector`, `QuadChunk`, `BlockFields`, the `QuadField` methods, the shape types, the six block functions, `EntityBlock` and `Connectivity`.
  - Crates/files: `gridcl-geometry/src/{grid.rs, evector.rs, field.rs, shape.rs, points_jac.rs, detj_pinv.rs, facets.rs}`, `gridcl-topology/src/entity.rs`.
  - Public API added or changed: none.
  - Tests: doctests.
  - Verify: `cargo test --doc -p gridcl-geometry -p gridcl-topology`; `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps`.
  - After: T1
  - Parallel with: T2, T4
  - Risks / notes: –
- [x] **T4 Rustdoc examples, gridcl (m2, part 2)**
  - Goal: examples for `Mesh` (type, aliases, constants, methods) and for `AnyMesh` and `AnyMeshRef` (`as_ref`, `tdim`, `gdim`).
  - Crates/files: `gridcl/src/{mesh.rs, any.rs}`.
  - Public API added or changed: none.
  - Tests: doctests.
  - Verify: `cargo test --doc -p gridcl`.
  - After: T1
  - Parallel with: T2, T3
  - Risks / notes: –
- [x] **T5 Policy block sizes, per-size tiles and the benchmark hooks (gridcl-device)**
  - Goal: the ADR-0015 items of gridcl-device, with no change to any launch.
  - Crates/files: `gridcl-device/src/{policy.rs, plan.rs, testing.rs}`, `gridcl-device/vector-policy.toml`, `gridcl-device/tests/launch_plan.rs`.
  - Public API added or changed: `VectorPolicy::{cpu_block_bytes, quad_tile(elem_bytes)}`; `testing::{with_kind_vector_size, with_cpu_block_bytes}`.
  - Tests:
    - makes `policy_covers_every_kind_and_class` pass;
    - new unit tests: the kind override takes precedence over the global override and the environment; the block override is bounded, and rounds to whole lines at 4, 16 and 64 KiB;
    - `cpu_blocks_are_whole_lines_of_about_16_kib` and `cpu_plan_one_unit_per_thread_whole_lines` read the policy's block;
    - `prop_plan_respects_injected_limits` also runs under block overrides.
  - Verify: `cargo test -p gridcl-device`; Metal; Locust CUDA and CPU `cargo test --release -p gridcl-device` (launch shapes).
  - After: T0
  - Parallel with: T1, T2, T7
  - Risks / notes: the `quad_tile` callers in gridcl-geometry pass `size_of::<F>()` from T0 on.
- [x] **T6 Scalar helper kernels plan as `KernelKind::Scalar` (m8)**
  - Goal: `record_plans`, per-kind overrides and block entries attribute only candidate kernels to candidate families.
  - Crates/files:
    - `gridcl-device/src/primitives/{histogram.rs, segmented.rs, gather.rs (scatter), unique.rs}`;
    - `gridcl-topology/src/{entity.rs, renumber.rs, connectivity.rs}`;
    - `gridcl-geometry/src/measure.rs`.
  - Public API added or changed: none. Primitives with a vectorized phase (radix sort, scan) keep one plan for every phase (ADR-0009).
  - Tests: unit tests assert the kinds `record_plans` reports for `histogram`, `create_entities(1)` and `measure`; the existing suites.
  - Verify: `cargo test -p gridcl-device -p gridcl-topology -p gridcl-geometry -p gridcl`; Metal; Locust CUDA and CPU for the same crates (launch code changes).
  - After: T5
  - Parallel with: T7
  - Risks / notes: –
- [x] **T7 Myriad CPU-only target `myriad-cpu` (ADR-0014 follow-up)**
  - Goal: `scripts/hpc/remote-test.sh myriad-cpu --cores 36 -- <command>` runs on a D node (`-ac allow=D`, no `-l gpu`) with `GRIDCL_TEST_DEVICE=cpu`.
  - Crates/files: `scripts/hpc/{lib/config.sh, remote-test.sh, jobs/myriad-cpu.sh, targets.example.env, acceptance.sh, tests/run.sh}`, `docs/dev-environment.md` (D-node row), and `.claude/skills/hpc-test/SKILL.md`. The sandbox denies writes to `.claude/skills`, so the maintainer approves or applies that edit.
  - Public API added or changed: none.
  - Tests:
    - new fake-SGE cases in `scripts/hpc/tests/run.sh`: no GPU request, `allow=D`, and `--gpus` rejected;
    - one real smoke job with `--cores 36` running `nproc` and `cargo test --release -p gridcl-device`.
  - Verify: `bash scripts/hpc/tests/run.sh`, `shellcheck`, and the smoke job's report.
  - After: T0
  - Parallel with: T1–T6, T8–T10
  - Risks / notes: CPU-runtime thread detection on Myriad follows the job's slots, so CPU work always requests `--cores 36`.
- [x] **T19 Kathleen target `kathleen` (ADR-0016)**
  - Goal: `scripts/hpc/remote-test.sh kathleen [--nodes 1|2] [--ranks N] [--time …] -- <command>` runs a Slurm job of whole Kathleen nodes with `GRIDCL_TEST_DEVICE=cpu`; the M3s acceptance data reads Kathleen as the CpuSimd512 target.
  - Crates/files:
    - `scripts/hpc/{lib/config.sh, lib/submit.sh, remote-test.sh, status.sh, cancel.sh, jobs/kathleen.sh, targets.example.env, acceptance.sh, tests/run.sh, tests/fake-bin/}` (scheduler kind `slurm`);
    - `docs/dev-environment.md` (a Kathleen section: setup, layout under Scratch, QoS, the job environment);
    - `gridcl-bench/tests/perf_record.rs` and `gridcl-bench/src/record.rs` (target names);
    - `.claude/skills/hpc-test/SKILL.md` (the maintainer applies it; sandbox).
  - Rules (ADR-0016): at most two nodes per job; QoS `singlenode` for 1 node, `test` for 2 nodes up to 1 h, else `small`; `--gpus` rejected; `--cores` ignored (whole nodes); crates fetched on the login node, jobs offline; temporary files and Open MPI's session directory on `/dev/shm`; `PSM2_TID=0`; `SLURM_STEP_ID` unset in the batch step; the gcc runtime first on `LD_LIBRARY_PATH`; the remote base under Scratch in `targets.env` only; one tree (quota).
  - Public API added or changed: none.
  - Tests:
    - new fake-Slurm cases in `scripts/hpc/tests/run.sh`: `sbatch` arguments for 1 and 2 nodes, `--nodes 3` and `--gpus` rejected, polling through `squeue`/`sacct`, exit status and `--max-wait` cancel;
    - `perf_record.rs`: `kathleen` CPU sweep and throughput records required; `myriad-cpu` records count only with a non-empty `fallback_reason`;
    - real smoke jobs: one node, `nproc` and `cargo test --release -p gridcl-device`; two nodes, `--ranks 4`, `scripts/mpi-test.sh 2 4`.
  - Verify: `bash scripts/hpc/tests/run.sh`, `shellcheck`, both smoke reports; `cargo test -p gridcl-bench --test perf_record` still fails only for missing data.
  - After: T7
  - Parallel with: T8–T10
  - Risks / notes: half of Kathleen's nodes were down on 2026-10-24 (ADR-0016); one-time setup of the toolchain on the login node (downloads only).
- [x] **T8 gridcl-bench protocol, records, and Fill, Copy, Reduce**
  - Goal: the protocol of "Benchmarks" and the first three families.
  - Crates/files: `gridcl-bench/src/{lib.rs, protocol.rs, record.rs, families/mod.rs, families/device.rs, bin/vector_sweep.rs}`.
  - Public API added or changed: none (unpublished crate).
  - Protocol:
    - one warm-up launch per configuration, checked against a host reference (host maths in f64 with `widen`, as CLAUDE.md requires);
    - the median of ten timed runs, each ending with `client.sync()`;
    - compile time per vector size: the warm-up time minus the median, with CubeCL's persistent kernel cache off, or the cache state recorded;
    - bandwidth relative to `copy` at the same size on the same device;
    - sizes 10^5, 10^6 and 10^7, and CPU blocks of 4, 16 and 64 KiB;
    - on the M3 Max also `cube_dim` = the performance-core count, through `with_properties`, with the count read by the binary from `sysctl`.
  - Configurations are set with `with_kind_vector_size(Fixed(n))` and `with_cpu_block_bytes`; the size that actually ran comes from `record_plans`.
  - Each record holds: backend, device, device class, target, family, scalar type, n, vector size, tile, block, units, `times_s`, `median_s`, compile time, bandwidth ratio, commit (`GRIDCL_BENCH_COMMIT`), profile, core counts, `fallback_reason`, and the node-load snapshot lines when present.
  - The CLI filters by family, type and size, so a busy part can be repeated.
  - Tests: unit tests of the median, the JSON round trip and the bandwidth formulas; a smoke run at 10^4 in `cargo test -p gridcl-bench`.
  - Verify: `cargo test -p gridcl-bench`; `cargo run --release -p gridcl-bench --bin vector_sweep -- --families fill --sizes 100000 --out $TMPDIR/x.json`.
  - After: T5, T6
  - Parallel with: T10
  - Risks / notes: –
- [x] **T9 gridcl-bench: ScanTile, SortDigits, Gather**
  - Goal: the remaining device-primitive families.
  - Crates/files: `gridcl-bench/src/families/device.rs`.
  - Scalar types: u32 for ScanTile and SortDigits; f32, f64 and u32 for Gather, with random in-range u32 indices.
  - Public API added or changed: none.
  - Tests: smoke runs at 10^4.
  - Verify: `cargo test -p gridcl-bench`.
  - After: T8
  - Parallel with: T10, T12
  - Risks / notes: –
- [ ] **T10 Cell-fastest connectivity and the topology `bench` hooks**
  - Goal: the copy of "Where vectors apply", behind the policy, plus `bench::{emit_keys, permutation_info}`.
  - Crates/files: `gridcl-topology/src/{keys.rs, permutation.rs, transpose.rs (new), bench.rs, lib.rs}`, `gridcl-topology/Cargo.toml` (feature `bench`).
  - Public API added or changed: the hidden `gridcl_topology::bench` module (feature `bench`).
  - Kernel rules (CLAUDE.md): generic over N and correct at N = 1; lanes are cells; arithmetic masks for padding lanes; at most six buffers; shape from `LaunchPlan`, with overflow-checked sizes; the transpose plans as `Scalar`.
  - Tests:
    - makes the T0 additions to `kernel_invariance` pass;
    - new unit test `transpose_round_trips` for 0, 1, 15, 16, 17 and 10^6 + 3 cells;
    - `every_vector_size_gives_the_same_keys` and `every_vector_size_gives_the_same_bits` cover both layouts.
  - Verify: `cargo test -p gridcl-topology -p gridcl`; Metal; `GRIDCL_VECTOR_SIZE=widest` on the CPU runtime; Locust CUDA and CPU for `-p gridcl-topology -p gridcl` (kernel change).
  - After: T6
  - Parallel with: T7, T8, T9
  - Risks / notes: the transient copy costs one extra buffer of the connectivity's size per launch at N > 1.
- [ ] **T11 gridcl-bench: TopologyKeys, PermutationInfo, EVector, GeomFactors**
  - Goal: the sweep for the mesh families, at tile widths 1, 4 and 16 for GeomFactors.
  - Crates/files: `gridcl-bench/src/families/mesh.rs`, `gridcl-bench/Cargo.toml` (gridcl-topology with `bench`), and possibly `gridcl-testing/src/meshes/` (question 9).
  - Workloads (question 9): items are cells, at 10^5, 10^6 and 10^7. EVector and GeomFactors run in f32 and f64; the topology families use u32.
  - Public API added or changed: none.
  - Tests: a smoke run at 10^3 cells, checking the records' cell counts.
  - Verify: `cargo test -p gridcl-bench`; a short local sweep.
  - After: T9, T10
  - Parallel with: T12
  - Risks / notes: –
- [ ] **T12 Decision rule and report generator**
  - Goal: `decide` and `report::render` implement the rule as read in question 4. `bin/policy_report.rs` writes `vector-policy.toml` and the generated tables of `docs/perf/vectorization.md` from `docs/perf/data/`.
  - Crates/files: `gridcl-bench/src/{decide.rs, report.rs, bin/policy_report.rs}`.
  - Public API added or changed: none.
  - Tests:
    - unit tests on synthetic data: 3–10% gains stay at 1; a 10% gain at only one size stays at 1; the smallest size within 3% wins; tile and block ties; multi-device geometric means with clamped sizes and missing f64; f32/u32 disagreement; families with no data (Pack, PullBack); CpuSimd256 copies CpuSimd512; element sizes no family uses;
    - a proptest that every decision satisfies both acceptance inequalities on random data.
  - Verify: `cargo test -p gridcl-bench --lib`.
  - After: T8
  - Parallel with: T9, T10, T11
  - Risks / notes: –
- [ ] **T13 End-to-end throughput binary**
  - Goal: entities created per second and geometric factors per second, with the chosen policy and with every kernel scalar (`--vector-size policy|scalar`).
  - Crates/files: `gridcl-bench/src/bin/mesh_throughput.rs`, `gridcl-bench/src/record.rs`.
  - Measures (question 10):
    - entities per second: entities created by `create_entities(d)` for every 0 < d < tdim on P1 tetrahedron and P1 hexahedron boxes of about 10^6 cells, over the time of those calls, on fresh meshes;
    - factors per second: (cell, quadrature point) pairs for which X, J, det J and K are computed on P2 tetrahedra with a degree-4 rule, over the time of the four mesh-level calls.
  - Public API added or changed: none.
  - Tests: a smoke run at 10^3 cells.
  - Verify: `cargo test -p gridcl-bench`; a local run.
  - After: T11, T19
  - Parallel with: T12
  - Risks / notes: –
- [ ] **T14 CPU `cube_dim` and test-time study (m6, question 11)**
  - Goal: the M2/M3 follow-up on the `sync_cube` slowness of 16-unit CPU cubes and on the test-time budgets.
  - Crates/files: a section of `docs/perf/vectorization.md` (draft) and this plan's Deviations log. Nothing is committed to library code.
  - Method: debug `primitives_prop` and debug `cargo test --workspace` on the idle M3 Max, timed at `cube_dim` 16, 12 and 4 through a temporary, uncommitted cap in `plan.rs`.
  - Public API added or changed: none. If a smaller CPU `cube_dim` meets the budgets and the sweep's `cube_dim` data (T8, T11) supports it, the maintainer decides on an ADR that changes the "CPUs" rule before T15. Otherwise the results are recorded and the follow-up is closed or handed on to M10.
  - Verify: the timings in the report.
  - After: T11
  - Parallel with: T12, T13
  - Risks / notes: an ADR here delays T15–T17, because the measurements must use the final CPU rule.
- [ ] **T15 Measurements on the M3 Max (CPU runtime and Metal)**
  - Goal: the M3 Max sweep JSON.
  - Crates/files: `docs/perf/data/m3s-sweep-{cpu,metal}-m3max.json`.
  - Public API added or changed: none.
  - Conditions ("Benchmarks"):
    - mains power and other applications closed, which the maintainer confirms;
    - release build, and no other cargo job running;
    - `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu` recorded.
    - The sandbox blocks Metal and `sysctl`, so these commands run outside it, with approval.
  - Verify: the M3 Max part of `benchmark_data_covers_required_targets`.
  - After: T12, T13, T14
  - Parallel with: T16
  - Risks / notes: the full CPU sweep may take hours; it runs in parts by family.
- [ ] **T16 Measurements on Locust (Grace CPU, H100) and a Kathleen node (AVX-512 CPU)**
  - Goal: the remote sweep JSON, under the idle rule.
  - Crates/files: `docs/perf/data/m3s-sweep-{cpu,cuda}-locust.json`, `docs/perf/data/m3s-sweep-cpu-kathleen.json` (ADR-0016).
  - Public API added or changed: none.
  - Runs:
    - Locust: `remote-test.sh locust -- 'env GRIDCL_TEST_DEVICE=cpu GRIDCL_BENCH_COMMIT=<sha> cargo run --release -p gridcl-bench --bin vector_sweep -- --families … --out results/$JOB_ID/…'`, and the same with `--features cuda`, in short parts by family. A part counts only if both snapshots are `idle`; otherwise it is repeated (ADR-0002, ADR-0005).
    - Kathleen: `remote-test.sh kathleen --time …` (one whole node, QoS `singlenode`), in parts by family; at most two scheduler jobs. A Myriad D node (`myriad-cpu --cores 36`) only as the fallback, with `fallback_reason` (ADR-0016).
    - Fallback (question 7): if Locust's GPU stays busy, a Myriad V100 or A100 CUDA run, with `fallback_reason`.
  - Verify: `benchmark_data_covers_required_targets` passes.
  - After: T7, T12, T13, T14, T19
  - Parallel with: T15
  - Risks / notes: question 7.
- [ ] **T17 Policy, report and throughput**
  - Goal: apply the rule.
  - Crates/files: `gridcl-device/vector-policy.toml`, `docs/perf/vectorization.md`, `docs/perf/data/m3s-throughput-cpu-{m3max,kathleen}.json`.
  - Steps:
    - `policy_report` writes the TOML and the generated tables;
    - the hand-written parts of the report: protocol, conditions with job ids and node load, observations, the `cube_dim` result (question 8), compile times and T14;
    - `mesh_throughput` runs on the M3 Max and a Kathleen node, with `policy` and with `scalar`.
  - Public API added or changed: none (policy values only).
  - Tests: all four `perf_record` tests and `policy_covers_every_kind_and_class` pass.
  - Verify: `cargo test -p gridcl-bench -p gridcl-device`.
  - After: T15, T16
  - Parallel with: –
  - Risks / notes: –
- [ ] **T18 Whole suite under the chosen policy on every target; test-time budgets**
  - Goal: acceptance bullet 5, and the m6 budgets re-measured.
  - Crates/files: this plan (Deviations), plus fixes found by the runs, each logged.
  - Public API added or changed: none.
  - Runs, through the test-runner subagent:
    - M3 Max: `cargo test --workspace` (debug, timed); release under `GRIDCL_VECTOR_SIZE` = `policy`, `scalar` and `widest`; `GRIDCL_QUAD_TILE=1`; Metal; `scripts/mpi-test.sh 1 2 3 4`.
    - Locust: CUDA and the CPU runtime.
    - Kathleen (one node): the CPU runtime, the only run that exercises the CpuSimd512 table.
    - Hosted CI: cpu, vulkan, mpi-mpich and mpi-openmpi.
  - Verify: the runs above.
  - After: T17
  - Parallel with: –
  - Risks / notes: –

## Acceptance coverage

Every acceptance bullet of the milestone in docs/design.md appears exactly once.

| Acceptance criterion (docs/design.md) | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| Benchmark JSON under `docs/perf/data/` for the M3 Max (CPU, Metal), Locust (Grace CPU, CUDA) and a Kathleen node (AVX-512 CPU), plus a Myriad D node if Kathleen was unavailable and V100/A100 if Locust was unavailable (ADR-0016) | `gridcl-bench/tests/perf_record.rs::benchmark_data_covers_required_targets` | T0, T7, T8, T9, T11, T15, T16, T19 | data: M3 Max CPU f32/f64/u32 and Metal f32/u32; Locust CPU and CUDA f32/f64/u32 (idle); Kathleen CPU f32/f64/u32 (one whole node); test: M3 Max CPU, hosted cpu |
| `vectorization.md` lists, per family, device class and scalar type, the speedup of each size over 1, the chosen size and the reason | `…::report_lists_every_family_class_and_type` | T0, T12, T17 | M3 Max CPU; hosted cpu |
| Every size above 1 is ≥ 10% faster than 1 at 10^6 and 10^7; every chosen (vector size, tile, block) is within 3% of the fastest | `…::policy_follows_the_decision_rule` | T0, T12, T17 | M3 Max CPU; hosted cpu |
| A test checks that `vector-policy.toml` covers every `KernelKind` and `DeviceClass` with powers of two from 1 to 16 | `gridcl-device/src/policy.rs::tests::policy_covers_every_kind_and_class` | T0, T5, T17 | M3 Max CPU; hosted cpu |
| With the chosen policy, the whole suite passes on every target, the vector-size invariance tests included | `cargo test --workspace` and the invariance tests below, under the committed policy and `GRIDCL_VECTOR_SIZE` = `policy`, `scalar`, `widest` | T18 | M3 Max: CPU f32/f64 (debug and release), Metal f32, MPI 1–4 ranks; Locust: CUDA f32/f64 and Grace CPU; Kathleen: CPU f32/f64 (CpuSimd512); hosted CI: cpu, vulkan, MPICH, Open MPI |
| Entities created per second and geometric factors per second on the M3 Max CPU runtime and a Kathleen node, with the chosen policy and with every kernel scalar (ADR-0016) | `…::throughput_recorded_with_policy_and_scalar` | T0, T13, T17, T19 | data: M3 Max CPU f64, Kathleen CPU f64 (one whole node); test: M3 Max CPU, hosted cpu |

Vector-size invariance of every candidate kernel. These tests exist from M2 and M3; T10 extends
the topology tests to the cell-fastest path, and T18 reruns all of them under the chosen policy.
Pack and PullBack get theirs in M6 and M9.

| Family | Test | Lengths |
| --- | --- | --- |
| Fill, Copy, Reduce, ScanTile, Gather | `gridcl-device/tests/vector_invariance.rs::{fill,copy,reduce,scan,gather}_sizes` | 0, 1, 15, 16, 17, 10^6 + 3 |
| SortDigits | `gridcl-device/tests/vector_invariance.rs::sort_digits_sizes` | as above |
| TopologyKeys | `gridcl/tests/kernel_invariance.rs::topology_keys_sizes`; `gridcl-topology/src/keys.rs::every_vector_size_gives_the_same_keys` | 0, 1, 15, 16, 17 cells of every type; 10^6 + 3 triangles |
| PermutationInfo | `gridcl/tests/kernel_invariance.rs::permutation_info_sizes`; `gridcl-topology/tests/permutation.rs::every_vector_size_gives_the_same_bits` | as above |
| EVector | `gridcl/tests/kernel_invariance.rs::evector_sizes` (f32, f64) | as above |
| GeomFactors | `gridcl/tests/factor_invariance.rs::factor_kernels_match_lvector_reference` | tiles 1 and 16; as above |

## Open questions and ADR candidates

Resolve every row before Status becomes approved.

| # | Question or conflict | Design section | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| 1 | The branch `tbetcke/tbetcke-m3s-simd-vectorization-and-benchmarks` is not named `m3s-<slug>`. | CLAUDE.md Workflow | Keep it and log it, as M0–M3 did. | answered (precedent) |
| 2 | Listing S2 has no hook for the CPU block size or for a per-kind vector size, and the policy has no field for block sizes. "Benchmarks" nevertheless measures block sizes, the rule chooses them, and "the choices go to `vector-policy.toml`". | Listing S2; "Benchmarks"; acceptance bullet 4 | ADR-0015 before T0, covering "Planned public API and rules". Blocks are stored per (kind, element size) in units of 4 KiB (1, 4, 16), so bullet 4 holds literally and the 3% rule can hold per family. | answered (maintainer, 2026-10-24: per kind and size); ADR-0015 (accepted) |
| 3 | TopologyKeys runs inside `create_entities`, where the sort dominates (about 2 s for 10^7 keys on the M3 Max CPU runtime). `Topology::new` also uploads ids and builds an `IndexMap` around PermutationInfo. Timed end to end, a kernel gain falls below the 10% threshold. | "Benchmarks"; "Cargo features"; crate table | (a) The hidden `bench` feature with `emit_keys` and `permutation_info` (ADR-0015). Alternatives: (b) per-kernel times from CubeCL 0.11's profiling log (`[profiling] logger`, level `basic`), with no API change but unknown accuracy on the CPU runtime; (c) time the public calls end to end and accept the dilution. | answered (maintainer, 2026-10-24: (a)); ADR-0015 (accepted) |
| 4 | The decision rule (a) leaves open how one class measured on two devices decides (CpuSimd128: M3 Max and Grace; Gpu: Metal and H100), and (b) conflicts with acceptance bullet 3 when a larger size is 3–10% faster. | "Benchmarks", decision rule; acceptance bullets 3 and 1 | (a) Geometric mean of the per-device time ratios, with clamped or missing sizes as in "Planned public API"; per-device numbers in the report. Alternative reading: "on its device class" means on every device of the class (stricter; adopts less). (b) The 3% rule applies among the configurations the 10% rule admits. Both go into ADR-0015, because they fix how acceptance bullets read. | answered (maintainer, 2026-10-24: geometric mean; 3% among admitted); ADR-0015 (accepted) |
| 5 | Lifetime and location of the cell-fastest copy. | "Where vectors apply"; ADR-0012 | Transient and private, built per launch at N > 1, with the transpose planned as `Scalar` but timed with the family. M3's lane-by-lane path at N > 1 is replaced. No public type changes. | answered (maintainer approved the plan, 2026-10-24) |
| 6 | m8: scalar helper kernels plan with candidate kinds, which misattributes them in `record_plans` and would give them a family's block size. | Listing S2 (`KernelKind`); ADR-0009 | Add `KernelKind::Scalar` (ADR-0015). Alternative: keep the kinds and filter in gridcl-bench by `max_vector`; no API change, but attribution stays misleading and the block entries stay shared. | answered (maintainer, 2026-10-24: `KernelKind::Scalar`); ADR-0015 (accepted) |
| 7 | Locust was busy in all three M2 benchmark runs. Bullet 1 requires Locust's Grace CPU data, which Myriad cannot replace (x86-64). Bullet 1 also names "Myriad's V100 and A100 nodes" for the fallback. | "Benchmarks", "Conditions"; ADR-0002, ADR-0014 | Repeat busy parts at different times over at least three days. CUDA falls back to a V100 or A100, with the reason recorded; one of the two is read as sufficient. If the Grace CPU never runs idle, stop and draft an ADR for the acceptance change (CpuSimd128 decided from the M3 Max alone). | answered (maintainer approved the plan, 2026-10-24) |
| 8 | `cube_dim` = performance-core count: the policy "may choose it", but no `DeviceProperties` field gives the performance-core count, and reading `hw.perflevel0.physicalcpu` in the library would be FFI outside the crates CLAUDE.md allows. | "Benchmarks", "Conditions"; CLAUDE.md (FFI) | Measure it with `with_properties`; the gridcl-bench binary reads `sysctl`. If it passes the rule on the M3 Max, draft a separate ADR for detection and a policy key (with T14). Otherwise record it and change nothing. | answered (maintainer approved the plan, 2026-10-24) |
| 9 | Workloads for the mesh families, and the slow host generation of 10^7-cell boxes (node deduplication through a `HashMap`). | "Benchmarks", "What it runs" | Items are cells. GeomFactors: X, J, det J and K (`physical_points_block`, `jacobians_block`, `detj_block`, `pseudo_inverse_block`), summed, on P2 tetrahedra and P2 triangles in 3D with a degree-4 rule. Normals and facet normals are left out: they need M9's `orient` for meaningful flags and share J's operation sequence. TopologyKeys and PermutationInfo: P1 tetrahedra. EVector: P2 tetrahedra. Meshes are built once per size. If building takes more than about a minute, T11 adds a lattice generator to gridcl-testing that needs no map. | answered (proposed) |
| 10 | Definitions of "entities created per second" and "geometric factors per second". | Acceptance bullet 6 | As in T13. | answered (proposed) |
| 11 | m6: the test-time budgets (hosted cpu 35:17 against about 30 min; debug workspace 16:01 against about 15 min) were to be revisited through the CPU `cube_dim`, which "CPUs" fixes at `max_units_per_cube`. | "CPUs"; M3 plan, question 20 | T14 measures the effect. A rule change is an ADR candidate, decided before T15; T18 re-measures. | answered (maintainer approved the plan, 2026-10-24); ADR only if T14 proposes a rule change |
| 12 | Element sizes that no family uses: Copy moves u32 words, and ScanTile, SortDigits, TopologyKeys and PermutationInfo are u32 only, so their 8-byte entries never apply. | "Where vectors apply" | Keep 1, with "not used" as the reason in the report. | answered (proposed) |
| 13 | SortDigits and ScanTile kernels are part of larger primitives, so their gains are diluted, as TopologyKeys' are. | "Benchmarks" | Time the smallest call whose launches all take the family's plan (ADR-0009: one partition per primitive). The policy entry then changes exactly the time measured. No hooks in gridcl-device. `radix_sort`'s inner `exclusive_scan` stays at its default through `with_kind_vector_size`. | answered (proposed) |
| 14 | Granularity of tiles: `quad_tile()` is one value per class (Listing S2), but the rule decides per scalar type, so f32 and f64 GeomFactors may prefer different tiles and bullet 3 could fail for one of them. Also, which tile is "size 1"'s baseline? (ADR-0012: tile 1 forces N = 1.) | Listing S2; "Benchmarks" | `quad_tile(elem_bytes)` with per-size TOML entries (ADR-0015). The baseline is the best size-1 configuration over all tiles and blocks. Alternative: one tile per class, chosen jointly, with a logged exception to bullet 3 if f32 and f64 disagree. | answered (maintainer, 2026-10-24: `quad_tile(elem_bytes)`); ADR-0015 (accepted) |
| 15 | The maintainer asked (2026-10-24) to add UCL Kathleen (Slurm, Xeon Gold 6248 with AVX-512) for CPU jobs, whose nodes are easier to get than Myriad's, and for multi-node tests on at most two nodes. Acceptance bullets 1 and 6 name a Myriad D node, and the runner has no Slurm support. | "Benchmarks", "Where"; "Development environment and test targets"; acceptance bullets 1 and 6 | Kathleen for CPU-only runs (CpuSimd512 sweeps, throughput, the suite) and for two-node MPI runs, with the D nodes of T7 as the fallback; a new task adds the `kathleen` target. | answered (maintainer, 2026-10-24: accepted); ADR-0016 (accepted) |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |
| 2026-10-24 | plan | Branch `tbetcke/tbetcke-m3s-simd-vectorization-and-benchmarks`, not `m3s-<slug>`. | Maintainer's worktree branch (question 1). | – |
| 2026-10-24 | plan | T1–T4 and T14 are M3 review follow-ups (m2, m3, m6, m8–m11), outside M3s's design scope. | Maintainer's follow-up schedule (M3 plan, Review). | – |
| 2026-10-24 | plan | The plan has an extra "Planned public API and rules" section. | M3's precedent: it fixes what ADR-0015 must decide before T0. | – |
| 2026-10-24 | T0 | `KernelKind::Scalar` is also swept on CPU targets for its `cpu_block` entries: histogram (u32) gives the 4-byte entry, segmented reduce sum (f64) the 8-byte entry. `perf_record` requires this data, and T9 adds the measurement. | ADR-0015 item 7 chooses Scalar's blocks "from the time of the scalar primitives (histogram, segmented_reduce)", but no task measured them. One type per element size avoids a shared-entry conflict between two different primitives. | ADR-0015 |
| 2026-10-24 | T0 | How `perf_record` reads the rule where the plan is silent: a configuration's class time is the geometric mean of the devices' medians, so ratios of class times are geometric means of per-device ratios. The 3% condition must hold at both 10^6 and 10^7, like the 10% condition. Records carry `nominal_n` (10^5, 10^6 or 10^7; mesh boxes give approximate cell counts) and a `cube_dim` mode (`Plan` or `PerformanceCores`), and only `Plan` records enter the rule. | The rule names no size for the 3% comparison; the mesh families cannot hit 10^6 cells exactly; question 8 keeps the performance-core `cube_dim` out of the configuration. | – |
| 2026-10-24 | T0 | The transpose check in `kernel_invariance` is stricter than "a `Scalar` plan before the candidate plan": above size 1, every TopologyKeys and PermutationInfo plan must directly follow a `Scalar` plan. `gridcl/tests/common/mod.rs` gains `at_vector_size_with_plans`. | After T6 other `Scalar` helpers also plan before them; with "directly follows", the mixed box fails until T10. | – |
| 2026-10-24 | T0 | Records name the Myriad CPU-only target `myriad-cpu`, the remote-test target of T7; the data file stays `m3s-sweep-cpu-myriad-d.json`. T0 also updates `gridcl-geometry/src/field.rs` (rustdoc of `quad_tile`). | Same names as the runner. | – |
| 2026-10-24 | T1 | `cells_without_nodes_read_no_node_memory` moved from `gridcl-geometry/tests/bounds.rs` to a unit test in `gridcl-geometry/src/evector.rs`, which calls the E-vector builder directly, in f32 and, where supported, f64. `bounds.rs` gains `empty_nodes_with_cells_are_rejected` (f32 and f64), and its skipped f64 cases now print why. | m11 makes `Geometry::new` reject exactly the input that test built (no nodes, a non-empty dofmap), so m10's kernel check is reachable only below the public constructor. | – |
| 2026-10-24 | T7, T19 | Kathleen replaces the Myriad D node as the CpuSimd512 and throughput target; D nodes become the fallback. New task T19 adds the `kathleen` target; T13, T16, T17 and T18 and the acceptance-coverage rows 1 and 6 now name Kathleen. T7 continues: `myriad-cpu` is the fallback target. | Maintainer's request; T7's smoke job waited about two hours for a D node. | ADR-0016 |
| 2026-10-24 | T7 | T7 committed before its real smoke job ran: Myriad job 10224 (--cores 36, nproc and cargo test --release -p gridcl-device) was still queued after about two hours; its result is recorded when it finishes, and a failure is fixed in a follow-up commit. | Maintainer's decision. | – |
| 2026-10-24 | T7 | `.claude/skills/hpc-test/SKILL.md` is not edited in this commit; the proposed change goes to the main session, which merges it with ADR-0016's changes. | The sandbox denies writes to `.claude/skills`. | – |
| 2026-10-24 | T7 | Two files outside the task's list change: `scripts/hpc/lib/submit.sh` exports `GRIDCL_TEST_DEVICE` per target (`cpu` on `myriad-cpu`), and `scripts/hpc/jobs/common.sh` records a prologue line `cpus:` (`nproc` without the OMP variables, online CPUs, `OMP_NUM_THREADS`). The smoke job therefore runs `nproc` in its prologue, and its command stays a plain `cargo test`, so build and run times are recorded separately. | The job settings and the prologue live in those files. | – |
| 2026-10-24 | T7 | The runner usage in docs/design.md ("Remote-test runner") still lists `targets: myriad-v100 \| myriad-a100 \| locust` and says jobs run with `GRIDCL_TEST_DEVICE=cuda`; the runner now lists `myriad-cpu` and its jobs run with `cpu`. design.md is not edited during implementation; the maintainer updates that block. | ADR-0014 (accepted) adds the target; this plan fixes `GRIDCL_TEST_DEVICE=cpu`. `adr: 0016 apply` (2843d4a) has since updated the target list; the `GRIDCL_TEST_DEVICE=cuda` sentence of step 4 remains. | ADR-0014 |
| 2026-10-24 | T7 | `myriad-cpu` without `--cores` is accepted with a note on stderr (one slot), not refused or defaulted to 36. `acceptance.sh myriad-cpu` runs `turnaround fail max-wait` (turnaround with `--cores 36`, env-check `--features cpu`, no `--expect-sm`) and refuses `nvrtc` and `gpus2`; it has not been run live. | Short CPU-only checks need no whole node; the GPU steps do not apply. | – |
| 2026-10-24 | T19 | T19 committed before its real smoke jobs ran. Every Kathleen node had failed the health check and was drained or draining ahead of the `rdss-upgrade` maintenance (all nodes from 2026-10-27 08:00, no end date set). The one-node job 608316 stayed pending (`ReqNodeNotAvail`) and was cancelled with `cancel.sh`. The one-node and two-node smoke jobs run when Kathleen accepts jobs again, before T16. Until then CPU-only remote runs (T13, T16–T18) use `myriad-cpu` with that fallback reason. | Maintainer's decision (2026-10-24): Kathleen may not accept jobs for several days; Myriad is the only option meanwhile. | ADR-0016 |
| 2026-10-24 | T19 | Files outside the task's list: `scripts/hpc/lib/{sync.sh, report.sh}` (Slurm liveness through `squeue`; the Slurm log moved into the results), `scripts/hpc/tests/fake-bin/{sbatch, squeue, scancel}`, and `ThroughputRecord::fallback_reason` in `gridcl-bench/src/record.rs`. The one-time setup on Kathleen's login node (root directories, rustup) ran as recorded commands in `docs/dev-environment.md`, not as a committed script. `acceptance.sh kathleen` (`turnaround fail max-wait`) has not run live. | Needed for the Slurm kind and the fallback rule; the setup is two commands. | – |
| 2026-10-24 | T7 | Job 10224 kept queued without its runner (stopped with SIGKILL at about 20:30, before its 5 h `--max-wait` would delete it); its results are copied back by hand from `results/10224/` of the T7 tree on Myriad. A second smoke job with `--cores 8` (10309) runs alongside: part of a D node may start sooner and still checks `-ac allow=D`, no GPU and `GRIDCL_TEST_DEVICE=cpu`. | Kathleen is drained, so the D-node queue is the only CPU path; maintainer's decision (2026-10-24). | ADR-0016 |
| 2026-10-24 | T8 | Additions to the planned design: the `Workload` trait sits in `protocol.rs` and `measure` takes it; `gridcl-bench` enables gridcl-device's `testing` feature directly; `vector_sweep --target` defaults to `GRIDCL_HPC_TARGET`, else `local` (so T15 passes `--target m3max`); `--cube-dim auto` adds the performance-core runs wherever `sysctl` reports a count; `--out` is rewritten after every record; a subcommand `vector_sweep annotate --node-load results/<job> FILE...` adds the runner's node-load snapshots afterwards, because the "after" snapshot is taken once the command has finished. The bandwidth reference is a u32 copy of the same length under the policy, measured once per size. Compile time is non-zero only for the first configuration of each vector size, because the block size does not change the kernel; the report takes the largest value per vector size. | The plan did not say how the snapshots or the copy reference reach the records. | – |
| 2026-10-24 | T9 | T9 also measures `KernelKind::Scalar` (histogram, u32, 256 bins; `segmented_reduce` sum, f64, segments of 0 to 16 values), on CPUs only and at vector size 1, as the T0 row requires. ScanTile times `inclusive_scan` then `exclusive_scan` as one region, so each configuration has one record. `with_cpu_block_bytes` applies to every kind, so a SortDigits block also reaches `radix_sort`'s inner `exclusive_scan` (and the Fill and Scalar helpers inside the timed calls); only the vector size stays per kind. SortDigits records have no bandwidth ratio (several passes). | Per-kind block overrides do not exist (ADR-0015 has one scoped block for all kinds); the policy entry of a family then changes the measured time together with the helpers' blocks, which the M10 re-run can revisit. | – |

## Review

<Filled by /review-milestone: reviewer findings and how each was resolved, test evidence per target and backend (command, job id for remote runs, result), follow-ups for later milestones.>
