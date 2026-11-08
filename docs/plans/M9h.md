# M9h High-order performance — plan

- **Status:** draft
- **Branch:** tbetcke/tbetcke-m9h-high-order-performance (user prefix, as for M3–M9)
- **Issue:** #37
- **Base commit:** d90f90ecb80faedf19f651a9bd39f49fdeb6aa4b
- **Design sections:**
  - "Implementation roadmap / M9h" (ADR-0041); "M10 Hardening" (benchmarks of M9h's kernels).
  - "Geometry": "Coordinate element and storage", "Tabulation and quadrature", "Geometric factors", "Pull-back".
  - "Vectorization and CPU execution": Listing S2 (`LaunchPlan`, `VectorPolicy`, `testing`), "How the CPU runtime runs a kernel", "Where vectors apply", "Determinism across vector sizes", "Benchmarks".
  - "Device model and backends": "Device limits"; "Workspace layout": "Cargo features".
  - "Testing and validation"; "Development environment and test targets"; "Risks and mitigations" (C18 row).
- **Depends on:** M9 (done, merged; docs/plans/M9.md `Status: done`).
- **Accepted ADRs that apply:**
  - ADR-0041: the milestone; its four open points (rows 1, 2, 3 and 10 below).
  - ADR-0012: factor kernels plan with `max_vector = tile`; fresh fields.
  - ADR-0013: CPU runtime fuses multiply-adds; lane-independent kernels bitwise across vector sizes on the CPU runtime.
  - ADR-0014, ADR-0037: Locust for every CUDA run and benchmark; A4000 first CUDA fallback (timings only if idle).
  - ADR-0015, ADR-0018, ADR-0019, ADR-0021: protocol, decision rule, backstop, classes by geometric mean.
  - ADR-0016, ADR-0020: one Kathleen socket for CpuSimd512 benchmarks (Myriad D node as fallback); two-node Kathleen MPI at the review.
  - ADR-0017, ADR-0034: CPU units; the local Mac is the M6 Mac mini; M3 Max data is history.
  - ADR-0024: remote runs set `GRIDCL_SKIP_PYTHON=1`.
  - ADR-0038: hidden cross-crate hooks.
  - ADR-0039: pull-back (items 13–15), device basis evaluator, coefficient cache; the copy-list clause for expansion coefficients.
- **ADR before approval:** [ADR-0042](../adr/0042-m9h-public-items-and-rules-the-design-leaves-open.md) (Proposed) "M9h public items and rules the design leaves open" (rules 1–13 below; rows 1–4 already decided by the maintainer).

<!--
Rules for this file (do not delete):
- Status moves draft -> approved (maintainer) -> implemented (/implement-milestone) -> done (maintainer, after merge).
- Every task is one commit. T0 is always the acceptance tests.
- The plan references design sections; it never restates or changes requirements.
- Tick a task ([x]) in the same commit that completes it.
-->

## Scope summary

M9h measures every degree-dependent device kernel family (EVector, GeomFactors with all six factors, PullBack, the device basis evaluator) at p = 1–8 with the M3s protocol. It then implements and compares the strategies of ADR-0041 against M9's direct evaluation: runtime loops, sum factorisation for quadrilaterals, hexahedra and prisms, shared-memory tabulations and coefficients, and a workgroup cooperating on one cell. A degree policy chooses one strategy per family, cell type, degree range and device class from the data, records it in `docs/perf/high-order.md`, and can be overridden for tests. C18 is isolated and closed.

## Findings from planning

Read against main at d90f90e.

1. **Degree enters every kernel at comptime, and node loops unroll fully.**
   - Factor kernel (`gridcl-geometry/src/layout.rs:233`): `GeomLayout { tdim, gdim, nodes_per_cell, num_points, tile }` is comptime. The point loop is a runtime loop, but the node loop is `#[unroll]` (:100–114, :265–276), about 6,561 FMAs per point for J at hexahedron p = 8. Five buffers.
   - E-vector (`evector.rs:135`): node and coordinate loops unrolled; three buffers.
   - Pull-back (`pull_back.rs:88`):
     - Set-up a = C'·x is a runtime N² loop over C' in global memory, into a local `Array` of nb·gd vectors (2,187 at hexahedron p = 8).
     - Each Newton step calls `expansion` (`basis_eval/mod.rs:212`), which unrolls over every basis function (about 8,748 FMAs at hexahedron p = 8).
     - `live = nb·gd + …` forces vector size 1. Five buffers.
   - The factor kernel cache key includes `num_points`. The node family never enters a kernel.
2. **No degree caps.** gridcl-refcell rejects only p = 0. GLL is offered on intervals, quadrilaterals and hexahedra (`lagrange.rs:154`). `gauss_rule(cell, 2p)` has no cap. Host partition of unity and reproduction are tested to p = 8 (`gridcl-refcell/tests/tabulation.rs:12`), and `coefficients_invert_the_vandermonde` to p = 4. Gmsh reaches order 9–10 (A.7).
3. **Tensor structure of the bases.** The hexahedron's orthogonal set is x-fastest lexicographic Legendre (`gridcl-refcell/src/polynomials.rs:157`), and the prism's is Dubiner(p) × Legendre(p) on layered nodes (:174). So C' = (diag(c₁)C₁)^⊗3 for hexahedra, and C'_tri ⊗ C'_1 for prisms, under a node permutation.
4. **Device tests stop at p ≤ 4.**
   - `basis_eval/tests.rs`: p ≤ 4. `pull_back.rs::recovers_reference_points`: p ≤ 3. `factor_invariance.rs`: p = 2, against an L-vector device reference. `determinism.rs`: p = 2.
   - `parallel::cases` hard-codes p = 1..=2 (`gridcl-testing/src/parallel.rs:166, 197, 291`).
   - No test compares factors with a gridcl-refcell f64 host evaluation.
5. **Test meshes.** Generators accept any p but are equispaced only (`meshes/mod.rs:134`). No GLL generator. `perturbed` keeps det J > 0 documented for p ≤ 4 only. Self-tests stop at p ≤ 4.
6. **gridcl-bench has no degree axis.**
   - `P = 2` and `RULE_DEGREE = 4` are fixed (`families/mesh.rs:43–45`). Sizes count cells, which does not fit at p = 8.
   - The compile time is warm-up minus median over the timed region (`protocol.rs:210`).
   - `load_dir` reads `m3s-*` only. `policy_report` applies the rule and splices generated tables.
7. **CUDA introspection without FFI.**
   - `CUBECL_CUDA_DUMP_PTX=<dir>` writes every kernel's PTX on cubecl-cuda 0.11's default LLVM path, cache hits included (`context.rs:474, 580`).
   - The LLVM path emits the newest PTX version the driver loads (`compiler.rs`, `ptx_version`), so `ptxas` must match the driver's CUDA version; `ncu` reports registers and local memory directly.
   - `launch_failed` drops the failing call's name; `compute-sanitizer` reports API errors by call.
   - CubeCL's compilation cache is off in gridcl (no `persistence` feature; cubecl-server `build.rs:13`).
   - `LOCUST_MODULES` sets `CUDA_CACHE_PATH`, so first-use timings need `CUDA_CACHE_DISABLE=1`.
8. **C18's entry names NVRTC, but cubecl-cuda 0.11 defaults to LLVM** (`compiler.rs:24–31`); gridcl enables `cuda-cpp` only in `tools/env-check`.
9. **No geometry kernel uses shared memory**, and `LaunchPlan` has only `elementwise` (`gridcl-device/src/plan.rs:90`), which checks no shared-memory size and carries no group size.
10. **CI and run times.** New gridcl integration modules land in shard 6. Hosted CI must stay under 20 min; shard 7 took 17.7 min after M9, and shards 2 (4.6) and 3 (6.6) have headroom. A full debug gridcl suite takes about 25 min locally. M3s: tests adding over about 2 min in debug become release-only.
11. **Local benchmarks run outside the sandbox** (sysctl, Metal). The committed vectorization data is from the M3 Max, so the M9 baseline is measured again on the Mac mini.

## Planned public API and rules (ADR-0042, to draft)

```rust
// gridcl-device — Listing S2 (maintainer, 2026-11-08: plan the cooperative strategy up front)
pub struct LaunchPlan {
    /* existing fields */
    /// Units that cooperate on one group: 1 for `elementwise`.
    pub units_per_group: u32,
}
impl LaunchPlan {
    /// `groups` independent work groups (a cell or a tile of cells), each run by
    /// `units_per_group` units: the largest power of two ≤ min(units, max_units_per_cube).
    /// GPUs: cube_dim = units_per_group × groups per cube (256 units per cube, or fewer);
    /// CPUs: cube_dim = units_per_group, and each cube visits blocks of consecutive groups.
    /// `shared_bytes` per cube is checked against the device's shared-memory limit.
    pub fn per_group(client: &Client, kind: KernelKind, groups: usize, units: usize,
                     elem_bytes: usize, max_vector: usize, live: usize, shared_bytes: usize)
        -> Result<Self>;
}

// gridcl-geometry — features `testing` and `bench` (hidden, ADR-0038); Strategy is `pub` in a
// private module, re-exported only through the hidden modules
#[doc(hidden)] #[cfg(feature = "testing")]
pub mod testing {
    pub enum Strategy { Direct, Loops, SumFactorised, Shared, Cooperative }
    pub enum StrategyFamily { EVector, Points, Jacobian, Normals, FacetNormals, PullBack }
    pub enum StrategyOverride { Policy, Fixed(Strategy) }
    pub fn with_strategy<R>(family: StrategyFamily, s: StrategyOverride, f: impl FnOnce() -> R) -> R;
    pub fn record_strategies<R>(f: impl FnOnce() -> R) -> (R, Vec<(StrategyFamily, Strategy)>);
}
#[doc(hidden)] #[cfg(feature = "bench")]
pub mod bench { /* basis-evaluator, pull-back set-up and Newton-step hooks */ }
```

Rules (each an ADR-0042 item):

1. **Bullet 3, read** (maintainer, 2026-11-08: amend via ADR).
   - At an arbitrary point, a Newton step touches all (p + 1)^d coefficients, so its leading cost grows as (9/5)^d from p = 4 to p = 8: about 5.8 for hexahedra, 3.2 for quadrilaterals.
   - Geometric factors keep "at most 3 times".
   - Pull-back Newton step: at most 1.1 · (9/5)^d (hexahedra 6.4, quadrilaterals 3.6), a no-regression bound at the node-count floor.
   - Pull-back set-up per point, where a strategy has one: at most 3 times. The tensor pull-back of T8 evaluates 1D Lagrange bases at ξ (barycentric form) and has no set-up.
   - The ratio is that of the strategy the policy selects at p = 8, timed at p = 8 and at p = 4.
   - Time per point = time per call / (cells × points per cell), with the Gauss rule of degree 2p, at 10^6 and 10^7 points. Newton step = (t(9 iterations) − t(1 iteration)) / 8; set-up = t(1) − one step.
   - Device classes use ADR-0015 item 5's geometric mean; per-device ratios are reported.
2. **Tolerances at high degree** (maintainer, 2026-11-08: Lebesgue-scaled).
   - f64: 10^−12 · max(1, A / A_ref), the same factor (proposed in ADR-0042 beyond the planning answer: the worst-case f64 bound for J on an equispaced hexahedron at p = 8 is about 2 · 10^−12; the factor is at most 1.01 for p ≤ 4).
   - f32: 10^−5 · max(1, A / A_ref). A is an amplification factor computed on the host in f64 at the test case's own points:
     - X: Λ = max Σ_i |ψ_i|;
     - J and derived quantities: Σ_i |∇ψ_i| scaled by the reference cell size; this grows like p² even for GLL;
     - a pull-back through coefficients: max Σ_j |φ_j| Σ_k |C'_jk|, times κ(J) at the point.
   - A_ref is the same factor for the p = 4 equispaced element of that cell type on the same point set.
   - The relative error of each quantity is |have − want| / max |want| over the block (per component), not max(1, |want|), so det J ~ h³ stays relative.
   - The same factor scales the f32 GPU invariance comparisons.
   - ADR-0042's drafting computes A on the host for every (cell, p, family) and lists it. If the pull-back factor makes the f32 bound vacuous for a cell, the ADR says so for that cell.
3. **Degree policy** (maintainer, 2026-11-08: crate-private).
   - `gridcl-geometry/degree-policy.toml` is compiled in. It holds one entry per (`StrategyFamily`, cell type, device class, scalar size) with ascending degree ranges and a strategy. The last range is open-ended and resolves to a strategy known to compile there (`loops` or better).
   - `GRIDCL_GEOM_STRATEGY` is `policy | direct | loops | sumfact | shared | cooperative`, or a comma list `<family>=<strategy>`. It is read once, and an invalid value panics with the variable's name.
   - Precedence: scoped hook, then environment, then TOML. `with_strategy` is thread-local, so thread-rank tests set it inside each rank thread.
   - An inapplicable forced strategy falls back to `loops`, and `record_strategies` shows what ran. Inapplicable means:
     - sum factorisation on a simplex or pyramid, or on a non-tensor point set;
     - `shared` where the data exceed the shared-memory limit.
   - A coverage test checks every family, cell type and class, and that p = 9–10 resolve.
   - The new features `testing` (enabled by gridcl-testing) and `bench` (by gridcl-bench) go in "Cargo features".
4. **`LaunchPlan::per_group`** (above).
   - `units_per_group` is a new public field, and `elementwise` sets it to 1.
   - Reductions over a group's units are pairwise trees over a fixed comptime leaf count L: each unit reduces its leaves pairwise, then the units combine pairwise. The result is the same for every power-of-two `units_per_group` dividing L, including 1.
   - `sync_cube` is called only in uniform control flow, outside `v < num_vectors`.
   - Plans with the family's `KernelKind`.
   - Checked against `max_cube_dim`, `max_units_per_cube`, `max_cube_count` and the shared-memory limit, with usize overflow checks.
   - Listing S2 changes.
5. **Sum factorisation data and the copy list.**
   - A `PointSet` built by a tensor rule (Gauss on quadrilaterals and hexahedra; triangle rule × Gauss on prisms) records its factors crate-privately. User point sets have none.
   - Uploaded once per (element, point set), under CLAUDE.md's clause for tabulations:
     - the 1D (or triangle × 1D) tabulations;
     - the lexicographic-to-gridcl node permutation, packed into the tabulation buffer so no kernel exceeds five buffers.
   - Uploaded once per geometry block, under the "expansion coefficients" clause:
     - the 1D nodes and barycentric weights of the tensor pull-back;
     - the prism's C'_tri factor.
   - CLAUDE.md's copy list and ADR-0039's clause are extended by name.
6. **CUDA attributes and compile time without FFI** (ADR-0041 open point).
   - Registers, spill and local memory per thread come from `ncu` metrics (`launch__registers_per_thread`, local memory) or, failing that, `ptxas -v` of the driver's CUDA version on `CUBECL_CUDA_DUMP_PTX` output. A runner script joins them to the records by kernel name.
   - Maximum threads per block is derived from the register count.
   - Failing driver calls are named by `compute-sanitizer`.
   - First-use compile and load time = the warm-up of a call that launches exactly one new kernel, minus its median, with `CUDA_CACHE_DISABLE=1`.
   - No FFI exception is needed.
7. **Bullet 2's "every kernel variant".**
   - Every (family, cell type, degree, precision, strategy, vector size used by the policy or the invariance tests) compiles, loads and launches on each backend. This is checked from `degree_sweep --launch-only` records per backend; the accuracy tests also force every strategy.
   - The 30 s bound on Locust applies to the variants selected for the Gpu class. The others are reported, and a strategy over 30 s on Locust cannot be selected for the Gpu class.
8. **Vector-size invariance at p = 3–8.**
   - The 10^6 + 3-cell case stays at p ≤ 2 (at p = 8 a hexahedron E-vector of 10^6 cells is about 17 GB in f64).
   - At p = 3–8 the large case is 16k + 3 cells with a Gauss rule of degree 1. The mesh is built outside `testing::with_properties`; the factor and pull-back calls run inside it, with `max_page_size` between the tabulation size and a third of the E-vector, so the E-vector and fields span at least three chunks and several cubes.
   - Cell counts 0, 1, 15, 16 and 17 and tile widths 1 and 16 are unchanged.
9. **Parallel invariance at p = 1–8.** Factors and pull-back results per (cell origin id, point) equal one rank's: bitwise on the CPU runtime (per-cell work does not depend on the partition), within the tolerances on GPUs.
   - Threads: 1–4 ranks at every p; every ghost mode at p ≤ 2, one per case above.
   - MPI target: 1–4 ranks at p ∈ {1, 4, 8}.
10. **Simplices** (ADR-0041 open point). No throughput bound. Collapsed-coordinate sum factorisation is out of scope. Simplices get loops, shared and cooperative, and are reported.
11. **"Than in M9"** (bullet 4).
    - The `direct` strategy stays bitwise M9's kernels (T4 changes only dispatch).
    - Bullet 4 is a paired comparison in one job per target: the policy against `direct`, at p ≤ 4, at most 1.05 by rule 1's class aggregation.
    - T3's baseline, at a pinned commit before any dispatch change, shows only that `direct` did not drift (dispatch overhead) and is reported, not bounded.
12. **Degree sweep protocol and decision rule.**
    - The sweep follows "Benchmarks" (medians, batches, warm-up, conditions), except that vector size, tile and CPU block come from the vector policy. M10 re-studies those for the final kernels.
    - Sizes are 10^6 and 10^7 quadrature points (Gauss degree 2p), plus 10^5 for small-launch behaviour.
    - A configuration exceeding 600 s per timed run on a CPU is recorded "too slow", as compile times over 300 s are.
    - A strategy is adopted over the baseline only if it is at least 10% faster at 10^6 and 10^7 points, and selectable. The baseline is `direct`, or `loops` where `direct` is not selectable (rule 7, or failing to compile, load or launch).
    - Among the strategies within 3% of the fastest, the simplest wins: direct < loops < sumfact < shared < cooperative.
    - Degree ranges are maximal runs of equal choices.
13. **Design text.** ADR-0042 changes:
    - "Geometry" ("Tabulation and quadrature", "Geometric factors", "Pull-back"): the strategies, the tensor pull-back, the policy;
    - Listing S2 (`per_group`, `units_per_group`) and "Cargo features";
    - "Benchmarks" (degree sweep, `high-order.md`);
    - "Testing and validation" (rules 2, 8, 9);
    - roadmap M9h bullets 2 and 3 (rules 1, 7, 11);
    - the copy list in "Handing this design to Claude Code" and CLAUDE.md (rule 5).

## Tasks

Every kernel task writes a SAFETY comment on each `launch_unchecked` and `from_raw_parts`, states each kernel's buffer count (at most five or six), keeps kernels generic over `N: Size` with lane-independent arithmetic masks (frozen pull-back lanes by selects, ADR-0039 item 13), and runs **the kernel-task check set** (M3, M9), extended by strategy:

1. `cargo test -p <crate> -p gridcl`, also with `GRIDCL_VECTOR_SIZE=scalar`, `=widest`, `GRIDCL_QUAD_TILE=1` and `GRIDCL_GEOM_STRATEGY=<the task's strategy>`;
2. `cargo test --release -p <crate> -p gridcl` (includes the p ≥ 5 cases, release-only from T0b);
3. Metal, outside the sandbox: `GRIDCL_TEST_DEVICE=metal cargo test --release -p <crate> -p gridcl --features gridcl-testing/metal`;
4. Locust after the commit: `scripts/hpc/remote-test.sh locust -- 'env GRIDCL_SKIP_PYTHON=1 CUDA_CACHE_DISABLE=1 cargo test --release -p <crate> -p gridcl --features gridcl-testing/cuda'` (A4000 fallback, ADR-0037). Record the job id in the Deviations log at the time.

Local cargo runs go one at a time. No edits while `remote-test.sh` syncs. Benchmarks follow "Benchmarks" (conditions; node load idle before and after on Locust). On the Mac they run outside the sandbox.

- [ ] **T0a Test inputs at p ≤ 8 (gridcl-testing)**
  - Goal: the acceptance inputs, implemented:
    - `HostMesh::with_node_family(NodeFamily::Gll)` for tensor cells;
    - generators validated to p = 8 (det J > 0 at Gauss points of degree 2p for `perturbed`, `ball`, `sphere`, `mixed_surface`, every box);
    - degree lists for `parallel` case builders (pyramid families dropped above p = 3);
    - `gridcl_testing::reference`: f64 host X, J, det J, K, normals and facet normals from `read_node` coordinates and gridcl-refcell tabulation, the amplification factors of rule 2, and a pull-back reference.
  - Crates/files: `gridcl-testing/src/{meshes/mod.rs, meshes/family.rs, reference.rs, parallel.rs, lib.rs}`.
  - Public API added or changed: none (gridcl-testing is unpublished).
  - Tests: GLL meshes reproduce their equispaced geometry to 10^−12; det J > 0 for every generator at p = 1–8; `reference` agrees with M9's device factors at p = 2.
  - Verify: `cargo test -p gridcl-testing`
  - After: ADR-0042 accepted and applied
  - Parallel with: –
  - Risks / notes: if `perturbed` tangles a cell above p = 4, scale its amplitude by 4/p there and log it.
- [ ] **T0b Acceptance tests (failing)**
  - Goal: every M9h bullet as an automated test, with stubs.
  - Crates/files:
    - `gridcl/tests/{high_order.rs, high_order_invariance.rs, parallel_threads.rs, main.rs, common/mod.rs}` (`for_each_strategy!`, rule-2 tolerances, p ≥ 5 release-only);
    - `gridcl-mpi-tests/{Cargo.toml, tests/high_order_parallel.rs}`;
    - `gridcl-bench/tests/high_order_record.rs`.
  - Stubs (`todo!()`):
    - gridcl-geometry `strategy.rs` and `testing` (feature `testing`, enabled by gridcl-testing);
    - `LaunchPlan::per_group` and the `units_per_group` field (set to 1 by `elementwise`);
    - the gridcl-bench `m9h-*` record loader.
  - Verify: `cargo test --workspace --no-run`; the new tests fail with `not yet implemented`, or with "no m9h data" in `high_order_record.rs`; existing tests unchanged.
  - After: T0a
  - Parallel with: –
  - Risks / notes: no `todo!()` on a launch path existing tests use. Dispatch is not yet wired, so `direct` is the only path.
- [ ] **T1 Degree axis in gridcl-bench (gridcl-bench, gridcl-geometry)**
  - Goal:
    - a `degree_sweep` binary (rule 12): families EVector, Points, Jacobian, det J, pseudo-inverse, Normals, FacetNormals, PullBack (set-up and step, rule 1) and BasisEval (hidden `bench` hook);
    - every cell type, p = 1–8 (pyramids 1–3), both node families, f32/f64, `--strategies`, `--launch-only`;
    - per-kernel first-use compile and load time (rule 6);
    - JSON `m9h-degree-<backend>-<target>.json` with degree, cell, node family and strategy, read by `load_dir`.
  - Crates/files: `gridcl-bench/src/{families/degree.rs, families/mod.rs, record.rs, bin/degree_sweep.rs}`; `gridcl-geometry/{Cargo.toml, src/bench.rs, src/lib.rs}`.
  - Public API added or changed: hidden `gridcl_geometry::bench` (feature `bench`).
  - Tests: smoke run at p = 1–2 and 10^4 points (CPU); records round-trip; a second identical call's compile time is near zero.
  - Verify: `cargo test -p gridcl-bench`; `cargo run --release -p gridcl-bench --bin degree_sweep -- --degrees 1,2 --sizes 10000 --out $TMPDIR/x.json`
  - After: T0b
  - Parallel with: –
  - Risks / notes: until T4 only `direct` exists. Estimated sweep times are recorded at T3 and used to split runs by family.
- [ ] **T2 CUDA resources and C18 (scripts, docs)**
  - Goal:
    - `scripts/bench/cuda-resources.sh` (rule 6): runs a sweep under `ncu` (or with `CUBECL_CUDA_DUMP_PTX` and a matching `ptxas -v`), and writes `m9h-cuda-resources-locust.json`;
    - isolate C18 on Locust with M9 T10a's first evaluator (commit 81e8739 on a scratch branch, never merged; each run under a 600 s timeout) and the current direct kernels at hexahedron p = 4–8: registers, local memory, derived max threads against cube_dim 256, the failing call by `compute-sanitizer`;
    - correct C18 (LLVM path) and record the cause.
  - Crates/files: `scripts/bench/cuda-resources.sh`, `docs/cubecl-upstream.md` (C18), `docs/dev-environment.md` (ncu, ptxas and compute-sanitizer on Locust).
  - Public API added or changed: none.
  - Tests: the script on a p = 1 sweep gives one record per launched kernel.
  - Verify: `scripts/hpc/remote-test.sh locust -- 'scripts/bench/cuda-resources.sh --degrees 1'`
  - After: T1
  - Parallel with: –
  - Risks / notes: filing the upstream report is outward-facing, so ask the maintainer first (row 13), then set C18's status.
- [ ] **T3 M9 baseline sweep (data)**
  - Goal: rule 11's drift baseline. `direct` at p = 1–8 at the T2 commit (recorded), on every M3s target: local Mac CPU and Metal, Locust Grace and H100, one Kathleen socket (Myriad D fallback). Failures and "too slow" are recorded, not retried. Includes an f32 error survey (error / rule-2 tolerance per cell, p, factor and pull-back).
  - Crates/files: `docs/perf/data/m9h-baseline-*.json`, `docs/perf/high-order.md` (Protocol, Conditions, Baseline, run times per target).
  - Public API added or changed: none.
  - Tests: `high_order_record.rs` loads the baseline (bullet checks still fail).
  - Verify: `cargo test -p gridcl-bench --test high_order_record -- baseline`
  - After: T2
  - Parallel with: –
  - Risks / notes: no code edits during the remote runs (the tree must stay at the recorded commit). If Locust stays busy, the run is repeated; A4000 timings only when idle, with the reason recorded.
- [ ] **T4 Strategy selection (gridcl-geometry)**
  - Goal: rule 3, with `strategy.rs` bodies and `degree-policy.toml` (all `direct`, open-ended last range):
    - environment override, hook and `record_strategies`;
    - dispatch at every launcher (`launch_factor`, `EVector::build`, `pull_back_block`, the basis-evaluator test kernel); `direct` is M9's kernels unchanged.
  - Crates/files: `gridcl-geometry/{degree-policy.toml, src/strategy.rs, src/testing.rs, src/layout.rs, src/evector.rs, src/pull_back.rs, src/lib.rs}`.
  - Public API added or changed: hidden `gridcl_geometry::testing` (rule 3).
  - Tests: `degree_policy_covers_every_family_and_class` (with p = 9–10); invalid TOML and environment values rejected; `record_strategies` shows `direct`; inapplicable strategies error until T6a, then fall back to `loops`.
  - Verify: kernel-task check set with `-p gridcl-geometry`
  - After: T3
  - Parallel with: T5
  - Risks / notes: results bitwise as before.
- [ ] **T5 `LaunchPlan::per_group` (gridcl-device)**
  - Goal: rule 4's constructor, field and checks.
  - Crates/files: `gridcl-device/src/{plan.rs, testing.rs}`.
  - Public API added or changed: `LaunchPlan::per_group`, `LaunchPlan::units_per_group`.
  - Tests:
    - plans against injected limits (`with_properties`), units 1 and non-powers of two clamped down, overflow, the shared-memory limit, block boundaries;
    - a test kernel with rule 4's tree over L = 64 leaves is bitwise equal at units 1, 2 and min(64, limit) on the CPU runtime, and within tolerance on GPUs.
  - Verify: kernel-task check set with `-p gridcl-device`
  - After: T3
  - Parallel with: T4, T6a, T6b
  - Risks / notes: `sync_cube` on the CPU runtime (every unit a thread); a CubeCL problem gets a `C<n>` entry.
- [ ] **T6a Runtime loops: EVector and factor kernels (gridcl-geometry)**
  - Goal: `loops`. Node loops become runtime loops, with nodes and points as runtime arguments, so one kernel serves every degree of a (tdim, gdim, factor, tile). Buffers: three (EVector), five (factors).
  - Crates/files: `gridcl-geometry/src/{layout_loops.rs, evector.rs, layout.rs}`.
  - Public API added or changed: none.
  - Tests: `high_order::factors_match_host_reference` under `loops` at p = 1–8; the geometry and gridcl suites under `GRIDCL_GEOM_STRATEGY=loops`; vector-size invariance bitwise on the CPU runtime, N = 1 included.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=loops`
  - After: T4
  - Parallel with: T5, T6b
  - Risks / notes: runtime zero and one as in `basis_eval` (C16).
- [ ] **T6b Runtime loops: basis evaluator and pull-back (gridcl-geometry)**
  - Goal: `loops` for the evaluator (recurrence coefficients from runtime integers, 1D values in small local arrays, runtime loops over basis functions) and for the pull-back set-up and step. Buffers: five; if the set-up needs global scratch for a, it travels as a trailer of an existing buffer or the kernel binds six.
  - Crates/files: `gridcl-geometry/src/{basis_eval/loops.rs, basis_eval/mod.rs, pull_back.rs}`.
  - Public API added or changed: none.
  - Tests: `basis_eval` tests at p = 1–8 (pyramids 1–3) under `loops`; `high_order::pull_back_recovers_reference_points` under `loops`; vector-size invariance.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=loops`
  - After: T4
  - Parallel with: T5, T6a
  - Risks / notes: the local array a (nb·gd) is the likely C18 cause. Choose its placement from T2's register and local-memory data.
- [ ] **T7a Sum factorisation I: tensor data, points and Jacobians (gridcl-geometry)**
  - Goal: rule 5's tensor point-set factors, 1D tabulations and packed permutation; a `sumfact` kernel for X and J on quadrilaterals and hexahedra (all (TD, GD) pairs), contracting one dimension at a time per cell. Buffers: five.
  - Crates/files: `gridcl-geometry/src/{point_set.rs, rule.rs, sumfact/{mod.rs, tensor.rs}, layout.rs}`.
  - Public API added or changed: none.
  - Tests: under `sumfact`, points and Jacobians match the host reference for quadrilaterals and hexahedra at p = 1–8, both families; a user `PointSet` falls back to `loops`; vector-size and tile invariance bitwise on the CPU runtime.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=sumfact`
  - After: T6a
  - Parallel with: T6b
  - Risks / notes: per-lane intermediates of (p + 1)^(d−1) · q values at p = 8 may spill. T9's shared and T10a's cooperative variants address that.
- [ ] **T7b Sum factorisation II: derived factors and facet normals (gridcl-geometry)**
  - Goal: det J, pseudo-inverse and normals on the sum-factorised J; facet normals per face where the facet point set is tensor (`facet_rule.rs` records its factors).
  - Crates/files: `gridcl-geometry/src/{sumfact/tensor.rs, facet_rule.rs, layout.rs}`.
  - Public API added or changed: none.
  - Tests: `factors_match_host_reference` complete for quadrilaterals and hexahedra under `sumfact`; invariance as T7a.
  - Verify: as T7a
  - After: T7a
  - Parallel with: T6b
  - Risks / notes: –
- [ ] **T7c Sum factorisation III: prisms (gridcl-geometry)**
  - Goal: triangle × interval partial factorisation for every prism factor (Finding 3).
  - Crates/files: `gridcl-geometry/src/{sumfact/prism.rs, sumfact/mod.rs, point_set.rs}`.
  - Public API added or changed: none.
  - Tests: as T7a and T7b for prisms.
  - Verify: as T7a
  - After: T7b
  - Parallel with: –
  - Risks / notes: –
- [ ] **T8 Tensor pull-back (gridcl-geometry)**
  - Goal: `sumfact` for the pull-back:
    - quadrilaterals and hexahedra evaluate the 1D Lagrange bases and derivatives at ξ in barycentric form and contract the E-vector one dimension at a time (no set-up, no C');
    - prisms use C'_tri (Dubiner) times the 1D Lagrange interval basis.
  - Buffers: five (the 1D data trail the coefficient buffer, rule 5).
  - Crates/files: `gridcl-geometry/src/{sumfact/pull_back.rs, sumfact/mod.rs, evector.rs}` (coefficient cache).
  - Public API added or changed: none.
  - Tests: `pull_back_recovers_reference_points` under `sumfact` for quadrilaterals, hexahedra and prisms at p = 1–8, both families; agrees with `loops` within tolerance; vector-size invariance bitwise on the CPU runtime.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=sumfact`
  - After: T6b, T7c
  - Parallel with: –
  - Risks / notes: barycentric evaluation at ξ equal to a node needs the standard exact-node branch as an arithmetic select (no lane branches).
- [ ] **T9 Shared memory (gridcl-geometry)**
  - Goal: `shared`. Factors stage the tabulation (or the 1D tabulation) in shared memory per cube; the pull-back stages C' or its factors. Planned by `per_group` with `units_per_group = 1`, for its shared-memory check; the cube's units load cooperatively behind `sync_cube` in uniform control flow.
  - Crates/files: `gridcl-geometry/src/{shared.rs, layout.rs, pull_back.rs}`.
  - Public API added or changed: none.
  - Tests: under `shared`, both accuracy tests at p = 1–8; data over the limit fall back to `loops` (`record_strategies`); vector-size invariance, N = 1 included.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=shared`
  - After: T5, T8
  - Parallel with: –
  - Risks / notes: shared memory on the CPU runtime is L1-sized; Metal's limit is 32 KiB.
- [ ] **T10a Cooperative factors (gridcl-geometry)**
  - Goal: `cooperative` for factors. Units of a group split the nodes (or the sum-factorised contractions) of a cell or tile and combine them by rule 4's trees.
  - Crates/files: `gridcl-geometry/src/{cooperative/{mod.rs, factors.rs}, layout.rs}`.
  - Public API added or changed: none.
  - Tests: under `cooperative`, `factors_match_host_reference` at p = 1–8; bitwise across `units_per_group` ∈ {1, 2, L} and across vector sizes on the CPU runtime.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=cooperative`
  - After: T9
  - Parallel with: –
  - Risks / notes: –
- [ ] **T10b Cooperative pull-back (gridcl-geometry)**
  - Goal: `cooperative` for the pull-back: the units of a group share one cell's set-up and split each step's contraction, with lanes frozen by selects.
  - Crates/files: `gridcl-geometry/src/{cooperative/{mod.rs, pull_back.rs}, pull_back.rs}`.
  - Public API added or changed: none.
  - Tests: as T10a for `pull_back_recovers_reference_points`.
  - Verify: as T10a
  - After: T10a
  - Parallel with: –
  - Risks / notes: every unit runs all `max_iterations` (no early exit), so `sync_cube` stays uniform.
- [ ] **T11a Strategy sweeps (data)**
  - Goal: `degree_sweep` over every strategy on every M3s target, including the paired policy-vs-`direct` runs of rule 11 (re-run after T11b's policy) and `--launch-only` on every backend; `cuda-resources.sh` on Locust.
  - Crates/files: `docs/perf/data/m9h-degree-*.json`, `m9h-launch-*.json`, `m9h-cuda-resources-locust.json`.
  - Public API added or changed: none.
  - Tests: `high_order_record.rs::{compile_and_load_within_30_s_on_locust, every_variant_launched}` pass.
  - Verify: `cargo test -p gridcl-bench --test high_order_record`
  - After: T10b
  - Parallel with: –
  - Risks / notes: run in parts by family. Budget per target from T3's run times. CPU runs on Kathleen use one socket under `numactl` (ADR-0020).
- [ ] **T11b Decision rule, policy and report (gridcl-bench, gridcl-geometry, docs)**
  - Goal: `decide_degree` (rule 12) and a `high_order_report` binary that writes `degree-policy.toml` and the generated tables of `docs/perf/high-order.md`, with `--check`. The tables cover cost per strategy, family, cell type, device class and degree range, the choice and the reason, compile times, and CUDA registers and local memory.
  - Crates/files: `gridcl-bench/src/{decide_degree.rs, report.rs, bin/high_order_report.rs}`, `gridcl-geometry/degree-policy.toml`, `docs/perf/high-order.md`.
  - Public API added or changed: none.
  - Tests: `high_order_record.rs::{tensor_cost_ratio_p8_over_p4, no_family_slower_than_m9_at_low_degree, report_covers_every_family_class_and_range, degree_policy_follows_the_decision_rule}` pass (after T11a's paired re-run).
  - Verify: `cargo test -p gridcl-bench`; `cargo run -p gridcl-bench --bin high_order_report -- --check`
  - After: T11a
  - Parallel with: –
  - Risks / notes: if a bound fails on a class, stop and draft an ADR; never a silent policy edit.
- [ ] **T12 Acceptance on the chosen policy; CI budget**
  - Goal: the whole suite under the policy on the local Mac, hosted CI and Locust; new modules placed in shards 2 and 3 of `scripts/ci-gridcl-shard.sh` (and vulkan's groups) to stay under 20 min.
  - Crates/files: `scripts/ci-gridcl-shard.sh`, `.github/workflows/ci.yml` (only if a shard must be added), `docs/dev-environment.md`.
  - Public API added or changed: none.
  - Tests: –
  - Verify: hosted CI job times on the push.
  - After: T11b
  - Parallel with: –
  - Risks / notes: –
- [ ] **T13 Final runs**
  - Goal: acceptance on every target; results in Review.
  - Runs:
    - local Mac: CPU f32/f64 and Metal f32 workspace suites; `scripts/mpi-test.sh 1` (CPU) and `GRIDCL_TEST_DEVICE=metal scripts/mpi-test.sh 1 2 3 4`;
    - hosted CI;
    - Locust: CUDA workspace (`CUDA_CACHE_DISABLE=1`), `--gpus 1 --ranks 2 -- scripts/mpi-test.sh 1 2`, and the aarch64 CPU-runtime run (`--test-threads=1`, `--time 06:00:00`, as at M9's review);
    - Kathleen two-node review (ADR-0016): `scripts/mpi-test.sh 2 4 8 16`, as in M9 T12.
  - Crates/files: `docs/plans/M9h.md`.
  - Public API added or changed: none.
  - Tests: –
  - Verify: the runs above pass.
  - After: T12
  - Parallel with: –
  - Risks / notes: –

## Acceptance coverage

| Acceptance criterion (docs/design.md) | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| Every cell type but pyramids at p = 1–8, both node families: X, J, det J, K, normals and facet normals match the host reference to 10^−12 / 10^−5, and the pull-back recovers reference points; pyramids p ≤ 3 | `gridcl/tests/high_order.rs::{factors_match_host_reference, pull_back_recovers_reference_points}`: every (TD, GD) pair, `gridcl_testing::reference`, rule-2 tolerances, under the policy and every strategy (`for_each_strategy!`); p ≥ 5 release-only | T0a, T0b, T4, T6a–T10b | Local Mac: CPU f32/f64, Metal f32; CI: cpu, vulkan; Locust: CUDA f32/f64 |
| Every kernel variant up to p = 8 compiles, loads and launches on each backend; first-use compile and load ≤ 30 s on Locust (rule 7) | `gridcl-bench/tests/high_order_record.rs::{every_variant_launched, compile_and_load_within_30_s_on_locust}` (`degree_sweep --launch-only` records; `CUDA_CACHE_DISABLE=1`) | T0b, T1, T2, T11a | launch: Local Mac CPU and Metal, Locust CPU and CUDA, Kathleen CPU; 30 s: Locust CUDA |
| Quadrilaterals and hexahedra: factors and a pull-back Newton step at p = 8 at most 3 times p = 4 per point on every device class (read as rule 1) | `high_order_record.rs::tensor_cost_ratio_p8_over_p4` | T0b, T7a–T8, T11a, T11b | CpuSimd128 (Mac mini, Locust Grace), CpuSimd512 (one Kathleen socket), Gpu (Metal, H100) |
| No family more than 5% slower than in M9 at p ≤ 4 on any device class (rule 11) | `high_order_record.rs::no_family_slower_than_m9_at_low_degree` (paired policy vs `direct`; drift against `m9h-baseline-*` reported) | T0b, T3, T11a, T11b | as above |
| Determinism, vector-size invariance and parallel invariance at p = 1–8 | `gridcl/tests/high_order_invariance.rs::{vector_size_invariance, tile_invariance, two_runs_are_bitwise_identical}` (rule 8); `parallel_threads.rs::high_order_parallel_invariance` (rule 9); `gridcl-mpi-tests::high_order_parallel` | T0a, T0b, T6a–T12 | Local Mac: CPU (bitwise), Metal f32; CI: cpu, vulkan, mpi-mpich, mpi-openmpi; Locust: CUDA f32/f64, MPI 1–2; Kathleen review |
| `docs/perf/high-order.md` lists every family, device class and degree range with each strategy's cost, the choice and the reason; JSON under `docs/perf/data/` from the M3s targets | `high_order_record.rs::{report_covers_every_family_class_and_range, degree_policy_follows_the_decision_rule}`; `high_order_report --check` | T3, T11a, T11b | data: Mac mini (CPU, Metal), Locust (Grace, H100), one Kathleen socket (Myriad D fallback) |

**Not acceptance bullets:**

| Requirement | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| C18 closed (cause isolated; upstream report if CubeCL's) | C18 entry; `m9h-cuda-resources-locust.json` | T2 | Locust CUDA |
| CUDA registers and local memory per thread | `cuda-resources.sh` records, tables in `high-order.md` | T2, T11a, T11b | Locust CUDA |
| Degree policy coverage and override parsing | gridcl-geometry `degree_policy_covers_every_family_and_class`, `override_parsing` | T4 | Local Mac CPU; CI cpu |
| `per_group` limits and fixed-order reductions | gridcl-device plan tests | T5 | Local Mac: CPU, Metal; CI; Locust |
| Device basis evaluator at p = 1–8 | `basis_eval/tests.rs` extended | T6b | Local Mac: CPU, Metal; CI; Locust |

## Open questions and ADR candidates

| # | Question or conflict | Design section | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| 1 | A pull-back Newton step at an arbitrary point grows as (9/5)^d from p = 4 to 8, so bullet 3's pull-back clause cannot hold (hexahedra 5.8, quadrilaterals 3.2). | Roadmap M9h, bullet 3 | Rule 1. | answered (maintainer, 2026-11-08: amend via ADR); ADR-0042 |
| 2 | f32 cannot meet 10^−5 at p = 8 on equispaced nodes; the pull-back amplifies through C'. | Testing and validation; ADR-0041 open point | Rule 2. | answered (maintainer, 2026-11-08: Lebesgue-scaled; pull-back factor and per-block relative error added after design review); ADR-0042 |
| 3 | Degree policy: public item or not; override. | Listing S2; ADR-0041 open point | Rule 3. | answered (maintainer, 2026-11-08: crate-private); ADR-0042 |
| 4 | The cooperative strategy needs a launch shape and group size that `elementwise` cannot give. | Listing S2; CLAUDE.md | Rule 4: `per_group`, `units_per_group`. | answered (maintainer, 2026-11-08: plan up front); ADR-0042 |
| 5 | Sum factorisation and the tensor pull-back upload data the copy list does not name. | CLAUDE.md copy list; Tabulation and quadrature | Rule 5. | ADR-0042 (Proposed) |
| 6 | Reading CUDA kernel attributes within the FFI rules. | CLAUDE.md FFI; ADR-0041 open point | Rule 6: ncu/ptxas, compute-sanitizer; no FFI. | ADR-0042 (Proposed) |
| 7 | "Every kernel variant" in bullet 2 across strategies. | Roadmap M9h, bullet 2 | Rule 7. | ADR-0042 (Proposed) |
| 8 | The 10^6 + 3-cell vector-size case does not fit at p = 8. | Testing and validation | Rule 8. | ADR-0042 (Proposed) |
| 9 | Parallel invariance at p = 1–8: quantities, ghost modes, MPI cost. | Testing and validation | Rule 9. | ADR-0042 (Proposed) |
| 10 | Throughput target for simplices. | ADR-0041 open point | Rule 10: none; reported. | ADR-0042 (Proposed) |
| 11 | "Than in M9": which data; noise of a 5% bound across weeks on open nodes. | Roadmap M9h, bullet 4; ADR-0034 | Rule 11: paired comparison against bitwise-M9 `direct`. | ADR-0042 (Proposed) |
| 12 | The degree sweep uses the vector policy's sizes, tiles and blocks rather than sweeping them; decision rule for strategies. | Benchmarks | Rule 12. | ADR-0042 (Proposed) |
| 13 | The upstream C18 report is outward-facing. | CLAUDE.md (cubecl-upstream) | Ask the maintainer at T2 before filing. | open |
| 14 | M9 question 22 (`set_vertex_sharers` wording) waits for an ADR touching "Ownership of vertices and entities". | Ownership of vertices and entities | Not touched by ADR-0042. | carried |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |

## Review

<Filled by /review-milestone.>
