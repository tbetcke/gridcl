# M9h High-order performance — plan

- **Status:** approved
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
  - ADR-0042 (accepted 2026-11-08, applied in `adr: 0042 apply`): [M9h public items and rules](../adr/0042-m9h-public-items-and-rules-the-design-leaves-open.md), items 1–13 = the plan's rules 1–13 and questions 1–12.

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

## Planned public API and rules (ADR-0042, accepted: "M9h public items and rules the design leaves open")

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
   - f64: 10^−12 · max(1, A / A_ref), the same factor (ADR-0042, accepted beyond the planning answer: the worst-case f64 bound for J on an equispaced hexahedron at p = 8 is about 2 · 10^−12; the factor is at most 1.01 for p ≤ 4).
   - f32: 10^−5 · max(1, A / A_ref). A is an amplification factor computed on the host in f64 at the test case's own points:
     - X: Λ = max Σ_i |ψ_i|;
     - J and derived quantities: Σ_i |∇ψ_i| scaled by the reference cell size; this grows like p² even for GLL;
     - a pull-back through coefficients: max Σ_j |φ_j| Σ_k |C'_jk|, times κ(J) at the point.
   - A_ref is the same factor for the p = 4 equispaced element of that cell type on the same point set.
   - The relative error of each quantity is |have − want| / max |want| over the block (per component), not max(1, |want|), so det J ~ h³ stays relative.
   - The same factor scales the f32 GPU invariance comparisons.
   - ADR-0046: for J, the quantities derived from it and the pull-back, the factor gains max(1, n / n_ref) (nodes of the element against the p = 4 equispaced element; p = 3 for pyramids) and max(1, R) (the largest |x − o| over the block's nodes divided by the block's largest entry of J).
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
   - At p = 3–8 the large case is 16k + 3 cells with a Gauss rule of degree 1. The mesh is built outside `testing::with_properties`; the factor and pull-back calls run inside it, with `max_page_size` between the tabulation size and a third of the E-vector, so the E-vector and fields span at least three chunks and several cubes. ADR-0046: the limit is also at least the largest per-block table the calls upload (C′); the E-vector spans three chunks at every degree, J wherever that limit allows, and the test names the exceptions.
   - Cell counts 0, 1, 15, 16 and 17 and tile widths 1 and 16 are unchanged.
9. **Parallel invariance at p = 1–8.** Factors and pull-back results per (cell origin id, point) equal one rank's: bitwise on the CPU runtime (per-cell work does not depend on the partition), within the tolerances on GPUs.
   - Threads: 1–4 ranks at every p; every ghost mode at p ≤ 2, one per case above.
   - MPI target: 1–4 ranks at p ∈ {1, 4, 8}.
10. **Simplices** (ADR-0041 open point). No throughput bound. Collapsed-coordinate sum factorisation is out of scope. Simplices get loops, shared and cooperative, and are reported.
11. **"Than in M9"** (bullet 4).
    - The `direct` strategy stays bitwise M9's kernels (T4 changes only dispatch), except that its pull-back clamps iterates to [−0.1, 1.1]^TD (ADR-0046) and so differs where an iterate left that box.
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

- [x] **T0a Test inputs at p ≤ 8 (gridcl-testing)**
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
- [x] **T0b Acceptance tests (failing)**
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
- [x] **T1 Degree axis in gridcl-bench (gridcl-bench, gridcl-geometry)**
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
- [x] **T2 CUDA resources and C18 (scripts, docs)**
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
- [x] **T3 M9 baseline sweep (data)**
  - Goal: rule 11's drift baseline. `direct` at p = 1–8 at the T2 commit (recorded), on every M3s target: local Mac CPU and Metal, Locust Grace and H100, one Kathleen socket (Myriad D fallback); on Locust, the pull-back families and the Grace CPU at p ∈ {1, 2, 4, 8} (pyramids {1, 2}; ADR-0043). Failures and "too slow" are recorded, not retried. Includes an f32 error survey (error / rule-2 tolerance per cell, p, factor and pull-back).
  - Crates/files: `docs/perf/data/m9h-baseline-*.json`, `docs/perf/high-order.md` (Protocol, Conditions, Baseline, run times per target).
  - Public API added or changed: none.
  - Tests: `high_order_record.rs` loads the baseline (bullet checks still fail).
  - Verify: `cargo test -p gridcl-bench --test high_order_record -- baseline`
  - After: T2
  - Parallel with: –
  - Risks / notes: no code edits during the remote runs (the tree must stay at the recorded commit). If Locust stays busy, the run is repeated; A4000 timings only when idle, with the reason recorded.
- [x] **T4 Strategy selection (gridcl-geometry)**
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
- [x] **T5 `LaunchPlan::per_group` (gridcl-device)**
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
- [x] **T6a Runtime loops: EVector and factor kernels (gridcl-geometry)**
  - Goal: `loops`. Node loops become runtime loops, with nodes and points as runtime arguments, so one kernel serves every degree of a (tdim, gdim, factor, tile). Buffers: three (EVector), five (factors).
  - Crates/files: `gridcl-geometry/src/{layout_loops.rs, evector.rs, layout.rs}`.
  - Public API added or changed: none.
  - Tests: `high_order::factors_match_host_reference` under `loops` at p = 1–8; the geometry and gridcl suites under `GRIDCL_GEOM_STRATEGY=loops`; vector-size invariance bitwise on the CPU runtime, N = 1 included.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=loops`
  - After: T4
  - Parallel with: T5, T6b
  - Risks / notes: runtime zero and one as in `basis_eval` (C16). First, ADR-0044's test support: K's and the pull-back's scales in `gridcl_testing::reference` with κ_F(J) / TD and its callers (the ADR lists them), and `try_clone` outside `with_properties` in `vector_size_invariance`'s large case (T4's check set).
- [x] **T6b Runtime loops: basis evaluator and pull-back (gridcl-geometry)**
  - Goal: `loops` for the evaluator (recurrence coefficients from runtime integers, 1D values in small local arrays, runtime loops over basis functions) and for the pull-back set-up and step. Buffers: five; if the set-up needs global scratch for a, it travels as a trailer of an existing buffer or the kernel binds six.
  - Crates/files: `gridcl-geometry/src/{basis_eval/loops.rs, basis_eval/mod.rs, pull_back.rs}`.
  - Public API added or changed: none.
  - Tests: `basis_eval` tests at p = 1–8 (pyramids 1–3) under `loops`; `high_order::pull_back_recovers_reference_points` under `loops`; vector-size invariance.
  - Verify: kernel-task check set with `-p gridcl-geometry`, `GRIDCL_GEOM_STRATEGY=loops`
  - After: T4
  - Parallel with: T5, T6a
  - Risks / notes: the local array a (nb·gd) is the likely C18 cause. Choose its placement from T2's register and local-memory data. First, ADR-0046: the clamp of the iterates to [−0.1, 1.1]^TD in `direct`'s pull-back and in `loops`', and its test support (the n / n_ref and R factors in `gridcl_testing::reference` and the callers ADR-0044 lists; the large case's page limit above C′).
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
  - Risks / notes: shared memory on the CPU runtime is L1-sized; Metal's limit is 32 KiB. First, ADR-0045: CPU cubes of `per_group` hold `units_per_group` × ⌊U / `units_per_group`⌋ units, U = `elementwise`'s CPU units (`gridcl-device/src/plan.rs` and its rustdoc, `cpu_cubes_visit_blocks_of_consecutive_groups`, the property and tree-kernel tests of `gridcl-device/tests/launch_plan.rs`); GPU plans are unchanged.
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
| 1 | A pull-back Newton step at an arbitrary point grows as (9/5)^d from p = 4 to 8, so bullet 3's pull-back clause cannot hold (hexahedra 5.8, quadrilaterals 3.2). | Roadmap M9h, bullet 3 | Rule 1. | answered (maintainer, 2026-11-08: amend via ADR); ADR-0042 (accepted, applied) |
| 2 | f32 cannot meet 10^−5 at p = 8 on equispaced nodes; the pull-back amplifies through C'. | Testing and validation; ADR-0041 open point | Rule 2. | answered (maintainer, 2026-11-08: Lebesgue-scaled; pull-back factor and per-block relative error added after design review); ADR-0042 (accepted, applied) |
| 3 | Degree policy: public item or not; override. | Listing S2; ADR-0041 open point | Rule 3. | answered (maintainer, 2026-11-08: crate-private); ADR-0042 (accepted, applied) |
| 4 | The cooperative strategy needs a launch shape and group size that `elementwise` cannot give. | Listing S2; CLAUDE.md | Rule 4: `per_group`, `units_per_group`. | answered (maintainer, 2026-11-08: plan up front); ADR-0042 (accepted, applied) |
| 5 | Sum factorisation and the tensor pull-back upload data the copy list does not name. | CLAUDE.md copy list; Tabulation and quadrature | Rule 5. | ADR-0042 (accepted, applied) |
| 6 | Reading CUDA kernel attributes within the FFI rules. | CLAUDE.md FFI; ADR-0041 open point | Rule 6: ncu/ptxas, compute-sanitizer; no FFI. | ADR-0042 (accepted, applied) |
| 7 | "Every kernel variant" in bullet 2 across strategies. | Roadmap M9h, bullet 2 | Rule 7. | ADR-0042 (accepted, applied) |
| 8 | The 10^6 + 3-cell vector-size case does not fit at p = 8. | Testing and validation | Rule 8. | ADR-0042 (accepted, applied) |
| 9 | Parallel invariance at p = 1–8: quantities, ghost modes, MPI cost. | Testing and validation | Rule 9. | ADR-0042 (accepted, applied) |
| 10 | Throughput target for simplices. | ADR-0041 open point | Rule 10: none; reported. | ADR-0042 (accepted, applied) |
| 11 | "Than in M9": which data; noise of a 5% bound across weeks on open nodes. | Roadmap M9h, bullet 4; ADR-0034 | Rule 11: paired comparison against bitwise-M9 `direct`. | ADR-0042 (accepted, applied) |
| 12 | The degree sweep uses the vector policy's sizes, tiles and blocks rather than sweeping them; decision rule for strategies. | Benchmarks | Rule 12. | ADR-0042 (accepted, applied) |
| 13 | The upstream C18 report is outward-facing. | CLAUDE.md (cubecl-upstream) | Ask the maintainer at T2 before filing. | answered (procedure: the maintainer decides at T2) |
| 14 | M9 question 22 (`set_vertex_sharers` wording) waits for an ADR touching "Ownership of vertices and entities". | Ownership of vertices and entities | Not touched by ADR-0042. | carried |
| 15 | The Locust baseline at p = 1–8 takes 9–12 h more; the host mesh build at 10^7 points dominates on Grace. | ADR-0042, item 11; T3 | Sample p ∈ {1, 2, 4, 8} for Locust's pull-back families and Grace CPU. | ADR-0043 (accepted, applied) |
| 16 | f32 K and ξ on the ball fail rule 2 from p = 3 (1.3e-5, 1.5e-5): J's error times κ(J), which grows without bound on the ball's singular edges, and rule 2 leaves κ out for K and cancels it for the pull-back. | Testing and validation, Tolerances; ADR-0042, item 2 | Scale K and ξ by the largest κ_F(J) / TD over the block (amended from pointwise at implementation); A_ref without κ. | ADR-0044 (accepted, applied, amended) |
| 17 | On CPUs a `per_group` plan runs `units_per_group` threads (`cube_dim` = `units_per_group`, ADR-0042 item 4, while each CubeCL CPU unit is one thread visiting every cube), so T9's plans (`units_per_group` = 1) run on one thread and T10's on at most the group size. | Launch plans and the vector policy, Listing S2; ADR-0042, item 4 | CPU cubes hold `units_per_group` × ⌊U / `units_per_group`⌋ units, U = `elementwise`'s CPU units; done as T9's first step. | ADR-0045 (accepted, applied) |
| 18 | Release runs reach p ≥ 5 for the first time (T6a): the f32 pull-back converges to preimages outside the cell at p = 8 (ball, both families); J's f32 error grows with the number of nodes, which A does not measure (GLL hexahedra from p = 6); the large invariance case's coordinates are 13–65 times the cell size; and the pull-back's C′ exceeds the large case's page limit from hexahedron p = 6. | Pull-back; Testing and validation, Tolerances; ADR-0042, items 2, 8 and 11; ADR-0044 | Clamp Newton iterates to [−0.1, 1.1]^TD; scale J-derived quantities and ξ by n / n_ref and R; page limit above the per-block tables. | ADR-0046 (accepted, applied) |
| 19 | A better initial guess for the pull-back (maintainer, 2026-11-11, after a comparison with MFEM, deal.II, gslib and DOLFINx): the centroid start needs 1–2 more Newton iterations than the affine fit of the cell's vertices on every test mesh; MFEM's closest-node start finds wrong roots at p = 8. | Pull-back; ADR-0042, item 11; ADR-0046 | Start from the affine fit of the vertices, clamped to the box, in every pull-back strategy; a task T6c before T7a. | ADR-0047 rejected (maintainer, 2026-11-11: modest gain at the default 16 iterations); centroid start kept, no T6c |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |
| 2026-11-08 | T0a | Rule 2's "per component" is read per quantity: the relative error of X, J, det J, K, normals and ξ is each measured against the largest \|want\| of that quantity over the block (`reference::rel_err`), not per matrix entry. | Strictly per entry, f32 J on a GLL hexahedron box at p = 3 fails (1.7e-5 > 1e-5) with no kernel defect: rounding errors are about 5e-7 on every entry, while off-diagonal entries are 18× smaller than diagonal ones (M9's p = 2 box was already at 8e-6). Maintainer, 2026-11-08: per quantity. | ADR-0042 item 2 (clarification noted in its decision record) |
| 2026-11-08 | T0a | Files beyond the task's list: `gridcl-testing/src/meshes/{chunk.rs, boxes.rs, manifold.rs, lattice.rs}` and `gridcl/tests/permutation_info.rs` (struct literals gain `HostBlock::family`; `HostChunkBlock` carries the family too); `gridcl-testing/tests/{main.rs, reference.rs}` for the "`reference` agrees with M9's device factors at p = 2" test (plus a GLL hexahedron box at p = 3). | `with_node_family` needs a per-block family that `HostMesh::build` and `HostChunk::builder` pass to `MeshBuilder`. | – |
| 2026-11-08 | T0a | The "GLL meshes reproduce their equispaced geometry" test uses rule 2's scaled f64 tolerance, not a flat 1e-12; `perturbed(0.05)` keeps det J > 0 to p = 8 on every box, so its amplitude is not scaled by 4/p. Boxes with pyramids are checked to p = 3; the self-tests above p = 4 (p = 5 for GLL) run in release builds only. | f64 J on an equispaced hexahedron box at p = 6 differs from the GLL mesh by 1.4e-12, within rule 2's scale. | – |
| 2026-11-08 | T0b | Files beyond the task's list: `gridcl-testing/src/parallel.rs` (`high_order_invariance`, the rule-9 check shared by `parallel_threads.rs` and `gridcl-mpi-tests`, as M6's checks are); `gridcl-testing/Cargo.toml` and `gridcl-geometry/{Cargo.toml, src/lib.rs}` (feature `testing`, enabled by gridcl-testing); `gridcl-bench/src/record.rs` (the M9h record types `DegreeRecord`, `CudaResources`, `HighOrderData` and the stub `load_high_order_dir`, a separate loader so `load_dir` and the M3s tests are unchanged). | The tests need the shared check, the feature wiring and the record types; T1 implements the loader and may fold it into `load_dir`. | – |
| 2026-11-08 | T0b | `high_order_record.rs` fixes the layout of `gridcl-geometry/degree-policy.toml` for T4: `[<Class>.<Family>]` tables, one key per cell type, per scalar size in bytes a list of `[first degree, strategy]` pairs from p = 1, the last range open-ended. The local Mac's records use target `macmini`. | T4 and T11b write the file the acceptance test reads; M3s records named the old machine `m3max`. | – |
| 2026-11-08 | T0b | Readings in the tests: `for_each_strategy!` runs the policy and every strategy forced through the scoped hook, but only the environment's setting when `GRIDCL_GEOM_STRATEGY` is set (so a task's check set exercises its own strategy); each run uses a copy of the mesh with an empty E-vector cache, so the EVector strategy runs too. Rule 9's check asserts through `record_strategies` that launches went through the dispatch. The decision-rule test sums J, det J and K for `Jacobian`, uses the 9-iteration pull-back for `PullBack`, and, if no admitted strategy is within 3% of the fastest at both sizes, takes the smallest worst ratio (simplest on ties). Timing tests use the equispaced records. | Choices the rules leave open; T11b may refine them with a log entry, not silently. | – |
| 2026-11-08 | T0b | Rule 8's large case sets `max_page_size` to a third of the E-vector or of the J field, whichever is smaller (above the tabulation and one E-vector tile), and asserts that the E-vector and J span at least three chunks; X and det J, with fewer components per point, may span fewer. | With one point per cell, a third of X is smaller than one E-vector tile at hexahedron p = 8 (131 KB against 280 KB in f64), so no single limit splits every field into three chunks. | – |
| 2026-11-08 | T1 | Files beyond the task's list: `gridcl-bench/Cargo.toml` (gridcl-geometry with feature `bench`, gridcl-dims), `gridcl-bench/src/families/mesh.rs` (`nearest_side` crate-visible for the degree boxes). `load_high_order_dir` stays separate from `load_dir`; `DegreeRecord` gains `vector_size_ran`, `check_error` and `check_tolerance` (T3's f32 error survey), and record.rs `annotate_node_load`. | The sweep reuses the M3s box sizing; the records need the size that ran and the check result. | – |
| 2026-11-08 | T1 | `gridcl_geometry::bench` holds only the device basis evaluator (`Interpolant`: Σ_j w_j ψ_j and its derivatives at given points). The pull-back's set-up and Newton step have no hooks: `pull_back_block` always runs `max_iterations`, so the sweep times it at 1 and 9 iterations as rule 1 defines them. | No hook is needed to measure what rule 1 measures. | – |
| 2026-11-08 | T1 | Sweep readings: unperturbed (affine) boxes, so the pull-back is checked at 1 iteration too; checks on at most 64 sampled cells, rule 2's relative error per quantity, with the J-derived quantities and ξ held to the scaled tolerance times the box side (absolute coordinates of size 1 sum into J ~ 1/side), the basis evaluator to its coefficient amplification, the E-vector bitwise. `--strategies` accepts every name but runs only `direct` and `policy` until T4, which wires the hooks in `gridcl-bench/src/families/degree.rs` (outside T4's file list). The unit tests sweep every family and cell at p = 1–2 launch-only (checks included) plus one small timed sweep; the full timed p = 1–2 smoke is the release verify command, because at 10^4 points the protocol's per-call sync costs about 4 s per record. | Choices the rules leave open; debug-build test time (M3s: about 2 min). | – |
| 2026-11-08 | T2 | Files beyond the task's list: `gridcl-bench/src/{bin/degree_sweep.rs, families/degree.rs}` (`--vector-sizes both\|scalar\|policy` for launch-only runs, `LaunchVectorSizes`); `scripts/hpc/lib/direct.sh` and `scripts/hpc/tests/run.sh` (`status.sh` prints a running Locust or a4000 job's last output line as progress; test `status_lists_jobs` extended). | cubecl-cuda names a dumped PTX file by the tail of the kernel id, so comptime variants overwrite each other (new entry C19): the script runs one process per (family, cell, degree, type, strategy, vector size). The maintainer asked for a progress view of long Locust runs. | – |
| 2026-11-08 | T2 | `ncu` is refused on Locust (`RmProfilingAdminOnly: 1`, `ERR_NVGPUCTRPERM`), so `cuda-resources.sh` uses ADR-0042 item 6's fallback: `ptxas -v` (CUDA 12.6) on the dumped PTX. `local_bytes` is ptxas's stack frame, and the maximum threads per block are derived from the register count, not read from the driver. | No FFI outside the allowed crates; the driver attributes need it. | ADR-0042 item 6 |
| 2026-11-08 | T2 | The current direct kernels were measured at hexahedron p = 4, 6, 8 with vector size 1 only (`--scalar-only`), not p = 4–8 at both vector sizes; the records stay in `target/hpc/` (T11a commits the full set). Findings for later tasks: no kernel exceeds the 512 KiB local-memory limit (largest frame 51 KB), but the direct pull-back takes 145 s to first use at p = 4 and does not compile within 900 s at p ≥ 6, so rule 7 fails for it from p = 4. Maintainer, 2026-11-09: file both; C18 and C19 reported upstream (tracel-ai/cubecl#2108, #2107). Jobs: 20261108195701-3090235 (tool probe), 20261108195835-3091086, 20261108200536-3100661 (p = 1 verify, 118 records), 20261108200940-3114247 (C18 probe of 81e8739 on a scratch branch), 20261108204949-3152865 (full p = 4–8 sweep, cancelled after 2 h), 20261108225929-3171689 (C19 check and the p = 4, 6, 8 sweep). | Maintainer, 2026-11-08: the full sweep was too slow; run the smaller one. | – |
| 2026-11-09 | T3 | Files beyond the task's list: `scripts/bench/degree-sweep.sh` (one `degree_sweep` process per (family, cell, degree, type) under a time limit, all resuming one JSON file), `gridcl-bench/src/bin/degree_sweep.rs` (`--resume`, `--stopped-after S`: the configurations a stopped process left become `too slow to compile` before their kernel's first record, else `too slow`, both "(process stopped after S s)"), `gridcl-bench/src/families/degree.rs` and `gridcl-bench/src/record.rs` (records written through a temporary file and a rename; the mesh's E-vector is made only for the families that read it, after the E-vector family, and at vector size 1: before, the set-up compiled the E-vector kernel at the policy's size first, which hid the E-vector family's compile time and cost each process up to 1,039 s at hexahedron p = 8, new entry C20); `docs/cubecl-upstream.md` (C20). The baseline is pinned at this runner's commit, not at T2's: neither touches a kernel or the dispatch, so `direct` is still M9's (rule 11). The baseline measures equispaced nodes only (what `baseline_covers_every_target_and_degree` and the timing tests read; direct's cost does not depend on the node family); GLL is measured with T11a's sweeps. | T2: the direct pull-back does not compile within 900 s on Locust from p = 6, so the in-process `too slow to compile` (warm-up minus calibration over 300 s) never fires and one sweep process would hang. | – |
| 2026-11-10 | T3 | Locust CUDA part b (PullBack, BasisEval) and the Grace CPU sweep run at p ∈ {1, 2, 4, 8} (pyramids {1, 2}) through `degree-sweep.sh --degrees`, still at a88ab2a; Locust part a and the Mac and Kathleen sweeps cover p = 1–8. Commit 44d7e70 (one sweep process per (cell type, degree)) is not used for the baseline, which stays at one commit; it serves T11a. | Locust's remaining runs would take 9–12 h, mostly the host mesh build at 10^7 points. | ADR-0043 (accepted) |
| 2026-11-10 | T3 | The baseline is at a88ab2a (the T3 commit, recorded in every record), not the T2 commit; neither changes a kernel. Runs: Mac CPU and Metal (local, 2026-11-09); Kathleen jobs 631525 and 631727 (one socket each, two nodes); Locust jobs 20261110055956-3563316 (CUDA part a), 20261110130020-3609133 (CUDA part b) and 20261110144213-3615352 (Grace CPU), each idle before and after; the busy first part a (20261109175723-3488291) is not used. The pinned runner's start command hung after each Locust job (fixed in f95b07d, after the pin), so their results were copied by hand once the exit status was written, and `degree_sweep annotate --node-load --commit-from` filled the records. The A4000 was not used. | The baseline stays at one commit; the runner fix came after it. | ADR-0043 |
| 2026-11-10 | T4 | Files beyond the task's list: `gridcl-geometry/Cargo.toml` (`toml`, as in gridcl-device); `src/geometry.rs` (the E-vector dispatches in `Geometry::evector`, the only library caller of `evector::build`, which knows no cell type or degree); `src/bench.rs` (`Interpolant::evaluate` dispatches as `PullBack`, so BasisEval records name what ran); the `# Errors` docs of `src/{points_jac, detj_pinv, facets}.rs`; `tests/{strategy.rs, main.rs}`; `gridcl-bench/{Cargo.toml, src/families/degree.rs}` (feature `testing`; `--strategies` forces every family through `with_strategy`, `strategy_ran` comes from `record_strategies`, and a strategy that does not run yet makes a `failed` record instead of a usage error). `degree-policy.toml` lists per `[class.family]` and cell type, per scalar size, `[first degree, strategy]` pairs ascending from 1. Until a strategy runs, `choose` returns `NotYetImplemented` naming its task, also for a fallback to `loops`. | The hook tests need a mesh, and a unit test of gridcl-geometry would build it through gridcl-testing's second copy of the crate, whose thread-local hooks are not the ones under test; integration tests share one copy. | – |
| 2026-11-10 | T4 | Check set (local CPU in debug with the four settings and in release, Metal release): gridcl-geometry passes everywhere. gridcl's `high_order::{factors_match_host_reference, pull_back_recovers_reference_points}` and `high_order_invariance::{vector_size_invariance, two_runs_are_bitwise_identical, tile_invariance}` fail with `NotYetImplemented` at their forced `loops` run (T6a, T6b), as planned. Under `GRIDCL_GEOM_STRATEGY=direct` only `direct` runs, which reaches three T0b checks that had never run, and they fail: K of the ball at p = 3 in f32 (relative error 1.30e-5 > 1e-5, scale 1), ξ of the ball at p = 4 in f32 (1.52e-5 > 1e-5, scale 1), and `vector_size_invariance`'s large case, whose `try_clone` runs inside `with_properties` and cannot copy the mesh under the reduced page size (`AllocationTooLarge { bytes: 262208, max: 87466 }`). T4 leaves `direct` bitwise M9's (`forced_direct_matches_the_policy_bitwise`), so these predate it; they are left for T6a and T6b, whose `loops` runs reach the same checks. | Found by the check set; not T4's to fix. | – |
| 2026-11-10 | T4 | CUDA check on the A4000, because Locust was running the baseline. Jobs: 20261110120040-3754425 (other users held 59 cores; the gridcl integration binary was killed with SIGKILL after 2.85 h, gridcl-geometry not reached); 20261110145407-3798950 (gridcl-geometry: 15 unit, 36 integration and 57 doc tests pass, idle before and after); 20261110145916-3801373 (gridcl: 66 unit and 85 doc tests pass; integration 53 pass and the five expected `NotYetImplemented` failures, then SIGKILL after 2.5 h with the node idle, while the thread-rank tests ran beside the high-order ones; likely host memory); 20261110181624-3823880 (rerun without `high_order*`, 4 threads; cancelled). No test failed beyond the expected five. | Maintainer, 2026-11-10: T4 counts as checked on CUDA from the completed tests; the remaining A4000 runs were dropped. The SIGKILLs are for the milestone's full CUDA run (Locust) to watch. | – |
| 2026-11-10 | T6a (prep) | ADR-0044's test support lands before T6a, while Locust ran the baseline: `gridcl_testing::reference::{conditioned_scale, pinv_scale}`, `pull_back_scale` without κ in A_ref, and the callers `gridcl-testing/src/parallel.rs`, `gridcl-testing/tests/reference.rs`, `gridcl/tests/{high_order.rs, high_order_invariance.rs}` and `gridcl-bench/src/families/degree.rs`; `vector_size_invariance`'s large case clones the mesh outside `with_properties`. `gridcl/tests/common/mod.rs` (`block_scale`) and `gridcl-testing/src/meshes/family.rs` scale only X, J and det J and stay unchanged. Under the pointwise factor of option 2, K of the GLL ball at p = 4 failed in f32 (1.40e-4 > 1.13e-4); the maintainer amended ADR-0044 to separate maxima. With it, `GRIDCL_GEOM_STRATEGY=direct` passes all six `high_order*` tests on the CPU runtime, and gridcl-testing and gridcl-bench pass (`timed_smoke_and_second_call_compile_time` failed once under load on its "nine iterations take longer than one" timing assert and passed 3 of 3 alone). | ADR-0044 (amended); T6a's first step. | ADR-0044 |
| 2026-11-10 | T5 | Files beyond the task's list: `gridcl-device/tests/launch_plan.rs` (the property test of `per_group` under injected limits, and the rule-4 tree kernel test, next to M2's plan tests). Readings the rule leaves open: groups are items as in `elementwise` (the same vector-size rule and overrides; lanes are groups; `num_vectors` counts vectors of groups, padded to `VECTOR_PAD`); slot s of cube c takes vectors (c · slots + s) · `items_per_unit` + i; `units_per_group` is also at most `max_cube_dim.0`, and on GPUs at most the cube of 256 units (or the smaller limits), which then holds ⌊cap / `units_per_group`⌋ groups whatever the group count; CPU blocks are sized as `elementwise`'s (whole 128-byte lines of `cpu_block_bytes` of N · elem_bytes), and the performance-core setting of ADR-0017 does not cap `units_per_group`; `shared_bytes` above `max_shared_memory_size` is `TooLarge`; `units` = 0 panics. The tree test's unit-level combine pairs adjacent units (u with u + s at stride s), which matches the per-unit pairwise tree; a stride-down combine (as in `reduce`'s `cube_tree`) differs at 8 units and the test catches it. | Choices the rules leave open. | ADR-0042 item 4 |
| 2026-11-10 | T5 | CubeCL's CPU runtime runs one thread per unit of a cube, and each thread visits every cube in turn (cubecl-llvm `src/cpu/entrypoint.rs`, cubecl-cpu `src/compute/threadpool/mod.rs`), so a `per_group` CPU plan (`cube_dim` = `units_per_group`, as rule 4 says) runs `units_per_group` threads: at `units_per_group` = 1 (T9's plan) a CPU launch is serial. No hazard from consecutive cubes sharing one block of shared memory: cubecl-llvm adds a cube barrier at the end of each cube when a kernel declares shared memory (`fence_cubes`), so no C entry. | For the maintainer: whether CPU plans should hold several groups per cube (an ADR, since rule 4 fixes the CPU shape), or T9/T10 leave CPUs to the policy. | – |
| 2026-11-10 | T5 | Check set (local part): debug default 435 passed, `GRIDCL_VECTOR_SIZE=scalar` 434, `=widest` 434, `GRIDCL_QUAD_TILE=1` 435, release 433; Metal release 432 (then gridcl-device alone again after the fix below: all pass). Every run fails the five `NotYetImplemented` tests of T6a/T6b (`high_order::{factors_match_host_reference, pull_back_recovers_reference_points}`, `high_order_invariance::{vector_size_invariance, two_runs_are_bitwise_identical, tile_invariance}`). Besides: on Metal, T5's unit test `cpu_cubes_visit_blocks_of_consecutive_groups` assumed the test device's `max_vector_size` (Metal: 4), fixed by injecting 16; under `scalar` and `widest`, gridcl's doctest `Mesh::create_connectivity` was killed by a signal once each (no assertion failed), and passed 10 of 10 reruns under `scalar`; it does not use `per_group`, and reads only host lengths, so it may return with launches queued, as in C17. Locust not run (after the merge). | Found by the check set. | – |
| 2026-11-10 | T5 | Locust check at ab809e3 (after the merge), CUDA release, idle before and after: 20261110212917-3877085 (gridcl-device: 36 unit, 62 integration and 75 doc tests pass); 20261110213056-3878376 (gridcl: 66 unit and 85 doc tests pass; integration: 62 pass and the five expected `NotYetImplemented` tests fail, then the binary died with SIGSEGV after 5,425 s with about 50 tests still running, mostly the thread-rank `*_threads` tests and `parallel_threads::high_order_parallel_invariance`). The runner's terminal was interrupted during the second job; its results were collected afterwards through the runner's own poll, collect and summary functions. 20261110202840-3857940 (gridcl-device and gridcl together) stopped at the 1 h limit at the same point. No `per_group` launch is among the failures. | As T4's SIGKILLs on the A4000: for the milestone's full CUDA run to isolate (one binary, fewer test threads). | – |
| 2026-11-11 | T6a | Files beyond the task's list: `gridcl-geometry/src/{geometry.rs, field.rs, lib.rs, strategy.rs}` and `tests/strategy.rs` (the E-vector strategy reaches `evector::build`; `loops` runs for every family but `PullBack`, whose `loops` stays `NotYetImplemented` until T6b; the hook tests force `cooperative` as the strategy that does not run yet). The tails of `factor_kernel` after J (J, det J, K, facet normals, normals) move into `layout::store_from_jacobian`, shared with `layout_loops::factor_loops_kernel`; `direct`'s output is bitwise unchanged (a fingerprint of the E-vector and every factor of hexahedra at p = 1–3 and the sphere's normals, f32, at the default, widest and scalar vector sizes). `loops` sums over the nodes in `direct`'s order and is bitwise equal to `direct` on the CPU runtime at every vector size. | One store path for both strategies; the dispatch needs the callers. | – |
| 2026-11-11 | T6a | Check set with `GRIDCL_GEOM_STRATEGY=EVector=loops,Points=loops,Jacobian=loops,Normals=loops,FacetNormals=loops` in place of `loops` (the pull-back's `loops` is T6b's). Local CPU debug (default, `scalar`, `widest`, `GRIDCL_QUAD_TILE=1`), release, Metal release: gridcl-geometry passes everywhere, gridcl's only failures are the five `NotYetImplemented` tests of T4's row. Under the setting: debug passes everything (p ≤ 4); release (CPU and Metal) fails three checks at p ≥ 5 that had never run: J of the GLL hexahedron box at p = 6 in f32 (relative error 2.03e-5 > 1e-5, scale 1; identical on both), ξ of the equispaced ball at p = 8 in f32 (1.335 > 4.24e-2, scale 4245), and `vector_size_invariance`'s large case (CPU: the pull-back's `AllocationTooLarge { bytes: 470656, max: 196800 }`; Metal: ξ of the quadrilateral box at p = 3, vector size 2, 3.65e-5 > 1.02e-5). `GRIDCL_GEOM_STRATEGY=direct` in release fails the same three with the same errors (J bitwise), so they predate T6a: the two pull-back checks are T6b's (`pull_back.rs`), and J at p = 6 is rule 2's tolerance (ADR candidate). In release, gridcl's doctest `Mesh::footprint_bytes` was killed by a signal once and passed 3 of 3 reruns (as `create_connectivity` in T5's row). | Found by the check set; not T6a's to fix. | – |
| 2026-11-11 | T6a | Locust check at 6dcc28a (CUDA release, idle before; the tree held only the temporary run script): job 20261111063919-3925105 ran gridcl-geometry (15 unit, 37 integration and 57 doc tests pass, including the strategy tests that force `loops`), then gridcl's `high_order*` tests with every family but the pull-back on `loops` (4 test threads). None of those finished within 2.5 h: three of them also run the pull-back, still `direct`, which does not compile on Locust within 900 s from p = 6 (T2, C18). The job was cancelled. T6a's kernels on CUDA at p = 1–8 are checked with T6b's Locust run instead, after ADR-0046's tolerances and the `loops` pull-back land; on the CPU runtime and Metal `loops` is bitwise equal to `direct` (CPU) up to p = 8. | Maintainer, 2026-11-11: stop the job; T6b's run covers T6a. | ADR-0046 |
| 2026-11-11 | T6b | ADR-0046's first step: `newton_update` (`pull_back.rs`) holds the Newton step shared by every pull-back kernel and clamps each iterate to [−0.1, 1.1]^TD by two selects (margins rounded to f32); `direct` is bitwise unchanged on a fingerprint of ξ and the last step (ball, triangle, quadrilateral, tetrahedron, prism, pyramid boxes, sphere, circle; p = 1–3; f32, f64; default, scalar and width-4 vector sizes; 144 cases). Test support: `gridcl_testing::reference::{node_ratio, origin_ratio, jacobian_scale}` and `host::jacobian_scale`; `conditioned_scale` includes the node ratio and `pinv_scale`, `pull_back_scale` the origin ratio. Callers as ADR-0044 lists, with readings: the J scale of `high_order_invariance.rs` is the largest over blocks (as before); `family.rs` measures R from the coordinates' zero (the host sums absolute coordinates); `degree.rs` keeps the box side as its R and gains the node ratio; `common::block_scale` now serves X only. The large case's page limit is at least C′, and the J assertion is waived on hexahedra from p = 6 (asserted as the only exception). Beyond ADR-0046's list: `gridcl/tests/pull_back.rs::outside_points_of_affine_cells_converge_outside_the_reference_cell` (M9) now stretches by 1.8 and 3: points whose preimage lies in the box converge (some outside the cell), points beyond it end on the box boundary with a last step above the tolerance; and `gridcl-testing/tests/reference.rs` bounds the pull-back scale by κ_F / TD times R. | The clamp changes that test's behaviour by design (ADR-0046: a point whose preimage lies outside the box does not converge); the ADR's list of affected tests missed it. | ADR-0046 |
| 2026-11-11 | T6b | Files beyond the task's list: `gridcl-geometry/src/{bench.rs, strategy.rs, basis_eval/tests.rs}`, `tests/strategy.rs`; `gridcl-bench/src/families/degree.rs` (`records_name_the_strategy_that_ran` expected `loops` to fail, stale since T6a; `cooperative` now plays that part). Readings: the `loops` pull-back is two kernels: a set-up kernel (three buffers) writes a(k, g) = Σ_j C′(k, j) x(j, g) per cell to a scratch buffer sized for the largest launch range (`[k][g][cell]`, at most one E-vector chunk, so within the page limit), and the Newton kernel (five buffers) reads it like the E-vector; no local array of nb · gd values (T2: the likely C18 cause). The evaluator (`basis_eval/loops.rs`) takes the degree at run time; recurrences run as rolling pairs, x's Legendre values sit in a local array and y and z roll, so the 1D arrays have a comptime capacity (`loops_capacity`: 4, 8 or 16 for p ≤ 2, ≤ 6, ≤ 14), and one kernel serves every degree up to its capacity. Lane-uniform coefficients (the `bench` `Interpolant`, the evaluator's unit test) are uploaded as broadcast rows of `VECTOR_PAD` copies. `loops` is bitwise equal across vector sizes on the CPU runtime but not bitwise equal to `direct`: one f32 ulp on a triangle at p = 3, where `direct`'s recurrence constants are comptime literals and `loops`' runtime values, and the backend fuses multiply-adds around them differently (C16's mechanism; no new C entry); the unit test `loops_agrees_with_direct` compares within the tolerance. The evaluator's unit tests run `loops` at p = 1–8 (pyramids 1–3) in debug builds too, with the tolerance scaled by the coefficient amplification ratio and, for derivatives, max(1, (p / 4)²). | Choices the task leaves open. | – |
| 2026-11-11 | T6b | Locust check before the commit, on 8a8365f plus the T6b tree (dirty, changes hashed 0e7ef4161198): job 20261111125439-4101401, CUDA release with `GRIDCL_GEOM_STRATEGY=loops` (every family; under the policy the high-order tests reach `direct`'s pull-back from p = 6, which does not compile within 900 s, T2 and C18) and `--test-threads=4`, idle before and after. All pass in 1,151 s: gridcl 66 unit, 113 integration (the six `high_order*` tests and `parallel_threads::high_order_parallel_invariance` at p = 1–8, f32 and f64) and 85 doc tests; gridcl-geometry 16 unit, 37 integration and 57 doc tests. This also covers T6a's `loops` kernels on CUDA at p = 1–8 (T6a's row). | Maintainer, 2026-11-11: start the Locust run before the local check set ends. | – |
| 2026-11-11 | T6b | Check set (local, CPU runtime unless named): debug, gridcl-geometry and gridcl in full, at the default settings, `GRIDCL_VECTOR_SIZE=scalar`, `=widest` and `GRIDCL_QUAD_TILE=1`: everything passes except the five `high_order*` tests, which stop with `NotYetImplemented` at their forced `shared` run (T9; under no setting `for_each_strategy!` forces every strategy, and `loops`, now running, no longer stops them earlier); `GRIDCL_GEOM_STRATEGY=loops` (debug): everything passes (gridcl 66 unit, 113 integration, 85 doc; gridcl-geometry 16, 37, 57); release at the default settings: as debug (the same five). Runs 7–9 shortened to the modules T6b touches (maintainer, 2026-11-11): release under `loops`: gridcl-geometry in full and gridcl's `high_order*`, `parallel_threads::high_order_parallel_invariance`, `pull_back`, `determinism`, `factor_invariance`, `kernel_invariance` and `manifold_factors` (26 tests, p = 1–8) pass; Metal release: gridcl-geometry in full and the same gridcl modules but `high_order*`, plus gridcl's doc tests, pass (`high_order::pull_back_recovers_reference_points`, matched by the `pull_back` filter, stops at `shared` as above); Metal release under `loops`: gridcl-geometry in full and gridcl's `high_order*`, `parallel_threads::high_order_parallel_invariance`, `pull_back` and `determinism` (18 tests, p = 1–8) pass. gridcl's threaded mesh tests (io, partition, submesh, tags and the like) ran on the CPU runtime only; they touch nothing T6b changed and run on Metal again at T12, T13 and the review. gridcl-testing and gridcl-bench pass in debug apart from `high_order_record.rs` (T11) and one load-sensitive timing assert of `timed_smoke_and_second_call_compile_time` (passed 3 of 3 alone, as in T6a (prep)). | Maintainer, 2026-11-11: shorten runs 7–9 (the full debug suite took 50–85 min per setting while another session's tests shared the machine). | – |

## Review

<Filled by /review-milestone.>
