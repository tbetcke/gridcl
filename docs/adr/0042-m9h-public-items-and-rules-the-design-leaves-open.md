# ADR-0042: M9h public items and rules the design leaves open

- **Status:** Accepted
- **Date:** 2026-11-08
- **Raised in:** M9h planning (docs/plans/M9h.md, "Planned public API and rules" and questions 1–12)
- **Design sections affected:**
  - "Vectorization and CPU execution": Listing S2 (`LaunchPlan`), "Benchmarks";
  - "Geometry": "Tabulation and quadrature", "Geometric factors", "Pull-back";
  - "Testing and validation": tolerances, parallel invariance, vector-size invariance;
  - "Workspace layout" / "Cargo features";
  - "Handing this design to Claude Code": the working-agreement copy list (also CLAUDE.md);
  - "Implementation roadmap": M9h, how its acceptance is read.

## Context

M9h's roadmap section (ADR-0041) asks for degree sweeps p = 1–8 of every degree-dependent kernel family. It asks for four strategies compared with M9's direct evaluation ("runtime loops instead of full unrolling …; sum factorisation for quadrilaterals, hexahedra and prisms; tabulations or coefficients in shared memory; a workgroup cooperating on one cell only where these fall short, with fixed-order reductions that do not depend on the plane size"), a degree policy "with an override for tests (as `GRIDCL_VECTOR_SIZE` is for vector sizes)", and C18 closed. Its acceptance:

> - every cell type except pyramids, at p = 1–8 and with both node families where offered: … match the host reference (gridcl-refcell's f64 tabulation) to 10^−12 (f64) and 10^−5 (f32), and the pull-back recovers reference points to the same tolerances; …
> - every kernel variant up to p = 8 compiles, loads and launches on each backend, and the first-use compile and load time of each is at most 30 s on Locust;
> - for quadrilaterals and hexahedra, the time per quadrature point of geometric factors and of a pull-back Newton step at p = 8 is at most 3 times that at p = 4 on every device class (direct evaluation: about 5.8 times);
> - no family is more than 5% slower than in M9 at p ≤ 4 on any device class;
> - the determinism, vector-size invariance and parallel invariance tests pass at p = 1–8: …
> - `docs/perf/high-order.md` lists, for every family, device class and degree range, the measured cost of each strategy, the chosen one and the reason, …

ADR-0041 left four points to the plan: f32 at p = 8, a throughput target for simplices, how CUDA kernel attributes are read within CLAUDE.md's FFI rule, and whether the degree policy needs a public item.

Planning read these against the code at main d90f90e (docs/plans/M9h.md, Findings):

- **Kernels.** The degree enters every kernel at comptime, and node loops are fully unrolled:
  - the factor kernel (`gridcl-geometry/src/layout.rs:233`, about 6,561 FMAs per point for J on a hexahedron at p = 8);
  - the E-vector kernel;
  - the pull-back (`pull_back.rs:88`): a runtime N² set-up a = C'·x into a local array of nb·gd vectors, then a Newton step that unrolls over every basis function (`basis_eval/mod.rs:212`).
- **Launch shapes.** `LaunchPlan` has only `elementwise` (`gridcl-device/src/plan.rs:90`). It carries no group size and checks no shared-memory size, and no geometry kernel uses shared memory.
- **Bases.** The hexahedron's orthogonal set is x-fastest lexicographic Legendre (`gridcl-refcell/src/polynomials.rs:157`), and the prism's is Dubiner × Legendre on layered nodes (:174). The coefficients therefore factor as Kronecker products under a node permutation.
- **CUDA.** cubecl-cuda 0.11 keeps `CUfunction` private and never calls `cuFuncGetAttribute`. It does write every kernel's PTX on its default LLVM path under `CUBECL_CUDA_DUMP_PTX` (`src/compute/context.rs:474, 580`), and NVIDIA's `ncu`, `ptxas -v` and `compute-sanitizer` report registers, local memory and failing driver calls from outside the process.
- **Benchmarks.** gridcl-bench has no degree axis, and its sizes count cells. A p = 8 hexahedron mesh of 10^6 cells needs about 17 GB of E-vector in f64. The committed vectorization data is from the M3 Max (ADR-0034).

## Problem

Each item matches the question of the same number in docs/plans/M9h.md.

1. **Bullet 3 cannot hold for the pull-back.** At an arbitrary point ξ, X(ξ) and J(ξ) depend on every one of the (p + 1)^d coefficients of the cell, so a Newton step costs at least about (p + 1)^d operations per point. Sum factorisation lowers this cost only on a tensor grid of points shared by a cell, which the pull-back does not have (each point has its own ξ). From p = 4 to p = 8 the leading cost grows by (9/5)^3 ≈ 5.8 on hexahedra and (9/5)^2 ≈ 3.2 on quadrilaterals, both above 3, on any compute-bound device.
2. **f32 cannot meet 10^−5 at p = 8.** The rounding error of evaluating Σ_j x_j ψ_j(ξ) grows with the amplification factor Σ_j |ψ_j(ξ)|, and the error of J with Σ_j |∇ψ_j(ξ)|. These are not kernel defects. Computed on the host in f64 at the Gauss points of degree 2p (gridcl-refcell `tabulate` and `coefficients`; reproduced by M9h T0a), relative to the p = 4 equispaced element of the same cell at the same points:

   | Cell, nodes | p = 5 | p = 6 | p = 7 | p = 8 |
   | --- | --- | --- | --- | --- |
   | Interval, equispaced (X / J / coeff.) | 1.3 / 1.6 / 1.3 | 1.6 / 2.7 / 1.4 | 2.3 / 4.8 / 2.0 | 3.6 / 8.8 / 2.6 |
   | Interval, GLL | 0.8 / 0.9 / 1.1 | 0.8 / 1.2 / 1.2 | 0.8 / 1.4 / 1.4 | 0.8 / 1.7 / 1.6 |
   | Triangle | 1.4 / 2.0 / 1.5 | 2.3 / 3.9 / 2.2 | 3.7 / 7.6 / 3.5 | 6.4 / 15 / 5.8 |
   | Quadrilateral, equispaced | 1.8 / 2.1 / 1.6 | 2.7 / 4.3 / 2.0 | 5.4 / 11 / 4.2 | 13 / 31 / 6.6 |
   | Quadrilateral, GLL | 0.6 / 0.8 / 1.2 | 0.6 / 0.9 / 1.5 | 0.6 / 1.1 / 2.0 | 0.7 / 1.4 / 2.4 |
   | Tetrahedron | 1.6 / 2.3 / 1.8 | 2.7 / 4.9 / 3.2 | 4.7 / 10 / 5.9 | 8.4 / 21 / 11 |
   | Hexahedron, equispaced | 2.3 / 2.8 / 2.0 | 4.3 / 7.1 / 2.8 | 12 / 25 / 8.5 | 46 / 112 / 17 |
   | Hexahedron, GLL | 0.5 / 0.6 / 1.3 | 0.4 / 0.6 / 1.8 | 0.5 / 0.8 / 2.9 | 0.5 / 1.1 / 3.8 |
   | Prism | 1.8 / 2.6 / 1.8 | 3.7 / 6.3 / 3.1 | 8.5 / 18 / 7.1 | 23 / 54 / 15 |

   "X" is max Σ_j |ψ_j|, "J" is max_d Σ_j |∂_d ψ_j|, and "coeff." is max Σ_k |φ_k| Σ_j |C_kj|, the path of the pull-back through `LagrangeElement::coefficients` (φ the orthonormal basis). The absolute factors at hexahedron p = 8 are 464, 19,019 and 6,910. At p ≤ 4 every ratio is at most 1.01, and on pyramids (p ≤ 3, reference p = 3) at most 1. For f64 the worst-case bound ε₆₄ · 19,019 ≈ 2 · 10^−12 for J on an equispaced hexahedron at p = 8 exceeds 10^−12 too, although typical errors are smaller.
3. **Degree policy.** ADR-0041 asks whether it needs a public item next to `VectorPolicy`. gridcl-device does not know cell types, and the override has no name.
4. **Cooperative launches.** A workgroup cooperating on one cell needs a cube per group of cells and a group size the kernel can read. `LaunchPlan::elementwise` gives neither, and CLAUDE.md requires every launch shape to come from `LaunchPlan`.
5. **Copies.** Sum factorisation and a tensor pull-back upload 1D tabulations, a lexicographic node permutation, 1D nodes and barycentric weights, and the prism's triangle coefficient factor. The copy list names "quadrature points and tabulations when a rule is created or a tabulation is first cached" and "a geometry block's expansion coefficients", but not these items by name.
6. **CUDA attributes.** CLAUDE.md allows FFI in gridcl-device only for GPU-aware MPI and the macOS core count, and CubeCL exposes no attribute query (Context).
7. **"Every kernel variant"** is undefined once there are five strategies. Forcing all of them through every vector size, cell, degree and precision would mean thousands of CUDA JIT compilations of up to 30 s each in one test run. The direct strategy may also take minutes or fail at p = 8 (C18).
8. **Vector-size invariance.** "Testing and validation" requires lengths up to 10^6 + 3. At p = 8 that case does not fit in memory, and a direct launch over it would take minutes per vector size.
9. **Parallel invariance at p = 1–8.** The design lists which quantities are compared (counts, volume, vertex lists, permutation bits, files) but not geometric factors, and the existing parallel cases stop at p = 2. Every ghost mode at every degree on 1–4 ranks would exceed the CI budget.
10. **Simplices** get no throughput target in the acceptance; ADR-0041 leaves open whether they should.
11. **"Than in M9"** names no data. The committed data is from another machine (the M3 Max), and a 5% bound compared across weeks on Locust (an open node) or across Kathleen nodes is within run-to-run noise.
12. **Protocol.** "Benchmarks" sweeps every vector size, tile width (1, 4, 16) and CPU block (4, 16, 64 KiB) at 10^5–10^7 items. Multiplied by five strategies, seven cells, eight degrees, two node families and two precisions, that is not feasible per target. The rule for choosing a strategy is not defined.
13. **Design text** for the strategies and the policy is missing from "Geometry". ADR-0041 says it changes "only if M9h changes what those sections describe, through M9h's own ADR".

## Options

1. **Keep the design as written.** Bullet 3's pull-back clause would fail by construction (item 1). Bullet 1 would fail in f32 at p ≥ 6 on equispaced tensor cells (item 2). The cooperative strategy could not be launched within CLAUDE.md (item 4). Items 7, 8 and 12 could not run in any time budget. Not chosen.
2. **Fix the open items now, item by item** (below). This adds one public constructor and field to `LaunchPlan`, two hidden modules on gridcl-geometry, copy-list entries, a reading of bullets 2–4, tolerances that scale with the element's amplification, and design text. Items 1–4 record the maintainer's planning answers of 2026-11-08.
3. **Option 2 without the cooperative strategy** (no `per_group`; "only where these fall short", decided after the other strategies are measured). This needs no public API change now, but the maintainer chose to plan it up front (2026-11-08). Not proposed.

## Proposed decision

Option 2, item by item.

1. **Bullet 3, read** (maintainer, 2026-11-08: amend):
   - Geometric factors keep "at most 3 times".
   - A pull-back Newton step: at most 1.1 · (9/5)^d, that is 6.4 for hexahedra and 3.6 for quadrilaterals. This is a no-regression bound at the node-count floor.
   - The pull-back's per-cell set-up per point, for a strategy that has one: at most 3 times. The tensor pull-back of item 5 has none.
   - The ratio is that of the strategy the policy selects at p = 8, timed at p = 8 and at p = 4.
   - Time per point is time per call / (cells × points per cell), with the Gauss rule of degree 2p, at 10^6 and 10^7 points.
   - A Newton step is (t(9 iterations) − t(1 iteration)) / 8, and the set-up is t(1) minus one step.
   - Device classes use ADR-0015 item 5's geometric mean, and the report lists per-device ratios.
2. **Tolerances at high degree** (maintainer, 2026-11-08: scaled by the element's amplification):
   - f32: 10^−5 · max(1, A / A_ref).
   - A is computed on the host in f64 at the test's own points: Σ_j |ψ_j| for X, max_d Σ_j |∂_d ψ_j| for J and the quantities derived from it, and Σ_k |φ_k| Σ_j |C_kj| for a pull-back through coefficients, times κ(J) at the point.
   - A_ref is the same factor for the p = 4 equispaced element of that cell type at the same points.
   - **f64: the same factor, 10^−12 · max(1, A / A_ref).** This goes beyond the planning answer, which kept f64 at 10^−12. The worst-case bound at hexahedron p = 8 is about 2 · 10^−12 (Problem 2). The factor is at most 1.01 for p ≤ 4, so no current test changes.
   - The relative error of a quantity is |have − want| / max |want| over the block, per component, so det J ~ h³ stays relative.
   - The factor is the element's. Test meshes at p ≥ 5 keep the ratio of coordinate size to cell size no larger than in M9's p ≤ 4 tests, so that coordinates relative to the origin add no amplification the factor does not cover.
   - The same factor scales the GPU invariance comparisons (item 8) and the f32-against-f64 comparisons.
3. **Degree policy** (maintainer, 2026-11-08: crate-private):
   - `gridcl-geometry/degree-policy.toml` is compiled in. It holds one entry per (family, cell type, device class, scalar size), each with ascending degree ranges and a strategy.
   - The families are EVector, Points, Jacobian (also det J and the pseudo-inverse), Normals, FacetNormals and PullBack.
   - The last range is open-ended and resolves to a strategy known to compile at p = 9–10 (Gmsh's highest orders).
   - `GRIDCL_GEOM_STRATEGY` is `policy | direct | loops | sumfact | shared | cooperative`, or a comma list `<family>=<strategy>`. It is read once, and an invalid value panics naming the variable.
   - Precedence: the scoped hook, then the environment, then the TOML. The hook is thread-local.
   - A forced strategy that does not apply falls back to `loops`. That covers sum factorisation on a simplex or pyramid or on a non-tensor point set, and `shared` above the shared-memory limit.
   - Hidden modules (ADR-0038 style):

     ```rust
     #[doc(hidden)] #[cfg(feature = "testing")]
     pub mod testing {
         pub enum Strategy { Direct, Loops, SumFactorised, Shared, Cooperative }
         pub enum StrategyFamily { EVector, Points, Jacobian, Normals, FacetNormals, PullBack }
         pub enum StrategyOverride { Policy, Fixed(Strategy) }
         pub fn with_strategy<R>(family: StrategyFamily, s: StrategyOverride, f: impl FnOnce() -> R) -> R;
         pub fn record_strategies<R>(f: impl FnOnce() -> R) -> (R, Vec<(StrategyFamily, Strategy)>);
     }
     #[doc(hidden)] #[cfg(feature = "bench")]
     pub mod bench { /* basis-evaluator, pull-back set-up and Newton-step timing hooks */ }
     ```

   - gridcl-testing enables `testing`, and gridcl-bench enables `bench`. No stable public item changes.
4. **`LaunchPlan::per_group`** (maintainer, 2026-11-08: plan the cooperative strategy up front):

   ```rust
   pub struct LaunchPlan { /* existing fields */ pub units_per_group: u32 }  // 1 for `elementwise`
   impl LaunchPlan {
       pub fn per_group(client: &Client, kind: KernelKind, groups: usize, units: usize,
                        elem_bytes: usize, max_vector: usize, live: usize, shared_bytes: usize)
           -> Result<Self>;
   }
   ```

   - `units_per_group` is the largest power of two ≤ min(`units`, `max_units_per_cube`).
   - On GPUs a cube holds `units_per_group` × (groups per cube) units, at most 256. On CPUs `cube_dim` = `units_per_group`, and each cube visits blocks of consecutive groups.
   - `shared_bytes` per cube is checked against the device's shared-memory limit, along with `max_cube_dim`, `max_units_per_cube` and `max_cube_count`, with usize overflow checks.
   - Reductions over a group's units are pairwise trees over a fixed comptime leaf count L: each unit reduces its leaves pairwise, then the units combine pairwise. The result is therefore the same for every power-of-two `units_per_group` dividing L, including 1, and does not depend on the plane size.
   - `sync_cube` is called only in uniform control flow.
5. **Sum factorisation data and the copy list.**
   - A `PointSet` built by a tensor rule (Gauss on quadrilaterals and hexahedra; triangle rule × Gauss on prisms) records its factors crate-privately. User point sets have none.
   - Uploaded once per (element, point set), as tabulations: the 1D (or triangle × 1D) tabulations, and the lexicographic-to-gridcl node permutation packed into the same buffer.
   - Uploaded once per geometry block, as expansion coefficients:
     - the 1D nodes and barycentric weights with which the tensor pull-back evaluates the 1D Lagrange bases at ξ (no set-up and no C' on quadrilaterals and hexahedra);
     - the prism's C'_tri factor.
   - The copy list names these.
   - No kernel binds more than six buffers.
6. **CUDA attributes and compile time without FFI** (ADR-0041 open point):
   - Registers, spill and local memory per thread come from `ncu` (`launch__registers_per_thread` and the local-memory metrics), or from `ptxas -v` of the driver's CUDA version on `CUBECL_CUDA_DUMP_PTX` output. A runner script joins them to the benchmark records by kernel name.
   - Maximum threads per block is derived from the register count.
   - `compute-sanitizer` names a failing driver call.
   - First-use compile and load time is the warm-up of a call that launches exactly one new kernel, minus its median, with `CUDA_CACHE_DISABLE=1`.
   - No FFI exception is added.
7. **Bullet 2, read:**
   - Every (family, cell type, degree, precision, strategy, vector size used by the policy or the invariance tests) compiles, loads and launches on each backend. This is checked from per-backend `degree_sweep --launch-only` records; the accuracy tests also force every strategy.
   - The 30 s bound on Locust applies to the variants the policy selects for the Gpu class.
   - Other variants are measured and reported. A strategy over 30 s on Locust, or one that fails to compile, load or launch, is not selectable for the Gpu class.
8. **Vector-size invariance at p = 3–8.**
   - The 10^6 + 3-cell case stays at p ≤ 2.
   - At p = 3–8 the large case is 16k + 3 cells with a Gauss rule of degree 1. The mesh is built outside `testing::with_properties`; the factor and pull-back calls run inside it, with `max_page_size` between the tabulation size and a third of the E-vector, so the E-vector and fields span at least three chunks and several cubes.
   - Cell counts 0, 1, 15, 16 and 17 and tile widths 1 and 16 are unchanged.
9. **Parallel invariance at p = 1–8:**
   - Geometric factors and pull-back results per (cell origin id, point) equal the one-rank run: bitwise on the CPU runtime (per-cell work does not depend on the partition), and within item 2's tolerances on GPUs.
   - Thread ranks 1–4 at every p, with every ghost mode at p ≤ 2 and one per case above.
   - The MPI target runs 1–4 ranks at p ∈ {1, 4, 8}.
10. **Simplices:** no throughput bound. Collapsed-coordinate sum factorisation is out of scope. Triangles, tetrahedra and pyramids get loops, shared and cooperative, and `high-order.md` reports them.
11. **Bullet 4, read:**
    - The `direct` strategy stays bitwise M9's kernels; only dispatch is added around them.
    - "Than in M9" is a paired comparison in one job per target: the policy against `direct` at p ≤ 4, at most 1.05 under item 1's class aggregation.
    - A baseline of `direct` at a pinned commit before any dispatch change, measured on the current machines, shows that dispatch adds no measurable cost. It is reported, not bounded. M3 Max data is not used.
12. **Degree sweep protocol and decision rule:**
    - The sweep follows "Benchmarks" (medians of ten batched runs, warm-up, conditions, node-load rule), except that vector size, tile and CPU block come from `vector-policy.toml`. M10 re-studies those for the final kernels.
    - Sizes are 10^6 and 10^7 quadrature points (Gauss degree 2p), plus 10^5.
    - A configuration over 600 s per timed run on a CPU is recorded as "too slow", and a compile over 300 s as "too slow to compile".
    - A strategy is adopted over the baseline (`direct`, or `loops` where `direct` is not selectable) only if it is at least 10% faster at 10^6 and 10^7 points.
    - Among strategies within 3% of the fastest, the simplest wins: direct < loops < sumfact < shared < cooperative.
    - Classes with several devices use ADR-0015 item 5's geometric mean.
    - Degree ranges are maximal runs of equal choices.
13. **Design text:** as listed under Consequences.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing S2:** the `LaunchPlan` struct gains `units_per_group` ("Units that cooperate on one group; 1 for `elementwise`"). `per_group` follows `elementwise`, with item 4's doc. After the "One kernel source" bullet comes a new bullet: "**Groups:** `per_group` plans cubes of cooperating units; reductions inside a group are pairwise trees over a fixed leaf count, so results do not depend on the group size or the plane size; `sync_cube` only in uniform control flow (ADR-0042)."
  - **"Benchmarks":** a new bullet, **Degree sweep (M9h):** item 12, with the record in `docs/perf/high-order.md` and `gridcl-geometry/degree-policy.toml`, raw JSON under `docs/perf/data/`, and CUDA resources from item 6.
  - **"Tabulation and quadrature":**
    - after the device-evaluation bullet: "Tensor rules keep their 1D (or triangle × 1D) factors; sum-factorised kernels use 1D tabulations and a lexicographic node permutation, uploaded with the tabulation (ADR-0042)";
    - the f32 tolerance remark goes to "Testing and validation".
  - **"Geometric factors":** after the list: "Each factor family runs one of the strategies direct, loops, sum-factorised (quadrilaterals, hexahedra, prisms on tensor rules), shared or cooperative, chosen per family, cell type, degree range and device class by gridcl-geometry's degree policy (`degree-policy.toml`, `docs/perf/high-order.md`); `GRIDCL_GEOM_STRATEGY` overrides it for tests and benchmarks (ADR-0042)."
  - **"Pull-back":** a new bullet: "The pull-back runs the same strategies. On quadrilaterals and hexahedra, the sum-factorised pull-back evaluates the 1D Lagrange bases at ξ (barycentric form) and contracts the E-vector one dimension at a time, with no expansion coefficients; prisms combine the triangle's coefficients with the 1D interval basis (ADR-0042)."
  - **"Testing and validation":**
    - **Tolerances** gains item 2's degree scaling, with the amplification factors defined there;
    - **Parallel invariance** gains "geometric factors and pull-back results per cell origin id (from M9h; ADR-0042)";
    - **Vector-size invariance** gains item 8's case for p = 3–8.
  - **"Cargo features":** "Test-support feature" adds gridcl-geometry's `testing` (the hidden `gridcl_geometry::testing`, enabled by gridcl-testing). "Benchmark-support feature" adds gridcl-geometry's `bench` (enabled by gridcl-bench).
  - **"Handing this design to Claude Code"**, the copy list (and CLAUDE.md): "and of quadrature points and tabulations when a rule is created or a tabulation is first cached" becomes "… first cached, including the 1D tabulations and lexicographic node permutations of sum-factorised kernels". "the upload of a geometry block's expansion coefficients" becomes "… expansion coefficients, or of the 1D nodes, barycentric weights and coefficient factors of a sum-factorised pull-back".
  - **Roadmap M9h:** the acceptance gains "(read as in ADR-0042: bullet 1 item 2, bullet 2 item 7, bullet 3 item 1, bullet 4 item 11, bullet 5 items 2, 8 and 9, bullet 6 item 12)".
  - **"Development environment and test targets":** no change (Locust, Kathleen and the local Mac as for M3s).
- **Public API changes:**
  - gridcl-device: `LaunchPlan::per_group` and the public field `LaunchPlan::units_per_group`. A struct-literal construction of `LaunchPlan` outside gridcl-device would break; there is none in the workspace.
  - gridcl-geometry: the hidden modules `testing` and `bench` behind new features (not stable API).
  - No signature of an existing public function changes.
- **Acceptance tests affected:** the M9h bullets, read as above; their tests are those of docs/plans/M9h.md. The existing p ≤ 4 tests are unchanged (the scale factor is at most 1.01 there).
- **Later milestones affected:**
  - M10: benchmarks geometric factors per second at p = 1, 4 and 8 with the chosen policy. The vectorization re-run covers the new kernels under their existing `KernelKind`s (EVector, GeomFactors, PullBack); the cooperative kernels' group size is measured there if it changes the vector choice.
  - CLAUDE.md's copy list changes with this ADR's apply commit.

## Decision record

Accepted by the maintainer on 2026-11-08, as proposed (option 2, items 1–13), including item 2's scaling of the f64 tolerance. Applied to docs/design.md and CLAUDE.md in `adr: 0042 apply`. When applied, one slip in Consequences was corrected: the roadmap's reading note names item 9 under bullet 5 (parallel invariance), not bullet 1, adds item 2 to bullet 5 (GPU tolerances) and item 12 to bullet 6 (the decision rule behind `high-order.md`).

Clarified by the maintainer on 2026-11-08 (M9h T0a): in item 2, "per component" means per quantity. The relative error of X, J, det J, K, normals and ξ is each taken against the largest |want| of that quantity over the block, so det J ~ h³ stays relative, while the entries of one matrix share its largest value (rounding errors scale with the matrix, not with each entry; strictly per entry, f32 J fails at p = 3 with no kernel defect). docs/design.md's "largest value of each component" is read the same way.

Amended by ADR-0043 (the Locust baseline of the pull-back families and of the Grace CPU samples p ∈ {1, 2, 4, 8}, pyramids {1, 2}), accepted 2026-11-10.

Amended by ADR-0044 (the tolerances of K and the pull-back scale with κ_F(J) / TD), accepted 2026-11-10.

Amended by ADR-0045 (CPU cubes of per-group plans hold several groups), accepted 2026-11-10.

Amended by ADR-0046 (items 2, 8 and 11: the pull-back clamps its iterates to the reference cell's box widened by 0.1, so `direct` differs from M9 only where an iterate left it; the J-derived and pull-back tolerances scale by n / n_ref and R; the large case's page limit is at least the per-block tables), accepted 2026-11-11.

Amended by ADR-0049 (items 1 and 5: the tensor pull-back contracts modal coefficients formed by a sum-factorised set-up, a = (C₁′ ⊗ C₁′ ⊗ C₁′)x, prisms C′_tri ⊗ C₁′, so rule 1's set-up bound applies to it; it uploads C₁′, C′_tri and the node permutation instead of 1D nodes and barycentric weights), accepted 2026-11-12.

Amended by ADR-0050 (items 7 and 12: the degree sweep, timed and launch-only, skips `direct` on hexahedra, prisms and tetrahedra at p ≥ 5; T3's baseline gives its cost there), accepted 2026-11-14.

Amended by ADR-0051 (items 1, 7 and 12: the degree sweep, timed and launch-only, leaves out Locust's Grace CPU, so CpuSimd128 is measured on the Mac mini alone; Grace keeps the drift baseline and item 11's paired check), accepted 2026-11-14.

Amended by ADR-0052 (items 1, 11 and 12: on the CPU classes the degree sweep times `direct`, `loops` and `sumfact` only, at 10^5 and 10^6 points with medians of five runs, and the rules use 10^6 points), accepted 2026-11-14.
