# ADR-0060: M10 public items and rules the design leaves open

- **Status:** Accepted
- **Date:** 2026-11-16
- **Raised in:** M10 planning (`/plan-milestone M10`, open questions 4–10)
- **Design sections affected:**
  - "API walkthrough": Listing B8 (the `local_gpu_index` comment);
  - "Distributed mesh model": Listing S2c (`local_gpu_index`);
  - "Vectorization and CPU execution", "Benchmarks" (**What it runs**: the candidate protocol; **Degree sweep (M9h)**: M10's check);
  - "Workspace layout" (the crate table: tetraloma-testing and a new tetraloma-examples row);
  - Roadmap M10 (the "read as" line).
  - Amends ADR-0053 (Consequences, "Later milestones affected") and ADR-0058 (item 5).

## Context

M10's scope and acceptance (Roadmap M10, read as in ADR-0057, ADR-0058 and ADR-0059) name the items below. The design does not fix any of them:

- `local_gpu_index` (ADR-0026 item 17), whose only appearance is the call in Listing B8: `let gpu = local_gpu_index(&comm); // node-local rank`;
- the candidate protocol of "Benchmarks" (ADR-0057 item 3), with the generic rule of ADR-0058;
- the p = 8 / p = 4 bullet;
- "every example runs on the CPU runtime and on at least one GPU backend";
- "every public item is documented".

M10 planning read the code against main at ae8d644 (`docs/plans/M10.md`, "Findings from planning").

## Problem

1. **`local_gpu_index` has no signature, crate or semantics.** No shared-memory communicator split exists. `tetraloma_testing::test_client()` always opens device 0.
2. **The candidate protocol is ambiguous for generic sets.**
   - "Vector size 1 at the entry's previous tile, block and units" has no single value for a Cpu entry: the three old CPU classes had different tiles, blocks and units.
   - Pack and PullBack were never measured. Their entries are defaults (`tetraloma-bench/src/decide.rs:499`), so their only candidate would be the baseline. The study would "cover" them without measuring anything.
   - The protocol does not say on which mesh and degree the geometry entries are measured, nor how "the configurations M3s or M9h chose for any device class" are found once T1 has replaced the old tables.
3. **The degree merge (ADR-0058 item 5) can multiply the run time.** M9h's records at p = 2, 3 and 5–7:
   - were measured under the old per-class vector policy;
   - are at 10^5 and 10^6 points on CPUs (ADR-0052);
   - have no Grace data (ADR-0051).

   Wherever M10 changes the EVector, GeomFactors or PullBack vector size, many ranges will move by more than 10% from M9h. Each then has to be "re-measured over its degree range", and no run plan budgets that.
4. **M9h bullet 4 conflicts with ADR-0058.** M9h bullet 4 bounds every policy family at 5% over M9's `direct` at p ≤ 4 (ADR-0055). Its test, `no_family_slower_than_m9_at_low_degree`, can contradict ADR-0058's per-platform bound of 1.25 for the generic policy.
5. **Bullet 2 versus ADR-0053.** ADR-0053 makes the p = 8 / p = 4 bound M10's target, "with option 1's GPU kernel as its first candidate". M10's bullet 2 allows "or the gap is reported with its cause". M9h measured the causes:
   - register spills of the sum-factorised kernels on GPUs (20–57 KB stack per thread at p = 8);
   - L1 overflow on CPUs;
   - the pull-back's (p + 1)^d step.

   A new kernel would need about two tasks, M9h's correctness and invariance tests at p = 1–8, and new GPU measurements.
6. **Examples need a home.** The workspace has none (only `ptscotch-sys/examples/threading_probe.rs`, built and not run). Dependency rule 4 allows runtime features only in "tetraloma-testing, examples, benchmarks and binaries", so a library crate cannot pick a backend for its examples.
7. **"Every public item is documented" has no defined scope.** `RUSTDOCFLAGS="-W missing_docs"` reports 223 items at default features: tetraloma 12, tetraloma-device 60, tetraloma-geometry 6, tetraloma-testing 11, tetraloma-bench 134. It does not say whether the unpublished crates count, nor which features are checked.

## Options

1. **Keep the design as written.** Every point above is then decided during the implementation, item by item, and the result is visible only in the code. Points 2, 3 and 4 decide which data choose the shipped tables, and point 5 contradicts an accepted ADR. Not chosen.
2. **Fix each point now (chosen by the maintainer in planning, 2026-11-16).** See the proposed decision.

## Proposed decision

Option 2, item by item.

1. **`local_gpu_index`** (tetraloma-parallel, re-exported in `tetraloma::prelude`):

   ```rust
   /// Index of this rank's GPU on its node: the rank within the node. Collective.
   pub fn local_gpu_index(comm: &dyn CommDyn) -> usize;
   ```

   - **With `comm.raw() = Some`:** the rank in `MPI_Comm_split_type(raw, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL)`. The split communicator is freed before return. This is an FFI call in tetraloma-parallel, with a SAFETY comment.
   - **Otherwise:** `comm.rank()`. Serial returns 0, and the ranks of an in-process communicator share one node.
   - It does not know the GPU count; Listing B8's `or_else` chain handles a missing device.
   - **tetraloma-testing (unpublished)** gains `test_client_at(index: usize) -> Client`. It opens GPU `index` on CUDA and HIP and ignores the index on the other backends. The existing MPI tests keep `test_client()`.
   - **Test:** it computes the expected index independently: an all-gather of a hash of each rank's host name, counting lower ranks on the same host. It therefore holds on Kathleen's two nodes.
2. **Candidate protocol** (with ADR-0057 item 3 and ADR-0058 item 4):
   - **Baselines.** The baseline of a Cpu entry is vector size 1 at the old CpuSimd128 entry's tile, block and units. The interim `[Cpu]` set of the collapse takes these values. A Gpu entry's baseline is size 1 at the Gpu entry's tile.
   - **Pack and PullBack** measure the baseline, the widest size, and the M3s choices of the nearest measured kind (Gather for Pack, GeomFactors for PullBack). They do not run the full M3s grid, which ADR-0057 item 2 excludes.
   - **Measurement set-up.** Geometry entries use M3s's family definitions: GeomFactors and EVector on P2 tetrahedra and P2 triangles in 3D, and PullBack on P2 tetrahedra under the current degree policy.
   - **Candidate sources.** "Chosen by M3s or M9h for any device class" means the output of the unchanged M3s rule (`decide.rs`) and M9h rule (`decide_degree.rs`) on the committed M3s and M9h records. The pre-M10 TOML files are kept as `tetraloma-bench/history/{vector,degree}-policy-pre-m10.toml`. The M3s and M9h record tests check against these files and keep the old class names through a crate-private `LegacyClass`.
   - **Records.** M10 records go to `docs/perf/data/m10-*.json`. Each carries the platform, the class, `load_width`, the core counts and the performance levels (ADR-0058 item 1).
3. **Degree merge** (amends ADR-0058 item 5):
   - **Comparison with M9h.** "Moves by more than 10% from M9h's record" compares at 10^6 points, per platform and strategy.
   - **Degrees in between.** p = 2, 3 and 5–7 come from M9h's records, at their own rule sizes (10^6 on CPUs, ADR-0052; 10^6 and 10^7 on GPUs), on the platforms that measured them (no Grace).
   - **Records left out.** M9h records whose `check_error` is not finite.
   - **Re-measurement.** A range that misses the bound or moves by more than 10% is re-measured, with candidates only, at the inner degrees next to each of its boundaries, not at every degree of the range.
4. **M9h bullet 4.** `no_family_slower_than_m9_at_low_degree` stays as a history test against `history/degree-policy-pre-m10.toml`. The generic policy is bound only by ADR-0058's rule. `docs/perf/high-order.md` lists, per platform, the M10 policy's ratio to `direct` at p ≤ 4.
5. **Bullet 2 is met by reporting** (amends ADR-0053). M10 adds no new strategy. `docs/perf/high-order.md` reports the p = 8 / p = 4 ratio per platform, cell type and family against ADR-0042 item 1's bounds, and the cause of every miss, with M9h's evidence and M10's records. ADR-0053's "with option 1's GPU kernel as its first candidate" becomes a candidate for a later performance milestone.
6. **Examples** live in a new unpublished workspace crate, `tetraloma-examples`.
   - **Features:** runtime features `cpu` (default), `metal`, `cuda` and `vulkan`, which forward to cubecl as in tetraloma-testing, and `mpi`.
   - **Examples:** `read_gmsh` (Listing B8, serial), `surface_submesh` (Listing B9), `geometric_factors` (p = 1, 4 and 8 on a curved sphere), `halo_exchange` (Listing B13, feature `mpi`, with `local_gpu_index`) and `vtk_output`.
   - **Structure:** each example is a thin `main` that picks its device as Listing B8 does (one `cfg` per compiled-in runtime) and its precision per device (f32 where f64 is unsupported). It then calls a public `run(&Client)` of the crate's library.
   - **Tests:** `tests/examples.rs` runs every `run` on `test_client()`. `scripts/run-examples.sh` runs the binaries, with `halo_exchange` under `mpirun -n 2`. CI's cpu job runs the serial examples, and its mpi jobs run `halo_exchange`.
7. **Rustdoc.** `missing_docs = "deny"` goes in `[workspace.lints.rust]`, and every workspace crate inherits it, unpublished ones included. Only bindgen-generated modules have `#[allow(missing_docs)]`.
   - `scripts/check-docs.sh` runs `cargo doc --workspace --no-deps` with `RUSTDOCFLAGS="-Dwarnings -Dmissing_docs"`, twice: at default features, and with `tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis`.
   - CI runs it in the cpu job, and the partitioner set in the ptscotch job.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B8:** the comment "arrives with M10 (ADR-0026)" becomes "node-local rank (ADR-0026, ADR-0060)".
  - **Listing S2c:** after `owner_hash`, add item 1's declaration with its doc comment.
  - **"Benchmarks", What it runs:** after the candidate list, add "A Cpu entry's baseline uses the old CpuSimd128 tile, block and units; Pack and PullBack, never measured before M10, also measure the widest size and the M3s choices of Gather and GeomFactors; geometry entries use M3s's P2 meshes; the M3s and M9h choices come from their unchanged rules on the committed records (ADR-0060)."
  - **"Benchmarks", Degree sweep (M9h):** the last sentence becomes "A family and cell type whose check misses the bound, or whose time at 10^6 points moves by more than 10% from M9h's record, is re-measured with candidates only at the inner degrees next to each boundary of its range (ADR-0058, ADR-0060)."
  - **"Workspace layout", crate table:** the tetraloma-testing row gains `test_client_at`. A new row: "tetraloma-examples (unpublished) | The examples of the API walkthrough, each over a tested `run(&Client)`; forwards runtime features | tetraloma, tetraloma-testing; cubecl with runtime features; mpi (optional)".
  - **"Cargo features", runtime features:** "(on tetraloma-testing and examples)" becomes "(on tetraloma-testing, tetraloma-examples and tetraloma-bench)".
  - **Roadmap M10:** "(read as in ADR-0057 and ADR-0058)" becomes "(read as in ADR-0057, ADR-0058, ADR-0059 and ADR-0060)".
- **ADR-0053:** a closing note that M10 meets its target by reporting (ADR-0060 item 5).
- **ADR-0058:** a closing note on item 5's re-measurement scope (ADR-0060 item 3).
- **Public API changes:**
  - `tetraloma_parallel::local_gpu_index`, re-exported in `tetraloma::prelude`;
  - `tetraloma_testing::test_client_at`, unpublished;
  - the crate `tetraloma-examples`, unpublished;
  - documentation only, otherwise.
- **Acceptance tests affected:**
  - M10's tests as planned: `local_gpu_index_is_the_node_local_rank`, `local_gpu_index_serial_is_zero`, `tetraloma-examples::examples`, `scripts/check-docs.sh`, and the candidate and generic-rule tests of tetraloma-bench;
  - `tensor_cost_ratio_within_bound_or_reported` reports and does not enforce;
  - M9h's record tests become history tests.
- **Later milestones affected:** a later performance milestone may take up ADR-0053's GPU kernel and M10's reported gaps.

## Decision record

Accepted by the maintainer on 2026-11-16, as proposed (option 2, items 1–7); the resolutions were chosen in M10 planning ("all questions as recommended").

Applied in `adr: 0060 apply`: docs/design.md (Listing B8, Listing S2c, "Benchmarks": What it runs and Degree sweep (M9h), the crate table, "Cargo features", Roadmap M10's reading line), closing notes in ADR-0053 and ADR-0058, and issue #25.

Note (ADR-0064, accepted 2026-11-18): item 2's `LegacyClass` is `pub` in the unpublished tetraloma-bench, so that its tests in `tetraloma-bench/tests/` can read it (M10 Deviations log, T1).

Amended by ADR-0065 (accepted 2026-11-18): from M10c, item 2's candidate sources and the wider sweep no longer apply to `[Cpu]` vector entries, which measure ADR-0065's grid.
