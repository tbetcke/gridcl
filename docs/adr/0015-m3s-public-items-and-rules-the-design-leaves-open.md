# ADR-0015: M3s public items and rules the design leaves open

- **Status:** Accepted
- **Date:** 2026-10-24
- **Raised in:** M3s planning (docs/plans/M3s.md, "Planned public API and rules", questions 2, 3, 4, 6 and 14)
- **Design sections affected:**
  - "Architecture": Listing B4b (the `QuadField` comment on `VectorPolicy::quad_tile`) and Listing B5 (`quad_tile()`);
  - "Device model and backends / Device primitives": the "one partition per primitive" bullet;
  - "Vectorization and CPU execution":
    - Listing S2 (`KernelKind`, `VectorPolicy`, the `testing` module);
    - the "CPUs" bullet after Listing S2;
    - "Where vectors apply" (the paragraph after the table);
    - "Benchmarks" ("Decision rule" and "Record");
  - "Workspace layout": the crate table (gridcl-topology, gridcl-bench) and "Cargo features";
  - "Implementation roadmap": M3s (acceptance bullet 3) and M10 (the re-run bullet).

## Context

M3s measures where SIMD vectors pay off. It writes `gridcl-device/vector-policy.toml` and
`docs/perf/vectorization.md` from benchmark data. "Vectorization and CPU execution / Benchmarks"
gives the protocol:

> **What it runs:** every candidate family at every vector size the device offers, at QuadField
> tile widths 1, 4 and 16 for geometry kernels, and on CPUs at block sizes of 4, 16 and 64 KiB.

> **Decision rule:** for each kernel family, device class and scalar type, a vector size above 1
> is adopted only if it is at least 10% faster than size 1 at both 10^6 and 10^7 items. Among the
> configurations within 3% of the fastest, the smallest vector size is chosen […]. Tile width and
> CPU block size follow the same rule.

M3s acceptance bullet 3 then requires:

> every vector size above 1 in `vector-policy.toml` is at least 10% faster than size 1 at 10^6 and
> 10^7 items on its device class, and every chosen configuration (vector size, tile width, block
> size) is within 3% of the fastest one measured.

Bullet 4 adds that "every value is a power of two from 1 to 16".

Planning read these sections against the code of M2 and M3. The relevant facts:

- **`VectorPolicy`** (gridcl-device/src/policy.rs) holds a vector size per (`KernelKind`, element size) and one `quad_tile` per device class.
- **`LaunchPlan::elementwise`** (plan.rs) sizes CPU blocks from a private constant, `CPU_BLOCK_BYTES = 16 KiB`. Nothing can change it.
- **The `testing` module** has `with_properties`, `with_vector_size`, `with_quad_tile` and `record_plans`. `with_vector_size` applies to every plan made inside its scope.
- **Composite primitives plan several kinds.** `radix_sort` plans SortDigits and calls `exclusive_scan`, which plans ScanTile.
- **TopologyKeys and PermutationInfo are private.** Key emission runs inside `Topology::create_entities`, and PermutationInfo inside `Topology::new`.
- **Scalar kernels borrow candidate kinds.** The scalar primitives `histogram`, `segmented_reduce`, `scatter` and `run_length`/`unique_by_key` plan with candidate kinds (Reduce or Gather), and so do scalar helper kernels in gridcl-topology and gridcl-geometry (Fill, Copy, Gather, Reduce). The M3 review flagged this as m8.

## Problem

1. **Block sizes have no hook and no place in the policy.**
   - The protocol measures 4, 16 and 64 KiB, and the rule chooses a block size ("Tile width and CPU block size follow the same rule").
   - "Record" puts the choices into `vector-policy.toml`.
   - But Listing S2 has no hook to set the block size and no policy field to hold it. The constant cannot be changed even for a benchmark.
2. **One block size and one tile per class cannot meet bullet 3.**
   - The rule decides per family and scalar type, and bullet 3 requires every chosen configuration to be within 3% of the fastest.
   - With one block size per class, a family whose best block differs from the class's choice fails bullet 3.
   - Likewise, `quad_tile()` is one value per class, shared by f32 and f64 GeomFactors. If f32 prefers tile 16 and f64 tile 4, one of them fails.
3. **No per-family vector size.** `with_vector_size(Fixed(n))` changes every plan in its scope. A SortDigits measurement therefore also changes the ScanTile kernels of the `exclusive_scan` inside `radix_sort`, and the measured time mixes two families.
4. **Key emission and PermutationInfo cannot be timed alone.**
   - Through the public API, TopologyKeys runs only inside `create_entities`. There the radix sort dominates: 2.1 s for 10^7 two-word keys on the M3 Max CPU runtime (docs/perf/m2-primitives.md).
   - PermutationInfo runs only inside `Topology::new`, which also uploads the origin ids from the host and builds an `IndexMap`.
   - Timed through those calls, a kernel speedup of 2× can show as a few percent, below the 10% threshold, and the policy would stay at 1 on noise.
5. **The decision rule is ambiguous for classes measured on several devices.**
   - CpuSimd128 is measured on the M3 Max and on Locust's Grace CPU; Gpu on Metal and the H100.
   - The rule says "on its device class" but not how two devices combine.
   - Metal also offers at most 4 lanes and no f64, so some sizes and types exist on only one device of the class.
6. **The 10% rule and the 3% bullet conflict.**
   - Say size 4 is 6% faster than size 1 at 10^6 and 10^7.
   - The 10% rule forbids size 4, so size 1 is chosen. But size 1 is then 6% slower than the fastest configuration measured, and bullet 3 ("within 3% of the fastest one measured") fails.
   - As written, both cannot hold whenever a larger size is 3–10% faster.
7. **Misattributed scalar kernels (m8).**
   - `record_plans` reports scalar helper kernels under candidate kinds.
   - Once block sizes are per kind (problem 2), these helpers would also take a candidate family's block size, chosen from measurements that never included them.
   - The design's "one partition per primitive" rule says a primitive with no vectorized phase plans "for its first phase", but the kind it should use for that phase is unnamed.

## Options

1. **Keep the design as written.**
   - One block size (the constant) and one tile per class.
   - Benchmarks change the block size only by editing the constant between runs.
   - Composite primitives are measured with every kind forced at once.
   - TopologyKeys and PermutationInfo are timed through `create_entities` and `Topology::new`.
   - The two-device and 3%/10% questions stay open.
   - Not chosen. Bullet 3 cannot be met in general (problems 2 and 6). The protocol needs code edits between runs. Diluted measurements would keep the mesh families at 1 whatever their kernel gains.
2. **Add the hooks and policy fields that M3s needs, and fix how the rule reads** (proposed). The items are listed under "Proposed decision".
3. **As option 2, but coarser policy granularity.**
   - One block size and one tile per class, chosen jointly over all families and types (for example by the geometric mean of time ratios).
   - Bullet 3 is relaxed to "within 3% for the family set as a whole".
   - This avoids changing `quad_tile`'s signature, but bullet 3 would have to be amended. Not chosen.
4. **Split device classes per device** (for example `GpuMetal`, `GpuCuda`, `CpuApple`, `CpuGrace`) instead of combining devices (problem 5).
   - This changes `DeviceClass` and `for_client` in Listing S2, and needs a rule for unmeasured devices of each vendor.
   - Not chosen: the design's classes are by SIMD width on purpose, and a new device would have no policy.
5. **Per-kernel times from CubeCL's profiling log** (`[profiling] logger`, level `basic`) instead of benchmark hooks (problem 4).
   - No API change.
   - Not chosen: the logged times' accuracy on the CPU runtime is unknown, and the extra syncs disturb what is measured.

## Proposed decision

Option 2. Each item answers the problem of the same number. The maintainer chose these in M3s
planning (2026-10-24).

1. **Block sizes in the policy, and a hook.**
   - Each CPU class of `vector-policy.toml` gets a table `[<Class>.cpu_block]`, laid out like `vector_size`: one entry per `KernelKind` and element size, in units of 4 KiB (1, 4 or 16, so bullet 4's "power of two from 1 to 16" holds as written).
   - `VectorPolicy::cpu_block_bytes(&self, kind: KernelKind, elem_bytes: usize) -> Option<usize>` returns the block in bytes, and `None` on `Gpu`.
   - `LaunchPlan::elementwise` uses it on CPUs in place of the constant, and still rounds to whole 128-byte lines as the "CPUs" bullet says. Until M3s's data, every entry is 4 (16 KiB), so no launch changes.
   - `testing::with_cpu_block_bytes<R>(bytes: usize, f: impl FnOnce() -> R) -> R` overrides the block for plans made inside `f`. The value must be a power of two from 4 KiB to 64 KiB; anything else panics. The override is scoped and thread-local, like the other hooks.
2. **Tiles per element size.**
   - `VectorPolicy::quad_tile(&self, elem_bytes: usize) -> usize`, with `quad_tile = { 4 = 16, 8 = 16 }` per class.
   - Factor kernels and `QuadField` pass `size_of::<F>()`.
   - `with_quad_tile` and `GRIDCL_QUAD_TILE` override both sizes.
3. **A per-family vector size.**
   - `testing::with_kind_vector_size<R>(kind: KernelKind, v: VectorSizeOverride, f: impl FnOnce() -> R) -> R`.
   - For `kind` it takes precedence over `with_vector_size`, `GRIDCL_VECTOR_SIZE` and the policy. Other kinds are unaffected.
   - `Fixed(n)` and `Widest` keep their clamps (ADR-0011).
4. **Benchmark hooks for the private mesh families.** gridcl-topology gets a Cargo feature `bench`, off by default and enabled only by gridcl-bench. It adds `#[doc(hidden)] pub mod bench` with:
   - `emit_keys(topology: &Topology, dim: usize) -> Result<BenchOutput>`: key emission of `create_entities(dim)` ("Entity creation", step 1);
   - `permutation_info(topology: &Topology, b: usize) -> Result<BenchOutput>`: the PermutationInfo launch of cell block `b`.

   Each runs exactly the family's launch sequence, including the cell-fastest connectivity copy (item 8), on inputs already on the device. `BenchOutput` is an opaque owner of the output buffers, so the caller can sync, time and drop it. The module is not part of the library API: it carries no stability promise and is hidden from the docs. Feature unification turns it on in `--workspace` builds, where it only adds functions.
5. **Classes measured on several devices.**
   - For each configuration the rule uses the geometric mean, over the class's measured devices, of the per-device time ratios.
   - A device that cannot run a vector size (Metal above 4 lanes) contributes its time at the size its launch is clamped to.
   - A device that lacks the scalar type (f64 on Metal) contributes nothing for that type.
   - The report also lists the per-device ratios.
   - CpuSimd256 copies the CpuSimd512 table unless an AVX2-only machine is measured, as "Where" already says.
6. **The 3% rule chooses among what the 10% rule admits.**
   - A configuration is (vector size, tile, block); the tile applies only to GeomFactors, the block only to CPU classes.
   - The admitted configurations are those at size 1, plus those at any larger size that is at least 10% faster than size 1 at 10^6 and at 10^7 items. A size's time is its best over tiles and blocks, and size 1's time is likewise its best.
   - Among the admitted configurations within 3% of the fastest admitted one, the smallest vector size is chosen, then the smallest tile, then the smallest block.
   - Bullet 3's "within 3% of the fastest one measured" reads "of the fastest admitted one".
   - **Shared 4-byte entries.** f32 and u32 share one policy entry (Fill, Reduce, Gather). That entry takes a size only if it is admitted for every 4-byte type measured; otherwise it takes the smaller size.
   - **Element sizes no family uses.** These entries stay 1: the 8-byte entries of Copy (which moves u32 words), ScanTile, SortDigits, TopologyKeys and PermutationInfo.
   - **Register budget.** Under `Policy` the register budget may lower a size at run time. The TOML records the size that `record_plans` shows ran under the chosen policy, so bullet 3 checks what actually runs.
7. **`KernelKind::Scalar`** for every kernel that is not on the candidate list.
   - Its `vector_size` entries are 1 in every class, which `parse` enforces. Its `cpu_block` entries are chosen like any other kind's, from the time of the scalar primitives (`histogram`, `segmented_reduce`).
   - **Primitives.** A primitive with no vectorized phase plans with `Scalar`. A primitive with a vectorized phase keeps one plan, of that phase's kind, for all its phases (unchanged).
   - **Not benchmarked.** `Scalar` is not a candidate family, so it is outside the vector-size study.
8. **Rules fixed here that change no design text:**
   - **Cell-fastest copy.** At N > 1, TopologyKeys and PermutationInfo build a transient, private `[local vertex][padded cell]` copy of the cell–vertex connectivity, and of the origin-id words for PermutationInfo. They read it with vector loads and drop it after the launch. At N = 1 they read the cell-major connectivity as before.
   - **The transpose.** It is an indirect copy, so it plans as `Scalar`, but the benchmark hooks time it with the family ("measured including the transpose").
   - **One timed region per family.** Each family is timed by the smallest call whose launches all take their shape from the family's plan:
     - the primitive itself for Fill, Copy, Reduce and Gather;
     - `inclusive_scan` and `exclusive_scan` for ScanTile;
     - `radix_sort` for SortDigits, with its inner `exclusive_scan` held at its default through `with_kind_vector_size`;
     - the hooks of item 4 for TopologyKeys and PermutationInfo;
     - `DeviceGrid::evector` on a fresh clone for EVector;
     - the block functions for GeomFactors.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing S2:**
    - `KernelKind` gains `Scalar`, commented "kernels not on the candidate list; always vector size 1".
    - `VectorPolicy`'s comment reads "Vector sizes, QuadField tiles and CPU block sizes chosen from benchmarks …, per (KernelKind, DeviceClass, element size)".
    - `pub fn quad_tile(&self, elem_bytes: usize) -> usize;` replaces `quad_tile(&self)`.
    - It adds `/// CPU block of one kind and element size in bytes (vector-policy.toml stores units of 4 KiB); None on Gpu.` with `pub fn cpu_block_bytes(&self, kind: KernelKind, elem_bytes: usize) -> Option<usize>;`.
    - The `testing` module adds:
      - `/// Vector size of one family; takes precedence over with_vector_size, the environment and the policy for that kind.` with `pub fn with_kind_vector_size<R>(kind: KernelKind, v: VectorSizeOverride, f: impl FnOnce() -> R) -> R;`;
      - `/// CPU block target, a power of two from 4 to 64 KiB, for plans made inside f.` with `pub fn with_cpu_block_bytes<R>(bytes: usize, f: impl FnOnce() -> R) -> R;`.
    - The module's precedence sentence becomes: "the scoped per-kind override, then the scoped override, then the environment variable, then vector-policy.toml".
  - **"CPUs" bullet:** "… blocks of `items_per_unit` consecutive vectors, `VectorPolicy::cpu_block_bytes` of the widest stream (16 KiB until M3s), rounded to a whole number of 128-byte lines …; the block size is chosen per kind and element size by the M3s benchmarks (ADR-0015)."
  - **Listing B4b:** the `QuadField` comment's "VectorPolicy::quad_tile" becomes "VectorPolicy::quad_tile(size_of::<F>())".
  - **Listing B5:** `let tile = VectorPolicy::for_client(g.client()).quad_tile(size_of::<F>());`.
  - **"Device primitives":** the "one partition per primitive" bullet reads "… for the `KernelKind` of its vectorized phase, or `KernelKind::Scalar` if none is vectorized, …".
  - **"Where vectors apply":** the last sentence after the table reads "All other kernels are scalar, plan as `KernelKind::Scalar`, and still take their CPU work distribution from `LaunchPlan`."
  - **"Benchmarks", "Decision rule":** append "With several devices in one class, the rule uses the geometric mean of the per-device time ratios; the 3% rule chooses among the configurations the 10% rule admits; a 4-byte entry shared by f32 and u32 must be admitted for both (ADR-0015)."
  - **"Benchmarks", "Record":** "the choices — vector sizes, QuadField tiles per element size, and CPU block sizes per kind and element size in units of 4 KiB — go to `gridcl-device/vector-policy.toml`".
  - **"Workspace layout":**
    - the gridcl-topology row adds "benchmark hooks (feature `bench`)";
    - the gridcl-bench row's dependencies read "gridcl, gridcl-device, gridcl-topology (feature `bench`), gridcl-testing; cubecl with runtime features; serde_json, toml";
    - "Cargo features" adds: "**Benchmark-support feature** (on gridcl-topology; not a library feature): `bench` enables the hidden `gridcl_topology::bench` module. gridcl-bench turns it on."
  - **Roadmap M3s, bullet 3:** "… at least 10% faster than size 1 at 10^6 and 10^7 items on its device class (the geometric mean over its measured devices), and every chosen configuration (vector size, tile width, block size) is within 3% of the fastest one the 10% rule admits (ADR-0015)".
  - **Roadmap M10:** "the M3s study covers every candidate `KernelKind`, including Pack and PullBack".
- **Public API changes:**
  - gridcl-device: `KernelKind::Scalar`; `VectorPolicy::quad_tile` takes `elem_bytes`; `VectorPolicy::cpu_block_bytes`; `testing::{with_kind_vector_size, with_cpu_block_bytes}`.
  - gridcl-topology: the hidden `bench` module behind feature `bench`.
  - `vector-policy.toml` gains `cpu_block` tables, per-size `quad_tile` entries and `Scalar` rows.
  - The one breaking change is `quad_tile`'s signature. Its callers are gridcl-geometry (layout.rs) and doc examples in gridcl-device; there are no external users.
- **Acceptance tests affected:**
  - M3s bullet 3 is read as in items 5 and 6; `perf_record::policy_follows_the_decision_rule` checks that reading.
  - Bullet 4's test (`policy_covers_every_kind_and_class`) also covers `cpu_block`, the per-size `quad_tile` and `Scalar`.
  - The M2 unit tests `the_policy_file_parses_with_one_and_sixteen_everywhere`, `cpu_blocks_are_whole_lines_of_about_16_kib` and `cpu_plan_one_unit_per_thread_whole_lines` change: they read the policy instead of constants. Their assertions are otherwise unchanged.
  - No M2 or M3 acceptance test changes meaning. The invariance tests also cover the cell-fastest path at N > 1.
- **Later milestones affected:**
  - M6's pack and unpack kernels plan as Pack, and the scalar parts of `Scatterer` as `Scalar`.
  - M9's pull-back plans as PullBack.
  - M10's re-run uses the same rule and records blocks and tiles in the same form.
  - A later `cube_dim` policy (performance cores, docs/plans/M3s.md question 8) would need its own ADR.

## Decision record

Accepted by the maintainer on 2026-10-24, as proposed (option 2, items 1–8), without conditions. Applied to docs/design.md in `adr: 0015 apply`.
