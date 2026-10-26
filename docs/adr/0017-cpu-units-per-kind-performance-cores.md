# ADR-0017: CPU units per kind: all hardware threads or the performance cores

- **Status:** Accepted
- **Date:** 2026-10-25
- **Raised in:** M3s T14 (the m6 follow-up; docs/plans/M3s.md question 8)
- **Design sections affected:** "Vectorization and CPU execution": Listing S2 (`VectorPolicy`, `testing`), the "CPUs" bullet of `LaunchPlan`, "Benchmarks" ("Conditions", "Decision rule", "Record"); "Development rules" (the FFI rule, mirrored in CLAUDE.md); "Workspace layout" (the gridcl-device row of the crate table); roadmap acceptance of M3s (bullets 3 and 4).

## Context

"Vectorization and CPU execution", `LaunchPlan`, "CPUs":

> `cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`, ADR-0011), one unit per hardware thread.

"Benchmarks", "Conditions":

> With static blocks, efficiency cores can finish last; M3s therefore also measures `cube_dim` equal to the performance-core count, and the policy may choose it.

No policy key, API or detection exists for that choice. M3s question 8 kept the performance-core `cube_dim` out of the decision rule's configuration and asked for a separate ADR "for detection and a policy key" if it passes the rule on the M3 Max. ADR-0015 says the same. The CubeCL CPU runtime reports `max_units_per_cube` = 16 on the M3 Max (12 performance and 4 efficiency cores) and no performance-core count. Reading one on macOS (`sysctlbyname("hw.perflevel0.physicalcpu")`) is an FFI call, which the development rules allow only in gridcl-parallel (MPI) and gridcl-device (GPU-aware).

T14 also had to study the M3 review follow-up m6: debug `cargo test --workspace` on the M3 Max took 16:01 against a budget of about 15 min, mostly in `primitives_prop`.

## Problem

T14's measurements (docs/perf/vectorization.md, "CPU `cube_dim` and test times") show three things.

1. **Large single launches are faster on the performance cores.** In a preliminary release sweep (vector sizes 1 and 4, 16 KiB blocks, 10^6 and 10^7 items), the median at `cube_dim` 12, divided by the median at 16, was:
   - Reduce: 0.11–0.51;
   - ScanTile: 0.34–0.50;
   - SortDigits: 0.60–0.75;
   - TopologyKeys: 0.36–0.52;
   - Fill, Copy and Gather: 0.42–0.99.

   GeomFactors (f64), which is compute-bound and has no barriers, was slower at 12: 1.11–1.23. The kernels with `sync_cube` barriers or several phases gain the most, because every barrier waits for the units on efficiency cores. That sweep ran with about 1.5 cores of macOS background load, so it does not meet T15's conditions. It is consistent with the debug test times, though: `primitives_large` took 90 s at 16 units and 38 s at 12.
2. **One rule for every kind costs some families.** Setting `cube_dim` to the performance-core count everywhere would cost GeomFactors 11–23% on the M3 Max. Keeping all threads everywhere costs Reduce, ScanTile and SortDigits up to 9×.
3. **The test-time budget has a second cause.** Debug `primitives_prop` took 541 s at 16 units with libtest's 16 test threads, but 96 s with one test thread: up to 256 workers on 16 cores wait at every `sync_cube`. The budgets can therefore be met without a library change: with 4 test threads, the workspace took 9:56. They can also be met through this ADR: at `cube_dim` 12 the workspace took 9:32.

So the design's "the policy may choose it" cannot be carried out: there is nothing to choose with and no way to detect the count. Acceptance bullet 3 fixes the configuration as "(vector size, tile width, block size)", which leaves the unit count out.

## Options

1. **Keep the design as written.** `cube_dim` stays at all hardware threads; "the policy may choose it" stays unimplemented; T17 records the performance-core runs in the report only. The m6 budget is met on the test side by limiting test threads: `RUST_TEST_THREADS` in the hosted cpu job, and locally if needed. That is a test-configuration change with no ADR. No API or FFI change. Reduce, ScanTile, SortDigits and TopologyKeys keep running up to 9× slower on the M3 Max, and the M10 re-run would revisit it.
2. **One rule: the performance cores where known.** `cube_dim` is the performance-core count on CPUs that report one, else `max_units_per_cube`. No policy key; it needs the detection and FFI change of option 3. GeomFactors loses 11–23% on the M3 Max. The decision rule is unchanged.
3. **A per-kind policy entry (proposed).**
   - Each CPU class gets `[<Class>.cpu_units]`, one entry per `KernelKind` and element size, like `cpu_block`, with the values `"all"` or `"performance"`.
   - On CPUs, `LaunchPlan::elementwise` uses the performance-core count p when the entry says `performance` and the device reports p < `max_units_per_cube`; otherwise it uses `max_units_per_cube`, as now.
   - gridcl-device reads p once per process. On macOS this is `sysctlbyname("hw.perflevel0.physicalcpu")` through a safe wrapper, with the declaration from the `libc` crate; everywhere else, and on any error, there is no count. Grace and the Xeons have one kind of core, so their plans do not change.
   - The decision rule's configuration becomes (vector size, tile, block, units). The units are chosen like tile and block, by the 3% rule among the admitted configurations, with ties going to `all` (the current rule). A device without a performance-core count gives the same time for both settings.
   - T15 and T16 are unaffected: `vector_sweep --cube-dim auto` already measures both settings on the M3 Max and only `all` elsewhere.
4. **A per-class entry.** Like option 3, but one value per CPU class. Simpler, but GeomFactors and Reduce share CpuSimd128 and want opposite settings, so one of them loses.

## Proposed decision

Option 3. The design already says that the policy may choose the performance-core count. The T14 data shows that the right choice differs by family, as block sizes do, so the entry follows `cpu_block`'s layout and its rule. The FFI call is one read-only `sysctlbyname` on macOS in gridcl-device, the crate that already owns device properties. The m6 budget on the hosted cpu job (4 vCPUs, one kind of core) is not addressed by this ADR. T18 re-measures it; if it is still over budget, `RUST_TEST_THREADS` is lowered in that job, logged without an ADR, as M3's question 20 handled CI steps.

## Consequences

- Text to change in docs/design.md:
  - Listing S2: `pub enum CpuUnits { All, Performance }`; `VectorPolicy::cpu_units(&self, kind: KernelKind, elem_bytes: usize) -> Option<CpuUnits>` (`None` on `Gpu`); in `testing`, `with_cpu_units<R>(units: CpuUnits, f: impl FnOnce() -> R) -> R`, which overrides the policy for plans made inside `f`, and `with_performance_cores<R>(cores: Option<u32>, f: impl FnOnce() -> R) -> R`, which replaces the detected count.
  - "CPUs" bullet: "`cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`), or the performance-core count where `VectorPolicy::cpu_units` says `Performance` and the device reports a smaller count (macOS); one unit per thread (ADR-0017)."
  - "Benchmarks", "Conditions": the last sentence ends "… and the policy chooses between the two per kind and element size (ADR-0017)."
  - "Decision rule": "Tile width, CPU block size and CPU units follow the same rule; ties in units go to all hardware threads."
  - "Record": "… CPU block sizes and CPU units per kind and element size …".
  - "Development rules" (and CLAUDE.md): "FFI calls only in gridcl-parallel (MPI) and gridcl-device (GPU-aware; the macOS performance-core count)".
  - M3s acceptance bullet 3: "(vector size, tile width, block size, CPU units)". Bullet 4: "… and that every numeric value is a power of two from 1 to 16, and every `cpu_units` value is `all` or `performance`".
- Public API changes: `CpuUnits`, `VectorPolicy::cpu_units`, and `testing::{with_cpu_units, with_performance_cores}`. gridcl-device gains a direct `libc` dependency on macOS (already in the lock file). `LaunchPlan` is unchanged.
- Acceptance tests affected:
  - `policy_covers_every_kind_and_class` (the `cpu_units` entries);
  - `perf_record::policy_follows_the_decision_rule` (both `cube_dim` modes enter, the TOML's units are checked);
  - `cpu_plan_one_unit_per_thread_whole_lines` (the unit count follows the entry);
  - `prop_plan_respects_injected_limits` (also under `with_performance_cores`).
- Plan changes: a new M3s task between T14 and T17 implements the API, the detection, the `decide` and `report` changes and the tests. The `vector_sweep` mode `PerformanceCores` switches from `with_properties` to `with_cpu_units`, with the same effect. T15 and T16 can start before it. Until T17 every entry is `all`, so no launch changes. T18 re-measures the debug workspace time under the chosen policy.
- Later milestones affected: M10's re-run decides the units for Pack and PullBack. Plans made inside `with_properties` with a lower `max_units_per_cube` keep that limit, because p is used only where it is smaller.

## Decision record

Accepted by the maintainer on 2026-10-25, as proposed (option 3), after a discussion of per-system compromise rules, which the maintainer decided against in favour of ADR-0015's per-kind scheme. Applied to docs/design.md and CLAUDE.md in `adr: 0017 apply`; the gridcl-device row of the crate table (the `libc` dependency) was added to the affected sections when applying. Implemented by M3s T20.
