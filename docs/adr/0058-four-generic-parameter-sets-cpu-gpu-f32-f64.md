# ADR-0058: Four generic parameter sets: CPU and GPU, f32 and f64

- **Status:** Accepted
- **Date:** 2026-11-16
- **Raised in:** before M10 planning (maintainer request, 2026-11-16)
- **Design sections affected:** "Fixed decisions" item 11; Requirements, **Development and test targets**; "Launch plans and the vector policy" (the `DeviceClass` listing); "Geometric factors" (degree policy); "Benchmarks" (**Where**, **Decision rule**, **Record**, **Degree sweep (M9h)**); "Development environment and test targets" (the Kathleen row and the CPU-only paragraph); Roadmap M10 (scope, acceptance bullets 2 to 5). Process documents: CLAUDE.md (the vector-size bullet and the Kathleen line), the hpc-test skill.

## Context

"Launch plans and the vector policy" defines `DeviceClass { CpuSimd128, CpuSimd256, CpuSimd512, Gpu }`. A CPU is classed by `load_width`. Two tables are keyed by this class:
- `vector-policy.toml`: vector sizes, QuadField tiles, CPU blocks and CPU units, by element size 4 or 8;
- `degree-policy.toml`: a strategy per family, cell type and degree range, by element size.

M3s and M9h chose each class's entries from the platforms measured for that class. In practice that means one platform per CPU class: the Mac and Grace for CpuSimd128, and Kathleen for CpuSimd512, which CpuSimd256 copies. The Gpu class used Metal and the H100. The tables are therefore tuned to the test platforms.

## Problem

A deployed gridcl runs on CPUs and GPUs that none of the test platforms represent. A table tuned to one platform can be far from the best on another:
- the Gpu class already combines Metal and the H100, which disagree on vector sizes (docs/perf/vectorization.md: "Wide vectors hurt geometry on GPUs");
- the CPU degree policies of CpuSimd128 and CpuSimd512 differ in many ranges (`degree-policy.toml`).

A class keyed by SIMD width suggests a precision the data do not support. The SIMD width does not predict the best block size, units or strategy: these follow from the cache, core count and compiler. The maintainer wants settings that are good enough on every platform rather than optimal on each test platform.

## Options

1. **Keep the design as written.** Per-class tables, with CPU classes by SIMD width, each tuned to the platforms measured for it.
2. **Four generic sets, enum collapsed (chosen by the maintainer).** `DeviceClass { Cpu, Gpu }`. Both tables hold one set per class and element size: Cpu and Gpu, each for 4-byte (f32, and u32 where the entry is shared; ADR-0015) and 8-byte (f64) scalars. Each set is chosen by a rule that bounds the loss on every platform of its class.
3. **Four generic sets, enum kept.** As option 2, but `DeviceClass` keeps its variants for reporting, and the tables key only on CPU against GPU. Not chosen: an enum documented as "classes the policy distinguishes" that the policy ignores would mislead.

## Proposed decision

Option 2, implemented in M10.

1. **API.** In "Launch plans and the vector policy": `pub enum DeviceClass { Cpu, Gpu }`. `DeviceClass::for_client` returns `Cpu` for a device with `hardware.num_cpu_cores`, and `Gpu` otherwise. `load_width` no longer classes a device. Benchmark records still store the hardware: `load_width`, core counts and performance levels.
2. **Tables.**
   - `vector-policy.toml` has the tables `[Cpu]` and `[Gpu]`, each with entries for element sizes 4 and 8. `cpu_block` and `cpu_units` belong to `[Cpu]`. `performance` keeps its meaning: the performance cores where the CPU reports them (ADR-0017), and otherwise every hardware thread.
   - `degree-policy.toml` has `[Cpu.<family>]` and `[Gpu.<family>]`, with the same per-cell-type, per-element-size degree ranges as before.

   A test checks that each table holds exactly these four sets and nothing else.
3. **Platforms.** The platforms of each class are the following.

   | Class | Platforms | Element sizes |
   | --- | --- | --- |
   | Cpu | the local Mac (CPU runtime), Locust's Grace CPU, one socket of a Kathleen node (ADR-0020) | f32 and f64 |
   | Gpu | Metal on the local Mac | f32 only, since Metal has no f64 |
   | Gpu | Locust's H100 | f32 and f64 |

   The Gpu f64 set therefore rests on the H100 alone. The fallbacks of "Benchmarks" stand in for an unavailable platform: a Myriad D node for Kathleen; V100 and A100 for Locust. The A4000 workstation does not vote (ADR-0037).
4. **Generic decision rule.** For one policy entry, class and element size:
   - For each platform d, rule size s and candidate configuration c, let r(c, d, s) = t(c, d, s) / min over c' of t(c', d, s). Each candidate is compared with the fastest candidate on that platform.
   - A shared 4-byte entry counts f32 and u32 as separate members (replacing ADR-0019's per-type rule).
   - Admission (the 10% rule, unchanged in spirit): a configuration other than the baseline is admitted only if the geometric mean, over the class's platforms, of its time ratio to the baseline is at most 1/1.10 at each rule size.
   - Bound: among the baseline and the admitted configurations, keep those with r ≤ 1.25 for every platform and size.
   - Choice: among those kept, take the smallest geometric mean of r over platforms and sizes. Among the configurations within 3% of that mean, take the simplest:
     - the smallest vector size, then the smallest tile, then the smallest block;
     - ties in units go to all hardware threads;
     - strategies rank direct < loops < sum-factorised < shared < cooperative.
   - Fallback: if no configuration meets the bound, take the one with the smallest worst-case r, and the report marks it with that worst case. This replaces ADR-0018's and ADR-0021's backstops (1.10 and 1.25) for these sets.
   - A strategy that fails, or takes over 30 s to compile and load on any platform of the class, is not selectable (ADR-0056).
5. **Degree ranges.** The choices at p = 2, 3 and 5–7 come from M9h's per-degree records in `docs/perf/data/`, with the platforms that measured them (the Mac mini, Kathleen, Metal and the H100). Grace votes at p = 1, 4 and 8 from M10's runs (ADR-0057). M10's runs at p = 1, 4 and 8 check each family and cell type. A range whose check misses the bound, or whose time moves by more than 10% from M9h's record, is re-measured over its degrees, with candidates only. Degree ranges remain maximal runs of equal choices, and the last range stays open-ended (ADR-0042).
6. **Reports.** For every entry of the four sets, `docs/perf/vectorization.md` and `docs/perf/high-order.md` list:
   - each platform's best configuration;
   - the generic choice;
   - each platform's ratio r for the generic choice;
   - the reason.

   They keep the per-platform data, so a deployment can see what tuning to its platform would gain.
7. **Overrides.** `GRIDCL_VECTOR_SIZE`, `GRIDCL_QUAD_TILE`, `GRIDCL_GEOM_STRATEGY` and the `testing` hooks keep their meaning. No new public tuning API is added.

## Consequences

- Text to change in docs/design.md:
  - Fixed decisions, item 11: per device class (CPU or GPU) and scalar size, as generic sets from M10.
  - Requirements, **Development and test targets**; "Development environment and test targets", the Kathleen row and the CPU-only paragraph: "CpuSimd512" becomes "x86-64 AVX-512".
  - The `DeviceClass` listing and its comment.
  - "Geometric factors": the degree policy per class (Cpu or Gpu).
  - "Benchmarks", **Where** (the platforms of each class; CpuSimd256 note removed), **Decision rule** (a paragraph on the generic rule from M10), **Record** and **Degree sweep (M9h)** (M10's check of the ranges).
  - Roadmap M10: scope and acceptance.
  - The M3s and M9h roadmap sections are not changed: they record what those milestones measured and chose.
- Public API changes: `DeviceClass` loses `CpuSimd128`, `CpuSimd256` and `CpuSimd512` and gains `Cpu`. This is a breaking change before 1.0; within the workspace it reaches gridcl-device (`policy.rs`, `plan.rs`), gridcl-geometry (`strategy.rs`) and gridcl-bench (decision, record and report code and their tests).
- Acceptance tests affected:
  - M3s's coverage test for `vector-policy.toml` checks the two classes;
  - gridcl-bench's policy-rule tests (`perf_record.rs`, `high_order_record.rs`) check the generic rule for the M10 records. The M3s and M9h records stay readable, with their old class names, for the reports.
- Later milestones affected: M10 implements this. A CubeCL pin bump re-chooses the four sets with ADR-0057's candidate protocol.

## Decision record

Accepted by the maintainer on 2026-11-16, with options chosen in the same session:
- `DeviceClass` collapses to `{Cpu, Gpu}`;
- the choice is the geometric mean with every platform within 1.25×;
- the platforms are the Mac mini (CPU and Metal), Locust (Grace and H100) and Kathleen (one socket);
- the code changes in M10, with only the documents now;
- the degree ranges come from M9h's data and are checked at p = 1, 4 and 8.

Applied in `adr: 0058 apply`: docs/design.md (the sections above), CLAUDE.md, the hpc-test skill and issue #25.

Amended by ADR-0060 (item 3, accepted 2026-11-16): item 5's comparison with M9h is at 10^6 points, and a range that misses the bound or moves by more than 10% is re-measured, with candidates only, at the inner degrees next to each of its boundaries.

Amended by ADR-0065 (accepted 2026-11-18): item 4's fallback applies the bound and the choice to all candidates when no admitted configuration meets the bound, and takes the admitted configuration with the smallest worst-case r only if none does; admission and r use the platforms present where a platform leaves out a size.
