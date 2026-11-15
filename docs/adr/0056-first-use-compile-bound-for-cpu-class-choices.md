# ADR-0056: A first-use compile bound for the CPU classes' choices in the degree policy

- **Status:** Accepted
- **Date:** 2026-11-15
- **Raised in:** M9h review (design-reviewer, Major 2)
- **Design sections affected:** "Benchmarks" (**Degree sweep (M9h)**); "Geometric factors" (the degree policy). Amends ADR-0042, items 7 and 12.

## Context

"Benchmarks", **Degree sweep (M9h)**: "A strategy is adopted over the baseline (`direct`, or `loops` where `direct` is not selectable) only if it is at least 10% faster at 10^6 and 10^7 points … A strategy that fails or takes over 30 s to compile and load on Locust is not selectable for the Gpu class." ADR-0042 item 7 limits the 30 s bound to the Gpu class, and `gridcl-bench/src/decide_degree.rs` (`selectable`) follows it: on the CPU classes a strategy is selectable whenever it has a time.

M9h's scope (Roadmap M9h, ADR-0041) names "runtime loops instead of full unrolling (bounding code size and JIT time)" as the first strategy, and C20 records that CubeCL 0.11.0's common-subexpression pass is quadratic in a region's operations, so fully unrolled kernels compile slowly on every backend, the CPU runtime included.

## Problem

Since `direct` is the baseline and another strategy replaces it only when it is at least 10% faster, the CPU policy keeps `direct` wherever nothing beats it by that margin, whatever its compile time. Crossing `gridcl-geometry/degree-policy.toml` with the first-use compile and load times of the committed records (`docs/perf/data/m9h-{launch,degree,baseline}-cpu-{macmini,kathleen}*.json`, the largest over the vector sizes that ran) gives, for the choices at p = 1–8:

| Class (device) | Family, cell | Type | p | Choice | First use | `loops` first use | Time per call at 10^6 points, `direct` / `loops` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CpuSimd128 (Mac mini) | E-vector, hexahedron | f64 | 7 | `direct` | **64.2 s** | 0.01 s | 1.07 ms / 1.05 ms |
| CpuSimd128 (Mac mini) | E-vector, hexahedron | f32 | 5 | `direct` | **37.1 s** | 0.02 s | 0.74 ms / 0.70 ms |
| CpuSimd128 (Mac mini) | E-vector, hexahedron | f64 | 6 | `direct` | 26.4 s | 0.01 s | 0.98 ms / 0.95 ms |
| CpuSimd128 (Mac mini) | E-vector, hexahedron | f32 | 4 | `direct` | 11.5 s | 0.02 s | (rule 11's range) |

No other choice of the CpuSimd128 class takes over 10 s to first use; on CpuSimd512 (one Kathleen socket) the largest is 24 s (`direct` pull-back, hexahedron, p = 8, f64; the `direct` hexahedron E-vector at p = 8 takes 15 s). The E-vector is a gather that a mesh builds once and caches, so the two entries above cost a first build 37 s and 64 s to save nothing measurable. `docs/perf/high-order.md` ("CPU compile time at high degree") says "The degree policy and T11a's strategies with runtime loops avoid the fully unrolled kernels", which is not true of these entries.

The implementation conforms to the design and ADR-0042; the acceptance bullets hold. The question is whether the decision rule should let this happen.

## Options

1. **Keep the design as written.** The policy stays; the report's sentence is corrected to name the two entries. A user of a hexahedral mesh at p = 5 (f32) or p = 7 (f64) on an Apple-class CPU waits 37 s or 64 s once per process for the E-vector kernel. M10's re-study of the final kernels may revisit it.
2. **The 30 s bound applies to every class, on the class's sweep devices (proposed).** A strategy whose first-use compile and load time exceeds 30 s on any device swept for the class (launch-only and timed records, the largest over the vector sizes that ran) is not selectable for that class: the Mac mini for CpuSimd128, one Kathleen socket for CpuSimd512 (CpuSimd256 copies it), Locust and Metal for Gpu (Locust alone today; no Gpu choice takes over 20 s to first use on Metal, so adding Metal changes nothing). Applied to the committed data, exactly two entries change, both CpuSimd128 E-vector on hexahedra: f32 `[1, direct], [5, loops]` (was `[6, loops]`) and f64 `[1, direct], [7, loops]` (was `[8, loops]`). No new measurement is needed: `loops` has records there. Rule 11 (p ≤ 4) is untouched. The E-vector under `loops` is bitwise `direct`'s on the CPU runtime (T6a).
3. **A relative rule: compile time enters the 10% comparison.** For example, a strategy must repay its extra first-use time within some number of calls. It needs a call count nobody can name for a library, and a second threshold.
4. **Leave it to M10.** As option 1, with an M10 follow-up to add a compile bound when the final kernels are re-benchmarked. The two entries ship as they are in between.

## Proposed decision

Option 2. It is the Gpu rule the design already has, applied where the same cause (C20) has the same effect, it changes two policy entries and no kernel, and it makes the report's sentence true. The bound stays 30 s so that there is one number; on the Gpu class it stays Locust's for bullet 2, and Metal is added only as a selectability check.

## Consequences

- Text to change in docs/design.md:
  - "Benchmarks", **Degree sweep (M9h)**: "A strategy that fails or takes over 30 s to compile and load on Locust is not selectable for the Gpu class" becomes "A strategy that fails, or takes over 30 s to compile and load on a device swept for a class, is not selectable for that class (ADR-0056); bullet 2's bound is Locust's".
  - "Geometric factors" (degree policy): no change.
- ADR-0042: item 7's "a strategy over 30 s on Locust cannot be selected for the Gpu class" and item 12's selectability gain "on the class's sweep devices, every class (ADR-0056)".
- Public API changes: none.
- Code: `gridcl-bench/src/decide_degree.rs` (`selectable` asks the class's launch records, not only Locust's); `high_order_report` regenerates `gridcl-geometry/degree-policy.toml` (two entries) and the tables and the "CPU compile time" sentence of `docs/perf/high-order.md`.
- Acceptance tests affected: `gridcl-bench/tests/high_order_record.rs::degree_policy_follows_the_decision_rule` recomputes the rule with the bound on every class; `compile_and_load_within_30_s_on_locust` is unchanged. A new assertion (same file) that no selected variant of any class exceeds 30 s on that class's sweep devices. The M9h tests under the policy at hexahedron p = 5 and p = 7 run the `loops` E-vector on CpuSimd128 devices: re-run on the Mac mini's CPU runtime in release (results bitwise as before).
- Later milestones affected: M10's re-study inherits the rule for every class.

## Decision record

Accepted by the maintainer on 2026-11-15, as proposed (option 2), at the M9h review. Applied in `adr: 0056 apply`: docs/design.md, "Benchmarks" (**Degree sweep (M9h)**); ADR-0042's decision record; the M9h plan. In code (`M9h review: first-use bound on every class (ADR-0056)`): `decide_degree.rs` (`Times::first_use_ok`), `high_order_record.rs` (`first_use_ok`, the new `policy_choices_first_use_within_30_s_on_every_class`), the regenerated `degree-policy.toml` and `docs/perf/high-order.md`. The regenerated policy changes exactly the two entries named under option 2. In the report, the baseline of four CpuSimd512 rows also becomes `loops` (Jacobian and FacetNormals on hexahedra at p = 8, f32 and f64: `direct`'s first use on Kathleen is over 30 s there), with the choice `sumfact` unchanged.

Corrected at the second review round (2026-11-15): the Problem section gave 15 s as the largest first use of a CpuSimd512 choice; it is 24.2 s (`direct` pull-back on hexahedra at p = 8 in f64, `m9h-degree-cpu-kathleen-hex8-f64.json`). The decision does not depend on it. In the report's cost table, `direct` excluded by the first-use rule is marked "f.u.; T3" (timed, not selectable) instead of "n.m.; T3" (not measured, ADR-0050).
