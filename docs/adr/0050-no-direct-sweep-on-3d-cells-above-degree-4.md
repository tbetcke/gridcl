# ADR-0050: The degree sweep skips `direct` on hexahedra, prisms and tetrahedra at p ≥ 5

- **Status:** Proposed
- **Date:** 2026-11-14
- **Raised in:** M9h T11a (strategy sweeps)
- **Design sections affected:** "Benchmarks" (the **Degree sweep (M9h)** bullet); Roadmap M9h, acceptance bullets 2 and 6 as read by ADR-0042 items 7 and 12. Amends ADR-0042, items 7 and 12.

## Context

"Benchmarks", **Degree sweep (M9h)**: the degree-dependent families "are measured at p = 1–8 for each strategy of 'Geometric factors'". Roadmap M9h, bullet 2, read by ADR-0042 item 7: "Every (family, cell type, degree, precision, strategy, vector size used by the policy or the invariance tests) compiles, loads and launches on each backend", checked from `degree_sweep --launch-only` records (`high_order_record.rs::every_variant_launched`). Bullet 6: `docs/perf/high-order.md` lists "the measured cost of each strategy".

T11a runs these sweeps on the six M3s targets: the Mac mini's CPU and Metal, Kathleen's CPU (one socket), Locust's Grace CPU and GH200. T3's drift baseline already measured `direct` at p = 1–8 on every target (on Locust at p ∈ {1, 2, 4, 8} for the pull-back families and the Grace CPU, ADR-0043).

## Problem

At p ≥ 5, `direct` on the three-dimensional cells costs a large share of the sweep and almost never wins. Measured on the sweeps finished so far (10^6 and 10^7 points, every family and scalar type):

| Data | Cell, p ≥ 5 | `direct` fastest or within 3% | `direct`'s sweep time |
| --- | --- | --- | --- |
| Mac mini CPU | Hexahedron (p = 5–7) | 8 of 93 | 0.53 h of 0.58 h for `direct` at p ≥ 5 |
| Mac mini Metal | Hexahedron | 2 of 72 | 0.42 h |
| Mac mini Metal | Prism | 6 of 72 | 0.17 h |
| Kathleen | Hexahedron p = 7 | 1 of 36 | 36% of the job |
| Kathleen | Hexahedron p = 8, f32 / f64 | 3 of 18 / 3 of 18 | 18% of 4.2 h / 18% of 2.7 h |
| Mac mini CPU | Interval | 143 of 160 | 0.01 h |
| Mac mini Metal | Triangle / Interval | 79 of 80 / 74 of 80 | under 0.01 h |

On CUDA, `direct`'s pull-back on tetrahedra, hexahedra and prisms at p ≥ 5 does not compile within T2's 900 s (C18). Its launch records for `every_variant_launched` needed a separate Locust job of up to 13 h at a 1800 s limit per configuration. The remaining sweeps would have run until about Monday 16 November. The maintainer judged that unacceptable (2026-11-14).

## Options

1. **Keep the design as written.** Time and launch `direct` everywhere, including the 13 h CUDA job, whose pull-back records at p ≥ 5 would be "stopped" and fail `every_variant_launched` anyway. Not chosen: it buys almost nothing and costs roughly 15–25 h of target time.
2. **No `direct` at p ≥ 5 on any cell.** Simpler to state, but on intervals and triangles `direct` is nearly always the fastest at p ≥ 5 and costs almost nothing to run. The policy would lose it there.
3. **Drop `direct` entirely.** The policy could never choose `direct`, which is the fastest at p ≤ 2 in most cases. Rule 11 (the policy at most 5% slower than `direct` at p ≤ 4) would very likely fail.
4. **No `direct` at p ≥ 5 on hexahedra, prisms and tetrahedra (chosen by the maintainer).** Every other (cell, degree) keeps `direct`, in both the timed and the launch-only sweeps. Where `direct` is not measured it is not selectable, so the decision rule's baseline there is `loops`, as ADR-0042 item 12 already provides. `direct`'s cost at p ≥ 5 on those cells is reported from T3's baseline.

## Proposed decision

Option 4. In the T11a sweeps, `direct` is neither timed nor launched on hexahedra, prisms and tetrahedra at p = 5–8, and the CUDA `direct` pull-back compile job is cancelled. Records of `direct` there that were already taken stay in the data: Mac mini CPU at p = 5–7 (hexahedra), Metal, Kathleen's hexahedra at p = 7–8, and Locust's CUDA job `a` for the families other than the pull-back. The decision rule uses them where they exist. `direct` stays bitwise M9's kernel and remains selectable everywhere it is measured. The last policy range is never `direct` (ADR-0042 item 12), so the choice at p ≥ 9 does not change.

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Benchmarks", **Degree sweep (M9h)**: after "for each strategy of 'Geometric factors'" add "(`direct` not on hexahedra, prisms and tetrahedra at p ≥ 5, where T3's drift baseline gives its cost; ADR-0050)".
  - Roadmap M9h, the line "Acceptance (read as in ADR-0042 …)": add "and ADR-0050".
- Public API changes: none.
- Acceptance tests affected:
  - `high_order_record.rs::every_variant_launched` skips (`direct`, Tetrahedron | Hexahedron | Prism, p ≥ 5).
  - `degree_policy_follows_the_decision_rule` is unchanged: it already treats a strategy without records as not selectable and falls back to `loops`.
  - `report_covers_every_family_class_and_range` is unchanged. `docs/perf/high-order.md` marks those cells "not measured (ADR-0050)" and cites T3's baseline cost.
- Plan: M9h T11a's sweep scope and the coverage-table row of bullet 2 name this ADR. The GitHub issue's acceptance text for bullet 2 gets the same note.
- Later milestones affected: M10 re-benchmarks the final kernels and may measure `direct` on these cells again if it is still a candidate there. C18's closure (M9h scope bullet 4) does not depend on the cancelled job: T2 recorded its cause.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
