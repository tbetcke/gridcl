# ADR-0051: The degree sweep leaves out Locust's Grace CPU; it keeps the paired check

- **Status:** Proposed
- **Date:** 2026-11-14
- **Raised in:** M9h T11a (strategy sweeps)
- **Design sections affected:** Roadmap M9h, acceptance bullets 2, 3 and 6 as read by ADR-0042 items 1, 7 and 12. Amends ADR-0042, items 1, 7 and 12.

## Context

Roadmap M9h, bullet 6: `docs/perf/high-order.md` lists the costs, choices and reasons "with benchmark JSON under `docs/perf/data/` from the targets of M3s". Those targets are the Mac mini's CPU and Metal, Kathleen's CPU, and Locust's Grace CPU and GH200. ADR-0042 item 7 checks bullet 2 from `--launch-only` records "on each backend". Item 12's decision rule aggregates a class with several devices by their geometric mean. Item 1 applies bullet 3's p = 8 / p = 4 bound "on every device class". The CpuSimd128 class has two devices, the Mac mini's M6 and Grace. `high_order_record.rs::required` lists both for the degree, launch and baseline records.

T3's drift baseline on Grace already samples p ∈ {1, 2, 4, 8}, because the host mesh build at 10^7 points dominates there (ADR-0043). The Mac mini covers the ARM CPU runtime for every correctness test.

## Problem

After ADR-0050, Locust's remaining T11a work was CUDA part B (about 1–2 h) and four Grace CPU jobs: launch-only, then timed sweeps of the small cells, the hexahedra and the prisms. Those four jobs had time limits of 37 h and an estimated 15–18 h. They must run one after another and after the GPU jobs, because Locust is an open node and a CPU timing counts only if the node is idle before and after. The Grace sweeps alone would have moved the end of T11a from the afternoon of 14 November to about 15 November midday. The maintainer judged that unacceptable (2026-11-14), since the M6 already covers ARM.

## Options

1. **Keep the design as written.** Sweep Grace in full. It gives CpuSimd128 a second device, a 72-core server with LPDDR5X, but delays T11a by about a day.
2. **Sample Grace at p ∈ {1, 2, 4, 8}, as ADR-0043 did for its baseline.** About half the time, and Grace still enters the class mean at the sampled degrees. The class rule then mixes one device at some degrees with two at others.
3. **Leave Grace out of the sweeps, but keep rule 11's paired check there (chosen by the maintainer).** CpuSimd128's degree policy, its p = 8 / p = 4 ratios and its launch coverage come from the Mac mini alone. T11b's paired policy-vs-`direct` run at p ≤ 4 still runs on Grace (about 1–2 h), so a policy that slows an ARM server at low degree still fails bullet 4. T3's Grace baseline stays.
4. **Leave Grace out entirely.** The fastest option, but nothing then checks the policy on an ARM server.

## Proposed decision

Option 3. T11a does not sweep Grace, timed or launch-only. CpuSimd128's measurements and policy come from the Mac mini's M6. Grace keeps T3's drift baseline (ADR-0043) and T11b's paired check of rule 11 at p ≤ 4. `docs/perf/high-order.md` says that CpuSimd128's choices were measured on the M6 only. M10 re-benchmarks the final kernels on the current targets.

## Consequences

- Text to change in docs/design.md:
  - Roadmap M9h, bullet 6: "from the targets of M3s" becomes "from the targets of M3s (Locust's Grace CPU: the drift baseline and the paired check only, ADR-0051)".
  - The same Roadmap section, the line "Acceptance (read as in ADR-0042 …, and ADR-0050)": add ADR-0051.
- Public API changes: none.
- Acceptance tests affected:
  - `high_order_record.rs`: the swept devices leave out `("locust", "cpu")` in `every_variant_launched`, `tensor_cost_ratio_p8_over_p4` and `degree_policy_follows_the_decision_rule` (through `required` and `devices`).
  - `baseline_covers_every_target_and_degree` and `no_family_slower_than_m9_at_low_degree` keep Grace. `report_covers_every_family_class_and_range` is unchanged.
- Plan: M9h T11a's targets and T11b's paired runs name this ADR, and so do the coverage-table rows of bullets 2, 3 and 6. Issue #37 gets the same note.
- Later milestones affected: M10's benchmark of the final kernels includes Grace again, as a CpuSimd128 device.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
