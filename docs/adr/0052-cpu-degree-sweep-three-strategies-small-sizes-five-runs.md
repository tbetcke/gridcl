# ADR-0052: On CPUs the degree sweep times direct, loops and sumfact at 10^5 and 10^6 points, five runs

- **Status:** Proposed
- **Date:** 2026-11-14
- **Raised in:** M9h T11a (strategy sweeps)
- **Design sections affected:** "Benchmarks" (the **Degree sweep (M9h)** bullet); Roadmap M9h, acceptance bullets 3, 4 and 6 as read by ADR-0042 items 1, 11 and 12. Amends ADR-0042, items 1, 11 and 12.

## Context

"Benchmarks", **Degree sweep (M9h)**: every strategy is measured at p = 1–8 "with the protocol and conditions above", which means medians of ten batched runs. The sizes are 10^5, 10^6 and 10^7 quadrature points. "A strategy is adopted over the baseline … only if it is at least 10% faster at 10^6 and 10^7 points." ADR-0042 item 1 applies bullet 3's p = 8 / p = 4 bound at both sizes on every device class, and item 11's paired check does the same. On the CPU runtime, `shared` and `cooperative` have no real shared memory: a cube's shared array is a per-thread local array, and a group's units are threads of one core (ADR-0045).

## Problem

After ADR-0050 and ADR-0051, the CPU sweeps still left were these:
- on the Mac mini: hexahedra p = 8, prisms and simplices, about 10–14 h;
- on Kathleen: the small cells with launch-only, prisms p = 1–7 and tetrahedra, about 6–9 h plus queue waits.

The maintainer asked for these to be cut substantially (2026-11-14). Measured on the CPU records so far (Mac mini: 2.9 h of recorded time; Kathleen: 8.5 h):

- **Where the time goes.** The pull-back takes 79% (Mac mini) and 91% (Kathleen) of the time, and its 9-iteration runs take 64% and 76%. Runs at 10^7 points take 64% and 76%. Each strategy takes 14–24%.
- **What dropping a strategy would cost.** At 10^6 and 10^7 points, this table counts the cases where the best remaining strategy is more than 5% slower:

| Dropped | Mac mini | Kathleen |
| --- | --- | --- |
| `shared` | 0 of 907 (worst 4%) | 0 of 72 |
| `cooperative` | 5 of 904 (worst 16%, hexahedron p = 8) | 0 of 71 |
| `loops` | 7 of 907 | 1 of 72 |
| `sumfact` | 303 of 907 | 40 of 72 |
| `direct` | 244 of 879 | 2 of 72 |

- **10^6 against 10^7.** The fastest strategy is the same at both sizes in 313 of 442 cases on the Mac mini and in 28 of 35 on Kathleen.

## Options

1. **Keep the design as written.** All five strategies, three sizes, ten runs. Not chosen: it would take until late tonight on the Mac mini and this evening on Kathleen.
2. **Drop only `shared` and `cooperative` on CPUs, and take five runs.** This cuts the time by about −38% and −40%, keeps 10^7 points, and keeps the rule as it is. The time would roughly halve.
3. **Option 2, and also drop 10^7 points on CPUs (chosen by the maintainer).** Runs at 10^7 points were 64–76% of the CPU time. The CPU decision then rests on 10^6 points alone. In about a quarter of the cases measured, the choice at 10^6 is not the one at 10^7, so the CPU policy is tuned for meshes of up to about 10^6 points per rank.

## Proposed decision

Option 3, for the CPU classes CpuSimd128 and CpuSimd512. The Gpu class is unchanged.

- **Strategies.** The timed sweep runs `direct`, `loops` and `sumfact` (simplices: `direct` and `loops`; ADR-0050's skips still apply), and the policy chooses among these. `shared` and `cooperative` stay correct on CPUs: the accuracy, determinism and invariance tests force every strategy, and the launch-only records keep them. The CPU timings of `shared` and `cooperative` taken before this ADR are reported but not selectable.
- **Sizes and runs.** Sizes are 10^5 and 10^6 points, with medians of five timed runs. The decision rule, item 1's p = 8 / p = 4 ratios and item 11's paired check use 10^6 points on the CPU classes. CPU records at 10^7 points and those with ten runs, taken before this ADR, stay in the data and the report, but the rules on CPUs read only 10^6 points. The configurations started after this ADR use five runs.
- **Report.** `docs/perf/high-order.md` states the CPU sizes and runs, and that the CPU policy is chosen at 10^6 points. M10 re-benchmarks the final kernels.

## Consequences

- Text to change in docs/design.md:
  - "Benchmarks", **Degree sweep (M9h)**: after "… only if it is at least 10% faster at 10^6 and 10^7 points;" add "on the CPU classes the sweep times `direct`, `loops` and the sum-factorised strategy only, at 10^5 and 10^6 points with medians of five runs, and the rule uses 10^6 points (ADR-0052);".
  - Roadmap M9h, the line "Acceptance (read as in ADR-0042 …)": add ADR-0052.
- Public API changes: none.
- Acceptance tests affected, in `high_order_record.rs`:
  - The rule sizes depend on the class: CPU classes use [10^6], the Gpu class [10^6, 10^7]. This affects `tensor_cost_ratio_p8_over_p4`, `no_family_slower_than_m9_at_low_degree` and `degree_policy_follows_the_decision_rule`.
  - `degree_policy_follows_the_decision_rule` on CPU classes considers only `direct`, `loops` and `sumfact`.
  - `check_protocol` accepts five or ten timed runs for CPU degree records.
  - `baseline_covers_every_target_and_degree` is unchanged: T3's baseline is complete.
- Plan: T11a's and T11b's CPU runs, and the coverage-table rows of bullets 3, 4 and 6, name this ADR. Issue #37 gets the same note.
- Later milestones affected: M10's benchmark of the final kernels includes 10^7 points on CPUs again.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
