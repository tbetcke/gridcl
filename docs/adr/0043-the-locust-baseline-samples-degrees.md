# ADR-0043: The Locust baseline samples degrees

- **Status:** Proposed
- **Date:** 2026-11-10
- **Raised in:** M9h T3
- **Design sections affected:** none in docs/design.md. Amends ADR-0042, item 11.

## Context

ADR-0042, item 11 asks for "a baseline of `direct` at a pinned commit before any dispatch change, measured on the current machines". It shows that dispatch adds no measurable cost, and it is "reported, not bounded". The plan's T3 runs the baseline at p = 1–8 on every M3s target. The acceptance test `baseline_covers_every_target_and_degree` (gridcl-bench/tests/high_order_record.rs) requires a record for every family, cell type, degree 1–8 (pyramids to 3), scalar type and rule size (10^6 and 10^7 points) on every target, all at one commit. No other acceptance test reads the baseline.

The baseline is pinned at a88ab2a, the T3 commit. Status on 2026-11-10:

- Complete: the Mac CPU and Metal sweeps, and the Kathleen CPU sweep in two jobs (16,538 s and 13,700 s).
- Locust (an open node, ADR-0002) runs CUDA part a (EVector and the factor families), then part b (PullBack and BasisEval), then the Grace CPU sweep, each only after an idle check.
- A first part a took 4.47 h but was invalid: another user held the GPU at 100%. The rerun started at 05:59 on an idle node.

## Problem

On Locust the sweep's time goes to the host, not the device:

- Of part a's 4.47 h, only 0.56 h was kernel compile (0.35 h) and timed runs (0.21 h).
- The rest is mostly building the host mesh at 10^7 points. A profile shows the time in the single-threaded numbering of the mesh pipeline (`gridcl::pipeline::number`).
- Locust's Grace cores do that work 2.5 times slower than the M6 Mac: 7,811 s against 3,097 s on the configurations both ran.

Estimated remaining times are about 4–4.5 h for part a, about 3.5–4 h for part b and 5–8 h for the Grace CPU sweep, plus an idle wait whenever another user is on the node.

Faster alternatives within the design:

- **Fewer processes.** One process per (cell type, degree) instead of per (family, cell type, degree, type) saves about 25% on Metal (253 s against 189 s for seven families at interval p = 8). It changes the sweep script, so the records would no longer be at one commit.
- **Fewer sizes or types.** Both are fixed by item 12 and by the acceptance test.

Per degree, part a's time was:

| p | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s | 936 | 1,251 | 1,461 | 1,564 | 1,784 | 2,195 | 2,788 | 4,110 |

Degrees 1, 2, 4 and 8 together take 49% of the total.

## Options

1. **Keep the design as written.** Locust's remaining runs take about 9–12 h, finishing around 21:00–23:00 on 2026-11-10 if Locust stays idle and later if not. Until then, Locust is not free for the kernel-task checks of later tasks, and the A4000 stands in. Not chosen, because of the time.
2. **The Locust baseline samples degrees.**
   - Part b and the Grace CPU sweep run at p ∈ {1, 2, 4, 8}. Pyramids, which stop at p = 3, run at {1, 2}.
   - Part a, already running, stays complete, as do the Mac and Kathleen baselines.
   - Still at a88ab2a: the sweep script's `--degrees` option needs no code change.
   - The sampled degrees cover the lowest, the quadratic, the middle and the highest degree. That is enough to show a dispatch cost at low degree, where item 11's paired comparison (p ≤ 4) applies, and a cost that grows with degree.
   - Locust then finishes about 4–6 h earlier.
3. **One process per (cell type, degree) at a new commit, all degrees.** About 25% less time. The test's one-commit check would accept commits that differ only under `scripts/`. A smaller saving than option 2, and the check becomes weaker.

## Proposed decision

Option 2. The baseline is reported, not bounded, and a degree sample on the slowest target shows the same thing as the full sweep: whether dispatch costs time and how that cost changes with degree. The complete Mac and Kathleen data and Locust part a stay as the full reference.

ADR-0042, item 11 gains: "On Locust, the baseline of the pull-back families (PullBack, BasisEval) and of the Grace CPU samples p ∈ {1, 2, 4, 8} (pyramids {1, 2}); every other target and family covers p = 1–8."

The degree sweep that chooses the policy (item 12, T11a) is unchanged: it covers p = 1–8 on every target.

## Consequences

- **Text to change in docs/design.md:** none. In ADR-0042's decision record, a line "Amended by ADR-0043 (the Locust baseline of the pull-back families and of the Grace CPU samples degrees)".
- **Public API changes:** none.
- **Acceptance tests affected:** `baseline_covers_every_target_and_degree` requires, for target `locust`, on backend `cpu` and for the families PullBack and BasisEval on backend `cuda`, only degrees 1, 2, 4 and 8 (pyramids 1 and 2). Its other checks stay: one commit, `direct` only, the protocol, the device class.
- **Plan:** T3's goal names the sample. Its Deviations-log row records the Locust job ids.
- **docs/perf/high-order.md** says which baseline entries are sampled.
- **Later milestones affected:** none. M10 does not read the M9h baseline.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
