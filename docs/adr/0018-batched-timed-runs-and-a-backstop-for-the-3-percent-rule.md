# ADR-0018: Batched timed runs, and a backstop for the 3% rule

- **Status:** Accepted
- **Date:** 2026-10-25
- **Raised in:** M3s T17 (trial `policy_report` on the M3 Max and Locust data)
- **Design sections affected:** "Vectorization and CPU execution", "Benchmarks" ("What it reports", "Decision rule"); roadmap acceptance of M3s (bullet 3).

## Context

"Vectorization and CPU execution", "Benchmarks":

> **What it reports:** the median of ten timed runs after a warm-up launch that absorbs JIT compilation, […]

> Among the configurations within 3% of the fastest, the smallest vector size is chosen […]

M3s acceptance, bullet 3: every chosen configuration "is within 3% of the fastest one the 10% rule admits (ADR-0015)". As T0 reads it (docs/plans/M3s.md, Deviations), the 3% condition must hold at both 10^6 and 10^7 items, like the 10% condition.

gridcl-bench implements this as one call per timed run: `protocol::measure` times `run()` plus `client.sync()` once, ten times, and records the median (T8). T12's `decide`, when no admitted configuration is within 3% at both sizes, takes the one with the smallest worst ratio and says so in the reason, and `perf_record::policy_follows_the_decision_rule` then fails, "and the case goes to the maintainer" (T12 deviation (b)).

## Problem

With the M3 Max (T15, T21) and Locust (T16) data, the trial `policy_report` leaves 9 entries with no admitted configuration within 3% of the fastest at both sizes:

| Entry | Closest configuration |
| --- | --- |
| Fill CpuSimd128 f32, u32 (shared 4-byte entry) | 6.2% slower |
| Fill CpuSimd128 f64 | 5.3% slower |
| Gather CpuSimd128 f32, u32 (shared) | 74.4% slower |
| Gather CpuSimd128 f64 | 11.9% slower |
| Reduce Gpu f32, u32 | 5.8% slower |
| Scalar CpuSimd128 f64 (`cpu_block`) | 4.4% slower |

The cause is timing noise, not the kernels. For each record, take the ten sorted timings and divide the spread between the 3rd and 8th (about the interquartile range, IQR) by the median. The median of that ratio over all `plan` configurations of a family and device is:

| Family, device | 10^6 | 10^7 | median time at 10^6 / 10^7 |
| --- | --- | --- | --- |
| Gather, Grace CPU | 0.33 | 0.30 | 0.18 ms / 1.07 ms |
| TopologyKeys, Grace CPU | 0.48 | 0.45 | 1.5 ms / 21 ms |
| Reduce, M3 Max CPU | 1.03 | 0.52 | 1.7 ms / 5.8 ms |
| ScanTile, M3 Max CPU | 0.83 | 0.33 | 4.1 ms / 19 ms |
| Fill, M3 Max CPU | 0.24 | 0.08 | 0.08 ms / 0.51 ms |
| PermutationInfo, M3 Max CPU | 0.84 | 0.10 | 0.87 ms / 6.5 ms |
| every family, H100 | ≤ 0.08 | ≤ 0.08 | |
| every family except Gather, Metal | ≤ 0.11 | ≤ 0.10 | |

Locust's node-load snapshots were `idle` before and after every part, and the M3 Max ran under T15's conditions, so the noise comes from the CPU runtime itself: thread wake-up and scheduling of a launch that lasts a fraction of a millisecond. A median of ten single calls cannot separate configurations 3% apart when individual runs differ by 30–100%. For example, Grace Gather f32 at 10^7 takes 0.50–1.12 ms depending on the configuration, and which configuration is fastest changes at random from one block size to the next. Repeating the sweep under the same protocol would only move the noise around.

## Options

1. **Keep the design as written.** The 9 entries fail `policy_follows_the_decision_rule`, so acceptance bullet 3 does not hold and M3s cannot reach `implemented`. Not chosen.
2. **Batched timed runs, plus a backstop (chosen by the maintainer, 2026-10-25).**
   - Each of the ten timed runs repeats the call k times and records the time per call.
   - k is chosen once per configuration, so that a run lasts at least 20 ms.
   - Entries that still have no configuration within 3% at both sizes take the closest one, bounded and marked in the report.
   - Every device is swept again.
3. **A noise-aware reading of 3%:** a configuration counts as within 3% if it is within 3% of the fastest or within the fastest's IQR. No new runs, but the rule gets looser exactly where the data is noisiest, and it changes how bullet 3 reads.
4. **Logged exceptions:** keep the protocol, and have `perf_record` accept a fixed list of entries that fail. This is the smallest change, but bullet 3 then holds only with exceptions, and the list has to be redone after every re-run (M10, CubeCL bumps).

## Proposed decision

Option 2.

**Protocol (`protocol::measure`):**
- After the checked warm-up, one untimed calibration call fixes k = ⌈20 ms / t_cal⌉, capped at 10,000 (k = 1 when one call already takes 20 ms).
- Each of the ten timed runs then makes k calls. Each call ends with `client.sync()`, and `prepare` plus its sync run untimed before each call, as now. A run's time is the sum of the k timed calls divided by k.
- `times_s` keeps ten per-call times. Records gain `calls_per_run` (k). The compile time stays the warm-up minus the median.
- Calls that already take 20 ms (most CPU runs at 10^7, all `mesh_throughput` runs) are measured exactly as before.

**Decision rule (backstop):**
- If no admitted configuration is within 3% of the fastest at both sizes (for every type of a shared entry), the rule takes the admitted configuration with the smallest worst-case ratio, provided that ratio is at most 1.10, the 10% admission threshold.
- The report marks the entry "closest, x% slower".
- Above 1.10 the entry is unresolved, `policy_follows_the_decision_rule` fails, and the case goes to the maintainer, as now.

**Data:**
- Sweep every device again under the new protocol: the M3 Max CPU (about 45 min, under T15's conditions, confirmed again by the maintainer), Metal (about 1 min), Locust CPU (about 11 min) and Locust CUDA (about 3 min).
- The maintainer's choice kept the GPU data. I propose sweeping the GPUs again as well, because it takes minutes and every record then follows one protocol; Reduce Gpu is one of the 9 entries.
- The Kathleen (or Myriad D node) CpuSimd512 sweep, which is still outstanding, uses the new protocol from the start.

## Consequences

- Text to change in docs/design.md:
  - "Benchmarks", "What it reports": "the median of ten timed runs, each repeating the call until it has lasted at least 20 ms and giving the time per call (ADR-0018), after a warm-up launch that absorbs JIT compilation, …".
  - "Benchmarks", "Decision rule": a final sentence, "If no admitted configuration is within 3% of the fastest at both sizes, the one with the smallest worst-case ratio is chosen if that ratio is at most 1.10, and the report marks it (ADR-0018)."
  - Roadmap, M3s acceptance, bullet 3: "… is within 3% of the fastest one the 10% rule admits (ADR-0015), or, where none is, the closest one, at most 10% slower and marked in the report (ADR-0018)".
- Public API changes: none. gridcl-bench is unpublished: `protocol::measure`, `SweepRecord` and `ThroughputRecord` gain `calls_per_run`, and `decide` and `report` gain the bound and the marking.
- Acceptance tests affected:
  - `perf_record::policy_follows_the_decision_rule`: its independent check accepts a marked backstop entry whose worst ratio is at most 1.10, and only when no admitted configuration meets 3%.
  - `benchmark_data_covers_required_targets`: also checks that every record has `calls_per_run` ≥ 1, and that `calls_per_run` × median ≥ 20 ms whenever `calls_per_run` > 1.
  - Unit tests of `measure` (the calibration and the cap) and of `decide` (the bound).
- Plan: a new M3s task before T17 implements the protocol and the backstop and repeats the T15 and T16 sweeps. The committed T15, T21 and T16 data are replaced. T17 and T18 are otherwise unchanged.
- Later milestones affected: M10's re-run and any re-run after a CubeCL bump use the batched protocol. Pack (M6) and PullBack (M9) are measured with it.

## Decision record

Accepted by the maintainer, 2026-10-25, as proposed (20 ms a run, the 1.10 bound). The GPUs (Metal, H100) are swept again as well.
