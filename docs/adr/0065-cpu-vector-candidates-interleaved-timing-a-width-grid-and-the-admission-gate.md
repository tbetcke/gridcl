# ADR-0065: Cpu vector candidates: interleaved timing, a width grid and the admission gate

- **Status:** Proposed
- **Date:** 2026-11-18
- **Raised in:** after M10 (maintainer request to improve the CPU policy, 2026-11-18; M10 Review, follow-ups; M10 Deviations log, T8)
- **Design sections affected:**
  - "Benchmarks": **What it runs**, **What it reports**, **Generic decision rule** (the fallback), **Record**.
  - "Implementation roadmap": the milestone table, and a new milestone M10c CPU vector policy after M10.
  - Amends ADR-0057 item 3, ADR-0058 item 4 and ADR-0060 item 2 for the vector entries of `[Cpu]`.

## Context

"Benchmarks", **What it runs** (ADR-0057 item 3, ADR-0060 item 2): each policy entry measures only its baseline, "the configurations or strategies M3s or M9h chose for any device class", and the one its own policy names. "A wider sweep runs only for an entry where no candidate meets the decision rule's bound on every platform." **What it reports**: "the median of ten timed runs (five in the candidate protocol)". The **Generic decision rule** (ADR-0058 item 4) admits a configuration other than the baseline only if it is at least 10% faster than the baseline (geometric mean over the platforms, at each rule size). It keeps the admitted configurations within r ≤ 1.25 on every platform and size, and otherwise falls back to "the smallest worst-case r".

In M10, 11 of the 26 `[Cpu]` vector entries took the fallback, with worst r from 1.28 to 2.63 (M10 Deviations log, T8). They were 4-byte Fill, Copy, Reduce, ScanTile, Gather, Pack, TopologyKeys and EVector, and 8-byte GeomFactors, PullBack and Scalar. The maintainer accepted them rather than wider sweeps. After M10 the maintainer asked for a better CPU policy, with two decisions (2026-11-18): same-device determinism is enough for any CPU launch-shape rule, and an opt-in tuner is a candidate for later.

Two probes followed. Their code and data are uncommitted (probe worktrees at 9327b9b, `target/probe-cpu-blocks/` and `target/probe-cpu-widths/`).

- **Probe 1, blocks capped by the thread count (the T8 follow-up): rejected.** Such a cap is neutral for Copy, TopologyKeys, Pack and Gather. It is 3–15 times slower for Reduce and ScanTile on Grace, whose barrier kernels run on pinned workers that spin (C21).
- **Probe 2, vector widths.** It covered the 10 fallback entries with a width (Scalar has none) on the three Cpu platforms: the Mac mini's CPU runtime, Locust's Grace CPU and one Kathleen socket.
  - **Configurations:** n ∈ {1, 2, 4, 8, 16}, blocks of 4, 16 and 64 KiB, and both unit settings on the Mac. GeomFactors used tile 4 at n ≤ 4 and tile 16 at n ≥ 8.
  - **Sizes:** 10^6 and 10^7.
  - **Timing:** each configuration first had a warm-up, a check and a calibration to at least 20 ms. Then came five rounds, each one batched run of every configuration of the entry in a seeded shuffled order.
  - **Processes:** two per platform (seeds 1 and 2). A configuration's time is the geometric mean of the two processes' medians.
  - **Analysis:** a re-implementation of `decide_generic::generic_rule`.
  - **Wall time:** 5.5–12.6 minutes per process on Grace and Kathleen, and about the same on the Mac outside GeomFactors.

## Problem

**1. Timing order.** M10 timed each configuration's five runs one after another, configuration by configuration. Under the interleaved protocol, 8 of the 10 fallback entries meet the bound with M10's own candidates. Only ScanTile 4B and Pack 4B still fall back. On Grace at 10^7 items, for example, Copy takes 293 µs at n = 1 and 62 µs at n = 8, the reverse of M10's records. ADR-0055 found the same order effect for CPU strategies and interleaved them. The vector candidates kept the sequential order.

The noise is close to the bound. For identical launches, the 90th percentile of the ratio between the two processes' medians is 1.19 on the Mac, 1.23 on Grace and 1.17 on Kathleen. Within one process, the 90th percentile of max/min over the five runs is 1.91, 1.92 and 1.57. A single process cannot resolve a bound of 1.25.

**2. The candidate set.** ADR-0060 item 2 takes the candidates from the choices of the old classes. Each candidate therefore comes as a fixed (width, block, units) triple, and some entries never see the width that suits all three platforms:

- Gather and TopologyKeys had only n = 1;
- Fill, Pack and EVector had n ∈ {1, 16}, and Copy n ∈ {1, 8};
- ScanTile and Reduce had n ∈ {1, 8, 16}, where ScanTile's n = 8 came from the old CpuSimd512 class and ran on all 12 threads of the Mac;
- GeomFactors and PullBack had n ∈ {1, 4, 16}.

The table gives each entry's choice and its worst r against the fastest configuration of the whole probe matrix. The M10 policy column comes from M10's records. The other columns are the generic rule on the probe's data, over the candidate set named in the heading. "FB" marks a fallback within that set.

| Entry | M10 policy | M10's set, interleaved | M10's set + n = 4 | Grid n ∈ {1, 4, 8, 16} |
| --- | --- | --- | --- | --- |
| Fill 4B | n16 64K perf, 1.10 | n16 64K perf, 1.10 | n16 64K perf, 1.10 | n8 64K perf, 1.12 |
| Copy 4B | n1 64K all, 4.97 | n8 64K all, 1.23 | n4 64K all, 1.10 | n4 64K all, 1.10 |
| Reduce 4B | n16 64K perf, 1.02 | n16 64K perf, 1.02 | n16 64K perf, 1.02 | n8 64K perf, 1.11 |
| ScanTile 4B | n1 64K perf, 2.35 | n16 64K perf, 1.55 FB | n4 64K perf, 1.36 FB | n8 64K perf, 1.29 FB |
| Gather 4B | n1 64K perf, 1.38 | n1 64K perf, 1.38 | n1 64K perf, 1.38 FB | n8 16K all, 1.24 |
| Pack 4B | n1 16K all, 1.96 | n1 16K all, 1.96 FB | n1 16K all, 1.96 FB | n1 16K all, 1.96 FB |
| TopologyKeys 4B | n1 64K perf, 1.22 | n1 64K perf, 1.22 | n1 64K perf, 1.22 | n1 64K perf, 1.22 |
| EVector 4B | n1 64K all, 1.69 | n16 64K all, 1.10 | n16 64K all, 1.10 | n16 16K all, 1.15 |
| GeomFactors 8B | n16 t16 4K all, 1.33 | n4 t4 64K all, 1.08 | n4 t4 4K all, 1.09 | n4 t4 4K all, 1.09 |
| PullBack 8B | n16 4K all, 1.32 | n16 4K all, 1.32 | n16 4K all, 1.32 | n8 4K all, 1.23 |

- **M10's set, even when interleaved,** leaves Copy at 1.23, and Gather and PullBack above the bound (1.38, 1.32).
- **Adding n = 4 to that set** fixes Copy but not Gather, ScanTile or PullBack.
- **The grid** meets the bound on 8 entries. It gives the same 10 choices as the full probe matrix with n = 2, which no entry chose.
- **Fill and Reduce** move from n16 to n8. Both are within the bound, and the 3% simplicity order prefers the smaller width.
- **ScanTile 4B is a real conflict.** Grace wants n ≤ 4: 1,479 µs at n = 1 against 2,286 µs at n = 16 at 10^6. The Mac's performance cores want n ≥ 8: 811 µs against 345 µs. No width balances them; the best worst case is n8 at 1.29.
- **The Mac's "n = 8 cliff" in M10's Reduce and ScanTile records came from the unit setting, not the width.** That candidate ran on all 12 threads. macOS does not pin threads (cubecl-cpu `src/compute/affinity/macos.rs`), so the spinning barrier (C21) also waits for the efficiency cores. On the performance cores, the time falls steadily with the width.

**3. The admission gate in the fallback.** Pack 4B's n16 64K all has worst r 1.11 on every platform, but it is only 1.076 times faster than the baseline at 10^6 (geometric mean over the platforms). It is therefore not admitted, and the rule falls back to the baseline, whose worst r is 1.96 (Grace). The gate is meant to keep the simple choice when a wider one does not pay overall. Here it overrides the bound that ADR-0058 introduced: no platform more than 25% slower than its own best.

The fallback code is shared by all four sets. `policy_report` and `high_order_report` were re-run on the committed M10 records with one change (uncommitted): admission is waived when no admitted configuration meets the bound. The results:

- **Vector fallbacks drop from 13 to 9.** Four entries change:
  - `[Cpu]` TopologyKeys 4B: 64 KiB on the performance cores becomes 16 KiB on all threads;
  - `[Cpu]` EVector 4B: n1 becomes n16;
  - `[Gpu]` Reduce 8B: n1 becomes n4;
  - `[Gpu]` Pack 8B: n1 becomes n4. This entry is never read (M10 Deviations log, T2a).
- **Degree fallbacks drop from 29 to 7.** 26 of the 1,520 per-degree choices at p = 1–10 change, 21 on Cpu and 5 on Gpu. Most go from `direct` to `loops` or `sumfact` at single degrees.

On the probe's data, the waiver turns Pack 4B into n16 64K all (1.11) and leaves ScanTile 4B as the only `[Cpu]` fallback of the ten.

**4. Memory.** GeomFactors in f64 at 10^7 points does not fit the Mac mini's 24 GB. The probe reached 8.9 GB resident with 12.4 of 13.3 GB of swap in use. M10's Mac records for this case (33–43 s per call, against 0.9 s on Kathleen) were swap-bound as well, and they vote in the current `[Cpu]` GeomFactors 8B entry. Why the case needs this much memory has not been investigated.

## Options

**Item 1: timing protocol of `[Cpu]` vector candidates**

1. **Keep the design as written (sequential runs).** The choices stay at the mercy of the order effect, as in M10.
2. **Interleaved rounds in one process.** This removes the order effect but not the noise between processes, whose 90th percentile is close to the bound.
3. **Interleaved rounds in two processes (proposed).** As in the probe: five shuffled rounds, two processes with different seeds, and the geometric mean of the two medians. This doubles the timed runs of these entries; the probe shows the cost is minutes.

**Item 2: candidate set of `[Cpu]` vector entries**

1. **Keep the design as written (ADR-0060 item 2, a wider sweep only for fallbacks).** Gather and PullBack stay above the bound. The wider sweeps run entry by entry after a first pass.
2. **ADR-0060 item 2's set plus n = 4.** This fixes Copy only.
3. **A grid (proposed).** Every `[Cpu]` vector entry measures:
   - n ∈ {1, 4, 8, 16};
   - blocks of 4, 16 and 64 KiB;
   - both unit settings where the CPU reports performance cores (ADR-0017). Elsewhere all threads count for both, as now.

   GeomFactors uses tile 4 at n ≤ 4 and tile 16 at n ≥ 8. Scalar measures blocks and units only. That is 24 configurations on the Mac and 12 on the other platforms. The grid replaces the candidate list and the wider sweep for these entries, and the baseline (n = 1 at the old CpuSimd128 block and units) is part of it. It does not apply to `[Gpu]` or to the degree policy.

**Item 3: the admission gate when no admitted configuration meets the bound**

1. **Keep the design as written.** Pack 4B keeps a baseline that is 1.96 times slower on Grace than its best, while n16 is within 1.11 everywhere.
2. **Waive admission in the fallback, in all four sets (proposed).** The fallback becomes:
   - If no admitted configuration meets the bound, the bound and the choice apply to all candidates.
   - Only if none meets the bound, take the admitted configuration with the smallest worst-case r. The report marks it.

   On the committed records this changes 4 vector entries and 26 per-degree choices (Problem 3) without new runs.
3. **Waive admission only for `[Cpu]` vector entries.** The four sets would then follow two different rules, for a reason that is not specific to CPUs.

**Item 4: cases that do not fit a platform's memory**

1. **Keep the design as written.** Swap-bound records keep voting.
2. **Leave out named cases (proposed).** GeomFactors 8B at 10^7 is left out on the Mac mini. The rule then uses the cells present: admission takes the geometric mean over the platforms present at each size, and r and the bound use the cells present. The milestone's smoke run measures the peak resident size of each geometry case at 10^7 on the Mac. A further case that does not fit is added to the list in the plan's Deviations log.
3. **A general swap guard.** The benchmark would read the system's swap use before and after each configuration and reject a record taken while it grew. The counter is system-wide, so other processes can trigger it, and the check adds code to every record. Not chosen now.

**Item 5: where the work happens**

1. **No milestone.** The working agreement requires a plan, an issue and a review for changes to the shipped policy.
2. **A small roadmap milestone, M10c CPU vector policy, after M10 (proposed).**

## Proposed decision

**Item 1: option 3; item 2: option 3; item 3: option 2; item 4: option 2; item 5: option 2.**

M10's `[Cpu]` set rests partly on records that the order effect and the noise between processes can reverse. Copy, one of the most frequent kernels, is 2–5 times slower than its best on all three platforms at the current n = 1. The probe shows that the two changes together are cheap:

- interleaved timing in two processes;
- a grid that lets every width meet every block and unit setting.

They bring 8 of the 10 probed entries within the bound. Waiving the gate in the fallback serves ADR-0058's aim, a set good enough on every platform, and leaves ScanTile 4B as the only real conflict among the probed entries. Every choice still comes from the generic rule on committed records. The tuner stays deferred (maintainer, 2026-11-18).

M10c's scope:

- **Implementation, in gridcl-bench:** items 1–4, as follows.
  - `vector_sweep` gains the interleaved protocol for `[Cpu]` entries, with the process seed in each record.
  - The candidate list for `[Cpu]` vector entries becomes the grid, with the memory omission.
  - `decide_generic::generic_rule` gets the new fallback, and the rule handles missing cells.
  - `check_protocol` and the coverage tests follow.
- **Runs:**
  - Every `[Cpu]` vector entry, 4- and 8-byte, on the three Cpu platforms, two processes each, at 10^6 and 10^7. The estimate, from the probe's 5.5–12.6 minutes for 10 entries, is 30–60 minutes per platform: one Locust run of under 2 hours with the node idle before and after, one whole-node Kathleen job on one socket, and the Mac on mains power.
  - Then the standing matrix.
- **Files:**
  - the records go to `docs/perf/data/m10c-vector-cpu-*.json`, and the `[Cpu]` rule reads them in place of M10's;
  - `vector-policy.toml` (`[Cpu]`, and `[Gpu]` Reduce 8B and Pack 8B through item 3);
  - `degree-policy.toml` (item 3, from the committed records);
  - `docs/perf/vectorization.md` and `docs/perf/high-order.md`, including each platform's noise between processes;
  - C21 gains the macOS note: the CPU runtime cannot pin threads there, so a barrier kernel on all threads also waits for the efficiency cores.

## Consequences

- **Text to change in docs/design.md:**
  - **"Benchmarks", What it runs:** after the ADR-0060 sentence, add: "From M10c, a Cpu vector entry measures the grid of vector sizes 1, 4, 8 and 16, blocks of 4, 16 and 64 KiB and both unit settings (GeomFactors at tile 4 up to size 4 and tile 16 above; Scalar blocks and units only). The grid replaces the candidate list and the wider sweep for these entries. GeomFactors in f64 at 10^7 items is left out on the Mac mini, whose 24 GB it does not fit (ADR-0065)."
  - **"Benchmarks", What it reports:** after "(five in the candidate protocol; ADR-0057)", add: "; on the Cpu class from M10c, vector candidates are timed in five rounds, each one batched run of every configuration of the entry in a seeded shuffled order, in two processes, and a configuration's time is the geometric mean of the two processes' medians; the report gives each platform's noise between processes (ADR-0065)".
  - **"Benchmarks", Generic decision rule, Fallback:** becomes "Fallback: if no admitted configuration meets the bound, the bound and the choice apply to all candidates; if none meets it, take the admitted configuration with the smallest worst-case r, and the report marks it. This replaces the backstops of ADR-0018 and ADR-0021. Where a platform leaves out a size, admission and r use the platforms present (ADR-0065)."
  - **"Benchmarks", Record:** after "and after any CubeCL pin bump (ADR-0057)", add "; the `[Cpu]` vector set is re-derived in M10c (ADR-0065)".
  - **"Implementation roadmap":**
    - the table gains "M10c CPU vector policy | M10";
    - a new section after M10:

      > ### M10c CPU vector policy
      >
      > **Scope (ADR-0065):** the `[Cpu]` vector set re-derived with interleaved timing in two processes over a grid of widths, blocks and units; the generic rule's fallback without the admission gate, applied to all four sets.
      >
      > **Acceptance:**
      >
      > - every `[Cpu]` vector entry is measured on the three Cpu platforms with the protocol and grid of "Benchmarks", and the records carry both processes; `docs/perf/vectorization.md` gives each platform's noise between processes;
      > - `vector-policy.toml` and `degree-policy.toml` follow the generic decision rule, with the fallback of ADR-0065, on the committed records; every entry that misses the bound is marked with its worst case and the cause;
      > - the standing matrix passes with the new sets.
- **ADR notes:** closing notes in ADR-0057 (item 3: the `[Cpu]` vector entries use the grid and the interleaved protocol from M10c), ADR-0058 (item 4: the fallback) and ADR-0060 (item 2: replaced for `[Cpu]` vector entries).
- **CLAUDE.md:** none; the vector bullet already requires benchmark data in docs/perf/vectorization.md.
- **Public API changes:** none. gridcl-bench is unpublished: its record format gains the process seed, and `vector_sweep` gains the interleaved mode.
- **Acceptance tests affected:**
  - `vector_policy_follows_the_generic_rule` and `degree_policy_follows_the_generic_rule` check against the new rule. The degree test's expected file changes in 26 per-degree choices.
  - `decide_generic`'s unit tests gain the waiver case and missing cells.
  - `m10_candidates_cover_every_kind_and_platform` stays for M10's records. A new coverage test checks the grid in M10c's records.
  - M10's records, tests and report sections stay as history.
- **Later milestones affected:**
  - M10c: its plan, an issue "M10c CPU vector policy", and a review.
  - M10's follow-ups: "CPU blocks sized to the thread count" is closed by probe 1 and taken over by this ADR.
  - The opt-in tuner stays a candidate for later.
  - Interleaved timing for `[Gpu]` candidates waits for evidence of the same effect on GPUs.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
