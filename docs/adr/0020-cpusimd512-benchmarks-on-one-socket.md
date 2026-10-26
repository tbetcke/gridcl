# ADR-0020: CpuSimd512 benchmarks on one socket of a Kathleen node

- **Status:** Proposed
- **Date:** 2026-10-26
- **Raised in:** M3s T16/T17 (Kathleen sweep, trial `policy_report`)
- **Design sections affected:** "Vectorization and CPU execution", "Benchmarks" ("Where", "Conditions"); "Development environment and test targets" (Kathleen paragraph); roadmap acceptance of M3s (bullets 1 and 6) and of M10 (benchmark bullet).

## Context

"Vectorization and CPU execution", "Benchmarks", "Where":

> […] plus a whole Kathleen node (2 × 20-core Xeon Gold 6248 with AVX-512) for the CPU runtime as CpuSimd512, which Locust cannot provide […]

M3s acceptance, bullet 3: every chosen configuration is within 3% of the fastest admitted one at 10^6 and 10^7 items, or, failing that, the closest admitted one if it is at most 10% slower (ADR-0018). Otherwise the entry is unresolved and goes to the maintainer.

The Kathleen sweep (T16, job 608437, one whole node, all 80 hardware threads allocated, no other job on the node) ran the CPU runtime with 40 threads, one per physical core, across both sockets. CubeCL's CPU runtime takes its thread count from `available_parallelism` and does not pin its threads.

## Problem

With the Kathleen data, `policy_report` leaves CpuSimd512 Fill unresolved (and CpuSimd256 Fill, which copies it), and which Fill entry fails depends on the run:

| Run | Fill 4-byte (f32, u32) | Fill 8-byte (f64) |
| --- | --- | --- |
| First sweep, job 608437 | resolved | unresolved: closest admitted 22.4% slower |
| Fill repeated, job 608489 | unresolved: closest admitted 13.7% slower | resolved |

The cause is not per-call noise; ADR-0018's batching made the ten runs of each configuration agree (spread of the 3rd to 8th run over the median at most 0.05 at 10^7). It is a state that is fixed for one configuration and changes between configurations and runs. For example, these are Fill medians at 10^7 items with 64 KiB blocks, in µs:

| Configuration | 608437 | 608489 | 608494 plain | 608494 plain, again | 608494 `--interleave=all` | 608494 one socket |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f32, size 16 | 340 | 369 | 422 | 98 | 421 | 236 |
| u32, size 4 | 450 | 103 | 109 | 126 | 112 | 275 |
| u32, size 16 | 387 | 112 | 378 | 121 | 305 | 231 |
| f64, size 1 | 408 | 436 | 449 | 348 | 369 | 1,393 |

Diagnostic job 608494 (Fill at 10^7 only, not in docs/perf/data/) ran the sweep four times in one job:

- **Two plain runs on both sockets:** 3–4× outliers appear in different configurations each time.
- **Memory interleaved over both sockets:** the outliers remain, so the placement of the pages is not the cause.
- **One socket (`numactl --cpunodebind=0 --membind=0`; 40 hardware threads on 20 cores, because the binding admits both hyperthreads):** no outliers. The times fall smoothly with the vector size and the block (f32 with 64 KiB blocks: 496 µs at size 1 to 236 µs at size 16), at about half the two-socket bandwidth.

On two sockets, the time of a configuration therefore depends on how the OS places 40 unpinned threads that share a launch's barriers across the two sockets. For some configurations the slow placement is the common one (f32, size 16, 64 KiB: slow in 4 of the 5 two-socket runs above). More repeats would not converge on one policy, and the 3% rule would choose between placements, not between vector sizes. The other families are less sensitive because their launches last longer, but the same placement effect is in all their two-socket times.

## Options

1. **Keep the design as written.** Fill CpuSimd512 stays unresolved (which entry depends on the run), `policy_follows_the_decision_rule` fails, and M3s cannot reach `implemented`. Not chosen.
2. **One socket per CPU benchmark on multi-socket nodes (proposed).**
   - The Kathleen sweep and the Kathleen throughput runs bind to one socket and its memory, one thread per physical core: `numactl --physcpubind=0-19 --membind=0` in the job (CPUs 0–19 are socket 0's first hardware threads in the job's CPU set). The CPU runtime then runs 20 threads.
   - The node is still reserved whole, so nothing else runs on the other socket.
   - CpuSimd512 then describes one 20-core AVX-512 socket, which is also what one MPI rank per socket sees on Kathleen.
   - The whole Kathleen sweep (about 30 minutes) is run again under the binding, with a second Fill run to confirm that the table no longer changes between runs.
   - The suite runs (T18) and MPI runs keep whole nodes; only benchmarks bind.
   - No API or runner change: the binding is part of the job's command, documented in docs/dev-environment.md and the hpc-test skill.
3. **Median over independent runs on both sockets.** Sweep three times in separate jobs and take, per configuration, the median of the three medians. This keeps the whole node (about 1.5 hours of Kathleen time) but chooses between OS placements where the slow one is the common one. It also gives one device class a protocol the others do not have.
4. **Logged exception.** Unresolved CpuSimd512 entries keep the defaults (size 1, 16 KiB, all threads), and `perf_record` accepts a named list. This is cheap, but the list changes with every run (the failing Fill entry already did), and bullet 3 then holds only with exceptions, for M10 too.

## Proposed decision

Option 2. The outliers come from thread placement across two sockets, which the vector policy cannot control and the protocol cannot average away. On one socket the times are stable and ordered by vector size and block, so the rule measures what it is meant to. One socket is also the unit an MPI run on Kathleen places one rank on. Thread pinning in the CubeCL CPU runtime, which would make two-socket runs stable, is left to M10 (and upstream CubeCL).

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Benchmarks", "Where": "… plus one socket of a whole Kathleen node (20 cores of a Xeon Gold 6248 with AVX-512, bound with its memory; ADR-0020) for the CPU runtime as CpuSimd512 …".
  - "Benchmarks", "Conditions": add "On a node with several sockets, CPU benchmarks bind to one socket and its memory, one thread per physical core, because the CPU runtime's unpinned threads across sockets make a configuration's time depend on their placement (up to 4×; ADR-0020)."
  - "Development environment and test targets", the Kathleen paragraph: "CPU-only runs, such as the CPU benchmarks on x86-64 with AVX-512 (CpuSimd512), use one whole Kathleen node; benchmarks bind to one socket (ADR-0020)."
  - Roadmap, M3s acceptance, bullet 1: "… and one socket of a whole Kathleen node (CPU runtime on its AVX-512 Xeon; ADR-0020) …"; bullet 6: "… on the M3 Max CPU runtime and one socket of a Kathleen node (ADR-0016, ADR-0020) …".
  - Roadmap, M10, benchmark bullet: "… and one socket of a whole Kathleen node (AVX-512; ADR-0020) …".
- Public API changes: none. gridcl-bench records already carry the thread count (`cores.threads`, 20 under the binding) and the units (`units`).
- Acceptance tests affected: none change their code. `benchmark_data_covers_required_targets` and `throughput_recorded_with_policy_and_scalar` read the new Kathleen records; `policy_follows_the_decision_rule` must then pass for CpuSimd512 and CpuSimd256.
- Data: `docs/perf/data/m3s-sweep-cpu-kathleen.json` is replaced by the bound sweep (T16); the Kathleen throughput (T17) runs bound. docs/perf/vectorization.md records the binding in its conditions, and the two-socket evidence above under observations.
- Later milestones affected: M10's re-run uses the binding on Kathleen (and on any Myriad D-node fallback, also two sockets), and investigates thread pinning in the CPU runtime. MPI milestones are unaffected.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
