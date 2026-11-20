# ADR-0062: The halo exchange's CUDA upload bound applies with one process per GPU

- **Status:** Accepted
- **Date:** 2026-11-17
- **Raised in:** M10 T6 (`halo_exchange` benchmark and records)
- **Design sections affected:** "Implementation roadmap", M10 Hardening, acceptance bullet 1.

## Context

M10's acceptance bullet 1 (as amended by ADR-0059 and ADR-0061) reads:

> the host-staged halo exchange (`forward` and `reverse`, halos of 10^3, 10^4 and 10^5 nodes with block 3) is recorded on the local Mac (CPU runtime, Metal) and Locust (CUDA, 1 and 2 ranks); for 10^5 nodes, its host-to-device copy takes at most 1.5 times its device-to-host copy on Metal and CUDA (…; ADR-0061), and `reverse` at most twice `forward` on each, …

`tetraloma-bench/tests/perf_record.rs::halo_exchange_recorded` checks the upload bound on every Metal and CUDA record set, so on Locust at 1 and at 2 ranks. Locust has one GH200, so its 2-rank run puts both MPI processes on the same GPU. The design's deployment is one GPU per rank on a node (`local_gpu_index`, Listing B8; ADR-0026 item 17, ADR-0060 item 1).

## Problem

M10 T6 recorded the bullet's runs at commit 5003a91 (`halo_exchange`, medians of 50 after 5 warm-ups, each run followed by a device sync, a run's time the largest over the ranks; docs/perf/halo-exchange.md).

At 10^5 nodes (3 · 10^5 values):

| Platform | Ranks | pack | D2H | H2D | unpack | H2D / D2H | reverse / forward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Locust GH200, CUDA f64 (job 20261117132829-1193763, idle before and after) | 1 | 17 | 334 | 403 | 18 | **1.21** | – |
| | 2 | 316 | 337 | 634 | 315 | **1.88** | 1.01 |
| Mac mini, Metal f32 | 2 | 310 | 412 | 413 | 309 | 1.00 | 1.31 |

Times in µs. The 2-rank Locust run misses the bound, and only that run:

- **Two processes time-slice one GPU.** At 2 ranks every kernel costs about 315 µs at every halo size, against 17 µs at 1 rank: the two CUDA contexts run in turn (no MPS on Locust), and each kernel waits for a slice. ADR-0059's probe found the same (a bare sync, 9 µs alone, cost 140 µs with two processes).
- **The upload picks up about one slice; the download does not.** D2H is 334 µs at 1 rank and 337 µs at 2. H2D rises from 403 to 634 µs (quartiles 622–725, minimum 430). The staged upload synchronises with the GPU more than once (T5: `Client::staging`'s blocking submit, then `Client::write` and the sync), and under time-slicing each extra round trip can cost a slice. This is a likely reading, not a measured breakdown.
- **With one process per GPU the bound holds:** 1.21 on the GH200. Metal at 2 ranks also holds (1.00), because there every step, the download included, costs about one slice (~300–400 µs).

So the 2-rank figure measures how two processes share one GPU, which the design's deployment avoids (one GPU per rank). It does not measure the exchange's upload path.

## Options

1. **Keep the design as written.** The bullet then fails on Locust's 2-rank run. Passing it would need either one GPU per rank on Locust, which it does not have, or an upload that synchronises with the GPU once. CubeCL 0.11 has no pinned buffer to fill in place (C23), and the unstaged `Client::write` costs 2.6–2.9 ms on the GH200 (T5), so the second needs an upstream change.
2. **Apply the CUDA upload bound where each process has its own GPU; report the shared-GPU ratio (proposed).**
   - On Locust the bound applies to the 1-rank records.
   - The 2-rank records (two processes on the one GPU) are kept: they give `forward`, `reverse`, the host exchange and the `reverse` ≤ 2 × `forward` check.
   - docs/perf/halo-exchange.md reports their H2D / D2H ratio with its cause, as ADR-0061 does for the CPU runtime.
   - Metal is unchanged (two processes on one GPU, within the bound).
   - No new runs.
3. **Check the bound at 2 ranks on two GPUs.**
   - `halo_exchange` opens `test_client_at(local_gpu_index(comm))` (T4).
   - The 2-rank CUDA run moves to a node with two GPUs, Myriad V100 `--gpus 2` (allowed for multi-GPU tests). Locust keeps its 1-rank run.
   - Costs a Myriad job and its queue time, and adds a third CUDA architecture to this bullet only.
   - Locust's 2-rank shared-GPU figures would still be recorded for `forward` and `reverse`.
4. **Investigate the upload under contention first.**
   - A probe splits H2D at 2 ranks into staging, write and sync, and checks whether dropping a sync keeps the 1-rank gain.
   - About one task. It may need a CubeCL change (C23) and may find nothing tetraloma can change.
   - It can follow option 2 as a follow-up, outside M10.

## Proposed decision

Option 2. The bound exists to keep the upload as fast as the download where the copy crosses a bus (ADR-0059, ADR-0061). On Locust with one process on its GPU it holds (1.21). The 2-rank run on the same GPU adds GPU time-slicing between two processes, which the design's one-GPU-per-rank deployment does not have, and which shows on every kernel (17 → 315 µs), not only on the upload. Reporting that ratio with its cause keeps the information without making M10 depend on a configuration tetraloma does not target. Option 3 would measure the right configuration, but on a different architecture and at the cost of a Myriad job. Option 4 is worth doing as a follow-up if shared-GPU runs become a target.

## Consequences

- **Text to change in docs/design.md:** "Implementation roadmap", M10 Hardening, acceptance bullet 1.
  - "its host-to-device copy takes at most 1.5 times its device-to-host copy on Metal and CUDA (on the CPU runtime, whose read returns a view of the runtime's storage, both are recorded and docs/perf/halo-exchange.md reports the ratio; ADR-0061)"
  - becomes "its host-to-device copy takes at most 1.5 times its device-to-host copy on Metal and on CUDA with one process per GPU (on the CPU runtime, whose read returns a view of the runtime's storage, and on CUDA with two ranks sharing one GPU, both are recorded and docs/perf/halo-exchange.md reports the ratio with its cause; ADR-0061, ADR-0062)".
- **Public API changes:** none.
- **Acceptance tests affected:**
  - `tetraloma-bench/tests/perf_record.rs::halo_exchange_recorded` applies the upload bound to Metal records and to CUDA records at 1 rank.
  - For CUDA records at more than one rank (Locust's one GPU), it checks that docs/perf/halo-exchange.md has the row with the 10^5-node ratio and a cause, as for the CPU runtime.
  - `reverse` ≤ 2 × `forward` is unchanged.
  - The test changes in M10 T6, with a Deviations-log row.
- **Later milestones affected:**
  - M10 T6: the report's cause column gains the shared-GPU cause.
  - Issue #25's bullet 1 is updated to the new text, with a comment naming this ADR.
  - Option 4 goes into the review's follow-ups.

## Decision record

Accepted by the maintainer on 2026-11-17, as proposed (option 2).

Applied in `adr: 0062 apply`: docs/design.md ("Implementation roadmap", M10 acceptance bullet 1) and issue #25. The test change and the report's cause are M10 T6's; option 4 goes into the review's follow-ups.
