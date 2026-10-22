# ADR-0014: Locust first for all CUDA runs and benchmarks; Myriad as the fallback

- **Status:** Proposed
- **Date:** 2026-10-22
- **Raised in:** M3, finish (maintainer's decision after the queued Myriad review jobs 3502 and 3503 were cancelled)
- **Design sections affected:** Overview (development and test targets; decision 10); "Supported backends" (CUDA row); "Vectorization and CPU execution" (benchmark "Where" and "Conditions"); "Testing and validation" (targets table); "Development environment and test targets" (targets table, "When the V100 and A100 run", the Locust paragraph, the job limit); "Workspace layout" (CI matrix rows cuda-v100, cuda-a100, gh200); roadmap acceptance of M3s and M10 (benchmark targets); "Risks and mitigations" (remote-targets row); the closing summary of targets. Supersedes ADR-0006.

## Context

ADR-0006 made Locust the routine CUDA platform but kept Myriad as the milestone gate. "Development environment and test targets" says:

> Myriad V100 runs at every milestone review, for multi-GPU and GPU-aware MPI tests (`--gpus 2`; Locust has one GPU), and as the fallback […]. Claude schedules A100 runs only: for the milestone reviews of M2, M3, M6, M9 and M10; […] for benchmarks; […]

The same section ends with "At most two remote jobs run at a time". "Vectorization and CPU execution" runs the benchmarks on "the M3 Max […] and a Myriad V100 node (CPU runtime on its AVX-512 Xeon as CpuSimd512, and CUDA)", with Locust and a Myriad A100 node optional.

At M3's finish, release `cargo test --workspace --features gridcl-testing/cuda` went to all three targets.
- On Myriad the V100 job (3502) and the A100 job (3503) were still queued after about 15 minutes. The V100 job had been requeued once (`Rq`). The A100 job waited behind a 2-hour walltime request.
- A first attempt failed because the two Myriad jobs were submitted at the same moment and raced in the shared remote tree (rsync exit code 23).
- Locust started its run at once.

The maintainer cancelled both Myriad jobs and decided: Locust is the preferred platform for all CUDA runs (routine tests, milestone reviews and benchmarks), and Myriad is only the fallback. On the same day CLAUDE.md was changed (bdb1328), with the hpc-test skill to match: the two-job limit covers only scheduler jobs (Myriad, Kathleen), and one Locust run may go in parallel with them.

## Problem

- **Milestone reviews:** the design's review gate depends on Myriad's queue. As measured in ADR-0006, a fresh A100 target waited 6383 s and a 2-GPU V100 job 7525 s, and M3's review jobs were still queued after 15 minutes. Every milestone finish waits on that queue, although Locust can run the same command within minutes.
- **The job limit:** "At most two remote jobs run at a time" in the design now conflicts with CLAUDE.md, which no longer counts Locust jobs.
- **Benchmarks:** the design's CPU benchmarks run on Myriad, but the maintainer wants Locust first for benchmarks too.

## Options

1. **Keep the design as written (ADR-0006).** The V100 and, where listed, the A100 gate every review, and benchmarks run on Myriad. Every review waits for Myriad's queue. This contradicts the maintainer's decision and the changed CLAUDE.md. Not chosen.
2. **Locust first for everything; Myriad only for what Locust cannot do, or as the fallback.** Locust runs the CUDA tests after every kernel task, at every milestone review and for benchmarks (GPU and CPU). Myriad V100/A100 runs only in these cases:
   - Locust is unreachable, or a run fails because another user occupies its GPU, or a benchmark cannot get an idle node (ADR-0002, ADR-0005). The run report says why.
   - The run needs what Locust lacks: more than one GPU (`--gpus 2`, the GPU-aware MPI tests of M10), an x86-64 AVX-512 CPU (the CpuSimd512 class of "Vectorization and CPU execution"), or sm_70/sm_80, to compare when a failure looks architecture-dependent, or for a change to CUDA-version- or architecture-specific code.
   - The user asks.

   The two-job limit applies to scheduler jobs only. This option is proposed.
3. **Locust first for tests only; benchmarks stay on Myriad.** Reviews move to Locust, but CPU and GPU benchmarks keep the design's targets. This keeps the AVX-512 benchmark data but ignores the maintainer's benchmarking preference. Not chosen.

## Proposed decision

Option 2. Locust answers in minutes and runs the newest architecture (sm_90) and an aarch64 host. Myriad stays available for everything Locust physically cannot provide, so no capability is lost.

The cost: sm_70, CUDA 12 on Volta and x86-64 CUDA hosts are no longer checked at every review. A Volta-only or x86-only CUDA regression may then go unnoticed until a fallback or comparison run. If Locust is unavailable, the review falls back to the V100, so Locust's experimental status never blocks a milestone.

To confirm in the decision record: whether x86 AVX-512 CPU benchmarks (CpuSimd512 in vector-policy.toml) count as "what Locust cannot do", and so still run on Myriad at M3s and M10.

## Consequences

- Text to change in docs/design.md:
  - Overview, the "Development and test targets" bullet, decision 10 and the closing summary: "Locust (GH200) is the CUDA platform for routine runs, milestone reviews and benchmarks; Myriad V100 and A100 run multi-GPU tests, x86-64/AVX-512 CPU benchmarks and architecture comparisons, and are the fallback when Locust is unavailable".
  - "Supported backends", CUDA row, last column: "Locust; Myriad V100/A100 for multi-GPU, comparisons and fallback".
  - "Testing and validation", targets table:
    - Locust row: "after every task that changes kernels; every milestone review; benchmarks".
    - V100 row: "multi-GPU and GPU-aware tests; fallback when Locust is unavailable; architecture comparisons".
    - A100 row: "when required" as rewritten below.
  - "Development environment and test targets":
    - the roles in the targets table;
    - "When the V100 and A100 run" becomes the list of option 2, with the milestone-review items (V100 at every review; A100 at M2, M3, M6, M9 and M10) and "for benchmarks" removed;
    - "At most two remote jobs run at a time" becomes "At most two scheduler jobs (Myriad) run at a time; one Locust run may run in parallel with them. Myriad jobs are submitted one after the other, because they sync into one remote tree".
  - "Vectorization and CPU execution", benchmark "Where": Locust (Grace CPU as CpuSimd128 and H100) and the M3 Max are required, and a Myriad node is required only for the CpuSimd512 data (pending the confirmation above). "Conditions" is unchanged.
  - "Workspace layout", CI matrix:
    - gh200: "after every task that changes kernels; every milestone review";
    - cuda-v100: "multi-GPU; fallback";
    - cuda-a100: "when required".
  - Roadmap acceptance:
    - M3s: benchmark JSON for the M3 Max and Locust, plus a Myriad node for the AVX-512 CPU class.
    - M10: the benchmark list follows the same rule. The GPU-aware tests stay on Myriad.
  - "Risks and mitigations", remote-targets row: "Locust (no queue) for all CUDA runs, and the V100 as the fallback; Locust runs never block a milestone".
- Also changed after acceptance:
  - CLAUDE.md's test-target rules, and its listing in "Handing this design to Claude Code";
  - `.claude/skills/hpc-test/SKILL.md`, section 2;
  - `.claude/agents/test-runner.md`, if it names review targets;
  - IMPLEMENTATION.md ("CUDA" and "Compare architectures");
  - the roles in `docs/dev-environment.md`;
  - the acceptance-coverage tables of open milestone plans: M3's lists Locust, V100 and A100 at the review.
  - ADR-0006 is set to "Superseded by ADR-0014".
- Public API changes: none.
- Acceptance tests affected: none of the tests change. Only their targets change: M3's review runs CUDA on Locust only, and M3s's and M10's benchmark targets change as above.
- Later milestones affected: every review from M3 on runs CUDA on Locust. M10's GPU-aware MPI tests stay on Myriad. M3s and M10 benchmarks follow the idle rule (ADR-0002, ADR-0005), and Myriad is the fallback when Locust stays busy.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
