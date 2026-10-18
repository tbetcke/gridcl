# ADR-0006: Locust as the routine CUDA test platform, Myriad V100 for reviews and multi-GPU tests

- **Status:** Accepted
- **Date:** 2026-10-18
- **Raised in:** E0 T8/T9 (maintainer's question after the Myriad A100 and Locust runs)
- **Design sections affected:** Overview (development and test targets; decision 10); "Supported backends" (CUDA row); "Testing and validation" (targets table); "Development environment and test targets" (targets table, "When A100 and Locust runs are required", the Locust paragraph, the CI matrix rows cuda-v100 and gh200); "Risks and mitigations" (remote targets row); the closing summary of targets

## Context

The design makes the Myriad V100 nodes the main CUDA platform. "Development environment and test targets" says:

> Myriad V100 is the default for every CUDA run. Claude schedules A100 or Locust runs only: …

The targets table in "Testing and validation" runs the V100 "after every task that changes kernels; every milestone review", and Locust "when required". The design wrote this before E0 had measured anything. Locust's access method and its scheduler were open then: Locust turned out to be an open node without a scheduler (ADR-0002).

E0 has now run the same env-check through the runner on both systems:

| Measured in E0 (2026-10-18) | Myriad V100 | Myriad A100 | Locust |
| --- | --- | --- | --- |
| Queue wait, 1 GPU | median 153 s (T7); 450 s (T6) | 6383 s (job 177336) | none (starts in 1 s) |
| Queue wait, 2 GPUs | 7525 s (job 175401) | – | – (one GPU) |
| Clean release build of env-check (`cuda`) | 1829 s on 8 cores | 5428 s on 8 cores | 85 s with 32 build jobs (`cuda,cpu`) |
| Rebuild after a feature change | 178 s on 8 cores, 952 s on 1 core | – | 7 s |
| Build environment | RHEL 7.9, glibc 2.17: Spack-built LLVM (ADR-0004), offline jobs, a fetch step on the login node | as V100 | RHEL 9.3: CubeCL's prebuilt LLVM bundle, internet access |
| Architecture | sm_70, CUDA 12.2 toolkit, x86-64 | sm_80 | sm_90, CUDA 12.6, aarch64 with 64 KiB pages |

## Problem

CUDA runs "after every task that changes kernels" will happen many times per milestone from M0 on. On Myriad each one waits in the queue and builds on 1–8 cores on a network filesystem. env-check is a small crate; the gridcl workspace will be larger. A single kernel change can therefore cost tens of minutes to hours before its result arrives. Claude has to run such jobs in the background and check them before the next dependent task, so the delay holds up every kernel task. On Locust the same check starts at once and its build takes about a minute. Yet the design uses Locust only "when required", and keeps its runs short because ARC calls it experimental.

Locust cannot replace Myriad entirely:
- It has one GPU, so `--gpus 2` and the GPU-aware MPI tests of M10 need Myriad.
- sm_90 is the newest architecture. The V100 (sm_70) is the oldest one the design supports, and the CUDA 12 toolkit rule exists because of it.
- Its host is aarch64 and Myriad's is x86-64.
- It is an experimental shared node: another user can occupy the GPU, as fmm saw on 2026-10-05 (a 74 GB process at 100% utilisation), and ARC gives no availability guarantee.
- Its environment reuses a Spack environment that the maintainer built for another project (docs/dev-environment.md, "Locust reference").

## Options

1. **Keep the design as written.** The V100 runs after every kernel task. Every kernel task then waits for a Myriad queue and a slow build, as measured above. Not chosen.
2. **Locust for routine CUDA runs; the V100 for milestone reviews, multi-GPU and as the fallback.** After every task that changes kernels, the CUDA tests run on Locust. Every milestone review runs the V100, and Locust as well when it is available. The V100 also runs multi-GPU and GPU-aware MPI tests (`--gpus 2`). It is the fallback when Locust is unreachable, or when a run fails because another user occupies the GPU; the run report says why. The A100 rules are unchanged. Locust runs still never block a milestone, because the review's CUDA check stays on the V100. Architecture-dependent failures on sm_70 surface at the review instead of after the task: the cost of this option.
3. **Locust for everything except multi-GPU.** Myriad is used only for `--gpus 2`, GPU-aware MPI and the A100 review list. This removes sm_70 and x86-64 CUDA from most reviews, and a Locust outage could then block a milestone. Not chosen.

## Proposed decision

Option 2. The frequent check moves to the machine that answers in minutes, and the oldest supported architecture, the multi-GPU tests and the milestone gate stay on Myriad. That puts every milestone review through both the newest and the oldest architecture. If Locust is unavailable, work falls back to today's V100 workflow, so nothing depends on the experimental system.

## Consequences

- Text to change in docs/design.md:
  - Overview: the "Development and test targets" bullet and decision 10: "Locust (GH200) is the routine CUDA platform; Myriad V100 runs at milestone reviews, for multi-GPU tests and as the fallback; Myriad A100 when required". The same in the closing summary of targets.
  - "Supported backends", CUDA row, last column: "Locust (routine); Myriad V100 (reviews, multi-GPU); Myriad A100 when required".
  - "Testing and validation", targets table: Locust row "after every task that changes kernels; milestone reviews when available"; V100 row "every milestone review; multi-GPU and GPU-aware tests; fallback when Locust is unavailable".
  - "Development environment and test targets": the roles in the targets table (Locust: routine CUDA platform, aarch64 host and Hopper GPU; V100: milestone reviews, multi-GPU, fallback, CPU benchmarks on AVX-512); the paragraph "When A100 and Locust runs are required" becomes "When the V100 and A100 run", with the fallback rule and the A100 list unchanged; the Locust paragraph keeps "kept short, never blocks a milestone" and adds that each run uses at most half of the node's cores to build. In the CI matrix, `gh200` gets "after every task that changes kernels", and `cuda-v100` gets "every milestone review; multi-GPU; fallback".
  - "Risks and mitigations", remote-targets row: "Locust (no queue) as the routine CUDA target and the V100 at every review and as the fallback; Locust runs never block a milestone".
- Also changed after acceptance: CLAUDE.md's test-target rules (and the CLAUDE.md listing in "Handing this design to Claude Code"), `.claude/skills/hpc-test/SKILL.md` (default target `locust`, fallback to `myriad-v100`), `.claude/agents/test-runner.md` (default CUDA target), IMPLEMENTATION.md ("CUDA" and "Compare architectures"), and the roles in `docs/dev-environment.md`.
- Public API changes: none.
- Acceptance tests affected: none. E0's per-target acceptance is unchanged. M0's "CUDA (f32, f64) on a Myriad V100 through `scripts/hpc`" stays, as part of M0's review.
- Later milestones affected: from M0 on, the kernel tasks of every milestone use Locust. Reviews of M2, M3, M6, M9 and M10 keep their A100 runs. M10's GPU-aware MPI tests stay on Myriad. M3s and M10 benchmarks follow the idle rule (ADR-0002, ADR-0005). If Locust's Spack environment changes, `acceptance.sh locust` must be re-run.

## Decision record

Accepted by the maintainer on 2026-10-18: option 2.
