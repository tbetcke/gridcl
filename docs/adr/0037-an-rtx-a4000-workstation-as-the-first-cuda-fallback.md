# ADR-0037: An RTX A4000 workstation as the first CUDA fallback

- **Status:** Accepted
- **Date:** 2026-11-03
- **Raised in:** M8 T5b (the Locust run after the commit; docs/plans/M8.md)
- **Design sections affected:**
  - "Overview": the development and test targets bullet;
  - "Testing and validation": the targets table;
  - "Development environment and test targets": the targets table, "When the V100 and A100 run", the open-node paragraph, the runner's target list;
  - CLAUDE.md "Test targets"; `.claude/skills/hpc-test`; docs/dev-environment.md.

## Context

"Development environment and test targets" makes Locust (GH200, sm_90) the CUDA platform for every run. Myriad's V100 and A100 nodes are the fallback "when Locust is unreachable, a run fails because another user occupies its GPU, or a benchmark cannot get an idle node" (ADR-0014). Locust is an open node without a scheduler. Myriad's GPU nodes are reached through SGE batch jobs, which can queue; E0 waited 450 s for a V100 and up to its 4 h `--max-wait` for an A100.

The maintainer now has access to a departmental workstation with one NVIDIA RTX A4000:

- 16 GB, compute capability 8.6, driver 580.178 (CUDA 13.0);
- x86-64, 64 hardware threads, 251 GB of RAM, Ubuntu 22.04 (glibc 2.35, libclang 14);
- no scheduler: SSH to the node;
- only CUDA 11.8 under `/usr/local`, no MPI;
- home on a network file system that is 94 % full, plus a local NVMe disk with about 500 GB free, reached as `~/local_disk`.

On 2026-11-03 (survey with read-only commands) the GPU was idle, but another user's process held about 50 of the 64 hardware threads and 126 GB of RAM (load average 63). The SSH alias and the real paths go in `scripts/hpc/targets.env`, as for every other target.

## Problem

The CUDA run after the T5b commit (Locust run 20261103105332-386925) found another user's process at 100 % GPU utilisation before and after the command. 21 gridcl-device doctests failed with `CUDA_ERROR_OUT_OF_MEMORY` when they created a client. Under the design, the only way on is a Myriad V100 batch job and its queue. The design has no CUDA target without a queue other than Locust. It also has no Ampere consumer architecture (sm_86) between the A100 (sm_80) and Hopper (sm_90).

## Options

1. **Keep the design as written.** Locust, with Myriad V100 or A100 as the fallback. Nothing changes; a busy Locust GPU still costs a queue wait. Not chosen: the workstation removes that wait at the cost of one runner target.
2. **Add the workstation as target `a4000`, the first CUDA fallback.**
   - **Role.** Claude runs CUDA tests there when Locust is unreachable or its GPU is occupied by another user, before falling back to Myriad V100. It also runs there to compare, when a result looks architecture-dependent (sm_86 beside sm_70, sm_80, sm_90). Myriad stays the fallback after it, and the only target for multi-GPU and GPU-aware MPI tests; `a4000` has one GPU and no MPI.
   - **Benchmarks.** Never CPU benchmarks: the node's CPU is shared. GPU timings only under the idle rule of ADR-0002, as on Locust. Locust remains the benchmark platform.
   - **Runner.** A direct target like Locust: `scripts/hpc/jobs/a4000.sh`, the `A4000_*` entries in `targets.example.env` (host alias, base directory, login and job environment lines), the node-load snapshot and the `node load:` line, and a run of `scripts/hpc/acceptance.sh a4000` before first use.
   - **Software.** All software for gridcl on the workstation lives under `~/local_disk`. That covers the rustup and cargo homes, a Spack environment with the CUDA toolkit, CubeCL's LLVM bundle, the remote trees and target directories, `TMPDIR` and the kernel caches (`CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, the CubeCL database through the tree's `cubecl.toml`). Nothing goes in the network home or the system directories. The toolkit is CUDA 12.6.3 from Spack, the version on Locust, so a difference between the two targets points at the architecture, not the toolkit; the driver also allows a later CUDA 13 toolkit. Builds use at most 16 jobs (`CARGO_BUILD_JOBS=16`), a quarter of the node, because other users share it.
   - **Setup.** The one-time setup on the workstation (Spack, the toolkit, rustup) needs SSH commands outside the runner. As in milestone E0, each such command needs the maintainer's approval. After the acceptance run, everything goes through `remote-test.sh`.
3. **Make the workstation the routine CUDA platform instead of Locust.** Not chosen: Locust's GH200 is the platform ADR-0014 chose for routine runs and benchmarks (Hopper, aarch64 host). The workstation's shared CPU rules out its benchmarks, and its consumer GPU runs f64 at a small fraction of its f32 rate.

## Proposed decision

Option 2. Most of the fallback runs are correctness runs, and a CUDA target without a queue makes them quick: today's busy Locust GPU needed a V100 batch job. It also adds a fourth CUDA architecture for comparisons. The cost is one direct runner target and an environment kept entirely under `~/local_disk`. The workstation never takes Locust's role as the benchmark platform, and Myriad keeps the multi-GPU and GPU-aware tests.

## Consequences

- **Text to change in docs/design.md:**
  - "Overview", the development and test targets bullet: an RTX A4000 workstation (sm_86) as the first CUDA fallback (ADR-0037).
  - "Testing and validation", targets table: a row "A4000 workstation (sm_86, x86-64, CUDA 12.x) | CUDA in f32 and f64 | first fallback when Locust is unavailable; architecture comparisons (ADR-0037)". The Myriad V100 row's "fallback when Locust is unavailable" becomes "fallback after the A4000 workstation".
  - "Development environment and test targets":
    - the targets table gains a row "A4000 workstation | one RTX A4000 16 GB (sm_86), x86-64, shared CPU | SSH to the node, no scheduler | first CUDA fallback and fourth architecture; no benchmarks, no MPI; software under `~/local_disk`";
    - "When the V100 and A100 run" becomes "When the fallbacks run": the A4000 workstation first, for Locust being unreachable or its GPU occupied, and for comparisons; Myriad for multi-GPU and GPU-aware MPI tests and when the workstation is unavailable too;
    - the open-node paragraph covers the workstation as well;
    - the runner's target list gains `a4000`.
- **CLAUDE.md "Test targets"** and **`.claude/skills/hpc-test`:** the same fallback order and target, and the rule that software on the workstation lives under `~/local_disk`.
- **docs/dev-environment.md:** a section for the workstation, without the host name, user names or real paths: hardware, driver, the Spack environment, the acceptance run.
- **Public API changes:** none.
- **Acceptance tests affected:** none. Milestone reviews still run CUDA on Locust, and the workstation only when Locust is unavailable.
- **Later milestones affected:** none directly. From M10 the GPU-aware tests stay on Myriad.
- **Work:** the runner target, the setup on the workstation and the acceptance run go on their own branch from main (as ADR-0034 did), not into M8's tasks.

## Decision record

Accepted by the maintainer on 2026-11-03, as proposed (option 2), without conditions. The maintainer asked for the setup in the same message, which covers its one-time SSH commands. Applied in `adr: 0037 apply` on its own branch, with the runner target.
