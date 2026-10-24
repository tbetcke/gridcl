# ADR-0016: Kathleen for CPU-only runs and two-node MPI tests; Myriad D nodes as the fallback

- **Status:** Proposed
- **Date:** 2026-10-24
- **Raised in:** M3s, during T7 (the maintainer's request to add UCL Kathleen as a target)
- **Design sections affected:** Overview (the "Development and test targets" bullet; decision 10); "Vectorization and CPU execution" (the device table row "Myriad's Xeon Gold CPUs"; "Benchmarks", "Where"); "Testing and validation" (targets table); "Development environment and test targets" (targets table, the CPU-only paragraph, "Remote-test runner" steps 1, 3, 4 and the `--cores` paragraph, the job limit); roadmap acceptance of M3s (bullets 1 and 6), M7 and M8 (the 8-rank runs), M10 (benchmark targets); the closing summary of targets.

## Context

ADR-0014 made Locust the platform for every CUDA run. It kept Myriad for what Locust cannot provide, among it the x86-64 AVX-512 CPU of the CpuSimd512 class. Its decision record adds that CPU-only Myriad runs use a CPU-only D node. "Development environment and test targets" now says:

> CPU-only Myriad runs, such as the CPU benchmarks on x86-64 with AVX-512 (CpuSimd512), which Locust cannot provide, use a D node (`-ac allow=D`), never a GPU node: CPU-only nodes queue far shorter.

M3s acceptance bullet 1 asks for benchmark JSON from "a CPU-only Myriad D node", and bullet 6 asks for throughput "on the M3 Max CPU runtime and a Myriad CPU". The M7 and M8 acceptance runs use "8 ranks on one Myriad node". No target runs MPI across nodes: the M3 Max runs up to 4 ranks, hosted CI one runner, and Myriad one node.

M3s T7 is adding the `myriad-cpu` target for the D nodes. Its smoke job (10224, `--cores 36`) was submitted at 16:39 on 2026-10-24 and was still queued at 18:27, after 1 h 48 min.

The maintainer asked on 2026-10-24 to add UCL Kathleen: "a modern CPU cluster with AVX512 and its nodes are easier accessible than Myriad for CPU jobs", usable "for multi-node tests", with "not more than two nodes". The project in ~/projects/fmm already uses Kathleen. Its `tools/kathleen/README.md` and `machine.md` record measured facts, from 2026-10-10 to 2026-10-15:

- **Nodes:** 190 diskless compute nodes, each with 2 × Xeon Gold 6248 (Cascade Lake, 2 × 20 cores, AVX-512 and AVX2), 192 GB, Intel Omni-Path; RHEL 9.6. Myriad's D nodes have the Xeon Gold 6240, also Cascade Lake.
- **Scheduler:** Slurm 26.05, whole nodes only. The QoS are `singlenode` (1 node, 6 h, 2 jobs per user), `test` (exactly 2 nodes, 1 h, 2 jobs per user, high priority) and `small` (2–6 nodes, 48 h).
- **Queue waits:** `singlenode` jobs mostly started within 1–2 s and at worst after 5 min 38 s (9 jobs). 2-node `test` jobs waited 2–4 min (median 3 min 43 s).
- **Environment:** no `$TMPDIR` space, so temporary files go to `/dev/shm`. Compute nodes reach crates.io and GitHub. Rust 1.99 installs through rustup, and the prebuilt tracel-llvm bundle works (RHEL 9, unlike Myriad's RHEL 7). Open MPI 4.1.6 runs over PSM2. Home and Scratch share one 250 GB quota.
- **Settings that a working setup needs:**
  - `PSM2_TID=0`, because PSM2's TID path corrupted data at 4 nodes;
  - unsetting `SLURM_STEP_ID` in the batch step, because otherwise MPI singletons started by `cargo test` abort;
  - the gcc module's runtime first on `LD_LIBRARY_PATH`, for CubeCL's CPU runtime.

Checked from the M3 Max at 18:40 on 2026-10-24 (`ssh kathleen`, login node only):

- **Nodes:** `sinfo -s` showed 190 nodes: 94 allocated, 1 idle and 95 in other states (down or drained). Half the machine was unavailable, so the waits above may not hold every day.
- **QoS:** as recorded by fmm.
- **Quota:** `lquota` showed 159 of 250 GiB used, mostly by the fmm project.

## Problem

- **CPU-only runs wait on Myriad's queue.** The D nodes queue shorter than Myriad's GPU nodes, but job 10224 has still waited about two hours for one whole node. M3s needs many CPU sweep parts on CpuSimd512 (T16) and throughput runs (T17). It then reruns the whole suite there (T18), and M10 repeats the study. Kathleen offers the same CPU microarchitecture and starts single-node jobs within seconds to minutes.
- **No multi-node MPI testing.** gridcl's MPI layer (M6 onwards: halo exchange, ownership, partitioners, parallel I/O) is tested only within one machine. Faults that appear only between nodes are never exercised: the transport, rank placement across nodes, and launch through a scheduler. One example is the PSM2 corruption fmm found at 4 nodes.
- **The design names Myriad for these runs.** M3s acceptance bullets 1 and 6 and the M7 and M8 acceptance runs name Myriad. Using Kathleen instead changes the targets of acceptance tests, which CLAUDE.md allows only through an ADR.
- **The runner has no Slurm support.** `scripts/hpc/remote-test.sh` knows the scheduler kinds `sge` (Myriad) and `direct` (Locust).

## Options

1. **Keep the design as written.** CPU-only runs and the 8-rank runs stay on Myriad D nodes, and there are no multi-node tests. This needs no runner work beyond T7. Not chosen: every CpuSimd512 run waits on Myriad's queue, and inter-node MPI stays untested.
2. **Kathleen for CPU-only runs and MPI runs beyond one machine, at most two nodes; Myriad D nodes as the fallback** (proposed).
   - The runner gains a target `kathleen` with scheduler kind `slurm`.
   - CpuSimd512 data, the M3s throughput runs and the 8-rank runs come from Kathleen.
   - Milestone reviews from M6 on add a two-node MPI run.
   - The D-node target of T7 stays and runs only as a fallback, with the reason recorded, as Myriad's GPU nodes do for CUDA.
3. **Kathleen in addition, Myriad D still required.** Both machines are measured for CpuSimd512, and the class uses the geometric mean of the two (ADR-0015, item 5). This gives more data but keeps Myriad's queue on the critical path, for two CPUs of the same microarchitecture. Not chosen.
4. **Kathleen for multi-node MPI only.** CPU benchmarks stay on Myriad D nodes. This adds inter-node coverage but leaves the queue problem. Not chosen.

## Proposed decision

Option 2. Kathleen gives the same Cascade Lake AVX-512 class as the D nodes, with a much shorter queue, an x86-64 RHEL 9 environment proven by the fmm project, and the only multi-node MPI coverage gridcl would have. Myriad D nodes stay available through `myriad-cpu` for the cases Kathleen cannot serve.

The rules:

- **Node limit:** at most two nodes per job, as the maintainer set. 1 node runs under QoS `singlenode` (up to 6 h). 2 nodes run under `test` (up to 1 h) or, for longer runs, `small`.
- **Job limit:** Kathleen jobs count towards the two scheduler jobs that may run at a time (CLAUDE.md already names Kathleen).
- **CPU-only runs:** CpuSimd512 benchmarks and every other CPU-only run use one whole Kathleen node, which is never shared, so the `--cores 36` rule becomes "one whole node".
- **When Myriad's D nodes run:** only as the fallback, when Kathleen is unreachable, or a job is still queued after `--max-wait`, or the user asks. The record's `fallback_reason` says why, as for CUDA (M3s question 7).
- **MPI:** the 8-rank runs of M7 and M8 run on one Kathleen node. From M6 on, each milestone review also runs the MPI tests on two Kathleen nodes. These are `scripts/mpi-test.sh` at 2, 4, 8 and 16 ranks, spread evenly over both nodes. The two-node runs add a target; they do not change what any test checks.

The runner follows `scripts/hpc` and the facts above:

- **Configuration:** the remote base under the user's Scratch, set in `targets.env` (no user ids or paths in committed files).
- **Sync and fetch:** the same rsync, and `cargo fetch` on the login node, so that jobs stay offline as on Myriad.
- **Submission:** `sbatch --parsable` with `--nodes`, `--qos` and `--time`, polled with `squeue` and finished with `sacct`. `--ranks N` spreads the ranks evenly over the nodes. `--gpus` is rejected.
- **Job environment:** temporary files and Open MPI's session directory on `/dev/shm`, `PSM2_TID=0`, `SLURM_STEP_ID` unset in the batch step, the gcc runtime first on `LD_LIBRARY_PATH`, and `GRIDCL_TEST_DEVICE=cpu`.
- **Status and cancel:** `status.sh` and `cancel.sh` learn `squeue` and `scancel`.

## Consequences

- **Text to change in docs/design.md:**
  - **Overview:** the "Development and test targets" bullet, decision 10 and the closing summary add "UCL Kathleen (Slurm, Xeon Gold 6248 Cascade Lake with AVX-512) for CPU-only runs and CPU benchmarks (CpuSimd512) and for MPI tests across two nodes; Myriad's CPU-only D nodes as the fallback".
  - **"Vectorization and CPU execution":**
    - the device table row "Myriad's Xeon Gold CPUs (x86-64, AVX-512)" reads "Kathleen's and Myriad's Xeon Gold CPUs (x86-64, AVX-512)";
    - "Benchmarks", "Where": "a whole Kathleen node (2 × 20-core Xeon Gold 6248 with AVX-512) for the CPU runtime as CpuSimd512; a Myriad D node only as the fallback (ADR-0016)".
  - **"Testing and validation", targets table:** a row "Kathleen (Slurm, x86-64, AVX-512, Omni-Path) | CPU runtime in f32 and f64; MPI on one and two nodes | CPU benchmarks; MPI runs beyond 4 ranks; two-node MPI at every review from M6 (ADR-0016)".
  - **"Development environment and test targets":**
    - the targets table gains a Kathleen row (hardware as above; "Slurm batch jobs, whole nodes, at most two nodes per job; QoS `singlenode` or `test`"; role as above);
    - the Myriad CPU row's role becomes "fallback for CPU-only runs (ADR-0016)";
    - the CPU-only paragraph reads: "CPU-only runs, such as the CPU benchmarks on x86-64 with AVX-512 (CpuSimd512), use one whole Kathleen node. A Myriad D node (`-ac allow=D`, never a GPU node) runs them only as the fallback.";
    - "Remote-test runner":
      - the target list adds `kathleen` (and `myriad-cpu` from T7);
      - step 1 adds the scheduler kind `slurm`;
      - step 3 says that Kathleen uses the prebuilt LLVM bundle;
      - step 4 adds `sbatch --parsable`;
    - a `--nodes N` option (1 or 2, Kathleen only) joins `--cores`, and "all 36 for CPU benchmarks" becomes "a whole node for CPU benchmarks";
    - the job-limit sentence reads "At most two scheduler jobs (Myriad, Kathleen) run at a time".
  - **Roadmap acceptance:**
    - M3s bullet 1: "… and a whole Kathleen node (CPU runtime on its AVX-512 Xeon), plus a Myriad D node if Kathleen was unavailable, and Myriad's V100 and A100 nodes if Locust was unavailable (ADR-0014, ADR-0016)".
    - M3s bullet 6: "… on the M3 Max CPU runtime and a Kathleen node …".
    - M7 and M8: "8-rank runs on one Kathleen node through `scripts/hpc`".
    - M10: its benchmark bullet follows M3s's.
- **Also changed after acceptance:**
  - CLAUDE.md's test-target rules: Kathleen for CPU-only runs; at most two nodes per job; the D-node rule becomes the fallback rule;
  - `.claude/skills/hpc-test/SKILL.md`; `.claude/agents/test-runner.md`, if it names targets;
  - `docs/dev-environment.md` (a Kathleen section like the fmm project's machine facts);
  - the M3s plan:
    - a new task, after T7, for the `kathleen` target and its fake-Slurm tests;
    - T13, T16, T17 and T18 use Kathleen in place of the D node;
    - `perf_record.rs` requires `kathleen` CPU records, and accepts `myriad-cpu` records only with a `fallback_reason`;
    - the data file `m3s-sweep-cpu-myriad-d.json` becomes `m3s-sweep-cpu-kathleen.json`, and the throughput file `m3s-throughput-cpu-kathleen.json`.
- **Public API changes:** none.
- **Acceptance tests affected:**
  - `gridcl-bench/tests/perf_record.rs::{benchmark_data_covers_required_targets, throughput_recorded_with_policy_and_scalar}`: which target supplies the CpuSimd512 and throughput records. The checks themselves are unchanged.
  - The M7 and M8 8-rank runs: target only.
- **Later milestones affected:**
  - from M6, a two-node MPI run at every review;
  - M7 and M8: their 8-rank runs;
  - M10: CpuSimd512 data for the re-run from Kathleen. Its GPU-aware MPI tests stay on Myriad, since Kathleen has no GPUs.
  - Kathleen's quota (250 GiB, of which 159 GiB were in use on 2026-10-24) limits gridcl to one tree with its `target/` and a shared toolchain, about 30 GB in all.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
