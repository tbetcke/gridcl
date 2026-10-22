---
name: hpc-test
description: Run gridcl tests on the UCL remote targets through scripts/hpc/remote-test.sh — Locust for every CUDA run (kernel tasks, milestone reviews, benchmarks); Myriad V100/A100 only for multi-GPU tests, architecture comparisons and as the fallback; Myriad CPU-only D nodes for AVX-512 CPU benchmarks. Use for every CUDA test run and for 8-rank MPI runs.
argument-hint: "[target] [-- command]"
---

Run a test command on a remote target and report the result. Arguments: $ARGUMENTS

## 1. Check that the runner exists

`scripts/hpc/remote-test.sh` is created in milestone E0. If it does not exist, say that CUDA
tests cannot run until E0 is merged, and stop.

## 2. Choose the target

- **`locust`** unless one of the cases below applies. It is the CUDA platform for every run:
  after kernel changes, at every milestone review, and for benchmarks. No queue, and a build
  takes about a minute (ADR-0014).
- **`myriad-v100`** or **`myriad-a100`** only:
    - as the fallback when Locust is unreachable, a run fails because another user occupies its
      GPU (the `node load:` line names another user's GPU process), or a benchmark cannot get
      an idle node;
    - multi-GPU and GPU-aware MPI tests (`--gpus 2` on V100, `--gpus 4` on A100; Locust has
      one GPU);
    - a Locust failure or result that looks architecture-dependent, to compare (sm_70, sm_80);
    - a change to CUDA-version- or architecture-specific code (CUDA paths of gridcl-device,
      the GPU-aware transport);
    - the user asked for it.

  State the reason in one line when you choose `myriad-v100` or `myriad-a100`.
- **CPU-only runs on Myriad** (x86-64 AVX-512 CPU benchmarks, CpuSimd512, which Locust cannot
  provide) use a CPU-only D node (`-ac allow=D`; 36 cores, Xeon Gold 6240), never a GPU node:
  CPU-only nodes queue far shorter. The runner has no such target yet (a `myriad-cpu` target is
  an M3s follow-up); until it exists, say so and stop rather than using a GPU node.

If the user named a target, use it.

## 3. Choose the command and resources

- Before M0 is merged: `cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda`.
- After M0: the command from the plan's verification line or acceptance-coverage table. The job
  sets `GRIDCL_TEST_DEVICE=cuda`; add `--features gridcl-testing/cuda` as recorded in docs/plans/M0.md.
- `--gpus 2` for multi-GPU or GPU-aware MPI tests on V100 (`--gpus 4` on A100); `--ranks N`
  for MPI runs; `--time 01:00:00` unless the test needs longer.
- Benchmarks (gridcl-bench, from M3s) on Myriad: `--cores 36` so the node's CPU is not shared,
  and a longer `--time`. Copy the JSON results from `target/hpc/<target>/<job id>/` to `docs/perf/data/`
  only when the plan's task asks for it.
- Benchmarks or timings on Locust: Locust has no scheduler and is shared with other users. Use
  the result only if the summary's `node load:` line is `idle` before and after the run;
  otherwise discard it and repeat the run later (ADR-0002).

## 4. Run

```sh
scripts/hpc/remote-test.sh <target> [options] -- <command>
```

- Runs that wait in a queue can take a long time. Run the command in the background and continue
  with work that does not depend on it; check the result before starting a task that does.
- At most two scheduler jobs (Myriad, Kathleen) at a time; check with `scripts/hpc/status.sh`.
  Locust jobs are independent of them: one Locust run may go in parallel with the scheduler jobs.
- Submit Myriad jobs one after the other, never at the same moment: they sync into one remote
  tree and race (rsync exit code 23). Locust syncs to its own host and is not affected.
- Cancel jobs you no longer need with `scripts/hpc/cancel.sh <job id>`, and leave none queued
  at the end of a session.

## 5. Report

Report the target, job id, commit (and whether the tree was dirty), the command, the result, and
the first failure from the runner's summary. On Locust also report the `node load:` line. The full logs are in `target/hpc/<target>/<job id>/`;
read them only as far as needed to locate a failure.

## 6. When it goes wrong

| Symptom | Meaning and action |
| --- | --- |
| SSH asks for a password or times out | The shared connection is closed or the VPN is down. Ask the user to run `ssh myriad true` (or `ssh locust true`); do not retry more than twice. |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` | Only with `cuda-cpp` (NVRTC; the default `cuda` path picks the PTX version for the driver, ADR-0001): the CUDA toolkit is newer than the node's driver. Use the toolkit recorded in docs/dev-environment.md. |
| A V100 build or load failure after loading a CUDA 13 toolkit | CUDA 13 dropped sm_70. Use CUDA 12.x. |
| Exit status 124 | The run exceeded `--max-wait` and was deleted. Report it; resubmit only if the queue was the cause. |
| Locust unavailable, or its GPU occupied by another user | Run on `myriad-v100` instead and say why; Locust runs never block a milestone. |
