---
name: hpc-test
description: Run gridcl tests on the UCL remote targets through scripts/hpc/remote-test.sh — Locust by default, Myriad V100 at milestone reviews, for multi-GPU tests and as the fallback, Myriad A100 only when required. Use for every CUDA test run and for 8-rank MPI runs.
argument-hint: "[target] [-- command]"
---

Run a test command on a remote target and report the result. Arguments: $ARGUMENTS

## 1. Check that the runner exists

`scripts/hpc/remote-test.sh` is created in milestone E0. If it does not exist, say that CUDA
tests cannot run until E0 is merged, and stop.

## 2. Choose the target

- **`locust`** unless one of the cases below applies. It is the routine CUDA platform: no
  queue, and a build takes about a minute (ADR-0006).
- **`myriad-v100`**:
    - at every milestone review (with `locust` as well when it is available);
    - multi-GPU and GPU-aware MPI tests (`--gpus 2`; Locust has one GPU);
    - as the fallback when Locust is unreachable, or a run fails because another user occupies
      its GPU (the `node load:` line names another user's GPU process).
- **`myriad-a100`** only when required:
    - the milestone review of M2, M3, M6, M9 or M10;
    - a Locust or V100 failure or result that looks architecture-dependent, to compare;
    - a change to CUDA-version- or architecture-specific code (CUDA paths of gridcl-device,
      the GPU-aware transport);
    - benchmarks;
    - the user asked for it.

  State the reason in one line when you choose `myriad-v100` outside a review, or `myriad-a100`.

If the user named a target, use it.

## 3. Choose the command and resources

- Before M0 is merged: `cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda`.
- After M0: the command from the plan's verification line or acceptance-coverage table. The job
  sets `GRIDCL_TEST_DEVICE=cuda`; add `--features gridcl-testing/cuda` as recorded in docs/plans/M0.md.
- `--gpus 2` for multi-GPU or GPU-aware MPI tests on V100 (`--gpus 4` on A100); `--ranks N`
  for MPI runs; `--time 01:00:00` unless the test needs longer.
- Benchmarks (gridcl-bench, from M3s): `--cores 36` so the node's CPU is not shared, and a
  longer `--time`. Copy the JSON results from `target/hpc/<target>/<job id>/` to `docs/perf/data/`
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
