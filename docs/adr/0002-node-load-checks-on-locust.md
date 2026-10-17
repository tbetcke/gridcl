# ADR-0002: Node-load checks on Locust, an open node without a scheduler

- **Status:** Accepted
- **Date:** 2026-10-17
- **Raised in:** E0 T4 (maintainer's note before T4 started)
- **Design sections affected:** "Vectorization and CPU execution / Benchmarks" (Conditions); "Development environment and test targets" (remote-test runner, steps 4 and 6; the Locust paragraph); the CLAUDE.md listing of test-target rules

## Context

The design already treats Locust as a `direct` target: the runner submits "through `ssh` with `timeout` for direct targets" ("Development environment and test targets", step 4), and E0 T4 implements that kind. For benchmarks the design fixes the conditions on two targets only ("Benchmarks", Conditions):

> on Myriad, CPU benchmarks request every core of the node (`--cores 36`) so that no other job shares it. On the M3 Max they run on mains power with other applications closed …

and CLAUDE.md adds "never draw SIMD conclusions from a node shared with other jobs". Locust (GH200: Grace CPU and H100) is listed as a benchmark target "when available".

The maintainer has confirmed that Locust has no job scheduler: it is an open multi-user node. Nothing reserves its GPU or CPU cores for a run.

## Problem

On Locust there is no equivalent of `--cores 36`. Any other user's process on the Grace cores or the H100 can distort a benchmark or a timing (the turnaround and run-time records of E0 T9, the M3s vector-size study, the M10 throughput benchmarks), and nothing in the design or the runner would notice. The design states benchmark conditions for Myriad and the M3 Max and none for Locust, so the gap affects milestones E0, M3s and M10.

## Options

1. **Keep the design as written.** Locust runs proceed with no check. Benchmark results from Locust may silently include interference from other users, and the M3s decision rule (10% speedup thresholds, 3% ties) could be decided by noise. Not chosen.
2. **Record the node load before and after every direct job, and accept Locust benchmarks only from an idle node.** The runner's direct kind takes a node-load snapshot immediately before the command starts and immediately after it ends: `nvidia-smi` GPU utilisation and memory use, the compute processes on the GPU with their owners (`nvidia-smi --query-compute-apps`, `ps`), the 1-minute load average, and the CPU share of processes owned by other users. Both snapshots are written to `results/<job id>/` and the summary gets one `node load:` line, `idle` or `busy` with the reason. "Idle" means: no GPU compute process of another user, GPU utilisation below 5%, other users' processes using less than one core in total, and a 1-minute load average below 2. Tests are unaffected (they are not timing-sensitive). Benchmark and timing results from Locust count only if both snapshots are `idle`; otherwise they are discarded and the run is repeated later. No CLI change.
3. **As option 2, plus a runner option `--require-idle`** that refuses to start (exit 75) when the "before" snapshot is busy, and marks the run failed when the "after" snapshot is busy. Saves a wasted benchmark run, but changes the runner CLI fixed in "Development environment and test targets", and still needs the "after" check of option 2.

## Proposed decision

Option 2. It closes the gap with the least change: the runner records the evidence on every Locust run, the summary makes a busy node visible to Claude and to developers, and the benchmark rule becomes explicit, like the `--cores 36` rule on Myriad. The CLI stays as designed; if repeated busy runs waste too much time, `--require-idle` (option 3) can follow in a later ADR.

## Consequences

- Text to change in docs/design.md:
  - "Vectorization and CPU execution / Benchmarks", Conditions: add "Locust has no scheduler and is shared with other users: a benchmark there counts only if the runner's node-load snapshots before and after the run both show the node idle; otherwise it is repeated."
  - "Development environment and test targets", runner step 4: for direct targets, the job records a node-load snapshot before and after the command; step 6: the summary includes a `node load:` line. The Locust paragraph ("Locust is an experimental system …") adds that it is an open multi-user node without a scheduler.
  - The CLAUDE.md listing ("Handing this design to Claude Code") and the repository's CLAUDE.md test-target rules: add "Locust has no scheduler: never draw benchmark or timing conclusions from a Locust run unless its `node load:` line is `idle` before and after."
  - Also updated: `.claude/skills/hpc-test/SKILL.md` (benchmarks on Locust; the `node load:` line in the report) and `docs/dev-environment.md` (Locust access: open node, no scheduler).
- Public API changes: none (no library code; the runner CLI is unchanged).
- Acceptance tests affected: E0 T4 adds a `scripts/hpc/tests/run.sh` assertion that a direct job writes both snapshots and that the summary has a `node load:` line (within the 40-line limit). E0 T9 records the node-load lines with Locust's turnaround runs.
- Later milestones affected: M3s and M10 benchmarks on Locust follow the idle rule; the benchmark JSON from Locust records both snapshots.

## Decision record

Accepted by the maintainer on 2026-10-17: option 2, with the idle thresholds as proposed.
