# ADR-0005: Node-load idle rule that does not count the runner's own work

- **Status:** Proposed
- **Date:** 2026-10-18
- **Raised in:** E0 T9 (Locust acceptance)
- **Design sections affected:** "Development environment and test targets" (remote-test runner, step 4); amends ADR-0002

## Context

ADR-0002 (accepted) added node-load snapshots to direct targets. "Development environment and test targets", step 4, defines the idle rule:

> The node counts as idle if no other user has a GPU compute process, GPU utilisation is below 5%, other users' processes use less than one core in total, and the 1-minute load average is below 2.

"Vectorization and CPU execution / Benchmarks" and CLAUDE.md accept a Locust benchmark or timing only if both snapshots are idle. E0 T4 implemented the rule in `scripts/hpc/jobs/common.sh`: the "before" snapshot follows the job's build, right before the command, and the "after" snapshot follows the command. Other users' CPU use is sampled over 1 s from `/proc`.

## Problem

The 1-minute load average counts every process on the node, including the job's own build and the runner's previous job. It decays with a time constant of one minute. E0 T9 ran `scripts/hpc/acceptance.sh locust` on 2026-10-18 on a node that was otherwise quiet: other users' processes used 0.05–0.7 cores, and no other process used the GPU. Even so, most snapshots were `busy`, and the load average was the only reason:

| Run | Before the command | After the command |
| --- | --- | --- |
| smoke run (85 s build with 32 jobs, then 2 s command) | busy: load average 5.07 | busy: load average 5.07 |
| turnaround-1 (no rebuild; 2 min after the smoke run) | busy: 2.83 | busy: 2.60 |
| turnaround-2 (14 s later) | busy: 2.20 | busy: 2.10 |
| turnaround-3 | idle | idle |
| nvrtc (7 s build) | idle | idle |
| gpus2 (2-rank MPI test, build included) | idle | busy: 4.14 |
| fail (10 s after gpus2) | busy: 3.73 | busy: 3.73 |

So the rule rejects almost every run that follows a build, and a benchmark always follows one (`cargo bench` builds first). The "after" snapshot has a worse problem: a CPU benchmark on all 72 Grace cores raises the load average by up to 72 by itself, so its "after" snapshot can never be idle. Under the rule as written, no multi-core CPU benchmark on Locust could ever count (M3s, M10).

## Options

1. **Keep the design as written.** Benchmarks on Locust would need a pause of a few minutes between the build and the command, and multi-core CPU benchmarks could still never pass the "after" check. Not chosen: the rule would block the benchmarks it is meant to protect.
2. **Wait for the load average to settle before the "before" snapshot.** After the build, the job polls until the 1-minute load average falls below 2 (at most 5 min), then takes the snapshot. The rule stays the same. This costs up to about 2 min per direct job, and it does not fix the "after" snapshot of a multi-core benchmark. Not chosen.
3. **Drop the load average from the idle rule and measure other users' CPU use over a longer window.** Both snapshots still record the 1-minute load average, but only as information. The rule becomes: no GPU compute process of another user, GPU utilisation below 5%, and other users' processes below one core in total, sampled over 5 s instead of 1 s. That criterion excludes the job's own processes by construction (the uid comparison of ADR-0002), and the longer sample takes over the load average's job of catching other users' bursty load. No CLI change, and each snapshot costs 4 s more.

## Proposed decision

Option 3. On a shared node, the question is whether other users interfere, and per-uid CPU accounting answers it directly. The node-wide load average cannot separate other users from the job itself. A 5 s sample keeps some of the protection against bursty load that the load average gave.

## Consequences

- Text to change in docs/design.md:
  - "Development environment and test targets", step 4: "… the 1-minute load average (recorded only), and the CPU share of other users' processes over 5 s. The node counts as idle if no other user has a GPU compute process, GPU utilisation is below 5%, and other users' processes use less than one core in total." The `node load:` line of step 6 and the benchmark condition in "Vectorization and CPU execution / Benchmarks" stay as they are.
  - ADR-0002 gets the note "Idle rule amended by ADR-0005".
- Implementation: `scripts/hpc/jobs/common.sh` (`gridcl_job_node_load` drops the load-average condition; `gridcl_job_cpu_others` samples 5 s); `docs/dev-environment.md` (Locust reference, node load). The run.sh assertions (`node_load=(idle|busy)`, the summary line) stay valid.
- Public API changes: none. The runner CLI and the summary format are unchanged.
- Acceptance tests affected: none of E0's acceptance criteria (tests are not timing-sensitive). E0 T9 records the node-load lines under the current rule. They would be re-recorded under the new rule only if the maintainer wants that.
- Later milestones affected: M3s and M10 benchmarks on Locust become possible under the idle rule; GPU utilisation in the "after" snapshot is still sampled right after the job's own GPU work, which M3s/M10 should watch.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
