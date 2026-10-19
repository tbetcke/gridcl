# Implementing gridcl with Claude Code

Each milestone of `docs/design.md` goes through the same loop: plan, implement, review, merge. You drive it with three skills: `/plan-milestone`, `/implement-milestone` and `/review-milestone`. Everything else (rules, subagents, templates) is in files Claude reads on its own.

## What this archive contains

| Path | Purpose |
| --- | --- |
| `IMPLEMENTATION.md` | This guide |
| `CLAUDE.md` | Working agreement (Listing H1 of the design) plus workflow rules; loaded into every session and subagent |
| `docs/design.md` | The revised Design B, the source of truth |
| `docs/dev-environment.md` | Development and test targets (M3 Max, Myriad V100 and A100, Locust), one-time setup, site reference; milestone E0 completes it |
| `docs/plans/TEMPLATE.md` | Template for milestone plans; plans are saved as `docs/plans/<milestone>.md` |
| `docs/adr/README.md`, `docs/adr/TEMPLATE.md` | How design changes are proposed and decided |
| `.claude/skills/plan-milestone/` | `/plan-milestone <milestone>`: drafts a plan, saves it after you approve |
| `.claude/skills/implement-milestone/` | `/implement-milestone <milestone> [task]`: works through the plan, one commit per task |
| `.claude/skills/review-milestone/` | `/review-milestone <milestone>`: design review, full test evidence, PR description |
| `.claude/skills/draft-adr/` | `/draft-adr <title>`: drafts an ADR; Claude also uses it when the design does not fit |
| `.claude/skills/verify/` | Pre-commit checks (fmt, clippy, tests of touched crates); Claude runs it before each commit |
| `.claude/skills/hpc-test/` | `/hpc-test [target] [-- command]`: runs tests on Myriad or Locust; Claude uses it for every CUDA run |
| `.claude/agents/test-runner.md` | Subagent that runs test suites and reports only failures |
| `.claude/agents/design-reviewer.md` | Read-only subagent that checks plans and diffs against the design |
| `.claude/settings.json` | Pre-approved cargo, git and `scripts/hpc` commands, denied history rewrites, `worktree.baseRef: "head"` |

The development tooling and the Rust workspace do not exist yet. Milestone E0 creates `rust-toolchain.toml`, the remote-test runner (`scripts/hpc/`) and the probe `tools/env-check/`; M0 then creates the workspace, CI and `scripts/mpi-test.sh`.

## One-time setup

1. **Unpack** the archive in the project root. The files sit at the top level of the archive, including the hidden `.claude/` directory:

   ```sh
   cd path/to/gridcl
   unzip gridcl-claude-workflow.zip
   ```

2. **Commit** the files to a git repository:

   ```sh
   git init -b main          # only for a new repository
   git add -A
   git commit -m "Design, working agreement and Claude Code workflow"
   ```

3. **Update Claude Code.** The workflow uses recent features: skills with named arguments, subagent worktrees that start from `HEAD` (`worktree.baseRef`), and Claude running a project `verify` skill before each commit (v2.1.286 or later). Check with `claude --version`.

4. **Check the setup.** Run `claude agents` in a shell; it should list `test-runner` and `design-reviewer`. Start `claude` in the project root and accept the workspace trust prompt, so that the project settings apply. Type `/` and check that the six skills appear.

5. **Prepare remote access** before E0, following "One-time setup" in `docs/dev-environment.md`: a Myriad account, Locust access from UCL ARC, SSH keys, and host aliases `myriad` and `locust` with connection sharing. Claude cannot answer password or second-factor prompts, so open the shared connections at the start of each working day with `ssh myriad true` and `ssh locust true`.

6. **Install local tools** on the M3 Max as the milestones need them. A missing tool is not a blocker: the test-runner reports the backend or tool as unavailable, and CI or a remote target covers it.

   | Needed from | Tool | Used for |
   | --- | --- | --- |
   | E0 | rustup (`rust-toolchain.toml` then installs Rust 1.99) | everything |
   | E0 | rsync 3.x from Homebrew | the remote-test runner |
   | E0 | Open MPI or MPICH (one of them; hosted CI covers both) | MPI smoke test; every milestone from M6 |
   | E0 | the M3 Max's GPU | Metal tests (Vulkan runs in hosted CI) |
   | M0, M1b | OpenBLAS (Linux) or Accelerate (macOS), and network access to codeberg.org | the conformance crate |
   | E0 (used from M4) | Gmsh, one pinned version | generating the fixture corpus once |
   | E0 (used from M5) | Python with `vtk` ≥ 9.2 and `meshio` | VTU read-back tests |
   | M7 | PT-Scotch; optionally ParMETIS | partitioner tests |
   | M8 | HDF5 built with MPI | VTKHDF output |
   | M10 | a Myriad node with a CUDA-aware MPI (E0 records whether one exists) | GPU-aware halo exchange |

## The loop for each milestone

M3 is used as the example. Replace the milestone id and branch name for other milestones. The first milestone is E0, on the branch `e0-dev-environment`.

### 1. Branch from an up-to-date main

Every milestone the new one depends on must be merged into main first (see the table in "Running milestones in parallel").

```sh
git switch main
git pull                         # only if you use a remote
git switch -c m3-serial-mesh-core
```

### 2. Plan (plan mode)

```text
$ claude --permission-mode plan
> /plan-milestone M3
```

Claude checks the starting point, reads the design, the existing code and the ADRs, and drafts the plan. For M3, M6, M7, M9 and plans with more than ten tasks, it first has the design-reviewer critique the draft. Then:

- Review the plan; press **Ctrl+G** to edit it directly.
- Check the acceptance-coverage table: every acceptance bullet of the milestone must appear.
- Answer the open questions in the conversation.
- Approve. Claude writes `docs/plans/M3.md`, commits `plan: M3` and stops.

### 3. Resolve design questions

If the plan lists ADR candidates:

1. Run `/draft-adr <short title>` for each, or let Claude do it.
2. Decide in the ADR file: set the status to Accepted or Rejected and fill in the Decision record.
3. For an accepted ADR, ask Claude: "Apply ADR-0002 to docs/design.md."
4. When nothing is open, set `Status: approved` in the plan (or ask Claude to) and commit.

### 4. Implement (fresh session)

```text
$ claude
> /implement-milestone M3
```

Claude writes the failing acceptance tests first (T0), then implements one task per commit. It runs the `verify` checks before each commit, ticks tasks in the plan, logs deviations, and delegates only the tasks the plan marks as parallel.

- When Claude suggests clearing the context, run `/clear` and then `/implement-milestone M3`. It resumes at the first unticked task. To start at a specific task, use `/implement-milestone M3 T5`.
- If the design does not fit, Claude drafts an ADR and stops. Decide as in step 3, then run `/implement-milestone M3` again.
- **Permission mode:** use the default mode (you approve each edit) for M0 and M1a while you get a feel for the work. Later, `claude --permission-mode acceptEdits` saves prompts. Routine cargo and git commands are pre-approved in `.claude/settings.json`; `git rebase`, `git reset --hard` and `cargo publish` are denied. Claude may push and merge.

At the end, Claude runs the full acceptance suite through the test-runner and sets `Status: implemented`.

### 5. Review (fresh session)

```text
$ claude
> /review-milestone M3
```

Claude runs the design-reviewer on `main...HEAD` and the test-runner on every required backend. It checks each acceptance criterion against a passing test, fixes Blockers and Majors, lists Minors for you, fills the plan's Review section and prints a PR description.

### 6. Merge (you or Claude)

Claude may do these steps when you ask.

1. Push the branch and open a pull request with the printed description.
2. Merge after CI passes.
3. On main, set `Status: done` in `docs/plans/M3.md` and commit (`plan: M3 done`).

## Running milestones in parallel

| Milestone | Can start when these are merged |
| --- | --- |
| E0 Development environment | – |
| M0 Workspace and CI | E0 |
| M1a Type layer | M0 |
| M1b Reference cells | M1a |
| M2 Device primitives and communication core | M0 |
| M3 Serial mesh core | M1b, M2 |
| M3s SIMD vectorization and benchmarks | M3 |
| M4 Gmsh I/O (serial) | M3 |
| M5 VTK output | M3 |
| M6 Parallel core | M3 |
| M7 Partitioners | M6 |
| M8 Parallel I/O | M4, M5, M7 |
| M9 Submeshes and manifolds | M6 |
| M10 Hardening | all |

M1a and M2 can run side by side after M0, M4, M5 and M6 after M3, and M7 and M9 after M6. Give each milestone its own git worktree and its own Claude Code session:

```sh
git worktree add ../gridcl-m4 -b m4-gmsh-io main
git worktree add ../gridcl-m5 -b m5-vtk-output main
cd ../gridcl-m4 && claude --permission-mode plan     # then /plan-milestone M4
```

Merge them into main one at a time, in dependency order. If a branch falls behind main, merge main into it yourself; Claude is not allowed to rebase.

`.claude/settings.json` sets `worktree.baseRef` to `"head"`, so that subagent worktrees inside a milestone start from the milestone branch. The setting also applies to `claude --worktree`. Create milestone worktrees with `git worktree add … main`, as shown, so they start from main.

## Milestone notes

- **E0:** the one milestone where you take part in the work. Have the accounts and SSH aliases ready, keep a terminal open for `ssh myriad true` and similar prompts, and expect Claude to ask you for host names, module names or project codes it cannot discover. The milestone is done when `docs/dev-environment.md` has no open fields left; Locust may stay pending without blocking M0.
- **M0:** the first plan with crates; it builds on the toolchain and runner from E0. M0 checks the `rust-toolchain.toml` that E0 created, and creates the workspace, gridcl-testing, gridcl-mpi-tests, `scripts/mpi-test.sh` and the CI jobs. Make sure `docs/plans/M0.md` records the exact commands for GPU backends and MPI; the test-runner and verify steps use them later.
- **M1b:** the conformance crate needs OpenBLAS (or Accelerate) and network access to codeberg.org for the pinned ndelement commit.
- **M3:** the largest milestone. Expect the design-reviewer critique during planning and several `/clear` cycles during implementation.
- **M3s:** a measurement milestone. Claude runs gridcl-bench on the M3 Max and, through `/hpc-test`, on whole Myriad V100 nodes (CPU and CUDA), and on Locust and A100 nodes if available. Before local benchmark runs, connect the power adapter and close other applications. The result is `vector-policy.toml` plus `docs/perf/vectorization.md`; later changes to the policy need new benchmark data. M3s can run in parallel with M4–M6.
- **M4:** generate the Gmsh fixture corpus once with a pinned Gmsh version and commit it; Claude can write the `.geo` scripts.
- **M6 and later:** run the MPI tests under both MPICH and Open MPI where you can; the `Comm` design depends on it.
- **CUDA:** Claude runs CUDA tests on Locust after kernel changes, on Myriad V100 at every review, for multi-GPU tests and when Locust is unavailable, and schedules Myriad A100 only when the policy in `CLAUDE.md` requires it (ADR-0006). Ask for a specific target with `/hpc-test myriad-a100 -- <command>`.
- **M10:** needs a Myriad node with a CUDA-aware MPI (2 ranks on a V100 node, 4 on an A100 node). If E0 found none, decide early whether to build one in your home directory.

## Useful one-off prompts

| Situation | Prompt |
| --- | --- |
| Resume without the skill | "Continue docs/plans/M3.md from the first unticked task, following CLAUDE.md." |
| Apply an accepted design change | "Apply ADR-0003 to docs/design.md and update the affected plans." |
| Re-plan after a design change | "Re-plan the unticked tasks of docs/plans/M6.md after ADR-0004; keep ticked tasks unchanged." |
| Check GPU backends mid-milestone | "Use the test-runner subagent to run the workspace tests on Metal and on locust and report failures." |
| Benchmark one kernel family | "Run gridcl-bench for GeomFactors on the M3 Max and on a whole myriad-v100 node (CPU and CUDA), and compare with docs/perf/vectorization.md." |
| Compare architectures | "Run the failing test on myriad-v100 and myriad-a100 with /hpc-test and compare with the Locust result." |
| Quick design check | "Use the design-reviewer subagent to check the last three commits against docs/design.md." |
| Start a milestone from scratch | "Discard docs/plans/M5.md and run /plan-milestone M5 again." |

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The skills are missing from the `/` menu | `.claude/skills/` did not exist when the session started: run `/reload-skills` or restart Claude Code |
| A subagent is not found | Restart Claude Code; a new `agents` directory is only picked up at session start |
| A subagent's worktree lacks recent commits | Check that `.claude/settings.json` still sets `worktree.baseRef` to `"head"` |
| Claude stops following a skill late in a long session | Run `/clear` and invoke the skill again; compaction keeps only the start of an invoked skill |
| Too many permission prompts | Add allow rules to `.claude/settings.local.json` (personal) or `.claude/settings.json` (shared) |
| A test needs a tool you do not have | The test-runner reports it as unavailable; CI or a remote target covers it |
| A remote run hangs at SSH or asks for a password | The shared connection is closed or the VPN is down: run `ssh myriad true` in a terminal |
| `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` on a remote target | The CUDA toolkit is newer than the node's driver; use the toolkit recorded in `docs/dev-environment.md` |
| A V100 job fails to compile for `sm_70` | A CUDA 13 toolkit was loaded; V100 jobs need CUDA 12.x |
| Remote jobs left in the queue | `scripts/hpc/status.sh`, then `scripts/hpc/cancel.sh <job id>` |
| The plan and the code disagree | The plan's Deviations log and the ADRs explain why; if they do not, ask the design-reviewer to compare them |
