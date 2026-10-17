---
name: implement-milestone
description: Implement an approved gridcl milestone plan task by task, committing after each task. Run as /implement-milestone <milestone> [task id]. Resumes from the plan after /clear.
argument-hint: <milestone> [task id to start from]
arguments: milestone task
disable-model-invocation: true
---

# Implement milestone $milestone

Start task (optional): $task

If the milestone is empty, ask the user which milestone to implement.

## 1. Check the starting point

- Read `docs/plans/<milestone>.md`. Its status must be `approved` (or `implemented` if you are
  asked to continue). If it is `draft`, or any row in "Open questions and ADR candidates" is
  still open, stop and list what must be resolved first.
- `git branch --show-current` must be the plan's branch, and `git status` must be clean
  (apart from untracked build output). If not, stop and say what is wrong.
- Read CLAUDE.md, the design sections the plan lists, and the accepted ADRs it names.
- Start at the given task, or else at the first unticked task. Tell the user which task you
  are starting with.

## 2. For each task, in order

1. **Re-read** the task, its design sections and its tests. Keep the change inside the task's
   crates and files; if that is impossible, note it in the Deviations log first.
2. **T0 only:** write all acceptance tests, plus stubs with the planned public signatures and
   `todo!()` bodies. Confirm that the tests compile and fail at run time with
   `not yet implemented`, not with compile errors.
3. **Implement** until the task's tests pass. Follow every rule in CLAUDE.md, in particular the
   kernel, precision, crate-boundary and origin-id rules.
4. **Verify:** run the task's verification command. The `verify` skill runs before each commit.
   When the task changed kernels, also run the affected crates on Metal
   (`GRIDCL_TEST_DEVICE=metal`) and, after committing, on a Myriad V100 with the `hpc-test`
   skill. Run the V100 job in the background; check its result before starting a task that
   depends on this one, and fix failures before `Status: implemented`.
5. **Commit** with the message `<milestone> T<n>: <summary>`, in the same commit tick the
   task (`- [x]`) in the plan and add any Deviations-log row.
6. **Context:** after each task, if the conversation is long or the next task touches other
   crates, tell the user to run `/clear` and then `/implement-milestone <milestone>`; the plan
   is enough to resume.

## 3. Parallel tasks

Use subagents only for tasks that the plan marks "Parallel with" and whose "after"
dependencies are already committed.

- Delegate each to a general-purpose subagent with worktree isolation. `.claude/settings.json`
  sets `worktree.baseRef` to `head`, so the worktree starts from the current commit.
- The delegation prompt contains:
  - the task text from the plan, the design sections it depends on, and the milestone id;
  - the instruction to commit in its worktree with the usual message and to report the branch name;
  - this rule: "If the design is ambiguous or an API must change, stop and return an ADR draft
    instead of deviating."
- When they finish, merge their branches one at a time with `git merge --no-ff`, run the
  affected tests after each merge, then tick the tasks in the plan.

## 4. When the design does not fit

If a public API, crate boundary, convention or acceptance test cannot be implemented as
designed, or the design is ambiguous in a way that affects other milestones: stop the task,
run the `draft-adr` skill, and wait for the user's decision. Do not edit docs/design.md.

## 5. Finish

When every task is ticked:

1. Ask the test-runner subagent for the full acceptance run: `cargo test --workspace`, lint,
   Metal, CUDA on Myriad V100 and any other targets in the plan's coverage table, MPI if the milestone touches
   gridcl-parallel or later crates, and conformance if gridcl-refcell or conformance/ changed.
2. Fix failures as new tasks or as commits to the task they belong to (note them in the
   Deviations log).
3. Set `Status: implemented` in the plan, commit, and tell the user to run
   `/review-milestone <milestone>` in a fresh session.
