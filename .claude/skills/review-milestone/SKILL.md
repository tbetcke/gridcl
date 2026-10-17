---
name: review-milestone
description: Review an implemented gridcl milestone before its pull request - design review, full test evidence, acceptance coverage - and draft the PR description. Run as /review-milestone <milestone>.
argument-hint: <milestone>
arguments: milestone
disable-model-invocation: true
---

# Review milestone $milestone

If the milestone is empty, ask the user which milestone to review.

## 1. Check the starting point

- `docs/plans/<milestone>.md` must have `Status: implemented` and every task ticked.
  If not, stop and list what is missing.
- Be on the plan's branch with a clean working tree.

## 2. Gather evidence (in parallel)

- Ask the **design-reviewer** subagent to review the range `main...HEAD` for this milestone.
  Pass it the milestone id, the plan path and the plan's design sections.
- Ask the **test-runner** subagent for the full run: lint, `cargo test --workspace`, Metal,
  CUDA on Myriad V100, every other target in the plan's acceptance-coverage table (Myriad A100
  and Locust where docs/design.md requires them for this milestone), MPI if the milestone touches
  gridcl-parallel or later crates, and conformance if gridcl-refcell or conformance/ changed.

## 3. Check acceptance coverage yourself

For each row of the plan's acceptance-coverage table, confirm that the named tests exist,
that they assert what the design's acceptance bullet requires (numbers, tolerances,
backends, rank counts), and that they passed in the test-runner report. A test that is
skipped or ignored on a required backend counts as missing. Exception: a required Locust run
that could not run because Locust was unavailable is recorded in the Review section, not
counted as missing.

## 4. Resolve findings

- **Blockers and Majors:** fix them in new commits named `<milestone> review: <summary>`,
  then ask the reviewers again for the affected part. Repeat until none remain.
- If a finding is really a design problem, run the `draft-adr` skill instead of working
  around it.
- **Minors:** list them for the user; fix only those the user chooses.

## 5. Record and hand over

1. Fill the plan's **Review** section: reviewer findings and how each was resolved, the
   test evidence per target and backend (command, job id for remote runs, result), targets
   that were unavailable,
   and follow-ups for later milestones.
2. Commit with the message `review: <milestone>`.
3. Print a PR description:
    - a summary;
    - the acceptance criteria with their tests and results;
    - deviations and ADRs;
    - targets and backends not tested, and why.
4. Tell the user the next steps: push the branch, open the PR, merge after CI passes, then
   set `Status: done` in the plan on main.
