# <Milestone id> <name> — plan

- **Status:** draft | approved | implemented | done
- **Branch:** <milestone id in lower case>-<slug>, for example e0-dev-environment
- **Base commit:** <sha of main when the branch was created>
- **Design sections:** <docs/design.md sections this plan implements, e.g. "Implementation roadmap / M3", "Topology construction on the device">
- **Depends on:** <milestones, each with status done and merged>
- **Accepted ADRs that apply:** <ADR-NNNN, ... or none>

<!--
Rules for this file (do not delete):
- Status moves draft -> approved (maintainer) -> implemented (/implement-milestone) -> done (maintainer, after merge).
- Every task is one commit. T0 is always the acceptance tests.
- The plan references design sections; it never restates or changes requirements.
- Tick a task ([x]) in the same commit that completes it.
-->

## Scope summary

<Two to four sentences: what this milestone delivers, in the design's words. No new requirements.>

## Tasks

- [ ] **T0 Acceptance tests (failing)**
  - Goal: every acceptance bullet of the milestone as an automated test.
  - Crates/files: <...>
  - Stubs: public items with the planned signatures and `todo!()` bodies, so the tests compile and fail at run time.
  - Verify: `<command>` compiles; the new tests fail with `not yet implemented`.
  - After: –
- [ ] **T1 <title>**
  - Goal: <one sentence>
  - Crates/files: <...>
  - Public API added or changed: <items, matching the Architecture listings exactly>
  - Tests: <acceptance tests from T0 that this task makes pass, plus new unit/property tests>
  - Verify: `<command>`
  - After: T0
  - Parallel with: <task ids, or –>
  - Risks / notes: <...>

## Acceptance coverage

Every acceptance bullet of the milestone in docs/design.md appears exactly once.

| Acceptance criterion (docs/design.md) | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| <criterion> | `<crate>::<test>` | T0, T3 | M3 Max: CPU f32/f64, Metal f32; CI: Vulkan f32; Locust: CUDA f32/f64 |

## Open questions and ADR candidates

Resolve every row before Status becomes approved.

| # | Question or conflict | Design section | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| 1 | <...> | <...> | <...> | open / answered / ADR-NNNN |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |

## Review

<Filled by /review-milestone: reviewer findings and how each was resolved, test evidence per target and backend (command, job id for remote runs, result), follow-ups for later milestones.>
