---
name: plan-milestone
description: Plan one gridcl milestone from docs/design.md into docs/plans/<milestone>.md. Run as /plan-milestone <milestone>, preferably in plan mode.
argument-hint: <milestone, e.g. E0, M0, M1a, M3>
arguments: milestone
disable-model-invocation: true
---

# Plan milestone $milestone

If the milestone above is empty, ask the user which milestone to plan, then continue.
Do not write or change any code in this skill. The only file you write is the plan, and only
after the user approves it.

## 1. Check the starting point

- Find the milestone's section under "Implementation roadmap" in docs/design.md and its row in
  the dependency table. If it does not exist, stop and say so.
- Every milestone it depends on must have `docs/plans/<dep>.md` with `Status: done`.
  If not, stop and list what is missing.
- Run `git status` and `git branch --show-current`. The working tree must be clean and the
  branch must be the milestone's branch (the milestone id in lower case plus a slug, for
  example `e0-dev-environment` or `m3-serial-mesh-core`). If not, tell the user the command
  to create it from an up-to-date main, and stop.
- If `docs/plans/<milestone>.md` already exists, read it and ask whether to revise it or start
  over.

## 2. Read

- The milestone's roadmap section and every docs/design.md section its scope or acceptance
  bullets refer to (listings, tables, Appendix A entries).
- CLAUDE.md and every ADR in docs/adr/ with status Accepted.
- The public API of the crates this milestone builds on, as they exist in the repository
  (use the Explore subagent for wide searches). Note every place where the code already
  differs from the design.
- The plans of the milestones it depends on, especially their Deviations logs.

## 3. Draft the plan

Follow docs/plans/TEMPLATE.md exactly.

- **T0** writes every acceptance test of the milestone, with stubs (planned signatures,
  `todo!()` bodies) so that the tests compile and fail at run time.
- **Later tasks** come in dependency order. Each is one reviewable commit: one crate or one
  concern, normally under about 600 changed lines.
- **Per task:** goal, crates and files, public items added or changed (matching the
  Architecture listings exactly), the tests it makes pass, a verification command, and its
  dependencies ("after").
- **Parallel with:** mark a task only if it shares no files and no unmerged API with the other
  task. Do not mark parallel tasks unless the gain is clear.
- **Acceptance coverage:** every acceptance bullet appears exactly once, with test names and
  tasks. Include the targets and backends from "Testing and validation": the M3 Max (CPU
  runtime, Metal), hosted CI (Vulkan, MPI flavours), Locust for CUDA, and Myriad (V100 or A100
  for multi-GPU tests, a CPU-only D node for AVX-512 CPU benchmarks) where "Development
  environment and test targets" requires them for this milestone (ADR-0014).
  Every kernel on the candidate list of "Vectorization and CPU execution" needs a vector-size
  invariance test in the table.
- **Open questions and ADR candidates:** every ambiguity in the design, every conflict with
  existing code, and every place where the design cannot be followed. Do not resolve these
  by guessing; propose a resolution and mark them open.

If the milestone is M3, M6, M7 or M9, or the plan has more than ten tasks, ask the
design-reviewer subagent to critique the draft before presenting it (pass it the draft text,
the milestone id and the design sections), and fix what it finds.

## 4. Present and wait

Present the plan, then a short list of the open questions that need the user's answer.
Wait for approval. If the user changes the plan, revise and present it again.

## 5. Save after approval

1. Write `docs/plans/<milestone>.md` with `Status: approved` if every open question is
   answered, otherwise `Status: draft`.
2. Fill in the branch and base commit (`git merge-base HEAD main`).
3. Commit with the message `plan: <milestone>`.
4. Stop. Tell the user the next step: resolve any open questions or ADR candidates
   (`/draft-adr`), then start a fresh session and run `/implement-milestone <milestone>`.
