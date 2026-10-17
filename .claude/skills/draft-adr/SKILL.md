---
name: draft-adr
description: Draft a gridcl architecture decision record in docs/adr/ when an implementation must deviate from docs/design.md or the design is ambiguous - a public API, crate boundary, convention or acceptance test that cannot be followed as written. Then stop for the maintainer's decision.
argument-hint: <short title>
---

# Draft an ADR

Title (may be empty; then derive one from the problem): $ARGUMENTS

1. **Number:** the next free four-digit number in docs/adr/ (start at 0001; TEMPLATE.md and
   README.md are not ADRs). File name: `docs/adr/NNNN-<kebab-case-title>.md`.
2. **Write it from docs/adr/TEMPLATE.md** with `Status: Proposed` and today's date:
    - Context: the design section and what it says (quote at most a few lines), and what the
      implementation found.
    - Problem: concrete evidence (compiler error, failing test, measurement, conflicting sections).
    - Options: at least two, one of which is "keep the design as written".
    - Proposed decision and its consequences, including the exact docs/design.md sections
      and listings that would change, public API changes, affected acceptance tests and later
      milestones.
3. **Link it** from the current milestone plan: a row under "Open questions and ADR
   candidates" with status `ADR-NNNN`, and a Deviations-log row if work already deviates.
4. **Commit** with the message `adr: NNNN <title>`.
5. **Stop.** Summarise the decision needed in two or three sentences and wait. Do not edit
   docs/design.md, and do not continue the affected task, until the maintainer sets the ADR
   to Accepted or Rejected. When asked to apply an accepted ADR, edit exactly the sections it
   names and commit with `adr: NNNN apply`.
