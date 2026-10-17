# Architecture decision records

An ADR records a change to, or a clarification of, docs/design.md. The design is the source of truth; nothing in it changes without an accepted ADR.

## When an ADR is needed

- A public type, trait, signature or listing in "Architecture" or "API walkthrough" must change.
- A crate boundary, dependency rule or feature changes.
- A convention changes (reference cells, orientation, node orderings, file formats).
- An acceptance test turns out to be wrong or impossible.
- The design is ambiguous and the choice affects other milestones.

A choice that stays private to one crate and changes no documented behaviour needs no ADR; record it in the plan's Deviations log instead.

## Process

1. Claude (or you) runs `/draft-adr <short title>`. It writes `docs/adr/NNNN-<slug>.md` from TEMPLATE.md with status Proposed, links it from the milestone plan and commits it.
2. You decide: set the status to Accepted or Rejected and fill in the Decision record.
3. If accepted, ask Claude to "apply ADR-NNNN to docs/design.md". The change is committed as `adr: NNNN apply`, and the plan continues.
4. A later ADR can supersede an earlier one; mark the old one "Superseded by ADR-NNNN".

## Numbering

Four digits, starting at 0001, never reused. TEMPLATE.md and this README are not ADRs.
