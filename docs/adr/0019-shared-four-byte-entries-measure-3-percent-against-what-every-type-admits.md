# ADR-0019: Shared 4-byte entries measure 3% against what every type admits

- **Status:** Proposed
- **Date:** 2026-10-25
- **Raised in:** M3s T22 (trial `policy_report` on the batched sweep data)
- **Design sections affected:** "Vectorization and CPU execution", "Benchmarks" ("Decision rule"); roadmap acceptance of M3s (bullet 3).

## Context

"Benchmarks", "Decision rule" (ADR-0015, ADR-0018):

> With several devices in one class, the rule uses the geometric mean of the per-device time ratios; the 3% rule chooses among the configurations the 10% rule admits; a 4-byte entry shared by f32 and u32 must be admitted for both (ADR-0015).

ADR-0015, item 6: f32 and u32 share one policy entry (Fill, Reduce, Gather). "That entry takes a size only if it is admitted for every 4-byte type measured; otherwise it takes the smaller size."

M3s acceptance, bullet 3: every chosen configuration "is within 3% of the fastest one the 10% rule admits (ADR-0015), or, where none is, the closest one, at most 10% slower and marked in the report (ADR-0018)".

`perf_record::policy_follows_the_decision_rule` and `decide` read "the fastest one the 10% rule admits" per scalar type: for f32 it is the fastest configuration at any size f32 admits, even a size u32 does not admit, so the shared entry cannot take it.

## Problem

When the two 4-byte types disagree on a size, item 6 keeps the entry at the smaller size, and bullet 3 then compares that size with the faster size of the type that admitted it. The gap is at least the 10% that admitted the faster size, so neither the 3% rule nor the ADR-0018 backstop (at most 10%) can meet it. The entry is unresolved by construction.

It happens with every sweep so far, and is not noise:

| Data | Entry | Admitted for | Closest |
| --- | --- | --- | --- |
| T15/T16 (single-call runs) | Gather CpuSimd128 4-byte | size 16 for f32 only | 74.4% slower |
| T22, first run (batched, ADR-0018) | Gpu Fill 4-byte | sizes 2, 4 for f32 only | 53.0% slower |
| T22, Metal Fill repeated | Gpu Fill 4-byte | size 2 for u32 only | 28.0% slower |

Fill moves the same bytes in f32 and u32, and the two runs of Gpu Fill disagree on which type admits size 2. Its class speedups at size 2 sit at the 10% threshold (geometric mean of Metal and H100 at 10^6: 1.06 and 1.11 in the first run, 1.11 and 0.98 in the second), so which type crosses it is chance. Every other entry of the T22 data is resolved (11 of them by the backstop).

## Options

1. **Keep the design as written.** Every sweep can leave shared entries unresolved, each one goes to the maintainer, and bullet 3 does not hold until they are waived by hand. The M10 re-run and every re-run after a CubeCL bump face the same. Not proposed.
2. **Admission per entry (proposed).** For a shared entry, "the configurations the 10% rule admits" are those at a size every 4-byte type measured admits. The 3% rule and the ADR-0018 backstop then measure each type against its fastest configuration among those, so a size only one type admits is no longer a reference for either. This is how item 6 already decides the size; only the reference of the 3% check follows it. No policy, API or data change.
3. **Separate f32 and u32 entries.** `vector-policy.toml` and `VectorPolicy::vector_size` gain a key for the scalar kind, not only its size. That changes a public API of gridcl-device (Listing S2), the TOML layout and every caller of the policy, for three families whose kernels move the same bytes. Not proposed.
4. **One combined time per shared entry:** the geometric mean over the 4-byte types, admitted and checked as one type. The rule stays one inequality per entry, but it contradicts "must be admitted for both" (ADR-0015), so a size could be adopted that one type does not gain from by 10%. Not proposed.

## Proposed decision

Option 2. ADR-0015 already makes admission a property of the shared entry ("admitted for every 4-byte type measured"), and the 3% rule "chooses among the configurations the 10% rule admits"; reading both at entry level removes a conflict that the noise of a threshold case decides, without loosening the 10% rule (a size above 1 still needs 10% for every type) or the 3% rule (each type is still checked against its own fastest admitted configuration).

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Benchmarks", "Decision rule": "… a 4-byte entry shared by f32 and u32 must be admitted for both (ADR-0015), and for such an entry the 3% rule measures each type against the fastest configuration at a size both admit (ADR-0019). …"
  - Roadmap, M3s acceptance, bullet 3: "… within 3% of the fastest one the 10% rule admits (ADR-0015; for a shared 4-byte entry, at a size every type admits, ADR-0019), or, …".
- Public API changes: none. gridcl-bench (unpublished): `decide` computes each type's fastest over the entry's admitted sizes, and the "admitted for f32 only" note stays in the reason.
- Acceptance tests affected: `perf_record::policy_follows_the_decision_rule` takes the reference of the 3% and backstop checks over the sizes every type with data admits. The other `perf_record` tests are unchanged. A `decide` unit test: a shared entry where one type admits size 2 keeps size 1 and is within 3%.
- Data: none; the committed T22 data are decided again. In the T22 trial, Gpu Fill 4-byte then keeps size 1 within the rule.
- Later milestones affected: M10's re-run and every re-run after a CubeCL bump use the same reading.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
