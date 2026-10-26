# ADR-0021: A wider backstop for shared 4-byte entries whose types disagree

- **Status:** Accepted
- **Date:** 2026-10-26
- **Raised in:** M3s T16/T17 (Kathleen sweep on one socket, trial `policy_report`)
- **Design sections affected:** "Vectorization and CPU execution", "Benchmarks" ("Decision rule"); roadmap acceptance of M3s (bullet 3).

## Context

"Benchmarks", "Decision rule":

> […] a 4-byte entry shared by f32 and u32 must be admitted for both (ADR-0015), and for such an entry the 3% rule measures each type against the fastest configuration at a size both admit (ADR-0019). If no admitted configuration is within 3% of the fastest at both sizes, the one with the smallest worst-case ratio is chosen if that ratio is at most 1.10, and the report marks it (ADR-0018).

ADR-0015, item 6: f32 and u32 share one policy entry per kind (`VectorPolicy::vector_size(kind, elem_bytes)` and the CPU block and units entries). The entry takes a size only if every 4-byte type admits it. ADR-0018 bounds the backstop at 1.10, and above that the entry is unresolved and goes to the maintainer.

## Problem

After ADR-0020, the Kathleen sweep on one socket (job 608520) is stable at 10^7. A second Fill sweep in the same job agrees within 0.76–1.08× there. CpuSimd512 Fill 4-byte (and CpuSimd256, which copies it) is still unresolved, because f32 and u32 Fill behave differently on this CPU:

| Fill on one Kathleen socket, best time per size over size 1 (10^6 / 10^7) | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | --- |
| f32 | 1.28 / 1.43 | 1.10 / 1.85 | 0.96 / 1.89 | 0.87 / 2.03 |
| u32 | 1.07 / 2.36 | 1.00 / 2.64 | 1.34 / 2.76 | 1.15 / 2.84 |

- **Admission:** f32 admits sizes 2 and 4, and u32 admits 8 and 16. No size above 1 is admitted for both, so the entry keeps size 1 (ADR-0015, item 6).
- **Blocks at size 1:** the types prefer different blocks. Each block's ratio to the type's fastest size-1 configuration, at 10^6 and 10^7:

  | Block | f32 | u32 | Worst |
  | --- | --- | --- | --- |
  | 4 KiB | 1.00 / 1.16 | 1.00 / 1.00 | 1.16 |
  | 16 KiB | 1.01 / 1.15 | 1.13 / 1.01 | 1.15 |
  | 64 KiB | 1.00 / 1.00 | 1.69 / 1.01 | 1.69 |

  The closest configuration is size 1 with 16 KiB blocks, 14.7% from the fastest for some type and size. That is above ADR-0018's 1.10.
- **Not noise alone:** with the geometric mean of the two Fill runs per configuration the entry is still unresolved (12.6%); with only the check run's Fill records it resolves. The 10^6 calls last 60–140 µs, so admission at 10^6 is close to the 10% threshold for both types. The underlying difference is real: at size 1, u32 Fill takes about 1.6× as long as f32 Fill at 10^7 for the same bytes (about 990 µs against 600 µs), so u32 gains much more from vectors.

A shared entry cannot serve two types that disagree this much within 10%, however often it is measured. `policy_follows_the_decision_rule` fails, and M3s cannot reach `implemented`.

## Options

1. **Keep the design as written.** The entry stays unresolved and bullet 3 fails. Not chosen.
2. **A wider backstop for shared entries whose types disagree (proposed; maintainer's direction, 2026-10-26).** For a shared 4-byte entry, the backstop takes the configuration with the smallest worst-case ratio over every type and both sizes. The bound is 1.25, not 1.10. The report marks it "closest across types, x% slower". The entry is unresolved only above 1.25. Single-type entries keep ADR-0018's 1.10. Here this gives size 1, 16 KiB blocks, all threads, 14.7%.
3. **Per-type 4-byte entries.** Split f32 and u32 in `vector-policy.toml` and in `VectorPolicy::vector_size` and its CPU block and unit lookups. This removes the conflict, but it changes Listing S2 and ADR-0015 item 6, and every kernel's plan call would pass a type instead of an element size. That is too large for M3s's last task.
4. **Defaults and a named exception list.** The entry keeps size 1, 16 KiB, all threads, and `perf_record` accepts a list of named entries. This has the same effect here, but the list has to be maintained by hand for M10 and every CubeCL bump, while option 2 is a rule.

## Proposed decision

Option 2. Admission already makes a shared entry follow its most conservative type (ADR-0015, ADR-0019). This change only lets the backstop, which already exists, account for two types pulling the block in different directions. The bound of 1.25 keeps a gross mismatch unresolved, and the report says how far the entry is from each type's fastest. Per-type entries (option 3) remain open for M10 if more shared entries hit this.

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Benchmarks", "Decision rule", last sentence: "If no admitted configuration is within 3% of the fastest at both sizes, the one with the smallest worst-case ratio is chosen if that ratio is at most 1.10, or at most 1.25 for a 4-byte entry shared by types that admit different sizes or prefer different configurations (ADR-0021), and the report marks it (ADR-0018)."
  - Roadmap, M3s acceptance, bullet 3: "… or, where none is, the closest one, at most 10% slower (25% for a shared 4-byte entry, ADR-0021) and marked in the report (ADR-0018);".
- Public API changes: none. gridcl-bench (unpublished): `decide` applies the 1.25 bound to shared 4-byte entries with more than one type measured, and its reason reads "closest across types, x% slower". `report` keeps the "closest" marking.
- Acceptance tests affected: `perf_record::policy_follows_the_decision_rule` accepts a backstop entry above 1.10 and at most 1.25 only for a shared 4-byte entry with more than one type measured, and only if its report rows say "closest across types, ". New `decide` unit tests: a shared entry at 1.20 resolves; a shared entry at 1.30 and a single-type entry at 1.20 stay unresolved.
- Data: none new. The committed Kathleen data then decide CpuSimd512 and CpuSimd256 Fill 4-byte as size 1, 16 KiB, all threads, 14.7%.
- Later milestones affected: M10's re-run and every re-run after a CubeCL bump apply the same bound. M10 may revisit per-type entries (option 3).

## Decision record

Accepted by the maintainer, 2026-10-26, as proposed.
