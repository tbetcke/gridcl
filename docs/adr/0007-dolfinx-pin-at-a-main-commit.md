# ADR-0007: Pin DOLFINx at a main commit for the Gmsh P3 acceptance arrays

- **Status:** Proposed
- **Date:** 2026-10-19
- **Raised in:** M1b T0 (plan question 7)
- **Design sections affected:** none in wording; the pin that A.10 ("Sources and conformance setup") and "High-order node orderings" leave open, and docs/plans/M1b.md question 7

## Context

M1b's acceptance requires the Gmsh generator to "reproduce DOLFINx's Gmsh P3 tables (tet20 after A.11; quad16, hex64, prism40 and pyramid30 unchanged)" (Roadmap M1b; A.9; "High-order node orderings"). A.10 lists DOLFINx `cpp/dolfinx/io/cells.cpp` as a source and says the fixtures carry "the pinned commits of ndelement, Basix, DOLFINx and Symfem", without naming them.

The M1b plan (question 7) resolved this as "pin each project's latest release tag at T0", with the condition: "T0 confirms the five arrays exist at the DOLFINx tag; if one is missing, stop and draft an ADR."

## Problem

T0 checked `cpp/dolfinx/io/cells.cpp` on 2026-10-19:

| DOLFINx ref | gmsh_tetrahedron 20 | gmsh_quadrilateral 16 | gmsh_hexahedron 64 | gmsh_prism 40 | gmsh_pyramid 30 |
| --- | --- | --- | --- | --- | --- |
| v0.11.0.post0 (latest release, 2026-06-10, fefdb22) | yes | yes | no (8, 27) | no (6, 15, 18) | no (5, 13, 14) |
| c9e0341 (main, #4617 "Support third-order Gmsh hexahedra, prisms and pyramids", 2026-10-10) | yes | yes | yes | yes | yes |
| be8352e (main HEAD, 2026-10-19) | yes | yes | yes | yes | yes |

The five arrays are identical from c9e0341 to be8352e. The commits in between (#4616, #4621, #4623, #4628, #4630, #4650) change only error types, VTK cell-type handling and logging in that file. No DOLFINx release contains hex64, prism40 or pyramid30, so the plan's resolution cannot supply three of the five acceptance arrays.

## Options

1. **Keep the plan as written (latest release tags).** Only tet20 and quad16 exist at v0.11.0.post0. The acceptance bullet for hex64, prism40 and pyramid30 could not be met until a DOLFINx release contains them. Not chosen.
2. **Pin DOLFINx at a main commit; Basix and Symfem keep their latest release tags.** DOLFINx main at be8352e677303b686d6862b70923bab086b279ff (HEAD on 2026-10-19), which has all five arrays. Basix v0.11.0 and Symfem v2025.12.0 stay release tags, since M1b uses them only as cited sources. The test's comment and `sources.json` give the full SHA and note that the arrays entered main in c9e0341. This mirrors the ndelement pin (a main commit until a release contains the change, Appendix A, "Version pin").
3. **Mix: tet20 and quad16 from v0.11.0.post0, the other three from main.** Two pins for one file, with no gain over option 2, since the shared arrays are identical. Not chosen.

## Proposed decision

Option 2. The design asks only for pinned commits, and pinning a main commit is the same choice the design already made for ndelement. All five arrays then come from one file at one commit, and the acceptance bullet is testable now. When a DOLFINx release contains the arrays, the pin can move to that tag; the test and `sources.json` change together.

## Consequences

- Text to change in docs/design.md: none. A.10's "pinned commits" covers a main commit. Optionally, A.10 could name the DOLFINx pin in the sources list; not required.
- Plan: docs/plans/M1b.md question 7 records the DOLFINx pin as main at be8352e, Basix v0.11.0 (SHA in `sources.json`) and Symfem v2025.12.0; a Deviations-log row.
- Public API changes: none.
- Acceptance tests affected: `gridcl-refcell::node_maps::gmsh_p3_matches_dolfinx` copies its arrays from DOLFINx at be8352e. `fixtures/sources.json` (T10) records that SHA.
- Later milestones affected: M4 (Gmsh I/O) cites the same pin; a later bump to a DOLFINx release that contains the arrays needs no ADR if the arrays are unchanged.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
