# ADR-0008: VTK Lagrange wedge: triangular faces and triangle layers are lexicographic

- **Status:** Proposed
- **Date:** 2026-10-19
- **Raised in:** M1b T9 (Deviations log), confirmed at the M1b review
- **Design sections affected:** "High-order node orderings" (the VTK bullet and the "Checks" list); A.5 "Inside a triangle (face or cell)"; A.9, last paragraph

## Context

"High-order node orderings" describes VTK's rule as:

> VTK: vertices, edges and faces with VTK's own edge directions, then the interior; triangle interiors are recursive, others lexicographic.

A.5 repeats it ("VTK orders triangle interiors recursively instead; the I/O maps handle this"), and the "Checks" list and A.9 ask for "VTK P3 prisms against VTK-written fixtures".

M1b T9 ported VTK v9.7.1's index functions to generate `vtk_to_internal`
(gridcl-refcell/src/node_maps/vtk.rs). VTK triangle cells (`vtkHigherOrderTriangle::Index`) and tetrahedron faces and interiors (`vtkHigherOrderTetra::Index`) are recursive, as the design says. The wedge is not: `vtkHigherOrderWedge::PointIndexFromIJK` places the interior nodes of its two triangular faces and of every interior triangle layer with `triangleDOFOffset`, which is lexicographic (x fastest, then y).

## Problem

The design's sentence reads as "every triangle interior in VTK is recursive", which includes the wedge's triangular faces. The code follows VTK, not that sentence:

- unit test `node_maps::vtk::tests::wedge_triangle_faces_are_lexicographic`: for p = 5 the bottom face's six interior nodes come row by row, (1,1), (2,1), (3,1), (1,2), (2,2), (1,3), where the recursive rule would start with (1,1), (3,1), (1,3);
- unit test `wedge_index_matches_collocation_points`: `PointIndexFromIJK` agrees with VTK's second description of the wedge, `vtkHigherOrderInterpolation::AppendWedgeCollocationPoints`, for p ≤ 9. Both are ports from the same VTK source, not output written by VTK;
- DOLFINx's `vtk_wedge` agrees for p ≤ 2.

The planned check cannot tell the two orders apart. A triangle face has (p−1)(p−2)/2 interior nodes: 1 at p = 3 and 3 at p = 4. With at most three nodes the recursive order (the corners of a degree p−3 triangle) and the lexicographic order coincide. The first difference appears at p = 5. A VTK P3 prism fixture would therefore pass with either rule, and a later implementer who follows the design's sentence could "correct" the generator without any test failing.

## Options

1. **Keep the design as written.** The text stays ambiguous, and read literally it contradicts VTK. The M5 check at P3 does not catch a recursive wedge. Not chosen.
2. **Clarify the wording and extend the VTK prism check to p = 5 and p = 6.** State which VTK cells are recursive and which are lexicographic, and check VTK prisms against VTK-written fixtures at p = 3, 5 and 6. p = 5 is the first degree at which the two orders differ (six interior nodes per triangle). At p = 6 (ten interior nodes) the recursive rule recurses a second time, into a degree-3 triangle with an interior node of its own. No API change; the code already matches.
3. **Clarify the wording only.** This fixes the text, but the acceptance check still cannot detect the difference. Not chosen.

## Proposed decision

Option 2. The code follows VTK and is already correct. The design should say so, and the VTK-written fixtures should cover the degrees at which the two orders differ: p = 5 at one level of recursion and p = 6 at two. The p = 5 and p = 6 wedges have 126 and 196 nodes, so the fixtures stay small.

## Consequences

- Text to change in docs/design.md:
  - "High-order node orderings", VTK bullet: "triangle interiors are recursive, others lexicographic" → "VTK triangle cells and tetrahedron faces and interiors are recursive; quadrilateral, hexahedron and wedge faces and interiors, including the wedge's triangular faces and triangle layers, are lexicographic".
  - "High-order node orderings", "Checks": "check VTK P3 prisms against VTK-written fixtures, since DOLFINx has no 40-node VTK wedge map" → "check VTK prisms at p = 3, 5 and 6 against VTK-written fixtures, since DOLFINx has no VTK wedge map above p = 2 and the wedge's triangular faces first differ from the recursive order at p = 5 (one level of recursion) and p = 6 (two levels)".
  - A.5, "Inside a triangle (face or cell)": "VTK orders triangle interiors recursively instead" → "VTK orders the interiors of triangle cells and tetrahedron faces recursively instead (its wedge's triangular faces are lexicographic)".
  - A.9, last paragraph: "VTK P3 prisms are checked against VTK-written fixtures" → "VTK prisms of degrees 3, 5 and 6 are checked against VTK-written fixtures".
- Public API changes: none.
- Acceptance tests affected: none in M1b. The M5 (VTK output) fixture corpus gains VTK-written p = 5 and p = 6 wedges.
- Later milestones affected: M5, which must accept this ADR before its plan is approved. gridcl-refcell's `node_maps/vtk.rs` module docs already describe the VTK order this way.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
