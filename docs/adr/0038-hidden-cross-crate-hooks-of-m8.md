# ADR-0038: Hidden cross-crate hooks of M8

- **Status:** Proposed
- **Date:** 2026-11-04
- **Raised in:** M8 review (design review of the review fixes, Minor 3; docs/plans/M8.md, question 19)
- **Design sections affected:** "Workspace layout" / "Dependency rules"; Listing H1 (the `footprint_bytes` comment)

## Context

CLAUDE.md: "Public types in 'Architecture' and 'API walkthrough' change only via an ADR". ADR-0028 item 3 made `IndexMap::{send_lists, recv_lists}` hidden test accessors and noted that "hidden items are still public items outside the listings"; ADR-0015 hides the `gridcl_topology::bench` module. ADR-0035 lists `Mesh::footprint_bytes` (Listing H1) and, in item 6, the gridcl-io items of chunked reading, but not hidden items in the lower crates.

M8 needed gridcl to call code in lower crates that is not part of their stable API:

- **The memory bound (bullet 2, ADR-0036).** gridcl-device `radix_sort_owned` (sorts through its input: two sets of keys on the device, not three) and `unique_by_key_runs` (groups and run offsets from one numbering); gridcl-topology `Topology::create_entities_releasing_pools` (entity creation that gives the empty pool pages back, C13), which mesh creation's stage 10 calls.
- **The measurement of that bound.** `footprint_bytes` on `DeviceBuffer` (gridcl-device), `IndexMap` (gridcl-parallel), `Topology` (gridcl-topology), `Geometry` (gridcl-geometry) and `Mesh` (gridcl).
- **The parallel reader.** gridcl-io `gmsh::{BlockHeader, Layout, CellChunk, complete, node_range, element_range, geometric_dim, FileNumbering, from_numbered_gmsh_mesh}` and `MshIndex::{block_headers, sections, bytes_read}`: the conversion steps both read paths share and the numbered writer.

## Problem

Rust has no visibility between `pub(crate)` and `pub`, so a function one workspace crate calls in another must be `pub`. The items above are `#[doc(hidden)] pub` and documented "not part of the stable API". Three of them (`radix_sort_owned`, `unique_by_key_runs`, `create_entities_releasing_pools`) are production code paths of mesh creation, not test accessors, so neither ADR-0028 (test accessors) nor ADR-0035 item 6 (gridcl-io items of chunked reading) covers them. No ADR says what a hidden item may be, what it must carry, or where it is recorded.

## Options

1. **Keep the design as written.** Either make the items stable public API and add them to the listings, which commits gridcl to internal primitives and an entity-creation variant that exist only for one memory measurement; or remove them, which loses the M8 memory bound (T5b measured 671–712 MiB without them under glibc's defaults and 470–485 MiB with them, against a bound of 525 MiB; ADR-0036). Not proposed.
2. **Hidden cross-crate hooks as a recorded category (proposed).** A workspace crate may expose a `#[doc(hidden)] pub` item for another workspace crate when no stable API serves the purpose. Each such item says "not part of the stable API" in its rustdoc, has a rustdoc example or a named test, and is listed in its milestone plan's Deviations log. Hidden items are semver-exempt: they may change or disappear in any release. The listings stay as they are.
3. **Public options instead of hooks.** For example a public `Topology` setting that turns the pool cleanups on, or public consuming variants of the primitives. This widens the stable API for an internal memory concern and still leaves the measurement accessors and the gridcl-io steps hidden. Not proposed.

## Proposed decision

Option 2. The hooks keep the stable API unchanged while letting gridcl's mesh creation meet the memory bound and the tests measure it, and the rules (documented as unstable, an example or test, listed in the plan) keep them reviewable. M8's items are those listed in the Context, recorded in docs/plans/M8.md's Deviations log (2026-11-04, hidden `pub` items).

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Workspace layout" / "Dependency rules": add rule 7: "A workspace crate may expose a `#[doc(hidden)] pub` item to other workspace crates when no stable API serves the purpose (hidden hooks and test accessors). Such an item is documented as not part of the stable API, has a rustdoc example or a named test, is listed in its milestone plan, and is semver-exempt (ADR-0015, ADR-0028, ADR-0038)."
  - No listing changes: Listing H1 already names `Mesh::footprint_bytes`.
- Public API changes: none to the stable API; the hidden items above stay as implemented.
- Acceptance tests affected: none.
- Later milestones affected: each plan lists the hidden items it adds, under the new rule 7.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
