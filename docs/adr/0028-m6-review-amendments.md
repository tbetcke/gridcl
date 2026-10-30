# ADR-0028: M6 review amendments

- **Status:** Proposed
- **Date:** 2026-10-30
- **Raised in:** M6 review (design review of T0, T5, T7, T10 and T14)
- **Design sections affected:** "Distributed mesh model" / "Communicators, IndexMap and Scatterer" (the paragraph after Listing S2c); "Workspace layout" (crate table: gridcl, gridcl-testing); "Parallel creation pipeline", stage 6; Listing H1 (CLAUDE.md, host↔device copies)

## Context

The M6 design review found five places where the implementation follows the plan's Deviations log but differs from, or adds to, the text of docs/design.md. None changes a public signature of the design's listings; each needs the design text to say what the code does.

1. **Scatterer precision check (T7).** "Communicators, IndexMap and Scatterer" says: "`ScatterNumeric` is implemented for f32, f64, i32 and u32. Device support for the chosen type is checked when the scatterer is created." The code checks in `Scatterer::reverse` (`UnsupportedPrecision`, gridcl-parallel/src/scatterer.rs:209-215); `forward` does not check.
2. **Optional `mpi` dependencies (T0).** The crate table lists `mpi (optional)` for gridcl-parallel only. gridcl and gridcl-testing gained an optional `mpi` dependency (enabled by their feature `mpi`) because their test communicators (`gridcl/src/test_comm.rs`, `ThreadComm`) implement `CommDyn::raw`, whose return type is `Option<mpi::ffi::MPI_Comm>` (Listing S2c).
3. **Hidden test accessors (T5).** `IndexMap::send_lists` and `IndexMap::recv_lists` are `#[doc(hidden)] pub`, read-only, for the exchange-pattern tests in gridcl-testing and gridcl/tests. They are not in Listing S2c.
4. **Unknown node id in `build_on` (T10).** Stage 6 reports a cell node id that no rank posted as `GridError::IndexOutOfRange { index: id (usize::MAX if negative), len: global node count }`. ADR-0026's input contract names errors for duplicates and role mismatches, not for an unknown id.
5. **Host↔device copies (T7, T14).** Listing H1 allows "the upload of an IndexMap's send and receive index lists when its scatterer is first created" and "the boundary-entity ownership exchange of entity creation (the candidate count and keys to the host; owners and global numbers back)". The code also:
   - uploads each scatterer's reverse lists (`rev_order`, `rev_segments`, `rev_first`, `rev_targets`, which depend on the block size) whenever `IndexMap::scatterer` is called, as the design's "The reverse segments are built once, when the scatterer is created" requires;
   - in the ownership exchange, reads the local indices of the candidates a rank owns to the host and uploads the ghosts' provisional ids and final local indices for one scatter into the renumbering (gridcl-topology/src/renumber.rs, `local_indices`, `place_ghosts`).

## Problem

1. `IndexMap::scatterer(&self, client, block_size)` is not generic over the value type (Listing S2c), so the scatterer cannot know at creation which type it will carry. `forward` moves u32 words, so it works for any `ScatterNumeric` (f64 included) on every device; only `reverse` computes in T. The sentence cannot be followed with the listed signature.
2. A crate that implements `CommDyn` under gridcl-parallel's `mpi` feature must name `mpi::ffi::MPI_Comm`, so it needs the `mpi` crate; the crate table does not show this for gridcl and gridcl-testing.
3. The tests must compare the host send and receive lists with an independent reference from other crates; the lists are otherwise crate-private. Hidden items are still public items outside the listings.
4. Without a named error, an unknown node id would have to be a panic or `NotYetImplemented`; ADR-0026 leaves it open.
5. The copies are part of the exchanges the rule already names, but its wording is narrower than the code, and the copy list is a CLAUDE.md rule that only the maintainer changes.

## Options

1. **Keep the design as written.**
   - (1) would need `IndexMap::scatterer::<T>` (a signature change to Listing S2c), or a check of every `ScatterNumeric` type at creation, which would reject f64 scatterers on Metal although `forward` works there.
   - (2) would need the test communicators to drop `CommDyn::raw` support, so gridcl-mpi-tests could not build with gridcl-testing.
   - (3) would need the tests to rebuild the lists from `ghosts()` and `ghost_owners()` alone, which checks less (send order).
   - (4) and (5) would leave the code outside the written rules.
2. **Amend the text to match the code (proposed).** Text changes only; no signature changes.
3. **Re-export the raw handle type** (`gridcl_parallel::RawComm = mpi::ffi::MPI_Comm`, feature `mpi`) instead of item 2's crate-table change, so gridcl and gridcl-testing need no direct `mpi` dependency. This adds a public item to gridcl-parallel; third-party `CommDyn` implementors would also benefit. It can follow later without breaking anything, so this ADR does not choose it.

## Proposed decision

Option 2:
1. Precision is checked by `Scatterer::reverse` (`UnsupportedPrecision`); `forward` moves u32 words and needs no check.
2. The crate table lists `mpi (optional, feature mpi)` for gridcl and gridcl-testing.
3. `IndexMap::{send_lists, recv_lists}` are documented as `#[doc(hidden)]` test accessors outside the stable API.
4. Stage 6 reports an unknown node id as `IndexOutOfRange { index: id, len: global node count }` (`usize::MAX` for a negative id), agreed like the other stage errors.
5. Listing H1 and CLAUDE.md name the extra copies explicitly.

## Consequences

- Text to change in docs/design.md (sections, listings, tables):
  - "Communicators, IndexMap and Scatterer": "Device support for the chosen type is checked when the scatterer is created." → "`reverse` checks device support for T (`UnsupportedPrecision`); `forward` moves u32 words and works for every `ScatterNumeric` type on every device." Add after Listing S2c: "`IndexMap::send_lists` and `recv_lists` are hidden, read-only test accessors and not part of the stable API."
  - "Workspace layout", crate table: gridcl's dependencies gain "mpi (optional)"; gridcl-testing's gain "mpi (optional)", with the note "for `CommDyn::raw` of the test communicators".
  - "Parallel creation pipeline", stage 6: "a cell node id that no rank posted is `IndexOutOfRange { index: id, len: global node count }`".
  - Listing H1 and CLAUDE.md: "the upload of an IndexMap's send and receive index lists when its scatterer is first created" → "the upload of an IndexMap's send and receive index lists when its first scatterer is created, and of a scatterer's reverse lists when it is created"; "(the candidate count and keys to the host; owners and global numbers back)" → "(the candidate count, keys and the local indices of owned candidates to the host; owners, global numbers and ghost placements back)".
- Public API changes: none (the items exist; the text describes them).
- Acceptance tests affected: none.
- Later milestones affected: M7 (`raw` for PT-Scotch and ParMETIS: the same `mpi` dependency rule applies to any crate implementing `CommDyn`); M10 (a GPU-aware transport keeps the reverse-list upload at scatterer creation).

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
