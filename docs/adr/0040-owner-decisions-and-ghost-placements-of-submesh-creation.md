# ADR-0040: Owner decisions and ghost placements of submesh creation on the copy list

- **Status:** Accepted
- **Date:** 2026-11-05
- **Raised in:** M9 T5a (found while planning it after T4b)
- **Design sections affected:** "Handing this design to Claude Code", the working-agreement
  copy list (also CLAUDE.md); ADR-0039 item 18

## Context

The copy list allows, for entity creation, "the boundary-entity ownership exchange of entity
creation (the candidate count, keys and the local indices of owned candidates to the host;
owners, global numbers and ghost placements back)". For submesh creation, ADR-0039 item 18
added:

> the per-block count reads of compaction, the agreed TooLarge, NonManifold and InteriorFacet
> flags, the child ghost lists for IndexMap::with_ghosts, the child sharer rows of shared
> vertices and the node-owner candidates of shared nodes, the upload of the closure node maps
> and the child's vertex shared flags

ADR-0039 item 6 orders child vertices and nodes [owned | ghost]: "owned in parent local order,
ghosts by (owner rank, child global index)". Vertex owners are chosen on the host from the
child sharer rows (`owner_hash(&[origin id]) mod n` among the owned-child-cell sharers), and
node owners from the node-owner candidates, which the parent node owner collects.

## Problem

The child topology lives on the device. Its vertex rows (and the child dofmaps) index the
child's local vertices (nodes), and the device builds the parent → child map by a compaction
(`gridcl/src/submesh/topology.rs`, `compact`). For an unshared vertex the device knows that
this rank owns it. For a shared vertex, which rank owns it, and so whether it goes to the owned
part or to a ghost slot (and which slot), is decided on the host. The device cannot place
shared vertices without that decision, and item 18 lists no host → device transfer for it.
Nodes (T5b) have the same gap. Entity creation, which places shared entities in the same way,
has this transfer on the list ("owners, global numbers and ghost placements back").

T4a and T4b also read two things that item 18 does not name word for word (Deviations log of
docs/plans/M9.md):

- the presence flags of the element check (ADR-0039 item 8: which parent cell blocks hold a
  child's parent cell), read in the same transfer as the agreed `TooLarge` flag;
- the size of each physical group of the child's tags (item 10), one count read per group,
  read as "per-block count reads of compaction".

## Options

1. **Keep the design as written.** The device would have to choose the owners of shared child
   vertices and nodes itself, through scatters (the host-staged halo exchange is on the list):
   for example, each holder in an owned child cell offers a score over the parent's vertex
   `IndexMap`, `reverse(Max)` picks the largest, and `forward` returns it. This replaces
   item 6's `owner_hash(&[origin id]) mod n` rule with a different one, an amendment to
   ADR-0039 item 6 instead of item 18. It costs two scatters and a kernel per child, and the
   ghost order by (owner rank, child global index) would still need a device sort.
2. **Add the transfer back to the copy list (proposed).** In submesh creation, the owners,
   child global numbers and ghost placements of shared child vertices and of non-corner shared
   nodes go back to the device, as for entity creation. Name the two reads of T4a and T4b too.
   Only shared vertices and nodes travel, as for the sharer rows that go to the host.

## Proposed decision

Option 2: it mirrors the entity-creation entry, keeps ADR-0039 item 6 unchanged, and limits
the transfer to shared vertices and nodes, which the host already handles. The two reads of
T4a and T4b are named so that the list says what the code does.

## Consequences

- Text to change in docs/design.md (sections, listings, tables): "Handing this design to
  Claude Code", the copy list (and CLAUDE.md, its copy). In the submesh-creation part, after
  "the child sharer rows of shared vertices and the node-owner candidates of shared nodes",
  insert "to the host, their owners, child global numbers and ghost placements back"; after
  "the agreed TooLarge, NonManifold and InteriorFacet flags", insert "(with the parent blocks
  that hold a child's parent cell, read in the same transfer)"; after "the per-block count
  reads of compaction", insert "(and of each physical group of the child's tags)". The
  ADR list in that paragraph gains ADR-0040.
- Public API changes: none.
- Acceptance tests affected: none.
- Later milestones affected: none. M9 T5a and T5b use the new entry; the Deviations-log rows
  of T4a and T4b that explain the two reads point to this ADR once it is accepted.

## Decision record

Accepted by the maintainer on 2026-11-05 (option 2), without conditions. Applied to the copy
list in docs/design.md ("Handing this design to Claude Code") and CLAUDE.md.
