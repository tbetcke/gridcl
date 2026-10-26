# ADR-0023: The mshio oracle cannot read 18-node prisms

- **Status:** Accepted
- **Date:** 2026-10-26
- **Raised in:** M4 T6 (docs/plans/M4.md, question 21)
- **Design sections affected:** "Implementation roadmap / M4 Gmsh I/O (serial)" (the third acceptance bullet); "Import and export / Gmsh MSH 4.1" (the "Own reader and writer" bullet)

## Context

M4's third acceptance bullet reads:

> - nodes and elements agree with mshio on every fixture;

"Import and export / Gmsh MSH 4.1" makes mshio "a test oracle, not the implementation", and
ADR-0022 pins it as a dev-dependency of gridcl-io, `mshio = "=0.4.2"` (the newest release on
crates.io). The plan for T6 says: "if mshio rejects a corpus type, record it and raise an ADR row
rather than dropping the fixture."

The committed corpus (T1) has 61 files written by Gmsh 4.15.2. Two of them contain Gmsh type 13,
the 18-node prism: `prism_p2.msh` and `prism_p2-binary.msh`.

## Problem

mshio 0.4.2 gives the 18-node prism 28 nodes. `src/mshfile.rs`, `ElementType::nodes`:

```rust
ElementType::Pri18 => 28,
```

Its element parser reads that many node tags per element, so it fails on both files:

```
prism_p2.msh: mshio: MshParserError([([36, 69, 110, 100, 69, 108, 101, 109, 101, 110, 116, 115, 10]
  /* "$EndElements\n" */, NomError(Digit)), ...
```

`nodes_and_elements_agree_with_mshio` therefore fails on these two files and on no other. The
other 59 files agree with mshio: node blocks (entity, coordinates, tags) and element blocks
(entity, type, tags, node tags), and so do the other prism types (6, 90, 91 and the serendipity
type 18). gridcl's own parser reads both files with Gmsh's node and element counts
(`corpus_reads_with_gmsh_node_counts` passes), and its binary parse of `prism_p2-binary.msh`
re-encodes to Gmsh's bytes exactly (`binary_byte_orders_and_size_t_widths`).

## Options

1. **Keep the design as written.** The bullet cannot pass with the pinned oracle, unless the
   type-13 files leave the corpus, which the plan forbids: type 13 is in A.7 and in the
   corpus-coverage bullet ("every element type up to order 4").
2. **Exempt exactly the files mshio cannot read.** The test compares every fixture with mshio,
   except the files that contain type 13. For those it asserts that mshio fails on them, so the
   exemption is a checked known defect, not a skipped file, and the test fails (and the exemption
   must go) once a fixed mshio is pinned. Type 13 stays covered by Gmsh's own counts
   (`corpus_reads_with_gmsh_node_counts`), the node-map check against the affine boxes (T8), the
   read → write → read round trip (T13), and the byte-exact binary re-encoding. No code or
   dependency changes.
3. **Patch mshio.** A vendored copy of mshio 0.4.2 with the one table entry fixed, wired in with
   `[patch.crates-io]` (a patch section applies to the whole workspace). The bullet then holds
   literally, but the oracle is no longer the published crate, the repository carries a fork of
   a third-party crate, and the patch must be maintained against later pins.
4. **A second oracle for type 13.** For example meshio through Python, as M5 does for VTK. It
   adds a Python dependency to the M4 tests and a second oracle with its own conventions for one
   element type.

## Proposed decision

Option 2. The oracle's defect is a single wrong table entry, reported by the test. Exempting the
two files keeps the corpus complete and the oracle unmodified. The exemption is narrow: by file,
because mshio fails on the whole file, and checked, because it asserts the failure. Type 13 keeps
three independent checks: Gmsh's own counts, Gmsh's own bytes, and the affine node positions.
Options 3 and 4 add a fork or a second toolchain to fix a gap that these checks already close.

## Consequences

- Text to change in docs/design.md:
    - Roadmap M4, the third acceptance bullet reads: "nodes and elements agree with mshio on
      every fixture mshio can read; mshio 0.4.2 cannot read the 18-node prism (type 13), and the
      test asserts that it fails on exactly the files that contain it (ADR-0023);".
    - "Gmsh MSH 4.1", "Own reader and writer": append "mshio 0.4.2 gives the 18-node prism 28
      nodes, so it cannot read files that contain one (ADR-0023)."
- Public API changes: none.
- Acceptance tests affected: `gridcl_io::gmsh_corpus::nodes_and_elements_agree_with_mshio`
  compares the files without type 13 with mshio, and asserts that mshio fails on each file with
  type 13. No other test changes.
- Later milestones affected: none. If a later mshio release fixes the entry, the test fails until
  the exemption is removed.

## Decision record

Accepted by the maintainer on 2026-10-26, as proposed (option 2), without conditions. Applied to docs/design.md in `adr: 0023 apply`.
