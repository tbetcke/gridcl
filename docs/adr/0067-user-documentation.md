# ADR-0067: User documentation: a book on GitHub Pages and API docs written for users

- **Status:** Proposed
- **Date:** 2026-11-20
- **Raised in:** D1 planning (maintainer request, 2026-11-20)
- **Design sections affected:** a new section "User documentation" (after "Import and export");
  "Workspace layout" (table rows tetraloma-parallel, tetraloma-partition and tetraloma-examples);
  "Cargo features" (runtime features: tetraloma-examples gains `hdf5`); "CI matrix" (a new row `docs`, a
  new row `pages`, and the rows `mpi-mpich, mpi-openmpi`: the example programs on 1–4 ranks);
  "Implementation roadmap" (dependency table and a new entry D1); Listing H1 (the working agreement).
  "Development environment and test targets" is not edited: rule 9 is an exception for D1's review only.
  Process documents: CLAUDE.md, IMPLEMENTATION.md, docs/dev-environment.md, the `.claude/` skills
  `implement-milestone` and `review-milestone` and the `design-reviewer` agent.

## Context

The design asks for documented public items only in the sense of rustdoc completeness: M10 bullet 7
("Rustdoc of every workspace crate builds without warnings, and every public item is documented") and the
workspace lint `missing_docs = "deny"` (ADR-0060, item 7). It has no user documentation: no overview for
someone who has not read docs/design.md, no statement of the mathematical conventions outside Appendix A
and the "Geometry" section, no tutorials beyond the five programs of tetraloma-examples, and no
description of how to choose a device.

The rendered rustdoc is written for the implementers. It cites developer documents about 430 times
(`docs/design.md, "Geometry"`, `ADR-0044`, `docs/plans/M3.md, question 15`). Its doctests obtain a
client and test meshes from tetraloma-testing, a crate users do not have, and the doctests of
tetraloma-parallel and tetraloma-partition define functions that take a client but never call them,
because those crates have no device in their dev-dependencies.

## Problem

1. A user who finds tetraloma on crates.io or GitHub has no way in except docs/design.md, a 2,600-line
   design record full of milestone history.
2. The citations point at files that are not part of a published crate; they are noise to users and dead
   links on docs.rs.
3. Items defined in the sub-crates (tetraloma-geometry, tetraloma-parallel, …) cannot link with
   intra-doc links to anything in the facade, because the dependency points the other way. A guide
   inside the facade's rustdoc would be reachable from only part of the API.
4. A guide inside rustdoc that is kept in `docs/user/` breaks `cargo package`: `include_str!` of files
   outside the package directory is not packaged.
5. docs.rs cannot build the features `mpi`, `hdf5`, `scotch`, `ptscotch`, `metis`, `parmetis` (no system
   libraries), so the API of distributed meshes, parallel I/O and the partitioners is missing there.

## Options

1. **Keep the design as written.** Rustdoc completeness only. Problems 1–5 stay.
2. **A guide inside rustdoc** (`tetraloma::guide` doc-only modules over `include_str!` of
   `docs/user/*.md`, KaTeX through `--html-in-header`). Works offline and on docs.rs, but adds a public
   module, needs the markdown inside the package or a copy step (problem 4), reaches only facade items with
   intra-doc links (problem 3), and gives a book-length guide only rustdoc's module navigation.
3. **A book (mdBook) on GitHub Pages, published together with the API docs of the library features, and
   API docs rewritten for users.** No public API change. Every crate links to the book the same way. The
   book has chapter navigation, search and pre-rendered maths. docs.rs still shows the default-feature API
   and links to the site.

## Proposed decision

Option 3, with these rules.

1. **Location.** The book's sources are in `docs/user/` (mdBook root: `book.toml`, `src/SUMMARY.md`,
   `src/**/*.md`, `src/figures/*.svg`). The developer documents stay where they are (`docs/design.md`,
   `docs/adr/`, `docs/plans/`, `docs/perf/`, …); `docs/README.md` says which is which.
2. **Site.** One site at `https://tetracl.github.io/tetraloma/`: the book at `/`, and at `/api/` the
   rustdoc of the nine published library crates (tetraloma, tetraloma-dims, tetraloma-refcell,
   tetraloma-device, tetraloma-parallel, tetraloma-topology, tetraloma-geometry, tetraloma-partition,
   tetraloma-io), built with `cargo doc --no-deps` and one `-p` per crate, with the features
   `tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis`, into an empty
   output directory. No unpublished crate and no test-support feature (`testing`, `bench`) is built into
   the site. `ParMetis` is not on the site (`metis` and `parmetis` are exclusive, ADR-0033); the book's
   features chapter describes it, and the audit of rule 7 checks it in a second run with `parmetis`.
   A CI job builds the site on every push and deploys it to GitHub Pages from `main`. Per-release copies
   under `/vX.Y/` start with the first crates.io release (a later decision); until then the site documents
   `main`.
3. **Tools.** mdBook and the `mdbook-katex` preprocessor at a pinned, tested pair of versions (same
   mdBook major version), installed with `cargo install --locked` by `scripts/install-doc-tools.sh`. Maths
   in the book is LaTeX between `$…$` and `$$…$$`, rendered at build time; the KaTeX stylesheet and fonts
   are served from the site, not from a CDN. `book.toml` sets the Rust edition 2024 and turns the
   playground's run buttons off. Figures are SVG files; the figures of reference cells and node orderings
   are generated from tetraloma-refcell's tables by a program in tetraloma-examples, and a test checks
   that the committed files equal its output.
4. **Contents.** The book has these parts, written for a graduate student who knows linear algebra and
   calculus but not necessarily finite elements, in British English:
   overview and installation; mathematical concepts (meshes and topology, reference cells and maps,
   Lagrange geometry, Jacobians, pseudo-inverses, measures, normals and orientation, quadrature,
   pull-back, precision); element definitions (reference cells, sub-entity numbering, Lagrange nodes,
   orientation bits, Gmsh and VTK maps, conversion to Basix and DOLFINx); tutorials (building a mesh,
   reading and writing files, iterating over cells and entities, geometric data and Jacobians,
   quadrature and high-order geometry, the pull-back, tags and submeshes and manifolds, distributed meshes
   and partitioners, halo exchange, parallel I/O, writing downstream kernels); compute devices (CPU
   runtime, Metal, CUDA, Vulkan, HIP; choosing a device; precision; one GPU per MPI rank; performance
   settings); Cargo features and system libraries; troubleshooting; a glossary.
5. **Code in the book.** Every fenced block has a language tag, and the book has no indented code blocks
   (rustdoc would compile both as Rust). Every `rust` block either runs as a doctest on the CPU runtime
   (a `#[cfg(doctest)]` module of tetraloma-examples includes each chapter with `include_str!`), or is
   marked `ignore` and consists only of an mdBook `{{#include}}` of a program in tetraloma-examples, which
   `scripts/run-examples.sh` runs on the CPU runtime, on Metal and on CUDA (MPI programs under `mpirun` on
   1–4 ranks). Snippets obtain devices the way a user does, through `cubecl::Device` (for example
   `Device::cpu()?.client()`), never through tetraloma-testing. Links from the book into `/api/` target
   item pages, not method anchors.
6. **API documentation of the nine published library crates.** In rendered docs (`///`, `//!`):
    1. no citation of developer documents: no `docs/` path, ADR number, plan question or milestone id.
       The provenance moves to a plain `//` comment directly above the doc comment
       (`// Design: "Geometry", "Geometric factors"; ADR-0044.`);
    2. these items link to at least one section of the book with an absolute URL under the site's base
       URL (a reference-style link is fine): each crate's root, every public module, struct, enum, union,
       trait, type alias, constant, static, function and macro, every inherent method and associated
       function, and every method of a trait defined in these crates. Fields, variants and associated types
       and constants need documentation (as today) but no link;
    3. every function and method of rule 6.2 has an `# Example` doctest that runs: not `ignore` and not
       `no_run`, and listed by `cargo test --doc -- --list`. An item is exempt only by an entry in
       `docs/user/doc-audit-allow.toml` with a reason (for example an accessor that its type's example
       calls). Examples of MPI-only items run as one-process programs in the `docs` CI job. API doctests may
       keep obtaining their client and test meshes from tetraloma-testing on hidden (`# `) lines, so they
       run on every backend of the standing matrix; a visible comment then names what the hidden line makes
       and, where a reader needs it, links to the tutorial that shows how to make it. To make this
       possible, tetraloma-parallel and tetraloma-partition gain the dev-dependency tetraloma-testing
       (as tetraloma-topology and tetraloma-geometry have);
    4. maths is written in Unicode, with the book's symbols, and links to the book for derivations.
   `#[doc(hidden)]` items and the test- and benchmark-support features keep their present rules
   (dependency rule 7). The unpublished crates and the two `-sys` crates get a crate page that says what
   they are and links to the book; their items are not audited.
7. **Checks.** `scripts/check-user-docs.sh` (run by the CI `docs` job, and locally) builds the site and
   fails on: a warning of mdBook (its log is searched for `WARN`) or of rustdoc; a broken link or anchor
   inside the site, including links into `/api/` and absolute links to the site's base URL, which it maps
   to the local build; a developer citation in a rendered doc block of the audited crates; an item of rule
   6.2 without a book link; a function of rule 6.3 without a running example and without an allowlist
   entry; a book chapter missing from the outline of rule 4, or one left as a stub; a fenced block without
   a language tag, an indented code block, or an `ignore` block that is not an include of a program that
   `scripts/run-examples.sh` runs; and, while D1 is open, a difference from D1's base commit in the
   sources (`src/**/*.rs`) of the nine library crates other than in comments (a stdlib tokenizer strips
   `//`, `///`, `//!` and block comments, respecting strings, raw strings, characters and lifetimes, and
   compares the remaining tokens file by file). The audit reads the generated HTML, so it sees exactly the items users
   see, and fails if it finds no items (a changed rustdoc markup). The `docs` job also runs the book's
   doctests and `cargo test --doc` of the nine crates with the site's features, and the audit once more for
   tetraloma-partition with `parmetis`.
8. **Keeping it current.** From D1 on, a milestone that adds or changes public items documents them by
   rule 6 and updates the book in the same task; `/review-milestone` runs `scripts/check-user-docs.sh`.
9. **D1's review runs (an exception to ADR-0014 and ADR-0016 for D1 only).** When the comments-only check
   of rule 7 passes, the compiled library code equals D1's base commit, which passed the standing matrix.
   D1's review then runs what D1 reaches, not the whole matrix: the D1 checks; the doctests of the
   workspace on the CPU runtime and Metal; the tests of tetraloma-examples on the CPU runtime and Metal; the
   tests of tetraloma-parallel and tetraloma-partition (changed dev-dependencies) on the CPU runtime; the
   MPI programs on 1–4 ranks (Open MPI locally; MPICH and Open MPI in hosted CI); and on Locust only the
   programs `devices` and `custom_kernel`. The hosted CI jobs (`cpu` with the full workspace suite,
   `mpi-mpich` and `mpi-openmpi` with `scripts/mpi-test.sh 1 2 3 4`, `vulkan`, `docs`) run on every push as
   usual. Locust's standing CUDA suite, the vector-size runs and Kathleen's two-node MPI run are not
   repeated. If D1 must change library code after all (through its own ADR), that change gets the runs it
   reaches, by ADR-0057.

## Consequences

- Text to change in docs/design.md: a new section "User documentation" stating rules 1–8;
  "Workspace layout", table rows: tetraloma-parallel and tetraloma-partition gain "dev: tetraloma-testing";
  tetraloma-examples gains the book's programs, the doctest harness of the book, the figure generator and
  the dependency tetraloma-refcell; "Cargo features": tetraloma-examples' features gain `hdf5`
  (forwarding `tetraloma/hdf5`); "CI matrix": a row `docs` (hosted Linux, the image of the `ptscotch` job
  with MPI, parallel HDF5 and Scotch; rule 7; every push), a row `pages` (builds and deploys the site;
  pushes to `main`), and the rows `mpi-mpich, mpi-openmpi` run the example programs on 1–4 ranks;
  "Implementation roadmap": table row "D1 User documentation | E1" and a D1 entry with the scope of rules
  1–5 and this acceptance: `scripts/check-user-docs.sh` passes in CI; the book's doctests pass on the CPU
  runtime; the book's programs pass on the CPU runtime, Metal and CUDA in f32 and, where supported, f64, and
  the MPI programs on 1–4 ranks under MPICH and Open MPI; the library sources differ from the base commit
  only in comments; the site is deployed from `main`; the hosted CI jobs pass. Listing H1: rules 6 and 8. M10 bullet 7 stays and is strengthened by rule 6.
- Public API changes: none. Doc comments only; no item, signature or behaviour changes. Dev-dependencies of
  tetraloma-parallel and tetraloma-partition change; tetraloma-examples (unpublished) gains programs, a
  feature, a dependency, the doctest harness and the figure generator.
- Acceptance tests affected: none of earlier milestones. New: D1's.
- Development environment and test targets: rule 9 is a one-off exception for D1's review; ADR-0014 and
  ADR-0016 are unchanged for every other milestone.
- Later milestones affected: every milestone after D1 follows rules 6 and 8. The first crates.io release
  adds versioned copies of the site (rule 2).
- Repository settings: GitHub Pages enabled for tetracl/tetraloma (public) with source "GitHub Actions"
  (maintainer action).

## Decision record

<Filled by the maintainer.>
