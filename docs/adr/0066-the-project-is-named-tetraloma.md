# ADR-0066: The project is named tetraloma

- **Status:** Accepted
- **Date:** 2026-11-19
- **Raised in:** E1 planning (maintainer request, 2026-11-19)
- **Design sections affected:** the title and "Summary" (**Name:**); "Workspace layout" (crate table, directory
  tree, dependency rules); "Testing and validation" and "CI matrix" (environment variables, job names);
  "Development environment and test targets"; "Cargo features"; Listing H1 (the working agreement);
  "Implementation roadmap" (crate names in milestone headings; a new entry E1); Appendix A.10; and every other
  section that names gridcl, a gridcl crate, a `gridcl_*` path or a `GRIDCL_*` variable (about 250 occurrences
  in about 60 sections, all mechanical). Process documents: CLAUDE.md, IMPLEMENTATION.md, docs/dev-environment.md,
  the `.claude/` skills, agents and settings.

## Context

"Summary": "**Name:** gridcl is a placeholder." The name runs through the whole project: 13 workspace crates
(`gridcl`, `gridcl-dims`, … `gridcl-examples`) and `gridcl-conformance`, their Rust paths, about 50 `GRIDCL_*`
environment variables, the remote-test runner (`.gridcl-hpc/`, `gridcl_*` shell functions), CI job names,
the licence copyright line, and about 7,600 occurrences in 484 tracked files. The repository is
github.com/tbetcke/gridcl.

The maintainer has chosen the permanent name **tetraloma** and a new home, the GitHub organisation `tetracl`.
The crate name `tetraloma` is reserved on crates.io (0.0.0, 2026-11-19, repository link
github.com/tetracl/tetraloma). No type, trait or function name contains "gridcl"; only crate names, paths,
variables and prose do.

## Problem

The placeholder cannot stay once the crates are published or the repository moves: crate names on crates.io
are permanent, and a repository under `tetracl` named after a placeholder misleads users. Renaming touches
crate boundaries, a convention (environment variable names) and every listing that names a crate, so it
needs an ADR.

## Options

1. **Keep the design as written.** Keep gridcl until a later release. Every milestone adds more occurrences,
   and the first crates.io release would have to rename anyway.
2. **Rename only the published surface.** Publish `tetraloma` as a facade over the unchanged `gridcl-*`
   crates and keep `GRIDCL_*`. Two names would live side by side in code, docs and error messages.
3. **Rename everything, mechanically, in one milestone (E1).** One name throughout, done while there are no
   outside users and no open PRs.

## Proposed decision

Option 3, with this mapping and these rules:

| Old | New |
| --- | --- |
| crate and directory `gridcl` | `tetraloma` |
| crates and directories `gridcl-<x>` (dims, refcell, device, parallel, topology, geometry, partition, io, testing, mpi-tests, bench, examples) | `tetraloma-<x>` |
| `gridcl-conformance` (directory `conformance/`) | `tetraloma-conformance` (directory unchanged) |
| Rust paths `gridcl_<x>` | `tetraloma_<x>` |
| environment variables `GRIDCL_*` | `TETRALOMA_*` |
| prose "gridcl", "GridCL" | "tetraloma", "Tetraloma"; copyright "The Tetraloma Developers" |
| runner internals `.gridcl-hpc/`, `gridcl-cache`, `gridcl_*` shell functions | `.tetraloma-hpc/`, `tetraloma-cache`, `tetraloma_*` |
| repository github.com/tbetcke/gridcl | github.com/tetracl/tetraloma (transfer and rename; GitHub redirects the old URLs) |

1. **Unchanged names:** `ptscotch-sys`, `parmetis-sys`, `tools/env-check`, and the spack repository
   namespace `gridcl` in `scripts/hpc/spack/` (renaming it re-concretises and rebuilds LLVM on Myriad; it is
   renamed at the next LLVM bump, when that rebuild happens anyway).
2. **Environment variables switch at once.** No `GRIDCL_*` variable is read after E1; there is no
   transition period (no outside users).
3. **Historical records are renamed too.** Accepted ADRs, finished plans, docs/perf and
   docs/cubecl-upstream.md are rewritten mechanically, so the paths, commands and links in them keep working;
   their meaning does not change. Recorded data stays verbatim: `docs/perf/data/` and the bench crate's
   `history/*.toml`.
4. **Manifest:** `[workspace.package]` gains `repository = "https://github.com/tetracl/tetraloma"`.
5. **Guard:** `scripts/check-name.sh`, run in CI, fails on any `gridcl` in tracked paths or content outside
   the allowlist of rules 1 and 3, on a workspace package not named `tetraloma` or `tetraloma-*` (the two `-sys`
   crates excepted), on a missing `repository`, and on any read of a `GRIDCL_` variable.
6. **Private configuration** (`scripts/hpc/targets.env`, remote base directories) is not covered: the
   remote base directories keep their names for now.

## Consequences

- Text to change in docs/design.md: the sections listed above, by the mapping; the **Name:** line states the
  name; "Implementation roadmap" gains a short E1 entry (scope: this rename; acceptance: rule 5 and an
  unchanged standing matrix).
- Public API changes: crate and module paths only (`gridcl::Mesh` becomes `tetraloma::Mesh`, and so on);
  no item, signature or behaviour changes. Cargo features keep their names; feature paths change with the
  crate names (`gridcl-testing/metal` becomes `tetraloma-testing/metal`).
- Acceptance tests affected: none change meaning; test names containing the name are renamed
  (`vtk_evaluates_curved_cells_like_gridcl`, `tetrahedron_basix_to_gridcl`). New: the checks of rule 5.
- Later milestones affected: all use the new names. The first crates.io release publishes the `tetraloma-*`
  crates under these names.

## Decision record

Accepted by the maintainer on 2026-11-19, as proposed. The repository transfer was done the same day.
