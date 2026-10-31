# ADR-0033: The features `metis` and `parmetis` are mutually exclusive

- **Status:** Proposed
- **Date:** 2026-10-31
- **Raised in:** M7 T9
- **Design sections affected:** "Partitioning" (the backend table); "Workspace layout", "Cargo
  features"; ADR-0029, item 14 (question 17 of docs/plans/M7.md).

## Context

ADR-0029, item 14:

> **METIS link clash.** If vendored METIS and ParMETIS's `libmetis` clash, `parmetis` switches the
> `metis` crate to its system build (`parmetis = [..., "metis?/use-system"]`), so both use one
> METIS. M7 T9 checks the build with both features.

The `metis` feature uses the `metis` crate (0.2.2) with its default feature `vendored`: metis-sys
0.3.2 compiles METIS 5.2.1 and GKlib from source into a static library. ParMETIS 4.0.3 uses
METIS 5.1.0: Debian's and Ubuntu's `libparmetis.so` depend on `libmetis.so.5`, and the build of
`scripts/hpc/build-partition-libs.sh` (Kathleen, M3 Max) links the METIS of its own tarball into
`libparmetis`. METIS 5.1 and 5.2 export the same names (`METIS_*`, `libmetis__*`, `gk_*`) with
different internals.

## Problem

1. **The clash happens, on Linux, at run time.** Both features link without error, but an
   executable that links the vendored METIS exports those symbols in its dynamic table, and on
   ELF they take precedence over the shared library's own definitions. ParMETIS's METIS calls
   then run METIS 5.2.1 code on 5.1.0 data:
   - Kathleen (job 616046, Open MPI 4.1.6, the script's ParMETIS): `partition_quality`,
     `partition_seed` and `partition_build` with `GRIDCL_MPI_FEATURES=partitioners,parmetis` die
     with SIGSEGV at 4 and 8 ranks, each when `ParMetis` runs (after Scotch, PT-Scotch and METIS
     succeed). The test executable exports 461 symbols that `libparmetis.so` also defines (job
     616060).
   - Ubuntu 24.04 (ptscotch workflow on a throwaway branch, run 41211890562, Ubuntu's
     `libparmetis-dev` and `libmetis-dev`): the executable exports 60 symbols that
     `libmetis.so.5` defines; `partition_quality` and `partition_seed` die with SIGSEGV at 2 ranks.
   - macOS (M3 Max): no clash, because the two-level namespace binds each library to its own
     METIS; every target passes at 1–4 ranks with both features.
2. **Item 14's remedy cannot be expressed.** Cargo features only add: `metis?/use-system` turns
   on metis-sys's `use-system` but cannot turn off its default `vendored`, and metis-sys builds
   the vendored library whenever `vendored` is on (`build.rs`: `#[cfg(feature = "vendored")] fn
   build_lib`), with or without `use-system`. Depending on `metis` with `default-features = false`
   and `metis = ["dep:metis", "metis/vendored"]` gives the same union when both features are on.
3. **Hiding ParMETIS's METIS is not enough.** Linking the script's `libparmetis` with
   `-Wl,--exclude-libs,ALL` removes METIS from its dynamic table, and then all three targets pass
   with both features at 4 and 8 ranks on Kathleen (job 616073; cut(ParMetis) / cut(Scotch)
   1.05–1.12). A system ParMETIS that links a shared `libmetis` (Ubuntu, item 1) still crashes.

## Options

1. **Keep the design as written** (ADR-0029, item 14). Not possible: problem 2.
2. **Hide ParMETIS's METIS in the build script's library only.** Fixes Kathleen and the M3 Max;
   every system ParMETIS with a shared `libmetis` (Debian, Ubuntu, most distributions, the hosted
   CI job) still crashes when both features are on. A segfault that depends on how the library
   was packaged is not acceptable for a library feature.
3. **Make `metis` and `parmetis` mutually exclusive.** gridcl-partition fails to compile with
   both (`compile_error!` naming this ADR). A program uses METIS on rank 0 or ParMETIS, not both;
   every distribution's ParMETIS works, and so does the vendored METIS. Costs: no binary holds
   both partitioners, and `--all-features` builds fail (nothing in the repository uses them). The
   tests that combine all four partitioners split into two builds (Consequences). Without the
   vendored METIS nothing else defines METIS's symbols, so neither crash mechanism applies; T9
   confirms this with `parmetis` alone on Kathleen and in the Ubuntu workflow.
4. **With `parmetis`, `Metis` calls the METIS that ParMETIS links**, through bindings in
   parmetis-sys. The `metis` crate still arrives with the feature `metis`, and whether a call to
   `METIS_PartGraphKway` binds to the vendored archive or to `libmetis` depends on the linker's
   order of the libraries, which gridcl does not control. Not robust.
5. **Vendor METIS in a gridcl crate with hidden symbols** (for example a crate replacing the
   `metis` crate, compiling METIS with `-fvisibility=hidden` and binding `METIS_PartGraphKway`).
   Every combination of features works on every platform. Costs: a new workspace crate and its
   bindings, a change to the crate table and dependency rules, and maintaining the vendored
   sources; more work than M7's remaining tasks.

## Proposed decision

Option 3: the features `metis` and `parmetis` are mutually exclusive. It is the only option that
is correct with every ParMETIS build and needs no new crate. ParMETIS is optional (the backend
table: "optional; check the licence terms"), and the default choice (PT-Scotch, else METIS, else
Block) never needs both. Option 5 remains open for a later milestone if both in one binary turn
out to be needed.

## Consequences

- Text to change in docs/design.md:
  - "Partitioning", backend table: the METIS and ParMETIS rows add "not together with
    `parmetis`" / "not together with `metis` (ADR-0033)".
  - "Workspace layout", "Cargo features": after "`ptscotch` and `parmetis` imply `mpi`." add
    "`metis` and `parmetis` are mutually exclusive: the vendored METIS and ParMETIS's METIS clash
    at run time (ADR-0033)."
  - ADR-0029, item 14 is superseded by this ADR (a note under its decision record).
- Public API changes: none (a compile error for the combination of the two features).
- Acceptance tests affected: none of the bullets; their tests run in builds without ParMETIS.
  - gridcl-mpi-tests: `partitioners = ["gridcl/ptscotch", "gridcl/scotch"]`, a new feature
    `metis = ["gridcl/metis"]`, and `parmetis`; `partition_quality` asserts the METIS ratio when
    `metis` is on. The acceptance run is `GRIDCL_MPI_FEATURES=partitioners,metis`; the ParMETIS
    run is `GRIDCL_MPI_FEATURES=partitioners,parmetis` (imbalance, seed, `partition_build`, the
    empty-rank check, and cut(ParMetis) / cut(Scotch) logged).
  - docs/plans/M7.md: the "all" targets of the acceptance table, T7's and T9's commands and T10's
    feature lists (`metis,scotch,ptscotch` and `scotch,ptscotch,parmetis`, each with
    `gridcl-testing/mpi`) follow; question 17 closes with this ADR.
- Later milestones affected: M8 (`GmshOptions::partitioner`, the default choice): none beyond
  the documented exclusion.
