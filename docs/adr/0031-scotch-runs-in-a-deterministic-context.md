# ADR-0031: Scotch and PT-Scotch run in a deterministic context

- **Status:** Accepted
- **Date:** 2026-10-31
- **Raised in:** M7 T4
- **Design sections affected:** "Partitioning" (the **Reproducibility** bullet); ADR-0029, item 8
  ("Seeds") and item 5 (ptscotch-sys's bindings, through the T5 allowlist).

## Context

"Partitioning", **Reproducibility**:

> PT-Scotch is called with a fixed seed (`SCOTCH_randomSeed`, then `SCOTCH_randomReset` before
> each call), METIS with `METIS_OPTION_SEED`. Scotch on rank 0 seeds and resets under a
> process-wide mutex […].

ADR-0029, item 8, repeats this: PT-Scotch calls `SCOTCH_randomSeed(seed)` and then
`SCOTCH_randomReset()` before each call, and Scotch on rank 0 does the same under one
process-wide mutex (`rank0::RANDOM_STATE`).

M7 T4 implemented `Scotch` exactly so. Its reproducibility test failed.

## Problem

Scotch 7 runs threaded algorithms when it is built with `SCOTCH_PTHREAD`. Homebrew's Scotch
7.0.14 is built that way: the library contains `SCOTCH_PTHREAD_NUMBER` and the thread-context
routines. By default those threaded algorithms are not deterministic. The Scotch 7 manual
(`ptscotch_user7.0.pdf`, §6.3 and §7.17) says:

> SCOTCH_DETERMINISTIC: when set to 0, faster, non-deterministic, multi-threaded algorithms may
> be used; when set to 1, fully deterministic, albeit sometimes slower, multi-threaded
> algorithms are always used.

> […] multi-threaded versions of the libScotch routines may rely by default on
> non-deterministic algorithms that are not only sensitive to the pseudo-random sequence but
> also to system artifacts.

So a fixed seed and a reset do not reproduce a partition. This is the evidence from T4, on the
M3 Max with Homebrew Scotch 7.0.14:

- **The design's mechanism fails.** `SCOTCH_randomSeed(7)`, `SCOTCH_randomReset()` and then
  `SCOTCH_graphMap` (QUALITY strategy, 3%), all under the mutex.
  `partition_threads::scotch_on_hex_prism_pyramid` fails with "seed 7 twice gives different
  parts" (n = 12, p = 2, 2 ranks, counts [2592, 2592]). The parts differ in a few vertices
  along the cut.
- **The environment variable fixes it.** The same binary with `SCOTCH_DETERMINISTIC=1` passes
  both Scotch tests.
- **The context option fixes it too.** This is the code committed in T4 (`b568eee`): a
  `SCOTCH_Context` with `SCOTCH_OPTIONNUMDETERMINISTIC` = 1 and a private random generator,
  bound to the graph with `SCOTCH_contextBindGraph`. It passes three repeated runs with `metis`
  and `scotch` on together (8 tests, concurrent test threads).

PT-Scotch (`libptscotch`) is built on the same `libscotch`. Its sequential phases and the
threaded algorithms on each rank are subject to the same default, so the design's mechanism
very likely fails there too. ptscotch-sys (T5) binds only `SCOTCH_(dgraph|strat|random|version).*`,
so it cannot set the option yet. `SCOTCH_contextBindDgraph` is declared in `ptscotch.h`, and the
other context routines are in `scotch.h`, which `ptscotch.h` includes.

Scotch builds without `SCOTCH_PTHREAD`, or built with the `SCOTCH_DETERMINISTIC` compile flag,
are deterministic either way. The supported builds are not known in advance: Ubuntu's in CI,
and the one T7 builds on Kathleen.

## Options

1. **Keep the design as written.** Use the global seed and reset only. On Homebrew the
   reproducibility bullet ("a fixed seed reproduces the partition for a fixed rank count")
   fails for `Scotch`, and probably for `PtScotch`. Not chosen.
2. **Require deterministic builds.** Document that Scotch must be built without
   `SCOTCH_PTHREAD` or with `SCOTCH_DETERMINISTIC`, set `SCOTCH_DETERMINISTIC=1` in the test
   scripts, and build Kathleen's copy that way in T7. Users of a threaded distribution build
   (Homebrew, and possibly Ubuntu) would silently lose reproducibility, and the environment
   variable would have to be set for every program, not only for the tests. Not chosen.
3. **Run every Scotch and PT-Scotch call in a deterministic context** (proposed). Each call:
   - creates a `SCOTCH_Context` and sets `SCOTCH_OPTIONNUMDETERMINISTIC` to 1;
   - clones a private random generator (`SCOTCH_contextRandomClone`), then calls
     `SCOTCH_contextRandomSeed(seed)` and `SCOTCH_contextRandomReset`;
   - binds the graph (`SCOTCH_contextBindGraph`, or `SCOTCH_contextBindDgraph` for PT-Scotch)
     and maps or partitions the bound container.

   This works with any Scotch 7 build and keeps the seed semantics. `rank0::RANDOM_STATE` stays
   around the clone and the call, as item 8 says. The clone reads the global generator, and
   `Scotch` and `PtScotch` share one `libscotch` in the same process. The deterministic
   algorithms may be slower. That matters for partition time only, not for the acceptance
   bullets, and the cost is measured in T6/T7.

## Proposed decision

Option 3. Reproducibility for a fixed seed must not depend on how the system's Scotch was
built. The context API is Scotch 7's documented way to get it per call, without environment
variables or rebuilds. It applies to `Scotch` (done in T4) and to `PtScotch` (T6). ptscotch-sys
therefore also binds `SCOTCH_context.*` and the `SCOTCH_OPTIONNUM*` constants. Both wrappers
take `rank0::RANDOM_STATE` around the clone, seed and call.
Scotch 6 has no context API; the supported Scotch is 7.x, as already required by Homebrew 7.0.14,
Ubuntu 24.04 (7.0.x; the T5 probe step confirms it in CI) and the 7.0.x that T7 builds.

## Consequences

- **Text to change in docs/design.md:**
  - "Partitioning", **Reproducibility**: "PT-Scotch and Scotch on rank 0 run each call in a
    `SCOTCH_Context` with `SCOTCH_OPTIONNUMDETERMINISTIC` set and a private random generator
    seeded with the fixed seed (`SCOTCH_contextRandomSeed`, then `SCOTCH_contextRandomReset`),
    under a process-wide mutex; threaded Scotch builds are otherwise not deterministic
    (ADR-0031). METIS uses `METIS_OPTION_SEED` under the same mutex; ParMETIS takes the seed in
    `options`. Results are reproducible for a fixed number of ranks only."
  - Backend table, PT-Scotch row: "Scotch 7.x (context API)".
- **ADR-0029:** a line "Amended by ADR-0031 (Scotch and PT-Scotch seeds in a deterministic
  context)" in its decision record, next to the ADR-0030 line. Item 5's list of ptscotch-sys
  bindings gains `SCOTCH_context*`.
- **Public API changes:** none. ptscotch-sys's bindings gain the context functions and the
  `SCOTCH_OPTIONNUM*` constants; the crate's public items are the raw bindings anyway.
- **Acceptance tests affected:** none change. `partition_seed` and `partition_threads` check
  reproducibility as before; this ADR is what lets them pass on threaded builds.
- **Later milestones affected:**
  - M7 T6 (`PtScotch`): a context with `SCOTCH_contextBindDgraph`, and the ptscotch-sys
    allowlist extended.
  - M7 T7: Kathleen's Scotch may be built with or without `SCOTCH_PTHREAD`; either works.
  - M8 (the default partitioner for distributed reads) inherits this unchanged.
- **docs/plans/M7.md:** question 21 and the T4 Deviations-log row refer to this ADR.

## Decision record

Accepted by the maintainer on 2026-10-31, as proposed (option 3), without conditions. Applied in `adr: 0031 apply`.
