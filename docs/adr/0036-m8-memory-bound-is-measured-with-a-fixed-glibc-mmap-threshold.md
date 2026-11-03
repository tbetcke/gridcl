# ADR-0036: The M8 memory bound is measured with a fixed glibc mmap threshold

- **Status:** Accepted
- **Date:** 2026-11-03
- **Raised in:** M8 T5b (docs/plans/M8.md)
- **Design sections affected:**
  - "Implementation roadmap / M8": the second acceptance bullet (through ADR-0035, item 13);
  - ADR-0035, item 13 ("Read and run");
  - docs/plans/M8.md: rule 10, the T5a and T12 commands; docs/dev-environment.md (T11).

## Context

The second M8 acceptance bullet reads: "reading a 10^7-cell binary file on 8 ranks of one Kathleen node keeps peak resident memory per rank below twice the rank's final mesh size plus 256 MiB (measured with getrusage)". ADR-0035 item 13 fixes the file (`cube_box(Tetrahedron, 119, 1)`, 10 110 954 cells), the read (`Block`, `GhostMode::None`, release build, CPU runtime) and the peak (`ru_maxrss`). On Kathleen the final mesh size (`Mesh::footprint_bytes`) is 134 MiB per rank, so the bound is 525 MiB.

T5a measured a peak of 1292–1306 MiB (job 618681). T5b's reductions so far, none of which changes a public item:

- **Host side.**
  - The builder frees its input after stage 1, and every stage drops its intermediates once they are used.
  - `Block`, which reads only the vertex distribution, gets a graph without edges instead of the facet dual graph.
  - Migration keeps the cells that stay on their rank in place.
  - Ordering and numbering move the cells instead of copying them; stage 8 builds its cell-number map only when a ghost cell asks.
  - Exchanges no longer hold their send lists twice, and stage 1 sizes its cell arrays exactly.
- **Device side.**
  - A hidden `gridcl_device::radix_sort_owned` sorts through its input buffers, so entity creation and tag matching hold two sets of keys instead of three.
  - Entity creation computes the touching cells before the key groups, so their scratch buffers never coexist.

The per-stage profile (`read_memory` prints, per rank, the pools, the peak RSS, a counting global allocator's live and peak heap, and Linux's current `RssAnon`/`RssFile`/`RssShmem`) now shows the following for rank 0 on Kathleen (job 618850; all ranks within 4 % of rank 0):

| Point | Live heap (peak) | Pools reserved | `RssAnon` now | Peak RSS |
| --- | --- | --- | --- | --- |
| after stages 7–8 | 172 MiB (272) | 0 | 303 MB | 316 MiB |
| after stage 9 (upload) | 170 MiB (337) | 113 MiB | 400 MB | 499 MiB |
| after stage 10 (tags) | 356 MiB (412) | 328 MiB | 643 MB | 671 MiB |

The heap counts every Rust allocation, the CPU runtime's pool pages included. Kernel compilation (stages 9 and 10, the first device work) costs about 55 MB of anonymous memory and 25 MB of mapped libraries; a 10^4-cell box peaks at 109 MiB (job 618825).

## Problem

With glibc's default settings the peak is 671–712 MiB per rank (job 618850), against a bound of 525 MiB. About 230 MB of `RssAnon` at stage 9 is memory that gridcl has freed and glibc keeps resident: the live heap is 170 MiB while `RssAnon` is 400 MB. glibc raises its mmap threshold dynamically up to 32 MiB whenever an mmapped block is freed. The pipeline's medium arrays (1–32 MiB: ids, offsets, hash tables, exchange buffers) then come from the brk heap and its thread arenas, and glibc returns them to the system only from the top of the heap. The CPU runtime's pool pages (about 61 MiB each) are always mmapped afresh, so they add to what glibc keeps.

gridcl cannot change glibc's behaviour from library code: `mallopt` and `malloc_trim` are FFI calls outside the crates that may make them (CLAUDE.md), and the measured process is a test binary that keeps the system allocator.

With the mmap threshold fixed at 1 MiB (`GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=1048576`, forwarded by `mpirun -x`), the same code peaks at 514–521 MiB and the test passes (job 618850). A second setting, `MALLOC_ARENA_MAX=1`, gains another 20 MiB (job 618829, before the device-side reductions). The margin is 4–11 MiB. What remains is the CPU runtime's pool reservation during entity creation (in-use peak about 230 MiB, reserved 328–344 MiB, of which 141 MiB is the final mesh), plus about 55 MiB of kernel compilation and the host arrays still alive in stage 10.

## Options

1. **Keep the design as written.** The run uses glibc's defaults. That needs about 190 MiB less, and most of the gap is allocator retention that only FFI (`malloc_trim` after each stage), a different allocator, or a much smaller set of medium allocations would remove. Not chosen: it measures glibc's caching rather than what gridcl holds, and the fixes either break the FFI rule or rewrite every stage's data layout.
2. **Fix glibc's mmap threshold for the measured run.** The bound, file, read and peak stay as ADR-0035 item 13 defines them. The measured `mpirun` exports `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=1048576` (plus `-x GLIBC_TUNABLES`). Allocations of 1 MiB or more are then mmapped and returned when freed, so `ru_maxrss` follows gridcl's live memory. The test passes today, with a margin of 4 MiB. No public API or crate change. The local Mac and CI run the small-box smoke test, which is unaffected.
3. **Option 2, and T5b continues for margin.** As option 2, and T5b also removes the remaining device-side duplication in entity creation before it is ticked:
   - `touching_cells`' `run_length` copies the unique keys (30 MB) only to get the run offsets that `unique_by_key` also computes;
   - both run `number_runs` (two 20 MB scratch buffers);
   - the tag-matching sort runs after entity creation, when the mesh already holds 141 MiB.

   Each change is a hidden gridcl-device helper (like `radix_sort_owned`), estimated at 30–60 MiB. T5b's acceptance becomes a margin of at least 32 MiB under option 2's measurement.
4. **Raise the constant.** For example, 2 × final mesh size + 384 MiB under glibc's defaults (652 MiB). Not chosen: glibc's defaults still fail at 671–712 MiB, and a larger constant hides growth that the bound exists to catch.

## Proposed decision

Option 3. The bullet is about how gridcl's own memory scales: rank-local chunks, no whole-file copy, intermediates freed by stage. With glibc's dynamic threshold, `ru_maxrss` measures allocator retention instead of that. A fixed mmap threshold is a documented setting of the measured run, comparable to the release profile ADR-0035 already fixes. The 4 MiB margin left today is too small to survive variation between runs or later changes to the pipeline, so T5b also takes the cheap device-side reductions and is ticked only with a margin of at least 32 MiB.

## Consequences

- **docs/design.md:** none. The bullet keeps its text. This ADR amends ADR-0035, item 13 ("Read and run"): the measured `mpirun` sets `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=1048576`. A line "Amended by ADR-0036" is added to ADR-0035's decision record.
- **Public API:** none. Hidden items, outside the stable API:
  - `gridcl_device::radix_sort_owned` (T5b);
  - under option 3, the helpers that share the run offsets between `run_length` and `unique_by_key`.
- **Acceptance tests:**
  - `gridcl-mpi-tests::read_memory`: unchanged code. Its rustdoc says the bound assumes the fixed threshold on Linux.
  - The T5a command and T12's runs export `GLIBC_TUNABLES` and pass `-x GLIBC_TUNABLES` in `GRIDCL_MPIRUN_FLAGS`.
  - T11's `docs/dev-environment.md` section names the setting.
- **Users:** the setting is not needed to use gridcl on Linux. Without it nothing leaks: the memory glibc keeps stays with the process and later allocations reuse it, but the read's peak resident memory is higher than what gridcl holds (about 190 MiB per rank for the 10^7-cell box on 8 ranks). T11's `docs/dev-environment.md` section says so and names the setting for runs near a node's memory limit; an application may also tune glibc or use another global allocator itself (the FFI rule binds gridcl's crates, not applications).
- **Later milestones:** none directly. A later memory bullet on Linux would use the same setting.

## Decision record

Accepted by the maintainer on 2026-11-03, as proposed (option 3), with the users' note under Consequences added. Applied in `adr: 0036 apply`; T5b is ticked only with a margin of at least 32 MiB under option 2's measurement.
