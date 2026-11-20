# ADR-0057: Minimum test and benchmark runs

- **Status:** Accepted
- **Date:** 2026-11-16
- **Raised in:** before M10 planning (maintainer request, 2026-11-16)
- **Design sections affected:** "Benchmarks" (**What it runs**, **What it reports**, **Record**, **Degree sweep (M9h)**); Roadmap M10 (scope, acceptance bullet 4); Listing H1 (CLAUDE.md). Process documents: CLAUDE.md, IMPLEMENTATION.md, the skills plan-milestone, review-milestone and hpc-test.

## Context

"Benchmarks", **What it runs**: "every candidate family at every vector size the device offers, at QuadField tile widths 1, 4 and 16 for geometry kernels, and on CPUs at block sizes of 4, 16 and 64 KiB. Sizes are 10^5, 10^6 and 10^7 items". **What it reports**: "the median of ten timed runs". The **Degree sweep (M9h)** measures every strategy at p = 1–8. The study "is re-run in M10 for every kernel family and after any CubeCL pin bump", and Roadmap M10 asks for a re-run "on every device class".

The design states the targets and when they run ("Testing and validation", "Development environment and test targets"). It does not say how far a plan may go beyond that matrix, so plans have defaulted to the widest reading.

## Problem

Exhaustive runs have cost more than they have bought:

- M3s swept every configuration on four platforms. M9h's degree sweeps would have taken 10–14 h on the Mac mini and 6–9 h plus queue waits on Kathleen, and had to be cut while running (ADR-0050 to ADR-0052).
- At the M9h review, a full matrix on every target takes 10–12 h. The maintainer cut the re-runs after the review fixes three times, to the runs those fixes reach.

Most of the configurations measured were never close to being chosen. In ADR-0052's data, dropping `shared` on CPUs changed none of 979 choices by more than 5%, and the 10^6-point choice equalled the two-size choice in 471 of 490 cases.

## Options

1. **Keep the design as written.** The M10 re-run repeats the full M3s and M9h protocols on every platform: many hours per platform, then the same again after a CubeCL pin bump.
2. **Minimum runs (chosen by the maintainer).** The standing correctness matrix stays. Beyond it, a plan runs only what shows that the code works and what gives the timings a decision needs. Benchmarks measure a candidate set rather than the full parameter space.
3. **Also thin the standing matrix.** For example, Locust only for the kernels a task changes, and two-node Kathleen MPI only for milestones that touch tetraloma-parallel. Not chosen: the maintainer keeps the matrix.

## Proposed decision

Option 2.

1. **Standing matrix unchanged.** The following stay as they are:
   - the targets and triggers of "Testing and validation" and "Development environment and test targets";
   - f32 and f64;
   - kernel tests at `TETRALOMA_VECTOR_SIZE=scalar` and `widest`;
   - MPI on 1–4 ranks under MPICH and Open MPI;
   - the review targets.
2. **Minimum beyond it.** Each extra test or benchmark run in a plan needs a reason: it shows that the code works on a backend, or it gives a timing that a decision or an acceptance bullet needs. A plan does not:
   - sweep parameters, sizes, degrees or targets exhaustively;
   - repeat a run that passed just to confirm it. A run is repeated only when it is invalid (the node is not idle; ADR-0002, ADR-0037), or when a timing lies within the noise of a decision threshold.

   After review fixes, only the runs those fixes reach are re-run. The earlier full run stands for the rest, and the plan records each cut.
3. **Candidate benchmark protocol (from M10, and after a CubeCL pin bump):**
    - **Configurations.** Each policy entry measures only:
      - its baseline: vector size 1 at the entry's previous tile, block and units; for strategies, `direct`, or `loops` where `direct` is not selectable;
      - each configuration or strategy that M3s or M9h chose for any device class;
      - each configuration or strategy the entry's own policy names.

      A wider sweep runs only for an entry where no candidate meets ADR-0058's bound on every platform, and covers that entry alone.
    - **Sizes and runs.** 10^6 and 10^7 items or points, on every class (CPU records at 10^7 points return; ADR-0052). Each record is the median of five timed runs, each batched to at least 20 ms (ADR-0018), after a warm-up launch that absorbs compilation.
    - **Degrees.** The degree-dependent families are measured at p = 1, 4 and 8. This is enough for M10's p = 8 / p = 4 bullet, and it checks M9h's choices. The degree ranges in between keep coming from M9h's per-degree records (ADR-0058).
    - First-use compile and load times are still recorded for each candidate. The 30 s bound of ADR-0056 still applies.
4. **Plans name their runs.** Each benchmark or remote-run task in a plan states:
   - its targets;
   - the candidate set, sizes and number of runs;
   - an estimate of the wall time.

## Consequences

- Text to change in docs/design.md:
  - "Benchmarks", **What it runs**: the M3s protocol is kept as the description of that study, and the candidate protocol of item 3 is added for M10 and pin bumps.
  - **What it reports**: five timed runs for candidate runs.
  - **Record**: "re-run in M10 … and after any CubeCL pin bump" becomes a re-run with the candidate protocol.
  - **Degree sweep (M9h)**: a closing sentence on M10's p = 1, 4 and 8 check.
  - Roadmap M10, scope and acceptance bullet 4.
  - Listing H1: the new run-plan bullet.
- CLAUDE.md: a new run-plan bullet in the working agreement. IMPLEMENTATION.md: the M10 note. Skills: plan-milestone (plans name their runs), review-milestone (the standing matrix once; reduced re-runs after fixes), hpc-test (no sweeps or confirmation repeats).
- Public API changes: none.
- Acceptance tests affected: tetraloma-bench's protocol checks (`check_protocol` and the coverage tests in `perf_record.rs` and `high_order_record.rs`) accept the candidate protocol for M10 records. This is done in M10.
- Later milestones affected: M10 and every later benchmark or CubeCL pin bump.

## Decision record

Accepted by the maintainer on 2026-11-16, with options chosen in the same session:
- the standing matrix is kept;
- candidates only, at 10^6 and 10^7 with five runs;
- degree runs at p = 1, 4 and 8, with the ranges in between taken from M9h's data.

Applied in `adr: 0057 apply`: docs/design.md ("Benchmarks", Roadmap M10, Listing H1), CLAUDE.md, IMPLEMENTATION.md, the skills plan-milestone, review-milestone and hpc-test, and issue #25.

Amended by ADR-0065 (accepted 2026-11-18): from M10c, the `[Cpu]` vector entries replace item 3's configurations with a grid (vector sizes 1, 4, 8 and 16, blocks of 4, 16 and 64 KiB, both unit settings) and are timed in shuffled interleaved rounds in two processes.
