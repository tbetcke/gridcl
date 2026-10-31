# ADR-0030: The slab bound cannot be met for few parts

- **Status:** Accepted
- **Date:** 2026-10-31
- **Raised in:** M7 T3
- **Design sections affected:** none in docs/design.md; ADR-0029, item 11 ("Slab bound"), and the
  tests that implement it (`gridcl-mpi-tests::partition_quality`, `gridcl/tests/partition_threads.rs`,
  `rank0::check`).

## Context

ADR-0029, item 11, adds two checks outside the M7 acceptance bullets. They guard against a poor
Scotch reference, because the one-sided "within 10%" reading cannot detect one. The second check
reads:

> **Slab bound.** Every partitioner's cut, Scotch's included, is at most the cut of k slabs
> separated by k − 1 grid planes normal to x: 2(k − 1)n² on the tetrahedral box […] and
> (k − 1)n² on the hexahedron/prism/pyramid box […]. A good k-way partition beats slabs (for
> k = 8, 2 × 2 × 2 blocks cut 3 planes instead of 7). Failing the bound is a test failure.

M7 T0 implemented this as `assert!(cut <= slab_cut_bound(mesh, nparts))`, in two places:

- `partition_quality`, on every rank count it runs: 1–4 on the M3 Max and in CI, and 4 and 8 on
  Kathleen;
- `partition_threads::rank0::check`, for `Metis` and `Scotch` on ThreadComm with 1–4 ranks
  (n = 12).

M7 T3 (`Metis`) fails the ThreadComm check at 2 ranks. The wrapper itself works: the imbalance is
at most 3% and the seeds reproduce.

## Problem

For k = 2, a single grid plane is essentially the minimum bisection of a cube. A heuristic
partitioner that works on the dual graph does not find that plane exactly. It cuts through
subcells and so cuts their internal facets. On the hexahedron/prism/pyramid box, those are the
triangular faces between the six pyramids of a subcell. For k = 3 and 4 the slab partition is
still close to optimal on that box.

The measurements below are METIS cuts on `host_dual_graph` on `Serial`, for seeds 0–4. The ratio
is cut / slab bound.

| Mesh | k = 2 | k = 3 | k = 4 | k = 5–8 |
| --- | --- | --- | --- | --- |
| tetrahedra n = 12 (bound 288 · (k − 1)) | 310–331 (1.08–1.15) | 0.84–0.89 | 0.75–0.79 | ≤ 0.70 |
| tetrahedra n = 24 (bound 1152 · (k − 1)) | 1276–1332 (1.11–1.16) | 0.85–0.95 | 0.74–0.79 | ≤ 0.70 |
| hex/prism/pyramid n = 12, p = 2 (bound 144 · (k − 1)) | 187–196 (1.30–1.36) | 1.09–1.21 | 0.88–0.94 | ≤ 0.93 |
| hex/prism/pyramid n = 24, p = 1 (bound 576 · (k − 1)) | 746–796 (1.30–1.38) | 1.11–1.16 | 0.88–0.99 | ≤ 0.89 |

The table has three consequences:

- At 2 ranks the check fails for every partitioner, on both meshes. At 3 ranks it fails on the
  hexahedron/prism/pyramid box. `partition_quality` therefore cannot pass
  `scripts/mpi-test.sh 1 2 3 4` (M3 Max, CI), and `rank0::check` cannot pass on ThreadComm.
- At 4 ranks, the acceptance case on the hexahedron/prism/pyramid box (n = 41) leaves METIS only
  1–12% of margin. PT-Scotch, which the acceptance allows to be 10% above Scotch, may exceed the
  bound.
- At k ≥ 5 every measured cut is at most 0.93 of the bound.

As written, the check fails good partitions, so it gives no information.

## Options

1. **Keep ADR-0029 as written.** The M7 MPI targets fail at 2 and 3 ranks on every target, and
   probably at 4 on Kathleen. Not chosen.
2. **Apply the bound only for k ≥ 5.** This keeps the strict bound where it has slack. But
   CI and the M3 Max (1–4 ranks) and the k = 4 acceptance runs would then check nothing against
   a poor reference; only the 8-rank Kathleen run would. Not chosen.
3. **Twice the slab bound, for every k** (proposed). Every cut must be at most
   2 · slab_cut_bound. The largest measured ratio is 1.38. A random or scrambled partition cuts
   about (1 − 1/k) of all dual edges, which is about 35 to 70 times the bound at k = 2 on these meshes
   and still far above twice the bound at k = 8. A scrambled reference therefore still fails.
   A merely mediocre Scotch configuration may pass, but the logged cuts and ratios of item 11
   expose it, as before.
4. **Two-sided agreement between METIS and Scotch** (|cut(Metis) / cut(Scotch) − 1| ≤ 10%) in
   place of the slab bound. Two independent libraries would cross-check each other. But this
   adds a second acceptance-like threshold between two libraries that may legitimately differ
   by more than 10% on a given mesh and rank count. Not chosen.

## Proposed decision

Option 3. The slab bound of ADR-0029, item 11, becomes "every partitioner's cut, Scotch's
included, is at most twice the slab cut". It applies on every target and at every rank count,
including k ≥ 2 on ThreadComm. Failing it is still a test failure. `slab_cut_bound` keeps its
meaning: the exact cut of the slab partition, as tested. The tests compare against
`2 * slab_cut_bound(..)`, and rank 0 logs cut / slab bound next to the ratios to Scotch's cut.

## Consequences

- **Text to change in docs/design.md:** none. docs/design.md does not mention the slab bound.
  This ADR amends ADR-0029, item 11. A line "Amended by ADR-0030 (factor 2 on the slab bound)"
  is added to ADR-0029's decision record when this ADR is applied.
- **Public API changes:** none.
- **Acceptance tests affected:** none of the three M7 bullets.
  - `partition_quality`: the slab assertion uses `2 * bound`, and the log line adds cut / bound.
  - `partition_threads::rank0::check`: the same assertion; the docs of both tests name this ADR.
  - docs/plans/M7.md: the "Reference sanity" row of the coverage table.
- **Later milestones affected:** none.

## Decision record

Accepted by the maintainer on 2026-10-31, as proposed (option 3), without conditions. Applied in `adr: 0030 apply`.
