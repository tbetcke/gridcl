# ADR-0027: The locality acceptance check counts owned node indices

- **Status:** Accepted
- **Date:** 2026-10-29
- **Raised in:** M6 T12 (docs/plans/M6.md)
- **Design sections affected:** "Implementation roadmap / M6", last acceptance bullet.

## Context

The last M6 acceptance bullet, checked on 1–4 ranks (docs/design.md, "Implementation roadmap / M6"):

> on a randomly permuted structured box, locality ordering at least halves the mean over cells of (max − min local node index in the cell).

The acceptance test is `gridcl_testing::parallel::locality`, called by the MPI target `gridcl-mpi-tests::locality` and by `gridcl/tests/parallel_threads.rs`. It shuffles the cells and nodes of a hexahedral or tetrahedral box at p = 1 and 2. It splits the box into even input chunks, builds it with `CellOrdering::Input` and with `CellOrdering::Hilbert` (block partitioner, `GhostMode::None`), and compares `mean_spread`: the mean over owned cells of all ranks of (max − min local node index). T0 chose the smallest box with at least 64 cells per rank.

Two rules fix the local node layout:

- the `IndexMap` layout `[owned | ghost]` (ADR-0009);
- ghost vertices and nodes ordered by (owner rank, global index) (ADR-0026, item 16).

The cell ordering therefore controls only the order of the owned nodes. Every ghost node sits after all owned nodes, in an order that does not depend on this rank's cell order.

## Problem

With T12, `build_on` runs the whole pipeline. The test fails, and on 2–4 ranks it cannot pass as written. Hilbert/Input ratios of `mean_spread` (must be ≤ 0.5), measured on the CPU runtime:

| Ranks | Hexahedra, n = 6 / 8 | Tetrahedra, n = 6 / 8 |
| --- | --- | --- |
| 1 | 0.32–0.34 / 0.24–0.26 | 0.28–0.36 / 0.20–0.27 |
| 2 | 0.77–0.81 / 0.73–0.78 | 0.74–0.85 / 0.71–0.82 |
| 3 | 0.86–0.91 / 0.82–0.88 | 0.85–0.93 / 0.84–0.92 |
| 4 | 0.89–0.95 / 0.88–0.94 | 0.90–0.97 / 0.90–0.94 |

The causes:

1. **Random partitions.** M6 has only the block partitioner, which splits by input position. On a randomly permuted input, each rank therefore gets a random set of cells, and most of its cells touch ghost nodes. A cell with an owned node and a ghost node spans about half the owned range, whatever the cell order. This adds the same large term to both orderings, so the ratio tends to 1 as the rank count grows.
2. **Better partitions do not fix it.** Geometric chunks (slabs of the box, shuffled within each rank) still give 0.51–0.64 on 4 ranks at n = 10 (0.30–0.55 on 2–3 ranks). A slab's interface layer is a fixed fraction of its cells, so the ratio falls only like P/n.
3. **The box is too small even on one rank.** T0's box (hexahedra n = 4, tetrahedra n = 3 on one rank) gives 0.55 and 0.81 serially. Hilbert spread grows like n², shuffled spread like n³, and the bound holds with margin only from n ≈ 6–8.

The Hilbert implementation is not at fault. Serially the ratio falls steadily: 0.85, 0.69, 0.55, 0.42, 0.34, 0.30, 0.26 for hexahedra n = 2…8 at p = 1. Reverse Cuthill–McKee behaves the same way.

**Owned-only metric.** Excluding ghost node indices from each cell's max − min measures only what the ordering controls. Cells with fewer than two owned nodes count as 0. On one rank there are no ghosts, so it equals the design's metric. With the test's own random partition (globally shuffled input, even chunks, block partitioner), the Hilbert/Input ratio is:

| Ranks | n = 6 | n = 8 |
| --- | --- | --- |
| 1 | 0.28–0.36 | 0.20–0.27 |
| 2 | 0.28–0.41 | 0.21–0.31 |
| 3 | 0.30–0.46 | 0.20–0.30 |
| 4 | 0.27–0.51 | 0.21–0.35 |

## Options

1. **Keep the design as written.** The test would need partitions with locality and boxes that grow with the rank count. M6 has no such partitioner, so the test would have to choose geometric chunks, and from the slab data it would need n ≈ 20 on 4 ranks (48 000 tetrahedra at p = 2 per case) and far more on Kathleen's 16 ranks. It would then measure partition quality (M7's subject) more than the ordering, and it would not use a "randomly permuted" box in any global sense. Not proposed.
2. **Count owned node indices only.** The metric becomes the mean over owned cells of (max − min local index over the cell's owned nodes). It is unchanged on one rank, isolates exactly what `CellOrdering` controls, and keeps the randomly permuted box and the block partitioner. The box has at least 8 cells per edge and at least 64 cells per rank. Proposed.
3. **Check the halving on one rank only.** On 2–4 ranks, require only that Hilbert does not increase `mean_spread`. This is weaker: it would pass even if the ordering had no effect in parallel.
4. **Move the parallel part of the criterion to M7,** with the real partitioners, and keep only the one-rank check in M6. This keeps the metric but defers the check, and M7's partitioners still leave the ghost-node term (cause 2).

## Proposed decision

Option 2. Ghost nodes come after the owned nodes in an order that ADR-0009 and ADR-0026 fix independently of the cell ordering. Their indices measure the partition and the ownership rule, not locality ordering. Counting owned node indices only keeps the bullet's intent: a cell's nodes lie close together in the part of the node array that this rank orders. It also keeps the bullet testable on every rank count with the block partitioner. On one rank the check is the design's check exactly. The box grows to at least 8 cells per edge, which gives a margin of at least 0.15 below the bound in every measured case on 1–4 ranks.

## Consequences

- **docs/design.md:** in "Implementation roadmap / M6", the last acceptance bullet becomes:
  > on a randomly permuted structured box, locality ordering at least halves the mean over owned cells of (max − min local index of the cell's owned nodes); ghost nodes, ordered by (owner rank, global index), are excluded (ADR-0027).
- **Public API:** none.
- **Acceptance tests affected:**
  - `gridcl_testing::parallel::{mean_spread, locality}`: `mean_spread` skips ghost node indices, and `locality` uses n ≥ 8 with at least 64 cells per rank;
  - `gridcl-mpi-tests::locality` and `gridcl/tests/parallel_threads.rs::locality` call these helpers unchanged.

  The plan's acceptance-coverage row and a Deviations-log row record the change.
- **Later milestones affected:** none directly. M7 may add a partition-quality check of its own. M8's parallel reader, which chooses the default ordering, inherits this metric for any locality test it adds.

## Decision record

Accepted by the maintainer, 2026-10-29 (option 2), without conditions.
