# ADR-0063: FacetNormals on intervals in f32 at sizes f32 can resolve

- **Status:** Accepted
- **Date:** 2026-11-17
- **Raised in:** M10 T9
- **Design sections affected:** "Benchmarks" (Degree sweep (M9h), the M10 sentences); Roadmap M10 bullet 4 as read by ADR-0057 item 3 and ADR-0060 items 2 and 3

## Context

"Benchmarks" measures the degree-dependent families in M10 "at p = 1, 4 and 8 only, with the candidate protocol (ADR-0057)". The protocol's rule sizes are 10^6 and 10^7 quadrature points on every platform (ADR-0057 item 3, ADR-0060 item 2). A strategy that fails on a platform of a class is not selectable for that class (ADR-0056). The acceptance test `high_order_record::m10_policy_first_use_within_30_s` requires each choice at p = 1, 4 and 8 to have run (`status == "ok"`) on every platform of its class.

The benchmark's FacetNormals workload on intervals uses `gridcl_testing::interval_box(side, p)`: `side` equal cells on [0, 1], with about size / 2 cells (two facet points per cell). Its check compares the device's unit normals with an f64 reference. The tolerance is scaled by `side`, because the Jacobian of a cell of length 1 / side loses accuracy in proportion.

## Problem

In f32, the interval mesh is beyond f32's resolution at the larger sizes.

**p = 4 at 10^7 points.** There are 5 × 10^6 cells and 2 × 10^7 node intervals of 5 × 10^-8. f32's spacing just below 1 is 6 × 10^-8, and one binade holds only 2^23 ≈ 8.4 × 10^6 values. So nodes coincide.

**p = 8 at 10^6 points.** The nodes are still distinct, but the Jacobian is a sum of node coordinates of size about 1 with large equispaced derivative weights. Its rounding error is then as large as the Jacobian itself, 1 / (8 · side).

M10 T9 (commit ff43a31, `docs/perf/data/m10-degree-*.json`) records FacetNormals on intervals in f32 as `failed: check: relative error inf`. The computed Jacobian rounds to exactly 0. This happens for both strategies (`direct`, `loops`), at both sizes where the problem shows, on all five platforms:

| p | 10^6 points | 10^7 points |
| --- | --- | --- |
| 1 | ok (error 0) | ok (error ≤ 6 × 10^-8) |
| 4 | ok (error ≤ 6 × 10^-8) | failed on all 5 platforms |
| 8 | failed on all 5 platforms | failed on all 5 platforms |

M9h's records show where the limit lies:

- **Clean:** at 10^5 points, every degree 1–8 is clean (error ≤ 1.2 × 10^-7). At 10^6 points, p ≤ 6 is clean.
- **Sign flips, accepted:** at 10^6 points, p = 7 and 8 give error 2.0, and at 10^7 points p ≥ 4 does too. An error of 2.0 is a flipped normal. M9h's size-scaled tolerance (50 at p = 4, 1,317 at p = 8 for 10^7) accepted it.
- **What changed:** the same configurations now give a Jacobian of exactly 0, whose error is infinite.

Consequences:

- At p = 4 and 8, no strategy of this entry is selectable for either class. `degree_policy` stores the rule's default (`loops`).
- `m10_policy_first_use_within_30_s` fails, and can never pass: whatever is chosen, its records there are `failed`.
- The f32 timings at these sizes would compare strategies on a degenerate mesh, so they would not be useful even if the check accepted them.

## Options

1. **Keep the design as written.** The entry stays without a selectable strategy at p = 4 and 8 in f32. The acceptance test `m10_policy_first_use_within_30_s` cannot pass, and M10 cannot be closed. Not chosen.
2. **Mark the entry as unmeasurable.** The first-use test would skip entries where every strategy fails on every platform, and the policy keeps the default `loops`. No re-runs. But it leaves an f32 policy entry chosen without any timing, and it widens an acceptance test for all entries. Not chosen.
3. **Time the degenerate configurations anyway.** The sweep would record the timing of a run whose check fails only from f32 resolution, and count it as ran. This needs a criterion for "fails only from resolution", which the check cannot tell apart from a wrong kernel. It also times strategies on a mesh with zero-length cells. Not chosen.
4. **Measure this entry at sizes f32 resolves (chosen by the maintainer, 2026-11-17).** For FacetNormals on intervals in f32 only, the rule sizes are lower by one decade where 10^6 or 10^7 points exceed f32's resolution:

   | Degrees | Rule sizes (points) | Note |
   | --- | --- | --- |
   | 1–3 | 10^6 and 10^7 | unchanged |
   | 4–6 | 10^5 and 10^6 | |
   | 7–8 | 10^4 and 10^5 | |

   These bands follow M9h's clean results above. The 10^4 size has fewer than 10^4 cells, but each timed run still lasts at least 20 ms (ADR-0018). Every other family, cell type and scalar keeps 10^6 and 10^7.

## Proposed decision

Option 4. The rest of the candidate protocol (ADR-0057 item 3, ADR-0060 items 2 and 3) applies unchanged at these sizes: the candidates, five runs, interleaving on CPUs, the generic rule's admission at both sizes and its bound on every size. The ADR-0060 item 3 re-measurement uses the bands too. For this entry, the generic rule and the selectability of ADR-0056 consider only records at its rule sizes. T9's records at 10^7 points (p = 4) and at 10^6 and 10^7 points (p = 8) stay in the data files as measured; they are history and are not read.

T9 adds the missing records on each of the five platforms, a few minutes per platform at the T9 commit:

- p = 4 at 10^5 points;
- p = 8 at 10^4 and 10^5 points.

The existing p = 4 records at 10^6 points are already clean.

## Consequences

- **Text to change in docs/design.md:** "Benchmarks", Degree sweep (M9h). After "In M10, and after a CubeCL pin bump, the degree-dependent families are measured at p = 1, 4 and 8 only, with the candidate protocol (ADR-0057).", add: "FacetNormals on intervals in f32 is measured at 10^5 and 10^6 points at p = 4–6 and at 10^4 and 10^5 points at p = 7–8, where 10^6 or 10^7 equispaced points exceed f32's resolution (ADR-0063)."
- **Public API changes:** none.
- **Code (gridcl-bench):**
  - one function gives the rule sizes of a (family, cell, scalar, degree) entry;
  - `degree_sweep --candidates` takes its default sizes from it;
  - `decide_generic::decide_degree` and the M10 report read it.
- **Acceptance tests affected:**
  - `high_order_record::m10_degree_check_covers_p1_4_8_on_every_platform` checks each entry's rule sizes instead of the fixed `M10_SIZES`;
  - `high_order_record::degree_policy_follows_the_generic_rule` recomputes the rule at those sizes;
  - `m10_policy_first_use_within_30_s` is unchanged.
- **Plan:** a T9 Deviations-log row (the re-measurement, its runs); T10 uses the new sizes.
- **Later milestones affected:** the candidate protocol after a CubeCL pin bump measures this entry at the same sizes. A later change to the benchmark mesh, for example one that stays resolvable in f32, could return it to 10^6 and 10^7 through a new ADR.

## Decision record

Accepted by the maintainer on 2026-11-17, as proposed (option 4). Applied in `adr: 0063 apply`: docs/design.md ("Benchmarks", Degree sweep (M9h)) and issue #25 (comment). The code, test and re-measurement changes are M10 T9's.
