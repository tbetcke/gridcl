# ADR-0032: The cut bullets apply at 4 and 8 ranks, and METIS within 25% of Scotch

- **Status:** Accepted
- **Date:** 2026-10-31
- **Raised in:** M7 T6
- **Design sections affected:** "Implementation roadmap / M7" (the **Acceptance** paragraph and its
  second bullet); ADR-0029, item 11 (the reading of "within 10%").

## Context

docs/design.md, "M7 Partitioners":

> **Acceptance** on a 10^6-cell tetrahedral box and a 2 · 10^5-cell mixed
> hexahedron/prism/pyramid mesh, at 4 and 8 ranks […], read as in ADR-0029 […]:
>
> - PT-Scotch's edge cut is within 10% of sequential Scotch on the same graph, and METIS's
>   within 10% of Scotch;

ADR-0029, item 11, reads "within 10%" as one-sided: cut(PtScotch) ≤ 1.10 · cut(Scotch) and
cut(Metis) ≤ 1.10 · cut(Scotch), with nparts equal to the number of ranks.

M7 T0 implemented this in `gridcl-mpi-tests::partition_quality`. The test asserts both ratios on
every rank count it runs: 1–4 on the M3 Max and in CI (`scripts/mpi-test.sh 1 2 3 4`), and 4
and 8 on Kathleen. It uses seed 0. M7 T6 (`PtScotch`) is the first task in which the test can
run, because it calls all three partitioners.

## Problem

The measurements were taken on the M3 Max with Homebrew Scotch 7.0.14, the vendored METIS of the
`metis` crate and the pipeline's dual graph (`dual_graph_on`) of the two acceptance meshes. The
8-rank runs were oversubscribed. The partitions depend only on the graph, the rank count and the
seed, not on timing. Each cell gives cut(X) / cut(Scotch) with the same seed: first seed 0 (the
test's), then the range over seeds 0–9.

| Mesh | k | PT-Scotch | METIS |
| --- | --- | --- | --- |
| tetrahedra n = 56 (1 053 696 cells) | 2 | 0.997 (0.997–1.076) | **1.160** (1.133–1.186) |
| | 3 | 0.917 (0.893–0.965) | 1.047 (1.035–1.121) |
| | 4 | 0.962 (0.890–0.991) | 1.075 (1.049–1.116) |
| | 8 | 1.040 (1.040–1.086) | **1.184** (1.184–1.227) |
| hex/prism/pyramid n = 41 (208 444 cells) | 2 | **1.218** (1.218–1.722) | **1.231** (1.216–1.362) |
| | 3 | **1.229** (1.104–1.229) | **1.128** (1.098–1.155) |
| | 4 | 1.051 (1.031–1.129) | 1.079 (1.017–1.085) |
| | 8 | 1.032 (0.979–1.032) | 1.079 (1.041–1.109) |

Scotch's reference is strong. On the tetrahedral box at k = 8 its cut is 1.08 times that of
2 × 2 × 2 blocks (18 816). At k = 2 it is within 0.5% of a single plane (6272). Every
partitioner's weighted imbalance stays at most 2.7%.

The table has three consequences:

1. **k = 2 and 3 are outside the design's acceptance, which names 4 and 8 ranks, but the test
   asserts there.** Both PT-Scotch and METIS fail there with seed 0. This is ADR-0030's finding
   again: at few parts Scotch comes close to the optimal plane, and small absolute differences
   become large ratios. `partition_quality` therefore cannot pass `scripts/mpi-test.sh 1 2 3 4`
   (M3 Max, CI).
2. **PT-Scotch meets the bullet at 4 and 8 ranks** with seed 0: 0.962 and 1.051 at k = 4, 1.040
   and 1.032 at k = 8. Two of ten seeds exceed 1.10 at k = 4 on the hexahedron/prism/pyramid mesh.
3. **METIS cannot meet the bullet at 8 ranks on the tetrahedral box.** Every seed gives 1.184 to
   1.227. Other METIS settings, measured with seeds 0–2, do not reliably fix it:

   | METIS setting | tetrahedra, k = 8 | worst at k = 4 | hex/prism/pyramid, k = 8 |
   | --- | --- | --- | --- |
   | `PartGraphKway`, as built (T3) | 1.184–1.201 | 1.112 | 1.079–1.109 |
   | `PartGraphKway`, `NITER` 50 | identical | identical | identical |
   | `PartGraphKway`, `NCUTS` 8 | 1.164–1.190 | 1.081 | 1.049–1.069 |
   | `PartGraphRecursive` | 1.106–1.146 | 1.074 | 1.062–1.154 |
   | `PartGraphRecursive`, `NCUTS` 8 | 1.075–1.104 | 0.992 | 1.025–1.045 |

   Even the best setting fails by a small margin with seed 0 (1.104), and it costs eight
   partitioning runs on rank 0.

The METIS bullet as written therefore fails at a rank count the design names, whatever the
seed or setting.

## Options

1. **Keep the design as written.** M7 cannot close: METIS fails at 8 ranks on the tetrahedral
   box, and the M3 Max and CI runs fail at 2 and 3 ranks. Not chosen.
2. **Tune METIS** (recursive bisection with `NCUTS` 8) and keep 1.10. This changes ADR-0029's
   METIS call (`METIS_PartGraphKway`) and multiplies its time by eight. It still fails at k = 8
   with seed 0 (1.104), so it would need another seed or a further relaxation, which only hides
   the variation from seed to seed. Not chosen.
3. **Drop the METIS bullet** and keep only its logged ratio. METIS on rank 0 is the default
   partitioner when `ptscotch` is off ("Default choice"), so its quality should stay checked
   against a bound. Not chosen.
4. **Apply the cut bullets at 4 and 8 ranks, as the design says; PT-Scotch within 10%, METIS
   within 25%** (proposed).
   - At 1–3 ranks the cuts and ratios are only logged.
   - Imbalance, reproducibility and the slab bound (ADR-0030) stay asserted at every rank count.
   - PT-Scotch keeps 1.10.
   - METIS's bound becomes 1.25: seed 0 gives at most 1.184, and the worst of ten seeds is 1.227.
     A random or scrambled partition would still fail by a factor of 10 or more.

## Proposed decision

Option 4.
- The design's acceptance already names 4 and 8 ranks, so asserting the cut ratios at 2 and 3
  ranks was a stricter reading in ADR-0029's tests, not a design requirement.
- PT-Scotch, the production default, meets the 10% bullet at those rank counts.
- For METIS on rank 0, the measurements show that 10% of a strong Scotch reference is not
  attainable at k = 8 with any setting tried. 25% still guards the rank-0 default against
  misconfiguration, and the logged ratios show its real distance.
- The METIS call stays as T3 built it.

## Consequences

- **Text to change in docs/design.md:**
  - "M7 Partitioners", **Acceptance** paragraph: the parenthesis becomes "(imbalance for every
    external partitioner; cut(PT-Scotch) ≤ 1.10 · Scotch's and cut(METIS) ≤ 1.25 · Scotch's;
    reproducibility per partitioner and rank count; ADR-0029, ADR-0032)".
  - The second bullet becomes "PT-Scotch's edge cut is within 10% of sequential Scotch on the
    same graph, and METIS's within 25% of Scotch;".
- **ADR-0029:** add the line "Amended by ADR-0032 (cut bullets at 4 and 8 ranks; METIS within
  25%)" to its decision record. In item 11, the cut condition becomes cut(Metis) ≤ 1.25 · cut(Scotch).
- **Public API changes:** none.
- **Acceptance tests affected:**
  - `gridcl-mpi-tests::partition_quality`:
    - the ratio assertions run only when nparts is 4 or 8, and the METIS factor is 1.25;
    - every ratio is still printed on rank 0 at every rank count;
    - imbalance and twice the slab bound stay asserted everywhere.
  - docs/plans/M7.md: the acceptance-coverage row of the cut bullet, and an open-question row.
  - GitHub issue #22: the acceptance text of the cut bullet.
- **Later milestones affected:** none. M8's default choice (PT-Scotch, else METIS on rank 0) is
  unchanged.
- **Risk kept:** PT-Scotch's margin at k = 4 on the hexahedron/prism/pyramid mesh depends on the
  seed (two of ten seeds exceed 1.10). The test uses seed 0, but other Scotch 7.0.x builds (Ubuntu
  in CI, Kathleen's from T7) may give other partitions for that seed. If one of them fails, the
  numbers go to a new ADR, not into a silent change of seed.

## Decision record

Accepted by the maintainer on 2026-10-31, as proposed (option 4), without conditions. Applied in `adr: 0032 apply`.
