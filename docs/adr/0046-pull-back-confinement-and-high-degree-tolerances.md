# ADR-0046: Pull-back confinement, and tolerances and page limits at high degree

- **Status:** Proposed
- **Date:** 2026-11-11
- **Raised in:** M9h T6a (the kernel-task check set in release builds)
- **Design sections affected:** "Pull-back" (the iteration bullet); "Testing and validation", **Tolerances** (the degree-scaling bullet). Amends ADR-0042, items 2, 8 and 11, and ADR-0044.

## Context

Release builds run the M9h acceptance tests at p = 5–8 (T0b). Until T6a, every release run stopped at p = 1 on a strategy that did not run yet, so the checks at p ≥ 5 had never run. T6a's check set ran them with `loops` forced for every family but the pull-back. In release, on the CPU runtime and on Metal, three tests fail:

- `high_order::factors_match_host_reference`: J of the GLL hexahedron box at p = 6 in f32, relative error 2.03e-5 against 1e-5 (scale 1).
- `high_order::pull_back_recovers_reference_points`: ξ of the equispaced ball at p = 8 in f32, relative error 1.335 against 4.24e-2 (scale 4245).
- `high_order_invariance::vector_size_invariance`, large case:
  - CPU: the pull-back fails with `AllocationTooLarge { bytes: 470656, max: 196800 }`.
  - Metal: ξ of the quadrilateral box at p = 3 differs by 3.65e-5 between vector sizes 1 and 2, against 1.02e-5.

`GRIDCL_GEOM_STRATEGY=direct` fails the same three with the same errors (J bitwise equal), so none comes from T6a: the M9 kernels and the test rules meet at p ≥ 5 for the first time.

Three different causes lie behind them, one per section below. Each touches an accepted rule, so they are proposed together.

The design says, in "Pull-back": "Iteration starts at the reference cell's vertex centroid with Δ = K(ξ)(x − X(ξ)) … A lane freezes through a lane select once |Δ|∞ ≤ tolerance. The loop always runs `max_iterations`, with no early exit; non-convergence shows in `last_step`." In "Tolerances": both tolerances "scale by max(1, A / A_ref)", with A the element's amplification factor and A_ref "the same factor of the p = 4 equispaced element of that cell type at the same points". ADR-0042 item 2 adds that test meshes at p ≥ 5 "keep the ratio of coordinate size to cell size no larger than in M9's p ≤ 4 tests". Item 8 sets the large case's `max_page_size` "between the tabulation size and a third of the E-vector". Item 11 requires `direct` to stay bitwise M9's.

## Problem

### 1. The f32 pull-back converges to preimages outside the cell at p = 8

A diagnostic pulled back every Gauss point of degree 2p of `ball(0, p)` (8 hexahedra) on the CPU runtime:

| case | points | ξ off by > 10^−3 | worst ξ error |
| --- | --- | --- | --- |
| f32, p = 4 and 6, both families | 1,000; 2,744 | 0 | 1.7e-4 |
| f32, p = 7, both families | 4,096 | 0 | 4.2e-4 |
| f32, p = 8, equispaced | 5,832 | 1 | 1.31 |
| f32, p = 8, GLL | 5,832 | 5 | 6.33 |
| f64, every case | | 0 | 1.6e-11 |

These are not inaccurate values; they are wrong roots:

- **Equispaced:** a point of cell 0 at ξ = (0.918, 0.984, 0.984) ends at (0.20, −0.33, 1.44) after 16 iterations. With 64 iterations it converges there (last step 2.6e-7). It is another preimage of the same x, outside the cell.
- **GLL:** a point at ξ ≈ (0.016, 0.016, 0.016) converges, with a last step of exactly 0, to (−0.25, −6.3, −2.1).

A degree-8 map has further preimages outside the reference cell. Newton from the centroid has no damping, line search or bound, so a long early step can leave the cell, and outside it the polynomial grows fast. In f64 the same points follow paths that stay in the cell's basin. In f32 the rounding of the expansion coefficients (C′, whose amplification the scale of 4245 measures) perturbs an early step enough to cross into another basin. The GLL ball fails too, so this is not Runge's effect of equispaced nodes. No tolerance covers a wrong root: the tests fail by O(1), and so would any user's point location.

### 2. The f32 error of J grows with the number of nodes, which A does not measure

An error budget measured every block of the accuracy set (`MeshSet::Accuracy`), p = 1–8, f32 and f64, 294 blocks, on the CPU runtime, against the current tolerances:

| block, f32 | nodes | A / A_ref | J error | error / tolerance (worst of J, det J, K) |
| --- | --- | --- | --- | --- |
| hexahedron box p = 4, GLL | 125 | 1.00 | 3.1e-6 | 0.37 |
| hexahedron box p = 6, GLL | 343 | 1.00 | 2.0e-5 | 2.03 |
| hexahedron box p = 8, GLL | 729 | 1.11 | 4.7e-5 | 4.22 |
| hexahedron box p = 8, equispaced | 729 | 112 | 4.7e-5 | 0.04 |
| ball p = 8, GLL | 729 | 1.11 | 3.8e-5 | 3.45 |
| quadrilateral box p = 7, GLL | 64 | 1.07 | 1.1e-5 | 1.03 |
| f64: hexahedron box p = 8, GLL | 729 | 1.11 | 1.2e-12 | 1.16 |

Nine blocks fail, all GLL at p ≥ 6 (hexahedra, the ball, quadrilaterals at p = 7–8; f64 once). At p = 8 the equispaced and GLL hexahedra have the same J error, 4.7e-5, although A differs by a factor of 100. The error follows the number of nodes summed, not A. This is the textbook bound of a computed sum of n products, γ_n Σ_k |x_k ∂φ_k| with γ_n ≈ n · u. A captures Σ_k |∂φ_k| but not n. For equispaced nodes A grows fast enough to hide that, but for GLL nodes it barely grows.

A second factor is outside A as well. The kernels read coordinates relative to the grid's origin (the centre of the bounding box), so the error of J is relative to |x − o|, while J is of the size of the cell. Let R be max |x − o| over the block's nodes divided by the block's largest |J| entry. On the accuracy meshes R is 0.85–2.2, so ADR-0042 item 2's premise holds there. Item 8's large case breaks it: 16k + 3 cells of `square_box(Quadrilateral, 129, p)` and `cube_box(Hexahedron, 26, p)` have R ≈ 65 and 13. That is the Metal failure: the Newton residual x − X(ξ) is formed from coordinates near 0.5 for cells of size 1/129, so f32 determines ξ only to about 10^−5–10^−4, and a different vector size changes the fused multiply-adds by that much. The CPU runtime compares bitwise and runs the same operations at every vector size, so it passes.

### 3. The pull-back's C′ cannot fit the large case's page limit

The pull-back uploads C′ once per block: nodes² values plus the runtime zero and one and the origin, padded. For the f32 hexahedron at p = 6 that is 343² + 5 values, padded to 117,664, or 470,656 bytes. That is the failing allocation exactly. The large case's limit is a third of the J field: 16,400 padded cells · 9 · 4 B / 3 = 196,800 bytes. p ≤ 5 fits (p = 5: 186,640 bytes). At p = 8, C′ takes 2.1 MB in f32 and 4.3 MB in f64. Item 8's lower bound names the tabulation and an E-vector tile but not C′, which is one array per block and cannot be chunked. Devices allow allocations of hundreds of MB, so this is the test's setting, not a device limit. The sum-factorised pull-back (T7b–T8) has no C′ for quadrilaterals and hexahedra. `loops` (T6b) and the simplex strategies keep it.

## Options

1. **Keep the design as written.** The ball fails bullet 1 of the acceptance at p = 8 under every strategy, on every device, and so do the GLL hexahedra from p = 6 and the large case at hexahedron p ≥ 6. Not chosen.
2. **Confine the Newton iterates, scale by n and R, and size the page limit above C′ (proposed).**
   - **Pull-back:** after each step, clamp ξ_d to [−m, 1 + m] with m = 0.1 (every reference cell lies in [0, 1]^TD). The step that `last_step` records and the freeze test use is the unclamped Δ, so a point that cannot be reached inside the box stays unconverged.
   - **Tolerances:** for J and the quantities derived from it (det J, K, normals, facet normals) and for the pull-back, the scale gains the factors max(1, n / n_ref) and max(1, R). n is the element's number of nodes and n_ref that of the reference element (p = 4 equispaced, p = 3 for pyramids). X keeps its factor: its error is relative to |x| itself.
   - **Page limit:** the large case's `max_page_size` becomes at least the largest per-block table the calls upload (C′, the tabulation, an E-vector tile). The E-vector must still span at least three chunks. J must span three wherever that limit allows, and the test states the cases where it does not.
3. **As option 2, but with factor max(A / A_ref, n / n_ref) instead of the product (A / A_ref) · (n / n_ref).** This is tighter for equispaced nodes, whose A already exceeds n / n_ref, and the same for GLL. It is not the error bound's form, only a fit of the data.
4. **Fix the numerics instead of the rules.**
   - **Ways to do it:**
     - sum J in coordinates relative to node 0 of each cell, which removes R;
     - use pairwise or compensated sums, which remove most of n;
     - use a damped or trust-region Newton instead of the clamp.
   - **Cost:** each changes `direct`'s output for every point, against item 11, and costs cycles in every strategy.
   - **Status:** not chosen. The numbers now are at the expected rounding level, and the tests should say so rather than demand more than f32 gives.

## Proposed decision

Option 2.

**Pull-back.** The clamp turns a wrong root into either the right root or a visible non-convergence. A margin of m = 0.1 keeps a small extrapolation for points just outside the cell. It was measured with a prototype of `pull_back_kernel` on the ball, every Gauss point of degree 2p:

| ball, f32 | margin none | m = 0 | m = 0.1 | m = 0.25 |
| --- | --- | --- | --- | --- |
| p = 8 equispaced: worst error (tolerance 4.24e-2) | 1.335 | 5.2e-4 | 5.2e-4 | 1.254 |
| p = 8 GLL: worst error (tolerance 9.2e-3) | 6.431 | 3.9e-4 | 3.9e-4 | 0.270 |
| p = 7, both: worst error | 4.2e-4, 2.7e-4 | the same | the same | the same |
| f64 p = 8 equispaced | 1.6e-11 | the same | the same | the same |

- **How it fixes the failures:** with m ≤ 0.1, every point of both p = 8 balls converges to its own preimage. With m = 0.25, the spurious roots stay within reach.
- **No effect elsewhere:** p = 7 and f64 are unchanged, because no iterate left the box.
- **Bitwise:** a lane whose iterates never leave the box computes exactly the same values, since the clamp is two selects that keep the value. `direct` stays bitwise M9's for every such point and differs only where M9 left the box. Item 11's rule ("`direct` bitwise M9's") is amended to that.
- **Cost:** two selects per coordinate per iteration, next to an iteration that evaluates the whole basis.
- **Unconverged points:** 8–11 of 5,832 points stay above the f32 step tolerance with or without the clamp, and 2 in f64. They lie near the ball's singular edges (ADR-0044), and their ξ error is within the tolerance.
- **Simplices and pyramids:** the box is larger than the reference cell, so a spurious root inside the box but outside the cell remains possible there. The tests show none at p ≤ 8 (pyramids p ≤ 3). The pyramid basis keeps its pole at ζ = 1 (ADR-0039, item 15), which the box contains, as now.

**Tolerances.** The product (A / A_ref) · (n / n_ref) is the γ_n bound normalised at p = 4, and R is the conditioning of coordinates relative to the origin. Under the product, the worst block of the 294 measured is at 0.74 of its tolerance (GLL hexahedron box at p = 6), and none fails. p ≤ 4 is unchanged except through R (at most 2.2 on the accuracy meshes). The large invariance case is the only place R is large.

**Page limit.** Item 8's purpose is chunked fields and E-vectors, not a table the design uploads whole. Raising the limit above that table keeps that purpose for the E-vector at every degree and for J where the sizes allow.

## Consequences

- **Text to change in docs/design.md:**
  - "Pull-back", the iteration bullet: after "Δ = K(ξ)(x − X(ξ))", add "each iterate is clamped to [−0.1, 1.1]^TD, the reference cell's box (every reference cell lies in [0, 1]^TD) widened by 0.1, while the freeze test and `last_step` use the unclamped step; a point whose preimage lies outside that box does not converge".
  - "Tolerances", the degree-scaling bullet:
    - "both tolerances scale by max(1, A / A_ref)" becomes "both tolerances scale by max(1, A / A_ref), times max(1, n / n_ref) and max(1, R) for J, the quantities derived from it and the pull-back";
    - it adds "n is the element's number of nodes and n_ref that of the p = 4 equispaced element (p = 3 for pyramids); R is the largest |x − o| over the block's nodes, o the grid's origin, divided by the block's largest entry of J".
  - ADR-0042's and ADR-0044's decision records: "Amended by ADR-0046".
- **Public API changes:** none. `PullBackOptions` is unchanged. The `PullBack` documentation gains the clamp, and `last_step` keeps its meaning.
- **Code:**
  - **Pull-back clamp:** `gridcl-geometry/src/pull_back.rs` (`direct`), and every pull-back strategy after it (T6b's `loops`, T7b–T8's `sumfact`, T9–T10's `shared` and `cooperative`).
  - **`gridcl_testing::reference`:** the scales gain n / n_ref, and R from the block's nodes and J. The callers are the same as ADR-0044's (`gridcl/tests/{common/mod.rs, high_order.rs, high_order_invariance.rs}`, `gridcl-testing/src/{parallel.rs, meshes/family.rs}`, `gridcl-testing/tests/reference.rs`, `gridcl-bench/src/families/degree.rs`).
  - **`high_order_invariance.rs`, the large case:** `max_page_size` is at least `EVector::coefficients_bytes` and the other per-block tables. The E-vector must span three chunks at every degree. The J assertion holds where the limit allows, with the exceptions named in the test.
- **Acceptance tests affected:**
  - `high_order::{factors_match_host_reference, pull_back_recovers_reference_points}`: J, det J, K, normals, facet normals and ξ get the new factors.
  - `high_order_invariance::vector_size_invariance`: the same factors on GPUs, and the page limit.
  - The parallel-invariance checks (rule 9) get the same factors on GPUs. They are bitwise on the CPU runtime and unchanged there.
  - X keeps its tolerance.
- **The baseline:** `direct`'s pull-back changes only on points whose path left the cell's box. The T3 sweep meshes are affine boxes and pull back in one or two steps from the centroid, so their records are unaffected; rule 11's paired comparison times both sides with the clamp.
- **Plan (M9h):** the clamp lands as T6b's first step, with `loops`, in both kernels; the test-support changes land with it, since T6b's check set runs the affected tests. T6a's Deviations rows record the findings.
- **Later milestones affected:** none known. Point location and interpolation between meshes (later users of the pull-back) get either a root inside the widened cell or an unconverged `last_step`, never a root far outside.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
