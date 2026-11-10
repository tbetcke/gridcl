# ADR-0044: Tolerances of K and the pull-back scale with the conditioning of J

- **Status:** Accepted
- **Date:** 2026-11-10
- **Raised in:** M9h T4 (the kernel-task check set)
- **Design sections affected:** "Testing and validation", **Tolerances** (the degree-scaling bullet). Amends ADR-0042, item 2.

## Context

ADR-0042, item 2 scales the tolerances of the high-order tests: 10^−5 · max(1, A / A_ref) in f32 and 10^−12 · max(1, A / A_ref) in f64. A is the element's amplification factor:

- for X, Σ_j |ψ_j|;
- for "J and the quantities derived from it", max_d Σ_j |∂_d ψ_j|;
- for a pull-back through coefficients, Σ_k |φ_k| Σ_j |C_kj| "times κ(J) at the point".

A_ref is "the same factor for the p = 4 equispaced element of that cell type at the same points". docs/design.md, "Tolerances", says the same, and adds: "The factor is at most 1.01 for p ≤ 4".

The implementation is in `gridcl_testing::reference`:

- `tolerance_scale` gives K the factor of J.
- `pull_back_scale` computes κ(J) on the mesh under test. It multiplies κ(J) into both A and A_ref, so κ cancels from the ratio.

The kernel-task check set of T4 ran gridcl's tests with `GRIDCL_GEOM_STRATEGY=direct`. That setting runs only `direct`, so it reached checks of `high_order.rs` (T0b) that had never run before:

- **K:** `factors_match_host_reference` fails in f32 on the ball at p = 3: K has relative error 1.30e-5 against 10^−5, at scale 1.
- **Pull-back:** `pull_back_recovers_reference_points` fails in f32 on the ball at p = 4: ξ has 1.52e-5 against 10^−5, at scale 1.

Without the variable, both tests fail earlier, at their forced `loops` runs (`NotYetImplemented` until T6a and T6b).

## Problem

The test mesh `gridcl_testing::ball(0, p)` maps the cube [−1, 1]³ onto the unit ball by a smooth map. As its documentation says, det J of that map "vanishes only on the cube's edges, as for every C¹ map of the cube onto the ball". As p grows, the Gauss points of degree 2p approach the cells' outer edges, so J's conditioning worsens without bound. The table gives the Frobenius condition number κ_F(J) = ‖J‖_F ‖K‖_F, maximised over cells and points, computed on the host in f64:

| p | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ball(0, p)` | 3.7 | 13.6 | 35.1 | 81.6 | 161 | 286 | 469 | 727 |
| `ball(1, p)` | 6.6 | 43.3 | 137 | 326 | 641 | 1,129 | 1,845 | 2,856 |

κ_F ≥ TD, with equality when J is a scaled rotation. Refining the ball makes it worse, because the cells then lie closer to the singular edges.

The f32 errors on `ball(0, p)` on the CPU runtime are relative to the f64 reference, per quantity over the block (ADR-0042, item 2). The last two columns isolate where the K error comes from:

| p | J | K (kernel) | K = exact inverse of the device's J | K = exact inverse of J_ref rounded to f32 |
| --- | --- | --- | --- | --- |
| 1 | 4.5e-8 | 1.3e-7 | 7.5e-8 | 3.0e-8 |
| 2 | 2.7e-7 | 1.3e-6 | 1.4e-6 | 4.0e-8 |
| 3 | 1.0e-6 | 1.30e-5 | 1.29e-5 | 5.9e-7 |
| 4 | 2.1e-6 | 6.20e-5 | 6.22e-5 | 1.3e-6 |

- **J** is within its tolerance at every degree.
- **K's error is J's error carried through the inverse.** Inverting the device's J exactly in f64 gives the same error as the kernel, so the K kernel adds none of its own.
- **The growth is the textbook bound.** A relative error δ in J becomes up to κ₂(J) · δ in K. At p = 4 that is 2.1e-6 · κ₂ ≈ 6e-5.
- **ξ is the same case.** The pull-back's Newton step solves with J, and ξ fails the same way at p = 4.
- **The factor stays 1.** The tolerances leave the conditioning out for K, and cancel it for ξ. At p ≤ 4 both factors stay at 1 (`pull_back_scale` measures 1.00–1.01 on the ball), so the tests demand 10^−5 of a quantity whose f32 error is κ times larger.
- **Higher degrees fail too.** p = 4 fails as well (6.2e-5; the test stops at p = 3), and p ≥ 5 (release only) fails by more, since κ keeps growing.
- **Neither strategy nor device fixes this.** These are not kernel defects. No strategy of T6a–T10b can meet the tolerance either, and f64 has the same structure (10^−12 against κ · 10^−16 · A).

## Options

1. **Keep the design as written.** `factors_match_host_reference` and `pull_back_recovers_reference_points` fail in f32 on the ball from p = 3 under every strategy, on every device. Bullet 1 of the acceptance cannot be met. Not chosen.
2. **Scale K and ξ by the conditioning of J, κ_F(J) / TD, pointwise.**
   - **K:** the factor becomes the largest over the block's cells and points of (κ_F(J) / TD) · A_J, divided by the largest A_ref over the points. A_J and A_ref are J's amplification factors.
   - **Pull-back:** the factor becomes the largest over cells and points of (κ_F(J) / TD) · A_C, divided by the largest A_C of the p = 4 equispaced element over the points. A_C is the coefficient factor. A_ref therefore no longer contains κ, so κ does not cancel.
   - **Both:** max(1, ·) as before. Since κ_F / TD ≥ 1, no tolerance gets tighter than now. On cells whose J is close to a scaled rotation the factor stays 1, as on the affine boxes of the degree sweep and the M9 tests.
   - **Ball, p = 3:** K's tolerance becomes 1.2e-4 (measured 1.3e-5).
   - **Ball, p = 4:** it becomes 2.7e-4 (measured 6.2e-5).
   - **Why κ_F rather than κ₂:** κ_F comes from J and K alone, with no singular values. Because κ₂ ≤ κ_F ≤ TD · κ₂, κ_F / TD can underestimate κ₂ by up to a factor of TD. The measured margins above are 4–9.
3. **Option 2 with κ₂(J) instead of κ_F(J) / TD.** κ₂ is exactly the factor of the bound. It needs the singular values of each J on the host: a symmetric eigenproblem of JᵀJ, at most 3 × 3. It costs more code in `gridcl_testing::reference` and gains at most a factor of TD in tightness. Not proposed, but it is the alternative if option 2's slack proves too large.
4. **Change the meshes, not the tolerance.**
   - **Ways to do it:** check K and ξ only on meshes with bounded κ, or replace the ball with a curved mesh whose map has no singular edges, for example a perturbed box with smooth curvature.
   - **Cost:** this changes bullet 1's test inputs (T0a), and the ball would then test only X, J and det J.
   - **What it loses:** the tests would no longer show how f32 behaves near degenerate cells, which users' meshes do reach.
   - **Status:** not chosen.

## Proposed decision

Option 2. The K and ξ errors are J's rounding error amplified by J's conditioning. The tolerance should state that bound instead of assuming well-shaped cells. Pointwise κ_F / TD leaves well-shaped meshes at the plain 10^−5 and 10^−12, and the tests keep their meshes. The design then reads consistently: "times κ(J)" for the pull-back now means what it says, because A_ref no longer cancels it.

## Consequences

- **Text to change in docs/design.md**, "Testing and validation", **Tolerances**, the bullet "Above degree 4 both tolerances scale…":
  - The bullet becomes "Above degree 4, or on distorted cells, both tolerances scale…".
  - It gains: "for K and the pull-back, A is multiplied by κ_F(J) / TD at the point (κ_F = ‖J‖_F ‖K‖_F, TD the topological dimension), and A_ref is the element's factor alone".
  - "The factor is at most 1.01 for p ≤ 4" gains "on cells with κ_F(J) = TD".
  - In ADR-0042's decision record, a line "Amended by ADR-0044 (the tolerances of K and the pull-back scale with κ_F(J) / TD)".
- **Public API changes:** none. Only test support changes:
  - **`gridcl_testing::reference`:** a K-specific scale, so `tolerance_scale` with `Amplification::Derivatives` plus a pointwise κ_F / TD over a block, and `pull_back_scale` without κ in A_ref. `Amplification`'s documentation is updated.
  - **The callers:**
    - `gridcl/tests/common/mod.rs` (`block_scale`);
    - `gridcl/tests/high_order.rs` (K's scale);
    - `gridcl/tests/high_order_invariance.rs` (`Scales`: the K and ξ comparisons of ADR-0042, item 8);
    - `gridcl-testing/src/{parallel.rs, meshes/family.rs}`, `gridcl-testing/tests/reference.rs`;
    - gridcl-bench's degree-sweep checks (`families/degree.rs`: `scale_derivatives` for Pinv; `pull_back_scale`).
- **Acceptance tests affected:**
  - **Changed tolerance:** `high_order::{factors_match_host_reference, pull_back_recovers_reference_points}` (K and ξ only) and the K and ξ comparisons of `high_order_invariance` and of the parallel-invariance tests.
  - **Unchanged:** X, J, det J, normals and facet normals keep their factors.
  - **The baseline:**
    - The T3 records store each check's error and tolerance under the old rule. The sweep meshes are affine boxes, so the new factor is the same or larger, and the stored ratios are upper bounds.
    - `baseline_covers_every_target_and_degree` reads no tolerance.
- **Plan (M9h):**
  - T0b's tests change with the next kernel task that runs them (T6a), which also moves `try_clone` out of `with_properties` in `vector_size_invariance`'s large case.
  - The Deviations log already records the check-set findings.
- **Later milestones affected:** none known. Every later test of K or the pull-back on curved meshes uses the same rule.

## Decision record

Accepted by the maintainer on 2026-11-10, as proposed (option 2, κ_F(J) / TD). Applied in `adr: 0044 apply`: docs/design.md, "Tolerances", and ADR-0042's decision record. The test-support changes in `gridcl_testing::reference` and its callers follow with T6a, the next task that runs the affected tests.

Amended by the maintainer on 2026-11-10, when the test support was written: the two maxima are taken separately, so the factor is max(1, A / A_ref) times the largest κ_F(J) / TD over the block's cells and points. The pointwise product of option 2 undershoots where the element's factor is small but κ is large. On the GLL ball at p = 4 it gives 11.3, though κ_F / TD reaches 27.2, and K's f32 error, 1.40e-4, exceeds that tolerance 1.24 times. The error comes from J's rounding error measured against the block's largest J, not from the factor at the same point. Measured ratios under separate maxima, f32 on the CPU runtime:

| Ball | p = 3 | p = 4 |
| --- | --- | --- |
| equispaced | 0.11 | 0.23 |
| GLL | 0.05 | 0.51 |

The alternative, the block condition number max ‖J‖_F · max ‖K‖_F / TD, was not chosen: it would also loosen the tolerance on graded meshes of well-shaped cells.
