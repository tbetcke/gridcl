# ADR-0049: The tensor pull-back contracts modal coefficients, not nodal values

- **Status:** Proposed
- **Date:** 2026-11-12
- **Raised in:** M9h T8 (Metal run of the kernel-task check set)
- **Design sections affected:** "Pull-back" (the strategies bullet); "Handing this design to Claude Code", the copy list (and CLAUDE.md). Amends ADR-0042, items 1 and 5.

## Context

"Pull-back" (ADR-0042) says: "On quadrilaterals and hexahedra, the sum-factorised pull-back evaluates the 1D Lagrange bases at ξ (barycentric form) and contracts the E-vector one dimension at a time, with no expansion coefficients; prisms combine the triangle's coefficients with the 1D interval basis." ADR-0042 item 1 adds: "The tensor pull-back of item 5 has none [no set-up]". Item 5 lists its uploads: the 1D nodes and barycentric weights, and the prism's triangle coefficient factor.

T8 implemented exactly that (`gridcl-geometry/src/sumfact/pull_back.rs`, uncommitted). Each Newton step evaluates the 1D Lagrange bases φ_a(ξ_d) and φ_a'(ξ_d) from the nodes and barycentric weights, as prefix and suffix products w_a Π_{b≠a}(ξ_d − t_b), so ξ on a node needs no branch. It then contracts the nodal E-vector one direction at a time. Results:

- In f64 the kernel agrees with `loops` to about 10^−9 at every iterate.
- Release builds on the CPU runtime pass every test.
- Locust CUDA passes every test (job 20261112192106-113324, f32 and f64, p = 1–8, node idle before and after).
- The new `forced_sumfact_pull_back_agrees_with_loops` passes at p = 3 on every tensor cell.

## Problem

On Metal (f32), `high_order::pull_back_recovers_reference_points` under `GRIDCL_GEOM_STRATEGY=sumfact` fails for the equispaced ball at p = 8: ξ has relative error 1.10 against the rule-2 tolerance 0.248 (scale 24,757). A diagnostic pulled back the 5,832 Gauss points of degree 16 of `ball(0, 8)`:

| f32 run | points off by > 10^−2 | worst ξ error |
| --- | --- | --- |
| `loops`, `direct` (CPU and Metal) | 0 | 5e-4 |
| nodal `sumfact` as specified (Metal) | 12, on a corner of the clamp box | 1.08 |
| nodal `sumfact` as specified (CPU) | 0 (the CPU run passes) | – |
| nodal `sumfact`, each line minus its linear interpolant (CPU and Metal) | 16 | 0.18 |
| every strategy in f64 | 0 | 4.6e-11 |

The worst point (cell 2, preimage (0.984, 0.016, 0.918)) shows the mechanism:

- **Iterate 1:** every strategy steps from the centroid to the box boundary, (1.1, −0.1, 1.08).
- **Iterate 2:**
  - f64 (both strategies): (0.983, 0.017, 0.917).
  - f32 `loops`: (0.985, −0.049, 0.958), which converges by iterate 4.
  - f32 nodal `sumfact`: (−0.1, 1.1, −0.1), step 3.8, where it stays.

The cause is the conditioning of the nodal form outside the reference cell. A rounding error δ in a nodal value, or in an intermediate sum of a direction-by-direction contraction, reaches X and J multiplied by the Lebesgue function Λ(ξ) = Σ_a |φ_a(ξ)| of each direction. At ADR-0046's box margins, Λ grows fast with the degree:

| p | equispaced: Λ(−0.1) | equispaced: max Λ on [0, 1] | GLL: Λ(−0.1) | GLL: max Λ on [0, 1] |
| --- | --- | --- | --- | --- |
| 4 | 8 | 2.2 | 6 | 1.6 |
| 6 | 45 | 4.5 | 22 | 1.9 |
| 8 | 296 | 10.9 | 76 | 2.1 |

At a corner of the box a hexahedron multiplies them: 296³ ≈ 2.6·10⁷ for equispaced p = 8. Times f32's 6·10^−8 this is O(1), so X and J at that iterate are noise. Subtracting each line's linear part (the last table row) shrinks the data but not the amplification of the contraction's own rounding.

`loops` and `direct` evaluate in the Legendre basis instead: Σ_k a_k φ̃_k(ξ) with a = C′x. For a smooth map the a_k decay, and φ̃_k(ξ) stays moderate just outside [0, 1]. The CPU and CUDA passes of the nodal kernel come from different rounding sending these iterates elsewhere: luck, not accuracy.

## Options

1. **Keep the design as written.** The nodal tensor pull-back stays. f32 iterates outside the cell are unreliable at p = 8 on equispaced nodes (and Λ grows quickly from p = 6), so `pull_back_recovers_reference_points` must exempt f32 `sumfact` at high degree, and the policy must never select it there. This weakens an acceptance check (every strategy recovers the reference points), and it passes or fails by rounding luck per backend. Not chosen.
2. **Modal tensor pull-back with a sum-factorised set-up (proposed).** Per cell, a set-up kernel forms the coefficients over the tensor Legendre basis:
   - quadrilaterals and hexahedra: a = (C₁′ ⊗ C₁′ ⊗ C₁′) x, by sum factorisation;
   - prisms: (C′_tri ⊗ C₁′) x;
   - C₁′ is the 1D interval element's C′.

   Mathematically a equals `loops`' C′x (Finding 3: C′ is the tensor product under the node permutation). The set-up costs 3 (p + 1)⁴ products per component on a hexahedron, against (p + 1)⁶ for `loops`' set-up. It writes a scratch buffer of `loops`' size (nodes × gdim per cell, one E-vector chunk range); three buffers.

   Each Newton step evaluates the 1D Legendre values and derivatives at ξ_d by the recurrences of the `loops` evaluator, O(p) each, and contracts a one direction at a time: 2 (p + 1)³ products per component, as the nodal kernel. On prisms it uses the Dubiner set times Legendre in z; five buffers. Conditioning is that of `loops` and `direct`.

   Rule 1 holds:
   - At the Gauss rule of degree 2p a hexahedron has (p + 1)³ points, so the set-up per point is 3 (p + 1) G products, a p = 8 to 4 ratio of 1.8 (bound 3).
   - The step's leading cost grows as (9/5)³ ≈ 5.8, within the 6.4 bound.

   Uploads per geometry block: C₁′ (and C′_tri), the node permutation, the runtime zero and one, and the origin. The copy list's "coefficient factors of a sum-factorised pull-back" already names them; the 1D nodes and barycentric weights are no longer needed.
3. **Nodal pull-back with a narrower box for `sumfact`** (iterates clamped to [0, 1]^TD, where Λ ≤ 10.9 per direction). The strategies would then converge differently for points near or outside the cell boundary, against ADR-0046 (one box, the same outside behaviour for every strategy) and the "agrees with `loops`" test. Not chosen.
4. **Nodal data converted to modal coefficients at every step** (no set-up). It costs (p + 1)⁴ products per component and step, more than `loops`' 4 (p + 1)³ from p = 4. Not chosen.

## Proposed decision

Option 2. The tensor pull-back keeps its sum factorisation and its tensor-cell scope. Its data become modal coefficients formed once per cell by a sum-factorised set-up, instead of nodal values contracted with barycentric Lagrange bases at ξ. That gives the strategy the conditioning of `loops` and `direct` at the iterates ADR-0046's box allows, and the "every strategy" accuracy test needs no exemption. Rule 1's set-up bound, written for strategies that have one, applies (ratio 1.8).

## Consequences

- **Text to change in docs/design.md:**
  - **"Pull-back", the strategies bullet,** becomes: "The pull-back runs the same strategies. On quadrilaterals and hexahedra, the sum-factorised pull-back forms each cell's coefficients over the tensor Legendre basis once, a = (C₁′ ⊗ C₁′ ⊗ C₁′)x by sum factorisation, then evaluates the 1D Legendre polynomials at ξ by their recurrences and contracts a one dimension at a time; prisms use C′_tri ⊗ C₁′ with the Dubiner set (ADR-0042, ADR-0049)."
  - **The copy list** in "Handing this design to Claude Code" (and CLAUDE.md): "or of the 1D nodes, barycentric weights and coefficient factors of a sum-factorised pull-back" becomes "or of the coefficient factors and node permutation of a sum-factorised pull-back".
- **ADR-0042:**
  - Item 1's "The tensor pull-back of item 5 has none" becomes "The tensor pull-back's set-up is sum-factorised (ADR-0049)".
  - Item 5's pull-back uploads become C₁′, C′_tri and the node permutation.
- **The M9h plan:**
  - T8's goal is rewritten to option 2.
  - The risk note on the barycentric exact-node branch is dropped.
- **Public API changes:** none.
- **Acceptance tests affected:** none. `pull_back_recovers_reference_points` and the invariance tests stay as they are, and T8's tests stay; `forced_sumfact_pull_back_agrees_with_loops` compares within the tolerances.
- **Later milestones and tasks affected:**
  - T9 (`shared`) stages C′ "or its factors", which includes C₁′.
  - T10b (`cooperative`) splits the step's contraction, as planned.
  - T11's sweep times the new set-up as rule 1 defines it.

## Decision record

<Filled by the maintainer.>
