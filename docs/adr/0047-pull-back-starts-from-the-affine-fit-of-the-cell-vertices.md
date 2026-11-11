# ADR-0047: The pull-back starts from the affine fit of the cell's vertices

- **Status:** Proposed
- **Date:** 2026-11-11
- **Raised in:** M9h, after T6b (the maintainer asked for a better initial guess as a small task before T7a and T8)
- **Design sections affected:** "Pull-back" (the iteration bullet). Amends ADR-0042, item 11 (as ADR-0046 amended it).

## Context

"Pull-back" says: "Iteration starts at the reference cell's vertex centroid with Δ = K(ξ)(x − X(ξ)) … Each iterate is clamped to [−0.1, 1.1]^TD … The loop always runs `max_iterations`, with no early exit". `PullBackOptions::default()` runs 16 iterations.

ADR-0046 added the clamp after the f32 pull-back converged to roots outside the cell at p = 8. With it, `high_order::pull_back_recovers_reference_points` passes at p = 1–8 on the CPU runtime, Metal and CUDA (M9h T6b). The maintainer then asked how other libraries start the iteration:

- MFEM's `InverseElementTransformation` starts from the closest point of a reference lattice or the closest physical node, and projects iterates onto the element.
- deal.II's `MappingQ` starts from the inverse of the cell's affine (or quadratic) approximation and damps the Newton steps.
- gslib's `findpts` runs a trust-region Newton constrained to the element.
- DOLFINx starts at the midpoint, as gridcl does now.

The question here is whether a better start belongs in M9h, before T8–T10b add three more pull-back strategies that share the start and `newton_update`.

## Problem

A prototype of the `loops` pull-back (M9h T6b's kernels, not committed) compared three starts.

**The starts:**

- **Centroid:** the current start.
- **Affine fit:** ξ₀ = ξ̄ + K_A (x − x̄), where x̄ is the mean of the cell's vertex coordinates and A the least-squares affine fit of the vertices. K_A is A's pseudo-inverse.
- **Closest node:** the reference point of the cell's node nearest to x (MFEM's "closest physical node").

**The measurement:** every Gauss point of degree 2p was pulled back on the CPU runtime, release build, with `max_iterations` = 2, 3, 4, 6, 8 and 16. The counts below are points whose last step exceeds the default tolerance; "error" is the worst |ξ − ξ_exact|∞. Columns give the iterations needed until no point is unconverged.

| Mesh | Centroid f32 | Centroid f64 | Affine fit f32 | Affine fit f64 | Closest node f32 | Closest node f64 |
| --- | --- | --- | --- | --- | --- | --- |
| hexahedron box p = 8, perturbed, equispaced and GLL | 4 | 6 | 3 | 4 | 3 | 4 |
| sphere p = 8 (triangles in 3D) | 6 | 6 | 3 | 6 | 3 | 6 |
| triangle box p = 8, tetrahedron and prism boxes p = 6, perturbed | 4 | 6 | 3 | 4 | 3 | 4 (tetrahedra 6) |

The ball (8 hexahedra with singular edges, ADR-0044) needs more iterations from every start:

| ball, unconverged points / error | Centroid | Affine fit | Closest node |
| --- | --- | --- | --- |
| p = 8 equispaced, f32, 8 iterations | 73 / 1.8e-3 | 19 / 5.3e-4 | 181 / 1.9e-1 |
| p = 8 equispaced, f32, 16 iterations | 9 / 5.1e-4 | 8 / 4.6e-4 | 13 / **1.9e-1** |
| p = 8 GLL, f32, 16 iterations | 8 / 3.9e-4 | 5 / 4.5e-4 | 6 / 4.4e-4 |
| p = 8 equispaced, f64, 8 iterations | 328 / 1.6e-3 | 32 / 1.2e-9 | 304 / 9.9e-3 |
| p = 8 equispaced, f64, 16 iterations | 2 / 1.6e-11 | 1 / 1.6e-11 | 0 / 1.6e-11 |
| p = 6 equispaced, f32, 6 iterations | 232 / 8.8e-3 | 13 / 2.0e-4 | 208 / 1.6e-2 |

What the tables show:

- **The affine fit is the best start.** It saves one to two iterations on every mesh, and on the ball it leaves 4–18 times fewer points unconverged at 6–8 iterations.
- **It is exact on affine cells.** There the first step is already below the tolerance.
- **The closest node fails at p = 8.** On the equispaced ball it starts some points in the basin of a wrong root inside the box (ξ off by 0.19 after 16 iterations). This is the failure ADR-0046 fixed for the centroid, back again.
- **At the default 16 iterations the gain is small.** The kernel runs every iteration, so the saved iterations do not save time. The points that stay unconverged lie at the ball's singular edges and do not converge from any start: 9 → 8 in f32 and 2 → 1 in f64.
- **What the gain buys:**
  - The pull-back converges with fewer iterations wherever a caller lowers `max_iterations`.
  - It reaches its fixed point earlier on strongly curved cells.
  - A start inside the right basin relies less on the clamp.

**Cost:** per point, about nv · GD · (TD + 1) multiply-adds and one TD × TD pseudo-inverse (about 120 operations for a hexahedron in 3D).

- At p = 8 that is under 2% of one Newton step of the `loops` evaluator.
- At p = 1 it is about one step, so a 16-iteration pull-back costs about 6% more.

The start needs each cell's vertex coordinates, which are the first nv nodes of the E-vector (gridcl-refcell numbers vertices first), and constant weights per cell type.

## Options

1. **Keep the design as written** (centroid start). Nothing changes, and the pull-back keeps passing its acceptance tests at p = 1–8. The iterations a caller can save stay unused. Each later strategy (T8, T9, T10b) inherits the centroid, and a change after T11a would invalidate the pull-back sweeps and the policy.
2. **Start from the affine fit of the vertices, clamped to the box (proposed).**
   - ξ₀ = clamp(ξ̄ + K_A (x − x̄)) to [−0.1, 1.1]^TD, with A = Σ_v x_v w_vᵀ and x̄ = Σ_v x_v / nv over the cell's vertices v.
   - The weights are W = Ξ (ΞᵀΞ)⁻¹, with Ξ the reference vertices minus their centroid ξ̄. They are constants of the cell type, built at comptime; a weight of 0 adds nothing and ±1 adds or subtracts, so no float constant 0 or 1 reaches the kernel (C16).
   - **Where it runs:**
     - `direct`: in the kernel, per cell, next to the expansion coefficients.
     - `loops`: in the set-up kernel, which writes x̄ and K_A to its scratch (TD · GD + GD more rows per cell).
     - Later strategies: wherever they read the E-vector.
   - The rest of the iteration, the clamp, the freeze test and `last_step` are unchanged.
3. **Start from the closest node** (MFEM's option). Rejected by the measurement: wrong roots at p = 8 on equispaced nodes. It also costs nb · GD operations per point, more than the affine fit from p = 2.
4. **As option 2, and lower the default `max_iterations`** (for example to 8). This would turn the saved iterations into time: about half the pull-back's cost. But on the ball at p = 8, 19 (f32) and 32 (f64) points stay unconverged at 8 iterations, against 8 and 1 at 16. It changes a public default and its documented example. Not proposed here: rule 12's sweep (T11a) can measure it first.
5. **Early exit once every lane of a vector is frozen.** This would keep the results identical (frozen lanes change nothing) and save time on easy cells. But it branches on lane values, which CLAUDE.md forbids. Not proposed.

## Proposed decision

Option 2. The affine fit is the cheapest start that the measurement shows to be better on every mesh, in f32 and f64. It is exact on affine cells and needs no branches or new buffers. Doing it now, before T7a, puts it into all five pull-back strategies once; after T11a it would invalidate the pull-back sweeps and the degree policy.

Its value at the default 16 iterations is modest: at p = 8, one unconverged point fewer on the ball. If the maintainer prefers to spend no task on that, option 1 is a sound choice, and the idea can wait for point location across meshes, where gslib-style robustness pays more.

## Consequences

- **Text to change in docs/design.md:** "Pull-back", the iteration bullet. "Iteration starts at the reference cell's vertex centroid with Δ = …" becomes:

  > Iteration starts at the preimage of x under the least-squares affine fit of the cell's vertex coordinates, clamped to the box below (exact on affine cells), with Δ = …
- **Public API changes:** none. `PullBackOptions` and its default of 16 iterations stay. The `pull_back_block` and `Mesh::pull_back` documentation names the start.
- **Code:**
  - `gridcl-geometry/src/pull_back.rs`: the start in `pull_back_kernel` (`direct`) and in `pull_back_setup_kernel` / `pull_back_loops_kernel` (`loops`), with the comptime vertex weights per cell type.
  - Later strategies (T8, T9, T10b) start the same way.
- **ADR-0042, item 11 (as amended by ADR-0046):** `direct` is no longer bitwise M9's for points whose iterates differ from M9's. That is now every point of a non-affine cell, not only those that left the box, though the converged results agree to the tolerance. The paired comparison of rule 11 (policy against `direct`, both with the new start) is unaffected. The drift against T3's baseline stays a report: `direct` costs about one iteration more at p = 1 (about 6% of a 16-iteration pull-back) and nothing measurable at p = 8.
- **Acceptance tests affected:** none change. The pull-back tests pass with either start. New unit or integration tests:
  - on affine cells (every cell type), one iteration converges from the affine start for points inside the box;
  - the ball at p = 8 converges in fewer iterations than from the centroid (a regression guard on the measurement above, at 8 iterations).
- **Plan (M9h):** a new task T6c "Pull-back start" after T6b and before T7a, in `gridcl-geometry/src/pull_back.rs`, with the kernel-task check set and a Locust run.
- **Later milestones affected:** none known. Point location and interpolation between meshes inherit the start.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
