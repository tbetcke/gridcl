# ADR-0054: M3's L-vector reference comparison under a degree policy that reorders sums

- **Status:** Proposed
- **Date:** 2026-11-14
- **Raised in:** M9h T11b (decision rule, policy and report)
- **Design sections affected:** Roadmap M3, acceptance ("factor kernels on E-vectors agree with a scalar L-vector reference kernel … bitwise on the CPU runtime"); "Testing and validation" (vector-size invariance: "comparisons with one are bitwise up to the sign of zero on the CPU runtime (ADR-0013)").

## Context

M3's acceptance: "factor kernels on E-vectors agree with a scalar L-vector reference kernel at every vector size from 1 up to 16 that the device offers and the tile holds, and at tile widths 1 and 16, bitwise on the CPU runtime, with +0 and −0 equal (ADR-0013), and within the tolerances on GPUs." `gridcl/tests/factor_invariance.rs` checks this at p = 2 (Gauss rule of degree 3) on every cell type. The reference sums over nodes in node order, as M9's `direct` kernels and T6a's `loops` do.

T7a's sum factorisation contracts one direction at a time, so its sums run in another order and are not bitwise equal to the reference. At T7b (Deviations log, 2026-11-12) the maintainer ruled: compare within the tolerances where `GRIDCL_GEOM_STRATEGY` forces a strategy that reorders sums (`common::environment_reorders_sums`), and "if T11b's policy selects such a strategy for a family at M3's p = 2 on the CPU, the default run meets the same comparison: an ADR then."

## Problem

The decision rule on T11a's data selects `sumfact` on the CPU classes at p ≤ 2 where it is much faster than `direct`:

- CpuSimd128 f32: points on quadrilaterals from p = 2, on hexahedra and prisms from p = 1; J, det J and K on hexahedra and prisms from p = 2;
- CpuSimd128 f64: points on quadrilaterals and hexahedra from p = 2, on prisms from p = 1;
- CpuSimd512: points, Jacobians and facet normals on hexahedra and prisms from p = 2 (some from p = 3).

For example, hexahedron points at p = 2, f32, 10^6 points: the Mac mini takes 0.976 ns per point with `direct` and 0.222 ns with `sumfact`.

With that policy in the tree, `factor_invariance::factor_kernels_match_lvector_reference` fails on the CPU runtime (release and debug). In the release run of the bitwise-comparing modules, every other test that finished passed: `kernel_invariance`, `manifold_factors`, `pull_back`, `determinism`, `high_order::pull_back_recovers_reference_points`, `high_order_invariance::{tile_invariance, two_runs_are_bitwise_identical}`, `parallel_threads::{determinism, high_order_parallel_invariance}`, `precision`, `unit_volume` and `sphere_area`. `high_order_invariance::vector_size_invariance` did not finish within the run's 90-minute limit. Determinism, vector-size and parallel invariance compare a strategy with itself, so they stay bitwise.

## Options

1. **Keep the design as written.** The decision rule would never select a reordering strategy where M3's test runs (p ≤ 2 on the CPU classes). That gives up `sumfact` at low degree on CPUs, up to 4.4 times on hexahedron points at p = 2, and adds an exception to rule 12 for a test's sake.
2. **Bitwise where the strategy that ran sums in the reference's order, within the tolerances elsewhere, plus a forced `direct` pass (proposed).** `factor_invariance` runs twice. (a) With `direct` forced through `testing::with_strategy` for every family: bitwise on the CPU runtime, as now. This keeps M3's check of M9's kernels against the reference. (b) Under the policy, with `testing::record_strategies`: bitwise where every launch ran `direct` or `loops`, and within the tolerances (10^−12 / 10^−5) where one ran `sumfact`, `shared` (whose tensor variant is sum-factorised) or `cooperative`. `common::assert_matches` takes that decision from the recorded strategies instead of from the environment variable. The vector-size and tile sweep is unchanged.
3. **A sum-factorised L-vector reference.** A second scalar reference kernel that sums in `sumfact`'s order, compared bitwise with `sumfact`. More reference code, and still none for `cooperative`'s tree order.

## Proposed decision

Option 2. M3's bitwise check keeps its purpose, catching vector-size and tile defects in M9's kernels against an independently written reference, through the forced `direct` pass. The policy's own kernels are checked against the same reference within the tolerances of "Testing and validation", as the GPU comparison already is. Determinism and the vector-size, tile and parallel invariance tests stay bitwise on the CPU runtime, because they compare a strategy with itself.

## Consequences

- Text to change in docs/design.md:
  - Roadmap M3 acceptance, the factor-kernel bullet: after "bitwise on the CPU runtime, with +0 and −0 equal (ADR-0013)," add "for the direct kernels and for strategies that sum over nodes in node order; within the tolerances for sum-factorised and cooperative kernels (ADR-0054)".
  - "Testing and validation", vector-size invariance, "A separately written reference kernel may fuse multiply-adds differently; comparisons with one are bitwise up to the sign of zero on the CPU runtime (ADR-0013)": add "where the kernel sums in the reference's order; sum-factorised and cooperative kernels are compared within the tolerances (ADR-0054)".
- Public API changes: none (`testing::with_strategy` and `record_strategies` exist, ADR-0042 item 3).
- Acceptance tests affected: `gridcl/tests/factor_invariance.rs` (the two passes); `gridcl/tests/common/mod.rs` (`assert_matches` decides from the recorded strategies; `environment_reorders_sums` becomes a special case of it). `pull_back::*`, which also uses `assert_matches`, follows the same rule.
- Plan: M9h T11b (with the policy) or T12 makes the change; a Deviations-log row cites this ADR.
- Later milestones affected: none.

## Decision record

<Filled by the maintainer.>
