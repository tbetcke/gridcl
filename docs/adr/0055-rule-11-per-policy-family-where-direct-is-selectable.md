# ADR-0055: Rule 11 compares policy families where `direct` is selectable, and counts consistent slowdowns

- **Status:** Proposed
- **Date:** 2026-11-14
- **Raised in:** M9h T11b (rule 11's paired runs)
- **Design sections affected:** Roadmap M9h, acceptance bullet 4 (read as ADR-0042 item 11, amended by ADR-0043, ADR-0046 and ADR-0052); "Benchmarks" (**Degree sweep (M9h)**). Amends ADR-0042, items 11 and 12.

## Context

Bullet 4: "no family is more than 5% slower than in M9 at p ≤ 4 on any device class". ADR-0042 item 11 reads it as "a paired comparison in one job per target: the policy against `direct`, at p ≤ 4, at most 1.05 by rule 1's class aggregation". `high_order_record.rs::no_family_slower_than_m9_at_low_degree` applies it to each of the nine bench families: EVector, Points, Jacobian, DetJ, Pinv, Normals, FacetNormals, the pull-back at 1 and 9 iterations, and BasisEval. It covers each cell type, scalar type, degree p ≤ 4 and rule size, with the geometric mean over the class's devices.

The policy is chosen per policy family (rule 12): J, det J and K together as `Jacobian`, by their summed time, and the pull-back by its 9-iteration time, with the basis evaluator following it. Rule 7 makes a strategy unselectable for the Gpu class if its first use on Locust takes over 30 s; the baseline is then `loops`.

T11b's paired runs at the policy of e9dd096, p ≤ 4, equispaced, the class's rule sizes:

- Mac mini CPU, Locust's Grace CPU and one Kathleen socket: `degree_sweep --interleave`, ten runs;
- the H100 (job 20261114151543-4171256), interleaved, ten runs;
- Metal (sequential, ten runs).

Locust was idle before and after both jobs. The data are in `docs/perf/data/m9h-degree-*-paired.json`. A device's ratio is the median over runs of the policy's run i over `direct`'s run i (Deviations log, T11b).

## Problem

The test fails on every class. The failures fall into four groups.

1. **Rule 7 against rule 11 (Gpu, p = 4).**
   - On tetrahedra, hexahedra and prisms, `direct`'s pull-back takes 43–138 s to first use on Locust (`m9h-launch-cuda-locust.json`). Rule 7 excludes it, so the policy runs `loops` or `sumfact`.
   - Rule 11 compares these with the excluded `direct`. The 9-iteration pull-back comes out 1.15–1.78 times slower, and the 1-iteration one 1.19–1.62.
   - Bullet 4 cannot hold there unless a pull-back strategy beats M9's at p = 4, which none of `loops`, `sumfact`, `shared` and `cooperative` does on the H100.
2. **Bench families the policy does not choose for.**
   - BasisEval is a bench-only view of the crate-private evaluator inside the pull-back, and it runs the pull-back's strategy. Where that is not `direct` it is 1.7–4.0 times slower on the Gpu class at p = 4. On the CPU classes it is 1.13–1.61 at hexahedra p = 3–4, where the CPU pull-back runs `sumfact`, which the evaluator does not have, so it runs `loops`.
   - J, det J and K are chosen by their sum but checked one by one. On Metal, prism det J at p = 2 under `shared` is 1.55, while the class's J + det J + K passes.
3. **Noise on one device.**
   - With ten interleaved runs, identical kernels (the policy is `direct`) on Kathleen give per-device ratios between 0.94 and 1.06 (5th to 95th percentile); 23 of 379 exceed 1.05.
   - On the Mac mini's CPU, 4 of 383 exceed it (at most 1.07), and on Metal 1 of 434.
   - CpuSimd512 has one device, so a single noisy configuration fails the class.
4. **Choices the paired run contradicts.**
   - On Kathleen, E-vectors of tetrahedra at p = 1 in f32 run `loops` under the policy. Interleaved, `loops` is 1.57 times slower than `direct` in every run pair (lower quartile 1.41).
   - T11a's sweeps timed each configuration's strategies one after another. On the CPU runtime that favours the later ones, as the first paired runs showed: identical kernels up to 1.70 apart, reversed when the order was reversed (Deviations log, T11b).
   - On CpuSimd128, the `loops` choices for interval normals at p = 3 (f32, 1.08) and quadrilateral det J at p = 1 (f64, 1.06) are smaller cases of the same kind.

Applying three readings together leaves one failure: Kathleen's tetrahedron E-vector at p = 1 in f32. The readings are:

- per policy family as rule 12 decides (`Jacobian` summed per run, the pull-back at 9 iterations);
- only where `direct` is selectable for the class;
- "slower" meaning slower by over 5% in at least three quarters of the run pairs (the class's geometric mean of the per-device lower quartiles of the per-run ratios above 1.05).

That leaves 0 of 294 configurations on CpuSimd128, 1 of 294 on CpuSimd512, and 0 of 576 on Gpu, with 6 Gpu configurations not compared.

## Options

1. **Keep the design as written.**
   - Group 1 needs a pull-back at p = 4 on 3D cells that matches M9's unrolled kernel on the H100 and compiles in under 30 s. That is new kernel work, with no candidate in M9h's strategies.
   - Group 2 needs the policy to choose BasisEval, det J and K separately from the pull-back and from J, which the policy families of ADR-0042 item 3 do not allow.
   - Group 3 needs many more runs on Kathleen.
   - Not proposed.
2. **Read rule 11 per policy family, where `direct` is selectable, counting consistent slowdowns, and re-time p ≤ 4 interleaved on the CPU classes (proposed).**
   - (a) **Families.** The paired ratio of a policy family uses the summed times of its members per run (`Jacobian`: J + det J + K), and the pull-back at 9 iterations. BasisEval, the 1-iteration pull-back and J, det J and K are reported, not bounded.
   - (b) **Selectability.** Only configurations where `direct` is selectable for the class (rule 7, ADR-0050) are bounded. The others are reported with their ratio to M9's `direct`; at p ≤ 4 they are the Gpu pull-back on tetrahedra, hexahedra and prisms at p = 4.
   - (c) **Consistency.** A class fails a configuration if the geometric mean over its devices of the lower quartile of the per-run ratios exceeds 1.05, that is, if the policy is over 5% slower in at least three quarters of the paired runs. The median ratio is reported.
   - (d) **CPU re-timing.** T11b re-times the CPU classes' sweep at p ≤ 4 with `degree_sweep --interleave`: the Mac mini and one Kathleen socket, `direct`, `loops` and `sumfact`, 10^5 and 10^6 points, ten runs. These records replace T11a's at p ≤ 4 in the decision rule. The policy is then regenerated, and the CPU paired runs repeated. The Gpu class keeps T11a's sweep: its paired check passes under (a)–(c), and Metal showed no order effect.
3. **Option 2 without (d).** Kathleen's tetrahedron E-vector at p = 1 (f32) stays `loops`, 1.57 times slower than M9. Bullet 4 is then not met for one configuration, and the report says so. Saves about 2–3 h of machine time.
4. **Option 2 with (d) on every device, the Gpu class included.** Metal and the H100 are re-timed interleaved at p ≤ 4 as well, about 4–5 h more on Locust. The Gpu policy at p ≤ 4 would rest on interleaved data, at the cost of a further paired run there.

## Proposed decision

Option 2.

- Readings (a)–(c) make bullet 4 test what the policy chooses (policy families, among selectable strategies) and what a paired run on one device can resolve.
- Re-timing (d) removes the sequential-order bias from the CPU decisions at p ≤ 4, where kernels take well under a millisecond and the bias is largest. That is where the one remaining failure comes from.
- The Gpu class is consistent under (a)–(c) as measured. Option 4 is the choice if the maintainer wants every class's low-degree policy on interleaved data.

## Consequences

- Text to change in docs/design.md:
  - Roadmap M9h, bullet 4: "no policy family is more than 5% slower than in M9 at p ≤ 4 on any device class, in a paired comparison where M9's direct kernel is selectable (ADR-0055)".
  - The "Acceptance (read as …)" line: add ADR-0055.
  - "Benchmarks", **Degree sweep (M9h)**: add "rule 11's paired runs interleave the policy's and `direct`'s timed runs on CPUs; at p ≤ 4 the CPU classes' strategies are timed interleaved (ADR-0055)".
- Public API changes: none.
- Acceptance tests affected:
  - `high_order_record.rs::no_family_slower_than_m9_at_low_degree` becomes per policy family, with the selectability and lower-quartile readings.
  - `degree_policy_follows_the_decision_rule` reads the interleaved CPU records at p ≤ 4.
  - `report_covers_every_family_class_and_range` is unchanged.
  - The report's rule-11 table lists the median and lower-quartile ratios per class, the excluded configurations, and BasisEval, J, det J, K and the 1-iteration pull-back as reported rows.
- Plan: T11b gains the CPU re-timing at p ≤ 4 and a repeated CPU paired run; the coverage-table row of bullet 4 names this ADR; issue #37's bullet 4 gets the new text.
- Later milestones affected: M10's benchmark of the final kernels uses interleaved timing for every comparison of strategies.

## Decision record

<Filled by the maintainer.>
