# ADR-0008: `LaunchPlan` is not `Copy`, and it computes its own cube count

- **Status:** Accepted
- **Date:** 2026-10-19
- **Raised in:** M2 T0 (stubs for Listing S2)
- **Design sections affected:**
  - "Vectorization and CPU execution": Listing S2 (launch plans) and the bullets after it;
  - "Kernels receive dimensions from types": the paragraph after Listing B5;
  - "Device model and backends": "Device limits" (cube count);
  - "Risks and mitigations": the device-limits row.

## Context

Listing S2 declares:

```rust
#[derive(Clone, Copy, Debug)]
pub struct LaunchPlan {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    ...
}
```

The paragraph after Listing B5 says "`LaunchPlan` takes the cube count from
`calculate_cube_count_elemwise`, which spreads cubes across the device's maximum cube count in
each dimension". "Device limits" says the cube count is "computed by
`calculate_cube_count_elemwise`, which respects `hardware.max_cube_count`".

ADR-0007 added `testing::with_properties`, which "replaces the properties that gridcl-device
reads (limits, device class, vector sizes) for buffers and plans made inside `f`". The M2
acceptance test `launch_plan::prop_plan_respects_injected_limits` relies on it: "with injected
device properties, no plan exceeds `max_units_per_cube`, `max_cube_dim` or `max_cube_count`".

While writing T0's stubs, the CubeCL 0.11.0 sources were read
(`cubecl-runtime-0.11.0/src/server/base.rs`, `cubecl-core-0.11.0/src/lib.rs`).

## Problem

1. **`CubeCount` is not `Copy`.** It is
   `enum CubeCount { Static(u32, u32, u32), Dynamic(BufferBinding) }` with a hand-written
   `Clone` (`base.rs:1584`, `impl Clone for CubeCount`). A struct with a `CubeCount` field cannot
   derive `Copy`, so Listing S2 as written does not compile (rustc rejects the derive with
   E0204, "the trait `Copy` cannot be implemented for this type").
2. **`calculate_cube_count_elemwise` cannot honour injected limits.** Its signature is
   `(client: &Client, num_elems, cube_dim)`, and it reads `client.properties().hardware.max_cube_count`
   (through `CubeCountSelection::new`). A property set injected by `testing::with_properties`
   never reaches it, so `prop_plan_respects_injected_limits` cannot pass if the plan uses it.
3. **Its spread can exceed `max_cube_count`.** `cube_count_spread` (`base.rs:1783`) halves x and
   doubles y until x fits, then halves y and doubles z until y fits, and never checks z.
   Powers of two overshoot: with `max_cube_count = (3, 3, 3)` and 27 cubes needed it returns
   (2, 2, 8), although (3, 3, 3) fits exactly. On the real development targets (x up to
   2^31 − 1 or 65 535, y and z 65 535) this happens only far beyond the u32 limit on launched
   units, so it matters only for injected or unusual limits; the design's claim that it
   "respects `hardware.max_cube_count`" is still not true in general.

## Options

1. **Keep the design as written.** Not possible for problem 1 (the listing does not compile).
   For problem 2, `with_properties` would not reach the cube count, and the acceptance test
   would have to be weakened to cube dimensions only. Not chosen.
2. **Replace the field by `(u32, u32, u32)`** and add a method `cube_count() -> CubeCount`.
   `LaunchPlan` stays `Copy`, but the public field changes type and Listing B5's launch call
   (`plan.cube_count`) changes to a method call in every launcher. Not chosen: it changes more
   public text than needed.
3. **Drop `Copy`, and let `LaunchPlan` compute its cube count itself.**
   - `#[derive(Clone, Debug)] pub struct LaunchPlan`, fields unchanged. Listing B5 still compiles:
     moving `plan.cube_count` out of the plan is a partial move, and the `Copy` fields
     (`num_vectors`, `items_per_unit`, …) stay usable after it. `record_plans` stores clones.
   - The cube count comes from a crate-private function of gridcl-device over the properties
     gridcl-device reads (the client's, or the injected ones): for `n` cubes and limits
     `(mx, my, mz)`, `x = min(n, mx)`, `y = min(⌈n / x⌉, my)`, `z = ⌈n / (x·y)⌉`, and
     `GridError::TooLarge` if `z > mz`. The result is always `CubeCount::Static`, never exceeds
     any limit, and fits whenever `n ≤ mx·my·mz`. In usize arithmetic with overflow checks.
   - With the real properties and `n ≤ mx`, the result equals `calculate_cube_count_elemwise`'s
     (`(n, 1, 1)`), so launches on the development targets keep their shape.
   - Kernels index by `ABSOLUTE_POS`, as Listing B5 does, and cubes past the needed count stay
     idle through the `v < num_vectors` check.

## Proposed decision

Option 3. It is the smallest change that compiles, keeps every public field and Listing B5's
launch code, and makes the injected-limit acceptance test meaningful. The cube count stays
the plan's job, as the design intends; only its source changes from a CubeCL helper that sees
the live client to gridcl-device's own function over the properties it reads.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing S2:** `#[derive(Clone, Debug)]` on `LaunchPlan`, with the comment "not `Copy`:
    `CubeCount` is not `Copy` in CubeCL 0.11".
  - **Listing S2, the "GPUs" bullet:** "the cube count comes from gridcl-device's spread over
    `max_cube_count` (x first, then y, then z), computed from the properties gridcl-device
    reads, so injected limits apply"; the "One kernel source" bullet keeps "Every plan is
    checked against `max_cube_dim`, `max_units_per_cube` and `max_cube_count`".
  - **Paragraph after Listing B5:** "`LaunchPlan` computes the cube count itself: it fills x up
    to the device's maximum, then y, then z, so no launch exceeds the 65 535-per-dimension limit
    of WebGPU-class devices." (instead of naming `calculate_cube_count_elemwise`).
  - **"Device limits", cube count:** "computed by `LaunchPlan`, which respects
    `hardware.max_cube_count`; `TooLarge` if no launch fits".
  - **"Risks and mitigations", device-limits row:** "use `LaunchPlan`" instead of
    "use `calculate_cube_count_elemwise`".
  - Unchanged: dependency rule 2 (the function stays part of the launch API anyone may use),
    and E0's acceptance text, which describes env-check, not library code.
- **Public API changes:** `LaunchPlan` loses `Copy`; it keeps `Clone`, `Debug` and all fields.
- **Acceptance tests affected:** none change. `launch_plan::prop_plan_respects_injected_limits`
  can now check `max_cube_count` under injected properties as written.
- **Later milestones affected:** launchers (M3 onwards) write `plan.cube_count` once per launch,
  or `plan.cube_count.clone()` if they launch twice from one plan. M2's multi-phase primitives
  clone the plan's cube count for each phase.

## Decision record

Accepted by the maintainer on 2026-10-19: option 3, without conditions. Applied to docs/design.md on the same day (Listing S2 and its "GPUs" bullet, the paragraph after Listing B5, "Device limits", and the device-limits row of "Risks and mitigations").
