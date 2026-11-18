# ADR-0064: Design text after M10 T5, and the strategies of M10's p = 8 / p = 4 ratio

- **Status:** Accepted
- **Date:** 2026-11-18
- **Raised in:** M10 review (design-reviewer, Major 1; acceptance-coverage check of bullet 2)
- **Design sections affected:**
  - "Distributed mesh model": Listing S2c (`Scatterer`'s private fields) and the paragraph after it; "Transports" (HostStaged).
  - "Workspace layout": the crate table's gridcl-bench row.
  - "Implementation roadmap", M10 Hardening, acceptance bullet 2 (as ADR-0042 item 1 reads it).

## Context

**Item 1: the scatterer after T5.** ADR-0059 added this to "Transports" (HostStaged):

> The received data is uploaded through pinned staging buffers that each scatterer reuses, and `reverse` uploads only the received values (ADR-0059).

Listing S2c still shows the M6 reverse fields `rev_order` ("received entries sorted by target owned index") and `rev_targets` ("owned local index of each segment"). The paragraph after it says that "every reverse exchange is a gather followed by a segmented reduce in a fixed order". ADR-0061 updated that paragraph's description of `reverse`'s steps, but left this sentence.

M10 T5 implemented the following (Deviations log, T5 rows; C23 and C24 in docs/cubecl-upstream.md):

- **Uploads:** CubeCL 0.11 hands out no pinned buffer that a caller can keep and fill in place (C23). On the Gpu class, `DeviceBuffer::write_host` copies the data once into a padded allocation, stages it through `Client::staging` (CubeCL's pinned pool), then calls `Client::write`.
- **Reuse:** each scatterer reuses, per scalar type, its device send and receive buffers, its host receive buffers, the targets' own values and the reverse buffer (`Buffers<T>`, `gridcl-parallel/src/scatterer.rs:123`).
- **Reverse:** `reverse` puts the targets' own values and the received values into the reduce's order with two element-wise `unpack`s (by `rev_own` and `rev_received`). This replaces the gather by `rev_order`. `Insert` gathers by `rev_first`, and the result reaches the targets by an `unpack` over the targets, which the map's shared indices hold. `rev_targets` is gone. The reduce sees the same values in the same order as before, so results are bitwise those of M9h.

**Item 2: bullet 2's ratio.** M10's acceptance bullet 2 asks for the time per point at p = 8 over p = 4 "under the generic degree policy (ADR-0042 item 1, ADR-0053, ADR-0058)". ADR-0042 item 1 says how the ratio is formed:

> The ratio is that of the strategy the policy selects at p = 8, timed at p = 8 and at p = 4.

M9h's `tensor_cost_ratio_reported` follows that definition. M10's T0 instead fixed "each degree under its own policy choice" (s4 at p = 4, s8 at p = 8). The Deviations log records this as a reading the tests fixed (T0, second row), not as a deviation. `tensor_cost_ratio_within_bound_or_reported` and M10's section of docs/perf/high-order.md use that reading, which ADR-0060 item 5 did not cover.

## Problem

**Item 1.** The design describes buffers and fields that do not exist:

- No pinned buffer is reused: each upload takes one from CubeCL's pool.
- There is no `rev_order` gather and no `rev_targets`.

The code cannot follow the text on CubeCL 0.11 (C23). The text is also out of date in two smaller places:

- The gridcl-bench row of the crate table lacks gridcl-geometry (`bench`, `testing`), gridcl-dims, bytemuck and the optional mpi that M10 T6 added.
- `LegacyClass` is `pub` in `gridcl_bench::record`, where ADR-0060 item 2 said crate-private. The tests in `gridcl-bench/tests/` are a separate crate and read it (Deviations log, T1).

**Item 2.** The two definitions give different reports. M10's section has 270 ratio rows (5 platforms, quadrilaterals and hexahedra, every factor family and the pull-back's step and set-up, 10^6 and 10^7 points). In 60 of them the policy chooses different strategies at p = 4 and p = 8. These counts come from a review script over the committed `m10-degree-*.json` records; the loader's choice among repeated records may differ slightly.

- **36 of the 60 cannot be computed with ADR-0042's definition from the records.** All are GPU rows: Locust CUDA 24 (`sumfact` 14, `shared` 10) and Metal 12 (`sumfact` 8, `shared` 4). There the p = 8 choice was not a candidate at p = 4, so it was not timed there.
- **For the other 24, the miss status differs in 4 rows.** All four are the CPU pull-back's set-up on hexahedra (`direct` at p = 4, `sumfact` at p = 8). Under ADR-0042's definition they miss the bound of 3; under M10's reading they are within it:

  | Platform | Scalar | Points | Ratio, M10's reading | Ratio, ADR-0042's definition |
  | --- | --- | --- | ---: | ---: |
  | Locust Grace | f32 | 10^6 | 1.10 | 8.74 |
  | Locust Grace | f32 | 10^7 | 1.09 | 4.24 |
  | Locust Grace | f64 | 10^6 | 0.51 | 3.70 |
  | Kathleen | f64 | 10^7 | 2.28 | 10.23 |

  The sum-factorised set-up is cheap at p = 4, so the ratio of `sumfact` alone is large.

Neither definition changes whether bullet 2 is met. ADR-0060 item 5 meets it by reporting each miss with its cause, and `tensor_cost_ratio_within_bound_or_reported` passes. What changes is which number the report gives.

## Options

**Item 1**

1. **Keep the design as written.** The scatterer would hold pinned host buffers and fill them in place. CubeCL 0.11 has no such API (C23), so this needs an upstream change. The `rev_order` gather would also come back, as one more kernel per `reverse`. Not chosen.
2. **Describe what T5 built (proposed).** Text only; no code changes.

**Item 2**

1. **Keep the design as written (ADR-0042's definition).**
   - Change `tensor_cost_ratio_within_bound_or_reported` and `render_m10_high_order` to time the p = 8 choice at both degrees.
   - Run the 36 missing configurations: GPU `sumfact` and `shared` at p = 4 on quadrilaterals and hexahedra, f32 and f64 (Metal f32), 10^6 and 10^7 points, candidate protocol. That is about one Locust job of under 30 min (idle before and after) and about 15 min on Metal.
   - Then re-run gridcl-bench's tests and `high_order_report --check`.
   - The report would then show how the p = 8 kernel itself scales, at the cost of a run outside the standing matrix and a later PR.
2. **Each degree's own choice (proposed).**
   - Amend ADR-0042 item 1's last-but-two point for M10: the ratio is the policy's time per point at p = 8 (its choice there) over the policy's time per point at p = 4 (its choice there).
   - No runs and no code changes; only the T0 row in the Deviations log is re-labelled as this ADR's.
   - M9h's frozen section keeps ADR-0042's definition, and says so.
3. **Report both.**
   - Option 2 as the bounded ratio, plus a column for the p = 8 strategy timed at p = 4 wherever records exist, marked "–" in the 36 GPU rows.
   - A report and test change, no runs. The column would be incomplete on exactly the GPU rows.

## Proposed decision

**Item 1: option 2.** The design should describe the exchange gridcl ships, and the per-call pinned staging is the only pinned path CubeCL 0.11 offers. T5 measured it: 0.43 ms against 0.32 ms for `to_host` on the GH200, against 2.6–2.9 ms for the unstaged write. That is within bullet 1's bound, so a reused pinned buffer is a possible upstream improvement (C23), not a requirement.

**Item 2: option 2.** Bullet 2 asks for the ratio "under the generic degree policy". The generic policy chooses a strategy per degree, so the cost a user sees from p = 4 to p = 8 is the policy's time at each degree, which is what M10 reports. ADR-0042's definition answered a different M9h question: whether the strategy chosen at p = 8 scales within the bound. With the bound reported rather than enforced (ADR-0053, ADR-0060 item 5), the scaling of each strategy stays visible in M9h's section and in the raw records. Option 1 would cost runs outside the standing matrix for a number that does not change whether the bullet is met.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing S2c**, in `Scatterer`'s private fields:
    - replace the `rev_order` and `rev_targets` lines with:
      - `rev_own: DeviceBuffer<u32>, // reverse: buffer index of each target's own value`
      - `rev_received: DeviceBuffer<u32>, // reverse: buffer index of each received value`
      - `rev_first: DeviceBuffer<u32>, // reverse (Insert): first contribution of each target`
    - add `buffers: /* per scalar type */, // device and host buffers reused across calls (ADR-0059)`.
  - **The paragraph after Listing S2c:** "so every reverse exchange is a gather followed by a segmented reduce in a fixed order" becomes "so every reverse exchange is a placement by `unpack` followed by a segmented reduce in a fixed order (ADR-0064)".
  - **"Transports", HostStaged:** "The received data is uploaded through pinned staging buffers that each scatterer reuses, and `reverse` uploads only the received values (ADR-0059)." becomes "On GPUs the received data is uploaded through CubeCL's pinned staging (`Client::staging`, one staging buffer per upload: CubeCL 0.11 has no pinned buffer a caller keeps; C23). Each scatterer reuses its device and host buffers across calls, and `reverse` uploads only the received values (ADR-0059, ADR-0064)."
  - **The crate table's gridcl-bench row:** the dependencies become "gridcl, gridcl-device, gridcl-geometry (features `bench`, `testing`), gridcl-dims, gridcl-refcell, gridcl-topology (feature `bench`), gridcl-testing; cubecl with runtime features; serde_json, toml, bytemuck; mpi (optional, feature `mpi`, for `halo_exchange`; ADR-0060); dev: proptest (ADR-0039)".
  - **M10 acceptance bullet 2:** "under the generic degree policy (ADR-0042 item 1, ADR-0053, ADR-0058)" becomes "under the generic degree policy, each degree under its own choice (ADR-0042 item 1 as amended by ADR-0064, ADR-0053, ADR-0058)".
- **ADR notes:** a closing note in ADR-0042 (item 1 amended by ADR-0064 item 2, from M10 on), and one in ADR-0060 (item 2: `LegacyClass` is `pub` in the unpublished gridcl-bench, so its tests can read it; ADR-0064).
- **Public API changes:** none. Only private fields change; gridcl-bench is unpublished.
- **Acceptance tests affected:** none under the proposed options.
  - `tensor_cost_ratio_within_bound_or_reported` already uses each degree's own choice. Its doc comment gains "(ADR-0064)".
  - Under item 2's option 1 or 3, that test and `render_m10_high_order` would change, and option 1 needs the runs listed there.
- **Later milestones affected:**
  - M10 review: Deviations-log rows for item 1 (T5's withdrawn finding (1) left the per-call staging unrecorded) and for item 2 (the T0 reading).
  - Issue #25: bullet 2's text updated, with a comment naming this ADR.
  - Follow-up, outside M10: a reused pinned upload buffer, if CubeCL adds one (C23).

## Decision record

Accepted by the maintainer on 2026-11-18, as proposed (item 1: option 2; item 2: option 2, each degree's own choice, no runs).

Applied in `adr: 0064 apply`: docs/design.md (Listing S2c, the paragraph after it, "Transports", the crate table's gridcl-bench row, Roadmap M10 bullet 2), closing notes in ADR-0042 and ADR-0060, the doc comment of `tensor_cost_ratio_within_bound_or_reported`, and issue #25.

Addendum at the M10 review (maintainer, 2026-11-18: fix all outstanding minors), applied in `M10 review: minors`:

- **"Transports":** "On GPUs" becomes "On CUDA and HIP … the other runtimes have no pinned staging". `write_host` now calls `Client::staging` only where it pins (`Client::name` is `cuda` or `hip`; C23). On Metal and wgpu it staged nothing but still made a blocking submit.
- **The paragraph after Listing S2c:** the reverse sentence adds "(`Insert`: a gather by `rev_first`; …)". `Insert` skips the placement and the reduce.
- **Listing S2c:** adds `targets`, the distinct owned indices that `reverse` packs from and writes back to.
- **The crate table's gridcl-bench row:** adds gridcl-device's feature `testing`.
