# ADR-0045: CPU cubes of per-group plans hold several groups

- **Status:** Accepted
- **Date:** 2026-11-10
- **Raised in:** M9h T5 (`LaunchPlan::per_group`)
- **Design sections affected:** "Launch plans and the vector policy", Listing S2 (the doc comment of `LaunchPlan::per_group`). Amends ADR-0042, item 4.

## Context

ADR-0042, item 4 adds `LaunchPlan::per_group` for the `shared` and `cooperative` strategies. A plan runs `groups` independent work groups (a cell or a tile of cells), each by `units_per_group` units. It fixes the cube shape per device kind:

> On GPUs a cube holds `units_per_group` × (groups per cube) units, at most 256. On CPUs `cube_dim` = `units_per_group`, and each cube visits blocks of consecutive groups.

Listing S2 in docs/design.md says the same in the doc comment of `per_group`.

docs/design.md, "How the CPU runtime runs a kernel", describes how CubeCL runs a cube on a CPU:

> Every unit of a cube is a separate worker thread, and each unit runs the whole cube loop: unit u visits the absolute positions cube · cube_dim + u for every cube of the launch.

The CubeCL 0.11 source confirms this:

- `cubecl-cpu` `src/compute/threadpool/mod.rs` starts `cube_dim.num_elems()` workers (`ensure_workers`, line 87), one per unit.
- `cubecl-llvm` `src/cpu/entrypoint.rs` wraps the kernel body in loops over `cube_count` z, y and x (`insert_skeleton`, lines 177–211). Each worker therefore visits every cube.
- When a kernel declares shared memory and `cube_dim` > 1, it adds a cube barrier at the end of each cube (`fence_cubes`, lines 54 and 94).

`elementwise` already plans for this. On CPUs its `cube_dim` is one unit per hardware thread, or the performance-core count where the vector policy says so (ADR-0017), and each unit takes blocks of consecutive vectors.

T5 implemented `per_group` as ADR-0042 states (commit 68115c4). It found that a CPU `per_group` plan therefore runs only `units_per_group` threads.

## Problem

On CPUs the number of threads a `per_group` launch uses is the group size. The device's thread count plays no part:

- **T9 (`shared`)** plans with `units_per_group` = 1 (plan, T9), so every `shared` launch on a CPU runs on one thread.
- **T10a and T10b (`cooperative`)** use `units_per_group` threads: at most the power of two at or below the units a cell can use, for example 8 of 12 threads on the M6, 16 of 20 on a Kathleen socket, 64 of 72 on Grace. That holds only where a cell has that many units of work. At low degree it is 1 or 2.

The plan runs both strategies on every target:

- T11a sweeps every strategy on every M3s target.
- T11b's decision rule picks the cheapest strategy per (class, family, cell type, degree range).

On CPUs, `shared` and `cooperative` would be measured on a fraction of the machine and could not win. The comparison would only show the launch shape, not the strategy. They would also lengthen T11a's CPU runs: T3's CPU sweeps of `direct` already took 4.0–8.4 h per target, and a single-threaded strategy at 10^7 points takes up to the thread count times longer.

ADR-0042's rule contradicts the design's own CPU model. `elementwise` exists to give each CPU thread contiguous blocks, and `per_group` on CPUs gives one group slot, and hence one thread, to the whole launch.

## Options

1. **Keep the design as written.** `shared` and `cooperative` run on `units_per_group` threads on CPUs. T11a measures them so, and T11b's rule then never picks them on CPUs. This needs no change. The cost is T11a's CPU sweep time spent on a handicapped variant, and no answer to whether either strategy helps on CPUs. Not chosen.
2. **CPU cubes hold several groups, as on GPUs.**
   - **The rule:** on CPUs a cube holds `units_per_group` × ⌊U / `units_per_group`⌋ units, where U is `elementwise`'s CPU unit count for the same (kind, element size): the hardware threads, or the performance cores where the vector policy says so (ADR-0017). `units_per_group` is at most U, so a cube holds at least one group.
   - **What stays the same:**
     - Each group slot takes blocks of `items_per_unit` consecutive vectors of groups, sized as `elementwise`'s CPU blocks. T5's indexing already covers this: slot s of cube c takes vectors (c · slots + s) · `items_per_unit` + i.
     - Kernels, the reduction trees and the `sync_cube` rule are unchanged, and results stay bitwise the same, because a group's result does not depend on its slot.
     - `shared_bytes` stays per cube and is checked as before. Staged data that every group reads (T9's tabulations) is shared by more groups per cube. Per-slot data must fit the CPU's L1-sized limit for U / `units_per_group` slots, as it must for a GPU cube's slots. Above the limit the strategy falls back to `loops`, as T9 plans.
   - **The new cost:** each `sync_cube`, and CubeCL's barrier at the end of each cube, then synchronises U threads instead of `units_per_group`. T11a measures this. Each cube covers `slots` × `items_per_unit` vectors, so the barrier is paid once per block per thread, not once per group.
3. **Keep rule 4, and leave CPUs out of `shared` and `cooperative`.** T11a sweeps them on GPU classes only, and the CPU entries of `degree-policy.toml` never name them. This saves the sweep time of option 1. But it changes T11a's coverage ("every strategy on every M3s target") and the report's acceptance row, and it still leaves the question for CPUs unanswered. Not chosen.

## Proposed decision

Option 2. A CPU plan then uses the threads `elementwise` uses, which is what the design's CPU model and every other launch plan assume. Kernels and results stay unchanged, so T9 and T10 are written once for both device kinds. T11a's measurements then compare strategies on CPUs, not launch shapes. The extra barrier cost is real but bounded, and the sweep measures it.

## Consequences

- **Text to change in docs/design.md:**
  - Listing S2, the doc comment of `LaunchPlan::per_group`: "CPUs: cube_dim = units_per_group, and each cube visits blocks of consecutive groups" becomes "CPUs: a cube holds units_per_group × (groups per cube) units, as many as elementwise's CPU units (ADR-0017), and each group slot visits blocks of consecutive groups".
  - In ADR-0042's decision record, a line "Amended by ADR-0045 (CPU cubes of per-group plans hold several groups)".
- **Public API changes:** none. The signature and fields of `per_group` stay; only its documented CPU shape changes.
- **Implementation:**
  - `gridcl-device/src/plan.rs`: the CPU branch of `per_group`, sharing `elementwise`'s CPU unit count, and its rustdoc.
  - Its unit test `cpu_cubes_visit_blocks_of_consecutive_groups`, and the property test in `gridcl-device/tests/launch_plan.rs`. On the CPU runtime, the rule-4 tree kernel test then also runs several group slots per cube.
  - This is done as the first step of T9, the first strategy that plans with `per_group`, with a Deviations-log row. A CUDA run is not needed for it, since GPU plans do not change.
- **Acceptance tests affected:** none. T5's tests change their expected CPU `cube_dim`, and T9–T10b's tests are unchanged.
- **Plan (M9h):** T9's note "Planned by `per_group` with `units_per_group = 1`" stays. On CPUs it now means one group per unit, with all threads busy.
- **Later milestones affected:** none known. Every later `per_group` user gets CPU plans that use the whole machine.

## Decision record

Accepted by the maintainer on 2026-11-10, as proposed (option 2). Applied in `adr: 0045 apply`: docs/design.md, Listing S2 (the doc comment of `LaunchPlan::per_group`), and ADR-0042's decision record. The change to `gridcl-device/src/plan.rs` and T5's CPU-shape tests follows as the first step of T9.
