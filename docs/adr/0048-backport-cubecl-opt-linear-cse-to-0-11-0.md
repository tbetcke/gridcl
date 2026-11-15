# ADR-0048: Backport CubeCL's linear common-subexpression pass to 0.11.0

- **Status:** Rejected
- **Date:** 2026-11-11
- **Raised in:** M9h, after T7a (test-suite run time; C20)
- **Design sections affected:** "API caveat" (the pin `cubecl = "=0.11.0"`); "Handing this design to Claude Code" (CLAUDE.md's CubeCL line); M0's scope line (the pin in `[workspace.dependencies]`)

## Context

The design pins CubeCL: "CubeCL is alpha and announces breaking changes between minor versions, so pin `cubecl = "=0.11.0"`" ("API caveat"), and CLAUDE.md repeats it. Every CubeCL compiler (cubecl-llvm for the CPU runtime and CUDA's default path, cubecl-cpp, cubecl-spirv, cubecl-wgpu) runs cubecl-opt's `SimpleCSEPass`.

C20 (docs/cubecl-upstream.md, M9h T3) found that this pass looks expressions up linearly: `ScopedMap` keeps each scope as a `Vec` and scans it on every lookup and insert (`cubecl-opt-0.11.0/src/scoped_map.rs:30-60`, `src/passes/simple_cse.rs:136-143`). A fully unrolled kernel puts every operation in one region, so the pass is quadratic in the operation count. On the Mac mini's CPU runtime, the hexahedron E-vector kernel at p = 8 compiles in 4.2 s at vector size 1, 179 s at size 8 (f64) and 1,039 s at size 16 (f32). A `sample` put all of the compiling thread's time in this pass.

CubeCL fixed this on main in tracel-ai/cubecl#1767, "fix(opt): make simple CSE linear in the operations of a region", merged 2026-10-06. It changes 2 files, +28 and −27 lines: each scope becomes a pliron `SmallMap`, which scans its first eight entries inline and hashes beyond them. `ExpressionKey` hashes its op, operands, result types and memory value, and keeps `AttributeDict`, which has no `Hash`, in the equality only. The PR reports that both versions remove the same operations and leave the same IR, with the pass up to 316 times faster on large regions. The fix is in 0.12.0-pre.1 and pre.2, but not in 0.11.1, 0.11.2 or 0.11.3 (checked against their `scoped_map.rs`). pliron 0.19.0, which cubecl-opt 0.11.0 uses, has `SmallMap` (`pliron-0.19.0/src/utils/table.rs:625`).

## Problem

Test-suite run time. The suite forces `direct` (M9's fully unrolled kernels) at p ≤ 8 in every M9h test (`for_each_strategy!`). `vector_size_invariance` sweeps every vector size up to 16, and the server thread of a test process compiles kernels one at a time, so one slow compile holds up every other test in the process. Measured:
- gridcl's six `high_order*` tests take 1,083 s in release under one strategy setting (M9h T7a).
- A full debug run of gridcl-geometry and gridcl takes 50–85 min per check-set setting (M9h T6b).
- Workspace crates at opt-level 1 gained 3.5% on gridcl-geometry's integration tests (301 → 290 s), within run-to-run spread, because kernel compilation, not host code, dominates.

The quadratic pass may also be what stops CUDA's LLVM path from emitting PTX for the direct pull-back within 900 s from p = 6 (C18, not checked).

The pass is equivalent before and after the fix. A key is inserted only after a lookup missed in every scope, so each scope's keys are unique, and a lookup that searches the innermost scope first finds the same entry whether the scope is scanned or hashed. The compiled kernels are therefore identical, and no check changes.

## Options

1. **Keep the design as written.** The pin stays at plain 0.11.0 and C20's compile times stay until gridcl moves to CubeCL 0.12, a minor version with announced breaking changes and a migration of its own. Every M9h check-set run (T7b to T13) and the milestone review pay the slow compiles.
2. **Backport #1767 into a vendored cubecl-opt 0.11.0, through `[patch.crates-io]`.**
   - `vendor/cubecl-opt/` holds cubecl-opt 0.11.0 from crates.io plus #1767's two files.
   - The workspace `Cargo.toml` gets `[patch.crates-io] cubecl-opt = { path = "vendor/cubecl-opt" }`.
   - `cubecl = "=0.11.0"` and every other crate stay as they are.
   - Remove the patch when gridcl moves to a CubeCL release that contains the fix.
   - Costs: 43 files (640 KB) of vendored source in the repository (licence files kept); `cargo update` and the hosted CI caches see a path dependency.
3. **Ask upstream for a 0.11.4 with #1767, then pin `cubecl = "=0.11.4"`.** Nothing to vendor. But the pin then also takes 0.11.1–0.11.3's other changes, among them "park the idle device server and wake it when a buffer fills" (#1806, which may bear on C14). Those need their own review and full suite. It also depends on upstream's schedule, and the request is outward-facing.

## Proposed decision

Option 2 now, and also ask upstream for a 0.11.4 backport (option 3); with the maintainer's agreement, that request goes on C20's upstream report. The backport is small, it changes no result, and it removes the largest fixed cost of every remaining M9h check-set run.

Acceptance of the backport:
- the CPU runtime's bitwise tests (vector-size, tile and determinism invariance, `forced_direct_matches_the_policy_bitwise`) pass unchanged;
- the IR that cubecl-opt emits for a sample of gridcl kernels is compared before and after the patch, with `CUBECL_DEBUG_LOG` or the compiler's dump, and is identical;
- the gain is measured on the E-vector compile at hexahedron p = 8, size 16 (C20's figure), and on the `high_order*` release run, and recorded in C20.

When a 0.11.x or later release with the fix is adopted under its own ADR, the vendor directory and the patch go.

## Consequences

- Text to change in docs/design.md:
  - "API caveat": after "so pin `cubecl = "=0.11.0"`", add "(cubecl-opt carries a backport of tracel-ai/cubecl#1767 through `[patch.crates-io]`, ADR-0048)".
  - "Handing this design to Claude Code", the CubeCL line of the CLAUDE.md text: the same addition.
  - CLAUDE.md, the same line.
- Public API changes: none.
- Acceptance tests affected: none. Results are unchanged, and the bitwise tests guard it.
- Later milestones affected: M10 and later inherit the patch until the CubeCL upgrade, which removes it. docs/cubecl-upstream.md C20: status "fixed upstream (#1767; 0.12.0-pre), not in 0.11.x; backported in gridcl (ADR-0048)".

## Decision record

Rejected by the maintainer on 2026-11-12. The pin stays at plain `cubecl = "=0.11.0"` with no vendored or patched crate, and no 0.11.4 is requested upstream. C20's compile times stay until gridcl moves to a CubeCL release that contains #1767, under its own ADR.
