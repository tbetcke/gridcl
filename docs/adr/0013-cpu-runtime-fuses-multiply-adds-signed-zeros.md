# ADR-0013: The CPU runtime fuses multiply-adds; reference comparisons treat +0 and −0 as equal

- **Status:** Accepted
- **Date:** 2026-10-22
- **Raised in:** M3 T14–T15 (`gridcl::factor_invariance::factor_kernels_match_lvector_reference`)
- **Design sections affected:**
  - "Vectorization and CPU execution": "How the CPU runtime runs a kernel" and "Determinism across vector sizes";
  - "Testing and validation": "Vector-size invariance";
  - "Implementation roadmap": M3, last acceptance bullet.

## Context

"How the CPU runtime runs a kernel" states:

> LLVM compiles each kernel for the host CPU at O3. Float additions and multiplications are not
> contracted into fused multiply-adds on the CPU, so each vector lane computes exactly what the
> scalar kernel computes.

M3's last acceptance bullet requires the factor kernels on E-vectors to "agree with a scalar
L-vector reference kernel at every vector size … bitwise on the CPU runtime".

M3 implemented the factor kernels (X, J, det J, K, facet normals, normals) as one vector-generic
kernel, and the acceptance test's reference as a separate scalar `#[cube]` kernel that performs
the same operation sequence (gridcl/tests/common/reference.rs).

## Problem

CubeCL 0.11 contracts multiply-adds on the CPU, too:

- cubecl-opt's `InstCombinePass` (`passes/inst_combine.rs`) rewrites `a * b + c`, `a * b - c` and `c - a * b` into an `FmaOp` whenever the product has a single use. cubecl-llvm adds this pass unconditionally for every target (`shared/base.rs`).
- `FmaOp` is lowered to `llvm.fmuladd`, which LLVM contracts on aarch64 (the M3 Max, Locust's Grace CPU) and on x86-64 with FMA.
- Only the separate fast-math `CONTRACT` flag is limited to GPU targets (`fma_contraction`); this is what the design's sentence describes.

Whether a product is fused depends on its number of uses, so it depends on the code around the
expression. Two different kernel sources computing the same formula can therefore fuse
differently.

**Effect on the vector sizes of one kernel.** None: the IR is the same for every N, and
every lane computes what the N = 1 kernel computes. M3's vector-size invariance tests
(`kernel_invariance`, and the size sweep of `factor_invariance` against its own N = 1 result)
pass bitwise.

**Effect on the comparison with the reference kernel.** On the CPU runtime, strictly bitwise
comparison fails only on the sign of zeros:

- K of a pyramid cell (`Pyramid box, 17 cells, f32`, entry 1167) is +0 in the library and −0 in the reference.
- The cofactor is J01·J21 − J02·J11 with J01 = +0, J21 = −0.25, J02 = 0.25, J11 = +0. IEEE evaluation gives (−0) − (+0) = −0. The fused form fma(J01, J21, −(J02·J11)), with the negated addend produced as 0 − x, gives +0.

Comparing bitwise but with +0 and −0 counted as equal, the whole acceptance test passes:

- every factor (X, J, det J, K, facet normals, normals);
- every cell type and every (TD, GD) pair;
- 0, 1, 15, 16 and 17 cells, and 10^6 + 3 P1 triangles for J and det J;
- every vector size the device offers, tile widths 1 and 16, f32 and f64.

No nonzero value differs in any bit.

## Options

1. **Keep the design as written (strictly bitwise).**
   - **Make both kernels fuse identically.** The reference would have to reproduce the library's use counts, which is fragile and still not guaranteed.
   - **Turn off the fusion.** That needs a patched CubeCL, against the exact pin `cubecl = "=0.11.0"`.

   Not chosen.
2. **Bitwise up to the sign of zero.** On the CPU runtime the reference comparison requires identical bits, or both values equal to zero (+0 or −0). Correct the design's sentence about contraction. Vector-size invariance of one kernel source stays strictly bitwise.
3. **Within a few ulps.** Allow small relative differences against the reference on the CPU too. This is weaker than the evidence requires: no nonzero value differs.
4. **Share the formula code.** Expose the library's `#[cube]` formula helpers (`#[doc(hidden)] pub`) and call them from the reference.
   - The reference would no longer be independent of the code it checks.
   - Equal fusion is still not guaranteed, because use counts also depend on the surrounding kernel.
   - It adds hidden public items.

## Proposed decision

Option 2. The test keeps its full strength for every nonzero value, the only difference observed
is an IEEE-equal one, and no gridcl code depends on the sign of a zero factor. The design text
is corrected to describe CubeCL 0.11 as it is: multiply-adds may be fused on every target, the
fusion is a property of the kernel source and so identical for every vector size, and two
different kernels computing the same formula agree bitwise only up to fusion. M3 confirms that
this is limited to signed zeros for the factor kernels.

## Consequences

- **Text to change in docs/design.md:**
  - "How the CPU runtime runs a kernel", third bullet. Replace "Float additions and multiplications are not contracted … exactly what the scalar kernel computes" with: "CubeCL 0.11 fuses a multiply and the add that consumes it into one fused multiply-add (`llvm.fmuladd`) on every target, the CPU included (ADR-0013). The fusion depends only on the kernel source, so each vector lane computes exactly what the same kernel computes at vector size 1."
  - "Determinism across vector sizes", first bullet: "because each lane performs the scalar operation sequence and nothing is contracted" becomes "because each lane performs the operation sequence of the same kernel at vector size 1 (ADR-0013)".
  - "Testing and validation", "Vector-size invariance": add "A separately written reference kernel may fuse multiply-adds differently; comparisons with one are bitwise up to the sign of zero on the CPU runtime (ADR-0013)."
  - Roadmap M3, last acceptance bullet: "bitwise on the CPU runtime" becomes "bitwise on the CPU runtime, with +0 and −0 equal (ADR-0013)".
- **Public API changes:** none.
- **Acceptance tests affected:**
  - `gridcl::factor_invariance::factor_kernels_match_lvector_reference`: `assert_matches` in gridcl/tests/common/mod.rs accepts equal bits or two zeros on the CPU runtime.
  - The tests that compare one kernel source across vector sizes and tile widths stay strictly bitwise.
- **Later milestones affected:**
  - **M3s** (benchmarks): none.
  - **M9:** the vector-generic pull-back kernel compares its vector sizes with each other, and so stays strictly bitwise; any comparison with a separate reference kernel follows this ADR.
  - **M10 hardening** may report the contraction to CubeCL; the CPU-side fusion also changes rounding against host f64 references by at most normal tolerances.

## Decision record

Accepted by the maintainer on 2026-10-22 (option 2), without conditions. Applied to docs/design.md in `adr: 0013 apply`.
