# ADR-0041: A high-order performance milestone before M10

- **Status:** Proposed
- **Date:** 2026-11-08
- **Raised in:** after the M9 review (docs/plans/M9.md, Review; docs/cubecl-upstream.md C18), maintainer discussion of 2026-11-08
- **Design sections affected:**
  - "Summary": the Scope bullet (the degree range the device kernels are verified and optimised for);
  - "Implementation roadmap": the introduction, the dependency table, a new section "M9h High-order performance", and "M10 Hardening" (its benchmarks and vectorization re-run cover M9h's kernels);
  - "Risks and mitigations": a new row for high-degree kernels.

## Context

The design promises "arbitrary geometry degree" ("Summary", Scope bullet). The host side and the file formats reach far beyond the degrees the device kernels have been checked at:

- **Reference cells (M1b):** node counts match A.4 for p ≤ 10, and partition of unity and reproduction hold to 10^−12 for p ≤ 8 (f64). "Tabulation and quadrature" calls equispaced points well conditioned up to about p = 10.
- **File formats:** the Gmsh element types of A.7 go to 11-node intervals, 66-node triangles, 1000-node hexahedra and 385-node pyramids (orders 9–10), and the high-order node orderings are generated for any degree ("High-order node orderings").
- **Device kernels:** tested to p = 4 at most. The geometric-factor acceptance tests use p = 1–4, the pull-back acceptance p = 1–3, and the device basis tests of M9 T10a degrees 1–4 (pyramids 1–3).

The degree-dependent device kernels evaluate every node at every point: the E-vector gather and the geometric factors (`KernelKind::EVector`, `GeomFactors`, through tabulations uploaded once per (element, point set)) and the pull-back (`KernelKind::PullBack`, through the orthonormal recurrences and `LagrangeElement::coefficients`). For a hexahedron of degree p, nodes and quadrature points both grow like (p + 1)^3, so this costs about (p + 1)^6 per cell, and a tabulation at p = 8 with a Gauss rule of degree 2p holds 729 × 729 × 4 values (about 17 MB in f64) per (element, point set).

## Problem

There is no evidence that the device kernels work, or perform acceptably, above p = 4, and some evidence that they need a different structure there:

- **C18:** the first version of M9's device basis evaluator kept the orthogonal set in a local array, fully unrolled. On Locust (CUDA, GH200) a hexahedron p = 4 kernel took about 200 s to compile and load, and the f64 one at vector size 2 failed to launch with an anonymous `CUDA_ERROR_INVALID_VALUE` (jobs 20261107001156-1923544 and 20261107004359-1931288). The accumulating rewrite fixed p ≤ 4, but its code still unrolls over all basis functions, so code size and driver JIT time still grow with the number of basis functions. The cause (registers, local memory or a launch limit) was never isolated.
- **Cost:** direct evaluation at p = 8 costs about (9/5)^6 ≈ 34 times as much per cell as at p = 4, and about (9/5)^3 ≈ 5.8 times as much per quadrature point. Sum factorisation, the standard technique for tensor-product cells (MFEM, libCEED, deal.II), reduces the per-point growth to about linear in p per dimension.
- **No milestone owns it:** M10 is hardening (GPU-aware transport, benchmarks, the vectorization re-run, documentation). Its benchmarks would record whatever the kernels do, and changing kernel structure there would mix risky algorithmic work into the last milestone.

The maintainer has decided that gridcl's device kernels are to be optimised up to geometry degree p = 8 (2026-11-08).

## Options

1. **Keep the design as written.** "Arbitrary geometry degree" stays unverified on the device above p = 4, and M10 benchmarks the current kernels. This is cheap now, but C18-like failures and the (p + 1)^6 cost would surface in users' meshes rather than in tests. It does not meet the maintainer's decision.
2. **Fold the work into M10.** Same content as option 4, but M10 grows by an open-ended study that changes kernels in gridcl-geometry, gridcl-device and gridcl. M10's own benchmarks and vectorization re-run would chase kernels that change in the same milestone.
3. **Narrow the scope instead.** Declare the device kernels optimised for p ≤ 4 and best-effort above. Contrary to the maintainer's decision.
4. **A new milestone, M9h High-order performance, between M9 and M10.** It measures every degree-dependent kernel family up to p = 8, studies the strategies, chooses one per family, degree range and device class from data, and adds acceptance tests at p = 1–8. M10 then benchmarks and re-studies the final kernels.

## Proposed decision

Option 4. A separate milestone keeps the algorithmic work, with its own plan, acceptance tests and review, apart from hardening, and placing it before M10 means that M10's benchmarks and vectorization re-run measure the kernels that will ship. Its name follows M3s (a study milestone attached to the milestone it builds on). The proposed text:

> ### M9h High-order performance
>
> **Scope:**
>
> - degree sweeps (p = 1–8) in gridcl-bench, with the protocol of "Vectorization and CPU execution", for every degree-dependent kernel family: EVector, GeomFactors (points, Jacobian, det J, pseudo-inverse, normals, facet normals), PullBack and the device basis evaluator; with, per kernel and device, the time per cell and per quadrature point, the first-use compile and load time, and on CUDA the registers and local memory per thread;
> - strategies, each compared with M9's direct evaluation: runtime loops instead of full unrolling (bounding code size and JIT time); sum factorisation for quadrilaterals, hexahedra and prisms; tabulations or coefficients in shared memory; a workgroup cooperating on one cell only where these fall short, with fixed-order reductions that do not depend on the plane size;
> - a degree policy: the strategy of each family chosen per degree range and device class from the measurements, recorded in `docs/perf/high-order.md`, with an override for tests (as `GRIDCL_VECTOR_SIZE` is for vector sizes);
> - C18 closed: its cause isolated with the CUDA kernel attributes, and reported upstream if it is CubeCL's.
>
> **Acceptance:**
>
> - every cell type except pyramids, at p = 1–8 and with both node families where offered: physical points, Jacobians, det J, pseudo-inverses, normals and facet normals from the device match the host reference (gridcl-refcell's f64 tabulation) to 10^−12 (f64) and 10^−5 (f32), and the pull-back recovers reference points to the same tolerances; on the CPU runtime, Metal (f32) and CUDA (Locust); pyramids as before (p ≤ 3, experimental above 2);
> - every kernel variant up to p = 8 compiles, loads and launches on each backend, and the first-use compile and load time of each is at most 30 s on Locust;
> - for quadrilaterals and hexahedra, the time per quadrature point of geometric factors and of a pull-back Newton step at p = 8 is at most 3 times that at p = 4 on every device class (direct evaluation: about 5.8 times);
> - no family is more than 5% slower than in M9 at p ≤ 4 on any device class;
> - the determinism, vector-size invariance and parallel invariance tests pass at p = 1–8: bitwise on the CPU runtime, to 10^−12 (f64) and 10^−5 (f32) on GPUs;
> - `docs/perf/high-order.md` lists, for every family, device class and degree range, the measured cost of each strategy, the chosen one and the reason, with benchmark JSON under `docs/perf/data/` from the targets of M3s.

Open points for the M9h plan, to be settled there or in an ADR of their own: whether f32 can meet 10^−5 at p = 8 (the Vandermonde inverse and Newton lose digits with p; if not, a tolerance per degree); a throughput target for simplices, where collapsed-coordinate sum factorisation is possible but harder; how the CUDA kernel attributes are read without FFI outside the crates that CLAUDE.md allows; and whether the degree policy needs a public item next to `VectorPolicy` (Listing in "Launch plans and the vector policy"), which would need its own ADR.

## Consequences

- Text to change in docs/design.md:
  - "Summary", Scope bullet: after "arbitrary geometry degree", add "(device kernels verified and optimised for p ≤ 8; pyramids p ≤ 3)".
  - "Implementation roadmap": the introduction gains a sentence that M9h studies the degree-dependent kernels before M10; the dependency table gains the row "M9h High-order performance | M9"; the new section "### M9h High-order performance" with the scope and acceptance above, after "M9 Submeshes and manifolds"; "M10 Hardening": its benchmark bullet records geometric factors per second at p = 1, 4 and 8, and its vectorization re-run includes any kernel family M9h adds.
  - "Risks and mitigations": a new row, "High-degree kernels: register pressure, code size and driver JIT time (C18); (p + 1)^6 cost of direct evaluation | M9h: degree sweeps, runtime loops, sum factorisation, a degree policy from benchmarks".
  - "Geometry" ("Tabulation and quadrature", "Geometric factors", "Pull-back") changes only if M9h changes what those sections describe, through M9h's own ADR.
- Public API changes: none from this ADR. A public degree-policy item, if M9h needs one, needs its own ADR.
- Acceptance tests affected: none of the existing ones change. M9h adds tests at p = 1–8; the existing tests at p ≤ 4 stay.
- Later milestones affected: M10 depends on M9h (it already depends on "all") and benchmarks M9h's kernels. A GitHub issue "M9h High-order performance" (label `enhancement`) is opened once this ADR is accepted, and M10's issue gains the dependency.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
