# gridcl - working agreement

- Design reference: docs/design.md (gridcl Design B, revised design, October 2026).
  Public types in "Architecture" and "API walkthrough" change only via an ADR in docs/adr/.
- Toolchain: stable Rust 1.99, pinned in rust-toolchain.toml.
- CubeCL: cubecl = "=0.11.0" in [workspace.dependencies].
  - Crates that define or launch kernels may depend on cubecl without runtime features.
  - Runtime types (runtime structs, get_resource, cubecl-runtime) only in gridcl-device.
  - Never add a Runtime type parameter to a public type; use cubecl::prelude::Client.
  - Runtime features only in gridcl-testing, examples, benches and binaries
    (exception: gpu-aware-cuda in gridcl-device).
- Kernels:
  - launch shape (cube count, cube dim, vector size, items per unit) only from LaunchPlan;
    usize arithmetic with overflow checks;
  - at most six buffers; no 64-bit atomics; no float atomics;
  - no assumption of forward progress between workgroups (no decoupled look-back);
  - never assume a plane size; correct for plane size 1;
  - results bitwise deterministic.
- SIMD only through CubeCL Vector<F, N> in kernels; never intrinsics or std::simd.
  - kernels on the candidate list are generic over N: Size and correct for N = 1;
  - lanes are independent items (cells, elements); no cross-lane operations except
    fixed-order reductions and fixed-order in-vector prefixes; no branches on lane values
    (use arithmetic masks);
  - vector sizes come from VectorPolicy (vector-policy.toml), changed only with benchmark
    data in docs/perf/vectorization.md.
- Precision: every kernel test runs in f32 and, where supported, f64.
  Tolerances: 1e-12 relative (f64), 1e-5 (f32).
- Host maths generic over F: Real uses widen()/narrow() or f64, never F's own methods.
- Orientation always uses origin ids, never local or post-partition global ids.
- Tests: cargo test --workspace must pass on the CPU runtime.
  GPU runs: GRIDCL_TEST_DEVICE=vulkan | metal | cuda.
- MPI tests: crate gridcl-mpi-tests ([[test]] targets, harness = false),
  run via scripts/mpi-test.sh 1 2 3 4 under MPICH and Open MPI.
- ndelement is a dependency of conformance/ only (outside the workspace). Never add it to a
  library crate, and never add `version =` to its git dependency. After bumping its pin, run
  cargo test --manifest-path conformance/Cargo.toml and re-export the fixtures with
  cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures.
- Host<->device copies in library code only for I/O (including the upload of input arrays
  at mesh creation, and of quadrature points and tabulations when a rule is created or a
  tabulation is first cached), explicit to_host()/read_node(), the count read of run-length
  and unique-by-key, the per-block read that finishes measure in f64, the debug-build index
  checks of gather and scatter, the host-staged halo exchange, the boundary-entity ownership
  exchange of entity creation (the candidate count and keys to the host; owners and global
  numbers back), the upload of an IndexMap's send and receive index lists when its scatterer
  is first created, and the vertex shared flags at mesh creation (ADR-0026).
- FFI declarations only in *-sys crates. FFI calls only in gridcl-parallel (MPI) and
  gridcl-device (GPU-aware; the macOS performance-core count, ADR-0017), behind safe wrappers. Every unsafe block needs a SAFETY comment.
- Each milestone: acceptance tests first (failing), then implementation, unit tests,
  property tests and rustdoc examples.
- If an API deviation is unavoidable, stop and draft an ADR.

## Workflow

The full procedure for humans is in IMPLEMENTATION.md. Rules for Claude:

- One milestone per branch, named after the milestone in lower case (for example
  `e0-dev-environment`, `m3-serial-mesh-core`),
  created from an up-to-date main after every milestone it depends on is merged.
- Every milestone has a plan, docs/plans/<milestone>.md, made from docs/plans/TEMPLATE.md
  by /plan-milestone. The plan is the hand-off between sessions: read it first, keep it current.
- Work only on tasks of an approved plan (Status: approved or implemented). Tick a task in
  the same commit that completes it. Record every departure from the plan in its Deviations log.
- Commit messages: `<milestone> T<n>: <summary>` for tasks, `plan: <milestone>`, `adr: NNNN <title>`,
  `review: <milestone>`.
- docs/design.md is the source of truth and is not edited during implementation. A change to a
  public API, crate boundary, convention or acceptance test needs an ADR (skill: draft-adr);
  the design is updated only after the maintainer accepts the ADR.
- CubeCL or CubeK bugs, surprising behaviour, missing settings and documentation gaps go into
  docs/cubecl-upstream.md as a `C<n>` entry (version, observation with source location,
  evidence, gridcl workaround, suggestion, status), in the same commit as the work that found
  them. Check claims against the crate source first; update the status when reported or fixed.
- Delegated work (subagents): include the task text, the design sections it depends on and
  this rule: "If the design is ambiguous or an API must change, stop and return an ADR draft
  instead of deviating." Subagents cannot ask the user questions.
- Use the test-runner subagent for full-suite, multi-backend and MPI runs, so logs stay out of
  the main context. Use the design-reviewer subagent before a plan is approved and before a
  milestone PR.
- A new worktree lacks the git-ignored `.venv` (pinned Python oracles) and
  `scripts/hpc/targets.env`: run `scripts/setup-worktree.sh` outside the sandbox before testing.
- Every roadmap milestone has a GitHub issue titled `<milestone> <name>` (label `enhancement`),
  named in its plan's **Issue** line. Keep it in step with the plan: tick a task in the issue
  after its commit; when an accepted ADR changes a milestone's scope or acceptance, update the
  issue text and comment with the ADR; tick the acceptance boxes at the review; the milestone PR
  says `Closes #<n>`. Tick a box with
  `gh issue view <n> --json body -q .body | sed 's/- \[ \] T<k> /- [x] T<k> /' | gh issue edit <n> --body-file -`.
- Claude may push branches and merge into main. Never force-push or rewrite history.

## Test targets

Targets and site details are in docs/dev-environment.md; the policy is in docs/design.md,
"Development environment and test targets".

- This machine is the M3 Max. Run tests here on the CubeCL CPU runtime (the default) and on
  Metal (`GRIDCL_TEST_DEVICE=metal`) on every change.
- CUDA runs go through `scripts/hpc/remote-test.sh` only (skill: hpc-test). Never run GPU work
  on a login node, and never submit jobs with hand-written `qsub` or `ssh` commands. The
  exception is milestone E0, while the runner is being built: then each such command needs the
  user's approval.
- Locust (`locust`) is the CUDA platform for every run: after every task that changes kernels,
  at every milestone review, and for benchmarks (ADR-0014).
- Myriad V100 (`myriad-v100`) and A100 (`myriad-a100`) only: as the fallback when Locust is
  unreachable, a run fails because another user occupies its GPU, or a benchmark cannot get an
  idle node; for multi-GPU and GPU-aware MPI tests (`--gpus 2`); to compare
  architecture-dependent results; for CUDA-version- or architecture-specific changes; or when
  the user asks. Say why when you schedule them.
- Kathleen (`kathleen`, Slurm, Xeon Gold 6248 with AVX-512) runs every CPU-only remote run
  (x86-64 AVX-512 CPU benchmarks, CpuSimd512) on one whole node, and the MPI runs beyond 4 ranks;
  from M6 on, every milestone review also runs the MPI tests on two Kathleen nodes (ADR-0016).
  Never more than two nodes per job.
- Myriad's CPU-only D nodes (`myriad-cpu`, `-ac allow=D`, never a GPU node) run CPU-only work
  only as the fallback: Kathleen unreachable, a job still queued after `--max-wait`, or the user
  asks; say why. They request the whole node (`--cores 36`).
- Never draw SIMD conclusions from a node shared with other jobs.
- Locust has no scheduler (open multi-user node): never draw benchmark or timing conclusions
  from a Locust run unless its `node load:` line is `idle` before and after (ADR-0002).
- At most two scheduler jobs (Myriad, Kathleen) at a time. Locust jobs are independent of them:
  one Locust run may go in parallel with the scheduler jobs. Cancel jobs you no longer need;
  leave none queued at the end of a session.
- Never put user ids, private host names, project codes or credentials in committed files; they
  live in `scripts/hpc/targets.env`, which git ignores.
- If a remote target is unreachable, ask the user to open the SSH connection (`ssh myriad true`)
  and continue with local work; do not retry more than twice.
