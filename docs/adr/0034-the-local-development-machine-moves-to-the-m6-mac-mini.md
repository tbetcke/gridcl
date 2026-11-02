# ADR-0034: The local development machine moves to the M6 Mac mini

- **Status:** Accepted
- **Date:** 2026-11-02
- **Raised in:** the move between M7 and M8 (M7 done in d015c2e; M8 not yet planned)
- **Design sections affected:**
  - "Overview" (the **Development and test targets** bullet) and decision 10;
  - "Backends" (the CPU runtime and Metal rows, and the PT-Scotch row);
  - "Vectorization and CPU execution": the vector-size table, the **CPUs** bullet of `LaunchPlan`, and "Benchmarks" ("Where", "Conditions");
  - "Partitioning" (the **Reproducibility** bullet);
  - "Testing and validation" (the targets table and the CI matrix);
  - "Development environment and test targets" (introduction, targets table, remote-test runner);
  - "Implementation roadmap": M10's benchmark bullet;
  - "Summary of decisions" (the targets line);
  - ADR-0017's performance-core rule, and ADR-0031's "works with any Scotch 7 build".

## Context

The design names one local machine, the Apple M3 Max. For example, "Development environment and
test targets":

> Claude Code runs on the M3 Max and tests there on every change.

Its targets table has the row "M3 Max (local) | Apple M3 Max, macOS, arm64 | local | Claude Code;
CPU runtime (f32, f64) and Metal (f32) on every change; MPI up to 4 ranks". `CLAUDE.md` says
"This machine is the M3 Max", and so do `docs/dev-environment.md`, `scripts/check-local-tools.sh`,
`.claude/agents/test-runner.md` and `.claude/skills/plan-milestone/SKILL.md`.

The maintainer has moved local development to an Apple M6 Mac mini:
- 24 GB of memory (the M3 Max has 64 GB);
- 12 cores in three performance levels: 2 Super, 4 Performance and 6 Efficiency
  (`hw.nperflevels` = 3; earlier Apple chips have two);
- macOS 27.2.

The machine was set up from a clean install, following the maintainer's handoff note, which is
not in git. It then ran the full local verification (that note's part D), on 2026-11-01 and 02,
on main at d015c2e:
- env-check;
- fmt and clippy;
- the workspace on the CPU runtime and on Metal, and at `GRIDCL_VECTOR_SIZE` = `scalar` and
  `widest`;
- the dependency check;
- `scripts/mpi-test.sh` on Metal (1–4 ranks) and on the CPU runtime (1 rank);
- the partitioner crates and the M7 MPI targets;
- a Locust CUDA smoke job.

Four points need a decision: three in the design and one in a library version.

ADR-0017 defines the performance-core count as `hw.perflevel0.physicalcpu`. ADR-0031 runs every
Scotch and PT-Scotch call in a deterministic `SCOTCH_Context` and says this "works with any
Scotch 7 build"; the design's backend table says "Scotch 7.x (context API)".

## Problem

1. **The design names a machine that is no longer the local target.** Every "M3 Max" sentence in
   the sections above is now wrong about where Claude runs and tests.
2. **The performance-core rule reads the wrong level on the M6.** On the M6, perflevel0 is
   "Super" with 2 cores, perflevel1 "Performance" with 4, and perflevel2 "Efficiency" with 6.
   - `gridcl-device/src/cores.rs` returns 2, not 6. A kernel family whose `cpu_units` is
     `performance` therefore launches with `cube_dim` 2 on a 12-core machine. Results stay
     correct, because launch shapes never change them, but those families run on 2 of 12 cores.
   - The host records have the same fault:
     - env-check reported `performance_cores: 2, efficiency_cores: 4`. The 4 Performance cores
       are recorded as efficiency cores, and the 6 Efficiency cores are missing.
     - gridcl-bench's `protocol::cores()` reads the same two sysctls.
   - The "Benchmarks" conditions name `sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu`.
3. **Homebrew's Scotch 7.0.18 is not deterministic, even in ADR-0031's context.** Homebrew now
   installs Scotch 7.0.18; the M3 Max and the M7 numbers had 7.0.14.
   - With 7.0.18, `partition_threads::scotch_on_tetrahedra` and `scotch_on_hex_prism_pyramid`
     fail with "seed 7 twice gives different parts", also with `--test-threads=1`.
   - The M7 MPI targets on Metal fail 5 runs (`partitioners,metis`) and 6 runs
     (`partitioners,parmetis`) of 36:
     - `partition_build`: owned cells differ between repeated builds;
     - `partition_seed`: Scotch's seeds 1 and 2 give the same hash, while repeated runs differ;
     - `partition_quality` at 4 ranks.
   - The same test binary passes 3 of 3 runs when Scotch 7.0.14 is loaded instead
     (`DYLD_LIBRARY_PATH` set to the 7.0.14 that `scripts/hpc/build-partition-libs.sh` builds),
     and fails 2 of 2 with 7.0.18.
   - With 7.0.14, every partitioner test passes: 188 tests (METIS set) and 186 (ParMETIS set),
     and all 24 + 24 M7 MPI runs. The 4-rank ratios match the M7 review's Kathleen ranges:
     cut(PT-Scotch)/cut(Scotch) is 0.9620 (tetrahedra) and 1.0509 (mixed).
   - With 7.0.18:
     - `SCOTCH_DETERMINISTIC=1` in the environment does not help (3 of 3 runs fail);
     - one Scotch thread (`SCOTCH_PTHREAD_NUMBER=1`) restores determinism (3 of 3 pass, and the
       M7 seed and build targets pass at 2–4 ranks);
     - but then PT-Scotch's 4-rank cut on the mixed box is 1.136 × Scotch's, above the 1.10
       bound of `partition_quality` (ADR-0032), against 1.0509 with 7.0.14.

   So 7.0.18 changed both the deterministic mode and the partitions. Whether this is Scotch's
   change or Homebrew's build of it is not known yet.

   There is also a lookup problem: gridcl cannot choose 7.0.14 while Homebrew's is installed.
   - cubecl-llvm's build script emits `-L /opt/homebrew/lib`, ahead of ptscotch-sys's
     `SCOTCH_DIR`.
   - scotch-sys 0.2.1 emits no search path at all.
   - So the link takes Homebrew's `libscotch` even with `SCOTCH_DIR`, `CPATH` and
     `LIBRARY_PATH` pointing at 7.0.14.
   - For now each command loads 7.0.14 at run time by its leaf name, `libscotch.7.0.dylib`
     (same 7.0 ABI), with `DYLD_LIBRARY_PATH`. MPI runs pass it on with `-x`, because SIP
     strips `DYLD_*` when `mpi-test.sh` starts through `/usr/bin/env bash`.
4. **MPI hangs on the M6 (hwloc).** hwloc 2.15.0's darwin backend never finishes topology
   discovery on three perflevels (https://github.com/open-mpi/hwloc/issues/852, open).
   - `lstopo` and every Open MPI `MPI_Init`, singletons included, spin at 100% CPU.
     `timeout 30 lstopo-no-graphics` exits 124.
   - With `HWLOC_COMPONENTS=-darwin`, hwloc returns at once, but with 12 PUs and no cores.
     Open MPI's by-core mapping then finds no slots.
   - What works: `--map-by slot` with `--oversubscribe`, or `--map-by slot --host localhost:4`
     without it (env-check).
   - With these, everything MPI passes:
     - the 4-rank allreduce;
     - `gridcl-parallel --features mpi` (60 tests);
     - `mpi-test.sh`: 24 of 24 runs on Metal and 6 of 6 on the CPU runtime.
   - The maintainer decided against a global setting (`~/.zshenv`). For now each local MPI
     command sets both by hand.
   - This is tooling, not design, but the design's local MPI rule depends on it.

What does not need a change:
- **Memory.** 24 GB is enough.
  - The 4-rank `partition_quality` on the 10^6-cell tetrahedral box (1,053,696 cells) peaks at
    1.06–1.13 GB resident per rank, about 4.3 GB in total.
  - From D4 to D6, the swap-out count stayed at 270189 and free memory at 79–81%.
- **Build times** are close to the M3 Max's:
  - env-check clean release build 91 s (M3 Max 81 s);
  - `cargo test --workspace` 938 s with build, Metal 400 s, `scalar` 414 s, `widest` 537 s.
- **The vector-size class** stays the same. The CPU runtime reports `load_width` 128 and
  `io_vector_sizes_f32` 16…1 (f64 8…1), as on the M3 Max: CpuSimd128.

## Options

### A. The local machine in the design

1. **Keep the design as written.** The design keeps naming the M3 Max while all local work
   runs elsewhere, and `CLAUDE.md` contradicts the machine Claude runs on. Not chosen.
2. **Replace "M3 Max" with "M6 Mac mini" everywhere.** Exact, but the next machine change needs
   another ADR to rename it again.
3. **Name the role in the design, and the machine in `docs/dev-environment.md`** (proposed).
   - The design says "the local Apple-silicon Mac (macOS, arm64)".
   - `docs/dev-environment.md` names the current machine (M6 Mac mini, 24 GB, 2 + 4 + 6
     cores) and keeps the M3 Max's records as history.
   - Done milestones' acceptance text (E0, M3s) stays as written, because it records where it
     was met.

### B. The performance-core count (ADR-0017)

1. **Keep `hw.perflevel0.physicalcpu`.** On the M6 the `performance` families run on 2 of 12
   cores. Not chosen.
2. **Match level names** ("Performance", "Super"). This depends on Apple's naming, which
   already changed once. Not chosen.
3. **Every level except the last** (proposed).
   - The count is the sum of `hw.perflevel<i>.physicalcpu` for i < n − 1, where
     n = `hw.nperflevels` ≥ 2. The last level is the efficiency level.
   - With n < 2, or on any error, there is no count, as now.
   - This gives 12 on the M3 Max (unchanged), 2 + 4 = 6 on the M6, and the same for any number
     of levels.
   - The names are only a cross-check in tests.

### C. Scotch

1. **Keep "Scotch 7.x" and Homebrew's current Scotch.** `partition_threads` and the M7 MPI
   targets fail on the local machine. Not chosen.
2. **Run Scotch with one thread** (a context with one thread, through
   `SCOTCH_contextThreadSpawn`). This restores determinism with 7.0.18, but the M7 acceptance
   still fails (PT-Scotch 1.136 × Scotch's cut at 4 ranks). It may also slow Scotch on every
   build. Not chosen.
3. **Pin the verified Scotch versions, and use the built 7.0.14 locally** (proposed).
   - The design's PT-Scotch row says "Scotch 7.0.x; verified with 7.0.4 (hosted CI, Ubuntu
     24.04) and 7.0.14 (local and Kathleen, `scripts/hpc/build-partition-libs.sh`); 7.0.18 is
     not deterministic (ADR-0034)".
   - On the local Mac, Homebrew's `scotch` (7.0.18) stays installed, and every Scotch build and
     run uses the 7.0.14 from `<prefix>/scotch-7.0.14`. Kathleen already uses that build.
     - The build sets `SCOTCH_DIR=<prefix>/scotch-7.0.14`,
       `CPATH=<prefix>/scotch-7.0.14/include:/opt/homebrew/include` and
       `LIBRARY_PATH=<prefix>/scotch-7.0.14/lib:/opt/homebrew/lib`. The link still records
       Homebrew's `libscotch.7.0.dylib`, because cubecl-llvm's `-L /opt/homebrew/lib` comes
       first.
     - Runs set `DYLD_LIBRARY_PATH=<prefix>/scotch-7.0.14/lib`, so dyld loads 7.0.14 by its
       leaf name (same 7.0 ABI; that directory holds only Scotch's libraries). MPI runs add
       `-x DYLD_LIBRARY_PATH=…` to `GRIDCL_MPIRUN_FLAGS`, because SIP strips `DYLD_*` when a
       script starts through `/usr/bin/env bash`.
     - A command without these settings runs 7.0.18. `partition_threads`, `partition_seed` and
       `partition_build` then fail, so the mistake shows, not silently.
     - `DYLD_PRINT_LIBRARIES=1 <test binary> --list` shows which Scotch loads.
     - Verified in part D: 188 and 186 partitioner tests, and 24 + 24 M7 MPI runs on Metal,
       pass this way.
     - Uninstalling or unlinking Homebrew's `scotch` (no installed formula depends on it)
       would make the `DYLD_LIBRARY_PATH` step unnecessary. It stays an option for later.
   - The 7.0.18 behaviour is reported to Scotch's issue tracker. The pin is lifted when a later
     version passes `partition_threads` and the M7 MPI targets.

### D. The M3 Max

1. **Keep it as a second local target.** Two local rows; every change tested on both.
2. **Retire it as a target** (proposed). Its committed benchmark data and env-check records stay
   as data from that machine.

## Proposed decision

- **A3, B3, C3 and D2.** The design names the local machine by role, so a future machine change
  is a `docs/dev-environment.md` edit, not an ADR.
- **B3** keeps ADR-0017's intent (the units that do not finish last) for any number of levels.
  It gives the same count as before on two-level chips.
- **C3** keeps M7's acceptance numbers and reproducibility claim valid, and states the Scotch
  versions they were verified with, without new library behaviour.
- **The local MPI rule stays** (Metal on 1–4 ranks, the CPU runtime on 1), because the memory
  check passed.
- **Benchmarks:**
  - The committed M3 Max data and `vector-policy.toml` stay valid as that machine's data.
  - New local benchmarks run on the Mac mini and record every performance level.
  - The `cpu_units = performance` entries measured on the M3 Max stay until the Mac mini's
    `cube_dim` comparison is measured; ADR-0017's decision rule then applies as usual.

## Consequences

- **Text to change in docs/design.md:**
  - "Overview", **Development and test targets**: "a local Apple-silicon Mac (Claude Code runs
    here; CubeCL CPU runtime and Metal)"; decision 10 and "Summary of decisions" (the targets
    line): "the local Mac" for "the M3 Max".
  - Backend table: the CPU runtime and Metal rows' last column becomes "the local Mac on every
    change …". The PT-Scotch row becomes "Scotch 7.0.x (context API; verified 7.0.4 and 7.0.14;
    7.0.18 is not deterministic, ADR-0034)".
  - Vector-size table: the row "M3 Max and Locust's Grace CPU (aarch64, NEON)" becomes "Apple
    silicon (local Mac) and Locust's Grace CPU (aarch64, NEON)", with the same values.
  - **CPUs** bullet of `LaunchPlan`: "…or the performance-core count (macOS: every
    `hw.perflevel` except the last, ADR-0034) where …".
  - "Benchmarks":
    - **Where:** "the local Mac (CPU runtime as CpuSimd128, and Metal)".
    - **Conditions:** "On the local Mac they run on mains power with other applications
      closed, and the report records every performance level by name with its core counts
      (`hw.nperflevels`, `hw.perflevel<i>.{name,physicalcpu}`). With static blocks, the cores
      of the last (efficiency) level can finish last; …".
  - "Partitioning", **Reproducibility**: one sentence: "Scotch 7.0.18's threaded build is not
    deterministic even in this context; the verified versions are listed in the backend table
    (ADR-0034)."
  - "Testing and validation": the targets table row "M3 Max, local" becomes "Local Mac (macOS,
    arm64)". In the CI matrix, the "local" row (developer machine), the conformance row ("or
    the local Mac with Accelerate") and the fallback sentence ("Without one, the local Mac
    covers …") follow suit.
  - "Development environment and test targets": the introduction ("Claude Code runs on the
    local Mac …"), the targets table row "Local Mac | Apple silicon, macOS, arm64 | local | …;
    MPI up to 4 ranks" with a pointer to `docs/dev-environment.md` for the current machine, and
    "**Remote-test runner.** … with one command from the local Mac".
  - M10's benchmark bullet: "on the local Mac (CPU runtime, Metal)".
  - Unchanged, as history: E0's scope and acceptance, M0's acceptance, and M3s's benchmark
    bullets (met on the M3 Max).
- **ADR-0017:** a line "Amended by ADR-0034 (the performance-core count on three or more
  perflevels)" in its decision record. **ADR-0031:** "Amended by ADR-0034 (Scotch 7.0.18 is not
  deterministic in the context; verified versions)".
- **CLAUDE.md:** "This machine is the M3 Max" becomes "This machine is the local Mac (currently
  the M6 Mac mini, docs/dev-environment.md)"; the rest of "Test targets" is unchanged.
- **docs/dev-environment.md** (no ADR needed, listed for the apply):
  - the targets table;
  - "Local tools on the M3 Max" and "Notes on the M3 Max" become "… on the local Mac", naming
    the M6 Mac mini; the M3 Max's lines stay as dated history;
  - the "Recorded by E0" host table gains a Mac mini column with part D's results, and keeps
    the M3 Max column;
  - the Partitioners section:
    - "Libraries" names Scotch 7.0.14 from `scripts/hpc/build-partition-libs.sh` for the local
      Mac, with Homebrew's 7.0.18 installed but not used;
    - the local test commands carry `SCOTCH_DIR`, `CPATH`, `LIBRARY_PATH` and
      `DYLD_LIBRARY_PATH` as in option C3, and the MPI ones add `-x DYLD_LIBRARY_PATH=…`;
    - a note says how to check which Scotch loads, and why the link alone does not choose it;
  - a hwloc note citing issue 852, the flags and "remove when Homebrew ships a fixed hwloc";
  - a change-history row.
- **Agent and skill files:** `.claude/agents/test-runner.md` and
  `.claude/skills/plan-milestone/SKILL.md` say "the local Mac" (outside the sandbox, as for
  ADR-0016); `scripts/check-local-tools.sh`'s header the same.
- **Code (its own small task after the apply, verified on the Mac mini; not part of M8):**
  - `gridcl-device/src/cores.rs`: rule B3 (reads `hw.nperflevels`, then the levels).
  - A unit test of the summation on fixed level lists, and the existing
    `performance_cores_match_the_os`.
  - `tools/env-check` (`report.rs`, `probe.rs`) and `gridcl-bench/src/protocol.rs`: the host
    record keeps `performance` (rule B3) and `efficiency` (the last level) for comparison with
    old records, and gains `levels: [{name, physical, logical}]`. env-check's report schema
    version goes up.
- **Tooling, without an ADR, with or before the apply:**
  - `scripts/mpi-test.sh`: on macOS with `hw.nperflevels` ≥ 3, it sets
    `HWLOC_COMPONENTS=-darwin` for its runs and adds `--map-by slot`, so the documented
    commands work unchanged. This is per script, not global, as the maintainer asked. Singleton
    tests and env-check's MPI test still need both on the command line, documented in
    `docs/dev-environment.md`.
  - `scripts/setup-worktree.sh` creates `.venv` with `python3.14` when it exists, else
    `python3`, and says so. Homebrew's default `python3` is 3.15, for which vtk 9.7.1 has no
    wheel.
  - `scripts/hpc` login-node fetch step: `CARGO_BUILD_JOBS=8`.
    - With the default parallelism (80 on Kathleen's login node), `rust-lld` died with SIGBUS 3 of
      3 times while linking build scripts into `target/kathleen-login` on Lustre.
    - Not quota or space: 3% of quota used, OSTs at 7%, cgroup peak 2 GB of 30 GiB, no OOM.
    - `-j 8` passes in 68 s.
- **Public API changes:** none. `performance_cores()` is crate-private; `with_performance_cores`
  is unchanged.
- **Acceptance tests affected:** none change.
  - `partition_threads`, `partition_seed`, `partition_build` and `partition_quality` pass
    unchanged with Scotch 7.0.14.
  - ADR-0017's tests keep their meaning: `cpu_plan_one_unit_per_thread_whole_lines` still
    injects the count through `with_performance_cores`.
- **Later milestones affected:**
  - M8 starts on the Mac mini: local MPI with the hwloc flags, HDF5 2.2.0 (Homebrew
    `hdf5-mpi`, was 2.0.0), Scotch 7.0.14. Its 8-rank 10^7-cell memory acceptance runs on
    Kathleen as planned.
  - M10's benchmark re-run measures the Mac mini, including the `cpu_units` comparison
    on 6 performance cores.
- **Open (outside this ADR):**
  - the Kathleen CPU smoke job from the Mac mini has not run yet: job 617653 waited 3 h in the
    queue and was deleted at `--max-wait`; the Locust CUDA smoke passed (job
    20261101231105-33368, idle before and after);
  - `/opt/homebrew` is owned by the old account, so `brew` cannot install, upgrade or remove
    formulae as the account Claude runs as until the maintainer changes the owner. Option C3 does not depend on
    it.
  - A Homebrew upgrade of `scotch` to 7.1 would break the leaf-name match. Binaries would then
    fail to load, not run 7.1; the setup would be revisited then.

## Decision record

Accepted by the maintainer on 2026-11-02, as proposed (A3, B3, C3 with Homebrew's Scotch left
installed, D2), including `scripts/mpi-test.sh` setting the hwloc workaround itself. Applied in
`adr: 0034 apply`. When applying, the sentence "MPI runs beyond the M3 Max's 4 ranks" in
"Development environment and test targets" also became "the local Mac's 4 ranks". The code change
(rule B3 and the host records) is a separate task after the apply.
