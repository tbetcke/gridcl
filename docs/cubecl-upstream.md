# CubeCL: upstream suggestions and observations

A running record of what gridcl development finds in CubeCL (and CubeK) that could be reported or
proposed upstream. Add an entry whenever a task meets a CubeCL bug, a surprising behaviour, a
missing knob or a documentation gap, with the evidence and gridcl's workaround. Entries are
observations, not decisions: anything that changes gridcl's design still needs an ADR.

**Entry format.** A heading `C<n> <short title>`, then:

- **Version:** the CubeCL (or CubeK) version and crate.
- **Observation:** what happens, with a source location or a reproducer when there is one.
- **Evidence:** the plan, ADR or job where it was found.
- **gridcl workaround:** what gridcl does about it now.
- **Suggestion:** the upstream change or report.
- **Status:** `open` (not reported), `reported` (with the issue or PR link), `fixed in <version>`,
  or `dropped` (with the reason).

Never put user ids, host names or project codes in an entry (CLAUDE.md).

## Entries

### C1 CPU runtime: no setting for the worker-thread count

- **Version:** cubecl-cpu 0.11.0.
- **Observation:** the runtime takes its worker count from `std::thread::available_parallelism()` (`src/runtime.rs:168`) and reports it as `max_units_per_cube` and `num_cpu_cores`. There is no environment variable or device option to set it; the only knob is the process's affinity mask (Linux) or cgroup quota. macOS has no affinity mask, so there the count cannot be limited at all.
- **Evidence:** M6 T14 (2–4 CPU MPI ranks on the M3 Max exceed the time limit; load average about 105 on 16 cores); M6 final run (hosted CI and Kathleen, see C2).
- **gridcl workaround:** multi-rank CPU MPI runs bind each rank (`GRIDCL_MPIRUN_FLAGS`, docs/dev-environment.md, "MPI and multi-rank tests"); on macOS, multi-rank MPI runs use Metal.
- **Suggestion:** a thread-count setting, for example `CUBECL_CPU_THREADS` or a field of the CPU device or `cubecl.toml`, that caps the pool (and `max_units_per_cube`) below the affinity mask.
- **Status:** open.

### C2 CPU runtime: workers are pinned by position in the process mask

- **Version:** cubecl-cpu 0.11.0.
- **Observation:** each worker thread pins itself to one CPU (`src/compute/threadpool/scheduler/mod.rs`, `Worker::spawn_thread` → `affinity::set_for_current`). The CPUs come from the process's affinity mask in a fixed order, one per physical core first and then SMT siblings (`src/compute/affinity/mod.rs`, `ordered_cores`). On Linux this is a hard pin (`sched_setaffinity` per thread, `affinity/linux.rs`); on macOS it is an affinity tag, which Apple Silicon rejects (`affinity/macos.rs`). When several unbound processes share a node, as MPI ranks do by default, they all see the same mask and pin their workers to the same CPUs in the same order, so they compete for the first cores while others may idle. The slowdown is far worse than proportional to the oversubscription.
- **Evidence:** M6 final run (2026-10-30). Hosted CI (4 vCPUs, unbound): `parallel_invariance` took 14 s on 2 ranks, 454 s on 3, and exceeded 600 s on 4. Kathleen, 2 nodes × 40 cores: unbound, job 614434 took 3172 s and timed out at 8 and 16 ranks; bound to 5 cores per rank, job 614468 passed everything in 217 s. ADR-0020's context says the runtime "does not pin its threads"; the 0.11.0 source does pin them on Linux.
- **gridcl workaround:** bind ranks so their masks are disjoint (CI: one hardware thread per rank; Kathleen: `--map-by node:PE=5 --bind-to core`). A planned ADR adds detection of overlapping masks among node-local ranks, and optionally divides the cores itself.
- **Suggestion:** an option to turn pinning off, or skip pinning when the process does not own its mask exclusively; at least document that co-located processes need disjoint masks.
- **Status:** open.

### C3 CPU runtime: `x % 1` with a comptime 1 folds to `x`

- **Version:** cubecl 0.11.0, CPU runtime.
- **Observation:** a remainder by a comptime 1 is folded to the dividend instead of 0, so at tile width 1 every store landed at o + v and overran the chunk.
- **Evidence:** M3 T13 (found by the tile-1 invariance case); M3 review follow-up.
- **gridcl workaround:** the factor kernels compute the lane as v − (v / vpt)·vpt.
- **Suggestion:** report with a minimal kernel.
- **Status:** open.

### C4 CPU runtime fuses multiply-adds

- **Version:** cubecl 0.11.0 (cubecl-opt `InstCombinePass` → `llvm.fmuladd`).
- **Observation:** float multiply-adds are contracted on the CPU runtime, so results differ from an uncontracted scalar reference (in the factor kernels, only in the sign of zeros).
- **Evidence:** M3 question 25, T14; ADR-0013.
- **gridcl workaround:** the tests accept the documented difference (ADR-0013).
- **Suggestion:** a compile option that disables contraction, for bitwise-reproducible kernels; document the default.
- **Status:** open.

### C5 `BufferArg::from_raw_parts` length does not bound the kernel's slice

- **Version:** cubecl-core 0.11.0 (`src/compute/launcher.rs:202`).
- **Observation:** the kernel sees `handle.size_in_used() / size_of::<C>()` elements; the `length` argument only feeds `BufferArg::len()`.
- **Evidence:** M2 question 6; ADR-0009.
- **gridcl workaround:** every kernel takes explicit lengths.
- **Suggestion:** honour the length, or document that it does not bound the slice.
- **Status:** open.

### C6 cubecl-cpu `Client::write` needs data that fills the binding

- **Version:** cubecl-cpu 0.11.0.
- **Observation:** writing fewer bytes than the allocation (for example to a padded buffer) is rejected.
- **Evidence:** M6 T4 (`DeviceBuffer::write_host`).
- **gridcl workaround:** `write_host` writes the whole allocation and zeroes the padding.
- **Suggestion:** allow a partial write from offset 0, or document the requirement.
- **Status:** open.

### C7 LLVM JIT crash (once) in `PMTopLevelManager::schedulePass`

- **Version:** cubecl-llvm 0.11.0 on macOS (CPU runtime).
- **Observation:** one SIGSEGV inside `cubecl_llvm::cpu::jit::PlironEngine::compile` in 18 workspace runs; not reproduced on reruns.
- **Evidence:** M3s review (macOS crash report `rust_out-2026-10-26-185750`).
- **gridcl workaround:** none.
- **Suggestion:** report with the crash report if it recurs.
- **Status:** open (watching).

### C8 Prebuilt LLVM bundle needs glibc ≥ 2.34

- **Version:** tracel-llvm-bundler 23.1.0-3 (cubecl-llvm 0.11.0; needed by cubecl-cuda and the CPU runtime).
- **Observation:** the downloaded LLVM does not run on older glibc (RHEL 7.9: 2.17). Building the same LLVM elsewhere relies on undocumented bundler internals (sentinel file, directory name tied to the bundler version, `llvm-config`).
- **Evidence:** E0 question 15; ADR-0004.
- **gridcl workaround:** LLVM 23.1.0 built with Spack in the bundler's layout on Myriad (ADR-0004).
- **Suggestion:** a documented way to point cubecl-llvm at a local LLVM, or a bundle for older glibc.
- **Status:** open.

### C9 wgpu fingerprints are named `spirv_…` whatever the shader language

- **Version:** cubecl-wgpu 0.11.0.
- **Observation:** on Metal (MSL), `identity.fingerprint` is `spirv_0_0`: every wgpu fingerprint is `spirv_{vendor}_{device}`.
- **Evidence:** docs/dev-environment.md, "Notes on the M3 Max" (E0).
- **gridcl workaround:** none needed.
- **Suggestion:** name the fingerprint after the backend actually used.
- **Status:** open.

### C10 CPU runtime reports no performance-core count

- **Version:** cubecl-cpu 0.11.0.
- **Observation:** on the M3 Max, `max_units_per_cube` is 16 (12 performance and 4 efficiency cores), with no way to learn the performance-core count; with static blocks, efficiency cores can finish last.
- **Evidence:** ADR-0017.
- **gridcl workaround:** gridcl-device reads `hw.perflevel0.physicalcpu` itself (ADR-0017).
- **Suggestion:** expose the core classes (performance and efficiency counts) in the CPU runtime's properties.
- **Status:** open.

### C11 CubeK lacks the device primitives gridcl needs

- **Version:** CubeK 0.3.0 (cubecl ^0.11).
- **Observation:** no scan, sort, histogram, unique or segmented reduce. cubek-reduce chooses its own cube count, ties `cube_dim` to the plane size, and is deterministic only per plane and vector configuration.
- **Evidence:** M2 question 14.
- **gridcl workaround:** gridcl-device implements its own primitives.
- **Suggestion:** offer launch-shape control (caller-chosen cube count and dim) and a deterministic mode independent of the plane size.
- **Status:** open.

### C12 MSRV declared only in the workspace manifest

- **Version:** cubecl 0.11.0.
- **Observation:** the Rust 1.95 minimum is declared in CubeCL's workspace, not in the published crates' manifests, so cargo cannot check it.
- **Evidence:** ADR-0003.
- **gridcl workaround:** the toolchain is pinned in `rust-toolchain.toml`.
- **Suggestion:** set `rust-version` in each published manifest.
- **Status:** open.
