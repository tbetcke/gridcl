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

### C13 Pools keep empty pages until an explicit cleanup

- **Version:** cubecl-server 0.11.0 (CPU runtime, `Adaptive` dynamic memory).
- **Observation:** the arena's current pool returns a page that has emptied only on `Client::memory_cleanup` (`Cleanup::Explicit`) or after its own deallocation period, which `MemoryManagement::reserve` checks on each reservation (`src/memory_management/dynamic/arena.rs`, `cleanup` and `drain`; `memory_manage.rs`, `reserve`). Its pages follow the largest allocation served so far (`dynamic/adaptive.rs`). A pipeline whose steps allocate buffers of different sizes therefore holds the empty pages of earlier steps beside the pages of the current one. On the CPU runtime the pages are host memory, so they count toward the process's resident memory.
- **Evidence:** M8 T5b, `read_memory` on Kathleen (10^7-cell box, 8 ranks). During face creation the pools reserved about 340 MB while about 230 MB was in use. Kathleen jobs 628342, 628348 and 628349: without cleanup, a peak of 510–520 MiB per rank; with cleanup after each entity dimension, 495–514 MiB; with cleanup also after the sort and unique steps of entity creation, 470–485 MiB.
- **gridcl workaround:** best-effort `Client::memory_cleanup` after the sort and unique steps of entity creation when mesh creation creates the entities (the hidden `Topology::create_entities_releasing_pools`, gridcl-topology `entity.rs`), and after each step of stage 10 of mesh creation (`gridcl/src/pipeline/tags.rs`). `Topology::create_entities` does not clean up: each cleanup waits for the device and frees cached pages, which slowed entity creation by about 2× on CUDA (M8 review).
- **Suggestion:** a pool setting that returns an emptied page of the current pool after a threshold (bytes or allocations), or document that long pipelines should call `memory_cleanup` between phases.
- **Status:** open.

### C14 Device-channel clients spin while the queue is full

- **Version:** cubecl-common 0.11.0 (every runtime; observed on the CPU runtime).
- **Observation:** a device has one server thread and a queue of `CHANNEL_MAX_TASK` = 32 tasks. A client that finds it full retries `available_index.fetch_add` in a loop: `SPIN_BUDGET_CLIENT` = 8192 × 64 = 524 288 `spin_loop` iterations, then 4096 `yield_now` calls, then 75 µs sleeps (`src/device/handle/channel.rs`, `DeviceClient::enqueue`). Every waiting client hits the same atomic. When the threads that submit to one device outnumber the cores, they take the CPU from the server thread that would empty the queue, and throughput collapses.
- **Evidence:** gridcl's `io_threads` tests (`ThreadComm`, a thread per rank, all submitting to the one CPU-runtime device) on the M6 Mac mini (12 cores): about 37 rank threads at once ran at 1100% CPU for over 30 min with one of eight `global_mesh_identical` tests done. `sample` of the process: 8533 on-CPU samples in `ChannelDeviceHandle::submit_inner`/`submit_blocking` and `swtch_pri` (yield), 3797 in kernel code (2026-11-05).
- **gridcl workaround:** `ThreadComm::run` (gridcl-testing `comm.rs`) limits the rank threads of a process to half the hardware threads; a run waits until enough others have finished. `io_threads` then took 311 s (808 s with the hardware threads as the limit).
- **Suggestion:** a short spin, then block on a condition variable or futex that the server signals after swapping buffers; or a backoff that grows per client.
- **Status:** reported, [tracel-ai/cubecl#2066](https://github.com/tracel-ai/cubecl/issues/2066) (2026-11-05), with a standalone reproducer: on 12 cores, 4779 launches/s from one thread and 245 from 36.

### C15 Compound assignment to a variable initialised from a comptime value fails at JIT

- **Version:** cubecl-core and cubecl-macros 0.11.0 (observed on the CPU runtime).
- **Observation:** in a `#[cube]` kernel, `let mut word = cycle;` with `cycle` a `#[comptime] u32` parameter, followed by `if comptime!(has_perm) { word ^= perm[c]; }`, compiles in Rust but fails when the kernel is compiled at launch: "the lowered module does not verify: Compilation error: verification failed. Expected operand type llvm.ptr, but found builtin.integer". The compound assignment ends in `assign_binop_expand` (cubecl-core `src/frontend/operation/base.rs`), which stores into `lhs.expand`; a variable initialised from a comptime value is apparently a constant rather than a mutable local, so the store targets an integer. The same kind of mix with a plain assignment (`let mut half = w;` with `w` comptime, then `half = 0usize;` in a runtime branch) is caught by rustc ("expected `usize`, found `NativeExpand<usize>`"). The variable initialised from a literal (`let mut slot = 1usize; if … { slot = 0usize; }`) works. Separately, the reported error names only a later kernel ("skipped `TableKernel`: it needed memory …, which carried the failure"), not the kernel that failed to compile, so finding it took host reads after each launch.
- **Evidence:** M9 T7, `gridcl-geometry/src/orient/parity.rs` (`sides_kernel`), test `orient::parity_is_zero_between_consistent_cells_and_one_across_flips`; isolated by restoring each of the two changed patterns in turn (2026-11-06).
- **gridcl workaround:** initialise such variables from a literal (`let mut word = 0u32; if comptime!(has_perm) { word = perm[c]; }`) and combine with the comptime value in a new expression (`(word ^ cycle) >> e`).
- **Suggestion:** have the macro make a `let mut` binding a mutable local even when its initialiser is comptime (or reject the compound assignment at compile time, as for plain assignment); name the kernel whose compilation failed in the error that dependent launches carry.
- **Status:** open.

### C16 Float folds apply to one-lane values only, so multiply-add fusion differs by vector size

- **Version:** cubecl-ir and cubecl-opt 0.11.0 (observed on the CPU runtime).
- **Observation:** the simplifications of `simplify!(FAddOp | FSubOp | FMulOp | FDivOp)` (cubecl-ir `src/dialect/math.rs`: 0 + x, x + 0, x − 0, 1 · x, x · 1, x / 1) and the constant evaluation of SCCP fire only where the constant operand is a one-lane float (`float_as_f64`; the `x − x` rule says "Only for one lane"), not for `Vector::broadcast` constants of more lanes. `InstCombine` (cubecl-opt `src/passes/inst_combine.rs`) then fuses each add or subtract with its left lone product, else its right one. A kernel generic over `N: Size` therefore reaches the fusion pass with different operations at N = 1 and N > 1, and fuses different products: in `a · 1 + c · d`, N = 2 fuses the exact `a · 1` and rounds `c · d`, while N = 1 drops the `· 1` and fuses `c · d`. Results at vector size 1 and 2 then differ in the last bits, deterministically.
- **Evidence:** M9 T10a, the device basis evaluator (`gridcl-geometry/src/basis_eval/`), test `basis_eval::tests::every_vector_size_gives_the_scalar_result`: a tetrahedron p = 3 in f32 differed in three near-zero derivative entries (−5.0e−8 at N = 1 against −3.5e−8 at N = 2). With `CUBECL_DEBUG_PLIRON`, the N = 1 kernel had 153 float constants after `sccp` against 112 at N = 2, lost 12 multiplies and 5 adds in `simplify_ops_rewrite` (none at N = 2), and `inst_combine_rewrite` made 125 FMAs against 130 (2026-11-06).
- **gridcl workaround:** kernels that must be bitwise equal across vector sizes use no float constant 0 or 1 and no operation on constants alone: `basis_eval` reads a runtime 0 and 1 from the end of its coefficient buffer and builds every other constant as an integer or half-integer times a runtime value (module docs of `basis_eval`). The factor kernels never met this: their sums start from a product and add products of runtime values.
- **Suggestion:** apply the float identities and constant folding to vector constants as to scalars (a broadcast is a constant), or skip them for floats when contraction is on; and document that fusion depends on operand order and on these folds.
- **Status:** open.

### C17 CPU runtime on aarch64 (64 K pages): intermittent abort or SIGSEGV at process exit, after the last test

- **Version:** cubecl-cpu and cubecl-llvm 0.11.0 (bundled LLVM 23.1.0) on Locust's Grace CPU (aarch64 with SVE, kernel `5.14.0-362.18.1.el9_3.aarch64+64k`, 64 KiB pages).
- **Observation:** the release lib test binary of `gridcl-testing` run with `--test-threads=1` passes all 49 tests and then dies in about half the runs: SIGSEGV, or SIGABRT with `LLVM ERROR: Cannot implicitly convert a scalable size to a fixed-width size in 'TypeSize::operator ScalarTy()'` on stderr. With libtest's default thread count it never failed (0 of 6), and no single test fails alone (0 of 8 each). The other eleven lib test binaries the check reached (`gridcl`, `gridcl-device`, `gridcl-geometry`, `gridcl-io`, `gridcl-parallel`, `gridcl-partition`, `gridcl-refcell` among them) never failed in 10 sequential runs each. It also never failed when a SIGSEGV handler was already installed before `main` (an `LD_PRELOAD` library; then std installs neither its stack-overflow handler nor per-thread alternate signal stacks): 0 of 15, against 8 of 15 in alternating runs without it. The core dumps go to systemd-coredump, which an unprivileged user cannot read, so there is no backtrace. At the M9 review (2026-11-08), 10 sequential runs per mode: 4 failed as built; 7 with SVE off in the JIT (a patched pliron-llvm 0.19.0 `LLVMLLJIT::new_with_default_builder` that gives LLJIT a target machine for the host CPU `neoverse-v2` with the host features followed by `-sve,-sve2`, confirmed on stderr); 5 with no LLJIT ever disposed (each kernel compile creates its own LLJIT, cubecl-llvm `src/cpu/jit/engine.rs:80`, held in `JitKernel::_lljit`). So neither SVE code generation nor the disposal of the JITs causes it. One run as built printed `LLVM ERROR: Cannot select: … v4i32 = insert_vector_elt …` and still exited with status 0, which suggests another thread was still compiling while the main thread exited normally. With an exit handler that calls `_exit(0)`, registered after every kernel compile (the patched `LLVMLLJIT::lookup_symbol`; handlers run newest first, so it runs before the destructors of LLVM's statics), the crash is gone: 0 of 15 runs, against 3 of 15 as built in alternating runs (all 49 tests passed in all 30). The crash therefore happens in the exit-time destructors. No backtrace yet: under gdb the program never started ("During startup program exited with code 127", with and without gdb's startup shell; gdb's child cannot enable tracing: `ptrace: Operation not permitted`). T12's gdb check (job 20261107222419-2489096), until now recorded here as "not reproduced under gdb (0 of 8)", almost certainly failed the same way: its output ends after `which gdb`, and the loop counted only logs containing SIGSEGV. Locust sets `kernel.yama.ptrace_scope` to 3 (no process may trace another; only a reboot lowers it), so neither gdb nor strace works there, even on `/bin/true`; jobs run without seccomp or no_new_privs.
- **Evidence:** M9 T12, Locust jobs 20261107213233-2466394 (the workspace run, `--test-threads=1`: 1059 passed, then the SIGSEGV), 20261107222224-2483467 (3 of 6 sequential runs fail, 0 of 6 parallel), 20261107222640-2492655 (test subsets), 20261107223235-2521463 (per binary), 20261107230628-2570934 (with and without the handler), 20261107230846-2581972 (the LLVM message) (2026-11-07); M9 review, Locust jobs 20261108020857-2608509 (the workspace run, 1061 passed, then the SIGSEGV), 20261108030224-2635812 (as built, SVE off, no disposal), 20261108030714-2655289 and 20261108030805-2656833 (the SVE-off target machine confirmed; gdb), 20261108081058-2678230 (the `_exit` handler), 20261108081941-2690014 (ptrace) (2026-11-08). Not seen on x86-64 (Kathleen, hosted CI) or macOS.
- **gridcl workaround:** none in code. The aarch64 review run uses libtest's default thread count for `gridcl-testing`'s lib tests.
- **Suggestion:** SVE is ruled out, and the crash is in the exit-time destructors (above). Check whether the static thread pool (`cubecl-cpu` `src/compute/threadpool/mod.rs`, a `OnceLock` never torn down) or a kernel compile on another thread can still use LLVM while the process exits and LLVM's static state is destroyed.
- **Status:** open (to report with a backtrace: a SIGSEGV handler installed after start-up that prints the stack and re-raises (one installed before `main` hid the crash), or core dumps from the Locust admins).

### C18 CUDA: a large unrolled kernel takes minutes to load and then fails to launch with an anonymous `CUDA_ERROR_INVALID_VALUE`

- **Version:** cubecl-cuda 0.11.0 on Locust (GH200, Hopper GPU, sm_90).
- **Observation:** the first version of the device basis evaluator (M9 T10a) kept the whole orthogonal set of a cell in a local array, fully unrolled. On CUDA, compiling and loading one kernel took about 200 s for a hexahedron p = 4 (f64, vector size 2), 28 s for a prism p = 4 and 14 s for a hexahedron p = 3; the f64 hexahedron p = 4 kernel at vector size 2 then failed to launch with `DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")` under "An unknown error happened during launch". cubecl-cuda compiles with NVRTC without a register limit or launch bounds (`src/compute/context.rs`: `--gpu-architecture=sm_<arch>`, sm_90a on Hopper, the CUDA and, if present, CCCL include paths, and `-lineinfo`) and loads the PTX with `cuModuleLoadData`, so the driver's PTX-to-SASS step takes the kernel as it is. `launch_failed` (`src/compute/context.rs`) drops the name of the failing call (`cuFuncSetAttribute` or `cuLaunchKernel`) for errors that do not poison the device, and nothing names the kernel or its register or local-memory use. The cause (registers, local memory or a launch limit) was not isolated.
- **Evidence:** M9 T10a, Locust jobs 20261107001156-1923544 (commit 81e8739: the lib tests took 1835 s and failed at the launch) and 20261107004359-1931288 (a probe timing each (cell, degree, family, vector size): `Hexahedron p=4 Equispaced n=2` done after 191 s with the launch error; p = 3 took 11.8 s) (2026-11-07). Not seen on the CPU runtime or Metal.
- **gridcl workaround:** the evaluator adds each basis function into the (1 + tdim) sums of its outputs as the recurrences produce it (`gridcl-geometry/src/basis_eval/`: `expansion`, `emit`), so it holds no array of basis values; the lib tests then take 14.6 s on Locust and every launch succeeds.
- **Suggestion:** keep the name of the failing driver call in `LaunchError::Unknown`, and name the kernel; after loading, read `CU_FUNC_ATTRIBUTE_NUM_REGS`, `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` and `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` and reject a cube dim above the last with a clear error; document that large unrolled kernels can take minutes in the driver's JIT.
- **Status:** open (not reproduced since the rewrite).
