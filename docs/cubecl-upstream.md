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
- **Observation:** the release lib test binary of `gridcl-testing` run with `--test-threads=1` passes all 49 tests and then dies in about half the runs: SIGSEGV, or SIGABRT with `LLVM ERROR: Cannot implicitly convert a scalable size to a fixed-width size in 'TypeSize::operator ScalarTy()'` on stderr. With libtest's default thread count it never failed (0 of 6), and no single test fails alone (0 of 8 each). The other eleven lib test binaries the check reached (`gridcl`, `gridcl-device`, `gridcl-geometry`, `gridcl-io`, `gridcl-parallel`, `gridcl-partition`, `gridcl-refcell` among them) never failed in 10 sequential runs each. It also never failed when a SIGSEGV handler was already installed before `main` (an `LD_PRELOAD` library; then std installs neither its stack-overflow handler nor per-thread alternate signal stacks): 0 of 15, against 8 of 15 in alternating runs without it. The core dumps go to systemd-coredump, which an unprivileged user cannot read, so there is no backtrace. At the M9 review (2026-11-08), 10 sequential runs per mode: 4 failed as built; 7 with SVE off in the JIT (a patched pliron-llvm 0.19.0 `LLVMLLJIT::new_with_default_builder` that gives LLJIT a target machine for the host CPU `neoverse-v2` with the host features followed by `-sve,-sve2`, confirmed on stderr); 5 with no LLJIT ever disposed (each kernel compile creates its own LLJIT, cubecl-llvm `src/cpu/jit/engine.rs:80`, held in `JitKernel::_lljit`). So neither SVE code generation nor the disposal of the JITs causes it. One run as built printed `LLVM ERROR: Cannot select: … v4i32 = insert_vector_elt …` and still exited with status 0, which suggests another thread was still compiling while the main thread exited normally. With an exit handler that calls `_exit(0)`, registered after every kernel compile (the patched `LLVMLLJIT::lookup_symbol`; handlers run newest first, so it runs before the destructors of LLVM's statics), the crash is gone: 0 of 15 runs, against 3 of 15 as built in alternating runs (all 49 tests passed in all 30). The crash therefore happens in the exit-time destructors. No backtrace yet: under gdb the program never started ("During startup program exited with code 127", with and without gdb's startup shell; gdb's child cannot enable tracing: `ptrace: Operation not permitted`). T12's gdb check (job 20261107222419-2489096), until now recorded here as "not reproduced under gdb (0 of 8)", almost certainly failed the same way: its output ends after `which gdb`, and the loop counted only logs containing SIGSEGV. Locust sets `kernel.yama.ptrace_scope` to 3 (no process may trace another; only a reboot lowers it), so neither gdb nor strace works there, even on `/bin/true`; jobs run without seccomp or no_new_privs. A crash handler installed at the first JIT creation (after std's start-up; it prints the crashing thread's stack with glibc's `backtrace` and re-raises) gave the backtrace: the crashing thread is cubecl's device channel server thread (`cubecl-common` `src/device/handle/channel.rs`, module `custom_channel`: spawned at :887, never joined; `Server::start` at :1051 → `ChannelDeviceHandle::submit_inner` → `CpuServer::launch` → `PlironCompiler::compile` → `PlironEngine::compile` → `LLVMOrcLLJITLookup`), inside LLVM's instruction selection of a kernel: SIGSEGV at address 0x8 in `EVT::isExtendedVector` (`VectorLegalizer::LegalizeOp`) or `EVT::isExtendedFixedLengthVector` (`DAGCombiner::visitSIGN_EXTEND_INREG` → `FoldConstantArithmetic`), or the abort of `report_fatal_error` from `SDNode::print_types` → `EVT::getEVTString` (the "scalable size" message). So a kernel compile is still running on the server thread when the main thread returns from `main`, and the exit-time destructors destroy LLVM state it is using (codegen's type tables, presumably), which explains the changing symptoms. With the handler the runs exit with status 0 (3 crash reports in 20 runs, against 13 of 20 plain runs failing): the handler delays the crashing thread, and the main thread's `exit` ends the process first. The same explains the `LD_PRELOAD` handler result above, and the `LLVM ERROR` with status 0. The compile in flight comes from `gridcl-testing`'s `meshes::sphere::tests::curved_meshes_build`: its mesh builds submit eight `permutation_info_kernel` launches (`gridcl-topology/src/permutation.rs`) and the test reads nothing from the device, so it returns with them queued. In every traced run the compile of one variant starts about 0.6 s later, while libtest runs the last 14 tests (host only, milliseconds), and never ends. Skipping that test removes the crash (0 of 20 runs, against 6 of 20). Later tests that read the device do not drain that work, presumably because each test thread has its own stream.
- **Evidence:** M9 T12, Locust jobs 20261107213233-2466394 (the workspace run, `--test-threads=1`: 1059 passed, then the SIGSEGV), 20261107222224-2483467 (3 of 6 sequential runs fail, 0 of 6 parallel), 20261107222640-2492655 (test subsets), 20261107223235-2521463 (per binary), 20261107230628-2570934 (with and without the handler), 20261107230846-2581972 (the LLVM message) (2026-11-07); M9 review, Locust jobs 20261108020857-2608509 (the workspace run, 1061 passed, then the SIGSEGV), 20261108030224-2635812 (as built, SVE off, no disposal), 20261108030714-2655289 and 20261108030805-2656833 (the SVE-off target machine confirmed; gdb), 20261108081058-2678230 (the `_exit` handler), 20261108081941-2690014 (ptrace), 20261108082259-2691108 (the backtraces), 20261108082946-2707203 and 20261108083347-2715443 (compile and launch traces), 20261108083522-2718495 (the test skipped), 20261108085344-2768015 (the workaround: 0 of 31 sequential `cargo test` runs fail) and 20261108085607-2780276 (the same without it: 11 of 30) (2026-11-08). Not seen on x86-64 (Kathleen, hosted CI) or macOS.
- **gridcl workaround:** `curved_meshes_build` ends with `client.sync()` (`gridcl-testing/src/meshes/sphere.rs`), so its launches finish before the test returns. A test that launches work and reads nothing back must sync before it returns.
- **Suggestion:** the device channel's server thread is never stopped or drained, so work submitted before `main` returns (here a kernel compile) can run during the exit-time destructors. Drain or join the server at exit (or make process exit wait for in-flight compiles) in `cubecl-common`/`cubecl-cpu`; LLVM's codegen statics are not safe to use while they are destroyed. Reproducer without gridcl code (Locust job 20261108090515-2819124): one kernel launched and synced (so LLVM's codegen state exists), then 64 not-yet-compiled variants of a small scalar kernel launched, a sleep, and a return from `main` without a sync: 14 of 40 runs crash with a 60 ms sleep and 7 of 40 with 200 ms (1 of 40 at 30 ms, 0 of 40 at 100 ms: it depends on the compile phase the exit meets); with a sync before the return, 0 of 40.
- **Status:** reported, [tracel-ai/cubecl#2101](https://github.com/tracel-ai/cubecl/issues/2101) (2026-11-08), with the reproducer; worked around in gridcl.

### C18 CUDA: a large unrolled kernel takes minutes to load and then fails to launch with an anonymous `CUDA_ERROR_INVALID_VALUE`

- **Version:** cubecl-cuda 0.11.0 on Locust (GH200, Hopper GPU, sm_90; driver 565.57.01, CUDA 12.7), default LLVM path.
- **Observation:** the first version of the device basis evaluator (M9 T10a, commit 81e8739) kept the whole orthogonal set of a cell in a local array, fully unrolled. cubecl-cuda 0.11 lowers kernels through its default LLVM path (pliron and LLVM's NVPTX back end, `src/compiler.rs`: `CudaBackend::Llvm` unless the `cpp` feature selects NVRTC; gridcl enables `cuda-cpp` only in `tools/env-check`) to PTX for the newest PTX version the driver loads (8.5 here), with `.reqntid` set to the cube dim, and loads it with `cuModuleLoadData` (`src/compute/context.rs`, `load_ptx`), so the driver's JIT compiles it to SASS. The unrolled evaluator becomes 3.6 to 7.3 MB of PTX (62,000 to 124,000 FMAs at hexahedron p = 4) with every basis value an SSA register; `ptxas` keeps 32 to 40 registers per thread and spills the rest to a per-thread stack frame (M9h T2, `ptxas -v` of the dumped PTX, `CUDA_CACHE_DISABLE=1`):

  | Kernel (`lagrange_kernel`, equispaced) | First launch | `ptxas` | Registers | Stack frame | Spill stores | Launch |
  | --- | --- | --- | --- | --- | --- | --- |
  | hexahedron p = 3, f64, n = 1 | 15 s | 8 s | 32 | 126,976 B | 158,852 B | ok |
  | hexahedron p = 3, f64, n = 2 | 27 s | 24 s | 32 | 243,632 B | 276,532 B | `CUDA_ERROR_OUT_OF_MEMORY` |
  | hexahedron p = 4, f32, n = 1 | 179 s | 36 s | 32 | 22,384 B | 67,728 B | ok |
  | hexahedron p = 4, f32, n = 2 | 261 s | 76 s | 32 | 67,016 B | 105,756 B | ok |
  | hexahedron p = 4, f64, n = 1 | 180 s | 89 s | 32 | 433,416 B | 555,856 B | `CUDA_ERROR_OUT_OF_MEMORY` |
  | hexahedron p = 4, f64, n = 2 | 360 s | 285 s | 40 | 777,400 B | 899,576 B | `CUDA_ERROR_INVALID_VALUE` |
  | prism p = 4, f64, n = 2 | 48 s | 44 s | 32 | 344,776 B | 387,004 B | ok |

  `compute-sanitizer` names `cuLaunchKernel` as the failing call in each case. **Cause:** local memory, not registers or the block size. The f64 hexahedron p = 4 kernel at vector size 2 needs a 777,400 B stack frame per thread, above CUDA's limit of 512 KiB of local memory per thread, and the launch is refused with `CUDA_ERROR_INVALID_VALUE`. Frames below the limit launch only if the driver can reserve the frame for every resident thread of the device: in this run another user's process held the GPU, and the 244 KB and 433 KB frames failed with `CUDA_ERROR_OUT_OF_MEMORY` (in M9 T10a's run they had launched). The minutes of first-use time are the driver's JIT on PTX of this size (`ptxas` alone takes up to 285 s). The current direct kernels (M9h T2, hexahedron p = 4, 6, 8, f32 and f64, vector size 1, cube dim 256) stay below the limit: the largest stack frame is 51,104 B (Pinv, p = 8, f64), the factor kernels spill up to 80 KB at p = 8 in f64, the basis evaluator (reworked, below) needs at most 4,656 B, and every kernel allows at least 256 threads per block. Their first-use time grows with the PTX instead: the direct pull-back (Newton iterations around the unrolled geometry map) takes 145 s at p = 4 (255 registers, 84 KB spill in f64, `ptxas` 40 s); at p = 6 its PTX is 105 MB and neither the driver's JIT nor `ptxas` finishes within 900 s; at p = 8 the LLVM path emits no PTX within 900 s. Large f64 kernels get 32 registers from `ptxas` and spill tens of KB rather than use the 255 registers that `.reqntid 256` allows. The kernel's design (an unrolled, dynamically sized set of values per lane) is gridcl's; CubeCL's part is the diagnosis: `launch_failed` (`src/compute/context.rs`) drops the name of the failing call (`cuFuncSetAttribute` or `cuLaunchKernel`) for errors that do not poison the device, and nothing names the kernel or reports its registers, local memory or maximum threads per block.
- **Evidence:** M9 T10a, Locust jobs 20261107001156-1923544 (commit 81e8739: the lib tests took 1835 s and failed at the launch) and 20261107004359-1931288 (a probe timing each (cell, degree, family, vector size)) (2026-11-07). M9h T2, Locust job 20261108200940-3114247 (2026-11-08): the evaluator of 81e8739 on a scratch branch (never merged), one kernel per process under a 600 s timeout, with its PTX (`CUBECL_CUDA_DUMP_PTX`), `ptxas -v` of CUDA 12.6 and `compute-sanitizer` on each failure; the table above. M9h T2, Locust job 20261108225929-3171689 (2026-11-08): `scripts/bench/cuda-resources.sh --cells Hexahedron --degrees 4,6,8 --scalar-only --timeout 900` with `CUDA_CACHE_DISABLE=1`, 44 kernel records from 48 configurations (the pull-back at p = 6 and 8 timed out). Not seen on the CPU runtime or Metal.
- **gridcl workaround:** the evaluator adds each basis function into the (1 + tdim) sums of its outputs as the recurrences produce it (`gridcl-geometry/src/basis_eval/`: `expansion`, `emit`), so it holds no array of basis values; the lib tests then take 14.6 s on Locust and every launch succeeds. M9h records registers, spill and stack frame per kernel and degree with `scripts/bench/cuda-resources.sh` (ADR-0042, item 6) and runs every variant at p ≤ 8 (`degree_sweep --launch-only`, ADR-0042, item 7).
- **Suggestion:** keep the name of the failing driver call in `LaunchError::Unknown`, and name the kernel; after loading, read `CU_FUNC_ATTRIBUTE_NUM_REGS`, `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` and `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`, reject a kernel whose local memory exceeds the per-thread limit (or a cube dim above the maximum threads per block) with a clear error at load time, and make them available to the caller; document that large unrolled kernels can take minutes in the driver's JIT.
- **Status:** reported, https://github.com/tracel-ai/cubecl/issues/2108 (2026-11-09; cause isolated in M9h T2).

### C19 CUDA: `CUBECL_CUDA_DUMP_PTX` names files by the tail of the kernel id, so kernels that differ in comptime settings overwrite each other

- **Version:** cubecl-cuda 0.11.0 (default LLVM path), cubecl-runtime 0.11.0.
- **Observation:** `dump_ptx` (`cubecl-cuda/src/compute/context.rs`) writes `<name>.ptx`, where `<name>` is the kernel id's `Display` with every non-alphanumeric character replaced by `_` and cut to its last 180 characters "to stay under NAME_MAX". `KernelId`'s `Display` (`cubecl-runtime/src/id.rs`) is the `Debug` of the launch info, which lists comptime values first and the runtime arguments' names last; it omits the kernel's type name. The last 180 characters are therefore the names of the runtime arguments (for gridcl's factor kernel `..._x_offset_x_stride_out_first_tile_first_cell_last_cell_num_vectors_items_per_unit_`), and every comptime variant of a kernel (degree, cell layout, vector size) writes the same file. Only the last kernel compiled survives. The PTX entry name inside the file (`factor_kernel_f_f32_n_1_fac8fa4b`) carries the type parameters and a hash and is unique.
- **Evidence:** M9h T2, Locust jobs 20261108195835-3091086 (the dump of one degree-sweep process: five files, named by their runtime arguments) and 20261108225929-3171689 (one process launching the factor kernel of Points, Jacobian and DetJ on hexahedra at p = 1 and 2: six launches, one factor file) (2026-11-08).
- **gridcl workaround:** `scripts/bench/cuda-resources.sh` runs each (family, cell, degree, type, strategy, vector size) as its own `degree_sweep --launch-only --vector-sizes scalar|policy` process with its own dump directory, and identifies a kernel by its PTX entry name.
- **Suggestion:** name the file by the entry point (which already holds a hash of the id), or by the kernel's short name plus a hash of the full id; keep the head and the tail when shortening; document the variable.
- **Status:** reported, https://github.com/tracel-ai/cubecl/issues/2107 (2026-11-09).
