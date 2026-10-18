# ADR-0004: Build CubeCL's LLVM with Spack on Myriad (the prebuilt bundle needs glibc ≥ 2.34)

- **Status:** Accepted
- **Date:** 2026-10-17 (revised 2026-10-18 after the Spack feasibility test; the first draft proposed an Apptainer container, now option 3)
- **Raised in:** E0 T6 (Myriad survey and smoke run)
- **Design sections affected:** "Development environment and test targets" (remote-test runner, step 3; Myriad rules); "Risks and mitigations" (the LLVM-bundle row and a new row); `docs/dev-environment.md` (Myriad reference, runner configuration)

## Context

The design makes Myriad V100 the main CUDA platform, with builds on the cluster's own operating system:

> 3. **Fetch.** It fetches crates on the login node, and the CPU runtime's LLVM bundle when needed, because compute nodes may have no internet access. …
> 4. **Submit.** … The job loads the modules, records `nvidia-smi` and the toolkit version, builds, and runs the command with `GRIDCL_TEST_DEVICE=cuda`.

("Development environment and test targets"). The risks table has "the CPU runtime downloads an LLVM bundle from github.com at build time | Fetch crates and the bundle on the login node".

The E0 T6 survey found:

- **Operating system.** Myriad's login nodes and its V100 compute nodes (probe job on `node-e00a-004`) run Red Hat Enterprise Linux 7.9 with **glibc 2.17** and kernel 3.10, and have no internet access. The system compiler is GCC 4.8.5 and the system linker GNU ld 2.27. Modules provide GCC 10.2, CUDA 12.2.2 and cmake 4.1.2. The V100 nodes have driver 550.127.05 (CUDA 12.4).
- **CubeCL needs LLVM for CUDA.** In cubecl-cuda 0.11.0, `cubecl-llvm` (features `std`, `nvptx`) is a **non-optional** dependency. Both env-check paths build it, `cuda` (LLVM → PTX) and `cuda-cpp` (NVRTC), and so does cubecl-cpu (the CPU runtime).
- **The bundle is prebuilt.** cubecl-llvm takes LLVM from `tracel-llvm-bundler` 23.1.0-3. That is a prebuilt archive (`linux-x64.tar.xz`, 318 MB) from github.com/tracel-ai/tracel-llvm, unpacked to `$XDG_DATA_HOME/tracel/tracel-llvm-23.1.0-3`. It contains LLVM 23.1.0 (Release, static, RTTI, no assertions), 122 LLVM and 6 lld static libraries, `llvm-config` and the lld binaries; no clang, no MLIR. There is no switch for a system LLVM. However, its build script returns early, without download or checksum, when `<that directory>/.tracel-llvm-installed` exists, and then reads everything it links through that directory's `llvm-config` (`--libs`, `--system-libs`, `--includedir`, `--libdir`).
- **Apptainer.** Apptainer 1.2.4 is installed on the login and compute nodes (setuid mode). User namespaces are disabled, so images cannot be built from a definition file on Myriad, only converted from existing images.

## Problem

The runner's fetch step on the Myriad login node (`cargo check --manifest-path tools/env-check/Cargo.toml --features cuda`, 2026-10-17) fails in cubecl-llvm's build script:

```text
…/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.33' not found
…/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.34' not found
…/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.32' not found
error: failed to run custom build command for `cubecl-llvm v0.11.0`
```

The bundle's static libraries also reference `__libc_single_threaded` (glibc ≥ 2.32) 242 times, so linking against glibc 2.17 would fail too. As designed, no CubeCL CUDA program, and no CPU-runtime program, can be built on Myriad. That covers the E0 V100 and A100 acceptance items and the AVX-512 CPU benchmarks of M3s.

## Options

1. **Keep the design as written.** Impossible without replacing CubeCL's LLVM; see the other options.
2. **Build LLVM 23.1.0 with Spack on Myriad and install it in the bundler's layout.** Tested on 2026-10-17/18; the details:
    - **Spack:** v1.2.2 in the remote home. A one-file overlay repo subclasses the builtin `llvm` recipe, adding version 23.1.0 (sha256 recorded) and `LLVM_ENABLE_ZLIB=OFF`, as in tracel's bundle. The spec is `llvm@23.1.0 +lld ~clang ~lldb ~polly ~lua ~gold ~mlir ~flang ~libomptarget ~offload ~python ~z3 ~zstd ~cuda ~llvm_dylib compiler-rt=none libcxx=none libunwind=none build_type=Release targets=x86,nvptx,amdgpu %gcc@10.2.0`, with `target=x86_64` for every package so that the tools run on any node. Of Myriad's own software it uses only GCC 10.2 and cmake 4.1.2; Spack builds its own Python and Perl, because RHEL 7's are too old (Perl lacks `IPC::Cmd`) or depend on module variables (the Python module's venv).
    - **Build:** sources are fetched on the login node (38 archives, 438 MB) and built in a CPU-only batch job (36 cores, D node). It took 2 h 34 min in total, but only about 20 min of that is the LLVM compile. The rest is Spack's install of many small files into home over myriadfs, which is very slow from compute nodes; a later build should install to node-local disk and copy one archive.
    - **Result:** LLVM 23.1.0, Release, static, RTTI, no assertions, targets X86 NVPTX AMDGPU, 122 LLVM + 6 lld libraries: the same configuration as the bundle. `llvm-config --system-libs` is `-lrt -ldl -lpthread -lm`.
    - **Layout:** `$XDG_DATA_HOME/tracel/tracel-llvm-23.1.0-3/{bin,include,lib}` link to the Spack prefix, plus the sentinel file; the runner's module lines export `XDG_DATA_HOME`.
    - **Linking:** RHEL 7's GNU ld 2.27 resolves each archive once, left to right. The bundler emits LLVM's archives from cubecl-llvm's build script, before `pliron_llvm` on the link line, so it leaves the ORC JIT and native-target symbols that only `pliron_llvm` uses undefined. Linking with LLD (`-fuse-ld=lld`, the Spack-built `ld.lld`) resolves them; Rust's own default on x86-64 Linux is LLD as well, but that default is lost with a custom linker.
    - **Evidence:** the login-node fetch step and a full runner run passed: `remote-test.sh myriad-v100 --cores 8 -- cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda`, job 172549, 2026-10-18, exit 0. The report shows `ptx_sm70`, toolkit 12.2 ≤ driver CUDA 12.4, compiler `llvm`, and f32 (vector sizes 1, 4), f64 (1, 2) and u32 histogram checks all passing bitwise. The binary needs at most `GLIBC_2.16` and links only glibc, GCC 10's `libstdc++` and `libgcc_s`.
    - **Costs:** it relies on undocumented bundler internals (the sentinel file, the directory name tied to the bundler version, the use of `llvm-config`). Every CubeCL upgrade that changes the bundler version needs a new Spack build, about 25 min of compiling plus the install, and a check that tracel's LLVM carries no patches cubecl-llvm depends on (none needed for 23.1.0-3). An LLVM build is a CPU batch job, which the runner (GPU targets only) does not submit, so it is a documented maintenance step.
    - **Benefits:** native jobs with Myriad's modules (CUDA 12.2.2, MPI), no container indirection, and the same LLVM for the CPU runtime. For M10, a CUDA-aware Open MPI can be built with Spack next to it.
3. **Run the fetch step, the build and the command inside an Apptainer image** (Ubuntu 22.04, CUDA 12.2.2, clang, Open MPI). It would be built with Docker on the M3 Max, copied to Myriad and converted on the login node, and the runner would wrap the fetch and the job in `apptainer exec --nv`. This uses the bundle unchanged and needs no LLVM builds. Its costs:
    - every image change is an emulated `linux/amd64` build on Apple Silicon plus a multi-GB transfer;
    - Myriad's modules are unusable inside the image, so every tool must be in it;
    - there are two environments to debug;
    - MPI comes from the image, and a CUDA-aware MPI for M10 would be built inside it;
    - the toolkit is frozen in the image.

    Kept as the fallback if a later bundler version cannot be reproduced with Spack.
4. **Make another machine the main CUDA platform.** Locust is an experimental open node without a scheduler. Fixed decision 10 relies on Myriad's many scheduled V100 nodes. Not proposed.
5. **Patch CubeCL so that `cuda-cpp` builds without cubecl-llvm.** This breaks the `=0.11.0` pin (fixed decision 2) and still leaves the CPU runtime unbuildable on Myriad. Not proposed.

## Proposed decision

Option 2. On Myriad, CubeCL's LLVM comes from a Spack build of the bundle's LLVM version in the bundler's layout, linked with LLD; everything else stays native. The Spack environment, the overlay recipe, the build job and the LLD wrapper are committed under `scripts/hpc/spack/`, so the build can be reproduced. The runner gains no new option: the target's module lines in `targets.env` export `XDG_DATA_HOME` and put the LLD wrapper on `PATH`. A CubeCL upgrade that changes the bundler version includes rebuilding this LLVM. If that ever fails, option 3 is taken up in a new ADR.

## Consequences

- Text to change in docs/design.md:
    - "Development environment and test targets", the runner step 3: "It fetches crates on the login node, and the LLVM bundle when needed — on Myriad, whose RHEL 7 is too old for the prebuilt bundle, a Spack-built LLVM of the same version in the bundler's layout — because compute nodes have no internet access. …";
    - "Myriad rules that shape the runner": a new bullet "Myriad runs RHEL 7.9 (glibc 2.17), older than CubeCL's prebuilt LLVM bundle allows (glibc ≥ 2.34). The same LLVM version is built with Spack (`scripts/hpc/spack/`, a CPU batch job) and linked with LLD; it is rebuilt when the bundler version changes.";
    - "Risks and mitigations": the LLVM-bundle row adds "The prebuilt bundle needs glibc ≥ 2.34; on Myriad the same LLVM version is built with Spack and linked with LLD", and a new row "Spack-built LLVM relies on undocumented bundler internals (sentinel file, directory name, `llvm-config`) | Rebuild and re-run env-check on Myriad with every CubeCL upgrade; Apptainer image as the fallback".
- Public API changes: none. The runner's CLI is unchanged; `targets.example.env` documents the module-line exports.
- Acceptance tests affected: none change. env-check's report and the prologue record the LLVM in use (`llvm-config --version` and prefix).
- E0 plan:
    - T6 commits `scripts/hpc/spack/` (spack.yaml, the overlay recipe, the build job script, the LLD wrapper) and the Myriad record, and closes with its smoke run, which already passed with this setup (job 172549);
    - a Deviations-log row records that the LLVM build job was a hand-written `qsub` under CLAUDE.md's E0 exception;
    - T7 also checks `--features cpu` on Myriad, since the CPU runtime uses the same LLVM, and investigates the slow first launches seen in job 172549 (26–147 s per check).
- Later milestones affected:
    - M0's CUDA runs on Myriad work unchanged (transparent);
    - M3s's AVX-512 CPU benchmarks become possible;
    - a CubeCL upgrade milestone includes the Spack rebuild;
    - M10 can build a CUDA-aware Open MPI with the same Spack setup.

## Decision record

Accepted by the maintainer on 2026-10-18 (option 2, Spack-built LLVM; option 3, the Apptainer image, stays the fallback).
