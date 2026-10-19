# Development environment

This file records the development and test targets of gridcl: how to reach them, how jobs run there, and the versions found on them. Milestone E0 fills in every field marked *(E0)* and keeps the file current afterwards. The rules for when each target is used are in `docs/design.md`, section "Development environment and test targets", and in `CLAUDE.md`.

Personal details (user ids, host names that are not public, project codes) never go in this file. They live in `scripts/hpc/targets.env`, which git ignores.

## Targets

| Target | Hardware | Access | Role |
| --- | --- | --- | --- |
| M3 Max (local) | Apple M3 Max, macOS, arm64 | local | Claude Code; CubeCL CPU runtime (f32, f64) and Metal (f32) on every change; MPI up to 4 ranks |
| Myriad V100 | E and F nodes: 36 cores (Xeon Gold 6140 or 6240), 192 GB RAM, 2 × V100 (compute capability 7.0) | SGE: `-l gpu=N` with `-ac allow=EF` | CUDA at milestone reviews, multi-GPU tests and as the fallback (ADR-0006); CPU benchmarks on AVX-512 |
| Myriad A100 | L nodes: 36 cores, 192 GB RAM, 4 × A100 40 GB (compute capability 8.0) | SGE: `-l gpu=N` with `-ac allow=L` | second CUDA architecture; when required (no CUDA 13: Myriad's driver supports CUDA 12.4) |
| Locust | NVIDIA GH200 Grace Hopper: Grace aarch64 CPU, Hopper GPU (compute capability 9.0) | UCL ARC experimental system; SSH to the node itself (alias `locust`, keys); open multi-user node without a scheduler (runner kind `direct`) | routine CUDA platform: after every task that changes kernels (ADR-0006); aarch64 host and Hopper GPU |
| Hosted CI | GitHub Actions, `ubuntu-24.04` x86-64 runners | automatic | CPU runtime, Vulkan (software), MPICH and Open MPI, conformance |

Myriad also has U and V nodes with 4 × A100 80 GB. The runner uses L nodes unless a test needs the larger memory.

## One-time setup (you, before E0)

1. **Accounts.**
    - **Myriad:** every UCL Research Computing account includes Myriad. Apply through the Research Computing sign-up process if you do not have one.
    - **Locust:** one of UCL ARC's experimental "Project Contender" systems. Request access from ARC (James Legg) with a short statement of use, for example "portability and performance testing of a GPU mesh library on Grace Hopper". ARC says these systems are not meant for large production runs.

2. **SSH keys.** Create a key on the M3 Max if you have none (`ssh-keygen -t ed25519`) and install it on Myriad (`ssh-copy-id myriad`) and on Locust. Off campus, the UCL SSH gateway accepts keys only.

3. **SSH configuration.** Add host aliases to `~/.ssh/config`. Connection sharing lets Claude's non-interactive commands reuse one authenticated connection:

    ```text
    Host ucl-gateway
        HostName ssh-gateway.ucl.ac.uk
        User <ucl-id>

    Host myriad
        HostName myriad.rc.ucl.ac.uk
        User <ucl-id>
        # Off campus without the UCL VPN:
        # ProxyJump ucl-gateway
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 8h
        ServerAliveInterval 60
        ConnectTimeout 15

    Host locust
        HostName <host name from ARC>
        User <ucl-id>
        # ProxyJump ucl-gateway, if ARC requires it
        ControlMaster auto
        ControlPath ~/.ssh/cm-%r@%h:%p
        ControlPersist 8h
        ServerAliveInterval 60
        ConnectTimeout 15
    ```

    At the start of a working day, open the shared connections once in a terminal, answering any password or second-factor prompt: `ssh myriad true` and `ssh locust true`. Claude cannot answer prompts.

4. **Local tools on the M3 Max.**

    | Tool | Install | Used for |
    | --- | --- | --- |
    | Xcode command-line tools | `xcode-select --install` | compilers, Metal |
    | rustup | rustup.rs; `rust-toolchain.toml` then installs Rust 1.99 | everything |
    | Claude Code ≥ 2.1.286 | as documented by Anthropic | the workflow |
    | Open MPI or MPICH (one of them) | Homebrew | local MPI runs; hosted CI covers the other |
    | rsync 3.x | Homebrew (the rsync that ships with macOS may lack the filter options the runner uses) | the remote-test runner |
    | Gmsh, one pinned version | Homebrew or gmsh.info | the M4 fixture corpus |
    | Python with `vtk` ≥ 9.2 and `meshio` | a virtual environment | M5 read-back tests |
    | HDF5 built with MPI | Homebrew (`hdf5-mpi`) | M8 VTKHDF output |

5. **Runner configuration.** After E0 has created `scripts/hpc/targets.example.env`, copy it to `scripts/hpc/targets.env` and fill in your aliases and paths.

## Myriad reference

From the UCL Research Computing documentation; E0 checks each item against the live system.

- **Scheduler:** SGE. Submit with `qsub`; monitor with `qstat` and `qstat -j <id>`; read errors with `qexplain <id>`; delete with `qdel <id>`; list finished jobs with `jobhist` (module `userscripts`); list node types with `nodetypes`.
- **GPU requests:** `#$ -l gpu=1`, `gpu=2` or `gpu=4`. Restrict to V100 with `#$ -ac allow=EF` and to A100 40 GB with `#$ -ac allow=L`. Keep these options inside the job script rather than on the command line, so `scriptfor <id>` shows them.
- **Other requests:** wall-clock time with `#$ -l h_rt=H:M:S`; memory per core with `#$ -l mem=…`.
- **Limits:** 72 hours of wall-clock time for 1 core and 48 hours for 2–36 cores. Home is also scratch, with a 1 TB quota (`gquota`).
- **Login nodes:** for managing files, compiling and short tests (under 15 minutes). Everything else runs as a job.
- **MPI with GPUs:** runs within a single node only.
- **Building against `libcuda`:** UCL notes that code linking against `libcuda` must be built on a GPU node. CubeCL does not link it: cudarc loads `libcuda` and NVRTC at run time, so builds need no driver library. cudarc does read the CUDA version at build time, from `nvcc --version`; without `nvcc` it silently assumes the newest CUDA it knows (13.x, feature `fallback-latest`). Every CUDA build must therefore run with the toolkit module loaded, as in the job.

### Checked by E0 (T6, 2026-10-17 and 18)

- **Login nodes:** Red Hat Enterprise Linux 7.9, kernel 3.10, glibc 2.17, bash 4.2, rsync 3.1.2, git 2.32 (module, loaded by default). Rust comes from rustup in the remote home (installed without changing `PATH`; the runner adds `~/.cargo/bin`), and the fetch step installs the toolchain of `rust-toolchain.toml`.
- **Compute nodes:** the V100 E nodes run the same RHEL 7.9 (glibc 2.17) and have no internet access (probe job: Xeon Gold 6240 with AVX-512, Tesla V100-PCIE-32GB, driver 550.127.05 = CUDA 12.4). Apptainer 1.2.4 is installed on login and compute nodes, in setuid mode; user namespaces are disabled, so images cannot be built from definition files on Myriad.
- **SSH:** key-only logins through the gateway work non-interactively (`BatchMode=yes`); each connection takes about 2 s without connection sharing.
- **Node types (`nodetypes`):** 16 E nodes (36 cores, 188 GB) and 6 L nodes (36 cores, 188 GB); no F nodes are listed. Also 1 U and 2 V nodes (48 cores, 251 GB), D, B and T nodes.
- **Modules:** CUDA up to 12.2.2 (`cuda/12.2.2/gnu-10.2.0`, which sets `CUDA_PATH`); no CUDA 13 module. NVIDIA HPC SDK up to 24.5. GCC up to 10.2.0 (`compilers/gnu/10.2.0` with `gcc-libs/10.2.0`); clang only 8.0.0. Open MPI 4.1.1 (gnu-4.9.2, the default) and 4.0.5 (gnu-10.2.0); Intel MPI up to 2021.11. Environment Modules 5.3.1.
- **Default environment:** the Intel 2018 compilers and Intel MPI are loaded by default, so the job's module lines start with `module purge`. `/usr/bin/cc` stays GCC 4.8.5 even with the GNU module loaded, and rustc links with `cc` by default; the module lines therefore set `CC=gcc`, `CXX=g++` and `CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc`. The shell function `module` is defined by the login shell but not exported, so the runner sources `$MODULESHOME/init/bash` in `bash -s` sessions and in jobs.
- **SGE requests:** `gpu` (per job), `memory` (`-l mem=…`, consumable per slot), `tmpfs` (per job, default 10G) and `h_rt`. Parallel environments include `smp` (one node) and `mpi`. The runner uses `-pe smp {cores}` for `--cores` and `--ranks` (GPU MPI runs stay within one node) and `-l mem=4G` per slot, so `--cores 36` asks for 144 GB of the 188 GB of a node.
- **Fetch step:** `cargo fetch`, then `cargo check` with the command's features on the login node, which runs the build scripts; jobs build offline in their own target directory. With `XDG_DATA_HOME` set as below, the LLVM bundle comes from the Spack build and nothing is downloaded.
- **Filesystem:** writing many small files into home (myriadfs) from a compute node is very slow: Spack's install of LLVM's headers took over two hours, against about 20 min for compiling LLVM on node-local disk.
- **Smoke run:** `remote-test.sh myriad-v100 --cores 8 -- cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda` passed on 2026-10-18 (job 172549): `ptx_sm70`, CUDA 12.2 toolkit ≤ driver 12.4, compiler path `llvm`, f32, f64 and histogram checks pass. Queue wait 450 s, build 1829 s on 8 cores from a clean target directory, command 396 s. The checks took 26–147 s each because the kernel caches were new and on myriadfs; see "Kernel caches".
- **Kernel caches (E0 T7, jobs 175329 and 175391):** on myriadfs, a new entry in the CUDA driver's JIT cache (`~/.nv/ComputeCache`) costs about 2 min per kernel, and creating CubeCL's cache database (`~/.cache/cubecl`, CubeCL's fallback when no `Cargo.toml` lies above the working directory) about 3 min. With both on the job's node-local `$TMPDIR` (a 10 GB tmpfs), env-check takes 1.9 s, the same as with the JIT cache disabled. The job script therefore sets `CUDA_CACHE_PATH` and `XDG_CACHE_HOME` below `$TMPDIR` and records them in the prologue (`kernel caches:`). Once the repository has a workspace `Cargo.toml` (M0), CubeCL 0.11 keeps its database in `<outermost Cargo.toml dir>/target/environment` instead, which is on myriadfs again (for env-check too, whose manifest lies inside the tree). No environment variable moves it; a `cubecl.toml` does, and the innermost one on the walk up from the working directory wins. Every job therefore writes `<tree>/cubecl.toml` with `[environment] path = "global"`, which puts the database in `$XDG_CACHE_HOME/cubecl`, below `$TMPDIR`; the prologue records it (`cubecl database=`). The file's content is the same for every job, so V100 and A100 jobs sharing a tree cannot point each other at another node's `$TMPDIR`. `/cubecl.toml` is git-ignored, so the sync neither copies nor deletes it (M0 T6).
- **MPI:** the Open MPI 4.0.5 module built with GCC 10 needs `numactl/2.0.12`, `binutils/2.36.1/gnu-10.2.0` and `ucx/1.9.0/gnu-10.2.0`. rsmpi's bindgen needs libclang; the only one on Myriad is LLVM 8 (`llvm/8.0.0/gnu-4.9.2`), which the module lines pass with `LIBCLANG_PATH` and `CLANG_PATH` without loading that module, whose `bin/` and `LIBRARY_PATH` would come before the Spack-built LLVM. bindgen 0.72 asks for libclang 9 or newer, but the bindings it generates with libclang 8 work for env-check. This Open MPI is not CUDA-aware (relevant for M10).
- **Acceptance (E0 T7, 2026-10-18):** `scripts/hpc/acceptance.sh myriad-v100` passed in two runs from the same sources: `gpus2` (job 175401), then `turnaround nvrtc fail max-wait` (jobs 175462–175496). The failing command exited 3 and its error was the summary's first failure. The `--max-wait 00:02:00` run was deleted while still queued, with exit status 124. `status.sh` was empty afterwards.
- **A100 acceptance (E0 T8, 2026-10-18 and 19):** `scripts/hpc/acceptance.sh myriad-a100` passed in two runs from the same env-check sources, after one warm-up job that built the A100 target directory (job 177336, 8 cores): `turnaround nvrtc fail max-wait` (jobs 177590–186144), then `gpus2` (job 186145), whose first attempt (job 178649) was deleted after its 4 h `--max-wait` in the queue. The L nodes run the same system, driver (550.127.05) and module lines as the E nodes, with Xeon Gold 6240 CPUs. `status.sh` was empty afterwards.

### CubeCL's LLVM on Myriad (ADR-0004)

cubecl-cuda (both `cuda` and `cuda-cpp`) and the CPU runtime need cubecl-llvm, whose build script downloads a prebuilt LLVM (tracel-llvm-bundler 23.1.0-3: LLVM 23.1.0) that needs glibc ≥ 2.34. On Myriad the same LLVM version is built with Spack and laid out where the bundler looks for it: `$XDG_DATA_HOME/tracel/tracel-llvm-23.1.0-3/` with the sentinel file `.tracel-llvm-installed`, which makes the bundler skip its download. The binaries link with LLD, because RHEL 7's GNU ld 2.27 resolves archives in one pass and misses LLVM symbols that only pliron-llvm uses.

1. On a Myriad login node, in the runner's tree: `bash scripts/hpc/spack/setup.sh`. It installs Spack 1.2.2 in `~/spack` if missing, renders the environment into `~/gridcl-llvm` (`GRIDCL_LLVM_ROOT`), concretizes and fetches every source.
2. `cd ~/gridcl-llvm && qsub build-llvm.sh`: a CPU-only job (36 cores, `h_rt` 6 h) that installs LLVM and lld, creates the bundler layout and `lld-bin/ld.lld`, and prints `llvm-config`. Progress is in `progress-<job id>.txt`.
3. In `targets.env`, LOGIN_MODULES and MODULES export `XDG_DATA_HOME=$HOME/gridcl-llvm/xdg PATH=$HOME/gridcl-llvm/lld-bin:$PATH` after loading GCC 10.

The Spack build matches the bundle: LLVM 23.1.0, Release, static, RTTI, no assertions, targets X86, NVPTX and AMDGPU, 122 LLVM and 6 lld libraries; `llvm-config --system-libs` is `-lrt -ldl -lpthread -lm`. Binaries need at most `GLIBC_2.16` and, at run time, GCC 10's `libstdc++` (`gcc-libs/10.2.0`). Spack builds its own Python and Perl (RHEL 7's are too old or depend on module variables) and uses Myriad's GCC 10.2 and cmake 4.1.2.

When a CubeCL upgrade changes the bundler version, add the new LLVM version to `scripts/hpc/spack/repo/spack_repo/gridcl/packages/llvm/package.py` and to the spec in `spack.yaml.in`, update `bundle=` in `build-llvm.sh`, rebuild, and re-run env-check on Myriad. If the bundle can no longer be reproduced, ADR-0004 names an Apptainer image as the fallback.

## Locust reference

Checked by E0 T9 on 2026-10-18.

- **Access:** SSH to the node itself (alias `locust`), with keys; non-interactive commands work through the shared connection. There is no scheduler: the runner's `direct` kind starts each job detached under `timeout` and tracks it with a pid file on the node. The node is shared and interactive.
- **Storage:** the home directory's quota is shared, so nothing goes there. Everything lives in the user's directory on the data disk, `/data/<user id>`, and the runner's trees are in `/data/<user id>/gridcl-hpc` (`LOCUST_BASE`). `/data` had 179 GB free (91% used).
- **Environment:** Locust has no site modules. The jobs reuse a Spack environment that the maintainer built there for another project (Spack 1.2.2: CUDA 12.6.3 and Open MPI 5.0.10, with the system GCC 11.4.1 and clang/libclang 17.0.6), that project's rustup and cargo homes and the downloaded `tracel-llvm` bundle 23.1.0-3 for aarch64; RHEL 9's glibc is new enough for the bundle, so no Spack-built LLVM is needed (unlike Myriad, ADR-0004). LOCUST_LOGIN_MODULES and LOCUST_MODULES in `targets.env` are the same lines: they point `HOME`, the XDG directories, `TMPDIR`, `CUDA_CACHE_PATH`, `CARGO_HOME` and `RUSTUP_HOME` at the data disk; unset `RUSTUP_TOOLCHAIN` (the other project pins Rust 1.99.0, and `rust-toolchain.toml` must decide); put cargo and the Spack view on `PATH` and `LD_LIBRARY_PATH`; export `CUDARC_CUDA_VERSION=12060`, `LIBCLANG_PATH=/usr/lib64` and `CARGO_BUILD_JOBS=32` (half of the shared node); keep Open MPI's shared-memory backing files in `TMPDIR`; and set the linker below. `CUDA_PATH` is the Spack view. If that Spack environment changes, re-run `scripts/hpc/acceptance.sh locust`.
- **Linker:** with the system's GNU ld 2.35.2, binaries that link CubeCL's LLVM fail with undefined `LLVMOrc*` symbols (archive order, as on Myriad). `CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS` links with the toolchain's `rust-lld` through rustup's `gcc-ld` wrapper of the `1.99` toolchain; change the toolchain name there when `rust-toolchain.toml` changes. `RUSTDOCFLAGS` repeats the same flags, because rustdoc ignores the target's `RUSTFLAGS` and doctests otherwise link with GNU ld and fail in the same way (M0 T2).
- **GPU:** one GH200, so `--gpus 1` is the maximum; the 2-rank MPI test runs both ranks on GPU 0. The driver's JIT cache and CubeCL's cache database go below `TMPDIR` on the data disk (`kernel caches:` in the prologue).
- **Acceptance:** `scripts/hpc/acceptance.sh locust` passed every step (jobs 20261018214145-3501138 to 20261018214423-3506568). The failing command exited 3 and its error was the summary's first failure; the `--max-wait 00:02:00` run was stopped on the node after 120 s, with exit status 124. The final `status.sh` check listed only the Myriad A100 job of T8, which was running at the same time; nothing was left on Locust.
- **Node load (ADR-0002):** both snapshots were `idle` only for the third turnaround run and the NVRTC run. Every other run was `busy` because of the 1-minute load average alone (2.1 to 5.1), left over from this runner's preceding build or job; other users' processes used at most 0.7 cores and no other process used the GPU. That rule counted the runner's own earlier work; since ADR-0005 the load average is recorded only, and the idle rule looks at other users' CPU use (sampled over 5 s) and the GPU.

## Recorded by E0

| Item | M3 Max | Myriad V100 | Myriad A100 | Locust |
| --- | --- | --- | --- | --- |
| SSH alias and scheduler kind | local | `myriad`, `sge` | `myriad`, `sge` | `locust`, `direct` |
| OS and architecture | macOS 27.0.1 (26A434), arm64 | Red Hat Enterprise Linux 7.9 (kernel 3.10, glibc 2.17), x86_64 | Red Hat Enterprise Linux 7 (kernel 3.10.0-1160.147.1.el7), x86_64 | Red Hat Enterprise Linux 9.3 (kernel 5.14.0-362.18.1.el9_3.aarch64+64k, 64 KiB pages), aarch64 |
| Rust toolchain | 1.99.1 (58cf8f9c3 2026-10-13), from `rust-toolchain.toml` (ADR-0003) | 1.99.1 (58cf8f9c3 2026-10-13), rustup in the remote home, from `rust-toolchain.toml` | 1.99.1 (58cf8f9c3 2026-10-13), the same rustup install and fetch step as the V100 | 1.99.1 (58cf8f9c3 2026-10-13), rustup on the data disk (`RUSTUP_HOME` shared with another project), from `rust-toolchain.toml` |
| Module lines for jobs | – | `module purge; module load gcc-libs/10.2.0 compilers/gnu/10.2.0; export CC=gcc CXX=g++ CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=gcc; export XDG_DATA_HOME=$HOME/gridcl-llvm/xdg PATH=$HOME/gridcl-llvm/lld-bin:$PATH; module load numactl/2.0.12 binutils/2.36.1/gnu-10.2.0 ucx/1.9.0/gnu-10.2.0 mpi/openmpi/4.0.5/gnu-10.2.0; export LIBCLANG_PATH=/shared/ucl/apps/llvm/8.0.0/gnu-4.9.2/lib CLANG_PATH=/shared/ucl/apps/llvm/8.0.0/gnu-4.9.2/bin/clang; module load cuda/12.2.2/gnu-10.2.0` (login node: the same without the CUDA module) | the same as the V100 (`cuda/12.2.2/gnu-10.2.0`, Myriad's newest CUDA module) | no site modules; environment lines in `targets.env` (see "Locust reference"), the same on the node for the fetch step and the job |
| CUDA toolkit and `CUDA_PATH` | – | 12.2.2 (`version.json`: 12.2.20230823), `CUDA_PATH=/shared/ucl/apps/cuda/12.2.2/gnu-10.2.0`; both compiler paths pass: LLVM (`cuda`) and NVRTC (`cuda-cpp`) | 12.2.2 (`version.json`: 12.2.20230823), `CUDA_PATH=/shared/ucl/apps/cuda/12.2.2/gnu-10.2.0`, the newest toolkit available that the driver supports (no CUDA 13: the driver supports CUDA 12.4); both compiler paths pass: LLVM (`cuda`) and NVRTC (`cuda-cpp`) | 12.6.3 from Spack (`version.json`: 12.6.20241114), `CUDA_PATH` = the Spack view; both compiler paths pass: LLVM (`cuda`) and NVRTC (`cuda-cpp`) |
| Driver version and its CUDA version (`nvidia-smi`) | – | 550.127.05, CUDA 12.4; toolkit 12.2 ≤ driver 12.4 | 550.127.05, CUDA 12.4; toolkit 12.2 ≤ driver 12.4 | 565.57.01, CUDA 12.7; toolkit 12.6 ≤ driver 12.7 |
| Compute capability (env-check) | – | 7.0 (`ptx_sm70`, same as `nvidia-smi`) | 8.0 (`ptx_sm80`, same as `nvidia-smi`); A100-PCIE-40GB | 9.0 (`ptx_sm90`, same as `nvidia-smi`) |
| f64 and u64 support (env-check) | CPU: f64 and u64; Metal: u64, no f64 (f64 checks reported unsupported); u32 atomic add on both | f64 and u64; u32 atomic add; f32 (sizes 1, 4), f64 (1, 2) and the histogram match the host bitwise on both GPUs and both compiler paths | f64 and u64; u32 atomic add; f32 (sizes 1, 4), f64 (1, 2) and the histogram match the host bitwise on both compiler paths | f64 and u64; u32 atomic add; f32 (sizes 1, 4), f64 (1, 2) and the histogram match the host bitwise on both compiler paths |
| `max_page_size`, `max_bindings` | CPU: 2^36 B (64 GiB), 2^32 − 1; Metal: 2^32 − 4 B, 28 | 8 518 139 904 B (7.9 GiB), 1024; plane size 32; IO-optimized sizes f32 4, f64 2 | 10 571 317 248 B (9.8 GiB), 1024; plane size 32; IO-optimized sizes f32 4, f64 2 | 25 501 368 320 B (23.75 GiB), 1024; plane size 32; IO-optimized sizes f32 4, f64 2 |
| CPU runtime: `load_width`, vector registers, hardware threads | 128 bits, 32, 16; plane size 1; IO-optimized sizes f32 16, f64 8 (Metal: 4, 2; plane size 32) | 512 bits, 32, 8 (the job's 8 slots); plane size 1; IO-optimized sizes f32 16, f64 8; all checks pass (job 175485) | 512 bits, 32, 8 (the job's 8 slots); plane size 1; IO-optimized sizes f32 16, f64 8; all checks pass (job 177595) | 128 bits, 32, 72; plane size 1; IO-optimized sizes f32 16, f64 8; all checks pass (job 20261018213942-3492085) |
| CPU model and SIMD extensions | Apple M3 Max, 12 P-cores and 4 E-cores; NEON (AdvSIMD with FP16, BF16, I8MM, DotProd), no SVE | Xeon Gold 6240 (every job so far, E nodes); AVX-512 (F, CD, BW, DQ, VL, VNNI), AVX2, FMA | Xeon Gold 6240 (every job so far, L nodes); AVX-512 (F, CD, BW, DQ, VL, VNNI), AVX2, FMA | Grace (Neoverse-V2), 72 cores, one thread each; NEON (AdvSIMD with FP16, RDM, DotProd, FHM, I8MM, BF16), SVE and SVE2 (with AES, PMULL, BitPerm, SHA3, SM4, I8MM, BF16) |
| MPI implementation and version | Open MPI 5.0.10 (Homebrew); 4-rank allreduce passes | Open MPI 4.0.5 (`mpi/openmpi/4.0.5/gnu-10.2.0`, UCX 1.9.0); with `--gpus 2 --ranks 2` the 2-rank allreduce passes and rank 0 on GPU 0 and rank 1 on GPU 1 both pass (job 175401) | Open MPI 4.0.5 (`mpi/openmpi/4.0.5/gnu-10.2.0`, UCX 1.9.0); with `--gpus 2 --ranks 2` the 2-rank allreduce passes and rank 0 on GPU 0 and rank 1 on GPU 1 both pass (job 186145) | Open MPI 5.0.10 (Spack); with `--gpus 1 --ranks 2` the 2-rank allreduce passes and ranks 0 and 1, both on GPU 0, pass (job 20261018214311-3504481) |
| CUDA-aware MPI available | – | no (`ompi_info`: `mpi_built_with_cuda_support` false) | no (`ompi_info`: `mpi_built_with_cuda_support` false) | no (`ompi_info`: `mpi_built_with_cuda_support` false) |
| Internet access on compute nodes | – | no (`curl` to crates.io times out) | no (`curl` to crates.io times out) | yes (`curl` to crates.io) |
| Clean build time of env-check | 81 s on Rust 1.99.1 (`--release --features cpu,metal,mpi`) | 1829 s on 8 cores (job 172549, `--release --features cuda`); after a feature change 178 s on 8 cores (`cuda-cpp,cpu`) and 952 s on 1 core (`cuda`) | 5428 s on 8 cores (job 177336, `--release --features cuda`); after a feature change 44 s on 8 cores (`cuda-cpp,cpu`) | 85 s with 32 build jobs (job 20261018213942-3492085, `--release --features cuda,cpu`); after a feature change 7 s (`cuda-cpp`) |
| Queue wait and run time (median of 3) | – | 1 GPU: queue 153 s, run 28 s (jobs 175462, 175480, 175482 on 2026-10-18; the first rebuilt CubeCL, 987 s); 2 GPUs: queue 7525 s (job 175401) | 1 GPU: queue 183 s, run 41 s (jobs 177590, 177591, 177593 on 2026-10-18; no rebuild); the first job of the target waited 6383 s (job 177336); 2 GPUs: queue 2762 s (job 186145; an earlier 2-GPU job, 178649, was deleted after 4 h in the queue) | no queue: start 1 s, run 3 s (jobs 20261018214145-3501138, …214159-3501789, …214212-3502445 on 2026-10-18; no rebuild, command 2 s each) |
| Gmsh, `vtk`, `meshio`, Claude Code versions | Gmsh 4.15.2-git (unpinned; M4 sets the pin); vtk 9.7.1 and meshio 5.3.5 in `.venv` (Python 3.14.7); Claude Code 2.1.295 | – | – | – |

Re-run env-check on a target after its maintenance window or a module change, and update this table.

### Notes on the M3 Max

- `scripts/check-local-tools.sh` checks and prints the local tool versions.
- env-check at 10^7 elements, release build, launch and read-back: CPU f32 add 38 ms at vector size 1 and 11 ms at 16, f64 add 34 ms at 1 and 15 ms at 8, u32 atomic histogram 170 ms; Metal f32 add 64 ms at 1 and 26 ms at 4, histogram 19 ms. Every check matches the host result bitwise.
- On Metal (wgpu MSL), `identity.fingerprint` is `spirv_0_0`: CubeCL 0.11 names every wgpu fingerprint `spirv_{vendor}_{device}`, whatever the shader language.
- Claude Code's Bash sandbox blocks Metal and `sysctl`: inside it, env-check finds no Metal device and records the CPU model and SIMD flags as unknown. Metal runs and the env-check host record need a command run outside the sandbox (`/sandbox` manages this).

## Hosted CI

GitHub Actions runs on the repository's GitHub copy (M0). Every job uses `ubuntu-24.04`, the toolchain of `rust-toolchain.toml`, and caches the cargo registry, `target/` and, where the CPU runtime builds, its LLVM bundle in `$XDG_DATA_HOME/tracel`. Actions are pinned by commit SHA.

| Workflow | Jobs | Trigger | System packages |
| --- | --- | --- | --- |
| `.github/workflows/ci.yml` | cpu, mpi-mpich, mpi-openmpi, vulkan | every push and pull request; manual | MPI jobs: `mpich libmpich-dev` or `openmpi-bin libopenmpi-dev`, and `libclang-dev`; vulkan: `mesa-vulkan-drivers vulkan-tools` |
| `.github/workflows/conformance.yml` | conformance | nightly; pushes touching `conformance/` or `gridcl-refcell/`; manual | `libopenblas-dev` |
| `.github/workflows/ptscotch.yml` | ptscotch (skeleton until M7) | nightly; pushes touching `gridcl-partition/` or `ptscotch-sys/`; manual | `libptscotch-dev` |

## Test commands

Fixed by M0; the verify skill, the test-runner and the hpc-test skill use them.

| Purpose | Command |
| --- | --- |
| workspace, CPU runtime | `cargo test --workspace` |
| Metal (M3 Max, unsandboxed) | `GRIDCL_TEST_DEVICE=metal cargo test --workspace --features gridcl-testing/metal` |
| Vulkan (hosted CI) | `GRIDCL_TEST_DEVICE=vulkan cargo test --workspace --features gridcl-testing/vulkan` |
| CUDA (the runner sets `GRIDCL_TEST_DEVICE=cuda`) | `scripts/hpc/remote-test.sh <target> -- cargo test --release --workspace --features gridcl-testing/cuda` |
| CPU runtime on Locust | `scripts/hpc/remote-test.sh locust -- env GRIDCL_TEST_DEVICE=cpu cargo test --release --workspace` |
| MPI | `scripts/mpi-test.sh 1 2 3 4` |
| MPI on CUDA (Myriad V100; both ranks use GPU 0 until per-rank selection) | `scripts/hpc/remote-test.sh myriad-v100 --gpus 2 --ranks 2 -- scripts/mpi-test.sh` |
| conformance | `cargo test --manifest-path conformance/Cargo.toml` |
| dependency check | `scripts/check-workspace-deps.sh` |
| lint (hosted cpu job) | `cargo fmt --all -- --check`; `cargo clippy --workspace --all-targets -- -D warnings`; `RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps` |

`scripts/hpc/remote-test.sh` runs everything after `--` as one shell command line (`bash -c`). Pass a compound command as one quoted argument, `-- 'A && B'`, not as `-- bash -c 'A && B'`.

`scripts/mpi-test.sh` takes its rank counts from its arguments, else `GRIDCL_RANKS`, else 1–4. `GRIDCL_MPIRUN_FLAGS` adds launcher flags and `GRIDCL_MPI_TEST_TIMEOUT` sets the per-run limit (600 s); each run's output is in `target/mpi-test/`. A `GRIDCL_TEST_DEVICE` other than `cpu` adds the gridcl-testing feature of the same name to the build, so inside `scripts/hpc` jobs the ranks open CUDA. The build uses the debug profile; on Myriad debug builds rely on the root `Cargo.toml`'s `[profile.dev.package.buildid]` override (buildid 1.0.5 otherwise aborts on an overflow), which separate cargo roots such as env-check and conformance do not inherit. Every rank opens `Device::cuda(0)` for now; choosing the node-local GPU per rank (`local_gpu_index` in the API walkthrough) is a follow-up.

## Change history

| Date | Change |
| --- | --- |
| 2026-10-17 | Created with the targets and the Myriad reference; E0 fills in the rest. |
| 2026-10-18 | Myriad checks of E0 T6; CubeCL's LLVM on Myriad from Spack (ADR-0004); Rust 1.99 (ADR-0003). |
| 2026-10-18 | Myriad V100 column (E0 T7): acceptance, Open MPI, kernel caches on node-local scratch. |
| 2026-10-18 | Locust column and reference (E0 T9): acceptance, environment on the data disk, node-load results. |
| 2026-10-19 | Myriad A100 column (E0 T8); target roles after ADR-0006. |
| 2026-10-19 | Record closed (E0 T10): no open E0 fields; the hosted CI provider is chosen in M0. |
| 2026-10-19 | M0: hosted CI on GitHub Actions, test commands, Locust doctest linker (`RUSTDOCFLAGS`), CubeCL's database under `$TMPDIR` via `cubecl.toml`. |
| 2026-10-19 | M0 review: MPI on CUDA row (Myriad V100); `mpi-test.sh` links the `GRIDCL_TEST_DEVICE` backend and needs no `jq`; debug profile and the buildid overflow override. |
