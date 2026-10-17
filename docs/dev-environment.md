# Development environment

This file records the development and test targets of gridcl: how to reach them, how jobs run there, and the versions found on them. Milestone E0 fills in every field marked *(E0)* and keeps the file current afterwards. The rules for when each target is used are in `docs/design.md`, section "Development environment and test targets", and in `CLAUDE.md`.

Personal details (user ids, host names that are not public, project codes) never go in this file. They live in `scripts/hpc/targets.env`, which git ignores.

## Targets

| Target | Hardware | Access | Role |
| --- | --- | --- | --- |
| M3 Max (local) | Apple M3 Max, macOS, arm64 | local | Claude Code; CubeCL CPU runtime (f32, f64) and Metal (f32) on every change; MPI up to 4 ranks |
| Myriad V100 | E and F nodes: 36 cores (Xeon Gold 6140 or 6240), 192 GB RAM, 2 × V100 (compute capability 7.0) | SGE: `-l gpu=N` with `-ac allow=EF` | main CUDA development platform; CPU benchmarks on AVX-512 |
| Myriad A100 | L nodes: 36 cores, 192 GB RAM, 4 × A100 40 GB (compute capability 8.0) | SGE: `-l gpu=N` with `-ac allow=L` | second CUDA architecture and CUDA 13; when required |
| Locust | NVIDIA GH200 Grace Hopper: Grace aarch64 CPU, Hopper GPU (compute capability 9.0) | UCL ARC experimental system; open multi-user node without a scheduler (runner kind `direct`) *(E0: access method)* | aarch64 host and Hopper GPU; when required |
| Hosted CI | Linux x86-64 runners *(provider to be chosen)* | automatic | CPU runtime, Vulkan (software), MPICH and Open MPI, conformance |

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
- **Building against `libcuda`:** UCL notes that code linking against `libcuda` must be built on a GPU node. CubeCL loads the driver library at run time, so building on a login node is expected to work; E0 confirms this or moves the build into the job.

## Recorded by E0

| Item | M3 Max | Myriad V100 | Myriad A100 | Locust |
| --- | --- | --- | --- | --- |
| SSH alias and scheduler kind | local | `myriad`, `sge` | `myriad`, `sge` | *(E0)* |
| OS and architecture | macOS 27.0.1 (26A434), arm64 | *(E0)* | *(E0)* | Linux, aarch64 *(E0: distribution)* |
| Rust toolchain | 1.99.1 (58cf8f9c3 2026-10-13), from `rust-toolchain.toml` (ADR-0003) | *(E0)* | *(E0)* | *(E0)* |
| Module lines for jobs | – | *(E0)* | *(E0)* | *(E0)* |
| CUDA toolkit and `CUDA_PATH` | – | 12.x *(E0)* | *(E0)* | *(E0)* |
| Driver version and its CUDA version (`nvidia-smi`) | – | *(E0)* | *(E0)* | *(E0)* |
| Compute capability (env-check) | – | 7.0 *(E0: confirm)* | 8.0 *(E0: confirm)* | 9.0 *(E0: confirm)* |
| f64 and u64 support (env-check) | CPU: f64 and u64; Metal: u64, no f64 (f64 checks reported unsupported); u32 atomic add on both | *(E0)* | *(E0)* | *(E0)* |
| `max_page_size`, `max_bindings` | CPU: 2^36 B (64 GiB), 2^32 − 1; Metal: 2^32 − 4 B, 28 | *(E0)* | *(E0)* | *(E0)* |
| CPU runtime: `load_width`, vector registers, hardware threads | 128 bits, 32, 16; plane size 1; IO-optimized sizes f32 16, f64 8 (Metal: 4, 2; plane size 32) | *(E0)* | *(E0)* | *(E0)* |
| CPU model and SIMD extensions | Apple M3 Max, 12 P-cores and 4 E-cores; NEON (AdvSIMD with FP16, BF16, I8MM, DotProd), no SVE | Xeon Gold 6140/6240, AVX-512 *(E0: confirm)* | *(E0)* | Grace (Neoverse V2), NEON and SVE2 *(E0: confirm)* |
| MPI implementation and version | Open MPI 5.0.10 (Homebrew); 4-rank allreduce passes | *(E0)* | *(E0)* | *(E0)* |
| CUDA-aware MPI available | – | *(E0)* | *(E0)* | *(E0)* |
| Internet access on compute nodes | – | *(E0)* | *(E0)* | *(E0)* |
| Clean build time of env-check | 81 s on Rust 1.99.1 (`--release --features cpu,metal,mpi`) | *(E0)* | *(E0)* | *(E0)* |
| Queue wait and run time (median of 3) | – | *(E0)* | *(E0)* | *(E0)* |
| Gmsh, `vtk`, `meshio`, Claude Code versions | Gmsh 4.15.2-git (unpinned; M4 sets the pin); vtk 9.7.1 and meshio 5.3.5 in `.venv` (Python 3.14.7); Claude Code 2.1.295 | – | – | – |

Re-run env-check on a target after its maintenance window or a module change, and update this table.

### Notes on the M3 Max

- `scripts/check-local-tools.sh` checks and prints the local tool versions.
- env-check at 10^7 elements, release build, launch and read-back: CPU f32 add 38 ms at vector size 1 and 11 ms at 16, f64 add 34 ms at 1 and 15 ms at 8, u32 atomic histogram 170 ms; Metal f32 add 64 ms at 1 and 26 ms at 4, histogram 19 ms. Every check matches the host result bitwise.
- On Metal (wgpu MSL), `identity.fingerprint` is `spirv_0_0`: CubeCL 0.11 names every wgpu fingerprint `spirv_{vendor}_{device}`, whatever the shader language.
- Claude Code's Bash sandbox blocks Metal and `sysctl`: inside it, env-check finds no Metal device and records the CPU model and SIMD flags as unknown. Metal runs and the env-check host record need a command run outside the sandbox (`/sandbox` manages this).

## Change history

| Date | Change |
| --- | --- |
| 2026-10-17 | Created with the targets and the Myriad reference; E0 fills in the rest. |
