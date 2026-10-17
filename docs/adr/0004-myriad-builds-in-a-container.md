# ADR-0004: Build and run Myriad jobs in an Apptainer container (CubeCL's LLVM bundle needs glibc ≥ 2.34)

- **Status:** Proposed
- **Date:** 2026-10-17
- **Raised in:** E0 T6 (Myriad survey and smoke run)
- **Design sections affected:** "Development environment and test targets" (remote-test runner, steps 1, 3 and 4; Myriad rules); "CUDA toolkit and driver"; "Risks and mitigations" (the LLVM-bundle row and a new row); `docs/dev-environment.md` (Myriad reference, runner configuration)

## Context

The design makes Myriad V100 the main CUDA platform, and has the runner build and run on the cluster's own operating system:

> 3. **Fetch.** It fetches crates on the login node, and the CPU runtime's LLVM bundle when needed, because compute nodes may have no internet access. …
> 4. **Submit.** … The job loads the modules, records `nvidia-smi` and the toolkit version, builds, and runs the command with `GRIDCL_TEST_DEVICE=cuda`.

("Development environment and test targets"). "CUDA toolkit and driver" assumes the toolkit comes from a module (`CUDA_PATH` must point at it). The risks table has "the CPU runtime downloads an LLVM bundle from github.com at build time | Fetch crates and the bundle on the login node".

The E0 T6 survey found:

- Myriad's login nodes run **Red Hat Enterprise Linux 7.9 with glibc 2.17** (kernel 3.10). Compute nodes: see "Problem" (probe job).
- In cubecl-cuda 0.11.0, `cubecl-llvm` (features `std`, `nvptx`) is a **non-optional** dependency. Both env-check paths, `cuda` (LLVM → PTX) and `cuda-cpp` (NVRTC), build it. So does cubecl-cpu, the CPU runtime.
- cubecl-llvm takes LLVM from `tracel-llvm-bundler` 23.1.0-3. That is a prebuilt archive (`linux-x64.tar.xz`, 318 MB) from github.com/tracel-ai/tracel-llvm, unpacked to `$XDG_DATA_HOME/tracel/tracel-llvm-23.1.0-3`. The path is fixed, there is no switch for a system LLVM, and Myriad has no LLVM 23 (its newest LLVM module is 8.0).
- Apptainer 1.2.4 is installed on Myriad (`/usr/bin/apptainer`, setuid mode). User namespaces are disabled (`max_user_namespaces=0`, no subuid entry), so images cannot be built from a definition file on Myriad. Existing images can be converted (`apptainer build x.sif docker://…` or `docker-archive://…`).

## Problem

The runner's fetch step on the Myriad login node (`cargo check --manifest-path tools/env-check/Cargo.toml --features cuda`, 2026-10-17) fails in cubecl-llvm's build script:

```text
/home/<user>/.local/share/tracel/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.33' not found
/home/<user>/.local/share/tracel/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.34' not found
/home/<user>/.local/share/tracel/tracel-llvm-23.1.0-3/bin/llvm-config: /lib64/libc.so.6: version `GLIBC_2.32' not found
error: failed to run custom build command for `cubecl-llvm v0.11.0`
```

The problem goes beyond `llvm-config`: the bundle's static libraries reference `__libc_single_threaded` 242 times. That symbol exists only from glibc 2.32 (and libstdc++ from GCC 11), so linking against glibc 2.17 would fail as well. As packaged, no CubeCL CUDA program and no CPU-runtime program can be built on a glibc 2.17 system. The E0 acceptance item "Myriad V100: `scripts/hpc/remote-test.sh myriad-v100 -- cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda` runs … the report shows sm_70 …" cannot be met as designed. The same holds for Myriad A100, and for the design's "CPU benchmarks on AVX-512" on Myriad (M3s), which use the CPU runtime.

Compute nodes: a probe job through the runner (job 164878: `/etc/os-release`, glibc, `nvidia-smi` driver and CUDA version, `apptainer --version`, CPU flags) was queued on 2026-10-17; its result is added here before the decision.

## Options

1. **Keep the design as written (host builds with modules).** Impossible on RHEL 7.9 without changing CubeCL or its LLVM bundle; see option 3 and 5. Not chosen.
2. **Run the fetch step, the build and the command inside an Apptainer container on Myriad.** An image based on `nvidia/cuda:12.2.2-devel-ubuntu22.04` (Ubuntu 22.04, glibc 2.35; CUDA 12.2.2, the toolkit Myriad's module provides) adds the build tools gridcl needs: GCC, clang and libclang (rsmpi's bindgen), pkg-config, Open MPI, git and curl. Its Dockerfile is committed under `scripts/hpc/containers/`. The image is built on the M3 Max (`docker buildx build --platform linux/amd64`), saved, copied with rsync, and converted on the login node (`apptainer build <base>/images/<name>-<hash>.sif docker-archive://…`); no registry or credentials are needed. A new `targets.env` key `CONTAINER` names the SIF. When it is set, the runner wraps the login-node fetch and the job's build and command in `apptainer exec --nv <sif> bash -c …` (`--nv` binds the host's NVIDIA driver; home and the tree are mounted). The module lines then stay empty, and `CUDA_PATH` is `/usr/local/cuda` inside the image. Rust remains the rustup toolchain in the remote home: its binaries run unchanged in the newer image, and the cargo cache and the LLVM bundle stay in home. MPI runs use the image's Open MPI within one node, which is enough for the design's single-node GPU MPI runs. The design's interface (`remote-test.sh` options, results, summary, exit statuses) does not change. Locust needs no container if its OS has glibc ≥ 2.34 (T9 checks).
3. **Build LLVM 23.1 from source on Myriad** (GCC 10, glibc 2.17) and install it where the bundler expects its archive, keeping host builds. This depends on undocumented bundler internals (directory layout, sentinel file, the components and patches of tracel's LLVM build), costs hours of build time on every CubeCL upgrade, and may fail on tracel's patches. Not proposed.
4. **Make another machine the main CUDA platform.** Locust is an experimental open node without a scheduler. The design (fixed decision 10) relies on Myriad's V100 nodes being plentiful and scheduled. Not proposed while option 2 works.
5. **Patch CubeCL so that `cuda-cpp` builds without cubecl-llvm.** This breaks the `cubecl = "=0.11.0"` pin (fixed decision 2) and leaves the CPU runtime unbuildable on Myriad. Not proposed.

## Proposed decision

Option 2. Builds and jobs on Myriad run inside an Apptainer image with a modern glibc and the CUDA 12.2.2 toolkit. The image is built from a committed Dockerfile on the M3 Max, copied to Myriad and converted there. The runner gains an optional per-target `CONTAINER` setting that wraps the fetch step and the job, and keeps its interface. This keeps Myriad as the main CUDA platform without touching CubeCL or its LLVM bundle, and also enables the CPU runtime on Myriad for the AVX-512 benchmarks. If the probe or the first container run shows that `--nv` or the driver does not work on the GPU nodes, option 4 is revisited in a new ADR.

## Consequences

- Text to change in docs/design.md:
    - "Development environment and test targets", the runner steps: step 1 adds "and, on targets whose OS is too old for CubeCL's LLVM bundle, the Apptainer image"; step 3 reads "It fetches crates on the login node, and the LLVM bundle when needed, inside the target's container if it has one, …"; step 4 reads "The job runs inside the target's container if it has one (`apptainer exec --nv`), otherwise it loads the modules; it records …";
    - "Myriad rules that shape the runner": a new bullet "Myriad runs RHEL 7.9 (glibc 2.17), older than CubeCL's prebuilt LLVM bundle allows (glibc ≥ 2.34). Builds and jobs therefore run in an Apptainer image (Ubuntu 22.04, CUDA 12.2.2) built from `scripts/hpc/containers/` on the M3 Max and converted on the login node; Myriad cannot build images itself.";
    - "CUDA toolkit and driver": "`CUDA_PATH` must point at the toolkit (in the container image on Myriad, `/usr/local/cuda`)";
    - "Risks and mitigations": the LLVM-bundle row adds "The prebuilt bundle needs glibc ≥ 2.34; on older systems (Myriad) builds run in a container."
- Public API changes: none (the runner's CLI is unchanged; `targets.example.env` gains `CONTAINER`).
- Acceptance tests affected: none change their wording. run.sh gains cases for the container wrapping (fake `apptainer`), and `acceptance.sh` runs inside the image on Myriad. env-check records the container's OS and glibc alongside the host's driver.
- E0 plan: a new task before the T6 smoke run, "Myriad container image and runner support" (Dockerfile, an image script `scripts/hpc/image.sh <target>` that builds, copies and converts, the `CONTAINER` key in `lib/config.sh`, the wrapping in the fetch step and `jobs/common.sh`, run.sh cases). T6 then finishes its smoke run in the container, and T7 and T8 use it.
- Later milestones affected: M0's gridcl-testing CUDA runs and M2–M10 CUDA runs on Myriad go through the container (transparent to them). The image's Open MPI is not CUDA-aware, so the M10 GPU-aware exchange needs a CUDA-aware MPI in the image (an Open MPI built with CUDA support, or a later image); M10's plan revisits this. A CubeCL upgrade that changes the bundle's requirements may need a new image.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
