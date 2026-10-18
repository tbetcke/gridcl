# ADR-0001: CUDA kernels compile with CubeCL's default LLVM backend, not NVRTC

- **Status:** Proposed
- **Date:** 2026-10-18
- **Raised in:** E0 plan, open question 1; measured in E0 T7 (Myriad V100 acceptance)
- **Design sections affected:** "CUDA toolkit and driver"; "Development environment and test targets" (remote-test runner, Myriad rules); "Risks and mitigations"; `.claude/skills/hpc-test/SKILL.md` (troubleshooting table)

## Context

"CUDA toolkit and driver" says:

> CubeCL compiles each kernel to PTX with NVRTC from the installed CUDA toolkit, and the driver JIT-compiles that PTX when it loads the module.

It derives two rules from this, "No CUDA 13 on V100" and "Toolkit no newer than the driver" (or module loading fails with `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`), and it says that `CUDA_PATH` must point at the toolkit "because CubeCL passes its include directory to NVRTC".

In cubecl-cuda 0.11.0 the default backend is different:

- **`cuda` (default):** CubeCL lowers each kernel through its own LLVM (cubecl-llvm, the NVPTX backend of LLVM 23.1) to PTX, and the driver JIT-compiles that PTX. No NVRTC is involved.
- **`cuda-cpp`:** the kernel is generated as CUDA C++ and compiled with NVRTC from the toolkit, as the design describes.

cubecl-llvm is a non-optional dependency of cubecl-cuda, so both features need the LLVM bundle (ADR-0004).

On the LLVM path, CubeCL chooses the PTX ISA version from the driver: `PtxVersion::for_driver(cuDriverGetVersion())` picks the newest version that driver loads, at most PTX 9.3 (cubecl-llvm, `nvptx/ptx_version.rs`). Its doc comment notes: "NVRTC writes its toolkit's version instead, which an older driver can refuse." With the LLVM backend, cubecl-cuda also restricts the device features it advertises to those the NVPTX target can compile (`restrict_features(…, GpuTarget::Nvptx)` in `runtime.rs`, mainly matrix-multiply-accumulate (MMA) instructions).

The toolkit is still needed at build time on both paths: cudarc reads the CUDA version from `nvcc --version`, and without `nvcc` it assumes the newest CUDA it knows (docs/dev-environment.md, Myriad reference).

## Problem

The design describes a compiler path that is not CubeCL's default, so its rules are wrong for the default path:

- The "toolkit no newer than the driver" rule does not apply to `cuda`, because the PTX version follows the driver, not the toolkit.
- `CUDA_PATH` is needed at run time only on `cuda-cpp`.
- The hpc-test troubleshooting rows (`CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, "A V100 compile error mentioning `sm_70`") describe NVRTC failures.

The design does not say which feature gridcl-testing, the examples and the benchmarks enable for CUDA. M0 needs that decision.

Measured on the Myriad V100 (E0 T7, 2026-10-18, Tesla V100-PCIE-32GB, driver 550.127.05 = CUDA 12.4, toolkit 12.2.2, env-check at 10^7 elements, release build, timings include launch and read-back):

| Path | Job | f32 ×1 | f32 ×4 | f64 ×1 | f64 ×2 | u32 histogram | Result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cuda` (LLVM) | 175462 | 86 ms | 40 ms | 145 ms | 69 ms | 28 ms | all pass bitwise; `ptx_sm70` |
| `cuda-cpp` (NVRTC) | 175485 | 743 ms | 111 ms | 216 ms | 135 ms | 97 ms | all pass bitwise; `ptx_sm70` |

The difference is the per-kernel NVRTC compile, about 0.1 s per kernel and about 0.65 s for the first. Both paths also passed on both GPUs of an E node in the 2-rank MPI test, which used `cuda` (job 175401). The A100 (T8) and Locust (T9) runs add the same comparison for sm_80 and sm_90.

## Options

1. **Keep the design as written: use `cuda-cpp` (NVRTC) for all CUDA work.** The toolkit rules and `CUDA_PATH` stay as the design says. Every kernel pays the NVRTC compile at its first launch in each process: test suites with hundreds of kernels would add minutes per run. `cuda-cpp` is not CubeCL's default, so it gets less upstream testing. The LLVM bundle is needed anyway.
2. **Use CubeCL's default `cuda` (LLVM) path, and correct the design text.** The PTX version follows the driver, so a toolkit newer than the driver no longer breaks module loading; the toolkit matters only at build time (cudarc's version detection). The first launch of each kernel is faster than with NVRTC. The device features advertised are those NVPTX can compile; the design uses no tensor-core MMA, so nothing it needs is lost. env-check keeps building `cuda-cpp` on every target, so a regression in either path is visible when a target is re-checked.
3. **Test both paths in every CUDA run.** This doubles the remote CUDA jobs and their queue waits (one hour or more for 2-GPU jobs on Myriad) for little gain, since gridcl code is identical on both paths. Not proposed.

## Proposed decision

Option 2. CUDA work uses CubeCL's default `cuda` feature (LLVM → PTX, with the PTX version chosen for the node's driver). `cuda-cpp` stays a diagnostic path that env-check builds and runs on each remote target. The design's "CUDA toolkit and driver" section is corrected to describe the LLVM path. It keeps "No CUDA 13 on V100", because cudarc reads the CUDA version from the toolkit at build time and the V100 nodes' driver is CUDA 12.4. The toolkit-vs-driver rule is kept only for `cuda-cpp`.

## Consequences

- Text to change in docs/design.md:
    - "CUDA toolkit and driver", first paragraph: "CubeCL 0.11 compiles each kernel to PTX with its own LLVM (NVPTX backend; feature `cuda`, the default), choosing the PTX ISA version from the node's driver, and the driver JIT-compiles that PTX when it loads the module. The alternative backend `cuda-cpp` compiles CUDA C++ with NVRTC from the toolkit; gridcl uses it only for diagnosis (env-check)."
    - "CUDA toolkit and driver", the rules: "No CUDA 13 on V100" stays, with the reason that cudarc takes the CUDA version from the toolkit at build time and the V100 driver is CUDA 12.4. "Toolkit no newer than the driver" applies only to `cuda-cpp`.
    - "CUDA toolkit and driver", `CUDA_PATH`: "`CUDA_PATH` and the toolkit module are needed at build time, because cudarc reads the CUDA version from `nvcc`; at run time only `cuda-cpp` (NVRTC) needs them."
    - "Development environment and test targets": no change to the runner. The job keeps loading the toolkit module.
    - "Risks and mitigations": new row "CubeCL's LLVM NVPTX backend advertises fewer device features than NVRTC (MMA) | gridcl uses no tensor-core MMA; env-check runs both backends on every remote target".
- `.claude/skills/hpc-test/SKILL.md`: the `CUDA_ERROR_UNSUPPORTED_PTX_VERSION` row applies to `cuda-cpp` only. The `sm_70` row becomes "A V100 build or load failure after loading a CUDA 13 toolkit: use CUDA 12.x".
- Public API changes: none. The CUDA feature of gridcl-testing (M0) forwards to cubecl `cuda`.
- Acceptance tests affected: none. E0's CUDA acceptance already uses `--features cuda`; T7–T9 also run `cuda-cpp`.
- Later milestones affected: M0 (gridcl-testing's `cuda` feature and the CI/hpc-test commands); M10 (the GPU-aware transport is unaffected; the compiler path does not change how device buffers are shared).

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
