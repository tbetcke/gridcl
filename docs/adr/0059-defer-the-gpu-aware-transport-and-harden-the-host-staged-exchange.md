# ADR-0059: Defer the GPU-aware transport and harden the host-staged exchange

- **Status:** Proposed
- **Date:** 2026-11-16
- **Raised in:** M10 planning (`/plan-milestone M10`; the maintainer asked what GPU-aware MPI would gain and for measurements)
- **Design sections affected:**
  - "Device model and backends", "Supported backends" (the GPU-aware MPI column);
  - "Distributed mesh model", "Transports", Listing S2b and Listing S2c (the `Transport` comment);
  - "Testing and validation" (targets table, the CUDA-aware MPI row);
  - "Workspace layout" (the gridcl-device row, dependency rule 4, "Cargo features", "CI matrix": the gpu-aware row);
  - "Risks and mitigations" (the GPU-aware MPI row);
  - "Implementation roadmap", M10 Hardening (scope and acceptance bullet 1).
  - Process documents: IMPLEMENTATION.md (the M10 rows).

## Context

"Transports" defines two transports for `Scatterer`:

> **HostStaged** (default, every backend): a pack kernel fills the send buffer, `Client::read_one` copies it to the host, and the data moves by a neighbourhood all-to-all-v. `DeviceBuffer::write_host` (`Client::write`) then uploads the receive buffer and an unpack kernel scatters it.
>
> **GpuAware** (CUDA, feature `gpu-aware-cuda` …): device pointers are handed to a CUDA-aware MPI, using `Client::get_resource` as in Listing S2b.

M6 implemented `HostStaged` only. `Transport::GpuAware` exists, but nothing constructs it (ADR-0026). Roadmap M10 asks for "the GPU-aware transport (CUDA)", with acceptance bullet 1: "on a Myriad node with a CUDA-aware MPI, the GPU-aware halo exchange is bitwise identical to the host-staged one on 2 ranks (V100 node) and 4 ranks (A100 node)".

M10 planning found the following:

1. **No CUDA-aware MPI on any CUDA target.** docs/dev-environment.md records `mpi_built_with_cuda_support` false on Myriad V100, Myriad A100 and Locust. M10 would first have to find or build one in the Myriad home, through the NVIDIA HPC SDK module or a Spack build of UCX and Open MPI with CUDA.
2. **CubeCL can supply the pointers, with caveats.** CubeCL 0.11 has no MPI support of its own. `get_resource` returns the `CUdeviceptr` and size of a handle and holds the allocation in place (`cubecl-runtime-0.11.0/src/client.rs:467`, `cubecl-cuda-0.11.0/src/compute/storage/gpu.rs:163–180`). The default `CudaAllocator::Sync` uses `cuMemAlloc`, and the runtime uses the primary context. The caveats:
   - synchronisation is the caller's job;
   - the `Async` allocator's default pool is not IPC-exportable;
   - handles are sub-allocations of pooled pages, and CubeCL calls the driver through cudarc, so UCX's free hooks may not see a page that is freed and reused. Its registration cache could then go stale.
3. **None of the cases where GPU-aware MPI pays off applies to gridcl's targets.** Those cases are GPUDirect RDMA across nodes, NVLink between the GPUs of a node, and large halos exchanged very often. On the targets:
   - GPU MPI jobs on Myriad run within one node;
   - the A100s are PCIe cards (`A100-PCIE-40GB`);
   - Locust has one GPU, and Kathleen has none.

## Problem

The maintainer asked what the GPU-aware transport would gain over host MPI. A throwaway probe, run on 2026-11-16 at main ae8d644 (an untracked crate, `halo-probe`, not committed), measured the host-staged exchange and its parts.

- **Set-up.** Each rank owns 10^6 nodes with block 3 and ghosts h nodes of another rank. Each value is the median of 50 runs after 5 warm-ups, in µs.
- **Parts.** Each part is timed alone, followed by a device sync: pack, `to_host` (device to host, D2H), the host `MPI_Neighbor_alltoallv` of the same bytes, `write_host` (host to device, H2D) and unpack.
- **Whole exchange.** `forward` and `reverse` (Add) are timed through `Scatterer` on two ranks.
- **Conditions.** The remote nodes were idle before and after every job.

Parts on one rank, uncontended GPU, f64:

| Device (job) | h (bytes) | pack | D2H | H2D | unpack | bare sync |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GH200, Locust (20261116215841-1041951) | 10^3 (24 KB) | 16 | 14 | 16 | 16 | 9 |
| | 10^4 (240 KB) | 16 | 26 | 32 | 16 | 9 |
| | 10^5 (2.4 MB) | 17 | 317 | **3,173** | 17 | 9 |
| RTX A4000, PCIe (20261116215719-69663) | 10^3 | 26 | 23 | 33 | 27 | 14 |
| | 10^4 | 27 | 43 | 108 | 27 | 15 |
| | 10^5 | 35 | 374 | **6,502** | 34 | 14 |

Whole exchange on two ranks:

| Device | h | forward | reverse | host MPI | D2H | H2D |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Locust CUDA f64, two ranks on one GPU (20261116215551-1032016) | 10^3 / 10^4 / 10^5 | 470 / 527 / 1,232 | 504 / 729 / 2,015 | 4 / 18 / 211 | 102 / 114 / 320 | 149 / 40 / 1,564 |
| Mac mini, CPU runtime, f64 | 10^3 / 10^4 / 10^5 | 91 / 252 / 923 | **42,261 / 44,594 / 35,621** | 2 / 9 / 90 | 5 / 8 / 33 | 6 / 25 / 309 |
| Mac mini, Metal, f32 | 10^3 / 10^4 / 10^5 | 734 / 731 / 966 | 725 / 746 / 1,397 | 1 / 5 / 46 | 216 / 321 / 318 | 353 / 342 / 396 |

What the measurements show:

1. **Small and medium halos.** Up to 10^4 nodes (240 KB), each copy costs 14–45 µs on a quiet GPU, about one kernel launch and sync. A GPU-aware transport removes two copies, but adds the latency of a device-memory transfer in MPI, so it saves at most a few tens of µs per exchange.
2. **Large halos.** At 10^5 nodes (2.4 MB) the cost is the H2D copy: 0.4–0.8 GB/s, against 6–8 GB/s for D2H on the same hardware. The upload path causes this, not the bus:
   - `write_host` builds a fresh padded `Vec` on every call (`gridcl-device/src/buffer.rs:264`);
   - `Client::write` then copies from that pageable memory (`cubecl-cuda-0.11.0/src/compute/driver.rs:146`);
   - reads, by contrast, go through CubeCL's pinned staging memory, and `Client::staging` (`cubecl-runtime-0.11.0/src/client.rs:844`) offers the same for writes.

   With H2D at D2H speed, the large-halo exchange drops from about 3.5–7 ms to about 0.7–0.9 ms on these GPUs, on every backend, without MPI changes. A CUDA-aware MPI without RDMA moves device data at about PCIe speed too, so beyond that it could gain at most about 2× on large halos.
3. **Fixed per-step latency elsewhere.** On Metal every step costs about 300 µs whatever its size. With two processes on one GPU, a bare sync costs 140 µs. Fewer steps and syncs per exchange help there; skipping copies does not.
4. **A defect unrelated to transports.** On the CPU runtime, `reverse` costs 35–45 ms at every halo size, against 0.1–0.9 ms for `forward` and about 1 ms on GPUs.

Against these modest gains, the GPU-aware path brings:
- a CUDA-aware MPI to find or build and maintain on Myriad;
- a second MPI in the runner's jobs;
- `ServerError`-level coupling to CubeCL's CUDA server type, already listed in "Risks and mitigations";
- the registration-cache risk of finding 2;
- tests that run only on Myriad's multi-GPU queues, which waited 0.8–2.1 h for two GPUs in E0.

## Options

1. **Keep the design as written.** M10 builds or finds a CUDA-aware MPI on Myriad and adds `device_ptr` (Listing S2b), transport selection, a test hook and the pointer-based exchange. It runs the 2-rank V100 and 4-rank A100 tests, then the gpu-aware review run at every later review. This costs about four tasks and Myriad queue time, for a gain the measurements put at tens of µs per exchange on small halos and at most about 2× on large halos once host staging is fixed. The transport would then still have to be kept working across CubeCL upgrades.
2. **Defer `GpuAware`, and harden the host-staged exchange in M10 (proposed).** `Transport::GpuAware` stays in the enum, reserved and never chosen. The feature `gpu-aware-cuda` stays declared but enables nothing. Listing S2b is kept as the design of the deferred path. M10 instead makes the host-staged exchange fast on every backend:
   - H2D through reused pinned staging buffers (`Client::staging`), so it moves at the speed D2H already does;
   - per-scatterer send, receive and staging buffers reused across calls, not allocated in every call;
   - `reverse` uploads only the received values, and packs the targets' own values on the device, instead of a host `[zeros | received]` buffer;
   - the CPU runtime's fixed cost in `reverse` found and removed;
   - fewer syncs per exchange where the order of steps allows it.

   gridcl-bench gains a `halo_exchange` benchmark (the probe's measurements, with records) that serves as the acceptance evidence. `local_gpu_index` stays in M10 (Listing B8; ADR-0026 item 17).
3. **Do both.** As option 2, plus option 1's transport. This has the highest cost, and option 2's fixes remove most of what option 1 would gain on the current targets.
4. **Remove `GpuAware` and `gpu-aware-cuda`.** Same work as option 2, but it changes a public enum and a library feature, and it discards a design that would pay off on RDMA or NVLink systems. Not proposed: keeping the variant reserved costs nothing.

## Proposed decision

Option 2. On gridcl's targets the host-staged exchange is limited by its own upload path and by per-step latency, not by the fact that it passes through the host. Fixing those gains more than a GPU-aware transport would, on every backend instead of CUDA alone, and without a second MPI, new CubeCL coupling or Myriad-only tests. The GPU-aware transport stays designed and reserved. A later milestone can take it up when a target has GPUDirect RDMA or NVLink, with measurements of the hardened host-staged exchange as its baseline.

## Consequences

- **Text to change in docs/design.md:**
  - **"Supported backends"**, the GPU-aware MPI column: CUDA "deferred (`gpu-aware-cuda` reserved; ADR-0059)"; HIP unchanged ("planned").
  - **"Transports"**:
    - HostStaged: the received data is uploaded through pinned staging buffers that each scatterer reuses, and `reverse` uploads only the received values (ADR-0059).
    - GpuAware: "deferred (ADR-0059); the design below is kept for a later milestone; no scatterer chooses it".
    - Listing S2b's caption gains "(deferred, ADR-0059)".
  - **Listing S2c**, the comment above `Transport`: "`HostStaged` is the only transport; `GpuAware` is reserved (ADR-0026, ADR-0059)".
  - **"Testing and validation"**, targets table: the row "Myriad node with a CUDA-aware MPI | GPU-aware halo exchange | from M10" is removed. The Myriad V100 row keeps "multi-GPU tests" and drops "GPU-aware".
  - **"Development environment and test targets"**: "for multi-GPU and GPU-aware MPI tests" becomes "for multi-GPU tests"; the bullet on CUDA-specific changes drops "or the GPU-aware transport".
  - **"Workspace layout"**:
    - the gridcl-device row: "GPU-aware pointer access" becomes "(GPU-aware pointer access deferred, ADR-0059)", and its dependency "cubecl-runtime (feature `gpu-aware-cuda`)" is removed;
    - dependency rule 4: "The one exception is `gpu-aware-cuda` …" becomes "`gpu-aware-cuda` is reserved and enables nothing (ADR-0059)";
    - "Cargo features": `gpu-aware-cuda` is listed as reserved.
  - **"CI matrix"**: the gpu-aware row is removed.
  - **"Risks and mitigations"**, the GPU-aware MPI row: the mitigation becomes "Deferred (ADR-0059); host-staged exchange hardened in M10".
  - **Roadmap M10:**
    - Scope: "the GPU-aware transport (CUDA) and `local_gpu_index` (ADR-0026)" becomes "a hardened host-staged halo exchange and `local_gpu_index` (ADR-0026, ADR-0059)".
    - Acceptance bullet 1 becomes: "the host-staged halo exchange (`forward` and `reverse`, halos of 10^3, 10^4 and 10^5 nodes with block 3) is recorded on the local Mac (CPU runtime, Metal) and Locust (CUDA, 1 and 2 ranks); for 10^5 nodes, its host-to-device copy takes at most 1.5 times its device-to-host copy and `reverse` at most twice `forward` on each, and results stay bitwise identical to M9h's (ADR-0059)".
- **CLAUDE.md:** line 11's "(exception: gpu-aware-cuda in gridcl-device)" becomes "(`gpu-aware-cuda` is reserved; ADR-0059)". The FFI rule's "gridcl-device (GPU-aware; …)" keeps its text, since the home stays valid for a later implementation. The copy list's "the host-staged halo exchange" is unchanged: staging through pinned buffers is the same copy.
- **IMPLEMENTATION.md:** the M10 rows ("a Myriad node with a CUDA-aware MPI"; the M10 note) are removed or reworded.
- **Code:** in gridcl-device's Cargo.toml, `gpu-aware-cuda` becomes an empty feature, dropping `cubecl/cuda` and `dep:cubecl-runtime`; gridcl keeps forwarding it. The `Transport` doc comments say "reserved (ADR-0059)".
- **Public API changes:** none. `Transport` keeps both variants and `Scatterer::transport` always returns `HostStaged`. `gpu-aware-cuda` still exists, but no longer pulls in cubecl's `cuda` runtime.
- **Acceptance tests affected:**
  - M10 bullet 1's planned `gridcl-mpi-tests::gpu_aware_halo` is not written;
  - new in its place: `gridcl-bench/tests/perf_record.rs::halo_exchange_recorded` (coverage and both bounds, from `docs/perf/data/m10-halo-*.json` written by a new `halo_exchange` binary);
  - the existing `scatter` (MPI, 1–4 ranks), `scatter_threads` and index-map tests guard the bitwise results of the hardened exchange;
  - no other milestone's acceptance changes.
- **Later milestones affected:**
  - M10: tasks T4, T6, T7 and T8 of the draft plan, and the Myriad gpu-aware review run, are replaced by one task, "host-staged exchange hardening", with a `halo_exchange` benchmark task;
  - M10 keeps `local_gpu_index`;
  - the GPU-aware transport has no milestone. A future one takes it up with ADR-0026's API and Listing S2b, measured against M10's records. Issue #25 is updated when this ADR is accepted.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
