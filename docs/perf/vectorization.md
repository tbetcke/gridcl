# Vectorization on gridcl's devices (M3s)

This report records how M3s chose `gridcl-device/vector-policy.toml` (docs/design.md,
"Vectorization and CPU execution", "Benchmarks"). The tables under "Results" are written by
`cargo run -p gridcl-bench --bin policy_report` from the JSON under `docs/perf/data/`, between
its marker comments; the rest is written by hand. The CpuSimd512 sweep and the Kathleen
throughput run wait for Kathleen, which was drained for maintenance during M3s (ADR-0016;
docs/plans/M3s.md, T16 and T17). Until then CpuSimd512 and CpuSimd256 keep vector size 1, tile
16, 16 KiB blocks and all hardware threads, and their rows say "no data".

## Protocol

- **Binary:** `vector_sweep` (gridcl-bench), release build, CubeCL's persistent kernel cache
  off. One timed region per family, the smallest call whose launches all take their shape from
  the family's plan (docs/plans/M3s.md, question 13): the primitives for Fill, Copy, Reduce
  and Gather; `inclusive_scan` then `exclusive_scan` for ScanTile; `radix_sort` for SortDigits;
  the hidden `gridcl_topology::bench` hooks for TopologyKeys and PermutationInfo, which include
  the cell-fastest transpose at sizes above 1; `DeviceGrid::evector` on a fresh clone for
  EVector; X, J, det J and K on P2 tetrahedra and P2 triangles in 3D for GeomFactors.
- **Configurations:** vector sizes 1, 2, 4, 8 and 16 on every device (a device that cannot run
  a size records the size it ran, "(ran m)"); QuadField tiles 1, 4 and 16 for GeomFactors; on
  CPUs, blocks of 4, 16 and 64 KiB, and on the M3 Max both CPU unit settings (all 16 threads,
  or the 12 performance cores; ADR-0017). Sizes 10^5, 10^6 and 10^7 items (cells for the mesh
  families). `KernelKind::Scalar` is measured on CPUs at vector size 1 only, for its blocks and
  units: histogram (u32) and segmented reduce (f64).
- **Timing (ADR-0018):** a warm-up call checked against a host reference in f64; one untimed
  calibration call that fixes k = ⌈20 ms / its time⌉, at most 10,000; ten timed runs of k calls,
  each call ending with `client.sync()`; the median of the ten per-call times. The compile time
  is the warm-up minus that median; the bandwidth ratio is relative to a u32 copy of the same
  length on the same device.
- **Decision rule:** "Benchmarks" in docs/design.md, as read in ADR-0015, ADR-0017, ADR-0018 and
  ADR-0019 (a summary heads the generated tables).

## Conditions

All sweep data were measured at commit 3ab0f31 (M3s T22).

| Device | Class | Run | Records | Conditions |
| --- | --- | --- | ---: | --- |
| M3 Max CPU runtime (Mac15,9; 12 performance and 4 efficiency cores) | CpuSimd128 | 2026-10-25 18:05–19:01, one family per part; Reduce repeated 19:12–19:14 | 1,800 | Mains power (100%, charged) and other applications closed, confirmed by the maintainer; each part started below a 1-minute load average of 3. Microsoft Defender (cannot be closed) used 0.3–0.7 of a core, with one burst of 1.7 cores before Fill; Outlook and Docker still ran during the first Reduce part, which was therefore repeated with both closed. |
| M3 Max Metal | Gpu | 2026-10-25 19:06–19:08; Fill repeated at 19:17 | 207 | As above. Sizes 8 and 16 run at 4. |
| Locust Grace CPU (Neoverse-V2, 72 threads; no performance-core count) | CpuSimd128 | job 20261025170542-3734076, 825 s | 900 | `node load: idle` before and after. |
| Locust H100 (GH200) | Gpu | job 20261025180049-1677957, 196 s | 294 | `node load: idle` before and after; no clamped sizes. |
| Kathleen node (2 × Xeon Gold 6248, AVX-512) | CpuSimd512 | – | – | Not yet measured: every node drained for the `rdss-upgrade` maintenance, and Myriad's D nodes (the fallback) started no job from 2026-10-24 on. |

Batching (ADR-0018) cut the noise of the sub-millisecond calls: the spread of the 3rd to 8th
sorted run over the median fell, for example, on the M3 Max CPU from 0.24 / 0.08 to
0.03 / 0.01 (Fill, 10^6 / 10^7) and from 1.03 / 0.52 to 0.38 / 0.40 (Reduce), on Grace from
0.33 / 0.30 to 0.15 / 0.09 (Gather), and on the H100 to below 0.01. Six entries (nine table
rows) still have no admitted configuration within 3% of the fastest at both sizes: Fill
CpuSimd128 4-byte, Reduce Gpu 4- and 8-byte, GeomFactors CpuSimd128 f32 and f64, and Scalar
CpuSimd128 f64. The rule then takes the closest (3.2–6.7% slower), and the tables mark them
"closest".

## Observations

- **Streaming kernels on CPUs gain most at 10^7.** Fill and Copy on Grace run 4.3–6.1× faster
  at sizes 4 to 16 than at 1 for 10^7 items, but only 0.9–1.2× at 10^6, where the data fit in
  cache and the launch dominates; on the M3 Max they gain up to 1.5× (Copy at 10^7: 0.95–1.08×). The 4-byte
  entries take 16 (Fill) and 8 (Copy); f64 Fill stays at 1, because Grace does not gain at 10^6.
- **Barrier-heavy kernels gain on GPUs.** Reduce and ScanTile run 1.3–2.4× faster at sizes 4 to
  16 on both the H100 and Metal; SortDigits gains 1.8–2.3× on Metal but only 1.1–1.3× on the
  H100 (0.86× at size 16 and 10^7). The Gpu class takes 16 (Reduce 4-byte), 4 (Reduce f64), 8
  (ScanTile) and 4 (SortDigits). On CPUs, SortDigits is flat (the scatter phase, scalar by
  design, takes about 90% of the sort after T21), and Reduce and ScanTile gain 1.1–1.9× on the
  M3 Max and at most 1.2× on Grace; the class means take the Reduce 4-byte and ScanTile entries
  to 16.
- **Gather is latency-bound.** At 10^7 no size makes Gather more than 10% faster on any device
  (and sizes 8 and 16 slow it on the H100, down to 0.54×); the gains at 10^6 (up to 1.9× on
  Grace for u32) are not repeated at 10^7, so it stays at 1.
- **The cell-fastest copy does not pay for itself yet.** At 10^7, TopologyKeys and
  PermutationInfo are slower at every size above 1 on every device (0.45–0.88×), and at 10^6
  on every device except the M3 Max CPU for PermutationInfo (up to 1.16×), because the
  transpose (scalar, `KernelKind::Scalar`) costs more than the vector loads save. They stay at 1, where no copy is
  made. A persistent cell-fastest layout, which would remove the per-launch transpose, is a
  question for the M10 re-run.
- **Geometry on CPUs gains from small tiles and vectors.** GeomFactors runs 2.0–3.0× faster in
  f32 at size 4 on both CPUs; in f64 it gains 1.6–1.7× on the M3 Max but 0.96–1.13× on Grace.
  The policy takes size 4 at tile 4 for both types (the tile entry is per class and element
  size). EVector takes 16 (f32) and 8 (f64) on
  CpuSimd128, driven by Grace.
- **Wide vectors hurt geometry on GPUs.** EVector and GeomFactors on the H100 run at 0.02–0.34×
  at sizes 8 and 16 (register pressure; f64 worst), and Metal's GeomFactors loses 22–29% at
  size 4. The Gpu
  class keeps 1 for EVector, GeomFactors, Gather, TopologyKeys and PermutationInfo, and the
  QuadField tile stays 16.

## CPU units (question 8, ADR-0017)

The sweep measured every CPU configuration with all hardware threads and with the performance
cores. Grace reports no performance-core count, so it gives both settings the same time and
the M3 Max decides. The policy takes the performance cores for 12 of the 17 CpuSimd128 entries
with data, mostly the kernels with barriers or several phases: Reduce (1.6× at 4 bytes; 3.7× /
1.2× at 8 bytes), ScanTile (1.7× / 2.6×), SortDigits (1.6× / 1.1×), Gather, TopologyKeys,
PermutationInfo, the 4-byte Fill entry, f64 EVector and both Scalar entries. GeomFactors keeps
all threads in both types, as T14's preliminary sweep found (below), and so do Copy, f64 Fill
and f32 EVector. The reasons in
the tables give each gain over the best admitted all-threads configuration.

## Compile times

The compile time of a configuration is its warm-up minus its median; only the first
configuration of each vector size compiles, and the first configuration of a family also pays
one-time setup. Per device, the median over families is a few milliseconds at every size (CPU
runtime 0.3–3.4 ms, CUDA below 0.1 ms, Metal 0.1–1.0 ms). The largest are the geometry kernels:

| Device | Family | 1 | 2 | 4 | 8 | 16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| M3 Max CPU | GeomFactors | 2,775 ms | 155 ms | 184 ms | 273 ms | 521 ms |
| M3 Max CPU | EVector | 105 ms | 28 ms | 45 ms | 86 ms | 241 ms |
| Grace CPU | GeomFactors | 988 ms | 245 ms | 286 ms | 409 ms | 738 ms |
| Grace CPU | EVector | 80 ms | 36 ms | 57 ms | 107 ms | 287 ms |
| H100 | GeomFactors | 216 ms | 210 ms | 216 ms | 243 ms | 295 ms |
| H100 | EVector | 34 ms | 46 ms | 82 ms | 178 ms | 476 ms |
| Metal | GeomFactors | 239 ms | 368 ms | 610 ms | (ran 4) | (ran 4) |

Above size 2 the compile time of the geometry kernels grows with the vector size, by about 2–3×
from 2 to 16 on the CPUs and up to 10× for EVector on the H100; the other families stay below
about 0.4 s at every size. These are one-time costs per process, which CubeCL's kernel cache
removes in later runs.

## Throughput (acceptance bullet 6)

`mesh_throughput` (f64, about 10^6 cells, the median of ten runs; ADR-0018 timing), with the
committed policy and with every kernel at vector size 1 (`--vector-size scalar`; blocks and
units still from the policy). Entities: edges and faces created by `create_entities(1)` and
`create_entities(2)` on fresh P1 meshes. Factors: (cell, quadrature point) pairs for which X, J,
det J and K are computed on P2 tetrahedra with a degree-4 rule.

| Device | Measure | Mesh | Policy | Scalar | Policy / scalar |
| --- | --- | --- | ---: | ---: | ---: |
| M3 Max CPU | entities / s | P1 tetrahedra (998,250 cells) | 1.48e7 | 1.41e7 | 1.05 |
| M3 Max CPU | entities / s | P1 hexahedra | 8.35e6 | 8.46e6 | 0.99 |
| M3 Max CPU | factors / s | P2 tetrahedra | 3.13e8 | 2.48e8 | 1.26 |
| Kathleen CPU | – | – | – | – | not yet measured |

M3 Max run: commit b96f49c (the committed policy), 2026-10-25 20:07, 32 s; mains power and
other applications closed (confirmed by the maintainer); 1-minute load average 2.98 before,
Microsoft Defender at 0.14 of a core and Spotlight at 0.08. Entity creation is dominated by the
radix sort and the scalar helpers, which the policy leaves at size 1 on CPUs, so it changes
little; the geometric factors follow GeomFactors' size-4 entry.

## CPU `cube_dim` and test times (T14)

The M3 review follow-up m6 (docs/plans/M3.md, question 20) found the test-time budgets exceeded:
debug `cargo test --workspace` took 16:01 on the idle M3 Max (budget about 15 min) and the
hosted cpu job 35:17 (budget about 30 min). gridcl-device's `primitives_prop` took most of it
(545 s debug). The M2 review attributed it to the `sync_cube` barriers of 16-unit CPU cubes.
The "CPUs" rule of `LaunchPlan` sets `cube_dim` to `max_units_per_cube`, which is 16 on the M3
Max (12 performance and 4 efficiency cores).

### Method

- Commit 41f2d64 plus a temporary, uncommitted cap in `plan.rs`, read from an environment
  variable, so that every run used the same debug build. The cap lowers the CPU `cube_dim` and
  changes nothing else.
- M3 Max on mains power. macOS background services (indexing, Defender, Teams) used about 1.5
  cores; no other cargo job ran. Runs were one at a time.
- Debug `cargo test -p gridcl-device --test primitives_prop`, and debug
  `cargo test --workspace --no-fail-fast`, at `cube_dim` 16, 12 and 4. libtest's default of
  16 test threads applies, unless stated otherwise.
- Wall times include cargo's fingerprint check (the build was done beforehand, 33 s).

### Results

| CPU `cube_dim` | Test threads | `primitives_prop` | `primitives_large` | `cargo test --workspace` |
| --- | --- | --- | --- | --- |
| 16 (current rule) | 16 | 541 s | 90 s | 949 s (15:49) |
| 12 (performance cores) | 16 | 340 s | 38 s | 572 s (9:32) |
| 4 | 16 | 100 s | 43 s | 292 s (4:52) |
| 16 | 4 | 230 s (203 s in the workspace run) | 88 s | 596 s (9:56) |
| 16 | 1 | 96 s | – | – |

Every workspace run passed 552 tests, except for the four `perf_record` tests, which wait for
T17's data. At `cube_dim` 12 and 4, `cpu_plan_one_unit_per_thread_whole_lines` also failed, as
expected: it asserts the current rule.

### Observations

- **Oversubscription dominates `primitives_prop`.** Its cases run 16 at a time, and each launch
  starts `cube_dim` worker threads that wait at every `sync_cube`. At `cube_dim` 16 with one
  test thread, `primitives_prop` takes 96 s, the same as at `cube_dim` 4 with 16 test threads.
  The barrier cost comes from up to 256 workers on 16 cores, not from 16-unit cubes as such.
- **Efficiency cores slow large single launches.** `primitives_large` runs its 10^6 + 3 cases
  with little test parallelism. It takes 88–90 s at 16 units, whatever the test threads, but
  38 s at 12. With static blocks, the units on efficiency cores finish last, and the barriers
  wait for them ("Benchmarks", "Conditions").
- **The budgets.** Debug `cargo test --workspace` on the M3 Max meets the 15 min budget either
  with `cube_dim` 12 (9:32) or with 4 test threads at `cube_dim` 16 (9:56). The hosted runner
  (4 vCPUs of one kind) has no efficiency cores, so only test-thread limits can help it; it was
  not measured here.

### Preliminary sweep: `cube_dim` 12 against 16

A short release sweep with the T8 binary (`vector_sweep --cube-dim both`, vector sizes 1 and 4,
16 KiB blocks, under the same background load) compared one launch per timed call. The table
gives the median at `cube_dim` 12 divided by the median at 16; below 1 means 12 is faster.
These numbers do not meet T15's conditions (other applications closed, confirmed by the
maintainer) and do not enter the policy.

| Family | Type | 10^6, v = 1 | 10^6, v = 4 | 10^7, v = 1 | 10^7, v = 4 |
| --- | --- | --- | --- | --- | --- |
| Fill | f32 | 0.73 | 0.82 | 0.90 | 0.69 |
| Fill | u32 | 0.69 | 0.79 | 0.68 | 0.92 |
| Copy | u32 | 0.92 | 0.60 | 0.99 | 0.96 |
| Reduce | f32 | 0.20 | 0.26 | 0.45 | 0.51 |
| Reduce | u32 | 0.11 | 0.18 | 0.41 | 0.33 |
| ScanTile | u32 | 0.43 | 0.35 | 0.50 | 0.34 |
| SortDigits | u32 | 0.75 | 0.67 | 0.61 | 0.60 |
| Gather | f32 | 0.78 | 0.76 | 0.90 | 0.86 |
| Gather | u32 | 0.42 | 0.44 | 0.89 | 0.87 |
| TopologyKeys | u32 | 0.52 | 0.36 | – | – |
| GeomFactors | f64 | 1.23 | 1.11 | – | – |

The kernels with barriers or several phases (Reduce, ScanTile, SortDigits, TopologyKeys) gain
the most. GeomFactors, which is compute-bound and has no barriers, is 11–23% slower at 12
units, because it loses the efficiency cores' work. A single rule for every kind would therefore
cost GeomFactors. ADR-0017 proposes letting the policy choose the unit count per kind and
element size from T15's data.

## Results

<!-- BEGIN GENERATED by policy_report; do not edit -->

### Vector sizes

Speedup of each vector size over size 1 at 10^6 / 10^7 items: the best time over tiles blocks and CPU units at size 1 over the best at that size; for a class measured on several devices, the geometric mean of the per-device ratios (ADR-0015). "(ran m)": every device ran the size at m lanes. Size, tile, block and units are the entries of `vector-policy.toml`; f32 and u32 share the 4-byte entries. Units: all hardware threads or the performance cores; a device without a performance-core count gives both the same time (ADR-0017). A reason "closest, x% slower" marks an entry where no admitted configuration is within 3% of the fastest at both sizes, so the rule took the one with the smallest worst-case ratio, at most 10% slower; "unresolved" marks one where even that is more than 10% slower (ADR-0018).

| Family | Class | Type | 2 | 4 | 8 | 16 | Size | Tile | Block | Units | Reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Fill | CpuSimd128 | f32 | 1.19 / 1.45 | 1.28 / 2.53 | 1.24 / 2.80 | 1.26 / 2.85 | 16 | – | 64 KiB | performance | closest, 3.9% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| Fill | CpuSimd128 | f64 | 1.07 / 1.37 | 1.05 / 2.19 | 1.08 / 2.35 | 1.06 / 2.28 | 1 | – | 64 KiB | all | no size at least 10% faster than 1 at 10^6 and 10^7 |
| Fill | CpuSimd128 | u32 | 1.19 / 1.47 | 1.28 / 2.81 | 1.25 / 2.93 | 1.23 / 2.87 | 16 | – | 64 KiB | performance | closest, 3.9% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| Fill | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Fill | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Fill | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Fill | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Fill | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Fill | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Fill | Gpu | f32 | 0.98 / 1.28 | 1.08 / 1.41 | 1.02 / 1.15 | 1.00 / 1.06 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7; size 2 admitted for u32 only (shared 4-byte entry) |
| Fill | Gpu | f64 | 1.06 / 1.10 | 0.87 / 0.68 | 0.83 / 0.57 | 0.76 / 0.28 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| Fill | Gpu | u32 | 1.11 / 1.28 | 1.09 / 1.42 | 1.04 / 1.19 | 1.02 / 1.02 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7; size 2 admitted for u32 only (shared 4-byte entry) |
| Copy | CpuSimd128 | u32 | 1.34 / 1.06 | 1.39 / 2.06 | 1.43 / 2.27 | 1.38 / 2.17 | 8 | – | 64 KiB | all | 1.43x / 2.27x over size 1; smallest admitted size within 3% of the fastest |
| Copy | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Copy | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Copy | Gpu | u32 | 1.12 / 1.07 | 1.11 / 1.14 | 1.10 / 1.14 | 1.07 / 1.01 | 4 | – | – | – | 1.11x / 1.14x over size 1; smallest admitted size within 3% of the fastest |
| Reduce | CpuSimd128 | f32 | 1.08 / 1.19 | 1.16 / 1.31 | 1.26 / 1.39 | 1.17 / 1.43 | 16 | – | 64 KiB | performance | 1.17x / 1.17x over size 1; smallest admitted size within 3% of the fastest; performance cores, 1.58x / 1.57x over all threads; size 8 admitted for f32 only (shared 4-byte entry) |
| Reduce | CpuSimd128 | f64 | 1.07 / 0.99 | 1.17 / 1.07 | 1.03 / 1.05 | 1.16 / 0.99 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 3.67x / 1.22x over all threads |
| Reduce | CpuSimd128 | u32 | 1.16 / 1.10 | 1.24 / 1.11 | 1.35 / 1.09 | 1.37 / 1.17 | 16 | – | 64 KiB | performance | 1.17x / 1.17x over size 1; smallest admitted size within 3% of the fastest; performance cores, 1.58x / 1.57x over all threads; size 8 admitted for f32 only (shared 4-byte entry) |
| Reduce | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Reduce | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Reduce | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Reduce | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Reduce | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Reduce | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Reduce | Gpu | f32 | 1.03 / 1.50 | 1.65 / 1.96 | 1.48 / 2.07 | 1.63 / 2.19 | 16 | – | – | – | closest, 6.7% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018); size 2 admitted for u32 only (shared 4-byte entry) |
| Reduce | Gpu | f64 | 1.05 / 1.39 | 1.36 / 1.60 | 1.34 / 1.57 | 1.29 / 1.65 | 4 | – | – | – | closest, 3.2% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| Reduce | Gpu | u32 | 1.28 / 1.47 | 1.43 / 1.86 | 1.56 / 2.02 | 1.47 / 1.89 | 16 | – | – | – | closest, 6.7% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018); size 2 admitted for u32 only (shared 4-byte entry) |
| ScanTile | CpuSimd128 | u32 | 1.11 / 1.05 | 1.08 / 1.11 | 1.14 / 1.12 | 1.21 / 1.11 | 16 | – | 64 KiB | performance | 1.21x / 1.11x over size 1; smallest admitted size within 3% of the fastest; performance cores, 1.71x / 2.61x over all threads |
| ScanTile | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| ScanTile | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| ScanTile | Gpu | u32 | 1.12 / 1.48 | 1.56 / 1.82 | 1.53 / 1.85 | 1.51 / 1.88 | 8 | – | – | – | 1.53x / 1.85x over size 1; smallest admitted size within 3% of the fastest |
| SortDigits | CpuSimd128 | u32 | 1.01 / 1.00 | 0.99 / 0.97 | 0.97 / 1.00 | 0.97 / 1.00 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.57x / 1.13x over all threads |
| SortDigits | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| SortDigits | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| SortDigits | Gpu | u32 | 1.36 / 1.46 | 1.49 / 1.72 | 1.45 / 1.63 | 1.44 / 1.42 | 4 | – | – | – | 1.49x / 1.72x over size 1; smallest admitted size within 3% of the fastest |
| Gather | CpuSimd128 | f32 | 1.24 / 0.99 | 1.25 / 0.98 | 1.27 / 1.00 | 1.39 / 0.98 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.11x / 1.03x over all threads |
| Gather | CpuSimd128 | f64 | 1.04 / 1.00 | 1.10 / 1.05 | 1.20 / 1.03 | 1.14 / 1.00 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.63x / 1.00x over all threads |
| Gather | CpuSimd128 | u32 | 1.09 / 1.00 | 1.30 / 0.99 | 1.44 / 1.01 | 1.38 / 1.03 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.11x / 1.03x over all threads |
| Gather | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Gather | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Gather | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| Gather | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Gather | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Gather | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| Gather | Gpu | f32 | 1.18 / 1.00 | 1.26 / 1.00 | 1.18 / 0.98 | 0.96 / 0.83 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| Gather | Gpu | f64 | 1.00 / 1.00 | 0.96 / 1.00 | 0.90 / 0.86 | 0.84 / 0.54 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| Gather | Gpu | u32 | 0.93 / 1.00 | 1.00 / 1.00 | 1.17 / 0.98 | 1.19 / 0.83 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| Pack | CpuSimd128 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| Pack | CpuSimd128 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| Pack | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no kernel before M6/M9; measured in the M10 re-run |
| Pack | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no kernel before M6/M9; measured in the M10 re-run |
| Pack | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| Pack | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| Pack | Gpu | f32 | – | – | – | – | 1 | – | – | – | no kernel before M6/M9; measured in the M10 re-run |
| Pack | Gpu | f64 | – | – | – | – | 1 | – | – | – | no kernel before M6/M9; measured in the M10 re-run |
| TopologyKeys | CpuSimd128 | u32 | 0.65 / 0.64 | 0.65 / 0.62 | 0.66 / 0.67 | 0.85 / 0.81 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.07x / 1.13x over all threads |
| TopologyKeys | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| TopologyKeys | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| TopologyKeys | Gpu | u32 | 0.77 / 0.75 | 0.75 / 0.75 | 0.65 / 0.65 | 0.62 / 0.60 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| PermutationInfo | CpuSimd128 | u32 | 0.76 / 0.77 | 0.86 / 0.87 | 0.88 / 0.85 | 0.87 / 0.82 | 1 | – | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.07x / 1.10x over all threads |
| PermutationInfo | CpuSimd256 | u32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| PermutationInfo | CpuSimd512 | u32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| PermutationInfo | Gpu | u32 | 0.72 / 0.54 | 0.76 / 0.54 | 0.76 / 0.51 | 0.69 / 0.45 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| EVector | CpuSimd128 | f32 | 1.06 / 1.02 | 1.07 / 1.03 | 1.13 / 1.06 | 1.38 / 1.24 | 16 | – | 64 KiB | all | 1.38x / 1.24x over size 1; smallest admitted size within 3% of the fastest |
| EVector | CpuSimd128 | f64 | 1.02 / 1.02 | 1.01 / 1.10 | 1.18 / 1.47 | 1.08 / 1.37 | 8 | – | 16 KiB | performance | 1.18x / 1.47x over size 1; smallest admitted size within 3% of the fastest; performance cores, 1.01x / 1.01x over all threads |
| EVector | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| EVector | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no data |
| EVector | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| EVector | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no data |
| EVector | Gpu | f32 | 0.97 / 0.96 | 0.79 / 0.79 | 0.55 / 0.53 | 0.57 / 0.56 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| EVector | Gpu | f64 | 0.88 / 0.90 | 0.47 / 0.44 | 0.22 / 0.17 | 0.21 / 0.15 | 1 | – | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| GeomFactors | CpuSimd128 | f32 | 1.68 / 1.57 | 2.45 / 2.83 | 2.23 / 2.24 | 1.93 / 2.09 | 4 | 4 | 4 KiB | all | closest, 5.0% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| GeomFactors | CpuSimd128 | f64 | 1.17 / 1.21 | 1.40 / 1.33 | 1.18 / 1.00 | 1.09 / 0.96 | 4 | 4 | 64 KiB | all | closest, 5.7% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| GeomFactors | CpuSimd256 | f32 | – | – | – | – | 1 | 16 | 16 KiB | all | copies CpuSimd512: no data |
| GeomFactors | CpuSimd256 | f64 | – | – | – | – | 1 | 16 | 16 KiB | all | copies CpuSimd512: no data |
| GeomFactors | CpuSimd512 | f32 | – | – | – | – | 1 | 16 | 16 KiB | all | no data |
| GeomFactors | CpuSimd512 | f64 | – | – | – | – | 1 | 16 | 16 KiB | all | no data |
| GeomFactors | Gpu | f32 | 0.98 / 0.99 | 0.80 / 0.83 | 0.21 / 0.21 | 0.26 / 0.25 | 1 | 16 | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| GeomFactors | Gpu | f64 | 1.00 / 1.01 | 0.54 / 0.57 | 0.04 / 0.04 | 0.02 / 0.03 | 1 | 16 | – | – | no size at least 10% faster than 1 at 10^6 and 10^7 |
| PullBack | CpuSimd128 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| PullBack | CpuSimd128 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| PullBack | CpuSimd256 | f32 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no kernel before M6/M9; measured in the M10 re-run |
| PullBack | CpuSimd256 | f64 | – | – | – | – | 1 | – | 16 KiB | all | copies CpuSimd512: no kernel before M6/M9; measured in the M10 re-run |
| PullBack | CpuSimd512 | f32 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| PullBack | CpuSimd512 | f64 | – | – | – | – | 1 | – | 16 KiB | all | no kernel before M6/M9; measured in the M10 re-run |
| PullBack | Gpu | f32 | – | – | – | – | 1 | – | – | – | no kernel before M6/M9; measured in the M10 re-run |
| PullBack | Gpu | f64 | – | – | – | – | 1 | – | – | – | no kernel before M6/M9; measured in the M10 re-run |

### Per device

The speedups of each measured device, at 10^6 / 10^7 items.

| Family | Type | Class | Device | 2 | 4 | 8 | 16 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Fill | f32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.02 / 1.58 | 1.11 / 4.79 | 1.03 / 5.74 | 1.04 / 5.95 |
| Fill | f32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.39 / 1.34 | 1.49 / 1.28 | 1.52 / 1.32 | 1.54 / 1.33 |
| Fill | f32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.05 / 1.42 | 1.10 / 1.64 | 1.01 / 1.07 | 0.96 / 0.94 |
| Fill | f32 | Gpu | m3max metal (Apple M3 Max) | 0.90 / 1.16 | 1.05 / 1.21 | 1.04 / 1.23 (ran 4) | 1.04 / 1.20 (ran 4) |
| Fill | f64 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.01 / 1.69 | 0.96 / 4.26 | 1.00 / 4.84 | 0.92 / 4.54 |
| Fill | f64 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.11 / 1.11 | 1.18 / 1.14 | 1.15 / 1.14 | 1.19 / 1.14 |
| Fill | f64 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.06 / 1.10 | 0.87 / 0.68 | 0.83 / 0.57 | 0.76 / 0.28 |
| Fill | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.08 / 1.62 | 1.09 / 5.66 | 1.03 / 6.06 | 1.01 / 5.86 |
| Fill | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.27 / 1.29 | 1.44 / 1.34 | 1.44 / 1.36 | 1.45 / 1.36 |
| Fill | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.07 / 1.42 | 1.09 / 1.64 | 1.00 / 1.08 | 0.96 / 0.93 |
| Fill | u32 | Gpu | m3max metal (Apple M3 Max) | 1.16 / 1.15 | 1.09 / 1.22 | 1.09 / 1.33 (ran 4) | 1.09 / 1.11 (ran 4) |
| Copy | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.16 / 1.35 | 1.14 / 4.51 | 1.16 / 5.46 | 1.11 / 4.72 |
| Copy | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.23 / 1.05 | 1.34 / 1.08 | 1.40 / 0.95 | 1.37 / 0.99 |
| Copy | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.06 / 1.25 | 1.07 / 1.30 | 1.03 / 1.28 | 0.98 / 1.00 |
| Copy | u32 | Gpu | m3max metal (Apple M3 Max) | 1.19 / 0.92 | 1.16 / 1.00 | 1.17 / 1.01 (ran 4) | 1.17 / 1.01 (ran 4) |
| Reduce | f32 | CpuSimd128 | locust cpu (Neoverse-V2) | 0.98 / 0.97 | 1.00 / 1.02 | 1.07 / 1.01 | 0.92 / 1.09 |
| Reduce | f32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.20 / 1.45 | 1.35 / 1.69 | 1.48 / 1.91 | 1.48 / 1.87 |
| Reduce | f32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.07 / 1.39 | 1.36 / 1.78 | 1.39 / 2.04 | 1.35 / 2.30 |
| Reduce | f32 | Gpu | m3max metal (Apple M3 Max) | 0.99 / 1.61 | 2.02 / 2.16 | 1.59 / 2.10 (ran 4) | 1.97 / 2.09 (ran 4) |
| Reduce | f64 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.08 / 0.95 | 1.07 / 0.94 | 0.91 / 0.98 | 1.03 / 0.97 |
| Reduce | f64 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.07 / 1.04 | 1.28 / 1.22 | 1.17 / 1.12 | 1.30 / 1.00 |
| Reduce | f64 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.05 / 1.39 | 1.36 / 1.60 | 1.34 / 1.57 | 1.29 / 1.65 |
| Reduce | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.09 / 0.97 | 1.20 / 0.92 | 1.24 / 0.95 | 1.18 / 0.93 |
| Reduce | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.24 / 1.24 | 1.29 / 1.33 | 1.47 / 1.26 | 1.60 / 1.47 |
| Reduce | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.08 / 1.39 | 1.33 / 1.79 | 1.41 / 2.01 | 1.36 / 2.38 |
| Reduce | u32 | Gpu | m3max metal (Apple M3 Max) | 1.51 / 1.56 | 1.55 / 1.92 | 1.72 / 2.02 (ran 4) | 1.59 / 1.50 (ran 4) |
| ScanTile | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.10 / 1.03 | 0.97 / 1.00 | 1.05 / 1.01 | 1.09 / 1.01 |
| ScanTile | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.12 / 1.07 | 1.20 / 1.23 | 1.23 / 1.24 | 1.33 / 1.22 |
| ScanTile | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.06 / 1.50 | 1.60 / 1.98 | 1.54 / 2.04 | 1.49 / 2.09 |
| ScanTile | u32 | Gpu | m3max metal (Apple M3 Max) | 1.19 / 1.46 | 1.52 / 1.68 | 1.52 / 1.68 (ran 4) | 1.53 / 1.69 (ran 4) |
| SortDigits | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.05 / 0.99 | 1.01 / 0.94 | 1.01 / 1.00 | 0.98 / 1.01 |
| SortDigits | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 0.98 / 1.01 | 0.97 / 1.00 | 0.94 / 1.00 | 0.95 / 1.00 |
| SortDigits | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.25 / 1.31 | 1.23 / 1.27 | 1.15 / 1.13 | 1.14 / 0.86 |
| SortDigits | u32 | Gpu | m3max metal (Apple M3 Max) | 1.49 / 1.62 | 1.81 / 2.34 | 1.81 / 2.34 (ran 4) | 1.83 / 2.35 (ran 4) |
| Gather | f32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.37 / 0.98 | 1.42 / 0.99 | 1.19 / 1.02 | 1.52 / 0.98 |
| Gather | f32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.10 / 1.02 | 1.23 / 0.98 | 1.30 / 0.98 | 1.30 / 1.02 |
| Gather | f32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.98 / 1.00 | 0.98 / 0.99 | 0.97 / 0.96 | 0.95 / 0.69 |
| Gather | f32 | Gpu | m3max metal (Apple M3 Max) | 1.42 / 1.00 | 1.61 / 1.00 | 1.44 / 0.99 (ran 4) | 0.98 / 1.00 (ran 4) |
| Gather | f64 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.01 / 1.00 | 1.08 / 1.09 | 1.32 / 1.05 | 1.35 / 1.05 |
| Gather | f64 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.08 / 1.00 | 1.12 / 1.01 | 1.16 / 1.02 | 1.07 / 0.96 |
| Gather | f64 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.00 / 1.00 | 0.96 / 1.00 | 0.90 / 0.86 | 0.84 / 0.54 |
| Gather | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.13 / 1.00 | 1.57 / 0.99 | 1.90 / 1.02 | 1.62 / 1.05 |
| Gather | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.02 / 1.00 | 1.09 / 1.00 | 1.14 / 1.00 | 1.19 / 1.02 |
| Gather | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.99 / 1.00 | 0.98 / 0.99 | 0.97 / 0.96 | 0.95 / 0.69 |
| Gather | u32 | Gpu | m3max metal (Apple M3 Max) | 0.87 / 1.01 | 1.02 / 1.00 | 1.41 / 1.00 (ran 4) | 1.48 / 1.00 (ran 4) |
| TopologyKeys | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 0.57 / 0.56 | 0.52 / 0.50 | 0.55 / 0.59 | 0.95 / 0.83 |
| TopologyKeys | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 0.77 / 0.72 | 0.79 / 0.77 | 0.78 / 0.77 | 0.76 / 0.79 |
| TopologyKeys | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.79 / 0.74 | 0.79 / 0.74 | 0.58 / 0.56 | 0.53 / 0.48 |
| TopologyKeys | u32 | Gpu | m3max metal (Apple M3 Max) | 0.75 / 0.76 | 0.72 / 0.75 | 0.73 / 0.75 (ran 4) | 0.72 / 0.75 (ran 4) |
| PermutationInfo | u32 | CpuSimd128 | locust cpu (Neoverse-V2) | 0.58 / 0.81 | 0.65 / 0.86 | 0.73 / 0.88 | 0.66 / 0.85 |
| PermutationInfo | u32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 0.99 / 0.71 | 1.15 / 0.84 | 1.10 / 0.79 | 1.16 / 0.80 |
| PermutationInfo | u32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.71 / 0.63 | 0.71 / 0.63 | 0.66 / 0.57 | 0.60 / 0.45 |
| PermutationInfo | u32 | Gpu | m3max metal (Apple M3 Max) | 0.74 / 0.46 | 0.82 / 0.46 | 0.88 / 0.46 (ran 4) | 0.79 / 0.45 (ran 4) |
| EVector | f32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.02 / 1.01 | 1.02 / 1.00 | 1.13 / 1.02 | 1.92 / 1.56 |
| EVector | f32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.08 / 1.02 | 1.13 / 1.06 | 1.13 / 1.09 | 0.98 / 0.97 |
| EVector | f32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.95 / 0.97 | 0.64 / 0.66 | 0.32 / 0.29 | 0.34 / 0.33 |
| EVector | f32 | Gpu | m3max metal (Apple M3 Max) | 0.99 / 0.95 | 0.98 / 0.95 | 0.95 / 0.95 (ran 4) | 0.98 / 0.94 (ran 4) |
| EVector | f64 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.00 / 1.01 | 0.97 / 1.12 | 1.38 / 2.14 | 1.28 / 1.98 |
| EVector | f64 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.05 / 1.03 | 1.04 / 1.04 | 0.99 / 1.00 | 0.90 / 0.92 |
| EVector | f64 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.88 / 0.90 | 0.47 / 0.44 | 0.22 / 0.17 | 0.21 / 0.15 |
| GeomFactors | f32 | CpuSimd128 | locust cpu (Neoverse-V2) | 1.39 / 1.42 | 2.03 / 2.63 | 1.83 / 1.64 | 1.61 / 1.60 |
| GeomFactors | f32 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.87 / 1.71 | 2.84 / 2.99 | 2.49 / 3.01 | 2.35 / 2.68 |
| GeomFactors | f32 | Gpu | locust cuda (NVIDIA GH200 480GB) | 0.98 / 0.99 | 0.90 / 0.89 | 0.06 / 0.05 | 0.10 / 0.08 |
| GeomFactors | f32 | Gpu | m3max metal (Apple M3 Max) | 0.98 / 0.99 | 0.71 / 0.78 | 0.71 / 0.78 (ran 4) | 0.71 / 0.78 (ran 4) |
| GeomFactors | f64 | CpuSimd128 | locust cpu (Neoverse-V2) | 0.94 / 0.82 | 1.13 / 0.96 | 1.02 / 0.79 | 1.08 / 0.75 |
| GeomFactors | f64 | CpuSimd128 | m3max cpu (Apple M3 Max) | 1.42 / 1.69 | 1.58 / 1.72 | 1.24 / 1.24 | 1.03 / 1.14 |
| GeomFactors | f64 | Gpu | locust cuda (NVIDIA GH200 480GB) | 1.00 / 1.01 | 0.54 / 0.57 | 0.04 / 0.04 | 0.02 / 0.03 |

### Scalar kernels

CPU blocks and units of `KernelKind::Scalar` (vector size 1), from histogram (u32) and segmented reduce (f64) (ADR-0015, item 7; ADR-0017).

| Class | Type | Block | Units | Reason |
| --- | --- | ---: | --- | --- |
| CpuSimd128 | u32 | 64 KiB | performance | no size at least 10% faster than 1 at 10^6 and 10^7; performance cores, 1.06x / 1.05x over all threads |
| CpuSimd128 | f64 | 4 KiB | performance | closest, 4.2% slower: no admitted configuration within 3% of the fastest at both sizes (ADR-0018) |
| CpuSimd256 | u32 | 16 KiB | all | copies CpuSimd512: no data |
| CpuSimd256 | f64 | 16 KiB | all | copies CpuSimd512: no data |
| CpuSimd512 | u32 | 16 KiB | all | no data |
| CpuSimd512 | f64 | 16 KiB | all | no data |

<!-- END GENERATED -->
