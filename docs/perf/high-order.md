# High-order geometry kernels on gridcl's devices (M9h)

This report records the degree sweeps of M9h (docs/design.md, "Benchmarks"; ADR-0041,
ADR-0042, ADR-0043) and, from T11b, how `gridcl-geometry/degree-policy.toml` was chosen. The
tables under "Results" will be written by gridcl-bench's report generator (T11b) from the JSON
under `docs/perf/data/m9h-*.json`, between its marker comments; the rest is written by hand.
Times are UTC.

## Protocol

- **Binary:** `degree_sweep` (gridcl-bench), release build, CubeCL's persistent kernel cache
  off, CUDA's driver cache off (`CUDA_CACHE_DISABLE=1`, ADR-0042 item 6), run by
  `scripts/bench/degree-sweep.sh` under a time limit per configuration (600 s on GPUs, 3600 s on
  CPUs), all processes of a run writing one JSON file with `--resume`. The baseline ran one
  process per (family, cell type, degree, scalar type); from 44d7e70 the script runs one
  process per (cell type, degree), which builds each mesh once, and after a stop one per
  configuration still without a record. A configuration whose process was stopped is recorded
  `too slow to compile (process stopped after S s)` if its kernel had no record yet, else
  `too slow (process stopped after S s)`; larger sizes carry the status over.
- **Families:** EVector; Points (X), Jacobian (J), DetJ, Pinv (K) and FacetNormals on every
  cell type; Normals on intervals, triangles and quadrilaterals one dimension up; PullBack at 1
  and 9 Newton iterations (rule 1: step = (t(9) − t(1)) / 8, set-up = t(1) − step); BasisEval,
  the device basis evaluator on its own (`gridcl_geometry::bench`).
- **Configurations:** p = 1–8 (pyramids 1–3); f32 and f64 (f32 only on Metal); the Gauss rule
  of degree 2p (the facet rule of degree 2p for facet normals) on unperturbed unit boxes with
  about 10^5, 10^6 and 10^7 quadrature points; equispaced nodes; vector size, QuadField tile and
  CPU block from `vector-policy.toml` (rule 12).
- **Timing (ADR-0018):** a warm-up call checked against an f64 host evaluation on up to 64
  sampled cells (rule 2's relative error and tolerance, both recorded); one calibration call
  fixing k = ⌈20 ms / its time⌉ calls per run; ten timed runs; the median per call. The
  compile time is the warm-up minus the median; sizes ascend, so the smallest size's record
  carries a kernel's first use. Time per point = median / (cells × points per cell).
- **Status rules (rule 12):** `too slow to compile` where the warm-up exceeds the calibration
  call by 300 s; on a CPU `too slow` where a call takes over 600 s, measured or extrapolated
  linearly from the previous size; both carry over to larger sizes without running.
- **Locust:** an open node without a scheduler. A run counts only if the runner's node-load
  snapshots say `idle` before and after it (ADR-0002); otherwise it is repeated.

## Conditions

The baseline (rule 11) is `direct` at a88ab2a, the T3 commit, before any dispatch change.

| Target | Backend, class | Device | Run | Wall time | Records | Conditions |
| --- | --- | --- | --- | --- | --- | --- |
| macmini | cpu, CpuSimd128 | Apple M6, 6 performance and 6 efficiency cores | 2026-11-09 18:07–22:19 | 4.2 h | 2,898 | local; load average 2.7 before the run |
| macmini | metal, Gpu | Apple M6 | 2026-11-09 22:19–23:44 | 1.4 h | 1,449 | local, right after the CPU run |
| kathleen | cpu, CpuSimd512 | Xeon Gold 6248, one socket (20 threads, `numactl --physcpubind=0-19 --membind=0`, ADR-0020) | job 631525: 2026-11-09 18:04–22:40; job 631727: 20:03–23:51 | 4.6 h + 3.8 h | 1,980 + 918 | whole nodes (two different nodes); EVector to FacetNormals in 631525, PullBack and BasisEval in 631727 |
| locust | cuda, Gpu | NVIDIA GH200 480GB | job 20261110055956-3563316 (part a): 2026-11-10 05:59–10:08; job 20261110130020-3609133 (part b): 13:00–14:22 | 4.1 h + 1.4 h | 1,980 + 468 | idle before and after, both jobs |
| locust | cpu, CpuSimd128 | Grace, 72 cores | job 20261110144213-3615352: 2026-11-10 14:42–18:43 | 4.0 h | 1,476 | idle before and after; every family at p ∈ {1, 2, 4, 8} (ADR-0043) |

Locust's CUDA run is split into part a (EVector to FacetNormals, p = 1–8) and part b (PullBack
and BasisEval, p ∈ {1, 2, 4, 8}, pyramids {1, 2}; ADR-0043). A first part a (job
20261109175723-3488291) ran while another user's process held the GPU at 100% and is not used.
The pinned runner's start command returned only long after each job had ended (fixed in
f95b07d), so the results of all three Locust jobs were copied from the node by hand after their exit status was
written.

## Baseline (T3)

### Records

Every configuration has a record. On the Mac CPU, EVector on hexahedra at p = 7 and 8 in f32
is `too slow to compile` (370 s and 974 s, at sizes 10^5–10^7: six records); every other
record is `ok`. Its compile time triples per degree (12 s at p = 4, 124 s at p = 6), at vector
size 16, the f32 policy size of CpuSimd128; f64 at vector size 8 compiles in 167 s at p = 8.
Kathleen (vector size 1) compiles the same kernel in 15 s, Metal in 5 s. The cost is CubeCL's
common-subexpression pass on the fully unrolled kernel (docs/cubecl-upstream.md, C20).

On Locust's GPU, PullBack at p = 8 on tetrahedra, hexahedra and prisms, in f32 and f64, is
`too slow to compile (process stopped after 600 s)` (six processes, 36 records, both iteration
counts at every size); it compiles at p = 4. T2 found the direct pull-back does not compile
within 900 s on Locust from p = 6: large unrolled kernels load slowly on CUDA (C18). Every other Locust CUDA record is `ok`.
On Locust's CPU, EVector on hexahedra at p = 8 in f32 is `too slow to compile` (994 s; three
records), as on the Mac; every other record is `ok`.

### Run times

Hours per family, from the start of one process to the start of the next (T11a splits its runs
by these). Locust's CPU run sampled every family at p ∈ {1, 2, 4, 8}, its CUDA run PullBack
and BasisEval only (ADR-0043), with
PullBack's p = 8 stopped after 600 s per process. The discarded busy part a took 4.47 h, of which
its timed and compiling parts were 0.56 h; the rest is host work.

| Target | EVector | Points | Jacobian | DetJ | Pinv | Normals | FacetNormals | PullBack | BasisEval | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macmini cpu | 0.83 | 0.22 | 0.28 | 0.28 | 0.28 | 0.10 | 0.40 | 1.74 | 0.07 | 4.20 |
| macmini metal | 0.08 | 0.09 | 0.14 | 0.16 | 0.13 | 0.05 | 0.21 | 0.52 | 0.02 | 1.40 |
| kathleen cpu | 0.52 | 0.55 | 0.69 | 0.68 | 0.69 | 0.31 | 1.09 | 3.68 | 0.10 | 8.33 |
| locust cuda | 0.51 | 0.53 | 0.59 | 0.62 | 0.58 | 0.28 | 1.00 | 1.34 (sampled) | 0.02 (sampled) | 5.50 |
| locust cpu (sampled) | 0.66 | 0.24 | 0.31 | 0.34 | 0.34 | 0.15 | 0.48 | 1.44 | 0.02 | 4.01 |

### f32 error survey

The largest error over tolerance (rule 2) of any f32 record, per family and cell type, over
p, size and target (Mac CPU, Metal, Kathleen, Locust CUDA and CPU); no record exceeds 1, and the largest
f64 ratio is 0.16.

| Family | Interval | Triangle | Quadrilateral | Tetrahedron | Hexahedron | Prism | Pyramid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Points | 0.007 | 0.012 | 0.013 | 0.017 | 0.038 | 0.026 | 0.014 |
| Jacobian | 0.058 | 0.090 | 0.106 | 0.104 | 0.185 | 0.164 | 0.071 |
| DetJ | 0.058 | 0.133 | 0.115 | 0.179 | 0.209 | 0.215 | 0.149 |
| Pinv | 0.910 | 0.135 | 0.106 | 0.174 | 0.185 | 0.216 | 0.138 |
| Normals | 0.053 | 0.067 | 0.059 | – | – | – | – |
| FacetNormals | 0.053 | 0.083 | 0.033 | 0.103 | 0.086 | 0.143 | 0.092 |
| PullBack | 0.301 | 0.028 | 0.023 | 0.052 | 0.055 | 0.044 | 0.057 |
| BasisEval | 0.001 | 0.001 | 0.002 | 0.002 | 0.001 | 0.001 | 0.002 |

The largest ratio per degree, over every family and cell type: 0.03 (p = 1), 0.07, 0.30, 0.22,
0.18, 0.91 (p = 6), 0.56, 0.17 (p = 8).

### Observations

- **Host time dominates the sweep.** Most of a configuration's time is building its meshes on
  the host: the single-threaded numbering of the mesh pipeline (`gridcl::pipeline::number`) at
  10^7 points. Locust's Grace cores do this 2.5 times slower than the M6 (7,811 s against
  3,097 s on the configurations both ran), so Locust's CUDA run takes as long as Kathleen's CPU
  run. One process per (cell type, degree) saves about 25% (Metal, seven families at interval
  p = 8: 253 s against 189 s); T11a uses it.
- **Intervals at 10^6 and 10^7 points are below f32 resolution.** The unit interval then has
  10^5–5 · 10^6 cells, and at high degree its nodes are h/p ≈ 10^−7 or less apart, one or two
  f32 ulps near 1 (6 · 10^−8). Normals and facet normals come out with the wrong sign in f32
  (error 2; facet normals from p = 4 at 10^7 points and p = 7 at 10^6, normals from p = 6 at
  10^7, identically on every target; Locust's CPU at its sampled degrees too), and K = 1/J
  reaches 0.91 of its tolerance (p = 6, 10^7 points, likewise on every device that ran p = 6). Rule 2's tolerance, which scales with the amplification and |K|, accepts both, and
  f64 is exact. Interval data at these sizes in f32 measures speed, not accuracy.
- **CPU compile time at high degree** (Records above, C20): on CpuSimd128, f32 kernels at vector
  size 16 compile in minutes from p = 6. The degree policy and T11a's strategies with runtime
  loops avoid the fully unrolled kernels.
