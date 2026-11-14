# High-order geometry kernels on gridcl's devices (M9h)

This report records the degree sweeps of M9h (docs/design.md, "Benchmarks"; ADR-0041,
ADR-0042, ADR-0043, ADR-0050 to ADR-0053) and how `gridcl-geometry/degree-policy.toml` was
chosen. The tables under "Results" are written by
`cargo run -p gridcl-bench --bin high_order_report` from the JSON under
`docs/perf/data/m9h-*.json`, between its marker comments (`--check` verifies them and the
policy); the rest is written by hand. Times are UTC.

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

## Strategy sweeps (T11a)

Every strategy at p = 1–8 (pyramids 1–3) on equispaced nodes, at f1c32de and later sweep-tool
commits (`GRIDCL_BENCH_COMMIT` and the runner's commit in each record), launch-only records
first:

- **Mac mini**, CPU (CpuSimd128) and Metal (Gpu), locally;
- **Kathleen**, one socket (CpuSimd512), jobs 633915, 633916, 633993, 634018, 634027, 634061
  and 634062;
- **Locust**, the H100 (Gpu): launch-only 20261114002010-3021357, `ptxas` on its PTX
  20261114041724-3295800, timed parts a and b 20261114044005-3485261 and
  20261114084900-3533108, each idle before and after. Locust's Grace CPU is not swept
  (ADR-0051).

Scope cuts, all accepted on 2026-11-14: `direct` is neither timed nor launched on tetrahedra,
hexahedra and prisms at p ≥ 5, where T3's baseline gives its cost (ADR-0050; records taken
before stay); on the CPU classes the timed sweep runs `direct`, `loops` and `sumfact` only, at
10^5 and 10^6 points with five runs, and the rules read 10^6 points (ADR-0052). `sumfact` does
not run on simplices (rule 10). Metal records three failures (`cooperative` K on intervals at
p = 6–8, f32, 10^7 points, where J rounds to 0); they are not selectable.

## Decision rule (T11b)

`gridcl-bench/src/decide_degree.rs`, per policy family, cell type, measured class (CpuSimd128:
the Mac mini's CPU; CpuSimd512: Kathleen; Gpu: Metal and the H100), scalar type and degree:

- a family's time per point is the sum of its bench families' (J + det J + K for `Jacobian`, the
  pull-back at 9 Newton iterations for `PullBack`), on the codimension-0 meshes (codimension 1
  for `Normals`);
- a strategy is selectable if every device of the class has `ok` records of it at every rule
  size (Gpu 10^6 and 10^7 points, CPU classes 10^6) that ran the strategy itself, not its
  fallback `loops`; on the Gpu class, also no failed launch and no first use over 30 s on
  Locust (rule 7). The CPU classes choose among `direct`, `loops` and `sumfact` (ADR-0052);
- the baseline is `direct`, or `loops` where `direct` is not selectable; a strategy is admitted
  if its class ratio to the baseline (geometric mean of the device ratios) is at most 0.9 at
  every rule size; among the admitted within 3% of the fastest at every size the simplest wins
  (`direct` < `loops` < `sumfact` < `shared` < `cooperative`), else the one with the smallest
  worst ratio;
- ranges are maximal runs of equal choices; after a `direct` choice at the highest measured
  degree the open-ended last range is `loops`; CpuSimd256 copies CpuSimd512.

The ranges follow the data degree by degree, so classes where two strategies are within about
10% of each other alternate between them (for example `direct` and `loops` on intervals and
triangles).

**Bullet 3 (ADR-0053).** The p = 8 / p = 4 ratios of the selected strategies exceed ADR-0042
item 1's bounds in 71 of 120 cases (factors 42 of 88, the pull-back 29 of 32): the
sum-factorised kernels keep per-lane intermediates of (p + 1)^2 · q values that spill at p = 8
(on CUDA a 20–57 KB stack per thread for the hexahedron Jacobian), and `direct` and `loops`
set up the pull-back with C′, (p + 1)^d work per point. At p = 8 the selected strategies are
nonetheless 16–161 times faster than M9's kernels on hexahedron factors. The ratios are reported
in "Rule 1" below, not enforced; M10 takes the bound as its target.

**M3's reference comparison (ADR-0054).** The CPU policy runs `sumfact` from p = 1–2 on tensor
cells, so `factor_invariance` compares with `direct` forced bitwise and, under the policy, within
the tolerances where a strategy that reorders the sums ran.

## Results

<!-- BEGIN GENERATED by policy_report; do not edit -->

### Degree policy

One row per entry of `gridcl-geometry/degree-policy.toml` and degree range. Ratio: the class time of the chosen strategy over the baseline's (`direct`, or `loops` where `direct` is not selectable), over the range's degrees and rule sizes. The last range is open-ended; after a `direct` choice at the highest measured degree it is `loops`, which compiles at every degree (ADR-0042, item 3). CpuSimd256 has CpuSimd512's table.

| Family | Cell | Class | Type | Degrees | Strategy | Ratio | Reason |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| EVector | Interval | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Interval | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| EVector | Interval | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| EVector | Interval | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| EVector | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| EVector | Interval | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Interval | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Triangle | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Triangle | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Triangle | CpuSimd256 | f32 | 1–5 | direct | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f32 | 6–6 | loops | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f32 | 7–7 | direct | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f32 | 8– | loops | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f64 | 1–5 | direct | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f64 | 6–6 | loops | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f64 | 7–7 | direct | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd256 | f64 | 8– | loops | – | as CpuSimd512 |
| EVector | Triangle | CpuSimd512 | f32 | 1–5 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd512 | f32 | 6–6 | loops | 0.78 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Triangle | CpuSimd512 | f32 | 7–7 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd512 | f32 | 8– | loops | 0.75 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Triangle | CpuSimd512 | f64 | 1–5 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd512 | f64 | 6–6 | loops | 0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Triangle | CpuSimd512 | f64 | 7–7 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | CpuSimd512 | f64 | 8– | loops | 0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Triangle | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Triangle | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Triangle | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd256 | f32 | 3–3 | loops | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd256 | f32 | 4–8 | direct | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| EVector | Quadrilateral | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | CpuSimd512 | f32 | 3–3 | loops | 0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Quadrilateral | CpuSimd512 | f32 | 4–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Quadrilateral | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Quadrilateral | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Tetrahedron | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd128 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd128 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd256 | f32 | 1–3 | direct | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd256 | f32 | 4– | loops | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd256 | f64 | 3–3 | loops | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd256 | f64 | 4–4 | direct | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd256 | f64 | 5– | loops | – | as CpuSimd512 |
| EVector | Tetrahedron | CpuSimd512 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd512 | f32 | 4– | loops | 0.89–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Tetrahedron | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd512 | f64 | 3–3 | loops | 0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Tetrahedron | CpuSimd512 | f64 | 4–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | CpuSimd512 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Tetrahedron | Gpu | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | Gpu | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Tetrahedron | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Tetrahedron | Gpu | f64 | 4– | loops | 0.87–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Hexahedron | CpuSimd128 | f32 | 1–5 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | CpuSimd128 | f32 | 6– | loops | 0.82–1.00 | direct not selectable at p = 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Hexahedron | CpuSimd128 | f64 | 1–7 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | CpuSimd128 | f64 | 8– | loops | 1.00 | direct not selectable at p = 8; no strategy 10% faster |
| EVector | Hexahedron | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| EVector | Hexahedron | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| EVector | Hexahedron | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| EVector | Hexahedron | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| EVector | Hexahedron | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Hexahedron | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Hexahedron | Gpu | f32 | 1–7 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | Gpu | f32 | 8– | loops | 1.00 | direct not selectable at p = 8; no strategy 10% faster |
| EVector | Hexahedron | Gpu | f64 | 1–7 | direct | 1.00 | no strategy 10% faster |
| EVector | Hexahedron | Gpu | f64 | 8– | loops | 1.00 | direct not selectable at p = 8; no strategy 10% faster |
| EVector | Prism | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd128 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Prism | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd128 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| EVector | Prism | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f32 | 5–7 | loops | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f32 | 8–8 | direct | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f64 | 1–4 | direct | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f64 | 5–7 | loops | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f64 | 8–8 | direct | – | as CpuSimd512 |
| EVector | Prism | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| EVector | Prism | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f32 | 5–7 | loops | 1.00 | direct not selectable at p = 5, 6, 7; no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f32 | 8–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Prism | CpuSimd512 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f64 | 5–7 | loops | 1.00 | direct not selectable at p = 5, 6, 7; no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f64 | 8–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Prism | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Prism | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| EVector | Prism | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| EVector | Pyramid | CpuSimd128 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | CpuSimd128 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| EVector | Pyramid | CpuSimd128 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | CpuSimd128 | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| EVector | Pyramid | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| EVector | Pyramid | CpuSimd256 | f32 | 3– | loops | – | as CpuSimd512 |
| EVector | Pyramid | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| EVector | Pyramid | CpuSimd256 | f64 | 3– | loops | – | as CpuSimd512 |
| EVector | Pyramid | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | CpuSimd512 | f32 | 3– | loops | 0.80 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Pyramid | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | CpuSimd512 | f64 | 3– | loops | 0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| EVector | Pyramid | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | Gpu | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| EVector | Pyramid | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| EVector | Pyramid | Gpu | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Points | Interval | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Interval | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| Points | Interval | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| Points | Interval | CpuSimd256 | f64 | 1–3 | direct | – | as CpuSimd512 |
| Points | Interval | CpuSimd256 | f64 | 4–5 | loops | – | as CpuSimd512 |
| Points | Interval | CpuSimd256 | f64 | 6–8 | direct | – | as CpuSimd512 |
| Points | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| Points | Interval | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Interval | CpuSimd512 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | CpuSimd512 | f64 | 4–5 | loops | 0.87–0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Interval | CpuSimd512 | f64 | 6–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Triangle | CpuSimd128 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd128 | f32 | 4–4 | loops | 0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd128 | f32 | 5–5 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd128 | f32 | 6–6 | loops | 0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd128 | f32 | 7–7 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd128 | f32 | 8– | loops | 0.73 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd128 | f64 | 5– | loops | 0.71–0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Triangle | CpuSimd256 | f32 | 2– | loops | – | as CpuSimd512 |
| Points | Triangle | CpuSimd256 | f64 | 1–3 | direct | – | as CpuSimd512 |
| Points | Triangle | CpuSimd256 | f64 | 4–4 | loops | – | as CpuSimd512 |
| Points | Triangle | CpuSimd256 | f64 | 5–7 | direct | – | as CpuSimd512 |
| Points | Triangle | CpuSimd256 | f64 | 8– | loops | – | as CpuSimd512 |
| Points | Triangle | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd512 | f32 | 2– | loops | 0.60–0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd512 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd512 | f64 | 4–4 | loops | 0.83 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | CpuSimd512 | f64 | 5–7 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | CpuSimd512 | f64 | 8– | loops | 0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Triangle | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Triangle | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Points | Triangle | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Points | Quadrilateral | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | CpuSimd128 | f32 | 2– | sumfact | 0.15–0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | CpuSimd128 | f64 | 2– | sumfact | 0.12–0.72 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Quadrilateral | CpuSimd256 | f32 | 2–2 | loops | – | as CpuSimd512 |
| Points | Quadrilateral | CpuSimd256 | f32 | 3– | sumfact | – | as CpuSimd512 |
| Points | Quadrilateral | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| Points | Quadrilateral | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| Points | Quadrilateral | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | CpuSimd512 | f32 | 2–2 | loops | 0.69 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | CpuSimd512 | f32 | 3– | sumfact | 0.17–0.49 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | CpuSimd512 | f64 | 3– | sumfact | 0.20–0.65 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | Gpu | f32 | 4– | shared | 0.54–0.83 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | Gpu | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Points | Quadrilateral | Gpu | f64 | 3–3 | sumfact | 0.73–0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Quadrilateral | Gpu | f64 | 4– | shared | 0.17–0.74 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | CpuSimd128 | f32 | 2– | loops | 0.74–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | CpuSimd128 | f64 | 2– | loops | 0.83–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd256 | f32 | 2– | loops | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd256 | f64 | 1–1 | direct | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd256 | f64 | 2–3 | loops | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd256 | f64 | 4–4 | direct | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd256 | f64 | 5– | loops | – | as CpuSimd512 |
| Points | Tetrahedron | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | CpuSimd512 | f32 | 2– | loops | 0.52–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | CpuSimd512 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | CpuSimd512 | f64 | 2–3 | loops | 0.73–0.76 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | CpuSimd512 | f64 | 4–4 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | CpuSimd512 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| Points | Tetrahedron | Gpu | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | Gpu | f32 | 5–6 | loops | 1.00 | direct not selectable at p = 5, 6; no strategy 10% faster |
| Points | Tetrahedron | Gpu | f32 | 7– | cooperative | 0.22–0.80 | direct not selectable at p = 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Tetrahedron | Gpu | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Points | Tetrahedron | Gpu | f64 | 5–5 | loops | 1.00 | direct not selectable at p = 5; no strategy 10% faster |
| Points | Tetrahedron | Gpu | f64 | 6– | cooperative | 0.08–0.47 | direct not selectable at p = 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | CpuSimd128 | f32 | 1– | sumfact | 0.02–0.74 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | CpuSimd128 | f64 | 1– | sumfact | 0.02–0.81 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Hexahedron | CpuSimd256 | f32 | 2– | sumfact | – | as CpuSimd512 |
| Points | Hexahedron | CpuSimd256 | f64 | 1–1 | direct | – | as CpuSimd512 |
| Points | Hexahedron | CpuSimd256 | f64 | 2– | sumfact | – | as CpuSimd512 |
| Points | Hexahedron | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Hexahedron | CpuSimd512 | f32 | 2– | sumfact | 0.02–0.37 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | CpuSimd512 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Hexahedron | CpuSimd512 | f64 | 2– | sumfact | 0.03–0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Points | Hexahedron | Gpu | f32 | 3–3 | shared | 0.48–0.51 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f32 | 4–4 | sumfact | 0.12 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f32 | 5–5 | shared | 0.05–0.06 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f32 | 6– | sumfact | 0.02–0.05 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Hexahedron | Gpu | f64 | 2–2 | shared | 0.74–0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Hexahedron | Gpu | f64 | 3– | sumfact | 0.06–0.41 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Prism | CpuSimd128 | f32 | 2– | sumfact | 0.12–0.46 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | CpuSimd128 | f64 | 1– | sumfact | 0.12–0.55 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Prism | CpuSimd256 | f32 | 2– | sumfact | – | as CpuSimd512 |
| Points | Prism | CpuSimd256 | f64 | 1–1 | direct | – | as CpuSimd512 |
| Points | Prism | CpuSimd256 | f64 | 2–2 | loops | – | as CpuSimd512 |
| Points | Prism | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| Points | Prism | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Prism | CpuSimd512 | f32 | 2– | sumfact | 0.10–0.50 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | CpuSimd512 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Prism | CpuSimd512 | f64 | 2–2 | loops | 0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | CpuSimd512 | f64 | 3– | sumfact | 0.16–0.41 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Prism | Gpu | f32 | 4–4 | sumfact | 0.76–0.79 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | Gpu | f32 | 5–7 | shared | 0.11–0.37 | at least 10% faster than the baseline; closest to the fastest at p = 6, 7 |
| Points | Prism | Gpu | f32 | 8– | sumfact | 0.08–0.10 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | Gpu | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Prism | Gpu | f64 | 2–4 | shared | 0.06–0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | Gpu | f64 | 5–5 | sumfact | 0.06–0.08 | at least 10% faster than the baseline; closest to the fastest at p = 5 |
| Points | Prism | Gpu | f64 | 6–7 | shared | 0.05–0.06 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Prism | Gpu | f64 | 8– | sumfact | 0.04–0.05 | at least 10% faster than the baseline; closest to the fastest at p = 8 |
| Points | Pyramid | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | CpuSimd128 | f32 | 2– | loops | 0.62–0.76 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Pyramid | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | CpuSimd128 | f64 | 3– | loops | 0.79 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Pyramid | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Points | Pyramid | CpuSimd256 | f32 | 2– | loops | – | as CpuSimd512 |
| Points | Pyramid | CpuSimd256 | f64 | 1–1 | direct | – | as CpuSimd512 |
| Points | Pyramid | CpuSimd256 | f64 | 2– | loops | – | as CpuSimd512 |
| Points | Pyramid | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | CpuSimd512 | f32 | 2– | loops | 0.52–0.66 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Pyramid | CpuSimd512 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | CpuSimd512 | f64 | 2– | loops | 0.72–0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| Points | Pyramid | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | Gpu | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Points | Pyramid | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Points | Pyramid | Gpu | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Jacobian | Interval | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Interval | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd256 | f64 | 1–5 | direct | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd256 | f64 | 6–6 | loops | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd256 | f64 | 7–8 | direct | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| Jacobian | Interval | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Interval | CpuSimd512 | f64 | 1–5 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | CpuSimd512 | f64 | 6–6 | loops | 0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Interval | CpuSimd512 | f64 | 7–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Triangle | CpuSimd128 | f32 | 1–6 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd128 | f32 | 7–7 | loops | 0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Triangle | CpuSimd128 | f32 | 8–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Triangle | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd128 | f64 | 5–5 | loops | 0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Triangle | CpuSimd128 | f64 | 6–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Triangle | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| Jacobian | Triangle | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| Jacobian | Triangle | CpuSimd256 | f64 | 1–6 | direct | – | as CpuSimd512 |
| Jacobian | Triangle | CpuSimd256 | f64 | 7– | loops | – | as CpuSimd512 |
| Jacobian | Triangle | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Triangle | CpuSimd512 | f64 | 1–6 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | CpuSimd512 | f64 | 7– | loops | 0.85–0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Triangle | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Triangle | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Triangle | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 3– | sumfact | 0.24–0.62 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 3– | sumfact | 0.19–0.77 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Jacobian | Quadrilateral | CpuSimd256 | f32 | 2– | sumfact | – | as CpuSimd512 |
| Jacobian | Quadrilateral | CpuSimd256 | f64 | 1–3 | direct | – | as CpuSimd512 |
| Jacobian | Quadrilateral | CpuSimd256 | f64 | 4– | sumfact | – | as CpuSimd512 |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 2– | sumfact | 0.40–0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 4– | sumfact | 0.38–0.75 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | Gpu | f32 | 4– | shared | 0.19–0.78 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Quadrilateral | Gpu | f64 | 4–5 | sumfact | 0.48–0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Quadrilateral | Gpu | f64 | 6– | shared | 0.14–0.51 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 4– | loops | 0.87–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 3– | loops | 0.87–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Tetrahedron | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| Jacobian | Tetrahedron | CpuSimd256 | f32 | 5– | loops | – | as CpuSimd512 |
| Jacobian | Tetrahedron | CpuSimd256 | f64 | 1–4 | direct | – | as CpuSimd512 |
| Jacobian | Tetrahedron | CpuSimd256 | f64 | 5– | loops | – | as CpuSimd512 |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| Jacobian | Tetrahedron | Gpu | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | Gpu | f32 | 5–7 | loops | 1.00 | direct not selectable at p = 5, 6, 7; no strategy 10% faster |
| Jacobian | Tetrahedron | Gpu | f32 | 8– | cooperative | 0.27–0.78 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Tetrahedron | Gpu | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Tetrahedron | Gpu | f64 | 5–5 | loops | 1.00 | direct not selectable at p = 5; no strategy 10% faster |
| Jacobian | Tetrahedron | Gpu | f64 | 6– | cooperative | 0.10–0.73 | direct not selectable at p = 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 2– | sumfact | 0.02–0.51 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 2– | sumfact | 0.02–0.47 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| Jacobian | Hexahedron | CpuSimd256 | f32 | 3– | sumfact | – | as CpuSimd512 |
| Jacobian | Hexahedron | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| Jacobian | Hexahedron | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 3– | sumfact | 0.01–0.75 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 3– | sumfact | 0.02–0.49 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | Gpu | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | Gpu | f32 | 2–5 | shared | 0.03–0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | Gpu | f32 | 6– | sumfact | 0.02–0.06 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | Gpu | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Hexahedron | Gpu | f64 | 3–4 | sumfact | 0.21–0.29 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | Gpu | f64 | 5–5 | shared | 0.25–0.31 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Hexahedron | Gpu | f64 | 6– | sumfact | 0.04–0.15 | direct not selectable at p = 8; at least 10% faster than the baseline; closest to the fastest at p = 7 |
| Jacobian | Prism | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | CpuSimd128 | f32 | 2– | sumfact | 0.13–0.70 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | CpuSimd128 | f64 | 2– | sumfact | 0.14–0.60 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| Jacobian | Prism | CpuSimd256 | f32 | 3– | sumfact | – | as CpuSimd512 |
| Jacobian | Prism | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| Jacobian | Prism | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| Jacobian | Prism | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | CpuSimd512 | f32 | 3– | sumfact | 0.03–0.70 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | CpuSimd512 | f64 | 3– | sumfact | 0.05–0.74 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | Gpu | f32 | 2–2 | shared | 0.88–0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 3–3 | sumfact | 0.71 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 4–4 | shared | 0.22–0.22 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 5–5 | sumfact | 0.14–0.15 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 6–7 | shared | 0.05–0.08 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f32 | 8– | sumfact | 0.02 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Prism | Gpu | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Prism | Gpu | f64 | 2– | shared | 0.01–0.80 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Pyramid | CpuSimd128 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | CpuSimd128 | f32 | 3– | loops | 0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Pyramid | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | CpuSimd128 | f64 | 3– | loops | 0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| Jacobian | Pyramid | CpuSimd256 | f32 | 1–3 | direct | – | as CpuSimd512 |
| Jacobian | Pyramid | CpuSimd256 | f32 | 4– | loops | – | as CpuSimd512 |
| Jacobian | Pyramid | CpuSimd256 | f64 | 1–3 | direct | – | as CpuSimd512 |
| Jacobian | Pyramid | CpuSimd256 | f64 | 4– | loops | – | as CpuSimd512 |
| Jacobian | Pyramid | CpuSimd512 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | CpuSimd512 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Jacobian | Pyramid | CpuSimd512 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | CpuSimd512 | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Jacobian | Pyramid | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | Gpu | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Jacobian | Pyramid | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Jacobian | Pyramid | Gpu | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| Normals | Interval | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Interval | CpuSimd256 | f32 | 1–6 | direct | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f32 | 7–7 | loops | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f32 | 8–8 | direct | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f64 | 1–6 | direct | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f64 | 7–7 | loops | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f64 | 8–8 | direct | – | as CpuSimd512 |
| Normals | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| Normals | Interval | CpuSimd512 | f32 | 1–6 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd512 | f32 | 7–7 | loops | 0.79 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Interval | CpuSimd512 | f32 | 8–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Interval | CpuSimd512 | f64 | 1–6 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd512 | f64 | 7–7 | loops | 0.82 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Interval | CpuSimd512 | f64 | 8–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Triangle | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd128 | f32 | 5–5 | loops | 0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Triangle | CpuSimd128 | f32 | 6–6 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd128 | f32 | 7– | loops | 0.87–0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Triangle | CpuSimd128 | f64 | 1–5 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd128 | f64 | 6– | loops | 0.81–0.85 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Triangle | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd256 | f64 | 1–4 | direct | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd256 | f64 | 5–5 | loops | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd256 | f64 | 6–6 | direct | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd256 | f64 | 7– | loops | – | as CpuSimd512 |
| Normals | Triangle | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Triangle | CpuSimd512 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd512 | f64 | 5–5 | loops | 0.83 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Triangle | CpuSimd512 | f64 | 6–6 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | CpuSimd512 | f64 | 7– | loops | 0.78–0.80 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Triangle | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Triangle | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| Normals | Triangle | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| Normals | Quadrilateral | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | CpuSimd128 | f32 | 2– | sumfact | 0.24–0.84 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Quadrilateral | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | CpuSimd128 | f64 | 2– | sumfact | 0.19–0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Quadrilateral | CpuSimd256 | f32 | 1–1 | direct | – | as CpuSimd512 |
| Normals | Quadrilateral | CpuSimd256 | f32 | 2– | sumfact | – | as CpuSimd512 |
| Normals | Quadrilateral | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| Normals | Quadrilateral | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| Normals | Quadrilateral | CpuSimd512 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | CpuSimd512 | f32 | 2– | sumfact | 0.51–0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Quadrilateral | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | CpuSimd512 | f64 | 3– | sumfact | 0.30–0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Quadrilateral | Gpu | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | Gpu | f32 | 5– | shared | 0.13–0.85 | at least 10% faster than the baseline; within 3% of the fastest |
| Normals | Quadrilateral | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| Normals | Quadrilateral | Gpu | f64 | 4–7 | shared | 0.04–0.77 | at least 10% faster than the baseline; closest to the fastest at p = 7 |
| Normals | Quadrilateral | Gpu | f64 | 8– | sumfact | 0.04–0.07 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Interval | CpuSimd128 | f32 | 1–6 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd128 | f32 | 7–7 | loops | 0.79 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Interval | CpuSimd128 | f32 | 8–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Interval | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f32 | 5–5 | loops | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f32 | 6–7 | direct | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f32 | 8– | loops | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f64 | 1–5 | direct | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f64 | 6–6 | loops | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f64 | 7–8 | direct | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| FacetNormals | Interval | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd512 | f32 | 5–5 | loops | 0.84 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Interval | CpuSimd512 | f32 | 6–7 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd512 | f32 | 8– | loops | 0.78 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Interval | CpuSimd512 | f64 | 1–5 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd512 | f64 | 6–6 | loops | 0.80 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Interval | CpuSimd512 | f64 | 7–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | CpuSimd128 | f32 | 1–6 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd128 | f32 | 7–7 | loops | 0.87 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Triangle | CpuSimd128 | f32 | 8–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd256 | f64 | 1–6 | direct | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd256 | f64 | 7–7 | loops | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd256 | f64 | 8–8 | direct | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| FacetNormals | Triangle | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | CpuSimd512 | f64 | 1–6 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd512 | f64 | 7–7 | loops | 0.80 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Triangle | CpuSimd512 | f64 | 8–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Triangle | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Triangle | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 2– | sumfact | 0.18–0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 3– | sumfact | 0.21–0.75 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Quadrilateral | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| FacetNormals | Quadrilateral | CpuSimd256 | f32 | 3– | sumfact | – | as CpuSimd512 |
| FacetNormals | Quadrilateral | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| FacetNormals | Quadrilateral | CpuSimd256 | f64 | 3– | sumfact | – | as CpuSimd512 |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 3– | sumfact | 0.42–0.68 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 3– | sumfact | 0.39–0.88 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Quadrilateral | Gpu | f32 | 1–5 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | Gpu | f32 | 6– | shared | 0.32–0.68 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Quadrilateral | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Quadrilateral | Gpu | f64 | 4– | shared | 0.41–0.90 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| FacetNormals | Tetrahedron | CpuSimd256 | f32 | 5– | loops | – | as CpuSimd512 |
| FacetNormals | Tetrahedron | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| FacetNormals | Tetrahedron | CpuSimd256 | f64 | 3– | loops | – | as CpuSimd512 |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 3– | loops | 0.63–1.00 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Tetrahedron | Gpu | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | Gpu | f32 | 5–7 | loops | 1.00 | direct not selectable at p = 5, 6, 7; no strategy 10% faster |
| FacetNormals | Tetrahedron | Gpu | f32 | 8– | cooperative | 0.49–0.73 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Tetrahedron | Gpu | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Tetrahedron | Gpu | f64 | 5–5 | shared | 0.57–0.90 | direct not selectable at p = 5; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Tetrahedron | Gpu | f64 | 6– | loops | 1.00 | direct not selectable at p = 6, 7, 8; no strategy 10% faster |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 2– | sumfact | 0.02–0.54 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 1– | sumfact | 0.03–0.85 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | CpuSimd256 | f32 | 1– | sumfact | – | as CpuSimd512 |
| FacetNormals | Hexahedron | CpuSimd256 | f64 | 1–1 | direct | – | as CpuSimd512 |
| FacetNormals | Hexahedron | CpuSimd256 | f64 | 2– | sumfact | – | as CpuSimd512 |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 1– | sumfact | 0.01–0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 2– | sumfact | 0.02–0.67 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f32 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Hexahedron | Gpu | f32 | 2–2 | shared | 0.77–0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f32 | 3–3 | sumfact | 0.17–0.23 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f32 | 4–5 | shared | 0.04–0.10 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f32 | 6– | sumfact | 0.03–0.05 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Hexahedron | Gpu | f64 | 3–4 | sumfact | 0.30–0.89 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Hexahedron | Gpu | f64 | 5–5 | shared | 0.42–0.66 | at least 10% faster than the baseline; closest to the fastest at p = 5 |
| FacetNormals | Hexahedron | Gpu | f64 | 6– | sumfact | 0.05–0.21 | direct not selectable at p = 8; at least 10% faster than the baseline; closest to the fastest at p = 7 |
| FacetNormals | Prism | CpuSimd128 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | CpuSimd128 | f32 | 3– | sumfact | 0.52–0.85 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd128 | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | CpuSimd128 | f64 | 2– | sumfact | 0.56–0.90 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd256 | f32 | 1–3 | direct | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f32 | 4–4 | sumfact | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f32 | 5–6 | loops | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f32 | 7– | sumfact | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f64 | 3–3 | loops | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f64 | 4–4 | sumfact | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f64 | 5–7 | loops | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd256 | f64 | 8– | sumfact | – | as CpuSimd512 |
| FacetNormals | Prism | CpuSimd512 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | CpuSimd512 | f32 | 4–4 | sumfact | 0.58 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd512 | f32 | 5–6 | loops | 1.00 | direct not selectable at p = 5, 6; no strategy 10% faster |
| FacetNormals | Prism | CpuSimd512 | f32 | 7– | sumfact | 0.12–0.86 | direct not selectable at p = 7; at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | CpuSimd512 | f64 | 3–3 | loops | 0.84 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd512 | f64 | 4–4 | sumfact | 0.54 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | CpuSimd512 | f64 | 5–7 | loops | 1.00 | direct not selectable at p = 5, 6, 7; no strategy 10% faster |
| FacetNormals | Prism | CpuSimd512 | f64 | 8– | sumfact | 0.13 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | Gpu | f32 | 4–5 | shared | 0.35–0.53 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | Gpu | f32 | 6– | sumfact | 0.08–0.31 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Prism | Gpu | f64 | 1–1 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Prism | Gpu | f64 | 2– | shared | 0.02–0.87 | at least 10% faster than the baseline; closest to the fastest at p = 4 |
| FacetNormals | Pyramid | CpuSimd128 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | CpuSimd128 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| FacetNormals | Pyramid | CpuSimd128 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | CpuSimd128 | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| FacetNormals | Pyramid | CpuSimd256 | f32 | 1–3 | direct | – | as CpuSimd512 |
| FacetNormals | Pyramid | CpuSimd256 | f32 | 4– | loops | – | as CpuSimd512 |
| FacetNormals | Pyramid | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| FacetNormals | Pyramid | CpuSimd256 | f64 | 3– | loops | – | as CpuSimd512 |
| FacetNormals | Pyramid | CpuSimd512 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | CpuSimd512 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| FacetNormals | Pyramid | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | CpuSimd512 | f64 | 3– | loops | 0.85 | at least 10% faster than the baseline; within 3% of the fastest |
| FacetNormals | Pyramid | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | Gpu | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| FacetNormals | Pyramid | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| FacetNormals | Pyramid | Gpu | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Interval | CpuSimd128 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | CpuSimd128 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Interval | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Interval | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Interval | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| PullBack | Interval | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Interval | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| PullBack | Interval | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Interval | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Interval | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Interval | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Interval | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Triangle | CpuSimd128 | f32 | 1–7 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | CpuSimd128 | f32 | 8– | loops | 0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Triangle | CpuSimd128 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | CpuSimd128 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Triangle | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Triangle | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| PullBack | Triangle | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Triangle | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| PullBack | Triangle | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Triangle | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Triangle | Gpu | f32 | 1–7 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | Gpu | f32 | 8– | loops | 1.00 | direct not selectable at p = 8; no strategy 10% faster |
| PullBack | Triangle | Gpu | f64 | 1–7 | direct | 1.00 | no strategy 10% faster |
| PullBack | Triangle | Gpu | f64 | 8– | loops | 1.00 | direct not selectable at p = 8; no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 5– | sumfact | 0.73–0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 5–5 | sumfact | 0.81 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 6–6 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 7– | sumfact | 0.82–0.86 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Quadrilateral | CpuSimd256 | f32 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Quadrilateral | CpuSimd256 | f32 | 9– | loops | – | as CpuSimd512 |
| PullBack | Quadrilateral | CpuSimd256 | f64 | 1–8 | direct | – | as CpuSimd512 |
| PullBack | Quadrilateral | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Quadrilateral | Gpu | f32 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | Gpu | f32 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Quadrilateral | Gpu | f64 | 1–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Quadrilateral | Gpu | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| PullBack | Tetrahedron | CpuSimd256 | f32 | 5– | loops | – | as CpuSimd512 |
| PullBack | Tetrahedron | CpuSimd256 | f64 | 1–4 | direct | – | as CpuSimd512 |
| PullBack | Tetrahedron | CpuSimd256 | f64 | 5– | loops | – | as CpuSimd512 |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 5– | loops | 1.00 | direct not selectable at p = 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Tetrahedron | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | Gpu | f32 | 4– | loops | 1.00 | direct not selectable at p = 4, 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Tetrahedron | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Tetrahedron | Gpu | f64 | 4– | loops | 1.00 | direct not selectable at p = 4, 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd128 | f32 | 5–6 | sumfact | 0.50–0.69 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd128 | f32 | 7–7 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd128 | f32 | 8– | sumfact | 0.60 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd128 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd128 | f64 | 3–3 | sumfact | 0.66 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd128 | f64 | 4–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd128 | f64 | 5– | sumfact | 0.50–0.78 | direct not selectable at p = 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd256 | f32 | 1–2 | direct | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd256 | f32 | 3– | sumfact | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd256 | f64 | 1–2 | direct | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd256 | f64 | 3–7 | sumfact | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd256 | f64 | 8–8 | direct | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd256 | f64 | 9– | loops | – | as CpuSimd512 |
| PullBack | Hexahedron | CpuSimd512 | f32 | 1–2 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd512 | f32 | 3– | sumfact | 0.40–0.64 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd512 | f64 | 1–2 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd512 | f64 | 3–7 | sumfact | 0.42–0.70 | at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Hexahedron | CpuSimd512 | f64 | 8–8 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | CpuSimd512 | f64 | 9– | loops | – | beyond p = 8: loops compiles at every degree |
| PullBack | Hexahedron | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | Gpu | f32 | 4– | loops | 1.00 | direct not selectable at p = 4, 5, 6, 7, 8; no strategy 10% faster |
| PullBack | Hexahedron | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Hexahedron | Gpu | f64 | 4–4 | loops | 1.00 | direct not selectable at p = 4; no strategy 10% faster |
| PullBack | Hexahedron | Gpu | f64 | 5– | sumfact | 0.72–0.88 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | CpuSimd128 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | CpuSimd128 | f32 | 5– | sumfact | 0.38–0.45 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | CpuSimd128 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | CpuSimd128 | f64 | 5– | sumfact | 0.38–0.45 | direct not selectable at p = 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | CpuSimd256 | f32 | 1–4 | direct | – | as CpuSimd512 |
| PullBack | Prism | CpuSimd256 | f32 | 5– | sumfact | – | as CpuSimd512 |
| PullBack | Prism | CpuSimd256 | f64 | 1–4 | direct | – | as CpuSimd512 |
| PullBack | Prism | CpuSimd256 | f64 | 5– | sumfact | – | as CpuSimd512 |
| PullBack | Prism | CpuSimd512 | f32 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | CpuSimd512 | f32 | 5– | sumfact | 0.35–0.75 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | CpuSimd512 | f64 | 1–4 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | CpuSimd512 | f64 | 5– | sumfact | 0.33–0.66 | direct not selectable at p = 5, 6, 7; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | Gpu | f32 | 4–4 | loops | 1.00 | direct not selectable at p = 4; no strategy 10% faster |
| PullBack | Prism | Gpu | f32 | 5–6 | sumfact | 0.64–0.75 | direct not selectable at p = 5, 6; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | Gpu | f32 | 7–7 | loops | 1.00 | direct not selectable at p = 7; no strategy 10% faster |
| PullBack | Prism | Gpu | f32 | 8– | sumfact | 0.66–0.86 | direct not selectable at p = 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Prism | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Prism | Gpu | f64 | 4– | sumfact | 0.46–0.72 | direct not selectable at p = 4, 5, 6, 7, 8; at least 10% faster than the baseline; within 3% of the fastest |
| PullBack | Pyramid | CpuSimd128 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | CpuSimd128 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Pyramid | CpuSimd128 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | CpuSimd128 | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Pyramid | CpuSimd256 | f32 | 1–3 | direct | – | as CpuSimd512 |
| PullBack | Pyramid | CpuSimd256 | f32 | 4– | loops | – | as CpuSimd512 |
| PullBack | Pyramid | CpuSimd256 | f64 | 1–3 | direct | – | as CpuSimd512 |
| PullBack | Pyramid | CpuSimd256 | f64 | 4– | loops | – | as CpuSimd512 |
| PullBack | Pyramid | CpuSimd512 | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | CpuSimd512 | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Pyramid | CpuSimd512 | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | CpuSimd512 | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Pyramid | Gpu | f32 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | Gpu | f32 | 4– | loops | – | beyond p = 3: loops compiles at every degree |
| PullBack | Pyramid | Gpu | f64 | 1–3 | direct | 1.00 | no strategy 10% faster |
| PullBack | Pyramid | Gpu | f64 | 4– | loops | – | beyond p = 3: loops compiles at every degree |

### Cost per strategy

Per measured class, policy family, cell type, scalar type and degree: the baseline's class time in ns per point and each selectable strategy's class ratio to it, at the rule sizes (10^6 / 10^7 points on Gpu, 10^6 on the CPU classes, ADR-0052); "–" where a strategy is not selectable (not measured on every device, not a candidate of the class, ran its fallback `loops`, or on Gpu a failed launch or a compile over 30 s on Locust). "n.m.; T3": `direct` not measured on tetrahedra, hexahedra and prisms at p ≥ 5 (ADR-0050), with its cost from T3's baseline (a88ab2a) over this run's baseline. Jacobian sums J, det J and K; PullBack is the pull-back at 9 Newton iterations.

| Family | Cell | Class | Type | p | Base | ns per point | direct | loops | sumfact | shared | cooperative | Choice | Reason |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EVector | Interval | CpuSimd128 | f32 | 1 | direct | 0.11 | 1.00 | 1.12 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 2 | direct | 0.11 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 3 | direct | 0.12 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 4 | direct | 0.11 | 1.00 | 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 5 | direct | 0.10 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 6 | direct | 0.11 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 7 | direct | 0.11 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f32 | 8 | direct | 0.12 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 1 | direct | 0.14 | 1.00 | 1.17 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 2 | direct | 0.16 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 3 | direct | 0.16 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 4 | direct | 0.17 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 5 | direct | 0.18 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 6 | direct | 0.18 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 7 | direct | 0.18 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd128 | f64 | 8 | direct | 0.18 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 1 | direct | 0.45 | 1.00 | 1.50 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 2 | direct | 0.16 | 1.00 | 2.62 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 3 | direct | 0.13 | 1.00 | 2.45 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 4 | direct | 0.13 | 1.00 | 2.25 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 5 | direct | 0.17 | 1.00 | 2.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 6 | direct | 0.16 | 1.00 | 2.32 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 7 | direct | 0.17 | 1.00 | 2.27 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f32 | 8 | direct | 0.19 | 1.00 | 2.07 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 1 | direct | 0.48 | 1.00 | 1.82 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 2 | direct | 0.19 | 1.00 | 2.72 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 3 | direct | 0.15 | 1.00 | 2.13 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 4 | direct | 0.17 | 1.00 | 1.89 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 5 | direct | 0.16 | 1.00 | 1.79 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 6 | direct | 0.19 | 1.00 | 1.30 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 7 | direct | 0.16 | 1.00 | 1.35 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | CpuSimd512 | f64 | 8 | direct | 0.16 | 1.00 | 1.45 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 1 | direct | 0.08 / 0.05 | 1.00 / 1.00 | 1.06 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 2 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 1.05 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 3 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.06 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 4 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.10 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 5 | direct | 0.10 / 0.02 | 1.00 / 1.00 | 1.02 / 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 6 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.07 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 7 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.02 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f32 | 8 | direct | 0.10 / 0.02 | 1.00 / 1.00 | 0.99 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 1 | direct | 0.02 / 0.02 | 1.00 / 1.00 | 1.15 / 0.75 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 2 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 1.07 / 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 3 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.13 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 4 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.20 / 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 5 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.17 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 6 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.06 / 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 7 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 0.99 / 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Interval | Gpu | f64 | 8 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.01 / 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 1 | direct | 0.12 | 1.00 | 1.12 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 2 | direct | 0.11 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 3 | direct | 0.18 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 4 | direct | 0.21 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 5 | direct | 0.25 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 6 | direct | 0.28 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 7 | direct | 0.32 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f32 | 8 | direct | 0.32 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 1 | direct | 0.16 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 2 | direct | 0.16 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 3 | direct | 0.16 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 4 | direct | 0.18 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 5 | direct | 0.19 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 6 | direct | 0.20 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 7 | direct | 0.20 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd128 | f64 | 8 | direct | 0.18 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 1 | direct | 0.15 | 1.00 | 1.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 2 | direct | 0.18 | 1.00 | 1.50 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 3 | direct | 0.28 | 1.00 | 1.19 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 4 | direct | 0.38 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 5 | direct | 0.70 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 6 | direct | 1.09 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| EVector | Triangle | CpuSimd512 | f32 | 7 | direct | 1.96 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f32 | 8 | direct | 1.27 | 1.00 | 0.75 | – | – | – | loops | loops 0.75 of direct |
| EVector | Triangle | CpuSimd512 | f64 | 1 | direct | 0.14 | 1.00 | 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 2 | direct | 0.13 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 3 | direct | 0.20 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 4 | direct | 0.30 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 5 | direct | 0.59 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 6 | direct | 0.86 | 1.00 | 0.82 | – | – | – | loops | loops 0.82 of direct |
| EVector | Triangle | CpuSimd512 | f64 | 7 | direct | 2.01 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | CpuSimd512 | f64 | 8 | direct | 1.19 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| EVector | Triangle | Gpu | f32 | 1 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 1.09 / 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 2 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.05 / 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 3 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 1.05 / 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 4 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.01 / 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 5 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 0.97 / 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 6 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.03 / 1.22 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 7 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.08 / 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f32 | 8 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.04 / 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.01 / 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 2 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.00 / 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.02 / 1.19 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 4 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.12 / 1.61 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 5 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 0.92 / 1.48 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 6 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.14 / 1.59 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 7 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 0.96 / 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Triangle | Gpu | f64 | 8 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.13 / 1.27 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 0.16 | 1.00 | 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 0.17 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 0.25 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 0.31 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 0.35 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 0.43 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 0.52 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 0.56 | 1.00 | 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 0.21 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 0.23 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 0.24 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 0.27 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 0.34 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 0.29 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 0.30 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 0.29 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 0.15 | 1.00 | 1.81 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 0.29 | 1.00 | 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 0.64 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 0.87 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 3.29 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 1.99 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 2.39 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 2.32 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 0.21 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 0.25 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 0.73 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 1.09 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 2.58 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 2.07 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 2.66 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 3.78 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 1 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.04 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 2 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.04 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 3 | direct | 0.11 / 0.03 | 1.00 / 1.00 | 0.99 / 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 4 | direct | 0.13 / 0.05 | 1.00 / 1.00 | 1.02 / 1.16 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 5 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 1.04 / 1.48 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 6 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 1.05 / 1.47 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 7 | direct | 0.14 / 0.05 | 1.00 / 1.00 | 1.04 / 1.15 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f32 | 8 | direct | 0.15 / 0.05 | 1.00 / 1.00 | 1.03 / 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 1 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 0.97 / 0.74 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.17 / 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.02 / 2.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 4 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 0.98 / 1.82 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 5 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 0.95 / 1.56 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 6 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 1.09 / 2.07 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 7 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 1.15 / 1.62 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Quadrilateral | Gpu | f64 | 8 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.11 / 1.41 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f32 | 1 | direct | 0.10 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f32 | 2 | direct | 0.20 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f32 | 3 | direct | 0.27 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f32 | 4 | direct | 0.25 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f32 | 5 | loops | 0.25 | n.m.; T3: 1.00 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f32 | 6 | loops | 0.24 | n.m.; T3: 0.94 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f32 | 7 | loops | 0.23 | n.m.; T3: 0.95 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f32 | 8 | loops | 0.23 | n.m.; T3: 0.94 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f64 | 1 | direct | 0.15 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f64 | 2 | direct | 0.11 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f64 | 3 | direct | 0.13 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f64 | 4 | direct | 0.13 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd128 | f64 | 5 | loops | 0.20 | n.m.; T3: 0.95 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f64 | 6 | loops | 0.40 | n.m.; T3: 0.99 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f64 | 7 | loops | 0.22 | n.m.; T3: 1.09 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd128 | f64 | 8 | loops | 0.23 | n.m.; T3: 0.94 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f32 | 1 | direct | 0.15 | 1.00 | 1.61 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f32 | 2 | direct | 0.39 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f32 | 3 | direct | 0.85 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f32 | 4 | direct | 0.80 | 1.00 | 0.89 | – | – | – | loops | loops 0.89 of direct |
| EVector | Tetrahedron | CpuSimd512 | f32 | 5 | loops | 0.63 | n.m.; T3: 1.24 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f32 | 6 | loops | 2.81 | n.m.; T3: 0.98 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f32 | 7 | loops | 0.58 | n.m.; T3: 1.26 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f32 | 8 | loops | 0.86 | n.m.; T3: 1.22 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f64 | 1 | direct | 0.11 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f64 | 2 | direct | 0.25 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f64 | 3 | direct | 0.60 | 1.00 | 0.89 | – | – | – | loops | loops 0.89 of direct |
| EVector | Tetrahedron | CpuSimd512 | f64 | 4 | direct | 2.00 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | CpuSimd512 | f64 | 5 | loops | 1.58 | n.m.; T3: 1.05 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f64 | 6 | loops | 3.33 | n.m.; T3: 0.97 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f64 | 7 | loops | 1.34 | n.m.; T3: 1.04 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | CpuSimd512 | f64 | 8 | loops | 1.17 | n.m.; T3: 1.21 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f32 | 1 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 0.89 / 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f32 | 2 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 1.04 / 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f32 | 3 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 0.94 / 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f32 | 4 | direct | 0.10 / 0.02 | 1.00 / 1.00 | 0.94 / 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f32 | 5 | loops | 0.11 / 0.02 | n.m.; T3: 1.02 / 1.01 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f32 | 6 | loops | 0.11 / 0.02 | n.m.; T3: 1.00 / 0.91 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f32 | 7 | loops | 0.12 / 0.02 | n.m.; T3: 1.00 / 0.93 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f32 | 8 | loops | 0.16 / 0.03 | n.m.; T3: 0.97 / 0.91 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f64 | 1 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.16 / 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.04 / 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f64 | 3 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.09 / 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Tetrahedron | Gpu | f64 | 4 | direct | 0.04 / 0.01 | 1.00 / 1.00 | 0.87 / 0.87 | – | – | – | loops | loops 0.87 / 0.87 of direct |
| EVector | Tetrahedron | Gpu | f64 | 5 | loops | 0.04 / 0.01 | n.m.; T3: 0.90 / 0.84 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f64 | 6 | loops | 0.05 / 0.01 | n.m.; T3: 0.81 / 0.79 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f64 | 7 | loops | 0.07 / 0.01 | n.m.; T3: 0.81 / 0.74 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Tetrahedron | Gpu | f64 | 8 | loops | 0.09 / 0.01 | n.m.; T3: 0.83 / 0.67 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Hexahedron | CpuSimd128 | f32 | 1 | direct | 0.20 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 2 | direct | 0.39 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 3 | direct | 0.67 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 4 | direct | 0.72 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 5 | direct | 0.70 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 6 | direct | 0.81 | 1.00 | 0.82 | – | – | – | loops | loops 0.82 of direct |
| EVector | Hexahedron | CpuSimd128 | f32 | 7 | loops | 0.68 | – | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Hexahedron | CpuSimd128 | f32 | 8 | loops | 0.70 | – | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Hexahedron | CpuSimd128 | f64 | 1 | direct | 0.28 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 2 | direct | 0.35 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 3 | direct | 0.51 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 4 | direct | 0.65 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 5 | direct | 0.83 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 6 | direct | 1.04 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 7 | direct | 1.21 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd128 | f64 | 8 | loops | 1.30 | n.m.; T3: 1.08 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Hexahedron | CpuSimd512 | f32 | 1 | direct | 0.37 | 1.00 | 1.12 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 2 | direct | 1.57 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 3 | direct | 3.62 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 4 | direct | 6.78 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 5 | direct | 7.01 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 6 | direct | 7.37 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 7 | direct | 7.04 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f32 | 8 | direct | 7.14 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 1 | direct | 0.29 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 2 | direct | 1.66 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 3 | direct | 4.18 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 4 | direct | 12.0 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 5 | direct | 12.0 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 6 | direct | 11.6 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 7 | direct | 11.8 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | CpuSimd512 | f64 | 8 | direct | 19.0 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.00 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 2 | direct | 0.13 / 0.05 | 1.00 / 1.00 | 1.05 / 1.44 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 3 | direct | 0.14 / 0.06 | 1.00 / 1.00 | 1.03 / 1.26 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 4 | direct | 0.18 / 0.07 | 1.00 / 1.00 | 1.05 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 5 | direct | 0.20 / 0.07 | 1.00 / 1.00 | 0.98 / 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 6 | direct | 0.28 / 0.08 | 1.00 / 1.00 | 1.04 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 7 | direct | 0.53 / 0.10 | 1.00 / 1.00 | 0.92 / 0.85 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f32 | 8 | loops | 0.52 / 0.11 | n.m.; T3: 1.16 / 1.18 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Hexahedron | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.00 / 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 2 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 1.03 / 2.84 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 3 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.24 / 1.91 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 4 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.15 / 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 5 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.49 / 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 6 | direct | 0.14 / 0.04 | 1.00 / 1.00 | 1.51 / 1.24 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 7 | direct | 0.27 / 0.05 | 1.00 / 1.00 | 1.33 / 1.33 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Hexahedron | Gpu | f64 | 8 | loops | 0.47 / 0.07 | n.m.; T3: 0.83 / 0.79 | 1.00 / 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f32 | 1 | direct | 0.15 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f32 | 2 | direct | 0.30 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f32 | 3 | direct | 0.42 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f32 | 4 | direct | 0.83 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f32 | 5 | loops | 0.43 | n.m.; T3: 1.05 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f32 | 6 | loops | 0.42 | n.m.; T3: 1.08 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f32 | 7 | loops | 0.41 | n.m.; T3: 1.12 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f32 | 8 | loops | 0.40 | n.m.; T3: 1.35 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f64 | 1 | direct | 0.33 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f64 | 2 | direct | 0.22 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f64 | 3 | direct | 0.26 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f64 | 4 | direct | 0.48 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd128 | f64 | 5 | loops | 0.32 | n.m.; T3: 0.98 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f64 | 6 | loops | 0.38 | n.m.; T3: 1.01 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f64 | 7 | loops | 0.46 | n.m.; T3: 1.03 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd128 | f64 | 8 | loops | 0.51 | n.m.; T3: 0.98 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f32 | 1 | direct | 1.06 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f32 | 2 | direct | 0.84 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f32 | 3 | direct | 4.27 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f32 | 4 | direct | 8.17 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f32 | 5 | loops | 1.45 | n.m.; T3: 1.23 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f32 | 6 | loops | 2.36 | n.m.; T3: 1.06 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f32 | 7 | loops | 2.28 | n.m.; T3: 1.08 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f32 | 8 | direct | 3.20 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f64 | 1 | direct | 0.63 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f64 | 2 | direct | 0.87 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f64 | 3 | direct | 3.39 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f64 | 4 | direct | 10.2 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | CpuSimd512 | f64 | 5 | loops | 4.63 | n.m.; T3: 1.03 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f64 | 6 | loops | 4.26 | n.m.; T3: 1.01 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f64 | 7 | loops | 4.51 | n.m.; T3: 1.00 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| EVector | Prism | CpuSimd512 | f64 | 8 | direct | 6.42 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 1 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.05 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 2 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.01 / 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 3 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.03 / 1.22 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 4 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 1.08 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 5 | direct | 0.13 / 0.04 | 1.00 / 1.00 | 1.11 / 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 6 | direct | 0.16 / 0.04 | 1.00 / 1.00 | 1.08 / 1.14 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 7 | direct | 0.20 / 0.06 | 1.00 / 1.00 | 0.98 / 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f32 | 8 | direct | 0.27 / 0.07 | 1.00 / 1.00 | 0.94 / 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.06 / 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.00 / 1.83 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.17 / 1.79 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 4 | direct | 0.05 / 0.01 | 1.00 / 1.00 | 1.16 / 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 5 | direct | 0.06 / 0.01 | 1.00 / 1.00 | 1.43 / 1.32 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 6 | direct | 0.08 / 0.01 | 1.00 / 1.00 | 1.32 / 1.40 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 7 | direct | 0.14 / 0.02 | 1.00 / 1.00 | 1.18 / 1.55 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Prism | Gpu | f64 | 8 | direct | 0.19 / 0.02 | 1.00 / 1.00 | 1.12 / 1.45 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f32 | 1 | direct | 0.12 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f32 | 2 | direct | 0.26 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f32 | 3 | direct | 0.34 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f64 | 1 | direct | 0.18 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f64 | 2 | direct | 0.18 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd128 | f64 | 3 | direct | 0.18 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd512 | f32 | 1 | direct | 0.23 | 1.00 | 1.22 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd512 | f32 | 2 | direct | 0.68 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd512 | f32 | 3 | direct | 1.28 | 1.00 | 0.80 | – | – | – | loops | loops 0.80 of direct |
| EVector | Pyramid | CpuSimd512 | f64 | 1 | direct | 0.16 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd512 | f64 | 2 | direct | 0.48 | 1.00 | 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | CpuSimd512 | f64 | 3 | direct | 0.91 | 1.00 | 0.89 | – | – | – | loops | loops 0.89 of direct |
| EVector | Pyramid | Gpu | f32 | 1 | direct | 0.12 / 0.03 | 1.00 / 1.00 | 0.87 / 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | Gpu | f32 | 2 | direct | 0.09 / 0.02 | 1.00 / 1.00 | 1.04 / 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | Gpu | f32 | 3 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.04 / 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | Gpu | f64 | 1 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.01 / 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.09 / 1.29 | – | – | – | direct | no strategy 10% faster than direct |
| EVector | Pyramid | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.17 / 1.39 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 1 | direct | 0.04 | 1.00 | 1.22 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 2 | direct | 0.05 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 3 | direct | 0.05 | 1.00 | 1.17 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 4 | direct | 0.07 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 5 | direct | 0.07 | 1.00 | 1.24 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 6 | direct | 0.08 | 1.00 | 1.20 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 7 | direct | 0.08 | 1.00 | 1.35 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f32 | 8 | direct | 0.09 | 1.00 | 1.31 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 1 | direct | 0.07 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 2 | direct | 0.08 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 3 | direct | 0.09 | 1.00 | 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 4 | direct | 0.10 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 5 | direct | 0.11 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 6 | direct | 0.16 | 1.00 | 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 7 | direct | 0.16 | 1.00 | 1.16 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd128 | f64 | 8 | direct | 0.19 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 1 | direct | 0.07 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 2 | direct | 0.08 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 3 | direct | 0.10 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 4 | direct | 0.07 | 1.00 | 1.48 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 5 | direct | 0.08 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 6 | direct | 0.05 | 1.00 | 1.31 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 7 | direct | 0.07 | 1.00 | 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f32 | 8 | direct | 0.08 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 1 | direct | 0.10 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 2 | direct | 0.10 | 1.00 | 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 3 | direct | 0.11 | 1.00 | 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 4 | direct | 0.12 | 1.00 | 0.90 | – | – | – | loops | loops 0.90 of direct |
| Points | Interval | CpuSimd512 | f64 | 5 | direct | 0.13 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Points | Interval | CpuSimd512 | f64 | 6 | direct | 0.09 | 1.00 | 1.39 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 7 | direct | 0.11 | 1.00 | 1.15 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | CpuSimd512 | f64 | 8 | direct | 0.10 | 1.00 | 1.28 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 1 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.07 / 1.07 | – | 1.04 / 1.05 | 1.27 / 1.29 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 2 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.07 / 1.16 | – | 1.10 / 1.15 | 1.29 / 1.53 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 3 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.03 / 1.10 | – | 1.02 / 1.15 | 1.29 / 1.91 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 4 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 0.97 / 1.18 | – | 0.94 / 1.12 | 1.27 / 2.45 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 5 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.01 / 1.15 | – | 1.04 / 1.13 | 1.39 / 2.14 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 6 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.02 / 1.27 | – | 1.02 / 1.24 | 1.29 / 2.39 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 7 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.15 / 1.26 | – | 1.14 / 1.22 | 1.87 / 3.77 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f32 | 8 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 1.00 / 1.20 | – | 1.02 / 1.15 | 1.84 / 4.39 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 0.87 / 1.08 | – | 0.97 / 1.11 | 1.04 / 1.37 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 2 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.34 / 0.97 | – | 1.13 / 0.98 | 1.22 / 1.46 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 3 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.08 / 1.11 | – | 1.07 / 1.05 | 1.48 / 2.35 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 4 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.13 / 1.03 | – | 1.12 / 0.95 | 1.57 / 2.79 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 5 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.10 / 1.25 | – | 1.17 / 1.14 | 1.96 / 3.10 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 6 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.06 / 1.13 | – | 1.14 / 1.17 | 1.55 / 2.87 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 7 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.02 / 1.17 | – | 0.95 / 1.20 | 1.83 / 4.79 | direct | no strategy 10% faster than direct |
| Points | Interval | Gpu | f64 | 8 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.24 / 1.32 | – | 1.02 / 1.17 | 2.25 / 5.94 | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 1 | direct | 0.08 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 2 | direct | 0.12 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 3 | direct | 0.25 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 4 | direct | 0.33 | 1.00 | 0.81 | – | – | – | loops | loops 0.81 of direct |
| Points | Triangle | CpuSimd128 | f32 | 5 | direct | 0.54 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 6 | direct | 0.85 | 1.00 | 0.89 | – | – | – | loops | loops 0.89 of direct |
| Points | Triangle | CpuSimd128 | f32 | 7 | direct | 0.94 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f32 | 8 | direct | 1.29 | 1.00 | 0.73 | – | – | – | loops | loops 0.73 of direct |
| Points | Triangle | CpuSimd128 | f64 | 1 | direct | 0.18 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f64 | 2 | direct | 0.23 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f64 | 3 | direct | 0.42 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f64 | 4 | direct | 0.69 | 1.00 | 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd128 | f64 | 5 | direct | 1.34 | 1.00 | 0.86 | – | – | – | loops | loops 0.86 of direct |
| Points | Triangle | CpuSimd128 | f64 | 6 | direct | 2.18 | 1.00 | 0.88 | – | – | – | loops | loops 0.88 of direct |
| Points | Triangle | CpuSimd128 | f64 | 7 | direct | 3.65 | 1.00 | 0.86 | – | – | – | loops | loops 0.86 of direct |
| Points | Triangle | CpuSimd128 | f64 | 8 | direct | 5.64 | 1.00 | 0.71 | – | – | – | loops | loops 0.71 of direct |
| Points | Triangle | CpuSimd512 | f32 | 1 | direct | 0.10 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f32 | 2 | direct | 0.14 | 1.00 | 0.82 | – | – | – | loops | loops 0.82 of direct |
| Points | Triangle | CpuSimd512 | f32 | 3 | direct | 0.28 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| Points | Triangle | CpuSimd512 | f32 | 4 | direct | 0.63 | 1.00 | 0.71 | – | – | – | loops | loops 0.71 of direct |
| Points | Triangle | CpuSimd512 | f32 | 5 | direct | 1.35 | 1.00 | 0.60 | – | – | – | loops | loops 0.60 of direct |
| Points | Triangle | CpuSimd512 | f32 | 6 | direct | 2.69 | 1.00 | 0.70 | – | – | – | loops | loops 0.70 of direct |
| Points | Triangle | CpuSimd512 | f32 | 7 | direct | 4.17 | 1.00 | 0.66 | – | – | – | loops | loops 0.66 of direct |
| Points | Triangle | CpuSimd512 | f32 | 8 | direct | 4.88 | 1.00 | 0.61 | – | – | – | loops | loops 0.61 of direct |
| Points | Triangle | CpuSimd512 | f64 | 1 | direct | 0.13 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 2 | direct | 0.14 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 3 | direct | 0.18 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 4 | direct | 0.24 | 1.00 | 0.83 | – | – | – | loops | loops 0.83 of direct |
| Points | Triangle | CpuSimd512 | f64 | 5 | direct | 0.22 | 1.00 | 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 6 | direct | 0.45 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 7 | direct | 0.55 | 1.00 | 1.15 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Triangle | CpuSimd512 | f64 | 8 | direct | 0.64 | 1.00 | 0.88 | – | – | – | loops | loops 0.88 of direct |
| Points | Triangle | Gpu | f32 | 1 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.02 / 1.04 | – | 1.04 / 1.04 | 1.18 / 1.36 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 2 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.11 / 1.15 | – | 1.12 / 1.13 | 1.63 / 2.20 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 3 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.19 / 1.32 | – | 1.20 / 1.18 | 2.60 / 4.62 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 4 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.36 / 1.56 | – | 1.24 / 1.41 | 2.38 / 4.76 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 5 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.68 / 2.26 | – | 1.78 / 2.58 | 4.47 / 9.44 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 6 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 2.62 / 3.50 | – | 2.40 / 3.31 | 3.99 / 8.88 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 7 | direct | 0.13 / 0.05 | 1.00 / 1.00 | 3.21 / 4.20 | – | 2.93 / 3.86 | 6.64 / 15.51 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f32 | 8 | direct | 0.14 / 0.05 | 1.00 / 1.00 | 4.19 / 6.00 | – | 4.49 / 5.28 | 6.49 / 17.90 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.06 / 1.04 | – | 1.08 / 0.99 | 1.42 / 1.87 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.08 / 1.10 | – | 1.22 / 1.05 | 2.74 / 4.50 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.60 / 1.67 | – | 1.69 / 1.98 | 5.35 / 10.76 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 4 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.91 / 4.56 | – | 2.07 / 5.17 | 4.87 / 11.86 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 5 | direct | 0.04 / 0.01 | 1.00 / 1.00 | 2.22 / 7.74 | – | 2.40 / 8.27 | 7.84 / 20.10 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 6 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 3.52 / 7.88 | – | 5.75 / 8.66 | 7.38 / 19.46 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 7 | direct | 0.06 / 0.02 | 1.00 / 1.00 | 6.51 / 9.30 | – | 10.81 / 10.38 | 9.18 / 26.09 | direct | no strategy 10% faster than direct |
| Points | Triangle | Gpu | f64 | 8 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 6.16 / 9.27 | – | 11.82 / 10.89 | 7.22 / 22.06 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 0.10 | 1.00 | 0.99 | 1.16 | – | – | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 0.16 | 1.00 | 0.98 | 0.82 | – | – | sumfact | sumfact 0.82 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 0.34 | 1.00 | 0.74 | 0.45 | – | – | sumfact | sumfact 0.45 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 0.64 | 1.00 | 0.83 | 0.41 | – | – | sumfact | sumfact 0.41 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 1.15 | 1.00 | 0.94 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 1.39 | 1.00 | 0.78 | 0.23 | – | – | sumfact | sumfact 0.23 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 1.90 | 1.00 | 0.67 | 0.19 | – | – | sumfact | sumfact 0.19 of direct |
| Points | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 2.49 | 1.00 | 0.69 | 0.15 | – | – | sumfact | sumfact 0.15 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 0.22 | 1.00 | 0.97 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 0.31 | 1.00 | 0.91 | 0.72 | – | – | sumfact | sumfact 0.72 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 0.69 | 1.00 | 0.82 | 0.49 | – | – | sumfact | sumfact 0.49 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 1.21 | 1.00 | 0.85 | 0.38 | – | – | sumfact | sumfact 0.38 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 3.91 | 1.00 | 1.05 | 0.26 | – | – | sumfact | sumfact 0.26 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 5.03 | 1.00 | 0.74 | 0.18 | – | – | sumfact | sumfact 0.18 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 7.44 | 1.00 | 0.74 | 0.15 | – | – | sumfact | sumfact 0.15 of direct |
| Points | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 13.5 | 1.00 | 0.74 | 0.12 | – | – | sumfact | sumfact 0.12 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 0.09 | 1.00 | 1.30 | 1.13 | – | – | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 0.21 | 1.00 | 0.69 | 0.87 | – | – | loops | loops 0.69 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 0.47 | 1.00 | 0.73 | 0.49 | – | – | sumfact | sumfact 0.49 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 1.48 | 1.00 | 0.59 | 0.35 | – | – | sumfact | sumfact 0.35 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 2.97 | 1.00 | 0.79 | 0.31 | – | – | sumfact | sumfact 0.31 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 4.22 | 1.00 | 0.61 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 7.46 | 1.00 | 0.56 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Points | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 9.81 | 1.00 | 0.55 | 0.17 | – | – | sumfact | sumfact 0.17 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 0.18 | 1.00 | 0.99 | 1.01 | – | – | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 0.20 | 1.00 | 1.02 | 0.92 | – | – | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 0.30 | 1.00 | 0.99 | 0.65 | – | – | sumfact | sumfact 0.65 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 0.36 | 1.00 | 0.95 | 0.58 | – | – | sumfact | sumfact 0.58 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 0.65 | 1.00 | 0.91 | 0.36 | – | – | sumfact | sumfact 0.36 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 0.74 | 1.00 | 0.76 | 0.39 | – | – | sumfact | sumfact 0.39 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 1.15 | 1.00 | 0.58 | 0.20 | – | – | sumfact | sumfact 0.20 of direct |
| Points | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 1.30 | 1.00 | 0.82 | 0.28 | – | – | sumfact | sumfact 0.28 of direct |
| Points | Quadrilateral | Gpu | f32 | 1 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.00 / 1.12 | 1.03 / 1.08 | 0.99 / 1.03 | 1.35 / 1.89 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | Gpu | f32 | 2 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.22 / 1.21 | 1.08 / 0.99 | 1.19 / 0.94 | 2.59 / 4.12 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | Gpu | f32 | 3 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.28 / 1.50 | 0.91 / 0.97 | 0.90 / 0.99 | 3.97 / 8.85 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | Gpu | f32 | 4 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 2.14 / 3.40 | 0.82 / 0.90 | 0.78 / 0.83 | 3.93 / 9.43 | shared | shared 0.78 / 0.83 of direct |
| Points | Quadrilateral | Gpu | f32 | 5 | direct | 0.14 / 0.05 | 1.00 / 1.00 | 2.61 / 6.42 | 0.63 / 0.73 | 0.63 / 0.66 | 6.14 / 15.90 | shared | shared 0.63 / 0.66 of direct |
| Points | Quadrilateral | Gpu | f32 | 6 | direct | 0.14 / 0.06 | 1.00 / 1.00 | 3.80 / 7.23 | 0.63 / 0.63 | 0.61 / 0.62 | 6.96 / 15.62 | shared | shared 0.61 / 0.62 of direct |
| Points | Quadrilateral | Gpu | f32 | 7 | direct | 0.21 / 0.06 | 1.00 / 1.00 | 3.73 / 10.06 | 0.57 / 0.86 | 0.54 / 0.80 | 5.45 / 16.81 | shared | shared 0.54 / 0.80 of direct |
| Points | Quadrilateral | Gpu | f32 | 8 | direct | 0.22 / 0.08 | 1.00 / 1.00 | 5.19 / 10.88 | 0.57 / 0.68 | 0.54 / 0.64 | 6.27 / 16.13 | shared | shared 0.54 / 0.64 of direct |
| Points | Quadrilateral | Gpu | f64 | 1 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.00 / 1.01 | 1.00 / 1.07 | 1.12 / 1.01 | 2.56 / 3.80 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.38 / 1.46 | 0.95 / 0.94 | 1.24 / 0.99 | 5.01 / 9.19 | direct | no strategy 10% faster than direct |
| Points | Quadrilateral | Gpu | f64 | 3 | direct | 0.04 / 0.01 | 1.00 / 1.00 | 1.57 / 4.77 | 0.73 / 0.88 | 0.73 / 0.87 | 6.83 / 17.15 | sumfact | sumfact 0.73 / 0.88 of direct, within 3% of shared (0.73 / 0.87) |
| Points | Quadrilateral | Gpu | f64 | 4 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 2.19 / 6.19 | 0.86 / 0.77 | 0.68 / 0.74 | 6.87 / 19.04 | shared | shared 0.68 / 0.74 of direct |
| Points | Quadrilateral | Gpu | f64 | 5 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 4.74 / 9.48 | 0.76 / 0.69 | 0.70 / 0.64 | 13.26 / 25.15 | shared | shared 0.70 / 0.64 of direct |
| Points | Quadrilateral | Gpu | f64 | 6 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 5.36 / 9.03 | 0.57 / 0.53 | 0.53 / 0.53 | 9.12 / 23.18 | shared | shared 0.53 / 0.53 of direct |
| Points | Quadrilateral | Gpu | f64 | 7 | direct | 0.19 / 0.07 | 1.00 / 1.00 | 3.62 / 4.36 | 0.25 / 0.26 | 0.19 / 0.24 | 3.83 / 10.17 | shared | shared 0.19 / 0.24 of direct |
| Points | Quadrilateral | Gpu | f64 | 8 | direct | 0.33 / 0.10 | 1.00 / 1.00 | 3.68 / 4.26 | 0.18 / 0.24 | 0.17 / 0.21 | 2.77 / 8.53 | shared | shared 0.17 / 0.21 of direct |
| Points | Tetrahedron | CpuSimd128 | f32 | 1 | direct | 0.15 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | CpuSimd128 | f32 | 2 | direct | 0.27 | 1.00 | 0.83 | – | – | – | loops | loops 0.83 of direct |
| Points | Tetrahedron | CpuSimd128 | f32 | 3 | direct | 0.98 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| Points | Tetrahedron | CpuSimd128 | f32 | 4 | direct | 1.70 | 1.00 | 0.74 | – | – | – | loops | loops 0.74 of direct |
| Points | Tetrahedron | CpuSimd128 | f32 | 5 | loops | 3.53 | n.m.; T3: 1.18 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f32 | 6 | loops | 16.5 | n.m.; T3: 1.04 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f32 | 7 | loops | 11.7 | n.m.; T3: 1.66 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f32 | 8 | loops | 20.2 | n.m.; T3: 2.29 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f64 | 1 | direct | 0.26 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | CpuSimd128 | f64 | 2 | direct | 0.80 | 1.00 | 0.88 | – | – | – | loops | loops 0.88 of direct |
| Points | Tetrahedron | CpuSimd128 | f64 | 3 | direct | 3.67 | 1.00 | 0.86 | – | – | – | loops | loops 0.86 of direct |
| Points | Tetrahedron | CpuSimd128 | f64 | 4 | direct | 9.00 | 1.00 | 0.83 | – | – | – | loops | loops 0.83 of direct |
| Points | Tetrahedron | CpuSimd128 | f64 | 5 | loops | 18.5 | n.m.; T3: 1.02 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f64 | 6 | loops | 61.3 | n.m.; T3: 1.01 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f64 | 7 | loops | 40.0 | n.m.; T3: 1.20 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd128 | f64 | 8 | loops | 48.1 | n.m.; T3: 1.50 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f32 | 1 | direct | 0.16 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | CpuSimd512 | f32 | 2 | direct | 0.64 | 1.00 | 0.68 | – | – | – | loops | loops 0.68 of direct |
| Points | Tetrahedron | CpuSimd512 | f32 | 3 | direct | 3.29 | 1.00 | 0.63 | – | – | – | loops | loops 0.63 of direct |
| Points | Tetrahedron | CpuSimd512 | f32 | 4 | direct | 5.93 | 1.00 | 0.52 | – | – | – | loops | loops 0.52 of direct |
| Points | Tetrahedron | CpuSimd512 | f32 | 5 | loops | 4.47 | n.m.; T3: 2.07 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f32 | 6 | loops | 21.9 | n.m.; T3: 1.27 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f32 | 7 | loops | 9.07 | n.m.; T3: 2.50 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f32 | 8 | loops | 13.4 | n.m.; T3: 2.42 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f64 | 1 | direct | 0.22 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | CpuSimd512 | f64 | 2 | direct | 0.27 | 1.00 | 0.76 | – | – | – | loops | loops 0.76 of direct |
| Points | Tetrahedron | CpuSimd512 | f64 | 3 | direct | 0.39 | 1.00 | 0.73 | – | – | – | loops | loops 0.73 of direct |
| Points | Tetrahedron | CpuSimd512 | f64 | 4 | direct | 0.73 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | CpuSimd512 | f64 | 5 | loops | 1.83 | n.m.; T3: 1.08 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f64 | 6 | loops | 14.0 | n.m.; T3: 0.61 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f64 | 7 | loops | 5.30 | n.m.; T3: 1.36 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | CpuSimd512 | f64 | 8 | loops | 11.5 | n.m.; T3: 1.39 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | Gpu | f32 | 1 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.21 / 1.04 | – | 1.10 / 1.03 | 1.53 / 1.71 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f32 | 2 | direct | 0.09 / 0.03 | 1.00 / 1.00 | 1.33 / 1.48 | – | 1.24 / 1.36 | 2.47 / 4.24 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f32 | 3 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 2.74 / 3.48 | – | 2.63 / 3.12 | 3.75 / 8.14 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f32 | 4 | direct | 0.22 / 0.07 | 1.00 / 1.00 | 3.49 / 5.11 | – | 3.34 / 3.67 | 5.30 / 14.20 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f32 | 5 | loops | 1.65 / 0.84 | n.m.; T3: 0.17 / 0.10 | 1.00 / 1.00 | – | – | 1.05 / 1.77 | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | Gpu | f32 | 6 | loops | 4.01 / 1.61 | n.m.; T3: 0.17 / 0.22 | 1.00 / 1.00 | – | – | 0.55 / 1.06 | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | Gpu | f32 | 7 | loops | 10.3 / 3.64 | n.m.; T3: 0.32 / 0.30 | 1.00 / 1.00 | – | – | 0.26 / 0.80 | cooperative | cooperative 0.26 / 0.80 of loops |
| Points | Tetrahedron | Gpu | f32 | 8 | loops | 22.8 / 6.66 | n.m.; T3: 0.31 / 0.37 | 1.00 / 1.00 | – | – | 0.22 / 0.59 | cooperative | cooperative 0.22 / 0.59 of loops |
| Points | Tetrahedron | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 0.91 / 1.08 | – | 1.04 / 1.17 | 1.99 / 3.02 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.95 / 1.99 | – | 2.17 / 4.78 | 4.51 / 8.74 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f64 | 3 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 6.06 / 10.27 | – | 6.92 / 9.38 | 6.72 / 17.25 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f64 | 4 | direct | 0.12 / 0.03 | 1.00 / 1.00 | 7.38 / 10.19 | – | 9.90 / 10.02 | 4.62 / 14.85 | direct | no strategy 10% faster than direct |
| Points | Tetrahedron | Gpu | f64 | 5 | loops | 3.68 / 0.60 | n.m.; T3: 0.13 / 0.14 | 1.00 / 1.00 | – | 0.88 / 1.21 | 0.25 / 1.22 | loops | no strategy 10% faster than loops |
| Points | Tetrahedron | Gpu | f64 | 6 | loops | 8.99 / 2.26 | n.m.; T3: 0.66 / 0.73 | 1.00 / 1.00 | – | 0.77 / 0.78 | 0.13 / 0.47 | cooperative | cooperative 0.13 / 0.47 of loops |
| Points | Tetrahedron | Gpu | f64 | 7 | loops | 19.6 / 4.55 | n.m.; T3: 0.61 / 0.65 | 1.00 / 1.00 | – | – | 0.08 / 0.34 | cooperative | cooperative 0.08 / 0.34 of loops |
| Points | Tetrahedron | Gpu | f64 | 8 | loops | 42.9 / 7.66 | n.m.; T3: 0.61 / 0.74 | 1.00 / 1.00 | – | – | 0.08 / 0.28 | cooperative | cooperative 0.08 / 0.28 of loops |
| Points | Hexahedron | CpuSimd128 | f32 | 1 | direct | 0.23 | 1.00 | 0.88 | 0.74 | – | – | sumfact | sumfact 0.74 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 2 | direct | 1.05 | 1.00 | 0.65 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 3 | direct | 3.02 | 1.00 | 0.72 | 0.14 | – | – | sumfact | sumfact 0.14 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 4 | direct | 9.23 | 1.00 | 0.78 | 0.07 | – | – | sumfact | sumfact 0.07 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 5 | direct | 24.5 | 1.00 | 0.88 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 6 | direct | 65.6 | 1.00 | 0.79 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 7 | direct | 158 | 1.00 | 0.76 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Points | Hexahedron | CpuSimd128 | f32 | 8 | loops | 218 | n.m.; T3: 1.39 | 1.00 | 0.02 | – | – | sumfact | sumfact 0.02 of loops |
| Points | Hexahedron | CpuSimd128 | f64 | 1 | direct | 0.48 | 1.00 | 0.93 | 0.81 | – | – | sumfact | sumfact 0.81 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 2 | direct | 2.19 | 1.00 | 0.92 | 0.28 | – | – | sumfact | sumfact 0.28 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 3 | direct | 13.2 | 1.00 | 0.93 | 0.12 | – | – | sumfact | sumfact 0.12 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 4 | direct | 68.7 | 1.00 | 0.88 | 0.07 | – | – | sumfact | sumfact 0.07 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 5 | direct | 120 | 1.00 | 0.88 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 6 | direct | 192 | 1.00 | 0.86 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 7 | direct | 294 | 1.00 | 0.82 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Points | Hexahedron | CpuSimd128 | f64 | 8 | loops | 339 | n.m.; T3: 1.30 | 1.00 | 0.02 | – | – | sumfact | sumfact 0.02 of loops |
| Points | Hexahedron | CpuSimd512 | f32 | 1 | direct | 0.21 | 1.00 | 0.99 | 0.93 | – | – | direct | no strategy 10% faster than direct |
| Points | Hexahedron | CpuSimd512 | f32 | 2 | direct | 2.47 | 1.00 | 0.60 | 0.37 | – | – | sumfact | sumfact 0.37 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 3 | direct | 11.4 | 1.00 | 0.47 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 4 | direct | 25.0 | 1.00 | 0.64 | 0.09 | – | – | sumfact | sumfact 0.09 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 5 | direct | 43.7 | 1.00 | 0.62 | 0.07 | – | – | sumfact | sumfact 0.07 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 6 | direct | 71.2 | 1.00 | 0.59 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 7 | direct | 107 | 1.00 | 0.58 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Points | Hexahedron | CpuSimd512 | f32 | 8 | direct | 153 | 1.00 | 0.57 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 1 | direct | 0.38 | 1.00 | 0.99 | 1.04 | – | – | direct | no strategy 10% faster than direct |
| Points | Hexahedron | CpuSimd512 | f64 | 2 | direct | 0.54 | 1.00 | 0.85 | 0.82 | – | – | sumfact | sumfact 0.82 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 3 | direct | 1.30 | 1.00 | 0.61 | 0.34 | – | – | sumfact | sumfact 0.34 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 4 | direct | 2.70 | 1.00 | 0.86 | 0.20 | – | – | sumfact | sumfact 0.20 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 5 | direct | 6.26 | 1.00 | 0.84 | 0.13 | – | – | sumfact | sumfact 0.13 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 6 | direct | 14.8 | 1.00 | 0.86 | 0.10 | – | – | sumfact | sumfact 0.10 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 7 | direct | 38.2 | 1.00 | 0.73 | 0.06 | – | – | sumfact | sumfact 0.06 of direct |
| Points | Hexahedron | CpuSimd512 | f64 | 8 | direct | 140 | 1.00 | 0.91 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Points | Hexahedron | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.14 / 1.20 | 1.03 / 1.05 | 0.95 / 1.01 | 1.77 / 3.13 | direct | no strategy 10% faster than direct |
| Points | Hexahedron | Gpu | f32 | 2 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 3.19 / 6.63 | 0.92 / 0.99 | 0.97 / 0.94 | 4.04 / 8.90 | direct | no strategy 10% faster than direct |
| Points | Hexahedron | Gpu | f32 | 3 | direct | 0.23 / 0.11 | 1.00 / 1.00 | 7.07 / 9.07 | 0.53 / 0.53 | 0.51 / 0.48 | 6.42 / 15.21 | shared | shared 0.51 / 0.48 of direct |
| Points | Hexahedron | Gpu | f32 | 4 | direct | 1.68 / 0.99 | 1.00 / 1.00 | 3.43 / 3.10 | 0.12 / 0.12 | 0.12 / 0.12 | 1.86 / 2.96 | sumfact | sumfact 0.12 / 0.12 of direct |
| Points | Hexahedron | Gpu | f32 | 5 | direct | 5.70 / 3.49 | 1.00 / 1.00 | 2.33 / 1.96 | 0.06 / 0.05 | 0.06 / 0.05 | 0.88 / 1.37 | shared | shared 0.06 / 0.05 of direct |
| Points | Hexahedron | Gpu | f32 | 6 | direct | 12.3 / 8.31 | 1.00 / 1.00 | 2.56 / 1.63 | 0.04 / 0.03 | 0.10 / 0.06 | 0.71 / 0.92 | sumfact | sumfact 0.04 / 0.03 of direct |
| Points | Hexahedron | Gpu | f32 | 7 | direct | 36.8 / 17.2 | 1.00 / 1.00 | 1.90 / 1.53 | 0.04 / 0.03 | 0.07 / 0.05 | 0.35 / 0.61 | sumfact | sumfact 0.04 / 0.03 of direct |
| Points | Hexahedron | Gpu | f32 | 8 | direct | 41.8 / 27.7 | 1.00 / 1.00 | 3.10 / 1.56 | 0.05 / 0.02 | 0.09 / 0.04 | 0.53 / 0.71 | sumfact | sumfact 0.05 / 0.02 of direct |
| Points | Hexahedron | Gpu | f64 | 1 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 1.27 / 1.49 | 0.98 / 1.01 | 1.13 / 1.04 | 3.38 / 5.91 | direct | no strategy 10% faster than direct |
| Points | Hexahedron | Gpu | f64 | 2 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 5.09 / 10.05 | 0.91 / 0.90 | 0.74 / 0.90 | 7.42 / 17.82 | shared | shared 0.74 / 0.90 of direct |
| Points | Hexahedron | Gpu | f64 | 3 | direct | 0.20 / 0.09 | 1.00 / 1.00 | 7.03 / 5.42 | 0.34 / 0.41 | 0.39 / 0.70 | 4.30 / 9.31 | sumfact | sumfact 0.34 / 0.41 of direct |
| Points | Hexahedron | Gpu | f64 | 4 | direct | 4.18 / 1.71 | 1.00 / 1.00 | 1.26 / 0.81 | 0.16 / 0.14 | 0.17 / 0.14 | 0.40 / 0.90 | sumfact | sumfact 0.16 / 0.14 of direct |
| Points | Hexahedron | Gpu | f64 | 5 | direct | 8.70 / 2.38 | 1.00 / 1.00 | 1.67 / 1.59 | 0.16 / 0.14 | 0.17 / 0.15 | 0.32 / 1.11 | sumfact | sumfact 0.16 / 0.14 of direct |
| Points | Hexahedron | Gpu | f64 | 6 | direct | 24.8 / 4.51 | 1.00 / 1.00 | 1.67 / 2.04 | 0.11 / 0.11 | 0.17 / 0.15 | 0.19 / 0.96 | sumfact | sumfact 0.11 / 0.11 of direct |
| Points | Hexahedron | Gpu | f64 | 7 | direct | 58.0 / 8.61 | 1.00 / 1.00 | 1.69 / 2.40 | 0.08 / 0.09 | 0.14 / 0.15 | 0.13 / 0.74 | sumfact | sumfact 0.08 / 0.09 of direct |
| Points | Hexahedron | Gpu | f64 | 8 | direct | 111 / 17.6 | 1.00 / 1.00 | 1.64 / 2.35 | 0.06 / 0.06 | 0.13 / 0.12 | 0.13 / 0.53 | sumfact | sumfact 0.06 / 0.06 of direct |
| Points | Prism | CpuSimd128 | f32 | 1 | direct | 0.20 | 1.00 | 0.97 | 0.92 | – | – | direct | no strategy 10% faster than direct |
| Points | Prism | CpuSimd128 | f32 | 2 | direct | 0.66 | 1.00 | 0.66 | 0.46 | – | – | sumfact | sumfact 0.46 of direct |
| Points | Prism | CpuSimd128 | f32 | 3 | direct | 1.70 | 1.00 | 0.76 | 0.30 | – | – | sumfact | sumfact 0.30 of direct |
| Points | Prism | CpuSimd128 | f32 | 4 | direct | 13.7 | 1.00 | 1.03 | 0.21 | – | – | sumfact | sumfact 0.21 of direct |
| Points | Prism | CpuSimd128 | f32 | 5 | loops | 7.36 | n.m.; T3: 1.60 | 1.00 | 0.22 | – | – | sumfact | sumfact 0.22 of loops |
| Points | Prism | CpuSimd128 | f32 | 6 | loops | 15.0 | n.m.; T3: 2.04 | 1.00 | 0.17 | – | – | sumfact | sumfact 0.17 of loops |
| Points | Prism | CpuSimd128 | f32 | 7 | loops | 29.1 | n.m.; T3: 2.21 | 1.00 | 0.14 | – | – | sumfact | sumfact 0.14 of loops |
| Points | Prism | CpuSimd128 | f32 | 8 | loops | 111 | n.m.; T3: 1.27 | 1.00 | 0.12 | – | – | sumfact | sumfact 0.12 of loops |
| Points | Prism | CpuSimd128 | f64 | 1 | direct | 0.76 | 1.00 | 0.97 | 0.55 | – | – | sumfact | sumfact 0.55 of direct |
| Points | Prism | CpuSimd128 | f64 | 2 | direct | 1.45 | 1.00 | 0.90 | 0.43 | – | – | sumfact | sumfact 0.43 of direct |
| Points | Prism | CpuSimd128 | f64 | 3 | direct | 7.29 | 1.00 | 0.88 | 0.29 | – | – | sumfact | sumfact 0.29 of direct |
| Points | Prism | CpuSimd128 | f64 | 4 | direct | 82.8 | 1.00 | 1.04 | 0.23 | – | – | sumfact | sumfact 0.23 of direct |
| Points | Prism | CpuSimd128 | f64 | 5 | loops | 41.9 | n.m.; T3: 1.24 | 1.00 | 0.18 | – | – | sumfact | sumfact 0.18 of loops |
| Points | Prism | CpuSimd128 | f64 | 6 | loops | 65.3 | n.m.; T3: 1.35 | 1.00 | 0.15 | – | – | sumfact | sumfact 0.15 of loops |
| Points | Prism | CpuSimd128 | f64 | 7 | loops | 95.2 | n.m.; T3: 1.45 | 1.00 | 0.13 | – | – | sumfact | sumfact 0.13 of loops |
| Points | Prism | CpuSimd128 | f64 | 8 | loops | 189 | n.m.; T3: 1.22 | 1.00 | 0.12 | – | – | sumfact | sumfact 0.12 of loops |
| Points | Prism | CpuSimd512 | f32 | 1 | direct | 0.22 | 1.00 | 0.96 | 0.90 | – | – | direct | no strategy 10% faster than direct |
| Points | Prism | CpuSimd512 | f32 | 2 | direct | 1.56 | 1.00 | 0.59 | 0.50 | – | – | sumfact | sumfact 0.50 of direct |
| Points | Prism | CpuSimd512 | f32 | 3 | direct | 7.41 | 1.00 | 0.78 | 0.35 | – | – | sumfact | sumfact 0.35 of direct |
| Points | Prism | CpuSimd512 | f32 | 4 | direct | 39.3 | 1.00 | 1.00 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Points | Prism | CpuSimd512 | f32 | 5 | loops | 9.86 | n.m.; T3: 2.43 | 1.00 | 0.40 | – | – | sumfact | sumfact 0.40 of loops |
| Points | Prism | CpuSimd512 | f32 | 6 | loops | 19.3 | n.m.; T3: 2.04 | 1.00 | 0.28 | – | – | sumfact | sumfact 0.28 of loops |
| Points | Prism | CpuSimd512 | f32 | 7 | loops | 26.2 | n.m.; T3: 2.24 | 1.00 | 0.25 | – | – | sumfact | sumfact 0.25 of loops |
| Points | Prism | CpuSimd512 | f32 | 8 | direct | 84.9 | 1.00 | 0.59 | 0.10 | – | – | sumfact | sumfact 0.10 of direct |
| Points | Prism | CpuSimd512 | f64 | 1 | direct | 0.38 | 1.00 | 1.01 | 0.96 | – | – | direct | no strategy 10% faster than direct |
| Points | Prism | CpuSimd512 | f64 | 2 | direct | 0.51 | 1.00 | 0.89 | 0.88 | – | – | loops | loops 0.89 of direct, within 3% of sumfact (0.88) |
| Points | Prism | CpuSimd512 | f64 | 3 | direct | 0.99 | 1.00 | 0.81 | 0.41 | – | – | sumfact | sumfact 0.41 of direct |
| Points | Prism | CpuSimd512 | f64 | 4 | direct | 6.94 | 1.00 | 1.02 | 0.21 | – | – | sumfact | sumfact 0.21 of direct |
| Points | Prism | CpuSimd512 | f64 | 5 | loops | 3.13 | n.m.; T3: 1.29 | 1.00 | 0.30 | – | – | sumfact | sumfact 0.30 of loops |
| Points | Prism | CpuSimd512 | f64 | 6 | loops | 7.18 | n.m.; T3: 1.23 | 1.00 | 0.29 | – | – | sumfact | sumfact 0.29 of loops |
| Points | Prism | CpuSimd512 | f64 | 7 | loops | 12.2 | n.m.; T3: 1.41 | 1.00 | 0.24 | – | – | sumfact | sumfact 0.24 of loops |
| Points | Prism | CpuSimd512 | f64 | 8 | direct | 32.4 | 1.00 | 0.90 | 0.16 | – | – | sumfact | sumfact 0.16 of direct |
| Points | Prism | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.07 / 1.14 | 1.01 / 1.09 | 1.04 / 1.00 | 1.40 / 1.97 | direct | no strategy 10% faster than direct |
| Points | Prism | Gpu | f32 | 2 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 2.13 / 2.99 | 0.98 / 0.93 | 0.99 / 0.93 | 3.95 / 8.12 | direct | no strategy 10% faster than direct |
| Points | Prism | Gpu | f32 | 3 | direct | 0.15 / 0.07 | 1.00 / 1.00 | 4.89 / 7.89 | 0.90 / 0.96 | 0.84 / 0.91 | 8.41 / 15.92 | direct | no strategy 10% faster than direct |
| Points | Prism | Gpu | f32 | 4 | direct | 0.47 / 0.26 | 1.00 / 1.00 | 4.67 / 5.20 | 0.79 / 0.76 | 0.81 / 0.79 | 4.14 / 6.18 | sumfact | sumfact 0.79 / 0.76 of direct |
| Points | Prism | Gpu | f32 | 5 | direct | 1.56 / 1.08 | 1.00 / 1.00 | 3.71 / 3.58 | 0.53 / 0.40 | 0.37 / 0.32 | 1.76 / 2.68 | shared | shared 0.37 / 0.32 of direct |
| Points | Prism | Gpu | f32 | 6 | direct | 8.05 / 2.71 | 1.00 / 1.00 | 1.69 / 2.93 | 0.13 / 0.20 | 0.12 / 0.22 | 0.55 / 1.66 | shared | closest: shared 0.12 / 0.22 of direct, 10% slower than the fastest at one size |
| Points | Prism | Gpu | f32 | 7 | direct | 17.3 / 6.66 | 1.00 / 1.00 | 1.52 / 2.14 | 0.12 / 0.13 | 0.11 / 0.14 | 0.35 / 0.97 | shared | closest: shared 0.11 / 0.14 of direct, 7% slower than the fastest at one size |
| Points | Prism | Gpu | f32 | 8 | direct | 20.2 / 14.6 | 1.00 / 1.00 | 2.51 / 1.65 | 0.10 / 0.08 | – | 0.59 / 0.64 | sumfact | sumfact 0.10 / 0.08 of direct |
| Points | Prism | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.33 / 1.15 | 1.00 / 1.00 | 1.16 / 1.00 | 2.43 / 3.24 | direct | no strategy 10% faster than direct |
| Points | Prism | Gpu | f64 | 2 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 2.96 / 7.94 | 0.83 / 0.93 | 0.84 / 0.87 | 5.64 / 14.99 | shared | shared 0.84 / 0.87 of direct |
| Points | Prism | Gpu | f64 | 3 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 7.71 / 9.11 | 0.60 / 0.62 | 0.62 / 0.49 | 7.46 / 15.22 | shared | shared 0.62 / 0.49 of direct |
| Points | Prism | Gpu | f64 | 4 | direct | 1.89 / 1.03 | 1.00 / 1.00 | 1.61 / 0.85 | 0.08 / 0.10 | 0.06 / 0.10 | 0.54 / 0.92 | shared | shared 0.06 / 0.10 of direct |
| Points | Prism | Gpu | f64 | 5 | direct | 5.42 / 1.95 | 1.00 / 1.00 | 1.78 / 1.20 | 0.06 / 0.08 | 0.07 / 0.08 | 0.35 / 0.80 | sumfact | closest: sumfact 0.06 / 0.08 of direct, 5% slower than the fastest at one size |
| Points | Prism | Gpu | f64 | 6 | direct | 14.2 / 3.60 | 1.00 / 1.00 | 1.74 / 1.57 | 0.05 / 0.06 | 0.05 / 0.06 | 0.20 / 0.68 | shared | shared 0.05 / 0.06 of direct |
| Points | Prism | Gpu | f64 | 7 | direct | 21.5 / 5.06 | 1.00 / 1.00 | 2.25 / 2.42 | 0.06 / 0.07 | 0.05 / 0.06 | 0.18 / 0.70 | shared | shared 0.05 / 0.06 of direct |
| Points | Prism | Gpu | f64 | 8 | direct | 37.5 / 9.58 | 1.00 / 1.00 | 2.46 / 2.42 | 0.05 / 0.04 | 0.05 / 0.05 | 0.20 / 0.54 | sumfact | closest: sumfact 0.05 / 0.04 of direct, 7% slower than the fastest at one size |
| Points | Pyramid | CpuSimd128 | f32 | 1 | direct | 0.27 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Pyramid | CpuSimd128 | f32 | 2 | direct | 0.52 | 1.00 | 0.62 | – | – | – | loops | loops 0.62 of direct |
| Points | Pyramid | CpuSimd128 | f32 | 3 | direct | 1.36 | 1.00 | 0.76 | – | – | – | loops | loops 0.76 of direct |
| Points | Pyramid | CpuSimd128 | f64 | 1 | direct | 0.32 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Pyramid | CpuSimd128 | f64 | 2 | direct | 1.14 | 1.00 | 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Pyramid | CpuSimd128 | f64 | 3 | direct | 6.02 | 1.00 | 0.79 | – | – | – | loops | loops 0.79 of direct |
| Points | Pyramid | CpuSimd512 | f32 | 1 | direct | 0.17 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Pyramid | CpuSimd512 | f32 | 2 | direct | 1.03 | 1.00 | 0.66 | – | – | – | loops | loops 0.66 of direct |
| Points | Pyramid | CpuSimd512 | f32 | 3 | direct | 5.14 | 1.00 | 0.52 | – | – | – | loops | loops 0.52 of direct |
| Points | Pyramid | CpuSimd512 | f64 | 1 | direct | 0.27 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Points | Pyramid | CpuSimd512 | f64 | 2 | direct | 0.37 | 1.00 | 0.82 | – | – | – | loops | loops 0.82 of direct |
| Points | Pyramid | CpuSimd512 | f64 | 3 | direct | 0.55 | 1.00 | 0.72 | – | – | – | loops | loops 0.72 of direct |
| Points | Pyramid | Gpu | f32 | 1 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.18 / 1.09 | – | 0.99 / 1.02 | 1.34 / 1.86 | direct | no strategy 10% faster than direct |
| Points | Pyramid | Gpu | f32 | 2 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.44 / 1.70 | – | 1.43 / 1.69 | 2.18 / 4.30 | direct | no strategy 10% faster than direct |
| Points | Pyramid | Gpu | f32 | 3 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 4.90 / 5.30 | – | 3.96 / 5.66 | 3.97 / 7.23 | direct | no strategy 10% faster than direct |
| Points | Pyramid | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.29 / 1.13 | – | 1.18 / 1.45 | 2.53 / 3.37 | direct | no strategy 10% faster than direct |
| Points | Pyramid | Gpu | f64 | 2 | direct | 0.04 / 0.01 | 1.00 / 1.00 | 2.44 / 5.19 | – | 2.47 / 5.92 | 3.89 / 9.23 | direct | no strategy 10% faster than direct |
| Points | Pyramid | Gpu | f64 | 3 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 10.30 / 12.16 | – | 10.10 / 10.99 | 8.32 / 19.29 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 1 | direct | 0.12 | 1.00 | 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 2 | direct | 0.14 | 1.00 | 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 3 | direct | 0.15 | 1.00 | 1.19 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 4 | direct | 0.19 | 1.00 | 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 5 | direct | 0.19 | 1.00 | 1.27 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 6 | direct | 0.21 | 1.00 | 1.25 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 7 | direct | 0.23 | 1.00 | 1.43 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f32 | 8 | direct | 0.25 | 1.00 | 1.37 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 1 | direct | 0.21 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 2 | direct | 0.22 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 3 | direct | 0.25 | 1.00 | 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 4 | direct | 0.29 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 5 | direct | 0.32 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 6 | direct | 0.41 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 7 | direct | 0.49 | 1.00 | 1.10 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd128 | f64 | 8 | direct | 0.54 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 1 | direct | 0.27 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 2 | direct | 0.30 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 3 | direct | 0.26 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 4 | direct | 0.22 | 1.00 | 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 5 | direct | 0.21 | 1.00 | 1.35 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 6 | direct | 0.15 | 1.00 | 1.63 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 7 | direct | 0.20 | 1.00 | 1.24 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f32 | 8 | direct | 0.25 | 1.00 | 1.16 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 1 | direct | 0.27 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 2 | direct | 0.32 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 3 | direct | 0.34 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 4 | direct | 0.33 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 5 | direct | 0.35 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 6 | direct | 0.38 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Jacobian | Interval | CpuSimd512 | f64 | 7 | direct | 0.35 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | CpuSimd512 | f64 | 8 | direct | 0.35 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 1 | direct | 0.20 / 0.05 | 1.00 / 1.00 | 1.03 / 1.09 | – | 1.06 / 1.14 | 1.14 / 1.30 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 2 | direct | 0.21 / 0.05 | 1.00 / 1.00 | 1.02 / 1.15 | – | 1.01 / 1.15 | 1.14 / 1.49 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 3 | direct | 0.20 / 0.05 | 1.00 / 1.00 | 1.10 / 1.17 | – | 1.09 / 1.17 | 1.33 / 1.94 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 4 | direct | 0.21 / 0.05 | 1.00 / 1.00 | 1.07 / 1.17 | – | 1.09 / 1.14 | 1.43 / 2.39 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 5 | direct | 0.21 / 0.06 | 1.00 / 1.00 | 1.12 / 1.18 | – | 1.12 / 1.19 | 1.42 / 2.24 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 6 | direct | 0.21 / 0.07 | 1.00 / 1.00 | 1.09 / 1.07 | – | 1.10 / 1.00 | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 7 | direct | 0.21 / 0.06 | 1.00 / 1.00 | 1.16 / 1.26 | – | 1.12 / 1.23 | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f32 | 8 | direct | 0.22 / 0.06 | 1.00 / 1.00 | 1.07 / 1.22 | – | 1.06 / 1.19 | – | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 1 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.03 / 1.08 | – | 1.10 / 1.10 | 1.18 / 1.36 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 2 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.08 / 1.10 | – | 1.13 / 1.08 | 1.23 / 1.62 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 3 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 0.95 / 1.05 | – | 1.04 / 1.08 | 1.42 / 2.25 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 4 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.08 / 1.02 | – | 1.11 / 1.09 | 1.72 / 2.75 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 5 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 0.97 / 1.18 | – | 0.94 / 1.12 | 1.47 / 2.77 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 6 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.16 / 1.16 | – | 1.13 / 1.14 | 1.71 / 2.79 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 7 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 1.08 / 1.27 | – | 1.13 / 1.23 | 2.01 / 4.76 | direct | no strategy 10% faster than direct |
| Jacobian | Interval | Gpu | f64 | 8 | direct | 0.08 / 0.02 | 1.00 / 1.00 | 1.16 / 1.22 | – | 1.07 / 1.15 | 2.37 / 5.32 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 1 | direct | 0.33 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 2 | direct | 0.55 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 3 | direct | 0.65 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 4 | direct | 1.09 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 5 | direct | 1.63 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 6 | direct | 2.67 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 7 | direct | 3.58 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Jacobian | Triangle | CpuSimd128 | f32 | 8 | direct | 3.94 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 1 | direct | 0.74 | 1.00 | 1.11 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 2 | direct | 0.97 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 3 | direct | 1.41 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 4 | direct | 2.27 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 5 | direct | 4.48 | 1.00 | 0.90 | – | – | – | loops | loops 0.90 of direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 6 | direct | 7.75 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 7 | direct | 13.0 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd128 | f64 | 8 | direct | 18.4 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 1 | direct | 0.31 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 2 | direct | 0.54 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 3 | direct | 1.03 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 4 | direct | 2.00 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 5 | direct | 3.87 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 6 | direct | 6.83 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 7 | direct | 10.5 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f32 | 8 | direct | 11.3 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 1 | direct | 0.76 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 2 | direct | 0.75 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 3 | direct | 0.92 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 4 | direct | 1.07 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 5 | direct | 1.13 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 6 | direct | 1.78 | 1.00 | 1.16 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 7 | direct | 2.73 | 1.00 | 0.86 | – | – | – | loops | loops 0.86 of direct |
| Jacobian | Triangle | CpuSimd512 | f64 | 8 | direct | 3.10 | 1.00 | 0.85 | – | – | – | loops | loops 0.85 of direct |
| Jacobian | Triangle | Gpu | f32 | 1 | direct | 0.25 / 0.10 | 1.00 / 1.00 | 1.04 / 1.00 | – | 1.00 / 0.99 | 1.28 / 1.64 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 2 | direct | 0.24 / 0.10 | 1.00 / 1.00 | 1.10 / 1.10 | – | 1.07 / 1.04 | 2.09 / 2.97 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 3 | direct | 0.26 / 0.11 | 1.00 / 1.00 | 1.15 / 1.23 | – | 1.12 / 1.14 | 3.59 / 6.51 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 4 | direct | 0.29 / 0.12 | 1.00 / 1.00 | 1.27 / 1.39 | – | 1.23 / 1.32 | 3.42 / 6.21 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 5 | direct | 0.38 / 0.15 | 1.00 / 1.00 | 1.43 / 1.76 | – | 1.32 / 2.01 | 5.46 / 11.74 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 6 | direct | 0.45 / 0.18 | 1.00 / 1.00 | 2.10 / 2.42 | – | 1.81 / 2.32 | 4.72 / 10.17 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 7 | direct | 0.56 / 0.22 | 1.00 / 1.00 | 2.48 / 2.63 | – | 1.94 / 1.84 | 8.09 / 18.55 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f32 | 8 | direct | 0.74 / 0.27 | 1.00 / 1.00 | 2.46 / 2.94 | – | 2.37 / 2.44 | 6.33 / 16.32 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 1 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 1.10 / 1.07 | – | 1.06 / 1.05 | 2.22 / 3.33 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 2 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.17 / 1.10 | – | 1.10 / 1.15 | 5.56 / 9.95 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 3 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.42 / 1.42 | – | 1.30 / 1.81 | 9.67 / 21.09 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 4 | direct | 0.14 / 0.05 | 1.00 / 1.00 | 1.56 / 1.86 | – | 1.50 / 3.15 | 7.41 / 20.12 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 5 | direct | 0.18 / 0.06 | 1.00 / 1.00 | 1.89 / 4.02 | – | 1.65 / 4.67 | 11.77 / 32.13 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 6 | direct | 0.25 / 0.08 | 1.00 / 1.00 | 2.37 / 5.14 | – | 3.06 / 5.26 | 8.93 / 25.32 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 7 | direct | 0.24 / 0.09 | 1.00 / 1.00 | 4.37 / 5.41 | – | 5.14 / 5.21 | 16.29 / 40.71 | direct | no strategy 10% faster than direct |
| Jacobian | Triangle | Gpu | f64 | 8 | direct | 0.44 / 0.12 | 1.00 / 1.00 | 3.31 / 5.40 | – | 4.34 / 6.35 | 9.59 / 32.54 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 0.39 | 1.00 | 1.05 | 1.55 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 0.59 | 1.00 | 1.04 | 1.03 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 1.12 | 1.00 | 1.02 | 0.62 | – | – | sumfact | sumfact 0.62 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 2.05 | 1.00 | 0.90 | 0.47 | – | – | sumfact | sumfact 0.47 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 3.28 | 1.00 | 0.90 | 0.39 | – | – | sumfact | sumfact 0.39 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 4.09 | 1.00 | 0.97 | 0.34 | – | – | sumfact | sumfact 0.34 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 4.99 | 1.00 | 0.92 | 0.27 | – | – | sumfact | sumfact 0.27 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 6.53 | 1.00 | 0.94 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 0.93 | 1.00 | 0.98 | 1.25 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 1.22 | 1.00 | 1.00 | 1.01 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 2.34 | 1.00 | 0.95 | 0.77 | – | – | sumfact | sumfact 0.77 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 4.09 | 1.00 | 0.92 | 0.54 | – | – | sumfact | sumfact 0.54 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 12.6 | 1.00 | 0.98 | 0.35 | – | – | sumfact | sumfact 0.35 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 16.0 | 1.00 | 0.80 | 0.28 | – | – | sumfact | sumfact 0.28 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 22.9 | 1.00 | 0.85 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Jacobian | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 40.7 | 1.00 | 0.81 | 0.19 | – | – | sumfact | sumfact 0.19 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 0.35 | 1.00 | 1.05 | 1.19 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 0.67 | 1.00 | 0.91 | 0.90 | – | – | sumfact | sumfact 0.90 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 1.49 | 1.00 | 1.01 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 3.65 | 1.00 | 0.99 | 0.66 | – | – | sumfact | sumfact 0.66 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 8.34 | 1.00 | 0.95 | 0.54 | – | – | sumfact | sumfact 0.54 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 10.5 | 1.00 | 0.97 | 0.57 | – | – | sumfact | sumfact 0.57 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 16.1 | 1.00 | 0.97 | 0.46 | – | – | sumfact | sumfact 0.46 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 19.9 | 1.00 | 0.95 | 0.40 | – | – | sumfact | sumfact 0.40 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 0.93 | 1.00 | 0.99 | 1.09 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 1.02 | 1.00 | 1.03 | 1.05 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 1.29 | 1.00 | 0.99 | 0.94 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 1.50 | 1.00 | 0.94 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 2.70 | 1.00 | 0.88 | 0.45 | – | – | sumfact | sumfact 0.45 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 3.34 | 1.00 | 0.88 | 0.49 | – | – | sumfact | sumfact 0.49 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 2.90 | 1.00 | 1.03 | 0.49 | – | – | sumfact | sumfact 0.49 of direct |
| Jacobian | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 5.55 | 1.00 | 0.95 | 0.38 | – | – | sumfact | sumfact 0.38 of direct |
| Jacobian | Quadrilateral | Gpu | f32 | 1 | direct | 0.26 / 0.11 | 1.00 / 1.00 | 1.00 / 1.04 | 1.01 / 1.05 | 1.04 / 0.98 | 1.87 / 2.75 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f32 | 2 | direct | 0.28 / 0.12 | 1.00 / 1.00 | 1.05 / 1.16 | 1.00 / 1.03 | 0.96 / 0.96 | 3.36 / 6.12 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f32 | 3 | direct | 0.29 / 0.12 | 1.00 / 1.00 | 1.25 / 1.37 | 0.97 / 0.97 | 0.97 / 0.92 | 6.92 / 13.49 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f32 | 4 | direct | 0.36 / 0.15 | 1.00 / 1.00 | 1.86 / 2.64 | 0.84 / 0.83 | 0.78 / 0.77 | 5.76 / 12.09 | shared | shared 0.78 / 0.77 of direct |
| Jacobian | Quadrilateral | Gpu | f32 | 5 | direct | 0.46 / 0.21 | 1.00 / 1.00 | 2.47 / 4.17 | 0.73 / 0.76 | 0.72 / 0.68 | 9.90 / 19.57 | shared | shared 0.72 / 0.68 of direct |
| Jacobian | Quadrilateral | Gpu | f32 | 6 | direct | 0.76 / 0.36 | 1.00 / 1.00 | 2.11 / 3.38 | 0.50 / 0.48 | 0.47 / 0.44 | 6.44 / 12.86 | shared | shared 0.47 / 0.44 of direct |
| Jacobian | Quadrilateral | Gpu | f32 | 7 | direct | 1.48 / 0.81 | 1.00 / 1.00 | 1.59 / 2.24 | 0.29 / 0.26 | 0.27 / 0.24 | 3.58 / 6.19 | shared | shared 0.27 / 0.24 of direct |
| Jacobian | Quadrilateral | Gpu | f32 | 8 | direct | 1.73 / 1.00 | 1.00 / 1.00 | 1.97 / 2.47 | 0.27 / 0.21 | 0.26 / 0.19 | 3.53 / 5.78 | shared | shared 0.26 / 0.19 of direct |
| Jacobian | Quadrilateral | Gpu | f64 | 1 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.19 / 1.03 | 1.39 / 1.39 | 1.31 / 1.37 | 4.97 / 8.59 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f64 | 2 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 1.19 / 1.25 | 1.11 / 1.35 | 1.22 / 1.35 | 8.82 / 19.78 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f64 | 3 | direct | 0.14 / 0.05 | 1.00 / 1.00 | 1.26 / 2.04 | 0.88 / 1.11 | 0.90 / 1.10 | 13.26 / 34.33 | direct | no strategy 10% faster than direct |
| Jacobian | Quadrilateral | Gpu | f64 | 4 | direct | 0.22 / 0.07 | 1.00 / 1.00 | 1.48 / 4.90 | 0.60 / 0.81 | 0.60 / 0.82 | 9.77 / 29.52 | sumfact | sumfact 0.60 / 0.81 of direct |
| Jacobian | Quadrilateral | Gpu | f64 | 5 | direct | 0.27 / 0.10 | 1.00 / 1.00 | 2.15 / 4.79 | 0.48 / 0.60 | 0.48 / 0.59 | 14.66 / 39.19 | sumfact | sumfact 0.48 / 0.60 of direct, within 3% of shared (0.48 / 0.59) |
| Jacobian | Quadrilateral | Gpu | f64 | 6 | direct | 0.36 / 0.13 | 1.00 / 1.00 | 2.90 / 6.13 | 0.48 / 0.55 | 0.45 / 0.51 | 12.27 / 32.98 | shared | shared 0.45 / 0.51 of direct |
| Jacobian | Quadrilateral | Gpu | f64 | 7 | direct | 1.03 / 0.32 | 1.00 / 1.00 | 1.89 / 2.79 | 0.23 / 0.27 | 0.18 / 0.25 | 4.53 / 14.18 | shared | shared 0.18 / 0.25 of direct |
| Jacobian | Quadrilateral | Gpu | f64 | 8 | direct | 1.66 / 0.65 | 1.00 / 1.00 | 1.98 / 1.91 | 0.20 / 0.18 | 0.14 / 0.15 | 3.17 / 7.68 | shared | shared 0.14 / 0.15 of direct |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 1 | direct | 1.02 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 2 | direct | 1.39 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 3 | direct | 2.76 | 1.00 | 0.90 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 4 | direct | 5.40 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 5 | loops | 12.7 | n.m.; T3: 1.22 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 6 | loops | 51.3 | n.m.; T3: 1.05 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 7 | loops | 42.8 | n.m.; T3: 1.37 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f32 | 8 | loops | 83.3 | n.m.; T3: 1.68 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 1 | direct | 2.42 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 2 | direct | 4.15 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 3 | direct | 16.2 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 4 | direct | 39.7 | 1.00 | 0.90 | – | – | – | loops | loops 0.90 of direct |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 5 | loops | 63.5 | n.m.; T3: 1.22 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 6 | loops | 177 | n.m.; T3: 1.20 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 7 | loops | 127 | n.m.; T3: 1.55 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd128 | f64 | 8 | loops | 159 | n.m.; T3: 1.90 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 1 | direct | 1.19 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 2 | direct | 3.73 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 3 | direct | 11.2 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 4 | direct | 15.7 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 5 | loops | 20.3 | n.m.; T3: 1.09 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 6 | loops | 78.3 | n.m.; T3: 0.95 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 7 | loops | 39.4 | n.m.; T3: 2.61 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f32 | 8 | loops | 60.2 | n.m.; T3: 4.66 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 1 | direct | 2.93 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 2 | direct | 2.88 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 3 | direct | 2.88 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 4 | direct | 4.08 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 5 | loops | 7.02 | n.m.; T3: 1.17 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 6 | loops | 43.7 | n.m.; T3: 1.06 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 7 | loops | 21.9 | n.m.; T3: 4.48 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | CpuSimd512 | f64 | 8 | loops | 47.3 | n.m.; T3: 5.26 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | Gpu | f32 | 1 | direct | 0.32 / 0.17 | 1.00 / 1.00 | 1.03 / 1.07 | – | 1.06 / 1.06 | 2.17 / 3.02 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f32 | 2 | direct | 0.36 / 0.17 | 1.00 / 1.00 | 1.38 / 1.45 | – | 1.31 / 1.37 | 4.38 / 7.67 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f32 | 3 | direct | 0.48 / 0.24 | 1.00 / 1.00 | 2.13 / 1.96 | – | 2.28 / 2.12 | 6.95 / 12.67 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f32 | 4 | direct | 1.45 / 0.51 | 1.00 / 1.00 | 2.00 / 2.55 | – | – | 5.41 / 14.05 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f32 | 5 | loops | 6.50 / 2.91 | n.m.; T3: 0.49 / 0.42 | 1.00 / 1.00 | – | – | 1.55 / 3.00 | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | Gpu | f32 | 6 | loops | 16.0 / 5.48 | n.m.; T3: 0.48 / 0.64 | 1.00 / 1.00 | – | – | 0.73 / 1.75 | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | Gpu | f32 | 7 | loops | 35.6 / 12.3 | n.m.; T3: 0.63 / 0.67 | 1.00 / 1.00 | – | – | 0.37 / 1.12 | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | Gpu | f32 | 8 | loops | 78.0 / 22.1 | n.m.; T3: 0.93 / 0.98 | 1.00 / 1.00 | – | – | 0.27 / 0.78 | cooperative | cooperative 0.27 / 0.78 of loops |
| Jacobian | Tetrahedron | Gpu | f64 | 1 | direct | 0.12 / 0.06 | 1.00 / 1.00 | 1.11 / 1.13 | – | 1.04 / 1.06 | 2.65 / 4.02 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f64 | 2 | direct | 0.18 / 0.07 | 1.00 / 1.00 | 1.74 / 1.73 | – | 1.28 / 1.90 | 4.84 / 10.27 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f64 | 3 | direct | 0.35 / 0.11 | 1.00 / 1.00 | 2.97 / 5.01 | – | 1.91 / 4.12 | 5.54 / 16.95 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f64 | 4 | direct | 1.12 / 0.33 | 1.00 / 1.00 | 3.20 / 3.98 | – | 1.94 / 1.88 | 3.64 / 11.35 | direct | no strategy 10% faster than direct |
| Jacobian | Tetrahedron | Gpu | f64 | 5 | loops | 14.1 / 2.49 | n.m.; T3: 0.28 / 0.33 | 1.00 / 1.00 | – | – | 0.40 / 1.81 | loops | no strategy 10% faster than loops |
| Jacobian | Tetrahedron | Gpu | f64 | 6 | loops | 33.2 / 7.62 | n.m.; T3: 0.48 / 0.50 | 1.00 / 1.00 | – | – | 0.18 / 0.73 | cooperative | cooperative 0.18 / 0.73 of loops |
| Jacobian | Tetrahedron | Gpu | f64 | 7 | loops | 66.7 / 15.3 | n.m.; T3: 0.46 / 0.52 | 1.00 / 1.00 | – | – | 0.11 / 0.47 | cooperative | cooperative 0.11 / 0.47 of loops |
| Jacobian | Tetrahedron | Gpu | f64 | 8 | loops | 146 / 26.2 | n.m.; T3: 0.44 / 0.57 | 1.00 / 1.00 | – | – | 0.10 / 0.34 | cooperative | cooperative 0.10 / 0.34 of loops |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 1 | direct | 1.33 | 1.00 | 1.01 | 1.06 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 2 | direct | 3.46 | 1.00 | 0.96 | 0.51 | – | – | sumfact | sumfact 0.51 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 3 | direct | 10.5 | 1.00 | 0.78 | 0.20 | – | – | sumfact | sumfact 0.20 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 4 | direct | 33.2 | 1.00 | 0.77 | 0.11 | – | – | sumfact | sumfact 0.11 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 5 | direct | 122 | 1.00 | 0.58 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 6 | direct | 320 | 1.00 | 0.51 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 7 | direct | 775 | 1.00 | 0.47 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f32 | 8 | loops | 656 | n.m.; T3: 2.18 | 1.00 | 0.03 | – | – | sumfact | sumfact 0.03 of loops |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 1 | direct | 3.06 | 1.00 | 1.00 | 0.98 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 2 | direct | 10.6 | 1.00 | 0.86 | 0.47 | – | – | sumfact | sumfact 0.47 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 3 | direct | 56.5 | 1.00 | 0.74 | 0.18 | – | – | sumfact | sumfact 0.18 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 4 | direct | 298 | 1.00 | 0.62 | 0.09 | – | – | sumfact | sumfact 0.09 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 5 | direct | 667 | 1.00 | 0.49 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 6 | direct | 1115 | 1.00 | 0.46 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 7 | direct | 1681 | 1.00 | 0.45 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Jacobian | Hexahedron | CpuSimd128 | f64 | 8 | loops | 1080 | n.m.; T3: 2.32 | 1.00 | 0.04 | – | – | sumfact | sumfact 0.04 of loops |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 1 | direct | 1.46 | 1.00 | 0.99 | 1.08 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 2 | direct | 7.77 | 1.00 | 0.95 | 1.01 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 3 | direct | 25.0 | 1.00 | 0.97 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 4 | direct | 120 | 1.00 | 0.48 | 0.17 | – | – | sumfact | sumfact 0.17 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 5 | direct | 471 | 1.00 | 0.19 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 6 | direct | 915 | 1.00 | 0.18 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 7 | direct | 1360 | 1.00 | 0.16 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f32 | 8 | direct | 2106 | 1.00 | 0.15 | 0.01 | – | – | sumfact | sumfact 0.01 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 1 | direct | 3.38 | 1.00 | 1.01 | 1.05 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 2 | direct | 3.89 | 1.00 | 0.96 | 0.95 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 3 | direct | 7.95 | 1.00 | 0.71 | 0.49 | – | – | sumfact | sumfact 0.49 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 4 | direct | 36.4 | 1.00 | 0.31 | 0.12 | – | – | sumfact | sumfact 0.12 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 5 | direct | 118 | 1.00 | 0.19 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 6 | direct | 315 | 1.00 | 0.17 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 7 | direct | 813 | 1.00 | 0.13 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Jacobian | Hexahedron | CpuSimd512 | f64 | 8 | direct | 1787 | 1.00 | 0.35 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 1 | direct | 0.36 / 0.19 | 1.00 / 1.00 | 1.15 / 1.22 | 1.12 / 1.16 | 1.13 / 1.15 | 3.93 / 6.41 | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | Gpu | f32 | 2 | direct | 0.54 / 0.28 | 1.00 / 1.00 | 2.45 / 3.63 | 0.92 / 0.86 | 0.86 / 0.82 | 6.39 / 11.55 | shared | shared 0.86 / 0.82 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 3 | direct | 3.14 / 2.23 | 1.00 / 1.00 | 1.81 / 1.45 | 0.19 / 0.14 | 0.19 / 0.14 | 2.77 / 4.12 | shared | shared 0.19 / 0.14 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 4 | direct | 13.0 / 7.58 | 1.00 / 1.00 | 1.48 / 1.34 | 0.08 / 0.08 | 0.08 / 0.07 | 1.11 / 1.80 | shared | shared 0.08 / 0.07 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 5 | direct | 44.4 / 29.4 | 1.00 / 1.00 | 1.00 / 0.78 | 0.04 / 0.03 | 0.03 / 0.03 | 0.48 / 0.68 | shared | shared 0.03 / 0.03 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 6 | direct | 96.9 / 64.5 | 1.00 / 1.00 | 1.09 / 0.70 | 0.06 / 0.05 | 0.06 / 0.05 | 0.35 / 0.46 | sumfact | sumfact 0.06 / 0.05 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 7 | direct | 257 / 157 | 1.00 / 1.00 | 0.91 / 0.56 | 0.04 / 0.02 | 0.05 / 0.03 | 0.19 / 0.24 | sumfact | sumfact 0.04 / 0.02 of direct |
| Jacobian | Hexahedron | Gpu | f32 | 8 | loops | 438 / 143 | n.m.; T3: 1.66 / 2.61 | 1.00 / 1.00 | 0.03 / 0.04 | 0.04 / 0.04 | 0.18 / 0.46 | sumfact | sumfact 0.03 / 0.04 of loops |
| Jacobian | Hexahedron | Gpu | f64 | 1 | direct | 0.13 / 0.07 | 1.00 / 1.00 | 1.33 / 1.49 | 1.37 / 1.49 | 1.29 / 1.50 | 5.65 / 9.09 | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | Gpu | f64 | 2 | direct | 0.29 / 0.11 | 1.00 / 1.00 | 3.16 / 5.87 | 0.79 / 1.08 | 0.76 / 1.09 | 7.64 / 17.95 | direct | no strategy 10% faster than direct |
| Jacobian | Hexahedron | Gpu | f64 | 3 | direct | 1.40 / 0.68 | 1.00 / 1.00 | 3.42 / 2.42 | 0.22 / 0.29 | 0.64 / 1.11 | 3.52 / 6.96 | sumfact | sumfact 0.22 / 0.29 of direct |
| Jacobian | Hexahedron | Gpu | f64 | 4 | direct | 9.35 / 5.93 | 1.00 / 1.00 | 1.98 / 0.80 | 0.24 / 0.21 | 0.50 / 0.36 | 0.81 / 1.19 | sumfact | sumfact 0.24 / 0.21 of direct |
| Jacobian | Hexahedron | Gpu | f64 | 5 | direct | 27.7 / 10.1 | 1.00 / 1.00 | 1.90 / 1.26 | 0.31 / 0.26 | 0.31 / 0.25 | 0.42 / 1.05 | shared | shared 0.31 / 0.25 of direct |
| Jacobian | Hexahedron | Gpu | f64 | 6 | direct | 470 / 72.9 | 1.00 / 1.00 | 0.32 / 0.42 | 0.04 / 0.05 | 0.04 / 0.05 | 0.04 / 0.22 | sumfact | sumfact 0.04 / 0.05 of direct |
| Jacobian | Hexahedron | Gpu | f64 | 7 | direct | 217 / 38.9 | 1.00 / 1.00 | 1.66 / 1.78 | 0.15 / 0.14 | 0.19 / 0.17 | 0.13 / 0.59 | sumfact | closest: sumfact 0.15 / 0.14 of direct, 20% slower than the fastest at one size |
| Jacobian | Hexahedron | Gpu | f64 | 8 | loops | 669 / 139 | n.m.; T3: 3.70 / 2.53 | 1.00 / 1.00 | 0.07 / 0.05 | 0.10 / 0.07 | 0.07 / 0.25 | sumfact | sumfact 0.07 / 0.05 of loops |
| Jacobian | Prism | CpuSimd128 | f32 | 1 | direct | 1.23 | 1.00 | 0.97 | 0.99 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd128 | f32 | 2 | direct | 2.38 | 1.00 | 0.94 | 0.70 | – | – | sumfact | sumfact 0.70 of direct |
| Jacobian | Prism | CpuSimd128 | f32 | 3 | direct | 5.69 | 1.00 | 0.83 | 0.41 | – | – | sumfact | sumfact 0.41 of direct |
| Jacobian | Prism | CpuSimd128 | f32 | 4 | direct | 43.9 | 1.00 | 1.01 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Jacobian | Prism | CpuSimd128 | f32 | 5 | loops | 26.4 | n.m.; T3: 1.45 | 1.00 | 0.26 | – | – | sumfact | sumfact 0.26 of loops |
| Jacobian | Prism | CpuSimd128 | f32 | 6 | loops | 56.1 | n.m.; T3: 1.87 | 1.00 | 0.20 | – | – | sumfact | sumfact 0.20 of loops |
| Jacobian | Prism | CpuSimd128 | f32 | 7 | loops | 107 | n.m.; T3: 2.45 | 1.00 | 0.17 | – | – | sumfact | sumfact 0.17 of loops |
| Jacobian | Prism | CpuSimd128 | f32 | 8 | loops | 335 | n.m.; T3: 2.12 | 1.00 | 0.13 | – | – | sumfact | sumfact 0.13 of loops |
| Jacobian | Prism | CpuSimd128 | f64 | 1 | direct | 3.11 | 1.00 | 1.03 | 1.01 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd128 | f64 | 2 | direct | 7.25 | 1.00 | 0.90 | 0.60 | – | – | sumfact | sumfact 0.60 of direct |
| Jacobian | Prism | CpuSimd128 | f64 | 3 | direct | 34.4 | 1.00 | 0.83 | 0.35 | – | – | sumfact | sumfact 0.35 of direct |
| Jacobian | Prism | CpuSimd128 | f64 | 4 | direct | 262 | 1.00 | 1.06 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Jacobian | Prism | CpuSimd128 | f64 | 5 | loops | 133 | n.m.; T3: 1.57 | 1.00 | 0.27 | – | – | sumfact | sumfact 0.27 of loops |
| Jacobian | Prism | CpuSimd128 | f64 | 6 | loops | 203 | n.m.; T3: 1.96 | 1.00 | 0.22 | – | – | sumfact | sumfact 0.22 of loops |
| Jacobian | Prism | CpuSimd128 | f64 | 7 | loops | 296 | n.m.; T3: 2.59 | 1.00 | 0.19 | – | – | sumfact | sumfact 0.19 of loops |
| Jacobian | Prism | CpuSimd128 | f64 | 8 | loops | 605 | n.m.; T3: 2.21 | 1.00 | 0.14 | – | – | sumfact | sumfact 0.14 of loops |
| Jacobian | Prism | CpuSimd512 | f32 | 1 | direct | 1.42 | 1.00 | 1.01 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd512 | f32 | 2 | direct | 5.74 | 1.00 | 1.00 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd512 | f32 | 3 | direct | 22.2 | 1.00 | 1.03 | 0.70 | – | – | sumfact | sumfact 0.70 of direct |
| Jacobian | Prism | CpuSimd512 | f32 | 4 | direct | 125 | 1.00 | 0.99 | 0.28 | – | – | sumfact | sumfact 0.28 of direct |
| Jacobian | Prism | CpuSimd512 | f32 | 5 | loops | 40.0 | n.m.; T3: 2.99 | 1.00 | 0.53 | – | – | sumfact | sumfact 0.53 of loops |
| Jacobian | Prism | CpuSimd512 | f32 | 6 | loops | 75.2 | n.m.; T3: 5.41 | 1.00 | 0.36 | – | – | sumfact | sumfact 0.36 of loops |
| Jacobian | Prism | CpuSimd512 | f32 | 7 | loops | 105 | n.m.; T3: 6.83 | 1.00 | 0.32 | – | – | sumfact | sumfact 0.32 of loops |
| Jacobian | Prism | CpuSimd512 | f32 | 8 | direct | 1169 | 1.00 | 0.16 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| Jacobian | Prism | CpuSimd512 | f64 | 1 | direct | 3.43 | 1.00 | 0.99 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd512 | f64 | 2 | direct | 3.14 | 1.00 | 1.00 | 1.00 | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Prism | CpuSimd512 | f64 | 3 | direct | 4.98 | 1.00 | 0.99 | 0.74 | – | – | sumfact | sumfact 0.74 of direct |
| Jacobian | Prism | CpuSimd512 | f64 | 4 | direct | 31.3 | 1.00 | 0.85 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Jacobian | Prism | CpuSimd512 | f64 | 5 | loops | 12.3 | n.m.; T3: 4.84 | 1.00 | 0.52 | – | – | sumfact | sumfact 0.52 of loops |
| Jacobian | Prism | CpuSimd512 | f64 | 6 | loops | 31.2 | n.m.; T3: 5.16 | 1.00 | 0.37 | – | – | sumfact | sumfact 0.37 of loops |
| Jacobian | Prism | CpuSimd512 | f64 | 7 | loops | 46.9 | n.m.; T3: 7.19 | 1.00 | 0.37 | – | – | sumfact | sumfact 0.37 of loops |
| Jacobian | Prism | CpuSimd512 | f64 | 8 | direct | 722 | 1.00 | 0.18 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| Jacobian | Prism | Gpu | f32 | 1 | direct | 0.35 / 0.18 | 1.00 / 1.00 | 1.12 / 1.12 | 1.04 / 1.06 | 0.99 / 1.00 | 2.21 / 3.02 | direct | no strategy 10% faster than direct |
| Jacobian | Prism | Gpu | f32 | 2 | direct | 0.45 / 0.23 | 1.00 / 1.00 | 1.62 / 2.04 | 0.88 / 0.90 | 0.89 / 0.88 | 7.09 / 13.07 | shared | shared 0.89 / 0.88 of direct |
| Jacobian | Prism | Gpu | f32 | 3 | direct | 0.85 / 0.45 | 1.00 / 1.00 | 3.12 / 3.90 | 0.71 / 0.71 | 0.79 / 0.79 | 9.52 / 16.20 | sumfact | sumfact 0.71 / 0.71 of direct |
| Jacobian | Prism | Gpu | f32 | 4 | direct | 4.83 / 2.89 | 1.00 / 1.00 | 1.67 / 1.61 | 0.25 / 0.23 | 0.22 / 0.22 | 2.18 / 3.20 | shared | shared 0.22 / 0.22 of direct |
| Jacobian | Prism | Gpu | f32 | 5 | direct | 15.0 / 8.23 | 1.00 / 1.00 | 1.30 / 1.59 | 0.15 / 0.14 | 0.15 / 0.16 | 0.90 / 1.64 | sumfact | sumfact 0.15 / 0.14 of direct |
| Jacobian | Prism | Gpu | f32 | 6 | direct | 53.9 / 24.6 | 1.00 / 1.00 | 0.85 / 1.08 | 0.07 / 0.08 | 0.06 / 0.08 | 0.36 / 0.78 | shared | shared 0.06 / 0.08 of direct |
| Jacobian | Prism | Gpu | f32 | 7 | direct | 105 / 55.7 | 1.00 / 1.00 | 0.86 / 0.85 | 0.07 / 0.06 | 0.06 / 0.05 | 0.24 / 0.46 | shared | shared 0.06 / 0.05 of direct |
| Jacobian | Prism | Gpu | f32 | 8 | direct | 409 / 207 | 1.00 / 1.00 | 0.43 / 0.38 | 0.02 / 0.02 | – | 0.11 / 0.17 | sumfact | sumfact 0.02 / 0.02 of direct |
| Jacobian | Prism | Gpu | f64 | 1 | direct | 0.13 / 0.07 | 1.00 / 1.00 | 1.08 / 1.27 | 0.95 / 1.01 | 0.95 / 1.00 | 2.81 / 4.14 | direct | no strategy 10% faster than direct |
| Jacobian | Prism | Gpu | f64 | 2 | direct | 0.28 / 0.09 | 1.00 / 1.00 | 2.02 / 4.85 | 0.61 / 0.83 | 0.58 / 0.80 | 6.70 / 19.38 | shared | shared 0.58 / 0.80 of direct |
| Jacobian | Prism | Gpu | f64 | 3 | direct | 0.76 / 0.26 | 1.00 / 1.00 | 3.02 / 4.15 | 0.35 / 0.57 | 0.29 / 0.46 | 5.51 / 15.08 | shared | shared 0.29 / 0.46 of direct |
| Jacobian | Prism | Gpu | f64 | 4 | direct | 4.65 / 3.05 | 1.00 / 1.00 | 2.37 / 0.97 | 0.14 / 0.11 | 0.11 / 0.11 | 1.19 / 1.71 | shared | shared 0.11 / 0.11 of direct |
| Jacobian | Prism | Gpu | f64 | 5 | direct | 14.8 / 5.65 | 1.00 / 1.00 | 2.27 / 1.40 | 0.11 / 0.10 | 0.09 / 0.10 | 0.59 / 1.26 | shared | shared 0.09 / 0.10 of direct |
| Jacobian | Prism | Gpu | f64 | 6 | direct | 37.2 / 10.5 | 1.00 / 1.00 | 2.41 / 1.81 | 0.09 / 0.07 | 0.07 / 0.07 | 0.33 / 0.97 | shared | shared 0.07 / 0.07 of direct |
| Jacobian | Prism | Gpu | f64 | 7 | direct | 266 / 57.5 | 1.00 / 1.00 | 0.66 / 0.71 | 0.02 / 0.02 | 0.02 / 0.02 | 0.05 / 0.24 | shared | shared 0.02 / 0.02 of direct |
| Jacobian | Prism | Gpu | f64 | 8 | direct | 1120 / 212 | 1.00 / 1.00 | 0.30 / 0.37 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.09 | shared | shared 0.01 / 0.01 of direct |
| Jacobian | Pyramid | CpuSimd128 | f32 | 1 | direct | 1.12 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd128 | f32 | 2 | direct | 1.86 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd128 | f32 | 3 | direct | 4.33 | 1.00 | 0.88 | – | – | – | loops | loops 0.88 of direct |
| Jacobian | Pyramid | CpuSimd128 | f64 | 1 | direct | 2.58 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd128 | f64 | 2 | direct | 5.76 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd128 | f64 | 3 | direct | 25.7 | 1.00 | 0.81 | – | – | – | loops | loops 0.81 of direct |
| Jacobian | Pyramid | CpuSimd512 | f32 | 1 | direct | 1.28 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd512 | f32 | 2 | direct | 4.58 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd512 | f32 | 3 | direct | 13.5 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd512 | f64 | 1 | direct | 3.06 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd512 | f64 | 2 | direct | 2.89 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | CpuSimd512 | f64 | 3 | direct | 3.37 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f32 | 1 | direct | 0.31 / 0.17 | 1.00 / 1.00 | 1.14 / 1.11 | – | 1.17 / 1.08 | 2.40 / 3.15 | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f32 | 2 | direct | 0.41 / 0.21 | 1.00 / 1.00 | 1.49 / 1.52 | – | 1.42 / 1.57 | 3.87 / 6.36 | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f32 | 3 | direct | 0.70 / 0.32 | 1.00 / 1.00 | 2.96 / 3.37 | – | 3.07 / 3.52 | 5.04 / 10.01 | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f64 | 1 | direct | 0.12 / 0.07 | 1.00 / 1.00 | 1.14 / 1.19 | – | 1.12 / 1.14 | 2.92 / 4.21 | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f64 | 2 | direct | 0.20 / 0.08 | 1.00 / 1.00 | 2.13 / 3.13 | – | 1.52 / 2.95 | 4.54 / 10.23 | direct | no strategy 10% faster than direct |
| Jacobian | Pyramid | Gpu | f64 | 3 | direct | 0.37 / 0.13 | 1.00 / 1.00 | 5.00 / 6.06 | – | 2.61 / 4.98 | 6.27 / 16.64 | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 1 | direct | 0.10 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 2 | direct | 0.18 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 3 | direct | 0.11 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 4 | direct | 0.15 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 5 | direct | 0.15 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 6 | direct | 0.16 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 7 | direct | 0.18 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f32 | 8 | direct | 0.19 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 1 | direct | 0.22 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 2 | direct | 0.24 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 3 | direct | 0.24 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 4 | direct | 0.24 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 5 | direct | 0.30 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 6 | direct | 0.33 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 7 | direct | 0.30 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd128 | f64 | 8 | direct | 0.36 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 1 | direct | 0.10 | 1.00 | 1.09 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 2 | direct | 0.10 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 3 | direct | 0.15 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 4 | direct | 0.14 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 5 | direct | 0.20 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 6 | direct | 0.20 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f32 | 7 | direct | 0.27 | 1.00 | 0.79 | – | – | – | loops | loops 0.79 of direct |
| Normals | Interval | CpuSimd512 | f32 | 8 | direct | 0.27 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 1 | direct | 0.28 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 2 | direct | 0.27 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 3 | direct | 0.31 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 4 | direct | 0.32 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 5 | direct | 0.23 | 1.00 | 1.51 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 6 | direct | 0.22 | 1.00 | 1.59 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | CpuSimd512 | f64 | 7 | direct | 0.39 | 1.00 | 0.82 | – | – | – | loops | loops 0.82 of direct |
| Normals | Interval | CpuSimd512 | f64 | 8 | direct | 0.36 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 1 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.04 / 1.05 | – | 1.07 / 1.06 | 1.15 / 1.33 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 2 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.03 / 1.07 | – | 1.07 / 1.04 | 1.20 / 1.47 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 3 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.08 / 1.10 | – | 1.08 / 1.10 | 1.45 / 1.88 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 4 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.09 / 1.11 | – | 1.10 / 1.08 | 1.59 / 2.08 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 5 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.11 / 1.16 | – | 1.11 / 1.12 | 1.57 / 2.00 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 6 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.14 / 1.21 | – | 1.13 / 1.17 | 1.55 / 2.13 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 7 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.14 / 1.20 | – | 1.13 / 1.19 | 2.21 / 3.70 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f32 | 8 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.16 / 1.26 | – | 1.15 / 1.20 | 2.43 / 4.18 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 1 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.16 / 1.03 | – | 1.15 / 1.05 | 1.69 / 1.79 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.03 / 1.05 | – | 1.05 / 1.02 | 1.52 / 2.00 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 3 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.10 / 1.08 | – | 1.13 / 1.06 | 2.35 / 3.57 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 4 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.15 / 1.11 | – | 1.14 / 1.07 | 2.48 / 4.02 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 5 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.14 / 1.16 | – | 1.14 / 1.13 | 2.43 / 4.02 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 6 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.15 / 1.19 | – | 1.12 / 1.20 | 2.33 / 4.22 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 7 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.16 / 1.29 | – | 1.13 / 1.24 | 4.09 / 8.91 | direct | no strategy 10% faster than direct |
| Normals | Interval | Gpu | f64 | 8 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.24 / 1.38 | – | 1.16 / 1.23 | 4.31 / 9.46 | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 1 | direct | 0.18 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 2 | direct | 0.30 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 3 | direct | 0.42 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 4 | direct | 0.59 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 5 | direct | 0.84 | 1.00 | 0.86 | – | – | – | loops | loops 0.86 of direct |
| Normals | Triangle | CpuSimd128 | f32 | 6 | direct | 1.20 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f32 | 7 | direct | 1.64 | 1.00 | 0.88 | – | – | – | loops | loops 0.88 of direct |
| Normals | Triangle | CpuSimd128 | f32 | 8 | direct | 2.01 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| Normals | Triangle | CpuSimd128 | f64 | 1 | direct | 0.40 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f64 | 2 | direct | 0.53 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f64 | 3 | direct | 0.93 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f64 | 4 | direct | 1.49 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f64 | 5 | direct | 2.66 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd128 | f64 | 6 | direct | 4.31 | 1.00 | 0.85 | – | – | – | loops | loops 0.85 of direct |
| Normals | Triangle | CpuSimd128 | f64 | 7 | direct | 8.16 | 1.00 | 0.84 | – | – | – | loops | loops 0.84 of direct |
| Normals | Triangle | CpuSimd128 | f64 | 8 | direct | 11.0 | 1.00 | 0.81 | – | – | – | loops | loops 0.81 of direct |
| Normals | Triangle | CpuSimd512 | f32 | 1 | direct | 0.15 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 2 | direct | 0.35 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 3 | direct | 0.78 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 4 | direct | 1.71 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 5 | direct | 2.43 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 6 | direct | 3.64 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 7 | direct | 6.37 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f32 | 8 | direct | 5.52 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 1 | direct | 0.43 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 2 | direct | 0.49 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 3 | direct | 0.56 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 4 | direct | 0.57 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 5 | direct | 0.73 | 1.00 | 0.83 | – | – | – | loops | loops 0.83 of direct |
| Normals | Triangle | CpuSimd512 | f64 | 6 | direct | 1.28 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| Normals | Triangle | CpuSimd512 | f64 | 7 | direct | 1.35 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| Normals | Triangle | CpuSimd512 | f64 | 8 | direct | 1.70 | 1.00 | 0.80 | – | – | – | loops | loops 0.80 of direct |
| Normals | Triangle | Gpu | f32 | 1 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 1.08 / 1.10 | – | 1.05 / 1.02 | 1.33 / 1.64 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 2 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.12 / 1.21 | – | 1.22 / 1.11 | 2.05 / 2.85 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 3 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.29 / 1.46 | – | 1.21 / 1.30 | 4.66 / 8.53 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 4 | direct | 0.10 / 0.05 | 1.00 / 1.00 | 1.60 / 1.58 | – | 1.49 / 1.37 | 4.42 / 7.34 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 5 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 2.35 / 4.76 | – | 2.28 / 3.30 | 7.85 / 16.67 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 6 | direct | 0.15 / 0.07 | 1.00 / 1.00 | 2.81 / 4.97 | – | 2.53 / 4.83 | 6.88 / 12.79 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 7 | direct | 0.19 / 0.09 | 1.00 / 1.00 | 3.74 / 5.39 | – | 3.12 / 4.65 | 10.82 / 22.17 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f32 | 8 | direct | 0.34 / 0.15 | 1.00 / 1.00 | 3.02 / 4.27 | – | 2.50 / 3.59 | 6.32 / 14.45 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.05 / 1.06 | – | 1.26 / 1.18 | 1.96 / 2.77 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.09 / 1.24 | – | 1.21 / 1.32 | 3.99 / 7.36 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 3 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 1.27 / 1.45 | – | 1.75 / 1.93 | 8.16 / 16.07 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 4 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.42 / 1.65 | – | 2.39 / 4.82 | 6.69 / 14.50 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 5 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 5.85 / 3.50 | – | 6.69 / 4.62 | 14.26 / 20.02 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 6 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 2.46 / 4.17 | – | 6.11 / 5.95 | 9.87 / 18.88 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 7 | direct | 0.16 / 0.05 | 1.00 / 1.00 | 1.90 / 4.77 | – | 4.99 / 7.31 | 8.29 / 24.66 | direct | no strategy 10% faster than direct |
| Normals | Triangle | Gpu | f64 | 8 | direct | 0.29 / 0.09 | 1.00 / 1.00 | 1.54 / 3.17 | – | 4.37 / 5.04 | 4.97 / 14.62 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 0.23 | 1.00 | 1.01 | 1.25 | – | – | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 0.41 | 1.00 | 1.00 | 0.84 | – | – | sumfact | sumfact 0.84 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 0.60 | 1.00 | 0.98 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 1.05 | 1.00 | 0.83 | 0.52 | – | – | sumfact | sumfact 0.52 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 2.17 | 1.00 | 0.88 | 0.37 | – | – | sumfact | sumfact 0.37 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 1.83 | 1.00 | 0.93 | 0.40 | – | – | sumfact | sumfact 0.40 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 2.70 | 1.00 | 0.85 | 0.34 | – | – | sumfact | sumfact 0.34 of direct |
| Normals | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 3.59 | 1.00 | 0.78 | 0.24 | – | – | sumfact | sumfact 0.24 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 0.46 | 1.00 | 1.00 | 1.13 | – | – | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 0.73 | 1.00 | 0.96 | 0.90 | – | – | sumfact | sumfact 0.90 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 1.31 | 1.00 | 0.91 | 0.80 | – | – | sumfact | sumfact 0.80 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 2.41 | 1.00 | 0.91 | 0.50 | – | – | sumfact | sumfact 0.50 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 7.97 | 1.00 | 0.93 | 0.30 | – | – | sumfact | sumfact 0.30 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 8.83 | 1.00 | 0.78 | 0.29 | – | – | sumfact | sumfact 0.29 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 15.0 | 1.00 | 0.76 | 0.22 | – | – | sumfact | sumfact 0.22 of direct |
| Normals | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 24.4 | 1.00 | 0.76 | 0.19 | – | – | sumfact | sumfact 0.19 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 0.18 | 1.00 | 0.99 | 1.11 | – | – | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 0.49 | 1.00 | 0.97 | 0.88 | – | – | sumfact | sumfact 0.88 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 1.08 | 1.00 | 0.84 | 0.79 | – | – | sumfact | sumfact 0.79 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 2.17 | 1.00 | 0.96 | 0.79 | – | – | sumfact | sumfact 0.79 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 4.21 | 1.00 | 0.93 | 0.58 | – | – | sumfact | sumfact 0.58 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 5.24 | 1.00 | 0.88 | 0.62 | – | – | sumfact | sumfact 0.62 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 7.61 | 1.00 | 0.87 | 0.58 | – | – | sumfact | sumfact 0.58 of direct |
| Normals | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 9.40 | 1.00 | 0.86 | 0.51 | – | – | sumfact | sumfact 0.51 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 0.44 | 1.00 | 1.06 | 1.15 | – | – | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 0.57 | 1.00 | 0.95 | 1.06 | – | – | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 0.89 | 1.00 | 0.87 | 0.81 | – | – | sumfact | sumfact 0.81 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 0.96 | 1.00 | 0.98 | 0.74 | – | – | sumfact | sumfact 0.74 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 1.50 | 1.00 | 0.81 | 0.46 | – | – | sumfact | sumfact 0.46 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 1.55 | 1.00 | 0.86 | 0.60 | – | – | sumfact | sumfact 0.60 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 1.99 | 1.00 | 0.81 | 0.42 | – | – | sumfact | sumfact 0.42 of direct |
| Normals | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 3.85 | 1.00 | 0.76 | 0.30 | – | – | sumfact | sumfact 0.30 of direct |
| Normals | Quadrilateral | Gpu | f32 | 1 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.08 / 1.12 | 1.11 / 1.17 | 1.08 / 1.09 | 1.89 / 2.35 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f32 | 2 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.15 / 1.32 | 1.13 / 1.19 | 1.05 / 1.10 | 4.12 / 7.74 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f32 | 3 | direct | 0.11 / 0.05 | 1.00 / 1.00 | 1.46 / 1.98 | 0.97 / 1.07 | 0.95 / 1.01 | 8.11 / 16.42 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f32 | 4 | direct | 0.13 / 0.06 | 1.00 / 1.00 | 2.79 / 4.16 | 0.92 / 1.02 | 0.91 / 0.97 | 7.47 / 14.78 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f32 | 5 | direct | 0.16 / 0.09 | 1.00 / 1.00 | 3.64 / 5.15 | 0.90 / 0.86 | 0.85 / 0.84 | 13.30 / 21.37 | shared | shared 0.85 / 0.84 of direct |
| Normals | Quadrilateral | Gpu | f32 | 6 | direct | 0.26 / 0.15 | 1.00 / 1.00 | 3.33 / 4.09 | 0.61 / 0.57 | 0.59 / 0.55 | 8.68 / 14.09 | shared | shared 0.59 / 0.55 of direct |
| Normals | Quadrilateral | Gpu | f32 | 7 | direct | 0.59 / 0.35 | 1.00 / 1.00 | 2.79 / 2.74 | 0.37 / 0.33 | 0.34 / 0.31 | 4.14 / 7.32 | shared | shared 0.34 / 0.31 of direct |
| Normals | Quadrilateral | Gpu | f32 | 8 | direct | 1.39 / 0.89 | 1.00 / 1.00 | 1.95 / 1.42 | 0.17 / 0.14 | 0.16 / 0.13 | 1.98 / 2.99 | shared | shared 0.16 / 0.13 of direct |
| Normals | Quadrilateral | Gpu | f64 | 1 | direct | 0.03 / 0.02 | 1.00 / 1.00 | 1.11 / 1.10 | 1.51 / 1.64 | 1.57 / 1.63 | 3.68 / 6.34 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f64 | 2 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 1.04 / 1.29 | 1.18 / 1.52 | 1.14 / 1.48 | 7.15 / 15.72 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f64 | 3 | direct | 0.06 / 0.03 | 1.00 / 1.00 | 1.10 / 2.11 | 0.88 / 0.94 | 0.84 / 0.92 | 11.07 / 21.32 | direct | no strategy 10% faster than direct |
| Normals | Quadrilateral | Gpu | f64 | 4 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 1.36 / 4.68 | 0.80 / 0.80 | 0.74 / 0.77 | 9.08 / 19.71 | shared | shared 0.74 / 0.77 of direct |
| Normals | Quadrilateral | Gpu | f64 | 5 | direct | 0.13 / 0.05 | 1.00 / 1.00 | 1.49 / 5.53 | 0.48 / 0.73 | 0.45 / 0.71 | 10.20 / 27.54 | shared | shared 0.45 / 0.71 of direct |
| Normals | Quadrilateral | Gpu | f64 | 6 | direct | 0.24 / 0.10 | 1.00 / 1.00 | 1.41 / 3.26 | 0.36 / 0.46 | 0.33 / 0.45 | 6.28 / 13.74 | shared | shared 0.33 / 0.45 of direct |
| Normals | Quadrilateral | Gpu | f64 | 7 | direct | 3.57 / 1.59 | 1.00 / 1.00 | 0.16 / 0.28 | 0.05 / 0.08 | 0.04 / 0.10 | 0.47 / 1.01 | shared | closest: shared 0.04 / 0.10 of direct, 15% slower than the fastest at one size |
| Normals | Quadrilateral | Gpu | f64 | 8 | direct | 6.47 / 2.06 | 1.00 / 1.00 | 0.13 / 0.26 | 0.04 / 0.07 | 0.05 / 0.08 | 0.30 / 0.90 | sumfact | sumfact 0.04 / 0.07 of direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 1 | direct | 0.08 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 2 | direct | 0.08 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 3 | direct | 0.08 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 4 | direct | 0.09 | 1.00 | 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 5 | direct | 0.17 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 6 | direct | 0.12 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 7 | direct | 0.18 | 1.00 | 0.79 | – | – | – | loops | loops 0.79 of direct |
| FacetNormals | Interval | CpuSimd128 | f32 | 8 | direct | 0.16 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 1 | direct | 0.15 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 2 | direct | 0.15 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 3 | direct | 0.16 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 4 | direct | 0.18 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 5 | direct | 0.23 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 6 | direct | 0.25 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 7 | direct | 0.31 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd128 | f64 | 8 | direct | 0.33 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 1 | direct | 0.07 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 2 | direct | 0.09 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 3 | direct | 0.09 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 4 | direct | 0.09 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 5 | direct | 0.09 | 1.00 | 0.84 | – | – | – | loops | loops 0.84 of direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 6 | direct | 0.10 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 7 | direct | 0.10 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f32 | 8 | direct | 0.14 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 1 | direct | 0.25 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 2 | direct | 0.25 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 3 | direct | 0.25 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 4 | direct | 0.25 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 5 | direct | 0.21 | 1.00 | 1.27 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 6 | direct | 0.35 | 1.00 | 0.80 | – | – | – | loops | loops 0.80 of direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 7 | direct | 0.33 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | CpuSimd512 | f64 | 8 | direct | 0.33 | 1.00 | 1.06 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 1 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.04 / 1.13 | – | 1.08 / 1.15 | 1.18 / 1.41 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 2 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.07 / 1.17 | – | 1.09 / 1.19 | 1.25 / 1.59 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 3 | direct | 0.07 / 0.02 | 1.00 / 1.00 | 1.05 / 1.21 | – | 1.09 / 1.23 | 1.45 / 2.28 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 4 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.07 / 1.14 | – | 1.08 / 1.15 | 1.54 / 2.48 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 5 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.09 / 1.18 | – | 1.11 / 1.18 | 1.52 / 2.29 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 6 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.08 / 1.22 | – | 1.09 / 1.21 | 1.47 / 2.18 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 7 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.09 / 1.22 | – | 1.12 / 1.23 | 2.01 / 3.34 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f32 | 8 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.07 / 1.16 | – | 1.14 / 1.11 | 2.07 / 3.54 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 1 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.05 / 1.21 | – | 1.07 / 1.26 | 1.33 / 1.91 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 2 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.08 / 1.17 | – | 1.11 / 1.20 | 1.38 / 2.17 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 3 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.09 / 1.17 | – | 1.13 / 1.20 | 1.67 / 2.54 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 4 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.10 / 1.09 | – | 1.10 / 1.10 | 1.80 / 2.89 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 5 | direct | 0.02 / 0.01 | 1.00 / 1.00 | 1.13 / 1.12 | – | 1.16 / 1.13 | 1.81 / 2.62 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 6 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.12 / 1.14 | – | 1.15 / 1.16 | 1.76 / 2.43 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 7 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.14 / 1.16 | – | 1.20 / 1.24 | 2.43 / 3.60 | direct | no strategy 10% faster than direct |
| FacetNormals | Interval | Gpu | f64 | 8 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.07 / 1.10 | – | 1.04 / 1.07 | 2.58 / 4.04 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 1 | direct | 0.16 | 1.00 | 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 2 | direct | 0.24 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 3 | direct | 0.39 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 4 | direct | 0.49 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 5 | direct | 0.66 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 6 | direct | 0.89 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 7 | direct | 2.10 | 1.00 | 0.87 | – | – | – | loops | loops 0.87 of direct |
| FacetNormals | Triangle | CpuSimd128 | f32 | 8 | direct | 1.40 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 1 | direct | 0.34 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 2 | direct | 0.45 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 3 | direct | 0.61 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 4 | direct | 0.91 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 5 | direct | 1.25 | 1.00 | 1.14 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 6 | direct | 1.66 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 7 | direct | 4.22 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd128 | f64 | 8 | direct | 3.95 | 1.00 | 1.15 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 1 | direct | 0.18 | 1.00 | 1.30 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 2 | direct | 0.32 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 3 | direct | 0.51 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 4 | direct | 0.83 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 5 | direct | 1.21 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 6 | direct | 1.82 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 7 | direct | 2.48 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f32 | 8 | direct | 3.00 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 1 | direct | 0.38 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 2 | direct | 0.38 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 3 | direct | 0.46 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 4 | direct | 0.52 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 5 | direct | 0.59 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 6 | direct | 0.69 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 7 | direct | 1.24 | 1.00 | 0.80 | – | – | – | loops | loops 0.80 of direct |
| FacetNormals | Triangle | CpuSimd512 | f64 | 8 | direct | 1.03 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 1 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.04 / 1.17 | – | 1.05 / 1.14 | 1.38 / 1.91 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 2 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.08 / 1.25 | – | 1.09 / 1.20 | 2.23 / 3.73 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 3 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.18 / 1.41 | – | 1.18 / 1.34 | 3.95 / 8.22 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 4 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.32 / 1.41 | – | 1.26 / 1.34 | 3.83 / 6.93 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 5 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.78 / 2.36 | – | 1.62 / 2.26 | 6.97 / 13.78 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 6 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 2.17 / 4.03 | – | 2.03 / 3.92 | 6.03 / 13.07 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 7 | direct | 0.15 / 0.06 | 1.00 / 1.00 | 2.41 / 4.79 | – | 2.21 / 4.50 | 9.98 / 24.09 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f32 | 8 | direct | 0.17 / 0.08 | 1.00 / 1.00 | 2.91 / 4.72 | – | 2.63 / 4.56 | 9.55 / 19.16 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.19 / 1.19 | – | 1.21 / 1.19 | 2.27 / 4.43 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.30 / 1.38 | – | 1.30 / 1.34 | 5.11 / 12.37 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 3 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.31 / 1.48 | – | 1.21 / 1.32 | 9.42 / 22.97 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 4 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 1.40 / 1.79 | – | 1.42 / 1.54 | 8.68 / 17.87 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 5 | direct | 0.34 / 0.02 | 1.00 / 1.00 | 1.08 / 4.01 | – | 0.99 / 2.69 | 2.85 / 31.50 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 6 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.90 / 3.97 | – | 1.48 / 3.90 | 11.47 / 23.29 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 7 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 2.35 / 4.82 | – | 1.70 / 4.26 | 17.30 / 35.50 | direct | no strategy 10% faster than direct |
| FacetNormals | Triangle | Gpu | f64 | 8 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 2.25 / 5.21 | – | 1.63 / 4.98 | 13.06 / 33.18 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 0.19 | 1.00 | 1.04 | 1.39 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 0.39 | 1.00 | 0.98 | 0.81 | – | – | sumfact | sumfact 0.81 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 0.53 | 1.00 | 0.99 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 1.59 | 1.00 | 0.97 | 0.39 | – | – | sumfact | sumfact 0.39 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 1.02 | 1.00 | 0.98 | 0.47 | – | – | sumfact | sumfact 0.47 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 1.53 | 1.00 | 0.95 | 0.37 | – | – | sumfact | sumfact 0.37 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 1.92 | 1.00 | 0.95 | 0.32 | – | – | sumfact | sumfact 0.32 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 5.78 | 1.00 | 0.96 | 0.18 | – | – | sumfact | sumfact 0.18 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 0.35 | 1.00 | 1.00 | 1.25 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 0.56 | 1.00 | 1.01 | 0.97 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 1.05 | 1.00 | 0.99 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 2.78 | 1.00 | 0.97 | 0.42 | – | – | sumfact | sumfact 0.42 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 2.05 | 1.00 | 0.91 | 0.45 | – | – | sumfact | sumfact 0.45 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 3.92 | 1.00 | 0.83 | 0.32 | – | – | sumfact | sumfact 0.32 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 5.87 | 1.00 | 0.82 | 0.27 | – | – | sumfact | sumfact 0.27 of direct |
| FacetNormals | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 11.9 | 1.00 | 0.92 | 0.21 | – | – | sumfact | sumfact 0.21 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 0.21 | 1.00 | 1.03 | 1.31 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 0.49 | 1.00 | 0.93 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 0.93 | 1.00 | 0.96 | 0.68 | – | – | sumfact | sumfact 0.68 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 1.95 | 1.00 | 0.99 | 0.60 | – | – | sumfact | sumfact 0.60 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 2.42 | 1.00 | 0.92 | 0.65 | – | – | sumfact | sumfact 0.65 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 3.04 | 1.00 | 0.94 | 0.60 | – | – | sumfact | sumfact 0.60 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 4.33 | 1.00 | 0.91 | 0.53 | – | – | sumfact | sumfact 0.53 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 6.24 | 1.00 | 0.88 | 0.42 | – | – | sumfact | sumfact 0.42 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 0.39 | 1.00 | 1.02 | 1.15 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 0.44 | 1.00 | 1.11 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 0.51 | 1.00 | 1.05 | 0.88 | – | – | sumfact | sumfact 0.88 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 0.75 | 1.00 | 0.91 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 0.79 | 1.00 | 1.20 | 0.86 | – | – | sumfact | sumfact 0.86 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 1.12 | 1.00 | 0.95 | 0.65 | – | – | sumfact | sumfact 0.65 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 1.58 | 1.00 | 0.92 | 0.52 | – | – | sumfact | sumfact 0.52 of direct |
| FacetNormals | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 2.10 | 1.00 | 0.79 | 0.39 | – | – | sumfact | sumfact 0.39 of direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 1 | direct | 0.07 / 0.03 | 1.00 / 1.00 | 1.20 / 1.21 | 1.13 / 1.27 | 1.22 / 1.21 | 2.11 / 3.45 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 2 | direct | 0.08 / 0.03 | 1.00 / 1.00 | 1.18 / 1.31 | 1.16 / 1.22 | 1.06 / 1.12 | 4.02 / 8.01 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 3 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.30 / 1.43 | 1.01 / 1.18 | 0.99 / 1.15 | 6.84 / 15.34 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 4 | direct | 0.11 / 0.05 | 1.00 / 1.00 | 1.99 / 2.98 | 0.94 / 1.04 | 0.91 / 0.94 | 6.54 / 12.87 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 5 | direct | 0.15 / 0.06 | 1.00 / 1.00 | 2.26 / 4.78 | 0.95 / 1.16 | 0.93 / 1.11 | 9.83 / 23.83 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 6 | direct | 0.21 / 0.10 | 1.00 / 1.00 | 2.57 / 4.04 | 0.73 / 0.72 | 0.68 / 0.66 | 7.79 / 15.30 | shared | shared 0.68 / 0.66 of direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 7 | direct | 0.33 / 0.21 | 1.00 / 1.00 | 2.14 / 2.79 | 0.46 / 0.44 | 0.42 / 0.41 | 5.40 / 8.26 | shared | shared 0.42 / 0.41 of direct |
| FacetNormals | Quadrilateral | Gpu | f32 | 8 | direct | 0.39 / 0.27 | 1.00 / 1.00 | 2.95 / 3.04 | 0.41 / 0.35 | 0.39 / 0.32 | 5.29 / 7.25 | shared | shared 0.39 / 0.32 of direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 1 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.12 / 1.22 | 1.52 / 1.74 | 1.38 / 1.74 | 4.85 / 11.09 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 2 | direct | 0.03 / 0.01 | 1.00 / 1.00 | 1.32 / 1.36 | 1.65 / 1.46 | 1.36 / 1.41 | 9.19 / 21.80 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 3 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.26 / 1.75 | 1.04 / 1.44 | 0.89 / 1.26 | 13.02 / 32.49 | direct | no strategy 10% faster than direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 4 | direct | 0.06 / 0.03 | 1.00 / 1.00 | 1.81 / 3.81 | 1.07 / 1.10 | 0.89 / 0.90 | 12.60 / 24.64 | shared | shared 0.89 / 0.90 of direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 5 | direct | 0.08 / 0.04 | 1.00 / 1.00 | 2.40 / 4.81 | 0.92 / 0.93 | 0.79 / 0.78 | 17.10 / 35.39 | shared | shared 0.79 / 0.78 of direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 6 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 2.61 / 5.27 | 0.93 / 1.02 | 0.71 / 0.84 | 13.46 / 31.37 | shared | shared 0.71 / 0.84 of direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 7 | direct | 0.20 / 0.10 | 1.00 / 1.00 | 1.86 / 3.16 | 0.48 / 0.69 | 0.41 / 0.60 | 7.76 / 15.08 | shared | shared 0.41 / 0.60 of direct |
| FacetNormals | Quadrilateral | Gpu | f64 | 8 | direct | 0.28 / 0.16 | 1.00 / 1.00 | 1.91 / 2.55 | 0.57 / 0.74 | 0.46 / 0.63 | 6.12 / 10.18 | shared | shared 0.46 / 0.63 of direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 1 | direct | 0.43 | 1.00 | 1.07 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 2 | direct | 0.64 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 3 | direct | 1.08 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 4 | direct | 1.86 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 5 | loops | 3.19 | n.m.; T3: 1.11 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 6 | loops | 6.41 | n.m.; T3: 1.22 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 7 | loops | 9.06 | n.m.; T3: 1.33 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f32 | 8 | loops | 49.5 | n.m.; T3: 1.16 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 1 | direct | 0.89 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 2 | direct | 2.16 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 3 | direct | 6.41 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 4 | direct | 13.2 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 5 | loops | 21.5 | n.m.; T3: 1.10 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 6 | loops | 34.6 | n.m.; T3: 1.25 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 7 | loops | 47.7 | n.m.; T3: 1.43 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd128 | f64 | 8 | loops | 165 | n.m.; T3: 1.16 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 1 | direct | 0.68 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 2 | direct | 2.05 | 1.00 | 0.98 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 3 | direct | 4.94 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 4 | direct | 7.67 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 5 | loops | 7.66 | n.m.; T3: 1.16 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 6 | loops | 9.69 | n.m.; T3: 1.27 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 7 | loops | 15.0 | n.m.; T3: 2.69 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f32 | 8 | loops | 96.9 | n.m.; T3: 1.33 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 1 | direct | 0.47 | 1.00 | 1.12 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 2 | direct | 0.61 | 1.00 | 0.95 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 3 | direct | 1.10 | 1.00 | 0.63 | – | – | – | loops | loops 0.63 of direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 4 | direct | 1.79 | 1.00 | 0.78 | – | – | – | loops | loops 0.78 of direct |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 5 | loops | 1.68 | n.m.; T3: 1.33 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 6 | loops | 3.29 | n.m.; T3: 2.09 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 7 | loops | 4.18 | n.m.; T3: 4.07 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | CpuSimd512 | f64 | 8 | loops | 27.5 | n.m.; T3: 1.66 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f32 | 1 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 1.13 / 1.20 | – | 1.10 / 1.14 | 2.82 / 4.71 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f32 | 2 | direct | 0.11 / 0.04 | 1.00 / 1.00 | 1.35 / 1.49 | – | 1.15 / 1.41 | 4.81 / 10.51 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f32 | 3 | direct | 0.16 / 0.07 | 1.00 / 1.00 | 1.99 / 2.58 | – | 2.32 / 2.41 | 7.19 / 15.29 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f32 | 4 | direct | 0.31 / 0.15 | 1.00 / 1.00 | 2.90 / 2.58 | – | – | 8.41 / 15.71 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f32 | 5 | loops | 1.96 / 0.77 | n.m.; T3: 0.41 / 0.52 | 1.00 / 1.00 | – | – | 1.53 / 3.53 | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f32 | 6 | loops | 4.03 / 1.85 | n.m.; T3: 0.60 / 0.69 | 1.00 / 1.00 | – | – | 1.03 / 1.97 | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f32 | 7 | loops | 6.52 / 3.27 | n.m.; T3: 0.63 / 0.80 | 1.00 / 1.00 | – | – | 0.78 / 1.19 | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f32 | 8 | loops | 12.0 / 6.68 | n.m.; T3: 0.90 / 0.83 | 1.00 / 1.00 | – | – | 0.49 / 0.73 | cooperative | cooperative 0.49 / 0.73 of loops |
| FacetNormals | Tetrahedron | Gpu | f64 | 1 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.15 / 1.21 | – | 1.06 / 1.20 | 2.86 / 5.28 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f64 | 2 | direct | 0.06 / 0.02 | 1.00 / 1.00 | 1.54 / 1.55 | – | 1.06 / 1.29 | 4.46 / 10.21 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f64 | 3 | direct | 0.15 / 0.05 | 1.00 / 1.00 | 1.78 / 3.09 | – | 1.22 / 2.93 | 4.48 / 12.65 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f64 | 4 | direct | 0.31 / 0.12 | 1.00 / 1.00 | 2.17 / 2.21 | – | 1.37 / 1.36 | 4.23 / 10.03 | direct | no strategy 10% faster than direct |
| FacetNormals | Tetrahedron | Gpu | f64 | 5 | loops | 1.91 / 0.50 | n.m.; T3: 0.39 / 0.54 | 1.00 / 1.00 | – | 0.57 / 0.90 | 0.87 / 3.00 | shared | shared 0.57 / 0.90 of loops |
| FacetNormals | Tetrahedron | Gpu | f64 | 6 | loops | 3.82 / 0.92 | n.m.; T3: 0.80 / 1.17 | 1.00 / 1.00 | – | – | 0.62 / 2.03 | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f64 | 7 | loops | 5.51 / 1.51 | n.m.; T3: 0.69 / 1.03 | 1.00 / 1.00 | – | – | 0.52 / 1.51 | loops | no strategy 10% faster than loops |
| FacetNormals | Tetrahedron | Gpu | f64 | 8 | loops | 12.5 / 2.73 | n.m.; T3: 0.70 / 0.87 | 1.00 / 1.00 | – | – | 0.25 / 1.05 | loops | no strategy 10% faster than loops |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 1 | direct | 0.58 | 1.00 | 1.00 | 0.99 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 2 | direct | 1.50 | 1.00 | 0.95 | 0.54 | – | – | sumfact | sumfact 0.54 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 3 | direct | 3.52 | 1.00 | 0.76 | 0.26 | – | – | sumfact | sumfact 0.26 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 4 | direct | 9.29 | 1.00 | 0.75 | 0.15 | – | – | sumfact | sumfact 0.15 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 5 | direct | 40.8 | 1.00 | 0.58 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 6 | direct | 66.8 | 1.00 | 0.42 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 7 | direct | 167 | 1.00 | 0.49 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f32 | 8 | loops | 129 | n.m.; T3: 2.24 | 1.00 | 0.04 | – | – | sumfact | sumfact 0.04 of loops |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 1 | direct | 1.50 | 1.00 | 0.91 | 0.85 | – | – | sumfact | sumfact 0.85 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 2 | direct | 7.02 | 1.00 | 0.99 | 0.51 | – | – | sumfact | sumfact 0.51 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 3 | direct | 25.3 | 1.00 | 0.85 | 0.25 | – | – | sumfact | sumfact 0.25 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 4 | direct | 68.5 | 1.00 | 0.67 | 0.14 | – | – | sumfact | sumfact 0.14 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 5 | direct | 219 | 1.00 | 0.50 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 6 | direct | 330 | 1.00 | 0.37 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 7 | direct | 560 | 1.00 | 0.47 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| FacetNormals | Hexahedron | CpuSimd128 | f64 | 8 | loops | 377 | n.m.; T3: 2.26 | 1.00 | 0.05 | – | – | sumfact | sumfact 0.05 of loops |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 1 | direct | 1.68 | 1.00 | 0.98 | 0.86 | – | – | sumfact | sumfact 0.86 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 2 | direct | 5.13 | 1.00 | 0.96 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 3 | direct | 9.25 | 1.00 | 1.01 | 0.50 | – | – | sumfact | sumfact 0.50 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 4 | direct | 41.2 | 1.00 | 0.32 | 0.12 | – | – | sumfact | sumfact 0.12 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 5 | direct | 155 | 1.00 | 0.20 | 0.04 | – | – | sumfact | sumfact 0.04 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 6 | direct | 310 | 1.00 | 0.13 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 7 | direct | 463 | 1.00 | 0.19 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f32 | 8 | direct | 695 | 1.00 | 0.14 | 0.01 | – | – | sumfact | sumfact 0.01 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 1 | direct | 0.62 | 1.00 | 0.92 | 1.10 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 2 | direct | 1.02 | 1.00 | 0.77 | 0.67 | – | – | sumfact | sumfact 0.67 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 3 | direct | 3.53 | 1.00 | 0.73 | 0.35 | – | – | sumfact | sumfact 0.35 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 4 | direct | 13.7 | 1.00 | 0.29 | 0.10 | – | – | sumfact | sumfact 0.10 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 5 | direct | 38.7 | 1.00 | 0.17 | 0.05 | – | – | sumfact | sumfact 0.05 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 6 | direct | 92.4 | 1.00 | 0.15 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 7 | direct | 173 | 1.00 | 0.36 | 0.03 | – | – | sumfact | sumfact 0.03 of direct |
| FacetNormals | Hexahedron | CpuSimd512 | f64 | 8 | direct | 311 | 1.00 | 0.13 | 0.02 | – | – | sumfact | sumfact 0.02 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.28 / 1.39 | 1.19 / 1.51 | 1.15 / 1.41 | 5.00 / 10.54 | direct | no strategy 10% faster than direct |
| FacetNormals | Hexahedron | Gpu | f32 | 2 | direct | 0.21 / 0.09 | 1.00 / 1.00 | 2.61 / 3.34 | 0.80 / 0.90 | 0.77 / 0.86 | 5.95 / 12.39 | shared | shared 0.77 / 0.86 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 3 | direct | 1.08 / 0.71 | 1.00 / 1.00 | 1.74 / 1.58 | 0.23 / 0.17 | 0.23 / 0.18 | 3.12 / 4.01 | sumfact | sumfact 0.23 / 0.17 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 4 | direct | 4.50 / 2.48 | 1.00 / 1.00 | 1.40 / 1.49 | 0.11 / 0.12 | 0.10 / 0.10 | 0.95 / 1.84 | shared | shared 0.10 / 0.10 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 5 | direct | 13.2 / 8.60 | 1.00 / 1.00 | 1.16 / 0.91 | 0.05 / 0.04 | 0.05 / 0.04 | 0.54 / 0.78 | shared | shared 0.05 / 0.04 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 6 | direct | 28.3 / 20.2 | 1.00 / 1.00 | 1.18 / 0.74 | 0.04 / 0.03 | 0.08 / 0.05 | 0.34 / 0.49 | sumfact | sumfact 0.04 / 0.03 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 7 | direct | 61.9 / 44.3 | 1.00 / 1.00 | 0.93 / 0.57 | 0.05 / 0.03 | 0.05 / 0.03 | 0.26 / 0.33 | sumfact | sumfact 0.05 / 0.03 of direct |
| FacetNormals | Hexahedron | Gpu | f32 | 8 | loops | 99.6 / 42.6 | n.m.; T3: 1.03 / 1.67 | 1.00 / 1.00 | 0.04 / 0.04 | 0.04 / 0.04 | 0.25 / 0.45 | sumfact | sumfact 0.04 / 0.04 of loops |
| FacetNormals | Hexahedron | Gpu | f64 | 1 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.40 / 1.26 | 1.79 / 1.93 | 1.63 / 1.77 | 5.53 / 10.08 | direct | no strategy 10% faster than direct |
| FacetNormals | Hexahedron | Gpu | f64 | 2 | direct | 0.12 / 0.05 | 1.00 / 1.00 | 2.79 / 3.60 | 0.99 / 1.14 | 0.97 / 1.06 | 6.65 / 13.52 | direct | no strategy 10% faster than direct |
| FacetNormals | Hexahedron | Gpu | f64 | 3 | direct | 0.69 / 0.30 | 1.00 / 1.00 | 1.88 / 1.64 | 0.30 / 0.37 | 0.30 / 0.46 | 2.56 / 5.30 | sumfact | sumfact 0.30 / 0.37 of direct |
| FacetNormals | Hexahedron | Gpu | f64 | 4 | direct | 2.79 / 1.62 | 1.00 / 1.00 | 1.52 / 0.66 | 0.89 / 0.55 | 1.01 / 0.55 | 0.94 / 1.46 | sumfact | sumfact 0.89 / 0.55 of direct |
| FacetNormals | Hexahedron | Gpu | f64 | 5 | direct | 7.16 / 2.69 | 1.00 / 1.00 | 1.42 / 0.89 | 0.63 / 0.45 | 0.66 / 0.42 | 0.55 / 1.33 | shared | closest: shared 0.66 / 0.42 of direct, 5% slower than the fastest at one size |
| FacetNormals | Hexahedron | Gpu | f64 | 6 | direct | 134 / 24.9 | 1.00 / 1.00 | 0.18 / 0.20 | 0.05 / 0.06 | 0.06 / 0.06 | 0.05 / 0.22 | sumfact | sumfact 0.05 / 0.06 of direct |
| FacetNormals | Hexahedron | Gpu | f64 | 7 | direct | 48.5 / 11.2 | 1.00 / 1.00 | 1.04 / 0.80 | 0.21 / 0.18 | 0.28 / 0.21 | 0.17 / 0.69 | sumfact | closest: sumfact 0.21 / 0.18 of direct, 23% slower than the fastest at one size |
| FacetNormals | Hexahedron | Gpu | f64 | 8 | loops | 117 / 17.4 | n.m.; T3: 4.12 / 4.66 | 1.00 / 1.00 | 0.12 / 0.15 | 0.14 / 0.17 | 0.13 / 0.66 | sumfact | sumfact 0.12 / 0.15 of loops |
| FacetNormals | Prism | CpuSimd128 | f32 | 1 | direct | 0.53 | 1.00 | 0.95 | 1.03 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd128 | f32 | 2 | direct | 1.07 | 1.00 | 0.99 | 0.92 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd128 | f32 | 3 | direct | 1.85 | 1.00 | 0.96 | 0.85 | – | – | sumfact | sumfact 0.85 of direct |
| FacetNormals | Prism | CpuSimd128 | f32 | 4 | direct | 14.5 | 1.00 | 1.02 | 0.55 | – | – | sumfact | sumfact 0.55 of direct |
| FacetNormals | Prism | CpuSimd128 | f32 | 5 | loops | 7.35 | n.m.; T3: 1.37 | 1.00 | 0.68 | – | – | sumfact | sumfact 0.68 of loops |
| FacetNormals | Prism | CpuSimd128 | f32 | 6 | loops | 13.5 | n.m.; T3: 1.68 | 1.00 | 0.66 | – | – | sumfact | sumfact 0.66 of loops |
| FacetNormals | Prism | CpuSimd128 | f32 | 7 | loops | 48.8 | n.m.; T3: 1.65 | 1.00 | 0.52 | – | – | sumfact | sumfact 0.52 of loops |
| FacetNormals | Prism | CpuSimd128 | f32 | 8 | loops | 44.5 | n.m.; T3: 2.26 | 1.00 | 0.55 | – | – | sumfact | sumfact 0.55 of loops |
| FacetNormals | Prism | CpuSimd128 | f64 | 1 | direct | 1.10 | 1.00 | 0.98 | 1.12 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd128 | f64 | 2 | direct | 4.11 | 1.00 | 1.06 | 0.90 | – | – | sumfact | sumfact 0.90 of direct |
| FacetNormals | Prism | CpuSimd128 | f64 | 3 | direct | 13.3 | 1.00 | 0.89 | 0.74 | – | – | sumfact | sumfact 0.74 of direct |
| FacetNormals | Prism | CpuSimd128 | f64 | 4 | direct | 85.7 | 1.00 | 1.08 | 0.60 | – | – | sumfact | sumfact 0.60 of direct |
| FacetNormals | Prism | CpuSimd128 | f64 | 5 | loops | 46.5 | n.m.; T3: 1.47 | 1.00 | 0.67 | – | – | sumfact | sumfact 0.67 of loops |
| FacetNormals | Prism | CpuSimd128 | f64 | 6 | loops | 71.1 | n.m.; T3: 1.84 | 1.00 | 0.63 | – | – | sumfact | sumfact 0.63 of loops |
| FacetNormals | Prism | CpuSimd128 | f64 | 7 | loops | 151 | n.m.; T3: 1.99 | 1.00 | 0.56 | – | – | sumfact | sumfact 0.56 of loops |
| FacetNormals | Prism | CpuSimd128 | f64 | 8 | loops | 165 | n.m.; T3: 2.45 | 1.00 | 0.56 | – | – | sumfact | sumfact 0.56 of loops |
| FacetNormals | Prism | CpuSimd512 | f32 | 1 | direct | 1.14 | 1.00 | 1.06 | 0.99 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd512 | f32 | 2 | direct | 3.54 | 1.00 | 0.95 | 1.07 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd512 | f32 | 3 | direct | 6.67 | 1.00 | 0.98 | 1.16 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd512 | f32 | 4 | direct | 41.5 | 1.00 | 1.00 | 0.58 | – | – | sumfact | sumfact 0.58 of direct |
| FacetNormals | Prism | CpuSimd512 | f32 | 5 | loops | 14.8 | n.m.; T3: 2.55 | 1.00 | 1.16 | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Prism | CpuSimd512 | f32 | 6 | loops | 21.2 | n.m.; T3: 6.27 | 1.00 | 1.12 | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Prism | CpuSimd512 | f32 | 7 | loops | 39.7 | n.m.; T3: 5.55 | 1.00 | 0.86 | – | – | sumfact | sumfact 0.86 of loops |
| FacetNormals | Prism | CpuSimd512 | f32 | 8 | direct | 390 | 1.00 | 0.13 | 0.12 | – | – | sumfact | sumfact 0.12 of direct |
| FacetNormals | Prism | CpuSimd512 | f64 | 1 | direct | 0.60 | 1.00 | 0.94 | 1.14 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd512 | f64 | 2 | direct | 0.89 | 1.00 | 1.08 | 1.08 | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | CpuSimd512 | f64 | 3 | direct | 1.86 | 1.00 | 0.84 | 0.92 | – | – | loops | loops 0.84 of direct |
| FacetNormals | Prism | CpuSimd512 | f64 | 4 | direct | 7.16 | 1.00 | 0.99 | 0.54 | – | – | sumfact | sumfact 0.54 of direct |
| FacetNormals | Prism | CpuSimd512 | f64 | 5 | loops | 3.86 | n.m.; T3: 4.27 | 1.00 | 1.02 | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Prism | CpuSimd512 | f64 | 6 | loops | 6.15 | n.m.; T3: 6.03 | 1.00 | 1.07 | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Prism | CpuSimd512 | f64 | 7 | loops | 11.1 | n.m.; T3: 6.51 | 1.00 | 0.93 | – | – | loops | no strategy 10% faster than loops |
| FacetNormals | Prism | CpuSimd512 | f64 | 8 | direct | 131 | 1.00 | 0.16 | 0.13 | – | – | sumfact | sumfact 0.13 of direct |
| FacetNormals | Prism | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.10 / 1.27 | 1.05 / 1.24 | 1.06 / 1.13 | 2.69 / 5.09 | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | Gpu | f32 | 2 | direct | 0.15 / 0.06 | 1.00 / 1.00 | 1.70 / 2.42 | 1.07 / 1.39 | 1.03 / 1.30 | 8.01 / 17.74 | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | Gpu | f32 | 3 | direct | 0.33 / 0.15 | 1.00 / 1.00 | 2.78 / 3.39 | 1.12 / 1.20 | 1.02 / 1.14 | 8.21 / 17.01 | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | Gpu | f32 | 4 | direct | 1.49 / 0.96 | 1.00 / 1.00 | 1.87 / 1.64 | 0.58 / 0.47 | 0.53 / 0.46 | 2.38 / 3.25 | shared | shared 0.53 / 0.46 of direct |
| FacetNormals | Prism | Gpu | f32 | 5 | direct | 4.39 / 2.42 | 1.00 / 1.00 | 1.36 / 1.62 | 0.40 / 0.39 | 0.35 / 0.38 | 1.02 / 1.66 | shared | shared 0.35 / 0.38 of direct |
| FacetNormals | Prism | Gpu | f32 | 6 | direct | 12.6 / 7.94 | 1.00 / 1.00 | 1.10 / 0.98 | 0.28 / 0.29 | – | 0.50 / 0.69 | sumfact | sumfact 0.28 / 0.29 of direct |
| FacetNormals | Prism | Gpu | f32 | 7 | direct | 22.2 / 14.3 | 1.00 / 1.00 | 1.17 / 0.90 | 0.31 / 0.21 | – | 0.42 / 0.59 | sumfact | sumfact 0.31 / 0.21 of direct |
| FacetNormals | Prism | Gpu | f32 | 8 | direct | 76.9 / 53.9 | 1.00 / 1.00 | 0.61 / 0.39 | 0.13 / 0.08 | – | 0.14 / 0.19 | sumfact | sumfact 0.13 / 0.08 of direct |
| FacetNormals | Prism | Gpu | f64 | 1 | direct | 0.05 / 0.02 | 1.00 / 1.00 | 1.23 / 1.27 | 1.05 / 1.30 | 1.01 / 1.27 | 2.72 / 5.31 | direct | no strategy 10% faster than direct |
| FacetNormals | Prism | Gpu | f64 | 2 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 2.09 / 1.82 | 0.86 / 1.01 | 0.76 / 0.87 | 7.10 / 14.44 | shared | shared 0.76 / 0.87 of direct |
| FacetNormals | Prism | Gpu | f64 | 3 | direct | 0.29 / 0.11 | 1.00 / 1.00 | 2.44 / 2.45 | 0.67 / 0.63 | 0.61 / 0.56 | 4.98 / 11.33 | shared | shared 0.61 / 0.56 of direct |
| FacetNormals | Prism | Gpu | f64 | 4 | direct | 1.37 / 0.78 | 1.00 / 1.00 | 1.37 / 0.81 | 0.38 / 0.29 | 0.31 / 0.30 | 1.37 / 2.24 | shared | closest: shared 0.31 / 0.30 of direct, 5% slower than the fastest at one size |
| FacetNormals | Prism | Gpu | f64 | 5 | direct | 3.19 / 1.40 | 1.00 / 1.00 | 1.58 / 0.93 | 0.38 / 0.36 | 0.32 / 0.35 | 0.88 / 1.70 | shared | shared 0.32 / 0.35 of direct |
| FacetNormals | Prism | Gpu | f64 | 6 | direct | 6.10 / 2.45 | 1.00 / 1.00 | 1.65 / 1.03 | 0.37 / 0.36 | 0.29 / 0.34 | 0.68 / 1.37 | shared | shared 0.29 / 0.34 of direct |
| FacetNormals | Prism | Gpu | f64 | 7 | direct | 13.0 / 3.95 | 1.00 / 1.00 | 1.47 / 1.13 | 0.31 / 0.26 | 0.23 / 0.24 | 0.42 / 1.14 | shared | shared 0.23 / 0.24 of direct |
| FacetNormals | Prism | Gpu | f64 | 8 | direct | 267 / 43.8 | 1.00 / 1.00 | 0.13 / 0.18 | 0.03 / 0.05 | 0.02 / 0.04 | 0.03 / 0.15 | shared | shared 0.02 / 0.04 of direct |
| FacetNormals | Pyramid | CpuSimd128 | f32 | 1 | direct | 0.50 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd128 | f32 | 2 | direct | 0.92 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd128 | f32 | 3 | direct | 1.50 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd128 | f64 | 1 | direct | 1.07 | 1.00 | 0.96 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd128 | f64 | 2 | direct | 3.66 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd128 | f64 | 3 | direct | 10.5 | 1.00 | 0.91 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f32 | 1 | direct | 0.99 | 1.00 | 1.01 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f32 | 2 | direct | 3.33 | 1.00 | 0.97 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f32 | 3 | direct | 6.37 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f64 | 1 | direct | 0.55 | 1.00 | 0.93 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f64 | 2 | direct | 0.75 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | CpuSimd512 | f64 | 3 | direct | 1.61 | 1.00 | 0.85 | – | – | – | loops | loops 0.85 of direct |
| FacetNormals | Pyramid | Gpu | f32 | 1 | direct | 0.10 / 0.04 | 1.00 / 1.00 | 1.16 / 1.21 | – | 1.15 / 1.16 | 2.83 / 5.02 | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | Gpu | f32 | 2 | direct | 0.15 / 0.06 | 1.00 / 1.00 | 1.52 / 1.48 | – | 1.35 / 1.77 | 3.90 / 8.29 | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | Gpu | f32 | 3 | direct | 0.29 / 0.10 | 1.00 / 1.00 | 2.41 / 2.90 | – | 2.63 / 2.60 | 4.64 / 10.98 | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | Gpu | f64 | 1 | direct | 0.06 / 0.02 | 1.00 / 1.00 | 1.14 / 1.24 | – | 0.99 / 1.19 | 2.59 / 5.22 | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | Gpu | f64 | 2 | direct | 0.10 / 0.03 | 1.00 / 1.00 | 1.77 / 1.46 | – | 1.11 / 1.53 | 3.27 / 8.67 | direct | no strategy 10% faster than direct |
| FacetNormals | Pyramid | Gpu | f64 | 3 | direct | 0.17 / 0.05 | 1.00 / 1.00 | 3.01 / 3.81 | – | 1.85 / 3.68 | 5.12 / 13.28 | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 1 | direct | 3.68 | 1.00 | 3.80 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 2 | direct | 8.08 | 1.00 | 2.87 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 3 | direct | 12.6 | 1.00 | 2.10 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 4 | direct | 21.0 | 1.00 | 1.48 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 5 | direct | 25.1 | 1.00 | 1.37 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 6 | direct | 32.0 | 1.00 | 1.27 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 7 | direct | 37.5 | 1.00 | 1.04 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f32 | 8 | direct | 39.5 | 1.00 | 1.17 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 1 | direct | 3.66 | 1.00 | 3.56 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 2 | direct | 8.35 | 1.00 | 2.53 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 3 | direct | 13.5 | 1.00 | 1.79 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 4 | direct | 22.4 | 1.00 | 1.50 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 5 | direct | 27.4 | 1.00 | 1.14 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 6 | direct | 33.2 | 1.00 | 1.29 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 7 | direct | 39.1 | 1.00 | 1.22 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd128 | f64 | 8 | direct | 45.9 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 1 | direct | 7.08 | 1.00 | 1.24 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 2 | direct | 9.20 | 1.00 | 2.11 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 3 | direct | 16.4 | 1.00 | 1.26 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 4 | direct | 23.2 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 5 | direct | 29.5 | 1.00 | 0.94 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 6 | direct | 28.8 | 1.00 | 1.23 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 7 | direct | 35.8 | 1.00 | 1.17 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f32 | 8 | direct | 44.9 | 1.00 | 1.13 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 1 | direct | 7.80 | 1.00 | 1.47 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 2 | direct | 11.0 | 1.00 | 1.88 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 3 | direct | 21.0 | 1.00 | 1.24 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 4 | direct | 24.5 | 1.00 | 1.02 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 5 | direct | 31.0 | 1.00 | 0.99 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 6 | direct | 32.0 | 1.00 | 1.03 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 7 | direct | 40.1 | 1.00 | 1.05 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | CpuSimd512 | f64 | 8 | direct | 38.2 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 1 | direct | 0.09 / 0.04 | 1.00 / 1.00 | 3.45 / 6.28 | – | 3.39 / 6.25 | 7.99 / 17.52 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 2 | direct | 0.10 / 0.05 | 1.00 / 1.00 | 3.75 / 5.77 | – | 3.65 / 5.79 | 8.34 / 15.37 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 3 | direct | 0.12 / 0.06 | 1.00 / 1.00 | 4.52 / 7.48 | – | 4.55 / 7.50 | 20.27 / 39.46 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 4 | direct | 0.16 / 0.09 | 1.00 / 1.00 | 3.90 / 6.07 | – | 3.87 / 6.06 | 17.14 / 31.04 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 5 | direct | 0.18 / 0.11 | 1.00 / 1.00 | 4.06 / 5.43 | – | 3.96 / 5.44 | 17.38 / 27.21 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 6 | direct | 0.22 / 0.14 | 1.00 / 1.00 | 3.65 / 4.83 | – | 3.67 / 4.83 | 15.31 / 23.80 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 7 | direct | 0.24 / 0.16 | 1.00 / 1.00 | 3.29 / 4.12 | – | 3.30 / 4.12 | 32.46 / 47.55 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f32 | 8 | direct | 0.29 / 0.20 | 1.00 / 1.00 | 2.96 / 3.56 | – | 3.00 / 3.55 | 27.88 / 40.40 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 1 | direct | 0.04 / 0.02 | 1.00 / 1.00 | 6.51 / 12.64 | – | 6.31 / 12.61 | 17.04 / 37.02 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 2 | direct | 0.05 / 0.03 | 1.00 / 1.00 | 6.20 / 8.99 | – | 6.16 / 8.99 | 15.80 / 25.33 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 3 | direct | 0.08 / 0.05 | 1.00 / 1.00 | 5.73 / 7.98 | – | 5.78 / 8.00 | 29.96 / 45.21 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 4 | direct | 0.10 / 0.08 | 1.00 / 1.00 | 4.75 / 5.44 | – | 4.73 / 5.44 | 23.80 / 29.98 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 5 | direct | 0.15 / 0.11 | 1.00 / 1.00 | 3.76 / 4.22 | – | 3.76 / 4.22 | 17.95 / 22.70 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 6 | direct | 0.18 / 0.13 | 1.00 / 1.00 | 3.39 / 3.90 | – | 3.42 / 3.90 | 15.16 / 20.62 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 7 | direct | 0.20 / 0.16 | 1.00 / 1.00 | 4.25 / 4.67 | – | 4.24 / 4.66 | 42.84 / 52.54 | direct | no strategy 10% faster than direct |
| PullBack | Interval | Gpu | f64 | 8 | direct | 0.26 / 0.20 | 1.00 / 1.00 | 3.65 / 3.94 | – | 3.67 / 3.94 | 34.24 / 43.91 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 1 | direct | 15.4 | 1.00 | 1.81 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 2 | direct | 21.5 | 1.00 | 1.72 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 3 | direct | 37.5 | 1.00 | 1.50 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 4 | direct | 53.5 | 1.00 | 1.27 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 5 | direct | 89.8 | 1.00 | 1.15 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 6 | direct | 145 | 1.00 | 1.00 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 7 | direct | 247 | 1.00 | 0.92 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f32 | 8 | direct | 404 | 1.00 | 0.81 | – | – | – | loops | loops 0.81 of direct |
| PullBack | Triangle | CpuSimd128 | f64 | 1 | direct | 15.6 | 1.00 | 1.84 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 2 | direct | 28.3 | 1.00 | 1.42 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 3 | direct | 43.6 | 1.00 | 1.26 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 4 | direct | 67.7 | 1.00 | 1.08 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 5 | direct | 77.7 | 1.00 | 1.16 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 6 | direct | 92.0 | 1.00 | 1.25 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 7 | direct | 125 | 1.00 | 1.31 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd128 | f64 | 8 | direct | 176 | 1.00 | 1.25 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 1 | direct | 13.6 | 1.00 | 1.96 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 2 | direct | 23.2 | 1.00 | 1.87 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 3 | direct | 29.7 | 1.00 | 1.97 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 4 | direct | 67.5 | 1.00 | 1.72 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 5 | direct | 136 | 1.00 | 1.54 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 6 | direct | 281 | 1.00 | 1.33 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 7 | direct | 513 | 1.00 | 1.19 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f32 | 8 | direct | 865 | 1.00 | 1.12 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 1 | direct | 14.9 | 1.00 | 1.99 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 2 | direct | 21.1 | 1.00 | 1.82 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 3 | direct | 34.0 | 1.00 | 1.80 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 4 | direct | 47.3 | 1.00 | 2.03 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 5 | direct | 74.1 | 1.00 | 1.69 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 6 | direct | 131 | 1.00 | 1.60 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 7 | direct | 232 | 1.00 | 1.51 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | CpuSimd512 | f64 | 8 | direct | 358 | 1.00 | 1.47 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 1 | direct | 0.14 / 0.07 | 1.00 / 1.00 | 3.51 / 5.12 | – | 3.50 / 5.13 | 8.59 / 14.45 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 2 | direct | 0.23 / 0.14 | 1.00 / 1.00 | 3.44 / 4.31 | – | 3.43 / 4.28 | 16.88 / 25.28 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 3 | direct | 0.36 / 0.28 | 1.00 / 1.00 | 2.96 / 3.27 | – | 2.97 / 3.27 | 36.84 / 46.83 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 4 | direct | 0.74 / 0.46 | 1.00 / 1.00 | 2.46 / 2.76 | – | 2.47 / 2.75 | 23.26 / 35.39 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 5 | direct | 0.97 / 0.85 | 1.00 / 1.00 | 2.62 / 2.06 | – | 2.63 / 2.07 | 50.57 / 55.74 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 6 | direct | 1.73 / 1.27 | 1.00 / 1.00 | 2.26 / 1.87 | – | 2.26 / 1.87 | 33.21 / 43.77 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 7 | direct | 2.48 / 1.71 | 1.00 / 1.00 | 2.28 / 1.71 | – | 2.27 / 1.71 | 65.26 / 92.47 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f32 | 8 | loops | 9.27 / 3.84 | n.m.; T3: 0.57 / 0.63 | 1.00 / 1.00 | – | 1.00 / 1.00 | 19.76 / 46.51 | loops | no strategy 10% faster than loops |
| PullBack | Triangle | Gpu | f64 | 1 | direct | 0.07 / 0.04 | 1.00 / 1.00 | 4.71 / 6.37 | – | 4.59 / 6.38 | 11.83 / 18.06 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 2 | direct | 0.17 / 0.13 | 1.00 / 1.00 | 3.50 / 3.63 | – | 3.49 / 3.63 | 16.70 / 20.96 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 3 | direct | 0.31 / 0.28 | 1.00 / 1.00 | 2.51 / 2.53 | – | 2.52 / 2.53 | 28.51 / 30.91 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 4 | direct | 0.83 / 0.54 | 1.00 / 1.00 | 2.03 / 1.83 | – | 2.02 / 1.83 | 13.14 / 18.91 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 5 | direct | 1.22 / 0.90 | 1.00 / 1.00 | 2.40 / 1.59 | – | 2.38 / 1.58 | 27.93 / 29.97 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 6 | direct | 2.00 / 1.46 | 1.00 / 1.00 | 2.07 / 1.36 | – | 2.06 / 1.35 | 17.31 / 22.52 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 7 | direct | 3.78 / 1.95 | 1.00 / 1.00 | 1.86 / 1.24 | – | 1.86 / 1.23 | 23.42 / 43.59 | direct | no strategy 10% faster than direct |
| PullBack | Triangle | Gpu | f64 | 8 | loops | 10.8 / 2.84 | n.m.; T3: 0.85 / 1.54 | 1.00 / 1.00 | – | 1.00 / 0.99 | 9.46 / 35.02 | loops | no strategy 10% faster than loops |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 1 | direct | 15.0 | 1.00 | 1.94 | 1.64 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 2 | direct | 23.0 | 1.00 | 1.82 | 1.48 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 3 | direct | 41.4 | 1.00 | 1.46 | 1.16 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 4 | direct | 51.8 | 1.00 | 1.36 | 1.17 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 5 | direct | 87.4 | 1.00 | 1.04 | 0.81 | – | – | sumfact | sumfact 0.81 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 6 | direct | 134 | 1.00 | 1.01 | 0.79 | – | – | sumfact | sumfact 0.79 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 7 | direct | 219 | 1.00 | 0.98 | 0.76 | – | – | sumfact | sumfact 0.76 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f32 | 8 | direct | 333 | 1.00 | 0.99 | 0.73 | – | – | sumfact | sumfact 0.73 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 1 | direct | 15.8 | 1.00 | 1.86 | 1.71 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 2 | direct | 25.3 | 1.00 | 1.57 | 1.46 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 3 | direct | 40.9 | 1.00 | 1.32 | 1.14 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 4 | direct | 54.4 | 1.00 | 1.20 | 1.07 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 5 | direct | 84.4 | 1.00 | 1.14 | 0.81 | – | – | sumfact | sumfact 0.81 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 6 | direct | 94.9 | 1.00 | 1.12 | 0.91 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 7 | direct | 136 | 1.00 | 1.06 | 0.86 | – | – | sumfact | sumfact 0.86 of direct |
| PullBack | Quadrilateral | CpuSimd128 | f64 | 8 | direct | 189 | 1.00 | 0.98 | 0.82 | – | – | sumfact | sumfact 0.82 of direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 1 | direct | 14.2 | 1.00 | 1.83 | 1.72 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 2 | direct | 22.0 | 1.00 | 1.62 | 1.76 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 3 | direct | 31.9 | 1.00 | 1.51 | 1.50 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 4 | direct | 59.8 | 1.00 | 1.46 | 1.43 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 5 | direct | 127 | 1.00 | 1.32 | 1.35 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 6 | direct | 225 | 1.00 | 1.18 | 1.18 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 7 | direct | 351 | 1.00 | 1.18 | 1.07 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f32 | 8 | direct | 560 | 1.00 | 1.13 | 1.10 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 1 | direct | 17.4 | 1.00 | 1.52 | 1.45 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 2 | direct | 19.6 | 1.00 | 1.71 | 1.74 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 3 | direct | 30.1 | 1.00 | 1.66 | 1.80 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 4 | direct | 36.3 | 1.00 | 1.56 | 1.71 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 5 | direct | 69.6 | 1.00 | 1.36 | 1.42 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 6 | direct | 111 | 1.00 | 1.28 | 1.28 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 7 | direct | 182 | 1.00 | 1.24 | 1.17 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | CpuSimd512 | f64 | 8 | direct | 295 | 1.00 | 1.12 | 1.09 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 1 | direct | 0.14 / 0.08 | 1.00 / 1.00 | 3.87 / 5.45 | 4.41 / 6.21 | 4.37 / 6.20 | 23.86 / 41.00 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 2 | direct | 0.23 / 0.15 | 1.00 / 1.00 | 3.92 / 4.82 | 3.86 / 4.73 | 3.86 / 4.72 | 54.24 / 84.07 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 3 | direct | 0.36 / 0.28 | 1.00 / 1.00 | 3.51 / 3.52 | 4.32 / 4.73 | 4.30 / 4.71 | 105.40 / 132.14 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 4 | direct | 0.71 / 0.59 | 1.00 / 1.00 | 3.12 / 2.46 | 3.65 / 3.79 | 3.65 / 3.78 | 60.76 / 70.65 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 5 | direct | 0.97 / 0.81 | 1.00 / 1.00 | 3.34 / 2.69 | 3.93 / 3.91 | 3.93 / 3.93 | 122.97 / 144.04 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 6 | direct | 2.77 / 1.81 | 1.00 / 1.00 | 2.05 / 2.40 | 2.16 / 3.07 | 2.17 / 3.04 | 50.39 / 75.01 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 7 | direct | 6.59 / 4.09 | 1.00 / 1.00 | 1.37 / 1.66 | 1.81 / 1.97 | 1.81 / 1.97 | 23.56 / 36.97 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f32 | 8 | direct | 12.8 / 7.57 | 1.00 / 1.00 | 1.00 / 1.11 | 1.35 / 1.46 | 1.35 / 1.45 | 14.73 / 24.14 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 1 | direct | 0.08 / 0.05 | 1.00 / 1.00 | 4.08 / 5.74 | 6.01 / 8.77 | 6.02 / 8.79 | 41.74 / 69.06 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 2 | direct | 0.13 / 0.09 | 1.00 / 1.00 | 4.47 / 5.02 | 5.25 / 6.03 | 5.25 / 6.01 | 78.34 / 116.56 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 3 | direct | 0.27 / 0.23 | 1.00 / 1.00 | 2.77 / 2.93 | 3.85 / 5.88 | 3.81 / 5.87 | 97.31 / 111.26 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 4 | direct | 0.62 / 0.39 | 1.00 / 1.00 | 2.58 / 2.55 | 3.24 / 5.21 | 3.20 / 5.20 | 44.64 / 68.66 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 5 | direct | 0.83 / 0.72 | 1.00 / 1.00 | 2.76 / 2.17 | 3.39 / 4.23 | 3.35 / 4.23 | 84.56 / 94.82 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 6 | direct | 1.54 / 1.06 | 1.00 / 1.00 | 2.88 / 3.09 | 3.52 / 3.64 | 3.50 / 3.63 | 50.34 / 70.01 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 7 | direct | 4.95 / 2.57 | 1.00 / 1.00 | 1.42 / 1.74 | 1.96 / 2.20 | 1.95 / 2.20 | 20.30 / 38.63 | direct | no strategy 10% faster than direct |
| PullBack | Quadrilateral | Gpu | f64 | 8 | direct | 8.99 / 3.44 | 1.00 / 1.00 | 1.32 / 1.75 | 1.70 / 1.82 | 1.69 / 1.82 | 13.44 / 34.37 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 1 | direct | 26.8 | 1.00 | 1.91 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 2 | direct | 48.0 | 1.00 | 2.07 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 3 | direct | 144 | 1.00 | 1.71 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 4 | direct | 468 | 1.00 | 1.49 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 5 | loops | 1684 | n.m.; T3: 0.70 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 6 | loops | 2610 | n.m.; T3: 0.74 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 7 | loops | 3355 | n.m.; T3: 0.74 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f32 | 8 | loops | 4429 | n.m.; T3: 0.79 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 1 | direct | 27.5 | 1.00 | 1.76 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 2 | direct | 44.3 | 1.00 | 2.18 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 3 | direct | 87.4 | 1.00 | 1.79 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 4 | direct | 242 | 1.00 | 1.59 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 5 | loops | 964 | n.m.; T3: 0.67 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 6 | loops | 2043 | n.m.; T3: 0.63 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 7 | loops | 3342 | n.m.; T3: 0.81 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd128 | f64 | 8 | loops | 4451 | n.m.; T3: 0.81 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 1 | direct | 19.2 | 1.00 | 12.15 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 2 | direct | 60.7 | 1.00 | 9.53 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 3 | direct | 336 | 1.00 | 5.05 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 4 | direct | 966 | 1.00 | 4.62 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 5 | loops | 10500 | n.m.; T3: 0.28 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 6 | loops | 14452 | n.m.; T3: 0.34 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 7 | loops | 17880 | n.m.; T3: 0.39 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f32 | 8 | loops | 21963 | n.m.; T3: 0.43 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 1 | direct | 24.2 | 1.00 | 2.96 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 2 | direct | 34.4 | 1.00 | 3.48 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 3 | direct | 137 | 1.00 | 2.53 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 4 | direct | 508 | 1.00 | 2.06 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 5 | loops | 2802 | n.m.; T3: 0.52 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 6 | loops | 8566 | n.m.; T3: 0.38 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 7 | loops | 11594 | n.m.; T3: 0.61 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | CpuSimd512 | f64 | 8 | loops | 15482 | n.m.; T3: 0.60 | 1.00 | – | – | – | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f32 | 1 | direct | 0.23 / 0.17 | 1.00 / 1.00 | 3.95 / 4.88 | – | 3.95 / 4.88 | 28.18 / 37.75 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f32 | 2 | direct | 0.60 / 0.38 | 1.00 / 1.00 | 3.84 / 4.09 | – | 3.84 / 4.09 | 39.62 / 58.63 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f32 | 3 | direct | 1.53 / 1.05 | 1.00 / 1.00 | 3.32 / 2.65 | – | 3.32 / 2.65 | 48.97 / 69.87 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f32 | 4 | loops | 13.0 / 5.28 | n.m.; T3: 0.69 / 0.69 | 1.00 / 1.00 | – | 1.00 / 1.00 | 18.56 / 43.83 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f32 | 5 | loops | 28.1 / 9.57 | – | 1.00 / 1.00 | – | 1.00 / 0.99 | 12.23 / 31.82 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f32 | 6 | loops | 56.5 / 18.5 | – | 1.00 / 1.00 | – | 1.00 / 1.00 | 7.64 / 21.58 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f32 | 7 | loops | 127 / 37.1 | – | 1.00 / 1.00 | – | – | 4.40 / 14.48 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f32 | 8 | loops | 270 / 66.6 | – | 1.00 / 1.00 | – | – | 3.40 / 10.59 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f64 | 1 | direct | 0.13 / 0.10 | 1.00 / 1.00 | 6.28 / 7.71 | – | 6.24 / 7.71 | 54.20 / 68.91 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f64 | 2 | direct | 0.54 / 0.32 | 1.00 / 1.00 | 5.32 / 4.87 | – | 5.32 / 4.88 | 44.94 / 69.29 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f64 | 3 | direct | 1.33 / 0.75 | 1.00 / 1.00 | 4.27 / 3.67 | – | 4.28 / 3.68 | 50.24 / 86.70 | direct | no strategy 10% faster than direct |
| PullBack | Tetrahedron | Gpu | f64 | 4 | loops | 18.6 / 5.35 | n.m.; T3: 0.78 / 0.52 | 1.00 / 1.00 | – | 1.01 / 1.00 | 10.98 / 36.23 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f64 | 5 | loops | 51.9 / 9.43 | – | 1.00 / 1.00 | – | 1.01 / 1.02 | 6.09 / 28.28 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f64 | 6 | loops | 106 / 17.8 | – | 1.00 / 1.00 | – | 1.00 / 1.01 | 3.41 / 19.77 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f64 | 7 | loops | 218 / 36.2 | – | 1.00 / 1.00 | – | 1.00 / 1.01 | 2.20 / 13.45 | loops | no strategy 10% faster than loops |
| PullBack | Tetrahedron | Gpu | f64 | 8 | loops | 460 / 72.1 | – | 1.00 / 1.00 | – | 1.00 / 1.00 | 2.15 / 8.75 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | CpuSimd128 | f32 | 1 | direct | 34.0 | 1.00 | 1.40 | 1.34 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 2 | direct | 78.8 | 1.00 | 1.29 | 1.03 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 3 | direct | 261 | 1.00 | 1.22 | 0.90 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 4 | direct | 826 | 1.00 | 1.21 | 1.01 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 5 | direct | 3244 | 1.00 | 0.73 | 0.50 | – | – | sumfact | sumfact 0.50 of direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 6 | direct | 3875 | 1.00 | 1.09 | 0.69 | – | – | sumfact | sumfact 0.69 of direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 7 | direct | 5807 | 1.00 | 1.54 | 1.29 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f32 | 8 | loops | 8962 | n.m.; T3: 1.46 | 1.00 | 0.60 | – | – | sumfact | sumfact 0.60 of loops |
| PullBack | Hexahedron | CpuSimd128 | f64 | 1 | direct | 32.0 | 1.00 | 1.53 | 1.45 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 2 | direct | 77.1 | 1.00 | 1.40 | 0.95 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 3 | direct | 226 | 1.00 | 0.94 | 0.66 | – | – | sumfact | sumfact 0.66 of direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 4 | direct | 512 | 1.00 | 1.79 | 1.54 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 5 | direct | 1683 | 1.00 | 0.96 | 0.57 | – | – | sumfact | sumfact 0.57 of direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 6 | direct | 4688 | 1.00 | 0.86 | 0.50 | – | – | sumfact | sumfact 0.50 of direct |
| PullBack | Hexahedron | CpuSimd128 | f64 | 7 | loops | 9904 | n.m.; T3: 0.59 | 1.00 | 0.78 | – | – | sumfact | sumfact 0.78 of loops |
| PullBack | Hexahedron | CpuSimd128 | f64 | 8 | loops | 10920 | n.m.; T3: 0.76 | 1.00 | 0.52 | – | – | sumfact | sumfact 0.52 of loops |
| PullBack | Hexahedron | CpuSimd512 | f32 | 1 | direct | 23.4 | 1.00 | 1.98 | 1.85 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 2 | direct | 113 | 1.00 | 1.22 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 3 | direct | 767 | 1.00 | 0.73 | 0.55 | – | – | sumfact | sumfact 0.55 of direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 4 | direct | 2552 | 1.00 | 0.74 | 0.64 | – | – | sumfact | sumfact 0.64 of direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 5 | direct | 9703 | 1.00 | 0.48 | 0.40 | – | – | sumfact | sumfact 0.40 of direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 6 | direct | 14073 | 1.00 | 0.65 | 0.54 | – | – | sumfact | sumfact 0.54 of direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 7 | direct | 20818 | 1.00 | 0.59 | 0.47 | – | – | sumfact | sumfact 0.47 of direct |
| PullBack | Hexahedron | CpuSimd512 | f32 | 8 | direct | 29911 | 1.00 | 0.57 | 0.46 | – | – | sumfact | sumfact 0.46 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 1 | direct | 27.2 | 1.00 | 2.36 | 2.03 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 2 | direct | 64.8 | 1.00 | 1.57 | 1.23 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 3 | direct | 390 | 1.00 | 0.82 | 0.63 | – | – | sumfact | sumfact 0.63 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 4 | direct | 1306 | 1.00 | 0.85 | 0.70 | – | – | sumfact | sumfact 0.70 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 5 | direct | 4823 | 1.00 | 0.57 | 0.44 | – | – | sumfact | sumfact 0.44 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 6 | direct | 13930 | 1.00 | 0.54 | 0.42 | – | – | sumfact | sumfact 0.42 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 7 | direct | 23815 | 1.00 | 0.59 | 0.44 | – | – | sumfact | sumfact 0.44 of direct |
| PullBack | Hexahedron | CpuSimd512 | f64 | 8 | direct | 34583 | 1.00 | 1.62 | 1.02 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f32 | 1 | direct | 0.30 / 0.23 | 1.00 / 1.00 | 3.51 / 4.08 | 3.80 / 3.97 | 3.81 / 3.97 | 69.27 / 90.30 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f32 | 2 | direct | 1.26 / 0.83 | 1.00 / 1.00 | 3.31 / 3.14 | 3.23 / 3.77 | 3.48 / 3.77 | 54.82 / 80.41 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f32 | 3 | direct | 17.3 / 11.2 | 1.00 / 1.00 | 0.90 / 0.88 | 0.90 / 1.05 | 0.90 / 1.05 | 14.59 / 21.97 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f32 | 4 | loops | 38.7 / 27.8 | n.m.; T3: 1.50 / 1.28 | 1.00 / 1.00 | 1.05 / 1.04 | 1.05 / 1.05 | 10.16 / 13.37 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f32 | 5 | loops | 92.6 / 53.2 | – | 1.00 / 1.00 | 1.02 / 1.24 | 1.02 / 1.24 | 6.95 / 10.69 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f32 | 6 | loops | 221 / 98.8 | – | 1.00 / 1.00 | 1.03 / 1.31 | 1.03 / 1.29 | 4.49 / 8.88 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f32 | 7 | loops | 629 / 246 | – | 1.00 / 1.00 | 0.86 / 1.01 | 0.86 / 1.01 | 2.59 / 5.47 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f32 | 8 | loops | 1271 / 412 | – | 1.00 / 1.00 | 0.74 / 0.94 | 0.74 / 0.94 | 2.15 / 5.52 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f64 | 1 | direct | 0.19 / 0.15 | 1.00 / 1.00 | 3.79 / 4.23 | 4.12 / 4.50 | 4.11 / 4.50 | 100.57 / 127.30 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f64 | 2 | direct | 0.89 / 0.53 | 1.00 / 1.00 | 3.78 / 3.68 | 3.17 / 4.42 | 3.16 / 4.42 | 52.46 / 83.72 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f64 | 3 | direct | 5.12 / 5.45 | 1.00 / 1.00 | 2.73 / 1.42 | 2.05 / 1.39 | 2.04 / 1.39 | 30.06 / 27.31 | direct | no strategy 10% faster than direct |
| PullBack | Hexahedron | Gpu | f64 | 4 | loops | 48.2 / 17.4 | n.m.; T3: 0.41 / 0.86 | 1.00 / 1.00 | 0.77 / 0.94 | 0.77 / 0.94 | 5.26 / 13.62 | loops | no strategy 10% faster than loops |
| PullBack | Hexahedron | Gpu | f64 | 5 | loops | 126 / 30.4 | – | 1.00 / 1.00 | 0.77 / 0.88 | 0.77 / 0.88 | 3.30 / 12.57 | sumfact | sumfact 0.77 / 0.88 of loops |
| PullBack | Hexahedron | Gpu | f64 | 6 | loops | 363 / 73.1 | – | 1.00 / 1.00 | 0.72 / 0.79 | 0.72 / 0.79 | 2.12 / 8.07 | sumfact | sumfact 0.72 / 0.79 of loops |
| PullBack | Hexahedron | Gpu | f64 | 7 | loops | 824 / 160 | – | 1.00 / 1.00 | 0.72 / 0.79 | 0.72 / 0.79 | 1.46 / 7.39 | sumfact | sumfact 0.72 / 0.79 of loops |
| PullBack | Hexahedron | Gpu | f64 | 8 | loops | 1482 / 309 | – | 1.00 / 1.00 | 0.75 / 0.77 | 0.75 / 0.77 | 1.37 / 5.63 | sumfact | sumfact 0.75 / 0.77 of loops |
| PullBack | Prism | CpuSimd128 | f32 | 1 | direct | 30.4 | 1.00 | 1.90 | 1.62 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f32 | 2 | direct | 61.9 | 1.00 | 2.01 | 1.29 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f32 | 3 | direct | 217 | 1.00 | 1.79 | 0.93 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f32 | 4 | direct | 601 | 1.00 | 3.39 | 3.26 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f32 | 5 | loops | 3089 | n.m.; T3: 0.68 | 1.00 | 0.45 | – | – | sumfact | sumfact 0.45 of loops |
| PullBack | Prism | CpuSimd128 | f32 | 6 | loops | 4899 | n.m.; T3: 0.71 | 1.00 | 0.43 | – | – | sumfact | sumfact 0.43 of loops |
| PullBack | Prism | CpuSimd128 | f32 | 7 | loops | 6924 | n.m.; T3: 0.74 | 1.00 | 0.41 | – | – | sumfact | sumfact 0.41 of loops |
| PullBack | Prism | CpuSimd128 | f32 | 8 | loops | 10122 | n.m.; T3: 0.73 | 1.00 | 0.38 | – | – | sumfact | sumfact 0.38 of loops |
| PullBack | Prism | CpuSimd128 | f64 | 1 | direct | 27.5 | 1.00 | 2.07 | 1.76 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f64 | 2 | direct | 66.7 | 1.00 | 1.93 | 1.03 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f64 | 3 | direct | 144 | 1.00 | 1.89 | 1.32 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f64 | 4 | direct | 336 | 1.00 | 3.28 | 3.14 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd128 | f64 | 5 | loops | 1879 | n.m.; T3: 0.49 | 1.00 | 0.45 | – | – | sumfact | sumfact 0.45 of loops |
| PullBack | Prism | CpuSimd128 | f64 | 6 | loops | 4241 | n.m.; T3: 0.44 | 1.00 | 0.43 | – | – | sumfact | sumfact 0.43 of loops |
| PullBack | Prism | CpuSimd128 | f64 | 7 | loops | 7697 | n.m.; T3: 0.43 | 1.00 | 0.45 | – | – | sumfact | sumfact 0.45 of loops |
| PullBack | Prism | CpuSimd128 | f64 | 8 | loops | 11009 | n.m.; T3: 0.42 | 1.00 | 0.38 | – | – | sumfact | sumfact 0.38 of loops |
| PullBack | Prism | CpuSimd512 | f32 | 1 | direct | 21.1 | 1.00 | 2.78 | 2.38 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f32 | 2 | direct | 85.8 | 1.00 | 2.33 | 1.39 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f32 | 3 | direct | 416 | 1.00 | 2.01 | 1.13 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f32 | 4 | direct | 1730 | 1.00 | 3.05 | 2.18 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f32 | 5 | loops | 8424 | n.m.; T3: 0.42 | 1.00 | 0.38 | – | – | sumfact | sumfact 0.38 of loops |
| PullBack | Prism | CpuSimd512 | f32 | 6 | loops | 13944 | n.m.; T3: 0.43 | 1.00 | 0.37 | – | – | sumfact | sumfact 0.37 of loops |
| PullBack | Prism | CpuSimd512 | f32 | 7 | loops | 19888 | n.m.; T3: 0.43 | 1.00 | 0.35 | – | – | sumfact | sumfact 0.35 of loops |
| PullBack | Prism | CpuSimd512 | f32 | 8 | direct | 18155 | 1.00 | 1.53 | 0.75 | – | – | sumfact | sumfact 0.75 of direct |
| PullBack | Prism | CpuSimd512 | f64 | 1 | direct | 25.3 | 1.00 | 3.40 | 2.34 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f64 | 2 | direct | 50.4 | 1.00 | 2.90 | 1.74 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f64 | 3 | direct | 213 | 1.00 | 2.42 | 1.25 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f64 | 4 | direct | 880 | 1.00 | 4.24 | 2.42 | – | – | direct | no strategy 10% faster than direct |
| PullBack | Prism | CpuSimd512 | f64 | 5 | loops | 4993 | n.m.; T3: 0.53 | 1.00 | 0.36 | – | – | sumfact | sumfact 0.36 of loops |
| PullBack | Prism | CpuSimd512 | f64 | 6 | loops | 12337 | n.m.; T3: 0.62 | 1.00 | 0.36 | – | – | sumfact | sumfact 0.36 of loops |
| PullBack | Prism | CpuSimd512 | f64 | 7 | loops | 23010 | n.m.; T3: 0.53 | 1.00 | 0.33 | – | – | sumfact | sumfact 0.33 of loops |
| PullBack | Prism | CpuSimd512 | f64 | 8 | direct | 17824 | 1.00 | 2.36 | 0.66 | – | – | sumfact | sumfact 0.66 of direct |
| PullBack | Prism | Gpu | f32 | 1 | direct | 0.27 / 0.20 | 1.00 / 1.00 | 3.89 / 4.79 | 3.94 / 4.64 | 3.90 / 4.64 | 24.93 / 32.64 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f32 | 2 | direct | 0.93 / 0.60 | 1.00 / 1.00 | 3.71 / 3.80 | 2.70 / 2.92 | 2.70 / 2.93 | 71.05 / 107.46 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f32 | 3 | direct | 5.74 / 3.54 | 1.00 / 1.00 | 1.75 / 1.59 | 1.61 / 2.39 | 1.61 / 2.38 | 42.07 / 65.68 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f32 | 4 | loops | 23.5 / 15.8 | n.m.; T3: 0.90 / 1.30 | 1.00 / 1.00 | 0.97 / 0.82 | 0.97 / 0.82 | 15.14 / 21.71 | loops | no strategy 10% faster than loops |
| PullBack | Prism | Gpu | f32 | 5 | loops | 66.8 / 36.4 | – | 1.00 / 1.00 | 0.68 / 0.74 | 0.68 / 0.74 | 8.69 / 14.16 | sumfact | sumfact 0.68 / 0.74 of loops, within 3% of shared (0.68 / 0.74) |
| PullBack | Prism | Gpu | f32 | 6 | loops | 158 / 72.9 | – | 1.00 / 1.00 | 0.64 / 0.75 | 0.64 / 0.75 | 5.29 / 10.26 | sumfact | sumfact 0.64 / 0.75 of loops, within 3% of shared (0.64 / 0.75) |
| PullBack | Prism | Gpu | f32 | 7 | loops | 302 / 132 | – | 1.00 / 1.00 | 0.68 / 0.92 | 0.68 / 0.92 | 4.05 / 7.93 | loops | no strategy 10% faster than loops |
| PullBack | Prism | Gpu | f32 | 8 | loops | 575 / 220 | – | 1.00 / 1.00 | 0.66 / 0.86 | 0.66 / 0.86 | 3.47 / 6.59 | sumfact | sumfact 0.66 / 0.86 of loops, within 3% of shared (0.66 / 0.86) |
| PullBack | Prism | Gpu | f64 | 1 | direct | 0.17 / 0.13 | 1.00 / 1.00 | 6.16 / 7.28 | 5.64 / 6.24 | 5.66 / 6.24 | 37.92 / 45.12 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f64 | 2 | direct | 0.65 / 0.37 | 1.00 / 1.00 | 7.48 / 6.72 | 3.47 / 4.36 | 3.44 / 4.36 | 71.82 / 121.36 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f64 | 3 | direct | 4.63 / 2.34 | 1.00 / 1.00 | 2.50 / 2.53 | 1.83 / 2.97 | 1.82 / 2.98 | 34.01 / 63.42 | direct | no strategy 10% faster than direct |
| PullBack | Prism | Gpu | f64 | 4 | loops | 42.2 / 17.0 | n.m.; T3: 0.34 / 0.69 | 1.00 / 1.00 | 0.54 / 0.72 | 0.53 / 0.72 | 5.64 / 13.42 | sumfact | sumfact 0.54 / 0.72 of loops, within 3% of shared (0.53 / 0.72) |
| PullBack | Prism | Gpu | f64 | 5 | loops | 125 / 33.1 | – | 1.00 / 1.00 | 0.49 / 0.50 | 0.49 / 0.50 | 3.63 / 10.76 | sumfact | sumfact 0.49 / 0.50 of loops, within 3% of shared (0.49 / 0.50) |
| PullBack | Prism | Gpu | f64 | 6 | loops | 314 / 48.7 | – | 1.00 / 1.00 | 0.46 / 0.67 | 0.46 / 0.67 | 2.25 / 10.90 | sumfact | sumfact 0.46 / 0.67 of loops, within 3% of shared (0.46 / 0.67) |
| PullBack | Prism | Gpu | f64 | 7 | loops | 598 / 104 | – | 1.00 / 1.00 | 0.50 / 0.69 | 0.50 / 0.69 | 1.33 / 7.23 | sumfact | sumfact 0.50 / 0.69 of loops, within 3% of shared (0.50 / 0.69) |
| PullBack | Prism | Gpu | f64 | 8 | loops | 1129 / 195 | – | 1.00 / 1.00 | 0.47 / 0.63 | 0.47 / 0.63 | 1.32 / 5.54 | sumfact | sumfact 0.47 / 0.63 of loops, within 3% of shared (0.47 / 0.63) |
| PullBack | Pyramid | CpuSimd128 | f32 | 1 | direct | 32.8 | 1.00 | 1.75 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd128 | f32 | 2 | direct | 56.5 | 1.00 | 2.17 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd128 | f32 | 3 | direct | 195 | 1.00 | 1.85 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd128 | f64 | 1 | direct | 26.5 | 1.00 | 2.05 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd128 | f64 | 2 | direct | 60.1 | 1.00 | 1.88 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd128 | f64 | 3 | direct | 117 | 1.00 | 1.96 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f32 | 1 | direct | 28.4 | 1.00 | 2.45 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f32 | 2 | direct | 84.7 | 1.00 | 2.70 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f32 | 3 | direct | 379 | 1.00 | 2.31 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f64 | 1 | direct | 28.2 | 1.00 | 3.27 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f64 | 2 | direct | 47.0 | 1.00 | 3.33 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | CpuSimd512 | f64 | 3 | direct | 199 | 1.00 | 2.27 | – | – | – | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f32 | 1 | direct | 0.26 / 0.20 | 1.00 / 1.00 | 4.32 / 5.13 | – | 4.33 / 5.13 | 28.38 / 35.89 | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f32 | 2 | direct | 0.80 / 0.50 | 1.00 / 1.00 | 4.28 / 4.58 | – | 4.27 / 4.57 | 37.71 / 55.91 | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f32 | 3 | direct | 3.48 / 1.47 | 1.00 / 1.00 | 2.53 / 3.22 | – | 2.52 / 3.23 | 29.42 / 67.69 | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f64 | 1 | direct | 0.16 / 0.13 | 1.00 / 1.00 | 6.22 / 7.33 | – | 6.22 / 7.34 | 50.66 / 60.77 | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f64 | 2 | direct | 0.64 / 0.38 | 1.00 / 1.00 | 6.40 / 5.89 | – | 6.42 / 5.90 | 46.90 / 72.85 | direct | no strategy 10% faster than direct |
| PullBack | Pyramid | Gpu | f64 | 3 | direct | 1.63 / 0.93 | 1.00 / 1.00 | 5.38 / 4.73 | – | 5.44 / 4.74 | 56.09 / 95.90 | direct | no strategy 10% faster than direct |

### Rule 1: p = 8 over p = 4 on tensor cells

The time per point at p = 8 over p = 4 of the strategy the policy selects at p = 8, the geometric mean over the class's devices, at each rule size, against the bounds of ADR-0042 item 1: 3 for the geometric factors and the pull-back's set-up, 1.1 · (9/5)^d for a Newton step (ADR-0049). Step = (t(9) − t(1)) / 8, set-up = t(1) − step. The bounds are reported, not enforced, in M9h; M10 takes them as its target (ADR-0053). Over M9: T3's `direct` time at p = 8 over the strategy's ("–" where `direct` has no baseline record, as Locust's hexahedron pull-back at p = 8, which did not compile).

| Class | Type | Cell | Family | Part | Strategy | Points | p = 8 / p = 4 | Bound | Within | Over M9 at p = 8 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| CpuSimd128 | f32 | Quadrilateral | Points | factor | sumfact | 10^6 | 1.42 | 3.00 | yes | 6.6 |
| CpuSimd128 | f32 | Quadrilateral | Jacobian | factor | sumfact | 10^6 | 1.57 | 3.00 | yes | 4.4 |
| CpuSimd128 | f32 | Quadrilateral | DetJ | factor | sumfact | 10^6 | 1.69 | 3.00 | yes | 4.4 |
| CpuSimd128 | f32 | Quadrilateral | Pinv | factor | sumfact | 10^6 | 1.56 | 3.00 | yes | 4.2 |
| CpuSimd128 | f32 | Quadrilateral | FacetNormals | factor | sumfact | 10^6 | 1.64 | 3.00 | yes | 5.6 |
| CpuSimd128 | f32 | Quadrilateral | Normals | factor | sumfact | 10^6 | 1.60 | 3.00 | yes | 4.1 |
| CpuSimd128 | f32 | Quadrilateral | PullBack | step | sumfact | 10^6 | 4.55 | 3.56 | no | 1.3 |
| CpuSimd128 | f32 | Quadrilateral | PullBack | set-up | sumfact | 10^6 | 0.13 | 3.00 | yes | 26.7 |
| CpuSimd128 | f32 | Hexahedron | Points | factor | sumfact | 10^6 | 5.54 | 3.00 | no | 82.3 |
| CpuSimd128 | f32 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 5.33 | 3.00 | no | 75.4 |
| CpuSimd128 | f32 | Hexahedron | DetJ | factor | sumfact | 10^6 | 5.53 | 3.00 | no | 77.0 |
| CpuSimd128 | f32 | Hexahedron | Pinv | factor | sumfact | 10^6 | 5.18 | 3.00 | no | 72.8 |
| CpuSimd128 | f32 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 3.50 | 3.00 | no | 59.9 |
| CpuSimd128 | f32 | Hexahedron | PullBack | step | sumfact | 10^6 | 6.45 | 6.42 | no | 2.1 |
| CpuSimd128 | f32 | Hexahedron | PullBack | set-up | sumfact | 10^6 | 8.50 | 3.00 | no | 35.2 |
| CpuSimd128 | f64 | Quadrilateral | Points | factor | sumfact | 10^6 | 3.59 | 3.00 | no | 8.1 |
| CpuSimd128 | f64 | Quadrilateral | Jacobian | factor | sumfact | 10^6 | 3.45 | 3.00 | no | 5.4 |
| CpuSimd128 | f64 | Quadrilateral | DetJ | factor | sumfact | 10^6 | 3.74 | 3.00 | no | 5.3 |
| CpuSimd128 | f64 | Quadrilateral | Pinv | factor | sumfact | 10^6 | 3.42 | 3.00 | no | 4.8 |
| CpuSimd128 | f64 | Quadrilateral | FacetNormals | factor | sumfact | 10^6 | 2.15 | 3.00 | yes | 4.7 |
| CpuSimd128 | f64 | Quadrilateral | Normals | factor | sumfact | 10^6 | 3.85 | 3.00 | no | 5.6 |
| CpuSimd128 | f64 | Quadrilateral | PullBack | step | sumfact | 10^6 | 3.09 | 3.56 | yes | 1.1 |
| CpuSimd128 | f64 | Quadrilateral | PullBack | set-up | sumfact | 10^6 | 0.56 | 3.00 | yes | 3.2 |
| CpuSimd128 | f64 | Hexahedron | Points | factor | sumfact | 10^6 | 1.67 | 3.00 | yes | 56.5 |
| CpuSimd128 | f64 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 1.74 | 3.00 | yes | 55.4 |
| CpuSimd128 | f64 | Hexahedron | DetJ | factor | sumfact | 10^6 | 1.79 | 3.00 | yes | 58.0 |
| CpuSimd128 | f64 | Hexahedron | Pinv | factor | sumfact | 10^6 | 1.60 | 3.00 | yes | 53.7 |
| CpuSimd128 | f64 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 1.78 | 3.00 | yes | 50.3 |
| CpuSimd128 | f64 | Hexahedron | PullBack | step | sumfact | 10^6 | 7.11 | 6.42 | no | 1.2 |
| CpuSimd128 | f64 | Hexahedron | PullBack | set-up | sumfact | 10^6 | 9.87 | 3.00 | no | 28.4 |
| CpuSimd512 | f32 | Quadrilateral | Points | factor | sumfact | 10^6 | 3.22 | 3.00 | no | 5.8 |
| CpuSimd512 | f32 | Quadrilateral | Jacobian | factor | sumfact | 10^6 | 3.21 | 3.00 | no | 2.3 |
| CpuSimd512 | f32 | Quadrilateral | DetJ | factor | sumfact | 10^6 | 3.54 | 3.00 | no | 2.6 |
| CpuSimd512 | f32 | Quadrilateral | Pinv | factor | sumfact | 10^6 | 3.23 | 3.00 | no | 2.4 |
| CpuSimd512 | f32 | Quadrilateral | FacetNormals | factor | sumfact | 10^6 | 2.23 | 3.00 | yes | 2.3 |
| CpuSimd512 | f32 | Quadrilateral | Normals | factor | sumfact | 10^6 | 2.75 | 3.00 | yes | 2.1 |
| CpuSimd512 | f32 | Quadrilateral | PullBack | step | direct | 10^6 | 8.65 | 3.56 | no | 1.0 |
| CpuSimd512 | f32 | Quadrilateral | PullBack | set-up | direct | 10^6 | 34.99 | 3.00 | no | 1.0 |
| CpuSimd512 | f32 | Hexahedron | Points | factor | sumfact | 10^6 | 1.57 | 3.00 | yes | 41.2 |
| CpuSimd512 | f32 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 1.11 | 3.00 | yes | 144.5 |
| CpuSimd512 | f32 | Hexahedron | DetJ | factor | sumfact | 10^6 | 1.87 | 3.00 | yes | 111.3 |
| CpuSimd512 | f32 | Hexahedron | Pinv | factor | sumfact | 10^6 | 1.09 | 3.00 | yes | 77.9 |
| CpuSimd512 | f32 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 1.96 | 3.00 | yes | 72.5 |
| CpuSimd512 | f32 | Hexahedron | PullBack | step | sumfact | 10^6 | 8.51 | 6.42 | no | 1.3 |
| CpuSimd512 | f32 | Hexahedron | PullBack | set-up | sumfact | 10^6 | 3.74 | 3.00 | no | 34.2 |
| CpuSimd512 | f64 | Quadrilateral | Points | factor | sumfact | 10^6 | 1.74 | 3.00 | yes | 3.6 |
| CpuSimd512 | f64 | Quadrilateral | Jacobian | factor | sumfact | 10^6 | 1.71 | 3.00 | yes | 2.3 |
| CpuSimd512 | f64 | Quadrilateral | DetJ | factor | sumfact | 10^6 | 2.37 | 3.00 | yes | 3.1 |
| CpuSimd512 | f64 | Quadrilateral | Pinv | factor | sumfact | 10^6 | 1.73 | 3.00 | yes | 2.4 |
| CpuSimd512 | f64 | Quadrilateral | FacetNormals | factor | sumfact | 10^6 | 1.46 | 3.00 | yes | 2.2 |
| CpuSimd512 | f64 | Quadrilateral | Normals | factor | sumfact | 10^6 | 1.64 | 3.00 | yes | 3.3 |
| CpuSimd512 | f64 | Quadrilateral | PullBack | step | direct | 10^6 | 7.46 | 3.56 | no | 1.0 |
| CpuSimd512 | f64 | Quadrilateral | PullBack | set-up | direct | 10^6 | 38.05 | 3.00 | no | 1.0 |
| CpuSimd512 | f64 | Hexahedron | Points | factor | sumfact | 10^6 | 8.88 | 3.00 | no | 29.5 |
| CpuSimd512 | f64 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 5.97 | 3.00 | no | 54.5 |
| CpuSimd512 | f64 | Hexahedron | DetJ | factor | sumfact | 10^6 | 11.05 | 3.00 | no | 83.4 |
| CpuSimd512 | f64 | Hexahedron | Pinv | factor | sumfact | 10^6 | 5.72 | 3.00 | no | 57.1 |
| CpuSimd512 | f64 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 3.92 | 3.00 | no | 59.9 |
| CpuSimd512 | f64 | Hexahedron | PullBack | step | direct | 10^6 | 25.57 | 6.42 | no | 0.9 |
| CpuSimd512 | f64 | Hexahedron | PullBack | set-up | direct | 10^6 | 35.67 | 3.00 | no | 1.0 |
| Gpu | f32 | Quadrilateral | Points | factor | shared | 10^6 | 1.49 | 3.00 | yes | 1.6 |
| Gpu | f32 | Quadrilateral | Points | factor | shared | 10^7 | 1.74 | 3.00 | yes | 1.5 |
| Gpu | f32 | Quadrilateral | Jacobian | factor | shared | 10^6 | 1.53 | 3.00 | yes | 4.1 |
| Gpu | f32 | Quadrilateral | Jacobian | factor | shared | 10^7 | 1.51 | 3.00 | yes | 6.1 |
| Gpu | f32 | Quadrilateral | DetJ | factor | shared | 10^6 | 1.63 | 3.00 | yes | 3.8 |
| Gpu | f32 | Quadrilateral | DetJ | factor | shared | 10^7 | 2.07 | 3.00 | yes | 4.5 |
| Gpu | f32 | Quadrilateral | Pinv | factor | shared | 10^6 | 1.65 | 3.00 | yes | 3.3 |
| Gpu | f32 | Quadrilateral | Pinv | factor | shared | 10^7 | 1.61 | 3.00 | yes | 4.4 |
| Gpu | f32 | Quadrilateral | FacetNormals | factor | shared | 10^6 | 1.55 | 3.00 | yes | 2.6 |
| Gpu | f32 | Quadrilateral | FacetNormals | factor | shared | 10^7 | 1.89 | 3.00 | yes | 3.2 |
| Gpu | f32 | Quadrilateral | Normals | factor | shared | 10^6 | 1.89 | 3.00 | yes | 6.3 |
| Gpu | f32 | Quadrilateral | Normals | factor | shared | 10^7 | 1.99 | 3.00 | yes | 7.7 |
| Gpu | f32 | Quadrilateral | PullBack | step | direct | 10^6 | 19.65 | 3.56 | no | 1.0 |
| Gpu | f32 | Quadrilateral | PullBack | set-up | direct | 10^6 | 10.66 | 3.00 | no | 1.0 |
| Gpu | f32 | Quadrilateral | PullBack | step | direct | 10^7 | 12.91 | 3.56 | no | 1.0 |
| Gpu | f32 | Quadrilateral | PullBack | set-up | direct | 10^7 | 10.22 | 3.00 | no | 1.0 |
| Gpu | f32 | Hexahedron | Points | factor | sumfact | 10^6 | 10.03 | 3.00 | no | 20.8 |
| Gpu | f32 | Hexahedron | Points | factor | sumfact | 10^7 | 5.08 | 3.00 | no | 46.1 |
| Gpu | f32 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 12.30 | 3.00 | no | 59.0 |
| Gpu | f32 | Hexahedron | Jacobian | factor | sumfact | 10^7 | 7.99 | 3.00 | no | 78.1 |
| Gpu | f32 | Hexahedron | DetJ | factor | sumfact | 10^6 | 13.97 | 3.00 | no | 71.9 |
| Gpu | f32 | Hexahedron | DetJ | factor | sumfact | 10^7 | 9.89 | 3.00 | no | 97.2 |
| Gpu | f32 | Hexahedron | Pinv | factor | sumfact | 10^6 | 11.83 | 3.00 | no | 36.0 |
| Gpu | f32 | Hexahedron | Pinv | factor | sumfact | 10^7 | 8.11 | 3.00 | no | 46.6 |
| Gpu | f32 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 7.56 | 3.00 | no | 28.6 |
| Gpu | f32 | Hexahedron | FacetNormals | factor | sumfact | 10^7 | 6.14 | 3.00 | no | 40.2 |
| Gpu | f32 | Hexahedron | PullBack | step | loops | 10^6 | 34.95 | 6.42 | no | – |
| Gpu | f32 | Hexahedron | PullBack | set-up | loops | 10^6 | 20.54 | 3.00 | no | – |
| Gpu | f32 | Hexahedron | PullBack | step | loops | 10^7 | 15.10 | 6.42 | no | – |
| Gpu | f32 | Hexahedron | PullBack | set-up | loops | 10^7 | 12.60 | 3.00 | no | – |
| Gpu | f64 | Quadrilateral | Points | factor | shared | 10^6 | 1.73 | 3.00 | yes | 4.9 |
| Gpu | f64 | Quadrilateral | Points | factor | shared | 10^7 | 1.85 | 3.00 | yes | 4.7 |
| Gpu | f64 | Quadrilateral | Jacobian | factor | shared | 10^6 | 1.77 | 3.00 | yes | 7.2 |
| Gpu | f64 | Quadrilateral | Jacobian | factor | shared | 10^7 | 1.61 | 3.00 | yes | 6.6 |
| Gpu | f64 | Quadrilateral | DetJ | factor | shared | 10^6 | 1.77 | 3.00 | yes | 8.2 |
| Gpu | f64 | Quadrilateral | DetJ | factor | shared | 10^7 | 1.81 | 3.00 | yes | 8.3 |
| Gpu | f64 | Quadrilateral | Pinv | factor | shared | 10^6 | 1.69 | 3.00 | yes | 6.8 |
| Gpu | f64 | Quadrilateral | Pinv | factor | shared | 10^7 | 1.76 | 3.00 | yes | 5.5 |
| Gpu | f64 | Quadrilateral | FacetNormals | factor | shared | 10^6 | 2.61 | 3.00 | yes | 2.2 |
| Gpu | f64 | Quadrilateral | FacetNormals | factor | shared | 10^7 | 4.15 | 3.00 | no | 1.6 |
| Gpu | f64 | Quadrilateral | Normals | factor | sumfact | 10^6 | 3.78 | 3.00 | no | 26.7 |
| Gpu | f64 | Quadrilateral | Normals | factor | sumfact | 10^7 | 5.16 | 3.00 | no | 14.2 |
| Gpu | f64 | Quadrilateral | PullBack | step | direct | 10^6 | 16.95 | 3.56 | no | 1.0 |
| Gpu | f64 | Quadrilateral | PullBack | set-up | direct | 10^6 | 6.13 | 3.00 | no | 1.0 |
| Gpu | f64 | Quadrilateral | PullBack | step | direct | 10^7 | 10.20 | 3.56 | no | 1.0 |
| Gpu | f64 | Quadrilateral | PullBack | set-up | direct | 10^7 | 5.55 | 3.00 | no | 1.0 |
| Gpu | f64 | Hexahedron | Points | factor | sumfact | 10^6 | 10.27 | 3.00 | no | 15.9 |
| Gpu | f64 | Hexahedron | Points | factor | sumfact | 10^7 | 4.61 | 3.00 | no | 16.5 |
| Gpu | f64 | Hexahedron | Jacobian | factor | sumfact | 10^6 | 41.70 | 3.00 | no | 60.7 |
| Gpu | f64 | Hexahedron | Jacobian | factor | sumfact | 10^7 | 9.82 | 3.00 | no | 52.4 |
| Gpu | f64 | Hexahedron | DetJ | factor | sumfact | 10^6 | 44.28 | 3.00 | no | 60.9 |
| Gpu | f64 | Hexahedron | DetJ | factor | sumfact | 10^7 | 9.98 | 3.00 | no | 53.4 |
| Gpu | f64 | Hexahedron | Pinv | factor | sumfact | 10^6 | 11.07 | 3.00 | no | 37.0 |
| Gpu | f64 | Hexahedron | Pinv | factor | sumfact | 10^7 | 3.55 | 3.00 | no | 33.8 |
| Gpu | f64 | Hexahedron | FacetNormals | factor | sumfact | 10^6 | 5.83 | 3.00 | no | 33.3 |
| Gpu | f64 | Hexahedron | FacetNormals | factor | sumfact | 10^7 | 3.02 | 3.00 | no | 30.2 |
| Gpu | f64 | Hexahedron | PullBack | step | sumfact | 10^6 | 30.26 | 6.42 | no | – |
| Gpu | f64 | Hexahedron | PullBack | set-up | sumfact | 10^6 | 11.04 | 3.00 | no | – |
| Gpu | f64 | Hexahedron | PullBack | step | sumfact | 10^7 | 14.59 | 6.42 | no | – |
| Gpu | f64 | Hexahedron | PullBack | set-up | sumfact | 10^7 | 5.68 | 3.00 | no | – |

### Rule 11: the policy against `direct` at p ≤ 4

In one paired run per device (the newest), per policy family (J + det J + K summed per run; the pull-back at 9 iterations), cell type, degree p ≤ 4 and rule size, where `direct` is selectable for the class (ADR-0055): a device's per-run ratios of the policy's run i over `direct`'s run i (`degree_sweep --interleave` alternates them on CPUs), their median and lower quartile, and the class's geometric means over its devices. A configuration fails if the lower quartile exceeds 1.05, that is, if the policy is over 5% slower in at least three quarters of the run pairs. Excluded: configurations where `direct` is not selectable (rule 7 on Gpu). CpuSimd128 includes Locust's Grace CPU (ADR-0051).

| Class | Type | Runs | Compared | Excluded | Missing | Worst median | At | Worst lower quartile | At | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |
| CpuSimd128 | f32 | macmini cpu paired-macmini-cpu-1794677271; locust cpu 20261114172752-62674 | 147 | 0 | 0 | 1.140 | FacetNormals Quadrilateral p=1 n=1000000 | 1.021 | FacetNormals Quadrilateral p=1 n=1000000 | none |
| CpuSimd128 | f64 | macmini cpu paired-macmini-cpu-1794677271; locust cpu 20261114172752-62674 | 147 | 0 | 0 | 1.052 | EVector Interval p=2 n=1000000 | 1.007 | Points Triangle p=2 n=1000000 | none |
| CpuSimd512 | f32 | kathleen cpu 634161 | 147 | 0 | 0 | 1.040 | PullBack Interval p=3 n=1000000 | 1.006 | EVector Quadrilateral p=3 n=1000000 | none |
| CpuSimd512 | f64 | kathleen cpu 634161 | 147 | 0 | 0 | 1.076 | FacetNormals Hexahedron p=1 n=1000000 | 1.004 | Jacobian Triangle p=1 n=1000000 | none |
| Gpu | f32 | macmini metal paired-macmini-metal-1794665922; locust cuda 20261114151543-4171256 | 288 | 6 | 0 | 1.097 | EVector Pyramid p=1 n=1000000 | 1.007 | PullBack Interval p=3 n=1000000 | none |
| Gpu | f64 | locust cuda 20261114151543-4171256 | 288 | 6 | 0 | 1.135 | EVector Quadrilateral p=2 n=1000000 | 1.032 | EVector Quadrilateral p=2 n=1000000 | none |

Reported, not bounded (ADR-0055): bench families the policy does not choose for on their own, over every configuration of the table above (`direct` selectable or not), the largest median class ratio of the policy over `direct`.

| Class | Type | Family | Largest median | At |
| --- | --- | --- | ---: | --- |
| CpuSimd128 | f32 | Jacobian | 1.059 | Jacobian Hexahedron p=1 n=1000000 |
| CpuSimd128 | f32 | DetJ | 1.010 | DetJ Pyramid p=2 n=1000000 |
| CpuSimd128 | f32 | Pinv | 1.017 | Pinv Hexahedron p=1 n=1000000 |
| CpuSimd128 | f32 | PullBack 1 it. | 1.016 | PullBack 1 it. Quadrilateral p=3 n=1000000 |
| CpuSimd128 | f32 | BasisEval | 1.047 | BasisEval Pyramid p=3 n=1000000 |
| CpuSimd128 | f64 | Jacobian | 1.029 | Jacobian Interval p=2 n=1000000 |
| CpuSimd128 | f64 | DetJ | 1.021 | DetJ Triangle p=2 n=1000000 |
| CpuSimd128 | f64 | Pinv | 1.009 | Pinv Interval p=1 n=1000000 |
| CpuSimd128 | f64 | PullBack 1 it. | 1.018 | PullBack 1 it. Quadrilateral p=3 n=1000000 |
| CpuSimd128 | f64 | BasisEval | 1.088 | BasisEval Prism p=1 n=1000000 |
| CpuSimd512 | f32 | Jacobian | 1.113 | Jacobian Quadrilateral p=2 n=1000000 |
| CpuSimd512 | f32 | DetJ | 1.009 | DetJ Interval p=2 n=1000000 |
| CpuSimd512 | f32 | Pinv | 1.020 | Pinv Interval p=2 n=1000000 |
| CpuSimd512 | f32 | PullBack 1 it. | 1.061 | PullBack 1 it. Interval p=1 n=1000000 |
| CpuSimd512 | f32 | BasisEval | 1.341 | BasisEval Hexahedron p=3 n=1000000 |
| CpuSimd512 | f64 | Jacobian | 1.019 | Jacobian Tetrahedron p=1 n=1000000 |
| CpuSimd512 | f64 | DetJ | 1.044 | DetJ Triangle p=1 n=1000000 |
| CpuSimd512 | f64 | Pinv | 1.013 | Pinv Pyramid p=2 n=1000000 |
| CpuSimd512 | f64 | PullBack 1 it. | 1.123 | PullBack 1 it. Interval p=3 n=1000000 |
| CpuSimd512 | f64 | BasisEval | 1.509 | BasisEval Hexahedron p=3 n=1000000 |
| Gpu | f32 | Jacobian | 1.009 | Jacobian Interval p=3 n=10000000 |
| Gpu | f32 | DetJ | 1.010 | DetJ Tetrahedron p=2 n=1000000 |
| Gpu | f32 | Pinv | 1.018 | Pinv Tetrahedron p=2 n=1000000 |
| Gpu | f32 | PullBack 1 it. | 1.195 | PullBack 1 it. Tetrahedron p=4 n=10000000 |
| Gpu | f32 | BasisEval | 3.956 | BasisEval Prism p=4 n=10000000 |
| Gpu | f64 | Jacobian | 1.008 | Jacobian Interval p=1 n=1000000 |
| Gpu | f64 | DetJ | 1.006 | DetJ Interval p=2 n=1000000 |
| Gpu | f64 | Pinv | 1.010 | Pinv Interval p=2 n=1000000 |
| Gpu | f64 | PullBack 1 it. | 1.622 | PullBack 1 it. Hexahedron p=4 n=1000000 |
| Gpu | f64 | BasisEval | 3.835 | BasisEval Prism p=4 n=10000000 |

### First-use compile and load times

The largest first-use compile and load time in seconds of each bench family and strategy over cell types, degrees, scalar types and vector sizes, from the launch-only records (rule 6; CUDA with `CUDA_CACHE_DISABLE=1`), with its cell type and degree. Gpu choice: the largest Locust time of the variants the policy selects for the Gpu class (bound 30 s, rule 7).

| Target | Backend | Family | direct | loops | sumfact | shared | cooperative | Gpu choice |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| kathleen | cpu | EVector | 0.4 (Hexahedron 4) | 0.0 (Quadrilateral 2) | 0.0 (Interval 6) | 0.0 (Tetrahedron 2) | 0.0 (Hexahedron 5) | – |
| kathleen | cpu | Points | 0.4 (Hexahedron 4) | 0.7 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Interval 7) | – |
| kathleen | cpu | Jacobian | 1.3 (Hexahedron 4) | 1.1 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Prism 1) | – |
| kathleen | cpu | DetJ | 1.2 (Hexahedron 4) | 1.1 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Hexahedron 7) | – |
| kathleen | cpu | Pinv | 1.4 (Hexahedron 4) | 1.1 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.1 (Tetrahedron 5) | – |
| kathleen | cpu | Normals | 0.4 (Quadrilateral 8) | 0.0 (Quadrilateral 6) | 0.1 (Quadrilateral 8) | 0.1 (Quadrilateral 8) | 0.1 (Quadrilateral 6) | – |
| kathleen | cpu | FacetNormals | 1.3 (Hexahedron 4) | 0.9 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.1 (Hexahedron 7) | – |
| kathleen | cpu | PullBack | 1.1 (Hexahedron 4) | 0.6 (Hexahedron 8) | 0.7 (Hexahedron 8) | 0.1 (Prism 7) | 0.1 (Hexahedron 8) | – |
| kathleen | cpu | BasisEval | 0.2 (Hexahedron 4) | 0.0 (Pyramid 3) | 0.0 (Hexahedron 6) | 0.0 (Triangle 3) | 0.1 (Pyramid 3) | – |
| locust | cuda | EVector | 61.8 (Hexahedron 8) | 0.1 (Hexahedron 2) | 0.0 (Triangle 7) | 0.0 (Quadrilateral 1) | 0.0 (Triangle 4) | 24.2 (Hexahedron 7 direct) |
| locust | cuda | Points | 25.1 (Hexahedron 8) | 0.4 (Hexahedron 8) | 22.9 (Hexahedron 8) | 24.2 (Hexahedron 8) | 0.4 (Hexahedron 8) | 22.9 (Hexahedron 8 sumfact) |
| locust | cuda | Jacobian | 45.6 (Hexahedron 8) | 0.6 (Hexahedron 8) | 25.9 (Hexahedron 8) | 26.4 (Hexahedron 8) | 0.6 (Hexahedron 8) | 25.9 (Hexahedron 8 sumfact) |
| locust | cuda | DetJ | 44.2 (Hexahedron 8) | 0.6 (Hexahedron 8) | 26.8 (Hexahedron 8) | 27.4 (Hexahedron 8) | 0.6 (Hexahedron 8) | 26.8 (Hexahedron 8 sumfact) |
| locust | cuda | Pinv | 45.8 (Hexahedron 8) | 0.6 (Hexahedron 8) | 26.1 (Hexahedron 8) | 26.4 (Hexahedron 8) | 0.6 (Hexahedron 8) | 26.1 (Hexahedron 8 sumfact) |
| locust | cuda | Normals | 0.6 (Quadrilateral 8) | 0.0 (Quadrilateral 1) | 0.6 (Quadrilateral 8) | 0.6 (Quadrilateral 8) | 0.1 (Quadrilateral 8) | 0.6 (Quadrilateral 8 sumfact) |
| locust | cuda | FacetNormals | 44.7 (Hexahedron 8) | 0.5 (Hexahedron 8) | 26.2 (Hexahedron 8) | 27.5 (Hexahedron 8) | 0.6 (Hexahedron 8) | 26.2 (Hexahedron 8 sumfact) |
| locust | cuda | PullBack | 138.0 (Hexahedron 4) | 0.4 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.6 (Hexahedron 8) | 0.4 (Hexahedron 8) | 29.0 (Hexahedron 3 direct) |
| locust | cuda | BasisEval | 3.6 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.1 (Hexahedron 7) | 0.1 (Pyramid 3) | 0.2 (Triangle 7 direct) |
| macmini | cpu | EVector | 960.5 (Hexahedron 8) | 0.0 (Hexahedron 8) | 0.0 (Hexahedron 1) | 0.0 (Triangle 4) | 0.0 (Hexahedron 7) | – |
| macmini | cpu | Points | 3.2 (Hexahedron 8) | 0.0 (Quadrilateral 2) | 0.2 (Hexahedron 8) | 0.0 (Hexahedron 8) | 0.0 (Interval 7) | – |
| macmini | cpu | Jacobian | 13.3 (Hexahedron 8) | 0.0 (Tetrahedron 6) | 0.2 (Hexahedron 8) | 0.0 (Hexahedron 8) | 0.0 (Hexahedron 8) | – |
| macmini | cpu | DetJ | 13.4 (Hexahedron 8) | 0.0 (Hexahedron 7) | 0.2 (Hexahedron 8) | 0.0 (Hexahedron 8) | 0.0 (Tetrahedron 2) | – |
| macmini | cpu | Pinv | 13.4 (Hexahedron 8) | 0.0 (Tetrahedron 2) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Tetrahedron 1) | – |
| macmini | cpu | Normals | 0.2 (Quadrilateral 8) | 0.0 (Triangle 3) | 0.0 (Quadrilateral 8) | 0.0 (Quadrilateral 8) | 0.0 (Triangle 3) | – |
| macmini | cpu | FacetNormals | 13.1 (Hexahedron 8) | 0.0 (Tetrahedron 1) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Hexahedron 8) | – |
| macmini | cpu | PullBack | 7.0 (Hexahedron 8) | 0.0 (Tetrahedron 8) | 0.2 (Hexahedron 8) | 0.0 (Pyramid 2) | 0.0 (Hexahedron 7) | – |
| macmini | cpu | BasisEval | 1.2 (Hexahedron 8) | 0.0 (Pyramid 3) | 0.0 (Hexahedron 7) | 0.0 (Prism 8) | 0.0 (Tetrahedron 7) | – |
| macmini | metal | EVector | 5.3 (Hexahedron 8) | 0.0 (Triangle 2) | 0.0 (Prism 8) | 0.0 (Interval 4) | 0.0 (Tetrahedron 2) | – |
| macmini | metal | Points | 2.9 (Hexahedron 8) | 0.0 (Tetrahedron 1) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 4) | 0.0 (Hexahedron 7) | – |
| macmini | metal | Jacobian | 14.4 (Hexahedron 8) | 0.0 (Tetrahedron 8) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Hexahedron 7) | – |
| macmini | metal | DetJ | 19.7 (Hexahedron 8) | 0.0 (Hexahedron 5) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Tetrahedron 8) | – |
| macmini | metal | Pinv | 15.8 (Hexahedron 8) | 0.0 (Tetrahedron 8) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Tetrahedron 5) | – |
| macmini | metal | Normals | 0.2 (Quadrilateral 8) | 0.0 (Triangle 7) | 0.0 (Quadrilateral 8) | 0.0 (Quadrilateral 8) | 0.0 (Triangle 5) | – |
| macmini | metal | FacetNormals | 14.9 (Hexahedron 8) | 0.0 (Tetrahedron 7) | 0.2 (Hexahedron 8) | 0.1 (Hexahedron 8) | 0.0 (Hexahedron 5) | – |
| macmini | metal | PullBack | 17.7 (Hexahedron 8) | 0.0 (Tetrahedron 4) | 0.2 (Hexahedron 8) | 0.0 (Hexahedron 8) | 0.1 (Hexahedron 7) | – |
| macmini | metal | BasisEval | 2.8 (Hexahedron 8) | 0.0 (Pyramid 3) | 0.0 (Quadrilateral 6) | 0.0 (Triangle 2) | 0.0 (Prism 8) | – |

### CUDA registers and local memory

Per bench family and strategy on Locust, over cell types, degrees, scalar types and vector sizes (`scripts/bench/cuda-resources.sh`, rule 6): the largest register count, spill and local (stack) bytes per thread, and the smallest maximum of threads per block the register count allows, with the cell type and degree of the largest stack.

| Family | Strategy | Kernels | Registers | Spill bytes | Local bytes | Threads per block | Largest stack at |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| EVector | direct | 102 | 255 | 0 | 0 | 256 | Pyramid 3 f64 |
| EVector | loops | 408 | 32 | 0 | 0 | 1024 | Pyramid 3 f64 |
| Points | direct | 102 | 255 | 37828 | 25368 | 256 | Hexahedron 8 f64 |
| Points | loops | 160 | 40 | 0 | 0 | 1024 | Tetrahedron 8 f64 |
| Points | sumfact | 48 | 255 | 101300 | 34920 | 256 | Hexahedron 8 f64 |
| Points | shared | 98 | 255 | 99540 | 44960 | 256 | Hexahedron 8 f64 |
| Points | cooperative | 102 | 40 | 20 | 16 | 1024 | Pyramid 2 f64 |
| Jacobian | direct | 102 | 255 | 69904 | 39664 | 256 | Hexahedron 8 f64 |
| Jacobian | loops | 163 | 48 | 0 | 0 | 1024 | Tetrahedron 8 f64 |
| Jacobian | sumfact | 48 | 255 | 143544 | 57400 | 256 | Hexahedron 8 f64 |
| Jacobian | shared | 95 | 255 | 132432 | 44696 | 256 | Hexahedron 8 f64 |
| Jacobian | cooperative | 102 | 64 | 20 | 16 | 1024 | Quadrilateral 8 f32 |
| DetJ | direct | 102 | 255 | 70080 | 39776 | 256 | Hexahedron 8 f64 |
| DetJ | loops | 163 | 40 | 20 | 24 | 1024 | Tetrahedron 8 f64 |
| DetJ | sumfact | 48 | 255 | 143556 | 57400 | 256 | Hexahedron 8 f64 |
| DetJ | shared | 95 | 255 | 132312 | 44696 | 256 | Hexahedron 8 f64 |
| DetJ | cooperative | 102 | 64 | 20 | 16 | 1024 | Quadrilateral 8 f32 |
| Pinv | direct | 102 | 255 | 68164 | 51656 | 256 | Hexahedron 8 f64 |
| Pinv | loops | 163 | 54 | 0 | 0 | 1024 | Tetrahedron 8 f64 |
| Pinv | sumfact | 48 | 255 | 144212 | 58624 | 256 | Hexahedron 8 f64 |
| Pinv | shared | 95 | 255 | 131476 | 56576 | 256 | Hexahedron 8 f64 |
| Pinv | cooperative | 102 | 64 | 20 | 16 | 1024 | Quadrilateral 8 f32 |
| Normals | direct | 48 | 255 | 8464 | 5408 | 256 | Quadrilateral 8 f64 |
| Normals | loops | 80 | 62 | 28 | 24 | 1024 | Triangle 8 f32 |
| Normals | sumfact | 16 | 255 | 5312 | 3864 | 256 | Quadrilateral 8 f64 |
| Normals | shared | 48 | 255 | 4848 | 3816 | 256 | Quadrilateral 8 f64 |
| Normals | cooperative | 48 | 64 | 0 | 0 | 1024 | Quadrilateral 8 f64 |
| FacetNormals | direct | 102 | 255 | 68336 | 51888 | 256 | Hexahedron 8 f64 |
| FacetNormals | loops | 161 | 54 | 0 | 0 | 1024 | Tetrahedron 8 f64 |
| FacetNormals | sumfact | 48 | 255 | 143436 | 58848 | 256 | Hexahedron 8 f64 |
| FacetNormals | shared | 97 | 255 | 132952 | 56424 | 256 | Hexahedron 8 f64 |
| FacetNormals | cooperative | 102 | 64 | 0 | 0 | 1024 | Pyramid 3 f64 |
| PullBack | direct | 78 | 255 | 107336 | 4024 | 256 | Hexahedron 4 f64 |
| PullBack | loops | 312 | 190 | 76 | 272 | 320 | Hexahedron 8 f64 |
| PullBack | sumfact | 96 | 126 | 40 | 4096 | 512 | Hexahedron 8 f64 |
| PullBack | shared | 204 | 190 | 40 | 4096 | 320 | Hexahedron 8 f64 |
| PullBack | cooperative | 204 | 186 | 196 | 416 | 320 | Hexahedron 8 f64 |
| BasisEval | direct | 102 | 255 | 2572 | 4672 | 256 | Prism 8 f64 |
| BasisEval | loops | 306 | 128 | 12 | 264 | 512 | Hexahedron 8 f64 |
| BasisEval | cooperative | 102 | 128 | 28 | 320 | 512 | Hexahedron 8 f64 |

<!-- END GENERATED -->
