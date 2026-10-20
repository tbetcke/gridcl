# M2 primitive throughput

The throughput of gridcl-device's radix sort and scans, per backend: the record that
docs/design.md asks of M2 ("the throughput of sorting 10^7 two-word keys is recorded per
backend"; Risks, sort and scan row: "benchmark in M2"). It is a baseline, not a tuning study:
every kernel runs at the vector sizes of the scalar-default `vector-policy.toml`, and M3s
decides where vectors help.

## Protocol

- **Binary:** `gridcl-bench/src/bin/primitive_throughput.rs`, release build:

  ```sh
  GRIDCL_TEST_DEVICE=<backend> cargo run --release -p gridcl-bench --features <backend> \
      --bin primitive_throughput -- --out <file>.json
  ```

- **Sort:** `radix_sort` of 10^7 keys of two u32 words, each word uniform random (SplitMix64,
  fixed seed), with an iota u32 payload. The time includes the allocation of the outputs and
  every pass; the inputs are on the device before timing starts.
- **Scan:** `inclusive_scan` and `exclusive_scan` (n + 1 outputs) of 10^7 u32 values, uniform in
  0..256 so the total does not wrap.
- **Timing:** one warm-up run (JIT compilation), whose result is checked against a host
  reference (sorted keys, stable permutation; exact prefix sums) before any time is reported;
  then the median of ten timed runs, each ending with `client.sync()`.
- **JSON:** one record per operation under `data/`: backend, device, operation, n, words,
  `median_s`, `items_per_s`, commit, and also the build profile, the `GRIDCL_VECTOR_SIZE`
  override (none in every run below) and the ten times (`times_s`).

## Results

Medians in milliseconds over ten runs, the range of the ten runs in brackets; rates in items
(keys or values) per second.

| Platform | Backend | Device | Sort, ms | Sort, keys/s | Inclusive scan, ms | Inclusive, values/s | Exclusive scan, ms | Exclusive, values/s | Data |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M3 Max | `cpu` | Apple M3 Max | 2086 (1910–2426) | 4.8e6 | 9.8 (8.6–15.7) | 1.0e9 | 10.1 (7.6–12.4) | 9.9e8 | `data/m2-primitives-cpu-m3max.json` |
| M3 Max | `wgpu<msl>` (Metal) | Apple M3 Max | 28.7 (28.6–47.8) | 3.5e8 | 1.14 (0.93–1.73) | 8.8e9 | 1.07 (1.00–1.53) | 9.3e9 | `data/m2-primitives-metal-m3max.json` |
| Hosted CI (software) | `wgpu<spirv>` (Vulkan, lavapipe) | llvmpipe (LLVM 20.1.2, 256 bits) | 46044 (45800–47700) | 2.2e5 | 936 (927–947) | 1.1e7 | 940 (926–948) | 1.1e7 | `data/m2-primitives-vulkan-lavapipe.json` |
| Locust | `cuda` | NVIDIA GH200 480GB | pending: node busy in both runs | | | | | | – |
| Locust | `cpu` | Neoverse-V2 (Grace) | pending: node busy in both runs | | | | | | – |
| Myriad V100 | `cuda`, `cpu` | V100, Xeon (36 cores) | pending: Myriad's scheduler refused jobs | | | | | | – |
| Myriad A100 | `cuda`, `cpu` | A100, Xeon (36 cores) | pending: Myriad's scheduler refused jobs | | | | | | – |

All rows were measured on 2026-10-20: the M3 Max CPU row at a8145b5, the others at 246dbae (the
binary is the same at both commits).

## Conditions

- **M3 Max:** release build, on mains power (`pmset`: AC power, battery charged), 12
  performance and 4 efficiency cores (`hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu`),
  64 GB. The CPU row was measured with no other test or build running (load average 1.6 before
  the run; Microsoft Defender's daemons and Outlook stayed open). A run just before it gave
  medians within 5 % for the sort (2.19 s) and within 40 % for the scans (13 to 14 ms); the
  scans, at about 10 ms, are sensitive to background work. A first CPU run that morning, with
  other sessions' tests running, gave twice the sort time and six to ten times the scan times,
  and was replaced. The Metal row was measured that morning with other work running; its times
  are tight all the same (sort 28.6 to 28.9 ms in nine of ten runs).
- **Hosted CI:** run 39396119752 of `.github/workflows/bench.yml` (push trigger), GitHub's
  `ubuntu-24.04` runner with 4 vCPUs of an AMD EPYC 7763, Mesa lavapipe (software Vulkan). The
  VM shares its host, and lavapipe is a CPU rasterizer: the row is labelled software and shows
  only that the primitives run there; it is not compared with hardware backends.
- **Locust:** jobs 20261020094911-3949350 and 20261020100636-3953887 (commit 246dbae; the tree
  was dirty only by these untracked JSON files). Both passed the result checks, but another
  user's GPU process held the GPU at 90 to 100 % utilisation before and after each run, so the
  times are discarded (ADR-0002) and the run is to be repeated when the node is idle.
- **Myriad:** the V100 and A100 jobs (`--cores 36`) could not be submitted: three `qsub` calls on
  2026-10-20 failed with "unable to send message to qmaster using port 6444". To be repeated when
  the scheduler is back.

## Observations

- On Metal, the sort of 10^7 two-word keys takes 29 ms (eight passes of 8-bit digits, about
  3.6 ms per pass) and a scan of 10^7 u32 values about 1.1 ms (80 MB of input and output, about
  70 GB/s). Both are well below the M3 Max's memory bandwidth; where the time goes is not
  measured here.
- The CPU runtime is about 70 times slower than Metal for the sort and about 9 times slower for
  the scans. Every kernel runs at vector size 1 and the sort's ranking runs through shared
  memory; M3s measures vector sizes and block sizes for SortDigits and ScanTile and decides from
  that.
- Topology creation sorts a few keys per cell, so at today's CPU-runtime rate a 10^6-cell mesh
  would spend about a second in sorting. This is the input the design's risk row asked
  for; it goes to M3s.
