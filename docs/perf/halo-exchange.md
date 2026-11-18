# The host-staged halo exchange (M10)

This report records the host-staged halo exchange of `Scatterer` before and after M10's
hardening (docs/design.md, "Distributed mesh model", "Transports"; ADR-0059, ADR-0061). The
tables under "M10 records" are written by `cargo run -p gridcl-bench --bin halo_exchange --
report` from `docs/perf/data/m10-halo-*.json`, between its marker comments; the rest is written
by hand.

## Protocol

- **Binary:** `halo_exchange` (gridcl-bench), release build, with the feature `mpi` for more
  than one rank.
- **Set-up:** each rank owns 10^6 nodes of 3 values and ghosts h = 10^3, 10^4 and 10^5 nodes
  of the next rank, spread evenly over its owned nodes; on two ranks each rank sends and
  receives one halo.
- **Parts:** the steps of `Scatterer::forward` one by one, with the map's own send and receive
  lists: the pack kernel, `to_host` of the packed blocks (D2H), the host
  `MPI_Neighbor_alltoallv` of the same bytes, `write_host` of the received blocks (H2D) and the
  unpack kernel; then the whole `forward` and `reverse` (`Add`). On one rank there is nothing
  to exchange: pack, D2H, H2D and unpack move the same blocks between the owned nodes and a
  buffer.
- **Timing:** 5 untimed and 50 timed runs per part, each after a barrier and followed by a
  device sync; a run's time is the largest over the ranks; the median of the 50.
- **Bounds (bullet 1, ADR-0061):** at 10^5 nodes, H2D ≤ 1.5 × D2H on Metal and CUDA, and
  `reverse` ≤ 2 × `forward` on every platform. On the CPU runtime the H2D / D2H ratio is
  reported with its cause: the runtime's read returns a view of its storage, so `to_host` costs
  one copy (the decode into a `Vec`) and `write_host` at least two (into an owned buffer, then
  into the runtime's storage).

## Baseline (ADR-0059)

Measured on 2026-11-16 at main ae8d644 by a throwaway probe with the same set-up, before M10
(medians in µs; full tables in ADR-0059):

| Device | h | forward | reverse | D2H | H2D |
| --- | --- | ---: | ---: | ---: | ---: |
| Locust CUDA f64, two ranks on one GPU | 10^3 / 10^4 / 10^5 | 470 / 527 / 1,232 | 504 / 729 / 2,015 | 102 / 114 / 320 | 149 / 40 / 1,564 |
| Locust GH200 f64, one rank | 10^5 | – | – | 317 | 3,173 |
| Mac mini, CPU runtime, f64 | 10^3 / 10^4 / 10^5 | 91 / 252 / 923 | 42,261 / 44,594 / 35,621 | 5 / 8 / 33 | 6 / 25 / 309 |
| Mac mini, Metal, f32 | 10^3 / 10^4 / 10^5 | 734 / 731 / 966 | 725 / 746 / 1,397 | 216 / 321 / 318 | 353 / 342 / 396 |

What M10 T5 changed (docs/plans/M10.md, Deviations log): `write_host` stages through
`Client::staging` on GPUs (CUDA's pageable upload was 9 times the download); each `Scatterer`
reuses its buffers; `reverse` uploads only the received values and places them and the
targets' own values in the reduce's order by `unpack`; on the Cpu class, `reverse` waits for
each kernel, because the CPU runtime's workers spin on queued kernels that are not ready (C24),
which caused its 35–45 ms.

## Conditions

All records at commit 5003a91, release build, 2026-11-17 (M10 T6), through
`scripts/bench/halo-exchange.sh`:

- **Mac mini (M6), CPU runtime f64 and Metal f32, 2 ranks:** Open MPI 5.0.11 with mpi-test.sh's
  hwloc workaround (ADR-0034); the load average was 2.8 when the CPU run started. The CPU runtime's `reverse` / `forward` of 1.90 is near its bound:
  the ratio of the quartiles spans 1.80–2.03, and `mpi` and `h2d` are bimodal at 10^5 nodes.
- **Locust, CUDA f64 on the GH200, 1 and 2 ranks:** one job (69 s), node load `idle` before
  and after (ADR-0002). Locust has one GPU, so the two ranks share it; with no MPS the two
  processes time-slice it, and every kernel costs about 315 µs against 17 µs on one rank
  (`pack` and `unpack` above). The upload bound applies to the 1-rank run, and the 2-rank
  ratio is reported with its cause (ADR-0062).

Against the baseline at 10^5 nodes: CUDA's upload on one rank fell from 3,173 to 403 µs
(`write_host` staged); the CPU runtime's `reverse` from 35.6 ms to 2.7 ms (C24); Metal's
`forward` and `reverse` are about the same (1,075 and 1,406 µs).

## M10 records

<!-- BEGIN GENERATED M10; do not edit -->

**locust cuda, f64, 1 rank** (NVIDIA GH200 480GB; commit 5003a91b7803). Median µs:

| part | 1000 | 10000 | 100000 |
| --- | ---: | ---: | ---: |
| pack | 16.5 | 16.0 | 17.5 |
| d2h | 23.6 | 34.1 | 334.1 |
| h2d | 19.2 | 32.6 | 403.3 |
| unpack | 16.1 | 15.9 | 17.7 |

**locust cuda, f64, 2 ranks** (NVIDIA GH200 480GB; commit 5003a91b7803). Median µs:

| part | 1000 | 10000 | 100000 |
| --- | ---: | ---: | ---: |
| pack | 316.4 | 316.0 | 316.0 |
| d2h | 28.6 | 189.5 | 336.7 |
| mpi | 15.3 | 173.9 | 310.7 |
| h2d | 321.5 | 177.9 | 634.3 |
| unpack | 315.2 | 315.3 | 315.2 |
| forward | 654.9 | 755.5 | 1554.0 |
| reverse | 987.0 | 776.1 | 1567.8 |

**macmini cpu, f64, 2 ranks** (Apple M6; commit 5003a91b7803). Median µs:

| part | 1000 | 10000 | 100000 |
| --- | ---: | ---: | ---: |
| pack | 41.3 | 71.5 | 614.9 |
| d2h | 6.0 | 8.6 | 36.8 |
| mpi | 5.2 | 12.6 | 307.5 |
| h2d | 4.4 | 17.5 | 359.3 |
| unpack | 41.9 | 52.6 | 253.2 |
| forward | 122.7 | 226.2 | 1409.5 |
| reverse | 333.7 | 649.6 | 2681.0 |

**macmini metal, f32, 2 ranks** (Apple M6; commit 5003a91b7803). Median µs:

| part | 1000 | 10000 | 100000 |
| --- | ---: | ---: | ---: |
| pack | 325.8 | 312.2 | 309.7 |
| d2h | 417.8 | 414.2 | 411.9 |
| mpi | 13.3 | 16.7 | 49.7 |
| h2d | 338.6 | 343.5 | 413.4 |
| unpack | 316.1 | 314.4 | 308.8 |
| forward | 693.0 | 726.8 | 1075.0 |
| reverse | 740.2 | 780.4 | 1406.0 |

At 100000 halo nodes:

| target backend | type | ranks | d2h µs | h2d µs | h2d / d2h | reverse / forward | bound | cause |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| locust cuda | f64 | 1 | 334.1 | 403.3 | 1.21 | – | h2d ≤ 1.5 × d2h; reverse ≤ 2 × forward | – |
| locust cuda | f64 | 2 | 336.7 | 634.3 | 1.88 | 1.01 | reverse ≤ 2 × forward; h2d reported | the ranks share one GPU and time-slice it, so each kernel waits for a slice (see pack and unpack against one rank), and the staged upload waits on the GPU more often than the download (ADR-0062) |
| macmini cpu | f64 | 2 | 36.8 | 359.3 | 9.76 | 1.90 | reverse ≤ 2 × forward; h2d reported | the CPU runtime's read returns a view of its storage, so `to_host` is one copy and `write_host` at least two (ADR-0061) |
| macmini metal | f32 | 2 | 411.9 | 413.4 | 1.00 | 1.31 | h2d ≤ 1.5 × d2h; reverse ≤ 2 × forward | – |

<!-- END GENERATED M10 -->
