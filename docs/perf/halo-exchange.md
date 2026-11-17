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

<!-- Filled in with the records (M10 T6). -->

## M10 records

<!-- BEGIN GENERATED M10; do not edit -->

At 100000 halo nodes:

| target backend | type | ranks | d2h µs | h2d µs | h2d / d2h | reverse / forward | bound | cause |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |

<!-- END GENERATED M10 -->
