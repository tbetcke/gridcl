# ADR-0061: The halo exchange's upload bound applies to GPU backends; the CPU runtime reports its ratio

- **Status:** Accepted
- **Date:** 2026-11-17
- **Raised in:** M10 T5 (host-staged exchange hardening), before T6 (`halo_exchange` benchmark and records)
- **Design sections affected:**
  - "Implementation roadmap", M10 Hardening, acceptance bullet 1;
  - "Distributed mesh model", the paragraph after Listing S2c that lists `Scatterer`'s device steps (the description of `reverse`).

## Context

M10's acceptance bullet 1 (as amended by ADR-0059) reads:

> the host-staged halo exchange (`forward` and `reverse`, halos of 10^3, 10^4 and 10^5 nodes with block 3) is recorded on the local Mac (CPU runtime, Metal) and Locust (CUDA, 1 and 2 ranks); for 10^5 nodes, its host-to-device copy takes at most 1.5 times its device-to-host copy and `reverse` at most twice `forward` on each, and results stay bitwise identical to M9h's (ADR-0059);

`gridcl-bench/tests/perf_record.rs::halo_exchange_recorded` (M10 T0) checks both bounds on every platform's records at 10^5 nodes. The host-to-device copy (H2D) is `DeviceBuffer::write_host` of the received values, the device-to-host copy (D2H) is `DeviceBuffer::to_host` of the packed send buffer: 3 · 10^5 values, 2.4 MB in f64 and 1.2 MB in f32.

The paragraph after Listing S2c describes `reverse` as "packs and exchanges the ghost values, gathers [owner values by `rev_targets` | received values] by `rev_order`, reduces over `rev_segments` and scatters the result to the targets (ADR-0026)".

## Problem

**The upload bound cannot hold on the CPU runtime.** T5 measured each copy alone (throwaway probe; medians of 50 after 5 warm-ups; each upload followed by `Client::sync`; docs/plans/M10.md, Deviations log of 2026-11-17):

| Platform | Values | `write_host` | `to_host` | Ratio | An owned buffer handed to `Client::write` |
| --- | --- | ---: | ---: | ---: | ---: |
| Locust H100, CUDA (node idle) | 3 · 10^5 f64 | 0.43 ms | 0.32 ms | 1.37 | – |
| Mac mini, Metal | 3 · 10^5 f32 | 0.41 ms | 0.30 ms | 1.35 | 0.40 ms |
| Mac mini, CPU runtime | 3 · 10^5 f64 | 0.08–0.13 ms | 0.027–0.030 ms | 3.0–4.4 | 0.07 ms (2.6) |

On the GPU backends, T5's staged `write_host` meets the bound (on CUDA it took 2.6–2.9 ms before, C23). On the CPU runtime, device memory is host memory:
- `Client::read_one` returns a view of the runtime's storage (0.003 ms for 2.4 MB), so `to_host` costs one copy, the decode into a `Vec`;
- an upload costs at least two: `write_host` copies the caller's slice into an owned buffer (its signature, Listing S1, takes `&[T]`), and the runtime copies that buffer into its storage.

Even with gridcl's copy removed, which would need a new public `DeviceBuffer` method taking an owned buffer, the ratio stays at about 2.6. No change inside gridcl brings it to 1.5. The bound compares two different kinds of work on this runtime and measures nothing about the exchange.

The other half of the bullet holds everywhere. `reverse` against `forward` at 10^5 nodes, 2 ranks:
- CPU runtime, f64: 1.5–1.8 (2.2–2.6 ms against 1.5 ms; before T5, 34–51 ms, C24);
- Metal, f32: 1.3;
- CUDA, f64: 1.15 (Locust job 20261117121850-1159598, node busy before, so a guide only; T6 records it under the idle rule).

**The design's description of `reverse` is out of date.** T5 replaced the `gather` by `rev_order` with element-wise `unpack`s that place the owner values and the received values directly in the reduce's order. It replaced the final `scatter` by an `unpack` over the (distinct) targets. On the Cpu class, each step waits for the previous one (C24). The reduce sees the same values in the same order, so the results are bitwise unchanged. Every step is still a gridcl-device primitive.

## Options

1. **Keep the design as written.** The bullet then fails on the CPU runtime with any implementation gridcl can make on CubeCL 0.11: even an owned-buffer upload, a public API change, is 2.6 times the read. M10 could not close.
2. **Apply the upload bound to the GPU backends; record and report the CPU runtime's ratio (proposed).** The bound stays for Metal and CUDA, where both copies cross a host–device boundary and T5's staged upload meets it. On the CPU runtime, both copies are recorded as planned, and docs/perf/halo-exchange.md gives the ratio at 10^5 nodes with its cause (the read is a view of the runtime's storage). `reverse` ≤ 2 × `forward` stays on every platform.
3. **Compare the CPU runtime's upload with a host copy instead.** For example, require H2D ≤ 1.5 × (D2H + a `memcpy` of the same bytes). This keeps a number on the CPU runtime, but the reference is arbitrary and is not part of the exchange.
4. **Drop the upload bound everywhere.** This loses the check that motivated ADR-0059 on the platforms where it matters (on CUDA the upload was 9 times the download before T5).

## Proposed decision

Option 2. The bound exists to keep the host-staged exchange's upload as fast as its download where the two cross a bus. That is true of Metal and CUDA, and T5 meets it there. On the CPU runtime, the download is a view plus one copy and the upload is two copies, so the ratio says nothing about the exchange. It is reported with its cause instead, as bullet 2 does for the degree ratios (ADR-0060 item 5). The description of `reverse` is updated to the steps T5 implemented, with the same fixed order.

## Consequences

- **Text to change in docs/design.md:**
  - "Implementation roadmap", M10 Hardening, acceptance bullet 1: "for 10^5 nodes, its host-to-device copy takes at most 1.5 times its device-to-host copy and `reverse` at most twice `forward` on each" becomes "for 10^5 nodes, its host-to-device copy takes at most 1.5 times its device-to-host copy on Metal and CUDA (on the CPU runtime, whose read returns a view of the runtime's storage, both are recorded and docs/perf/halo-exchange.md reports the ratio; ADR-0061), and `reverse` at most twice `forward` on each".
  - "Distributed mesh model", the paragraph after Listing S2c: "`reverse` packs and exchanges the ghost values, gathers [owner values by `rev_targets` | received values] by `rev_order`, reduces over `rev_segments` and scatters the result to the targets (ADR-0026)" becomes "`reverse` packs and exchanges the ghost values, uploads only the received values, places the targets' own values and the received values in the reduce's order by `unpack` (each segment the owner's value, then its contributions by neighbour rank and ghost position), reduces over `rev_segments` and unpacks the result to the targets; on the Cpu class each step waits for the previous one (ADR-0026, ADR-0059, ADR-0061)".
- **Public API changes:** none.
- **Acceptance tests affected:**
  - `gridcl-bench/tests/perf_record.rs::halo_exchange_recorded` applies `h2d ≤ 1.5 × d2h` to the `metal` and `cuda` records only. For the `cpu` records, it checks that docs/perf/halo-exchange.md has a CPU-runtime row with the 10^5-node ratio and a cause. The `reverse` ≤ 2 × `forward` check is unchanged.
  - The test changes in M10 T6, with a Deviations-log row.
  - No other test changes.
- **Later milestones affected:**
  - M10 T6 writes the report row.
  - Issue #25's bullet 1 is updated to the new text, with a comment naming this ADR.
  - No other milestone.

## Decision record

Accepted by the maintainer on 2026-11-17, as proposed (option 2).

Applied in `adr: 0061 apply`: docs/design.md ("Implementation roadmap", M10 acceptance bullet 1; "Distributed mesh model", the paragraph after Listing S2c) and issue #25. The test change is M10 T6's.
