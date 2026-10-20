# ADR-0011: M2 review clarifications: finite Min/Max identities, fill launches for the fixed-size outputs of empty inputs, vector sizes clamped by `max_vector_size`

- **Status:** Accepted
- **Date:** 2026-10-20
- **Raised in:** M2 review (design-reviewer, Major 1 and Minor 5); the T0, T2, T4, T5, T8 and T9 rows of the M2 Deviations log
- **Design sections affected:**
  - "Device model and backends": Listing S1b (`reduce`, `segmented_reduce`) and the paragraph after it ("An empty input never launches a kernel");
  - "Vectorization and CPU execution": Listing S2 (`VectorSizeOverride`), the "Vector size" and "CPUs" bullets after it, and "Padding and tiles";
  - "Distributed mesh model": Listing S2c (`CommDyn::exscan_bytes`);
  - ADR-0009, "Empty inputs" (amended, not superseded).

## Context

M2 implemented three behaviours that the design text does not state, or states differently. Each
one is logged in the M2 Deviations log, but none has an ADR. The milestone review flagged them
because later milestones rely on them.

1. **Identity of Min and Max.** Listing S1b says of `reduce`: "an empty input gives the identity of
   `op`". Listing S2c says of `exscan_bytes`: "rank 0 receives the identity of `op`". For floats
   the mathematical identities are +∞ (Min) and −∞ (Max). The implementation uses
   `T::max_value()` and `T::min_value()` instead, which for floats are the largest finite values
   (`f32::MAX`, `f64::MAX`). This applies to the identity of an empty reduce, an empty segment, a
   masked lane and `Serial::exscan_bytes` (T0 and T9 rows).
2. **Empty inputs.** The design says twice: "An empty input never launches a kernel" (the paragraph
   after Listing S1b, and "Padding and tiles"). ADR-0009 says the same. Three primitives have an
   output of fixed size even when their input is empty, and they write it with the fill kernel:
   - `reduce` writes `[identity]`;
   - `exclusive_scan` writes `[0]`;
   - `run_length` writes the offsets `[0]`.

   No kernel runs over the empty input itself (T4, T5 and T8 rows). The list is not complete;
   see the amendment in the decision record.
3. **Vector-size clamps.** Listing S2 says `Fixed(n)` and `Widest` "skip the register budget but
   stay clamped by max_vector". The "Vector size" bullet lists the policy size, `max_vector` and
   the register budget. The implementation also clamps every vector size, `Fixed(n)` and `Widest`
   included, by the device's `hardware.max_vector_size`. On the CPU it also clamps `cube_dim` by
   `max_cube_dim.0` (T2 row).

## Problem

1. **Infinite identities.** A device identity of ±∞ would need infinity constants in CubeCL
   kernels, for every `DeviceScalar`. `Numeric::max_value()` and `min_value()` are what CubeCL
   0.11 provides, and they are finite. ±∞ also has no meaning for the integer types, so i32 and u32
   would need their own rule anyway.

   The finite values are an identity for every finite input. They differ from ±∞ only in what an
   empty input returns. M3 reductions and M6 `Scatterer::reverse` (Min/Max over empty segments)
   will see these values, so the choice is visible beyond M2.
2. **Empty fixed-size outputs.** The output has to hold a value on the device. "Never launch" would
   leave only a host upload to write it, and CLAUDE.md's host-copy rule excludes that. As written,
   the design cannot be followed for these three primitives.
3. **`max_vector_size`.** A launch above the device's vector limit fails, so a forced
   `Fixed(16)` on a 128-bit device has to be clamped. The same holds for a CPU `cube_dim` above
   `max_cube_dim.0`. The design's own check, "Every plan is checked against `max_cube_dim`",
   implies the second clamp but not the first.

## Options

1. **Keep the design as written.**
   - Infinite float identities in kernels; integers keep MAX/MIN.
   - A host upload for the three fixed-size outputs, against CLAUDE.md, or else an ADR that
     changes the host-copy rule.
   - An unclamped `Fixed(n)`, which fails at launch on narrow devices.

   Not chosen: it needs kernel infinity constants, it breaks the host-copy rule, and it makes
   `GRIDCL_VECTOR_SIZE=16` unusable on Metal and the M3 Max CPU.
2. **Record the implemented behaviour** (proposed).
   - (a) The identities of Sum, Min and Max are 0, `T::max_value()` and `T::min_value()`, for
     floats the largest finite values. This holds on the device and in `exscan_bytes`.
   - (b) An empty input never launches a kernel *over that input*. An output whose size does not
     depend on it is written by the fill kernel, for example `reduce` → `[identity]`,
     `exclusive_scan` → `[0]`, `run_length` offsets → `[0]` (amended: see the decision record).
   - (c) Every vector size, `Fixed(n)` and `Widest` included, is also clamped by
     `hardware.max_vector_size`. The CPU `cube_dim` is also clamped by `max_cube_dim.0`.
3. **Infinite float identities only** (a'), with (b) and (c) as in option 2.
   - Adds infinity constants to the kernels and to `DType`'s byte-level identities.
   - Changes the M2 tests' expected values for empty float inputs.
   - Gains the mathematically exact identity for empty float reductions.

## Proposed decision

Option 2. It documents what M2 already does and tests. It keeps the kernels free of non-finite
constants and keeps CLAUDE.md's host-copy rule intact. Every primitive and collective also gets
one identity rule for floats and integers alike. A later milestone that needs ±∞ for empty float
inputs can raise its own ADR.

## Consequences

- **Text to change in docs/design.md:**
  - Listing S1b, `reduce`: "Length 1; an empty input gives the identity of `op` (0,
    `T::max_value()`, `T::min_value()`; finite for floats), written by the fill kernel."
    `segmented_reduce`: "an empty segment gives the identity of `op`."
  - The paragraph after Listing S1b: "An empty input never launches a kernel over that input; a
    fixed-size output (reduce, exclusive scan, the offsets of run-length) is written by the fill
    kernel."
  - "Padding and tiles": the same sentence.
  - Listing S2, `VectorSizeOverride`: "…but stay clamped by max_vector and the device's
    `max_vector_size`". The "Vector size" bullet adds `hardware.max_vector_size`. The "CPUs" bullet:
    "`cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`)".
  - Listing S2c, `exscan_bytes`: "rank 0 receives the identity of `op` (0, MAX, MIN; finite for
    floats)".
- **ADR-0009:** add an "Amended by ADR-0011" line to its "Empty inputs" section.
- **Public API changes:** none. The signatures are unchanged. The rustdoc already states (a) and
  (c); (b) is not visible through the API. Only the design text changes.
- **Acceptance tests affected:** none. The M2 tests already assert these values.
- **Later milestones affected:**
  - M3 reductions over cells and M6 `Scatterer::reverse` (Min/Max of empty segments) inherit (a).
  - M6 `Mpi::exscan_bytes` must write the same identities on rank 0, since MPI leaves rank 0's
    buffer undefined.

## Decision record

Accepted by the maintainer on 2026-10-20 (option 2, decisions (a)–(c)), without conditions. Applied to docs/design.md in `adr: 0011 apply`.

**Amendment, 2026-10-20 (approved by the maintainer at the M2 review).** The design-reviewer's follow-up found that the list in (b) read as complete but was not:
- `histogram` of empty values returns `zeros(num_bins)`, a fill launch;
- `gather` from an empty `src` returns `zeros(idx.len())`;
- `segmented_reduce` of empty values with non-empty offsets launches its kernel over the segments. It binds the empty `values` buffer but reads nothing from it.

The Context's original sentence, "The empty input itself is never bound", was therefore wrong for `segmented_reduce`. Decision (b) now reads: an empty input never launches a kernel over that input. Outputs whose size does not depend on it are still written:
- by the fill kernel, for example in `reduce`, `exclusive_scan`, the offsets of `run_length`, `histogram`, and `gather` from an empty source;
- or by a kernel over another, non-empty input, which may bind the empty buffer but reads nothing from it (`segmented_reduce`).

The two sentences in docs/design.md (the paragraph after Listing S1b, and "Padding and tiles") changed accordingly, in `adr: 0011 amend`. Code and tests are unchanged.
