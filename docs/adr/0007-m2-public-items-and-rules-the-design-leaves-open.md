# ADR-0007: M2 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-19
- **Raised in:** M2 planning (docs/plans/M2.md, questions 2–6 and 11–13)
- **Design sections affected:**
  - "Device model and backends": Listing S1, "Data layouts and index widths", "Device primitives";
  - "Vectorization and CPU execution": Listing S2 (launch plans), "Padding and tiles", "Where vectors apply";
  - "Distributed mesh model": the communicator listing (also called Listing S2);
  - "Testing and validation" (vector-size invariance);
  - "Workspace layout" (crate table);
  - "Handing this design to Claude Code" (the CLAUDE.md listing: kernels, SIMD, host copies).

## Context

M2 builds gridcl-device's primitives and gridcl-parallel's communication core. M3's topology
construction is the first user of both. The design fixes the containers (Listing S1), the launch
plan (Listing S2) and the communicator traits. For the primitives it gives only a list and five
rules ("Device primitives"):

> inclusive and exclusive scan, built as reduce-then-scan; stable LSD radix sort over keys of one
> to four u32 words with 8-bit digits and a u32 payload; run-length encoding and unique-by-key;
> histogram, …; gather and scatter; segmented reduce (sum, min, max) for f32, f64, i32 and u32;
> fill and copy.

M2's acceptance tests need two things the design does not specify:
- **Injected device properties:** "with injected device properties, no plan exceeds …"; "with an
  injected limit, an allocation above `max_page_size` returns `AllocationTooLarge`".
- **Forced vector sizes:** "every vector-generic primitive gives the same results at vector size 1
  and at every larger power of two up to 16".

While planning M2, the CubeCL 0.11.0 sources were read. They differ from the design in three ways:
- **Kernel slice length.** `BufferArg::from_raw_parts(handle, length)` does not set the length of the kernel's slice. The kernel sees `handle.size_in_used() / size_of::<C>()` (`cubecl-core-0.11.0/src/compute/launcher.rs:202`). The `length` argument only feeds `BufferArg::len()`.
- **Plane operations.** The CPU runtime has plane size 1 and registers no plane operations; shuffles and broadcasts fail with `PlaneOpsUnsupported`.
- **Device properties.** `DeviceProperties` is `Clone`, and its fields are public. It cannot be injected into a live `Client`.

## Problem

T0 of M2 writes the acceptance tests against stubs with their final signatures, and M3 programs
against those signatures. The design leaves six things open.

1. **Public items without a definition:**
   - the signatures of the primitives;
   - the key layout of the sort;
   - the crate that holds `GridError`, and its M2 variants;
   - where `ReduceOp` lives: gridcl-device's reduce needs it, but the listing puts it in gridcl-parallel, which depends on gridcl-device;
   - a constructor for `IndexMap`;
   - the type `Neighbourhood`, which `CommDyn::neighbour_all_to_all_v_bytes` names but no listing defines;
   - `DeviceClass` is "derived from `Client::properties()`" without a rule.
2. **Vector scope.** CLAUDE.md says "kernels on the candidate list are generic over N: Size".
   SortDigits and Gather are on the candidate list, but the M2 scope names only fill, copy,
   reduce and the tile-local scan phase.
   - **Lanes over keys.** For SortDigits, the lanes must be independent keys. With interleaved
     key words, a vector of u32 spans parts of several keys.
3. **Data-dependent sizes.** Unique-by-key and run-length produce outputs whose length the
   host must know before it can allocate them or set `len()`. CLAUDE.md allows host copies only
   for "I/O, explicit to_host()/read_node(), and the host-staged halo exchange".
4. **Tests within one process.**
   - The environment variables `GRIDCL_VECTOR_SIZE` and `GRIDCL_QUAD_TILE` are read once per process, so a test cannot sweep vector sizes through them.
   - A clamped size can silently replace the requested one, so a test cannot tell which size ran.
   - No live client accepts injected limits.
5. **Phases of one primitive.**
   - **Splitting the data.** Reduce-then-scan and the radix sort write per-cube results (tile
     sums, digit counts) that the next phase reads, so all phases must split the data
     identically. The design says only that each primitive "takes its launch shape from
     `LaunchPlan`". The scan's vectorized tile phase and a scalar scatter would otherwise get
     different plans.
   - **Cross-lane prefix.** The tile phase of the scan needs an in-vector prefix, a cross-lane
     operation. CLAUDE.md allows "no cross-lane operations except fixed-order reductions".
6. **Errors in the design text:**
   - **`arg()`.** Listing S1 says `arg()` is a "scalar kernel argument over the logical length", which CubeCL 0.11 does not do (see Context).
   - **Empty buffers.** `pad_len(0) = 0` gives a zero-size allocation, and wgpu rejects zero-size bindings.
   - **Duplicate listing name.** Two listings are both called "Listing S2".

## Options

1. **Keep the design as written.**
   - Every item above is decided ad hoc during M2 and recorded only in the plan's Deviations log.
   - M3 would then rely on API that docs/design.md does not record.
   - Problem 3 has no compliant solution. Exactly sized outputs need a host read, so the outputs
     would be sized at input capacity with a wrong `len()`.
   - Not chosen.
2. **Fix every open item in this ADR, and add it to the design** (detailed below). The public
   API grows by the items listed under Consequences. CLAUDE.md gains two clarifications.
3. **Keep the M2 scope literal.** SortDigits and Gather stay scalar until M3s, and sort keys use
   the interleaved (key-major) layout. This keeps M2 smaller, but it contradicts CLAUDE.md's
   candidate-list rule. M3s would then change both the sort kernels and the key layout that M3's
   key emission writes. Not chosen.

## Proposed decision

Option 2. Each item is decided as follows.

### gridcl-device: errors and scalars

- **`GridError`.** `#[non_exhaustive] pub enum GridError` and `pub type Result<T, E = GridError>` live in gridcl-device, the lowest crate that returns them. gridcl-parallel and gridcl re-export both.
  - M2 variants:
    - `TooLarge`;
    - `AllocationTooLarge { bytes: usize, max: u64 }`;
    - `UnsupportedPrecision { scalar: &'static str, device: &'static str }`;
    - `LengthMismatch { expected: usize, found: usize }`;
    - `Server(ServerError)`.
  - It implements `Display`, `std::error::Error` and `From<ServerError>`.
  - Later milestones add their variants (`MessageTooLarge`, `DimensionMismatch`, …) when they first use them.
- **`ReduceOp`.** `ReduceOp { Sum, Min, Max }` moves to gridcl-device with the listing's derives. gridcl-parallel re-exports it, so `gridcl_parallel::ReduceOp` stays valid.
- **`DeviceScalar`.** A sealed trait `DeviceScalar: CubeElement + Numeric`, implemented for f32, f64, i32 and u32: the four types of the design's segmented reduce.

### gridcl-device: the `primitives` module

The module is re-exported at the crate root. In the signatures below, buffers are `DeviceBuffer<T>`.

- **Fill:** `fill<T: DeviceScalar>(x: &mut DeviceBuffer<T>, v: T) -> Result<()>`. It writes the padded length.
- **Copy:** `copy<T: CubeElement>(src, dst: &mut DeviceBuffer<T>) -> Result<()>`.
  - It moves u32 words, so `size_of::<T>()` must be a multiple of 4.
  - Unequal lengths return `LengthMismatch`.
  - `DeviceBuffer::deep_clone` uses it. CubeCL 0.11 has no same-device copy: `to_client` moves data between devices.
- **Reduce:** `reduce<T: DeviceScalar>(x, op: ReduceOp) -> Result<DeviceBuffer<T>>`.
  - The result has length 1.
  - An empty input gives the identity of `op`.
- **Segmented reduce:** `segmented_reduce<T: DeviceScalar>(values, offsets: &DeviceBuffer<u32>, op) -> Result<DeviceBuffer<T>>`.
  - `offsets` are CSR offsets (segments + 1 entries).
  - There is one result per segment, combined in index order.
  - Offsets are clamped to `[0, values.len()]`, and a decreasing pair gives an empty segment.
- **Scans:** `inclusive_scan<T: DeviceScalar>(x) -> Result<DeviceBuffer<T>>` and `exclusive_scan<T: DeviceScalar>(x) -> Result<DeviceBuffer<T>>`.
  - Both compute sums.
  - The exclusive scan returns n + 1 elements, the last being the total, so that its output can serve as CSR offsets.
- **Gather:** `gather<T: DeviceScalar>(src, idx: &DeviceBuffer<u32>) -> Result<DeviceBuffer<T>>`, so that `out[i] = src[idx[i]]`.
  - An index ≥ `src.len()` yields 0 through an arithmetic mask; the read address is clamped, so it is never out of bounds.
  - An empty `src` gives zeros without reading.
- **Scatter:** `scatter<T: DeviceScalar>(src, idx, dst: &mut DeviceBuffer<T>) -> Result<()>`, so that `dst[idx[i]] = src[i]`.
  - Out-of-range writes are dropped.
  - Distinct indices are a documented precondition. Duplicates are memory-safe, but they leave the result unspecified and not deterministic.
- **Histogram:** `histogram(values: &DeviceBuffer<u32>, num_bins: u32) -> Result<DeviceBuffer<u32>>`. Values ≥ `num_bins` are not counted.
- **Sort keys:** `SortKeys` holds `len` keys of `num_words` ∈ 1..=4 u32 words in one buffer.
  - **Layout.** Planar and word-major: word w of key i is at `w · pad_len(len) + i`. Each plane is padded to `VECTOR_PAD`, so the lanes of a vector are independent keys.
  - **Order.** Word 0 is the most significant.
  - **Methods:** `uninit(client, len, num_words)`, `from_host(client, keys, num_words)` (key-major host data), `to_host()` (key-major), `len()`, `num_words()`, `words()`, `words_mut()`.
- **Radix sort:** `radix_sort(keys: &SortKeys, payload: &DeviceBuffer<u32>) -> Result<(SortKeys, DeviceBuffer<u32>)>`.
  - A stable LSD sort with 8-bit digits.
  - Unequal lengths return `LengthMismatch`, and `len ≥ 2^32` returns `TooLarge`.
- **Run-length:** `run_length(keys: &SortKeys) -> Result<Runs>`, with `pub struct Runs { pub keys: SortKeys, pub offsets: DeviceBuffer<u32> }`. It returns the runs of consecutive equal keys; `offsets` has runs + 1 entries.
- **Unique-by-key:** `unique_by_key(sorted: &SortKeys, payload: &DeviceBuffer<u32>) -> Result<Unique>`, with `pub struct Unique { pub keys: SortKeys, pub first: DeviceBuffer<u32>, pub group: DeviceBuffer<u32> }`.
  - `first` is the payload of each group's first key.
  - `group` gives the group of each input position.

### gridcl-device: rules for every primitive

- **Data-dependent sizes.** Run-length and unique-by-key read their count with one blocking 4-byte `read_one`, and return exactly sized buffers. This is an explicit read in the sense of CLAUDE.md's host-copy rule, and their rustdoc says so. No other primitive reads from the device.
- **One partition per primitive.** A primitive with several kernels computes one `LaunchPlan`: for the `KernelKind` of its vectorized phase, or for its first phase if none is vectorized.
  - **Ranges.** Every kernel of that primitive takes its scalar ranges from this plan: a unit owns `items_per_unit · vector_size` consecutive items, and a cube owns `cube_dim` times as many. Scalar phases launch with the same cube count and cube dimension.
  - **Recursive levels** (the partials of reduce and the tile sums of scan) use one new `LaunchPlan::elementwise` per level, until one cube remains.
  - Launch shapes come from nowhere else.
- **Cross-lane operations.** A fixed-order in-vector prefix (lane extracts, unrolled at comptime) is allowed, like a fixed-order reduction. ScanTile uses it.
- **No plane operations.** Primitives combine values within a cube through shared memory and `sync_cube`, so they are correct with plane size 1 and on the CPU runtime.
- **Empty inputs.**
  - An empty input never launches a kernel.
  - `pad_len` rounds 0 up to one `VECTOR_PAD` block, so no handle has zero size, and `len()` stays 0. `padded_len()` is therefore at least `VECTOR_PAD`.

### gridcl-device: `DeviceClass`

`DeviceClass::for_client(&Client) -> DeviceClass` is a public function.
- A device with `hardware.num_cpu_cores` set is a CPU. It is classed by `hardware.load_width`: 512 bits gives CpuSimd512, 256 gives CpuSimd256, and anything else gives CpuSimd128.
- Every other device is Gpu.
- `load_width` alone is not enough: wgpu also reports 128 bits.

### gridcl-device: test hooks

A public module, `gridcl_device::testing`, holds scoped overrides. They are thread-local, because
gridcl runs one host thread per mesh ("Streams and threading").

- **`with_properties`.** `with_properties<R>(props: DeviceProperties, f: impl FnOnce() -> R) -> R` replaces the properties that gridcl-device reads (limits, device class, vector sizes) for buffers and plans made inside `f`. Tests clone `client.properties()` and modify it.
- **`with_vector_size` and `with_quad_tile`.** `with_vector_size<R>(v: VectorSizeOverride, f) -> R` and `with_quad_tile<R>(tile: usize, f) -> R`.
- **`record_plans`.** `record_plans<R>(f) -> (R, Vec<(KernelKind, LaunchPlan)>)` returns the plans made inside `f`, so that a test can assert the vector size it ran at.
- **`VectorSizeOverride`.** `pub enum VectorSizeOverride { Scalar, Policy, Widest, Fixed(usize) }` is also what `GRIDCL_VECTOR_SIZE` parses into.
  - `Widest` is the first entry of `io_optimized_vector_sizes(elem_bytes)`.
  - `Fixed(n)` and `Widest` skip the register budget, but stay clamped by `max_vector`, the data layout.
  - `Policy` applies the full rule of Listing S2.
- **Precedence:** the scoped override, then the environment variable, then `vector-policy.toml`. An invalid `GRIDCL_VECTOR_SIZE` or `GRIDCL_QUAD_TILE` panics and names the variable.

### gridcl-parallel

- **`IndexMap::new`.** `IndexMap::new(comm: &dyn CommDyn, num_owned: usize) -> Result<IndexMap>` builds a map without ghosts.
  - The owned range comes from `exscan_sum` of `num_owned`, and `size_global` from `sum`.
  - An i64 overflow returns `TooLarge`.
  - Added accessors: `owned_range() -> (i64, i64)`, `ghosts() -> &[i64]` and `ghost_owners() -> &[i32]`.
  - M6 adds ghosts, `scatterer` and the device field.
- **`Neighbourhood`.** `pub struct Neighbourhood` is opaque, with `sources() -> &[usize]` and `destinations() -> &[usize]`, plus `Neighbourhood::empty()`.
  - `Serial` accepts only the empty neighbourhood.
  - M6 adds construction over a distributed-graph communicator.

### Dependencies

- gridcl-testing gets a dev-dependency on gridcl-device, for the `LaunchPlan` of M0's add test.
- gridcl-bench gets a dependency on gridcl-device, for the throughput record.
- New workspace dependencies:
  - `proptest` (dev-dependency of gridcl-device);
  - `bytemuck`, for `CommScalar: bytemuck::Pod` in gridcl-parallel;
  - `toml`, which parses `vector-policy.toml` in gridcl-device.
- None of these brings in MPI, BLAS or a runtime feature.

### Corrections to the design text

- **`arg()` and `vector_arg()`.**
  - Their comments say that the handle's size, not `length`, bounds the kernel's slice.
  - Every kernel takes its lengths as explicit arguments (`len`, `num_vectors`), as Listing B5 does.
  - The signatures are unchanged.
- **Listing names.** The communicator listing becomes "Listing S2c — communicators, ownership and halo exchange", and the launch-plan listing stays Listing S2.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing S1:**
    - the comments of `arg` and `vector_arg`;
    - `pad_len` rounds 0 up to `VECTOR_PAD`;
    - the paragraph after the listing says `GridError` and `Result` live in gridcl-device and lists the M2 variants;
    - `deep_clone` is "copy kernel" instead of "device-to-device copy".
  - **"Device primitives":** a new Listing S1b, "primitives (gridcl-device)", with `DeviceScalar`, the signatures, `SortKeys`, `Runs` and `Unique`, and the edge-case rules above. Three more rules join the five:
    - one partition per primitive;
    - no plane operations;
    - the count read of run-length and unique-by-key.
  - **"Data layouts and index widths":** sort keys are planar, word-major, word 0 most significant.
  - **Listing S2 (launch plans):**
    - `DeviceClass::for_client` and its rule;
    - `VectorSizeOverride`;
    - the `testing` module.
  - **"Padding and tiles":** the empty-buffer rule.
  - **"Where vectors apply":** ScanTile's fixed-order in-vector prefix.
  - **Communicator listing:**
    - renamed Listing S2c;
    - `ReduceOp` marked "re-exported from gridcl-device";
    - `IndexMap::new` and the three accessors;
    - `Neighbourhood`.
  - **References** at the `Mesh` field comment, `DeviceGrid::comm` and the generic-algorithm example say S2c.
  - **"Testing and validation":** vector-size invariance is "forced with `GRIDCL_VECTOR_SIZE` or `testing::with_vector_size`".
  - **Crate table:**
    - gridcl-testing gains "dev: gridcl-device";
    - gridcl-bench gains gridcl-device;
    - gridcl-device gains toml, and dev proptest;
    - gridcl-parallel gains bytemuck.
  - **"Handing this design to Claude Code"** (the CLAUDE.md listing) and CLAUDE.md:
    - "no cross-lane operations except fixed-order reductions and fixed-order in-vector prefixes";
    - "Host<->device copies in library code only for I/O, explicit to_host()/read_node(), the count read of run-length and unique-by-key, and the host-staged halo exchange".
- **Public API changes:**
  - **Additions:** `GridError`'s home and M2 variants, `DeviceScalar`, the `primitives` module and its result types, `SortKeys`, `DeviceClass::for_client`, `VectorSizeOverride`, `gridcl_device::testing`, `IndexMap::new` and its accessors, and `Neighbourhood`.
  - **Relocation:** `ReduceOp` moves to gridcl-device, with a re-export at its old path.
  - **No changes** to existing signatures.
- **Acceptance tests affected:** none change. The M2 tests use `testing::with_properties` for "injected" properties and limits, and `testing::with_vector_size` with `record_plans` for the vector-size sweep.
- **Later milestones affected:**
  - **M3:**
    - it emits topology keys into `SortKeys`'s planar layout;
    - it builds entities with `radix_sort` and `unique_by_key`;
    - it uses `testing::with_properties` for "with an injected limit, `QuadField` results are identical with 1 and 7 chunks".
  - **M3s** benchmarks SortDigits and Gather, which exist in vector-generic form, through the same overrides.
  - **M6:**
    - extends `IndexMap` and `Neighbourhood`;
    - adds `MessageTooLarge` to `GridError`;
    - builds `Scatterer::reverse` on `gather` and `segmented_reduce`.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
