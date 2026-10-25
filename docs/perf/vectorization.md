# Vectorization on gridcl's devices (M3s)

Draft. The protocol, conditions, observations, generated tables and compile times are added in
M3s T17 (docs/plans/M3s.md); `policy_report` writes the generated part between its marker
comments.

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
