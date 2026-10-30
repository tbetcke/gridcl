# ADR-0029: M7 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-30
- **Raised in:** M7 planning (docs/plans/M7.md, "Planned public API and rules" and questions 1–13, 17, 18)
- **Design sections affected:**
  - "Device model and backends": the `GridError` paragraph;
  - "Distributed mesh model" / "Parallel creation pipeline": stages 2 and 3 of the stage table;
  - "Partitioning": Listing S3, the backend table, the bullets "Rank-0 partitioners" and "Reproducibility";
  - "Workspace layout": the crate table (gridcl-partition, ptscotch-sys, parmetis-sys), "Cargo features";
  - "Implementation roadmap": M7 acceptance;
  - "Handing this design to Claude Code": Listing H1, the FFI rule. CLAUDE.md changes the same way.

## Context

M7's roadmap section gives the scope:

> vertex weights on M6's distributed dual graph (ADR-0026), ptscotch-sys and its wrapper, parmetis-sys and its wrapper (feature-gated), METIS on rank 0 (vendored) and Scotch on rank 0.

Its acceptance, on a 10^6-cell tetrahedral box and a 2 · 10^5-cell hexahedron/prism/pyramid mesh at 4 and 8 ranks, is "weighted imbalance ≤ 5%", "PT-Scotch's edge cut is within 10% of sequential Scotch on the same graph, and METIS's within 10% of Scotch", and "a fixed seed reproduces the partition for a fixed rank count".

The public items are Listing S3 (`DistGraph`, `Partitioner`, `Block`) and `PtScotch::default()` in Listing B8. "Partitioning" adds the backend table, "Rank-0 partitioners gather the graph on rank 0, partition it and scatter the destinations", and "PT-Scotch is called with a fixed seed (`SCOTCH_randomSeed`, then `SCOTCH_randomReset` before each call), METIS with `METIS_OPTION_SEED`". Stage 3 of the pipeline takes "vertex weights from the cost model by cell type and degree".

Planning read these against the code after M6 (main at 489d2bb) and the available libraries:

- **gridcl-partition** matches Listing S3. `Block` ignores `vwgt`. Its private graph check (`block.rs:67-117`) tests lengths and monotonicity, not `adjncy` ranges, self-loops, symmetry or weight signs.
- **Stage 2** (`gridcl/src/pipeline/dual_graph.rs:21-76`) builds the facet graph with `vwgt: None`. The cell type and degree of every graph vertex are at hand there (`BlockSpec { cell, degree, family }`, `num_lagrange_nodes`). Stage 2 is crate-private.
- **Features.** gridcl's `ptscotch`, `parmetis`, `metis` and `scotch` forward to gridcl-partition. `metis` and `scotch` are empty. `ptscotch` and `parmetis` enable the sys crates, which are empty workspace members (M0 question 4: "ADR candidate due before M7"), and they do not enable `gridcl-parallel/mpi`, so `CommDyn::raw` is not visible in gridcl-partition.
- **Libraries.**
  - M3 Max: Homebrew Scotch 7.0.14 with PT-Scotch (`SCOTCH_Num` is a 32-bit `int`; built against Open MPI 5.0.10) and METIS 5.1.0 (32-bit `idx_t`). No ParMETIS.
  - Kathleen (2026-10-30, `module --show_hidden avail`): no Scotch, METIS or ParMETIS.
  - Crates: `scotch` 0.2.1 needs a system libscotch and offers no seed API and no strategy builder; `metis` 0.2.x builds METIS from source with `vendored`. mpi-sys 0.2.4 already runs bindgen 0.72.1, so libclang is present wherever MPI builds.
- **gridcl-parallel** has no gather or scatter collective; `CommExt::all_to_all_v` can do both.

The maintainer answered three planning questions on 2026-10-30: the cost model is the Lagrange node count (item 2); ParMETIS is built from source on the M3 Max as well as on Kathleen; the tests read the pipeline's graph through a hidden accessor (item 15).

Two planning decisions need no ADR and are recorded in the plan only: the hexahedron/prism/pyramid test mesh (a new gridcl-testing generator, test infrastructure) and the source build of Scotch and ParMETIS on Kathleen (tooling, question 16).

## Problem

Each problem has the number of the item that answers it.

1. **FFI calls have no allowed home.** Listing H1 and CLAUDE.md say "FFI calls only in gridcl-parallel (MPI) and gridcl-device (GPU-aware; the macOS performance-core count, ADR-0017), behind safe wrappers." The crate table puts the PT-Scotch and ParMETIS wrappers in gridcl-partition. Scotch on rank 0 also needs raw `scotch-sys` calls (`SCOTCH_randomSeed`, `SCOTCH_stratGraphMapBuild`), because the `scotch` crate wraps neither, and `scotch-sys` is not in the crate table.
2. **The cost model has no formula.** Listing B12 shows an example ("tetrahedron 1.0, pyramid 1.3, prism 1.6, hexahedron 2.2, times p³", 3D only), and `DistGraph::vwgt` is integer.
3. **Only `PtScotch` is named.** No listing gives the other backends' type names, constructors or a seed parameter. "A fixed seed reproduces the partition" needs a seed the caller can set.
4. **The sys crates break dependency rule 5.** The crate table gives both an `mpi` dependency. As workspace members they would make `cargo test --workspace` build MPI and run bindgen ("No crate depends on MPI unless its `mpi` feature is on"; `scripts/check-workspace-deps.sh`).
5. **Library discovery and width probing are unspecified.** "`SCOTCH_Num` width probed at build time", "`idx_t` and `real_t` widths probed" name no mechanism. bindgen would also emit its own `MPI_Comm` type, which differs from `mpi::ffi::MPI_Comm`.
6. **No error fits a library failure,** or a distributed partitioner on a communicator without an MPI handle (`Serial` with nparts > 1 cannot happen, but `ThreadComm` with 2 ranks can).
7. **No gather or scatter collective** for the rank-0 partitioners.
8. **Seeds beyond PT-Scotch and METIS.** Scotch's random state is process-global, and possibly METIS's (GKlib) too. `cargo test` runs tests on threads, and every `ThreadComm` rank is a thread, so concurrent calls would share it and results would depend on timing.
9. **Safe wrappers could pass invalid graphs to C.** `Partitioner::partition` is public. An `adjncy` entry out of range, a self-loop or an asymmetric graph is undefined behaviour inside PT-Scotch or ParMETIS. Narrowing each `i64` value does not prevent overflow of totals: 10^7 P5 hexahedra (216 nodes each) sum to 2.2 · 10^9 > `i32::MAX`.
10. **Balance targets** of the libraries are not given; the acceptance limit is 5%.
11. **The acceptance wording is open:** which partitioners "weighted imbalance ≤ 5%" covers (`Block` ignores weights by ADR-0026's definition); whether "within 10%" is one- or two-sided; how edges are counted; how reproducibility is shown.
12. **ParMETIS 4 fails when a rank holds no vertices.**
13. **Threading.** "Streams and threading": "MPI is initialised with `Threading::Single`." A PT-Scotch built with `SCOTCH_PTHREAD_MPI` requires `MPI_THREAD_MULTIPLE`. Homebrew's `libptscotch.7.0.14.dylib` contains no such check string; the Ubuntu and Kathleen builds are not yet known.
14. **Link clash.** Vendored METIS (static, from the `metis` crate) and the system `libmetis` that ParMETIS links both define `METIS_*` symbols when `metis` and `parmetis` are both on.
15. **The tests need "the same graph"** that the pipeline gives the partitioner, and stage 2 is crate-private. A full `build_on` of the 10^6-cell box runs every kernel; on the CPU runtime at 8 ranks this is slow (M6 Deviations log, T14–T16).

## Options

1. **Keep the design as written.** The PT-Scotch and ParMETIS wrappers cannot be written without breaking the FFI rule (problem 1); the sys crates cannot be workspace members (problem 4); reproducibility cannot be tested without a seed parameter (problem 3); and the acceptance bullets can be read several ways (problem 11). Not chosen.
2. **Fix the items below** (proposed).
3. **Put the FFI wrappers in gridcl-parallel** (problem 1), which already holds MPI FFI. This keeps the FFI rule but adds partitioner code and the optional sys-crate dependencies to the communication crate, against the crate table ("gridcl-partition: METIS, Scotch, PT-Scotch and ParMETIS behind features"). Not chosen.
4. **Move the sys crates out of the workspace** (`exclude`, problem 4), as conformance/. `cargo test --workspace`, clippy and fmt would then skip them, and they would need their own lock file and CI. Not chosen.
5. **B12's weights as integers** (problem 2): `round(10 · w · p^TD)`, with w still to be chosen for intervals, triangles and quadrilaterals. Rejected by the maintainer on 2026-10-30.
6. **A recording partitioner inside a full `build_on`** (problem 15). No new item, but every acceptance run builds the 10^6-cell mesh on the device. Rejected by the maintainer on 2026-10-30.

## Proposed decision

Option 2. Each item answers the problem of the same number. The maintainer chose items 2 and 15 on 2026-10-30.

1. **FFI calls in gridcl-partition.**
   - The FFI rule becomes: "FFI declarations only in *-sys crates. FFI calls only in gridcl-parallel (MPI), gridcl-device (GPU-aware; the macOS performance-core count, ADR-0017) and gridcl-partition (Scotch, PT-Scotch, ParMETIS; ADR-0029), behind safe wrappers."
   - gridcl-partition depends on `scotch-sys` (optional, feature `scotch`) besides `scotch`, at the version `scotch` uses.
   - An MPI call that a partitioner needs beyond the library's own (for example a thread-level query, item 13) goes through a safe gridcl-parallel function, not `mpi::ffi` in gridcl-partition.
2. **Cost model.**
   - Stage 2 sets `vwgt = Some(w)`, where `w[i]` is the number of Lagrange nodes of cell i's (cell type, degree), `gridcl_refcell::num_lagrange_nodes` (Appendix A.4).
   - P1 weights: interval 2, triangle 3, quadrilateral 4, tetrahedron 4, pyramid 5, prism 6, hexahedron 8. The 3D ratios 1 : 1.25 : 1.5 : 2 are close to B12's 1 : 1.3 : 1.6 : 2.2. Degree enters as the node count, which grows more slowly than p^TD (a tetrahedron has 4 nodes at P1 and 10 at P2).
   - `Block` keeps ignoring `vwgt` (ADR-0026's definition).
3. **Public partitioner types (gridcl-partition).**
   ```rust
   #[cfg(feature = "ptscotch")] #[derive(Clone, Copy, Debug, Default)] pub struct PtScotch { /* seed */ }
   #[cfg(feature = "parmetis")] #[derive(Clone, Copy, Debug, Default)] pub struct ParMetis { /* seed */ }
   #[cfg(feature = "metis")]    #[derive(Clone, Copy, Debug, Default)] pub struct Metis    { /* seed */ } // on rank 0
   #[cfg(feature = "scotch")]   #[derive(Clone, Copy, Debug, Default)] pub struct Scotch   { /* seed */ } // on rank 0
   impl PtScotch { pub fn seed(self, seed: u32) -> Self; }   // likewise ParMetis, Metis, Scotch
   impl Partitioner for PtScotch { /* … */ }                // name(): "ptscotch", "parmetis", "metis", "scotch"
   ```
   - The default seed is 0. The balance target (item 10) is fixed and not public.
   - gridcl re-exports the four types at its root and in its prelude, each under its feature.
4. **Sys crates and features.**
   - ptscotch-sys and parmetis-sys each have one feature, `mpi`, off by default. It enables their optional dependencies `mpi`, `bindgen` (build) and `build-probe-mpi` (build), the bindgen step and linking. Without it the crate is empty and its build script does nothing.
   - gridcl-partition: `ptscotch = ["dep:ptscotch-sys", "ptscotch-sys/mpi", "gridcl-parallel/mpi"]`, `parmetis` likewise, `metis = ["dep:metis"]`, `scotch = ["dep:scotch", "dep:scotch-sys"]`.
   - gridcl's `ptscotch` and `parmetis` also enable gridcl's `mpi`. A crate that builds tests on `ThreadComm` with them must enable `gridcl-testing/mpi` (ADR-0028 item 2).
   - This answers M0 question 4.
5. **Library discovery and widths.**
   - The build scripts read `SCOTCH_DIR` (ptscotch-sys), `PARMETIS_DIR` and `METIS_DIR` (parmetis-sys), each a prefix with `include/` and `lib/`; without them, the system paths. MPI include paths come from `build-probe-mpi`, as for mpi-sys.
   - bindgen blocklists the MPI types; the bindings use `mpi::ffi::MPI_Comm`.
   - The widths are those of the bound types, exported as `pub const SCOTCH_NUM_BITS: u32` (ptscotch-sys) and `pub const IDX_BITS: u32`, `pub const REAL_BITS: u32` (parmetis-sys). No separate probe program.
   - ptscotch-sys links `ptscotch`, `scotch` and `ptscotcherr`; parmetis-sys links `parmetis` and `metis`. parmetis-sys's documentation repeats the design's licence note.
6. **Errors (gridcl-device).**
   - `GridError` gains `PartitionerFailed { name: &'static str, code: i32 }` (a library status other than success; gridcl's own conditions use negative codes: −1 item 12, −2 and −3 item 9) and `NeedsMpiCommunicator(&'static str)` (`PtScotch` or `ParMetis` with `comm.raw() == None` on more than one rank).
   - Both are agreed like every collective error (ADR-0026 item 15).
7. **Rank-0 gather and scatter.** Crate-private in gridcl-partition, through `CommExt::all_to_all_v` with counts to rank 0 only: degrees, `adjncy` and `vwgt` to rank 0, destinations back. No new `CommExt` method.
8. **Seeds.**
   - PT-Scotch: `SCOTCH_randomSeed(seed)` then `SCOTCH_randomReset()` before each call, as the design says.
   - Scotch on rank 0: the same, under one process-wide mutex in gridcl-partition. METIS (`METIS_OPTION_SEED`) takes the same mutex if GKlib's random state proves global (checked in M7 T3).
   - ParMETIS: `options = [1, 0, seed]`.
9. **Validation before any library call.** Every external partitioner checks, and agrees on the result:
   - `Block`'s length and monotonicity checks;
   - `adjncy` in `[0, N)` (`IndexOutOfRange`);
   - no self-loops (`PartitionerFailed { name, code: -2 }`) and symmetry, where the owner of each edge's far end confirms the reverse edge (collective; `PartitionerFailed { name, code: -3 }`);
   - `vwgt ≥ 0`;
   - global totals within the library's index type: N, the number of edges and the sum of `vwgt` (`TooLarge`).
   - `nparts == 1` returns zeros without a library call.
10. **Balance targets.** 3% for every backend: METIS `ufactor` 30, Scotch and PT-Scotch `SCOTCH_STRATQUALITY` strategies built with 0.03, ParMETIS `ubvec` 1.03. No edge weights.
11. **Reading of the M7 acceptance.**
    - "Weighted imbalance" is max_k W_k · nparts / W − 1, with W_k the summed `vwgt` of part k, for every external partitioner (PtScotch, Metis, Scotch, and ParMetis when built); not `Block`.
    - The edge cut counts each undirected dual-graph edge whose ends lie in different parts once. "Within 10%" is one-sided: cut(PtScotch) ≤ 1.10 · cut(Scotch) and cut(Metis) ≤ 1.10 · cut(Scotch).
    - nparts equals the number of ranks, and every partitioner works on the graph that stage 2 gives for even input chunks.
    - Reproducibility: per partitioner, the same seed twice in one run, with a different seed in between, gives identical parts on every rank, and the hash of the parts agrees between two `mpirun` runs.
12. **ParMETIS with empty ranks.** When any rank holds no vertices, `ParMetis` returns an agreed `PartitionerFailed { name: "parmetis", code: -1 }`, documented. M8's even read chunks give every rank cells when there are at least as many cells as ranks.
13. **Threading.** gridcl supports PT-Scotch builds without `SCOTCH_PTHREAD_MPI`, so MPI stays at `Threading::Single`. M7 T5 probes the Homebrew and Ubuntu builds under `Single` on 2 ranks, and the Kathleen build is configured without it. If a probe fails, M7 stops and a further ADR decides.
14. **METIS link clash.** If vendored METIS and ParMETIS's `libmetis` clash, `parmetis` switches the `metis` crate to its system build (`parmetis = [..., "metis?/use-system"]`), so both use one METIS. M7 T9 checks the build with both features.
15. **Hidden graph accessor (gridcl).** `#[doc(hidden)] pub fn MeshBuilder::dual_graph_on<C: Comm>(self, comm: C) -> Result<DistGraph>` runs stages 1–2 and returns the graph `build_on` would give the partitioner. It is collective and host-only, a test accessor outside the stable API like ADR-0028 item 3.

## Consequences

- **Text to change in docs/design.md:**
  - **The `GridError` paragraph:** add "M7 adds `PartitionerFailed { name: &'static str, code: i32 }` and `NeedsMpiCommunicator(&'static str)` (ADR-0029)."
  - **Stage table:** stage 2 "(M6; M7 adds vertex weights; ADR-0026)" becomes "(M6; vertex weights from M7: the Lagrange node count of the cell's type and degree; ADR-0026, ADR-0029)"; stage 3 "vertex weights from the cost model by cell type and degree" adds "(ADR-0029)".
  - **Listing S3:** add the four types and `seed` of item 3, with their `cfg` lines, and a comment on the hidden `MeshBuilder::dual_graph_on`.
  - **Backend table:** the Scotch row's integration reads "`scotch` crate, plus `scotch-sys` for the seed and strategy"; the PT-Scotch and ParMETIS rows add "feature `mpi` of the sys crate; libraries from `SCOTCH_DIR`, `PARMETIS_DIR`, `METIS_DIR` or the system paths".
  - **"Partitioning" bullets:**
    - "Rank-0 partitioners" adds "through `all_to_all_v`".
    - "Reproducibility" adds "Scotch on rank 0 seeds and resets under a process-wide mutex (and METIS, if its random state is global); ParMETIS takes the seed in `options`."
    - New bullet "**Checks and balance:** every external partitioner validates the graph (ranges, self-loops, symmetry, non-negative weights, totals within the library's index type) before calling the library, returns zeros for one part, and targets 3% imbalance. PT-Scotch builds with `SCOTCH_PTHREAD_MPI` are not supported (MPI runs at `Threading::Single`). ParMETIS needs every rank to hold a vertex (ADR-0029)."
  - **Crate table:** gridcl-partition depends on "parallel; metis, scotch, scotch-sys, ptscotch-sys, parmetis-sys (all optional)"; ptscotch-sys and parmetis-sys depend on "mpi, bindgen, build-probe-mpi (all behind the crate's feature `mpi`)".
  - **"Cargo features":** add "`ptscotch` and `parmetis` imply `mpi`. The sys crates have a feature `mpi`, off by default, that gridcl-partition enables (ADR-0029)."
  - **Roadmap M7 acceptance:** append "(ADR-0029: imbalance for every external partitioner; cut ≤ 1.10 · Scotch's; reproducibility per partitioner and rank count)".
  - **Listing H1:** the FFI rule as in item 1.
- **CLAUDE.md:** the same change to the FFI rule.
- **Public API changes:**
  - gridcl-partition: `PtScotch`, `ParMetis`, `Metis`, `Scotch` and their `seed`, each behind its feature;
  - gridcl-device: `GridError::{PartitionerFailed, NeedsMpiCommunicator}`;
  - ptscotch-sys and parmetis-sys: the bindings and the width constants, behind `mpi`;
  - gridcl: root and prelude re-exports; hidden `MeshBuilder::dual_graph_on`.
  - No existing signature changes. Pipeline graphs now carry `vwgt: Some`; `Block`'s results do not change.
- **Acceptance tests affected:** M7's three bullets, read as in item 11 (new tests `gridcl-mpi-tests::{partition_quality, partition_seed}`). No earlier acceptance test changes.
- **Later milestones affected:**
  - M8: chooses the default partitioner for distributed reads by the existing rule and adds `GmshOptions::partitioner` (ADR-0022); even read chunks satisfy item 12.
  - M10: benchmarks may revisit the cost model and the balance target.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
