# M1a Type layer — plan

- **Status:** approved
- **Branch:** tbetcke/tbetcke-m1a-type-layer (question 1)
- **Base commit:** ad44333511adf48990520db36477b833522a8e67
- **Design sections:** "Implementation roadmap / M1a Type layer (gridcl-dims)"; "Architecture" (intro); "Dimension markers" (Listing B1); "Cell families per topological dimension" (Listing B2); "Mesh type" (Listing B3, for the stub); "Methods that exist only where they make sense" (Listing B4, impl bounds for the stub); "Errors the compiler catches" (Listing B10); "Testing and validation" (targets table, compile-fail tests); "Workspace layout" (crate table); "Development environment and test targets"
- **Depends on:** M0 (done, merged in 27fdeb8, PR #2)
- **Accepted ADRs that apply:** ADR-0003 (Rust 1.99: the `.stderr` baselines depend on the pinned rustc); ADR-0001, ADR-0002 (amended by ADR-0005), ADR-0004, ADR-0006 for the review runs on Myriad and Locust only

<!--
Rules for this file (do not delete):
- Status moves draft -> approved (maintainer) -> implemented (/implement-milestone) -> done (maintainer, after merge).
- Every task is one commit. T0 is always the acceptance tests.
- The plan references design sections; it never restates or changes requirements.
- Tick a task ([x]) in the same commit that completes it.
-->

## Scope summary

M1a fills gridcl-dims with Listings B1 and B2: the dimension markers `D0`–`D3`, `Dim`,
`EmbeddedIn`, `Codim0`, `Codim1`, `CellType`, `Cell1`–`Cell3`, `TopoDim` and the `WrongCellDim`
conversions. A test-only stub `Mesh` with Listing B4's impl bounds lets trybuild check the
compile-fail and compile-pass cases of Listing B10 before the real `Mesh` exists (M3). No kernels,
no device code.

## Tasks

- [x] **T0 Acceptance tests (failing)**
  - Goal: every M1a acceptance bullet as an automated test.
  - Crates/files:
    - root `Cargo.toml`: `trybuild = "1"` in `[workspace.dependencies]`; `Cargo.lock`.
    - `gridcl-dims/Cargo.toml`: `[dev-dependencies] trybuild.workspace = true` (question 2).
    - `gridcl-dims/src/lib.rs`: crate docs, private modules `dim` and `cell`, items re-exported at
      the crate root (`gridcl_dims::D3`, `gridcl_dims::Cell2`, …).
    - `gridcl-dims/tests/compile.rs`: one test `ui` with one `trybuild::TestCases`:
      `compile_fail("tests/ui/fail/*.rs")` and `pass("tests/ui/pass/*.rs")`.
    - `gridcl-dims/tests/ui/stub/mesh.rs` (test-only stub, included by every case with
      `#[path = "../stub/mesh.rs"] mod stub;`; question 3): stand-ins `Real` (f32, f64), `Comm`,
      `Serial`, `QuadRule<F, TD>`, `EntityMap`; `Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial>`
      (PhantomData only) with the aliases `CurveMesh`, `PlanarMesh`, `SurfaceMesh`, `VolumeMesh`
      of Listing B3; the four impl blocks of Listing B4 with its bounds verbatim — `EmbeddedIn`
      (`tdim`, `gdim`), `Codim0` (`inverse_jacobians`), `Codim1` (`normals(&self, rule:
      &QuadRule<F, TD>)`), and `TD::Lower: TopoDim + EmbeddedIn<GD>` (`boundary(&self) ->
      Result<(Mesh<F, TD::Lower, GD, C>, EntityMap), StubError>`); `CellBlock<TD: TopoDim> { pub
      cell: TD::Cell }` with `new(cell: TD::Cell)`. Bodies `unimplemented!()`; never executed.
      `#![allow(dead_code, unused)]` at the top of the stub and of every case, so no warning
      enters a baseline.
    - Shape of every case: the calls sit in `fn check(v: &VolumeMesh<f64>, r: &QuadRule<f64, D3>, …)`
      and `fn main() { let _ = check as fn(&_, &_); }`, so they are type-checked, nothing runs and
      the stub needs no constructors.
    - `gridcl-dims/tests/ui/fail/`: `volume_normals.rs` (`normals` on `VolumeMesh<f64>`),
      `cell_block_wrong_dim.rs` (`CellBlock::<D2>::new(Cell3::Tetrahedron)`),
      `curve_boundary.rs` (`boundary` on `Mesh<f64, D1, D2>` as in Listing B10, and on
      `CurveMesh<f64>`). No `.stderr` files yet (T3).
    - `gridcl-dims/tests/ui/pass/`: `normals_surface_planar_curve.rs` (`SurfaceMesh<f64>`,
      `Mesh<f64, D1, D2>`), `boundary_volume_surface.rs`, `boundary_boundary_volume.rs`.
    - `gridcl-dims/tests/cell_type.rs`: `cell_n_round_trips` (every `Cell1`/`Cell2`/`Cell3`
      variant → `CellType` → back), `cell_type_round_trips` (every `CellType` except `Point` →
      the `TD::Cell` of its dimension → back), `wrong_dimension_returns_wrong_cell_dim` (every
      `CellType` against `D1`, `D2`, `D3` where the dimension differs, `Point` against all three:
      `Err(WrongCellDim { found, expected_tdim: TD::N })`). The variant lists are checked for
      completeness by an exhaustive `match` in the test, so a new variant breaks the build.
  - Stubs (in `gridcl-dims/src/`):
    - `dim.rs`: Listing B1 except the six `EmbeddedIn` impls and the `Codim0`/`Codim1` impls
      (sealed module, `Dim`, `D0`–`D3` with their `Dim` impls, the three trait declarations).
    - `cell.rs`: Listing B2 types and `TopoDim` impls; `From<CellN> for CellType` and
      `TryFrom<CellType> for CellN` written out with `todo!()` bodies.
  - Verify: `cargo test -p gridcl-dims` builds; the `cell_type` tests fail with
    `not yet implemented`; `compile::ui` fails: the fail cases for missing `.stderr` files, the
    pass cases with E0599 (no `EmbeddedIn`/`Codim1` impls yet).
  - After: –
  - Risks / notes:
    - holding back the pair impls departs from the template's "planned signatures" stubs, so that
      the pass cases fail at run time (logged);
    - trybuild builds each case from its source path, so the `#[path]` include resolves; if not,
      the stub is inlined into each case (logged);
    - trybuild is not in the local cargo cache: T0 needs crates.io;
    - with no `.stderr`, trybuild writes a `wip/` copy: check where it lands and make sure it is
      never committed (`.gitignore` if it is outside `target/`).

- [x] **T1 Dimension pairs (Listing B1)**
  - Goal: the six `EmbeddedIn` pairs and the `Codim0`/`Codim1` impls, with rustdoc.
  - Crates/files: `gridcl-dims/src/dim.rs`.
  - Public API added or changed: `impl EmbeddedIn<GD> for TD` for the six pairs, `impl Codim0`
    for (D1, D1), (D2, D2), (D3, D3), `impl Codim1` for (D1, D2), (D2, D3), exactly as Listing B1.
  - Tests:
    - the three `pass` cases of `compile::ui` (the test stays red until T3 adds the `.stderr`
      files);
    - unit tests: `N` of each marker; the `Lower` chain (`D0::Lower = D0`); `Array<T>` lengths;
      `Jacobian`/`PseudoInverse` shapes for the six pairs (outer and inner lengths);
    - rustdoc examples: a Jacobian type for a surface pair; `compile_fail` doctests for a
      downstream `impl Dim` (sealed), `D3: Codim1<D3>` and `D0: TopoDim`.
  - Verify: `cargo test -p gridcl-dims --lib && cargo test -p gridcl-dims --doc`;
    `cargo test -p gridcl-dims --test compile` reports the three pass cases `ok`.
  - After: T0
  - Parallel with: T2 (disjoint files: `dim.rs` vs `cell.rs`, `tests/cell_type.rs`)

- [x] **T2 Cell conversions (Listing B2)**
  - Goal: `From<CellN> for CellType` and `TryFrom<CellType> for CellN` from one internal
    `macro_rules!`, replacing the T0 stubs, with rustdoc.
  - Crates/files: `gridcl-dims/src/cell.rs`.
  - Public API added or changed: none beyond T0's stubs (bodies only); no `Display` or
    `std::error::Error` for `WrongCellDim` (question 5).
  - Tests: `cell_type::cell_n_round_trips`, `cell_type::cell_type_round_trips`,
    `cell_type::wrong_dimension_returns_wrong_cell_dim`; rustdoc example of a round trip and of
    `Cell2::try_from(CellType::Tetrahedron)`. No property tests (question 4).
  - Verify: `cargo test -p gridcl-dims --test cell_type && cargo test -p gridcl-dims --doc`.
  - After: T0
  - Parallel with: T1

- [x] **T3 Compile-fail baselines**
  - Goal: the `.stderr` files of the three fail cases, generated and reviewed.
  - Crates/files: `gridcl-dims/tests/ui/fail/*.stderr`.
  - Public API added or changed: none.
  - Tests: `compile::ui` passes (all six cases).
  - Review, recorded in the commit message and in this plan's Deviations log (code and bound per
    case). Expected:
    - `volume_normals`: E0599 on `Mesh<f64, D3, D3>`, unsatisfied `D3: Codim1<D3>`;
    - `cell_block_wrong_dim`: E0308, expected `Cell2`, found `Cell3`;
    - `curve_boundary`: E0599 on `Mesh<f64, D1, D2>` and `Mesh<f64, D1, D3>` (rustc drops the
      `Serial` default and the alias), unsatisfied `D0: TopoDim` and `D0: EmbeddedIn<D2>` /
      `<D3>`;
    - paths normalised by trybuild (`$DIR`, `$WORKSPACE`), no warnings, no machine-specific text.
    Different wording is logged (Listing B10: "the comments show the error codes and the
    unsatisfied bound, not rustc's exact text"); a different error code (for example E0277)
    changes an acceptance test: stop and draft an ADR.
  - Verify: `TRYBUILD=overwrite cargo test -p gridcl-dims --test compile`, review, then
    `cargo test -p gridcl-dims --test compile` without it; then the full M3 Max set:
    `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`,
    `RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps`, `scripts/check-workspace-deps.sh`,
    and, unsandboxed (the sandbox hides the GPU),
    `GRIDCL_TEST_DEVICE=metal cargo test --workspace --features gridcl-testing/metal`.
  - After: T1 (the baselines); T2 (the final full-suite run)
  - Parallel with: –
  - Risks / notes:
    - the baselines hold line numbers of `tests/ui/stub/mesh.rs` and possibly of `src/dim.rs`
      and `src/cell.rs` (rustc's "bound introduced here" and "defined here" notes): editing any
      of them means regenerating and re-reviewing all three baselines, logged;
    - they are tied to rustc 1.99 (ADR-0003); `channel = "1.99"` takes the newest 1.99.x, so if
      CI and the M3 Max disagree on a baseline, compare `rustc -V` first;
    - remote runs (review) run trybuild's nested cargo build with `CARGO_NET_OFFLINE=true`;
      confirm it at the first Locust or V100 run.

## Acceptance coverage

Every acceptance bullet of the milestone in docs/design.md appears exactly once.

Targets for every row: M3 Max: `cargo test --workspace` (CPU runtime) and the Metal workspace run;
hosted CI: the cpu job (`cargo test --workspace`) and the vulkan job; at the milestone review
Myriad V100 and Locust run `cargo test --release --workspace --features gridcl-testing/cuda`. The
tests are host-only (no device code), so the device runs only show that they build and pass on
every target. MPI: not applicable. Myriad A100: not required ("Development environment and test
targets" lists no M1a review). Vector-size invariance: no kernels in M1a.

| Acceptance criterion (docs/design.md) | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| compile-fail: `normals()` on a volume mesh gives E0599 | `gridcl-dims::compile::ui`, `tests/ui/fail/volume_normals.rs` | T0, T1, T3 | as above |
| compile-fail: a `Cell3` in a `CellBlock<D2>` gives E0308 | `compile::ui`, `tests/ui/fail/cell_block_wrong_dim.rs` | T0, T3 | as above |
| compile-fail: `boundary()` on a curve mesh gives E0599 | `compile::ui`, `tests/ui/fail/curve_boundary.rs` | T0, T1, T3 | as above |
| the `.stderr` files are generated with `TRYBUILD=overwrite` and reviewed | the three `tests/ui/fail/*.stderr`; review in the T3 commit message and Deviations log | T3 | M3 Max (generation); checked on every target by `compile::ui` |
| compile-pass: `normals()` on surface and planar-curve meshes | `compile::ui`, `tests/ui/pass/normals_surface_planar_curve.rs` | T0, T1 | as above |
| compile-pass: `boundary()` on volume and surface meshes | `compile::ui`, `tests/ui/pass/boundary_volume_surface.rs` | T0, T1 | as above |
| compile-pass: `boundary().boundary()` on a volume mesh | `compile::ui`, `tests/ui/pass/boundary_boundary_volume.rs` | T0, T1 | as above |
| `CellType` ↔ `CellN` round trips for every cell; a wrong dimension returns `WrongCellDim` | `gridcl-dims::cell_type::cell_n_round_trips`, `cell_type_round_trips`, `wrong_dimension_returns_wrong_cell_dim` | T0, T2 | as above |

## Open questions and ADR candidates

Resolve every row before Status becomes approved.

| # | Question or conflict | Design section | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| 1 | Branch `tbetcke/tbetcke-m1a-type-layer`, not `m1a-<slug>`. | CLAUDE.md Workflow | Keep it and log it, as in M0 (question 1 there). | answered |
| 2 | The crate table lists no dev-dependencies for gridcl-dims; trybuild ("Testing and validation", compile-fail tests) needs one. The real `Mesh` will live in gridcl. | Workspace layout; Testing and validation | Compile tests in `gridcl-dims/tests/` with a `trybuild` dev-dependency (cheap: no cubecl build in trybuild's target dir); logged, no ADR. M3 moves them to gridcl when the real `Mesh` replaces the stub (follow-up). | answered |
| 3 | The stub needs `Real` (gridcl-device), `Comm`/`Serial` (gridcl-parallel), `QuadRule`, `EntityMap`, `Result`, none of which exist before M2/M3. Listing B10 calls `CellBlock::<D2>::new(…)`, but Listing B3 gives `CellBlock` no constructor. | Mesh type; Listing B4; Listing B10 | Stand-ins in `tests/ui/stub/mesh.rs` with B3/B4's names, generic parameters and impl bounds verbatim; return types simplified. Stub `CellBlock::new(cell)`. Follow-up for M3: whether the real `CellBlock` gets a `new` (else the E0308 case uses a struct literal). | answered |
| 4 | CLAUDE.md asks for property tests each milestone; the domain here is eight cell types and six dimension pairs. | CLAUDE.md; Testing and validation | Exhaustive tests instead; no proptest dependency. | answered |
| 5 | Listing B2 derives only `Clone, Copy, Debug, PartialEq, Eq` for `WrongCellDim`; `Display` and `std::error::Error` would make it usable with `?`. | Cell families | Exactly the listing in M1a; M3's `GridError` can wrap it. An ADR if a later milestone needs the impls. | answered |
| 6 | Listing B1 imports from `core::`, but nothing says whether gridcl-dims is `#![no_std]`. | Dimension markers | Not in M1a; `#![no_std]` only if a later milestone needs it. | answered |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |
| 2026-10-19 | plan | Branch `tbetcke/tbetcke-m1a-type-layer`, not `m1a-<slug>`. | Maintainer's worktree branch, as in M0 (question 1). | – |
| 2026-10-19 | plan | gridcl-dims gets a `trybuild` dev-dependency, not in the crate table. Follow-up for M3: moving the cases to gridcl makes trybuild rebuild gridcl's dependencies, cubecl included, in `target/tests/trybuild` on every target. | Compile-fail tests need trybuild ("Testing and validation"); no public API or crate boundary changes (question 2). | – |
| 2026-10-19 | plan | T0's stubs leave out the `EmbeddedIn`/`Codim0`/`Codim1` impls instead of having every planned item. | Type-level items have no bodies to `todo!()`; without the impls the pass cases fail at run time, as the template requires of acceptance tests. | – |
| 2026-10-19 | T0 | `tests/cell_type.rs` has a fourth test, `lists_are_complete`, which checks the variant lists against the exhaustive `match` helpers. | The exhaustive `match` alone does not catch a variant missing from a list. | – |
| 2026-10-19 | T0 | Root `.gitignore` ignores `/gridcl*/wip/`. | trybuild writes pending `.stderr` files to `gridcl-dims/wip/` (outside `target/`); it adds its own `*` `.gitignore` there, the root entry is a second guard. | – |
| 2026-10-19 | T0 | The `#[path]` include works; trybuild prints locations in the stub and in `src/` without line numbers (`tests/ui/fail/../stub/mesh.rs`, `src/dim.rs`). | Observation: T3's baselines depend on the text of the bound lines, not on their line numbers. | – |
| 2026-10-19 | T1 | The `compile_fail` doctests carry `E0277`, but stable rustdoc does not check error codes; each snippet was compiled against the crate with `rustc` and fails with E0277 for the intended bound (`D4: Sealed`, `D3: Codim1<D3>`, `D0: TopoDim`). The `D0: TopoDim` doctest sits on `D0` in `dim.rs`, so T1 stays out of `cell.rs`. | Make sure the doctests fail for the right reason. | – |
| 2026-10-19 | T3 | Baseline review (rustc 1.99.1): `volume_normals` E0599 on `Mesh<f64, D3, D3>`, unsatisfied `D3: Codim1<D3>`; `cell_block_wrong_dim` E0308, expected `Cell2`, found `Cell3`; `curve_boundary` E0599 on `Mesh<f64, D1, D2>` and `Mesh<f64, D1, D3>`, unsatisfied `D0: TopoDim` and `D0: EmbeddedIn<D2>` / `<D3>`. Codes and bounds as expected. Wording differs from Listing B10: "exists for reference `&Mesh<…>`" instead of "for struct", and rustc qualifies some paths (`stub::Mesh`, `gridcl_dims::D1`). Paths are relative, the stub and `src/dim.rs` appear without line numbers, no warnings, no machine-specific text. | Listing B10 shows codes and bounds, not rustc's exact text. | – |

## Review

<Filled by /review-milestone.>
