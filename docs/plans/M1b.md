# M1b Reference cells — plan

- **Status:** implemented
- **Branch:** tbetcke/tbetcke-m1b-reference-cells (question 1)
- **Base commit:** 1b94e3ce1d3d2685d2339b9bdd42d541db894225
- **Design sections:** "Implementation roadmap / M1b Reference cells (gridcl-refcell, gridcl-conformance)"; "Reference cells and conventions"; Appendix A.1–A.6, A.8 (VTK pyramids), A.9, A.10, A.11; "Geometry / Tabulation and quadrature"; "High-order node orderings"; "Mesh type" (Listing B3, `NodeFamily`); "Kernels receive dimensions from types" (Listing B5, `gridcl_refcell::num_lagrange_nodes`); "Precision policy"; "Testing and validation"; "Development environment and test targets"; "Workspace layout" (crate table, dependency rules 5 and 6, CI matrix)
- **Depends on:** M1a (done, merged in e66b5f8, PR #3)
- **Accepted ADRs that apply:** ADR-0003 (Rust 1.99); ADR-0001, ADR-0002 (amended by ADR-0005), ADR-0004, ADR-0006 for the review runs on Myriad and Locust only

<!--
Rules for this file (do not delete):
- Status moves draft -> approved (maintainer) -> implemented (/implement-milestone) -> done (maintainer, after merge).
- Every task is one commit. T0 is always the acceptance tests.
- The plan references design sections; it never restates or changes requirements.
- Tick a task ([x]) in the same commit that completes it.
-->

## Scope summary

M1b fills gridcl-refcell with gridcl's own reference-cell definitions under the DefElement
conventions: the tables A.1–A.4 for the eight cells, Lagrange layouts for any p (A.5, including
the prism and pyramid rules and DefElement's pyramid space), equispaced and GLL point families,
quadrature, host f64 tabulation with the crate's own dense LU, the orientation bits of A.6 and
the Gmsh/VTK permutation generators of A.9. gridcl-conformance compares them with ndelement at
f8cbffe and exports the JSON fixtures that the workspace tests read (A.10). Host code only: no
kernels, no device code; gridcl-refcell keeps its single dependency gridcl-dims.

## Planned public API of gridcl-refcell (question 2)

This section is not in TEMPLATE.md: no Architecture listing defines the crate's API (B3 and B5
use only `NodeFamily` and `num_lagrange_nodes(cell, degree)`), and later milestones build on it.
All items at the crate root, f64 only ("Precision policy"). Points are flat `[f64]`, node-major,
`tdim(cell)` coordinates per point.

```rust
// cell.rs — A.1–A.3
pub fn tdim(cell: CellType) -> usize;
pub fn vertices(cell: CellType) -> &'static [&'static [f64]];                     // A.1
pub fn sub_entities(cell: CellType, dim: usize) -> &'static [&'static [usize]];   // A.2, A.3; empty if dim > tdim
pub fn sub_entity_types(cell: CellType, dim: usize) -> &'static [CellType];
pub fn num_sub_entities(cell: CellType, dim: usize) -> usize;

// lagrange.rs — A.4, A.5
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NodeFamily { Equispaced, Gll }
pub fn num_lagrange_nodes(cell: CellType, degree: u32) -> usize;                  // Listing B5 (question 12)
pub fn num_interior_lagrange_nodes(cell: CellType, degree: u32) -> usize;
pub fn lagrange_points(cell: CellType, degree: u32, family: NodeFamily) -> Result<Vec<f64>, RefCellError>;
pub fn lagrange_entity_nodes(cell: CellType, degree: u32, dim: usize, index: usize) -> core::ops::Range<usize>;

// quadrature.rs (named apart from gridcl-geometry's QuadRule<F, TD>, which builds on it)
pub struct QuadratureRule { pub cell: CellType, pub degree: u32, pub points: Vec<f64>, pub weights: Vec<f64> }
pub fn gauss_rule(cell: CellType, degree: u32) -> QuadratureRule;   // exactness: question 11

// tabulation.rs
pub struct LagrangeElement { /* cell, degree, family, points, coefficients (private) */ }
impl LagrangeElement {
    pub fn new(cell: CellType, degree: u32, family: NodeFamily) -> Result<Self, RefCellError>;
    pub fn cell(&self) -> CellType; pub fn degree(&self) -> u32; pub fn family(&self) -> NodeFamily;
    pub fn dim(&self) -> usize;                                       // = num_lagrange_nodes
    pub fn points(&self) -> &[f64];
    /// nderivs 0 (values) or 1 (values and first derivatives).
    pub fn tabulate(&self, points: &[f64], nderivs: u32) -> Result<Tabulation, RefCellError>;
}
/// data[(deriv * num_points + point) * num_basis + basis]; deriv 0 = value, 1 + d = ∂/∂x_d.
/// gridcl-geometry reorders to B5's dphi [q][k][d] when it narrows and uploads.
pub struct Tabulation { pub num_derivs: usize, pub num_points: usize, pub num_basis: usize, pub data: Vec<f64> }
impl Tabulation { pub fn get(&self, deriv: usize, point: usize, basis: usize) -> f64; }

// orientation.rs — A.6
pub fn orientation_bits(cell: CellType, vertex_ids: &[i64]) -> u32;  // origin ids; 0 for tdim < 2

// node_maps.rs — A.9, "High-order node orderings"; entry k = internal index of the file's k-th node
pub fn gmsh_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError>;
pub fn vtk_to_internal(cell: CellType, degree: u32) -> Result<Vec<usize>, RefCellError>;   // pyramid p ≥ 2: Err (A.8)

// error.rs (question 3)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefCellError {
    UnsupportedNodeFamily { cell: CellType, family: NodeFamily },   // GLL on a non-tensor cell
    UnsupportedDegree { cell: CellType, degree: u32 },              // p = 0 for layouts, elements, maps
    UnsupportedDerivativeOrder(u32),
    PointsShape { len: usize, tdim: usize },
    NotInFormat { format: &'static str, cell: CellType, degree: u32 },
    Singular,                                                       // LU breakdown (unisolvence failure)
}   // + Display, std::error::Error
```

Internal (`pub(crate)`), each declared in T0 as an empty module so that later tasks never share
`lib.rs`: `generator` (lexicographic sub-entity generator), `jacobi` (Gauss–Jacobi and GLL
points by Newton iteration on [0, 1]), `polynomials` (orthonormal sets with first derivatives;
`polynomials/pyramid.rs` for A.5's pyramid space), `lu` (dense LU with partial pivoting),
`node_maps/gmsh.rs`, `node_maps/vtk.rs`.

## Tasks

- [x] **T0 Acceptance tests (failing)**
  - Goal: every M1b acceptance bullet as an automated test.
  - Crates/files:
    - root `Cargo.toml`: `proptest = "1"` in `[workspace.dependencies]`; `Cargo.lock`;
      `gridcl-refcell/Cargo.toml`: `proptest.workspace = true` under `[dev-dependencies]`
      (question 6).
    - `gridcl-refcell/tests/node_counts.rs`: `node_counts_match_a4` — the eight cells (Point:
      N = 1, interior 0, question 12), p = 1..=10: both counts against A.4's formulas, and
      `num_lagrange_nodes` against `lagrange_points(..).len() / tdim`.
    - `gridcl-refcell/tests/tabulation.rs` (f64): `partition_of_unity` and
      `reproduces_own_space` — every cell with D_TOP ≥ 1, p = 1..=8, equispaced, plus GLL on
      the tensor cells; at the points of `gauss_rule(cell, 2p)` and at the nodes: the basis sums
      to 1, and interpolating each monomial of the element's own space (A.5; pyramid space of
      Definition 9) reproduces its values; tolerances of question 10.
    - `gridcl-refcell/tests/quadrature.rs`: `integrates_monomials` — every cell with D_TOP ≥ 1,
      degree 0..=20, every monomial within the rule's degree (rationomials on pyramids,
      question 11) against closed-form integrals written in the test, 1e-13 relative.
    - `gridcl-refcell/tests/node_maps.rs`: `gmsh_reproduces_a9`, `vtk_reproduces_a9` (every A.9
      row as a literal; the VTK pyramid 14 row as `Err(NotInFormat)`), `gmsh_p3_matches_dolfinx`
      (the arrays of `gmsh_tetrahedron` case 20 converted with A.11, and `gmsh_quadrilateral`
      16, `gmsh_hexahedron` 64, `gmsh_prism` 40 and `gmsh_pyramid` 30 unchanged, copied from
      DOLFINx `cpp/dolfinx/io/cells.cpp` at the pinned commit, question 7).
    - `conformance/tests/tables.rs`: rewritten. `sub_entities_match_ndelement` replaces
      `triangle_edges_match_ndelement`: for all eight cells, `tdim`, `vertices`, and
      `sub_entities`/`sub_entity_types` of every dimension against ndelement's
      `reference_cell::{dim, vertices, entity_indices/edges/faces, entity_types}`.
    - `conformance/tests/orientation.rs`: `bits_match_compute_orientation` — the six cells with
      D_TOP ≥ 2, 10 000 random numberings each (fixed-seed splitmix64 in the test; distinct
      non-negative ids below 2^62, as `i64` to gridcl and `usize` to ndelement):
      `orientation_bits == compute_orientation as u32`.
    - `conformance/tests/lagrange.rs`: `points_match_ndelement` — intervals, triangles,
      quadrilaterals, tetrahedra, hexahedra, p = 1..=4, `LagrangeVariant::Equispaced` and
      (tensor cells) `Gll`, `Continuity::Standard`; ndelement's `interpolation_points()`
      concatenated by dimension, entity and point (its DOF order) against `lagrange_points`,
      node by node (question 10).
  - Stubs: `gridcl-refcell/src/lib.rs` and `cell.rs`, `lagrange.rs`, `quadrature.rs`,
    `tabulation.rs`, `orientation.rs`, `node_maps.rs`, `error.rs` with the planned API above and
    `todo!()` bodies (types complete; `RefCellError`'s `Display` and `Error` written out); the
    internal modules as empty files.
  - Verify: `cargo test -p gridcl-refcell` and `cargo test --manifest-path conformance/Cargo.toml`
    compile, and every new test fails with `not yet implemented`; `scripts/check-workspace-deps.sh`
    passes (proptest added).
  - After: –
  - Risks / notes: large for one commit (about 25 stubs, seven test files, the closed-form
    integrals and the A.9/DOLFINx literals), logged; the literals need github.com, the
    conformance build codeberg.org (cached locally).

- [x] **T1 Reference-cell tables (A.1–A.3)**
  - Goal: `const` tables for the eight cells and the lexicographic generator that reproduces them.
  - Crates/files: `gridcl-refcell/src/cell.rs`, `src/generator.rs`.
  - Public API added or changed: bodies of `tdim`, `vertices`, `sub_entities`,
    `sub_entity_types`, `num_sub_entities` (question 4).
  - Tests: `conformance::tables::sub_entities_match_ndelement`; unit tests: the generator
    (facets from the cell's facet planes, edges as the vertex pairs shared by D_TOP − 1 facets,
    each dimension sorted lexicographically) equals the `const` tables for all eight cells; the
    side notes of A.2 and A.3 (opposite-vertex rules, hexahedron face planes); rustdoc example.
  - Verify: `cargo test -p gridcl-refcell --lib && cargo test --manifest-path conformance/Cargo.toml --test tables`
  - After: T0
  - Parallel with: T3 (disjoint files)

- [x] **T2 Orientation bits (A.6)**
  - Goal: `orientation_bits` with ndelement's layout, from origin ids.
  - Crates/files: `gridcl-refcell/src/orientation.rs`.
  - Public API added or changed: body of `orientation_bits`.
  - Tests: `conformance::orientation::bits_match_compute_orientation`; unit tests: the A.6 worked
    example (quadrilateral face → `0b001`, edge (30, 4) → bit 1), the bits-used table of A.6
    (no bit set above each cell's count, bits 30 and 31 always zero), negative ids; proptest
    `bits_depend_only_on_order` (a strictly increasing relabelling of the ids keeps the word);
    rustdoc example.
  - Verify: `cargo test -p gridcl-refcell orientation && cargo test --manifest-path conformance/Cargo.toml --test orientation`
  - After: T1
  - Parallel with: T3

- [x] **T3 Point rules and quadrature**
  - Goal: Gauss–Jacobi and GLL points by Newton iteration, and `gauss_rule` on every cell as in
    "Tabulation and quadrature".
  - Crates/files: `gridcl-refcell/src/jacobi.rs`, `src/quadrature.rs`.
  - Public API added or changed: body of `gauss_rule`.
  - Tests: `quadrature::integrates_monomials`; unit tests: weights sum to the cell volume, points
    inside the cell, GLL points symmetric with endpoints 0 and 1 and known values for n ≤ 5,
    Gauss–Jacobi nodes against closed forms for small n, `gauss_rule(Point, n)` is one point of
    weight 1; rustdoc example.
  - Verify: `cargo test -p gridcl-refcell --test quadrature && cargo test -p gridcl-refcell --lib`
  - After: T0
  - Parallel with: T1, T2 (disjoint files)
  - Risks / notes: pyramids use Gauss–Jacobi(2, 0) in z under the collapse of question 11.

- [x] **T4 Lagrange layouts (A.4, A.5)**
  - Goal: node counts and node coordinates in internal order for any p ≥ 1, equispaced on every
    cell and GLL on intervals, quadrilaterals and hexahedra.
  - Crates/files: `gridcl-refcell/src/lagrange.rs`.
  - Public API added or changed: bodies of `num_lagrange_nodes`, `num_interior_lagrange_nodes`,
    `lagrange_points`, `lagrange_entity_nodes`.
  - Tests: `node_counts::node_counts_match_a4`, `conformance::lagrange::points_match_ndelement`;
    unit tests: block layout (`lagrange_entity_nodes` ranges tile 0..N in A.5's block order),
    the in-entity rules of A.5 per entity type, points lie on their entity, GLL on triangles,
    tetrahedra, prisms and pyramids returns `UnsupportedNodeFamily`, p = 0 returns
    `UnsupportedDegree`, the overflow panic of question 12; rustdoc example.
  - Verify: `cargo test -p gridcl-refcell --test node_counts && cargo test -p gridcl-refcell --lib && cargo test --manifest-path conformance/Cargo.toml --test lagrange`
  - After: T1, T3 (GLL points)
  - Parallel with: –

- [x] **T5 Orthonormal polynomial sets**
  - Goal: orthonormal sets with first derivatives on intervals (Legendre), quadrilaterals and
    hexahedra (tensor Legendre), triangles and tetrahedra (Dubiner) and prisms (Dubiner ×
    Legendre), as in "Tabulation and quadrature".
  - Crates/files: `gridcl-refcell/src/polynomials.rs` (dispatch; pyramid arm `todo!()` until T6).
  - Public API added or changed: none.
  - Tests: unit tests: orthonormality to 1e-12 with `gauss_rule(cell, 2p)` for p ≤ 8;
    derivatives against central differences; the span equals the cell's space (rank of the
    monomial Vandermonde).
  - Verify: `cargo test -p gridcl-refcell --lib`
  - After: T3
  - Parallel with: T8, T9

- [x] **T6 Pyramid rationomial basis**
  - Goal: an orthonormal basis of A.5's pyramid space with first derivatives.
  - Crates/files: `gridcl-refcell/src/polynomials/pyramid.rs`; the pyramid arm in
    `polynomials.rs`.
  - Public API added or changed: none.
  - Tests: unit tests: orthonormality with `gauss_rule(Pyramid, 2p)` for p ≤ 8; span equals
    A.5's rationomial set (rank test); dimension equals A.4's N(p); derivatives against central
    differences away from the apex.
  - Verify: `cargo test -p gridcl-refcell --lib`
  - After: T5
  - Parallel with: T8, T9
  - Risks / notes: proposed construction (reviewed): with ξ = x/(1−z), η = y/(1−z),
    P_i(ξ) P_j(η) (1−z)^max(i,j) P_k^(2max(i,j)+2, 0)(2z−1), max(i,j) + k ≤ p, normalised. It
    spans exactly A.5's set and is orthogonal on the pyramid; the apex is handled by the limit.

- [x] **T7 Dense LU and Lagrange tabulation**
  - Goal: gridcl-refcell's own dense LU with partial pivoting, and `LagrangeElement`: the
    generalised Vandermonde of the orthonormal set at the nodes, solved with it; values and
    first derivatives at arbitrary points.
  - Crates/files: `gridcl-refcell/src/lu.rs`, `src/tabulation.rs`.
  - Public API added or changed: bodies of `LagrangeElement` and `Tabulation`.
  - Tests: `tabulation::partition_of_unity`, `tabulation::reproduces_own_space`; unit tests: LU
    on known systems, pivoting on a zero leading entry, `Singular` on a rank-deficient matrix;
    Kronecker property at the nodes; reproduction of first derivatives (1e-10 relative to the
    max norm of the derivative, p ≤ 8; not an acceptance tolerance); `PointsShape`,
    `UnsupportedDerivativeOrder`, GLL on a triangle → `UnsupportedNodeFamily`; rustdoc example.
  - Verify: `cargo test -p gridcl-refcell --test tabulation && cargo test -p gridcl-refcell --lib`
  - After: T4, T6
  - Parallel with: –
  - Risks / notes: equispaced conditioning grows with p ("well conditioned up to about
    p = 10"). If p = 8 misses 1e-12 on any cell, record the achieved error, stop and draft an
    ADR (the tolerance is an acceptance criterion).

- [x] **T8 Gmsh permutation generator (A.9)**
  - Goal: `gmsh_to_internal` for any p ≥ 1, generated from Gmsh's recursive rule and matched to
    `lagrange_points` by coordinates ("High-order node orderings").
  - Crates/files: `gridcl-refcell/src/node_maps.rs` (`gmsh_to_internal`, shared coordinate
    matching), `src/node_maps/gmsh.rs`.
  - Public API added or changed: body of `gmsh_to_internal`.
  - Tests: `node_maps::gmsh_reproduces_a9`, `node_maps::gmsh_p3_matches_dolfinx`; proptest
    `gmsh_maps_are_permutations` (every cell, p ≤ 6: a bijection onto 0..N that starts with
    A.9's vertex map); rustdoc example of the direction `internal[map[k]] = file[k]`.
  - Verify: `cargo test -p gridcl-refcell --test node_maps gmsh && cargo test -p gridcl-refcell --lib`
  - After: T4
  - Parallel with: T5, T6
  - Risks / notes: Gmsh's edge and face lists and directions come from Gmsh's sources, cited in
    the module docs; equispaced nodes are matched on the integer lattice p·x, so no tolerance.

- [x] **T9 VTK permutation generator (A.9)**
  - Goal: `vtk_to_internal` for any p ≥ 1 (pyramids p = 1 only, A.8), from VTK's rules.
  - Crates/files: `gridcl-refcell/src/node_maps/vtk.rs`, `vtk_to_internal` in
    `src/node_maps.rs`; `conformance/tests/vtk.rs`.
  - Public API added or changed: body of `vtk_to_internal`.
  - Tests: `node_maps::vtk_reproduces_a9`; proptest `vtk_maps_are_permutations`; extra
    conformance test `vtk_matches_ndelement` (not an acceptance test): `vtk_to_internal`
    equals ndelement's `reorderings::vtk_ordering` (same direction) for its six cells, p ≤ 4.
  - Verify: `cargo test -p gridcl-refcell --test node_maps vtk && cargo test --manifest-path conformance/Cargo.toml --test vtk`
  - After: T8 (shared `node_maps.rs`)
  - Parallel with: T5, T6

- [x] **T10 Fixture export and workspace fixture tests**
  - Goal: the `export-fixtures` binary of A.10, the committed fixtures, and gridcl-refcell tests
    that check tables, orientation bits and point sets against them without ndelement.
  - Crates/files: `conformance/Cargo.toml` (`[[bin]] name = "export-fixtures"`, path
    `src/bin/export_fixtures.rs`), `conformance/src/bin/export_fixtures.rs`,
    `conformance/src/lib.rs` (docs: drop "(from M1b)"),
    `gridcl-refcell/tests/fixtures/{tables,orientation,lagrange_points,sources}.json`,
    `gridcl-refcell/tests/fixtures.rs`.
  - Public API added or changed: none.
  - Fixtures: `tables.json` (ndelement's vertices, sub-entities and types, eight cells);
    `orientation.json` (each cell with D_TOP ≥ 2: `compute_orientation` of every permutation of
    0..V in lexicographic order, question 5); `lagrange_points.json` (the five cells, p ≤ 4,
    equispaced, and GLL on intervals, quadrilaterals and hexahedra); `sources.json` (ndelement
    f8cbffe and the pinned Basix, DOLFINx and Symfem commits, question 7).
  - Tests: `fixtures::tables_match_fixture`, `fixtures::orientation_matches_fixture`,
    `fixtures::lagrange_points_match_fixture` (serde_json `Value`, no serde derive); the whole
    conformance suite.
  - Verify: `cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures && git diff --exit-code gridcl-refcell/tests/fixtures` (after the first commit of the fixtures: a re-export changes nothing), `cargo test --manifest-path conformance/Cargo.toml`, `cargo clippy --manifest-path conformance/Cargo.toml --all-targets -- -D warnings`, `cargo test -p gridcl-refcell --test fixtures`, `scripts/check-workspace-deps.sh`; then the full M3 Max set: `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `RUSTDOCFLAGS=-Dwarnings cargo doc --workspace --no-deps`, `cargo fmt --all --check`, and unsandboxed `GRIDCL_TEST_DEVICE=metal cargo test --workspace --features gridcl-testing/metal`.
  - After: T2, T4 (fixtures); T7, T9 (final full-suite run)
  - Parallel with: –

## Acceptance coverage

Every acceptance bullet of the milestone in docs/design.md appears exactly once.

Targets for every row: M3 Max: `cargo test --workspace` (CPU runtime) and the Metal workspace
run; conformance rows on the M3 Max with Accelerate and in the hosted conformance job with
OpenBLAS (it triggers on pushes touching `gridcl-refcell/` or `conformance/`); hosted CI: cpu and
vulkan jobs; at the milestone review Myriad V100 and Locust run
`cargo test --release --workspace --features gridcl-testing/cuda`. The tests are host-only (f64
host maths, no device code): the device runs show only that they build and pass on every target,
and the f32/f64 kernel rule does not apply. MPI: not applicable. Myriad A100: not required (no
M1b review in "Development environment and test targets"). Vector-size invariance: M1b has no
kernels, so none is on the candidate list.

| Acceptance criterion (docs/design.md) | Test name(s) | Task(s) | Targets and backends |
| --- | --- | --- | --- |
| node counts match A.4 for p ≤ 10 | `gridcl-refcell::node_counts::node_counts_match_a4` | T0, T4 | as above |
| partition of unity and reproduction of each element's own space to 10^−12 for p ≤ 8 (f64) | `gridcl-refcell::tabulation::partition_of_unity`, `reproduces_own_space` | T0, T7 | as above |
| sub-entity tables equal ndelement's for all eight cells | `gridcl-conformance::tables::sub_entities_match_ndelement`; workspace mirror `gridcl-refcell::fixtures::tables_match_fixture` | T0, T1, T10 | conformance (Accelerate, OpenBLAS); mirror as above |
| orientation bits equal `compute_orientation` for 10 000 random vertex numberings per cell type with D_TOP ≥ 2 | `gridcl-conformance::orientation::bits_match_compute_orientation`; mirror `gridcl-refcell::fixtures::orientation_matches_fixture` | T0, T2, T10 | conformance; mirror as above |
| point sets equal ndelement's node by node for intervals, triangles, quadrilaterals, tetrahedra and hexahedra, p ≤ 4, equispaced and (tensor cells) GLL | `gridcl-conformance::lagrange::points_match_ndelement`; mirror `gridcl-refcell::fixtures::lagrange_points_match_fixture` | T0, T4, T10 | conformance; mirror as above |
| the generators reproduce every A.9 row and DOLFINx's Gmsh P3 tables | `gridcl-refcell::node_maps::gmsh_reproduces_a9`, `vtk_reproduces_a9`, `gmsh_p3_matches_dolfinx` | T0, T8, T9 | as above |
| quadrature integrates monomials (rationomials on pyramids) up to its degree to 10^−13 | `gridcl-refcell::quadrature::integrates_monomials` | T0, T3 | as above |
| `cargo test --workspace` still builds neither ndelement nor BLAS | `scripts/check-workspace-deps.sh` (M0) | T0, T10 (the only tasks that change dependencies or add fixtures) | M3 Max; hosted cpu job |

## Open questions and ADR candidates

Resolve every row before Status becomes approved.

| # | Question or conflict | Design section | Proposed resolution | Status |
| --- | --- | --- | --- | --- |
| 1 | Branch `tbetcke/tbetcke-m1b-reference-cells`, not `m1b-<slug>`. | CLAUDE.md Workflow | Keep it and log it, as in M0 and M1a. | answered: as proposed |
| 2 | No Architecture listing defines gridcl-refcell's API, yet later milestones "depend only on the public interfaces of earlier ones". | Architecture; Roadmap intro; Workspace layout | Record the API in this plan only ("Planned public API"); no listing changes, no ADR. | answered: plan only |
| 3 | "Requesting GLL on any other cell returns `GridError::UnsupportedNodeFamily`", but `GridError` lives in gridcl-device, which does not depend on gridcl-refcell; gridcl-refcell depends only on gridcl-dims. A `From` impl fits in neither crate (orphan rule). | Reference cells and conventions; Device containers; Workspace layout | gridcl-refcell returns `RefCellError::UnsupportedNodeFamily { cell, family }`; gridcl-geometry (M3) maps it with `map_err` onto `GridError::UnsupportedNodeFamily`, so the gridcl-level behaviour is as designed. Logged, follow-up for M2/M3; an ADR only if `GridError` must carry a refcell type or a `From` impl. | answered: as proposed |
| 4 | "Tables … generated by the lexicographic rule" and "usable inside `comptime!` blocks". | Roadmap M1b; Reference cells and conventions; A.2 | `const` literal tables plus an internal generator (facet planes, lexicographic sort) that a unit test requires to agree for all eight cells; conformance checks the tables. | answered: const + generator test |
| 5 | The workspace mirror of the orientation test needs data without ndelement. | A.10 Fixtures | Conformance keeps the 10 000-random acceptance test; the fixture holds every permutation of 0..V (the bits depend only on the id order), about 300 kB for the hexahedron. | answered: all permutations |
| 6 | CLAUDE.md asks for property tests and "Testing and validation" names node-ordering permutations; the crate table gives gridcl-refcell only serde_json. | Testing and validation; Workspace layout | `proptest = "1"` as a dev-dependency (workspace dependency), logged like trybuild in M1a; no ADR. | answered: as proposed |
| 7 | A.10 names pinned Basix, DOLFINx and Symfem commits, and the P3 test needs DOLFINx's arrays, but no commits are given. DOLFINx main (checked 2026-10-19) has all five arrays (`gmsh_tetrahedron` 20, `gmsh_quadrilateral` 16, `gmsh_hexahedron` 64, `gmsh_prism` 40, `gmsh_pyramid` 30). | A.9; A.10; High-order node orderings | Pin each project's latest release tag at T0 (full SHA in `sources.json` and the test's comment). T0 confirms the five arrays exist at the DOLFINx tag; if one is missing, stop and draft an ADR. | answered: latest release tags |
| 8 | Not in M1b's scope or acceptance: reference facet normals (Geometry, "Facet normals"), the A.6 storage slots, the A.7/A.8 type tables, Gmsh-file and VTK P3 prism fixtures ("High-order node orderings"). | Roadmap M1b, M3–M5; Geometry; A.6–A.8 | Facet normals and storage slots with M3; type tables and file fixtures with M4/M5. Logged as follow-ups. T9's ndelement VTK check covers VTK p ≤ 4 on six cells meanwhile. | answered: as proposed |
| 9 | Degree range: "any p"; ndelement allows p = 0 (discontinuous). | A.4, A.5 | p ≥ 1 for layouts, elements and maps (`UnsupportedDegree` for 0); no upper limit; tests to p = 10 (counts), 8 (tabulation), 6 (map proptests). | answered: as proposed |
| 10 | Tolerances: "equal node by node", "to 10^−12" and "to 10^−13" give no norm. | Roadmap M1b; Testing and validation | Points: 1e-14 absolute per coordinate. Quadrature: 1e-13 relative to the exact integral. Partition of unity: |Σφ − 1| ≤ 1e-12 absolute. Reproduction: 1e-12 relative to the max norm of the monomial on the cell. | answered: as proposed |
| 11 | "Up to its degree" for pyramid rules (and the `gauss_rule(Pyramid, 2p)` used by T5–T7) is undefined. | Tabulation and quadrature; A.5 | `gauss_rule(Pyramid, n)` is exact for the rationomials of A.5's space of degree n, i.e. ξ^a η^b (1−z)^max(a,b) z^c with max(a,b) + c ≤ n in collapsed coordinates (times the Jacobian (1−z)^2). Then `2p` integrates products of two degree-p functions. | answered: as proposed |
| 12 | `num_lagrange_nodes` returns a plain `usize` (Listing B5), so p = 0, A.4's p−1 and 2p−3 factors at small p, overflow at huge p and the Point cell need a defined behaviour. | A.4; Listing B5 | Point: N = 1, interior 0. Other cells: A.4's formulas in checked u64 arithmetic; interior counts are 0 below each cell's first interior degree; p = 0 gives 1 node (no interior); overflow panics with a message naming cell and degree. | answered: as proposed |
| 13 | DOLFINx v0.11.0.post0 (latest release) lacks `gmsh_hexahedron` 64, `gmsh_prism` 40 and `gmsh_pyramid` 30; they entered main in c9e0341 (2026-10-10). | A.9; A.10; High-order node orderings | Pin DOLFINx at main be8352e; Basix v0.11.0 and Symfem v2025.12.0 stay release tags. | answered: ADR-0007 accepted |

## Deviations log

| Date | Task | Deviation from plan or design | Reason | ADR |
| --- | --- | --- | --- | --- |
| 2026-10-19 | plan | Branch `tbetcke/tbetcke-m1b-reference-cells`, not `m1b-<slug>`. | Maintainer's worktree branch, as in M0 and M1a (question 1). | – |
| 2026-10-19 | plan | gridcl-refcell returns `RefCellError`, not `GridError::UnsupportedNodeFamily`. Follow-up for M3: gridcl-geometry maps it with `map_err`. | Crate graph (question 3). | – |
| 2026-10-19 | plan | gridcl-refcell gets a `proptest` dev-dependency, not in the crate table. | Property tests for node-ordering permutations (question 6). | – |
| 2026-10-19 | plan | Reference facet normals and A.6 storage slots deferred to M3; A.7/A.8 tables, Gmsh-file and VTK P3 prism fixtures to M4/M5. | Not in M1b's scope or acceptance (question 8). | – |
| 2026-10-19 | T0 | DOLFINx pinned at main be8352e677303b686d6862b70923bab086b279ff, not the latest release tag (question 7); Basix v0.11.0 and Symfem v2025.12.0 stay release tags. | v0.11.0.post0 lacks hex64, prism40 and pyramid30 (question 13). | ADR-0007 |
| 2026-10-19 | T0 | gridcl-conformance gets a `gridcl-dims` path dependency and, in `src/lib.rs`, `CELLS` and `reference_cell_type` (gridcl → ndelement cell type), shared by the tests and T10's exporter. | The tests need `CellType`, which gridcl-refcell does not re-export. | – |
| 2026-10-19 | T0 | The stub modules carry `#![allow(unused_variables)]`, removed by the task that fills each one in; `polynomials.rs` and `node_maps.rs` declare their submodules instead of being empty. | `todo!()` bodies leave the parameters unused; empty parents could not hold `pyramid.rs`, `gmsh.rs` and `vtk.rs`. | – |
| 2026-10-19 | T1 | `tdim`, `vertices`, `sub_entities`, `sub_entity_types` and `num_sub_entities` are `pub const fn`, not `pub fn`; `generator.rs` is `#![cfg(test)]`. | Usable in `const` and `comptime!` contexts ("Reference cells and conventions"); the generator is only used by the test that checks the tables. | – |
| 2026-10-19 | T3 | `gauss_rule` points are in tensor order of the collapsed coordinates, first coordinate fastest, with degree/2 + 1 points per direction (documented on `gauss_rule`). | The design gives no point order; a choice, not an API change. | – |
| 2026-10-19 | T3 | `gll_points` rounds each lower-half point to 1 − fl(1 − t) before mirroring (shift ≤ 1.1e-16), so the points are exactly symmetric. | Exact symmetry; `points_match_ndelement` (GLL, 1e-14) confirms it in T4. | – |
| 2026-10-19 | T3 | Done by a worktree subagent and merged with `--no-ff` (b060e45); the task was ticked in a separate commit after the merge, as the parallel-task procedure requires. | Parallel with T2. | – |
| 2026-10-19 | T5 | The span test checks Parseval (‖m‖² = Σ (m, φ_k)² for every monomial of the space, plus the dimension) instead of computing the rank of the monomial Vandermonde. | Equivalent for an orthonormal set and needs no rank-revealing factorisation. | – |
| 2026-10-19 | T6 | At the apex the rational pyramid functions (min(i, j) ≥ 1) return their limit along the axis x = y = s/2: value and first derivatives 0. Unit test `apex_is_the_axis_limit`. | Their derivatives have no limit over all directions; the plan said only "the apex is handled by the limit". | – |
| 2026-10-19 | T7 | Root `Cargo.toml` gets `[profile.dev.package.gridcl-refcell] opt-level = 3` (debug assertions and overflow checks stay on). | Unoptimised, `tests/tabulation.rs` took 80 s (729 × 729 Vandermonde systems for hexahedra, p = 8); with it 7 s. | – |
| 2026-10-19 | T7 | Achieved p = 8 errors (partition of unity = reproduction, max over test points): hexahedron equispaced 5.6e-13, prism 1.2e-13, pyramid 4.6e-14, hexahedron GLL 6.0e-14, all others ≤ 2.5e-14. All within 1e-12, so no ADR; the equispaced hexahedron keeps a margin of only about 2. | Recorded per T7's risk note. | – |
| 2026-10-19 | T8 | Gmsh rules ported from Gmsh 4.15.2 (tag `gmsh_4_15_2`): `pointsGenerators.cpp` and the `M*.h` edge/face tables, cited in `node_maps/gmsh.rs`. The prism interior is Gmsh's non-recursive triangle(p−3) × line(p−2) rule; pyramid lattices map as (m0, m1, p − m2). Cross-checked in scratch against the gmsh 4.15.2 Python wheel for 68 (cell, p) pairs (up to p = 10 / 9 / 8). The shared lattice matcher `match_lattice` is private to `node_maps`. | Done by a worktree subagent, merged with `--no-ff`, ticked after the merge (parallel with T5–T7). | – |
| 2026-10-19 | T9 | VTK rules ported from VTK v9.7.1 (`vtkHigherOrder{Triangle,Quadrilateral,Tetra,Hexahedron,Wedge,Curve}.cxx`, `vtkHigherOrderInterpolation.cxx`), VTK 9 hexahedron numbering (XML ≥ 2.2). VTK's wedge orders its triangle faces and interior triangle layers lexicographically, not recursively; prism p ≥ 3 checked against VTK's independent `AppendWedgeCollocationPoints` (p ≤ 9) and DOLFINx `vtk_wedge` (p ≤ 2). Done by a worktree subagent, merged with `--no-ff`, ticked after the merge. | "High-order node orderings" says "triangle interiors are recursive, others lexicographic"; true for VTK triangles and tetrahedra, not for prism faces. Design-wording clarification for the review (candidate ADR, no API change). | – |
| 2026-10-19 | T10 | `orientation.json` and `lagrange_points.json` are written compact (409 kB and 16 kB); `sources.json` reads ndelement's commit from conformance/Cargo.lock and records repository, ref and commit for each source. | Deterministic output that follows the pin; a re-export is byte-identical. | – |
| 2026-10-19 | finish | Finish run: workspace tests, clippy, doc, fmt and Metal pass on the M3 Max; conformance passes; Locust CUDA (`cargo test --release --workspace --features gridcl-testing/cuda`, job 20261019233247-3771866, 09d8b9c, clean tree) passes. Myriad V100 not run: both submissions failed because Myriad's SGE qmaster refused connections (util01, port 6444); nothing queued. The V100 run moves to the milestone review. | Remote scheduler outage; CLAUDE.md allows two attempts. | – |
| 2026-10-19 | plan | T0 exceeds the usual ~600 changed lines. | All acceptance tests with their literals in one failing commit, as the template requires. | – |

## Review

<Filled by /review-milestone.>

