# ADR-0012: M3 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-21
- **Raised in:** M3 planning (docs/plans/M3.md, questions 2–11 and 21–23)
- **Design sections affected:**
  - "Architecture":
    - Listing B3 (`CellBlock`, `CoordinateElement`, `GeometryBlock`, the origin-id comment);
    - Listing B4 (`boundary`);
    - Listing B4b (`QuadField`, `PointSet`, `FacetQuadRule`);
    - Listing B5 (`gb.evector()`);
    - Listing B7 (`DeviceGrid`, and the paragraph after it);
  - "Reference cells and conventions" (origin ids);
  - "Device model and backends":
    - the paragraph after Listing S1 (`GridError` variants, public helpers);
    - "Data layouts and index widths";
  - "Topology construction on the device": "Entity creation" steps 1, 2, 4, 5 and 9, and Listing T1 (`ConnKey`);
  - "Geometry / Geometric factors" (caching, facet normals);
  - "Testing and validation" (vector-size invariance at tile width 1);
  - "Implementation roadmap": M3 (scope and the last acceptance bullet) and M9 (scope);
  - "Workspace layout" (crate table);
  - "Handing this design to Claude Code" (Listing H1) and CLAUDE.md (the host-copy rule).

## Context

M3 builds the serial mesh core: `Mesh` from host arrays, device entity creation, connectivity,
permutation info, geometry blocks, quadrature rules, E-vectors, the tiled `QuadField`, the factor
kernels and `measure`. It is the first user of most of the Architecture listings.

The listings name these types and methods but leave open the items that M3's T0 must stub with
final signatures. Some listings also do not compile or conflict with the crate graph once their
items are placed in crates. Planning read the M3-relevant listings against the code of M1a–M2.
The design-reviewer then critiqued the draft plan; its findings are folded in here.

The following facts from the existing code matter:

- **gridcl-device depends only on cubecl and toml.** It cannot name gridcl-dims, gridcl-refcell or gridcl-topology types, but `GridError` lives there (ADR-0009).
- **The test overrides are crate-private.** `testing::properties` is `pub(crate)` in gridcl-device. A kernel launcher in gridcl-topology or gridcl-geometry that reads `client.properties()` does not see `testing::with_properties`. M3's acceptance test "with an injected limit, `QuadField` results are identical with 1 and 7 chunks" depends on these overrides.
- **Helpers other crates need are crate-private too.** `pad_len`, which gives the `SortKeys` plane stride, and `debug_assert_bindings` are `pub(crate)`.
- **gridcl-refcell lacks three host items M3 needs:** reference facet normals, facet-to-cell point maps and the A.6 storage slots. M1b deferred them to M3 (docs/plans/M1b.md, question 8).

## Problem

1. **Crate placement of `DeviceGrid`.**
   - The crate table and Listing B7 put `DeviceGrid` in gridcl.
   - Listing B4b's `QuadRule::for_grid<G: DeviceGrid<…>>` and Listing B5's `jacobians_block<…, G: DeviceGrid<…>>` are in gridcl-geometry.
   - gridcl depends on gridcl-geometry, so gridcl-geometry cannot name a trait defined in gridcl.
2. **The E-vector cache and the origin.**
   - **No place for the cache.** Listing B5 calls `let xe = gb.evector()?; // cached` on a `GeometryBlock<TD>`, which Listing B3 defines without a scalar parameter. It cannot hold a `DeviceBuffer<F>` cache.
   - **The cache contradicts B7's rule.** The paragraph after Listing B7 says "Accessors never create data". A lazy cache does create data, yet the M3 scope requires E-vectors "built on demand and cached per geometry block".
   - **No way to read the origin.** `physical_points` must add back the per-mesh origin ("Coordinates relative to an origin"), but `DeviceGrid` has no method that returns it. Generic block functions therefore cannot compute X.
3. **`ConnectivityNotCreated(key)`.** Listing B7 documents `Err(GridError::ConnectivityNotCreated(key))`, and Listing T1 places `ConnKey` in gridcl-topology. `GridError` lives in gridcl-device, which does not depend on gridcl-topology.
4. **Error variants and helpers.** "Later milestones add their variants when they first use them". Their names, payloads, and the cell and family types in them are not given, and the cell and family types cannot be named in gridcl-device. M3 also needs `pad_len`, a properties accessor that honours the test overrides, and `debug_assert_bindings`, all from outside gridcl-device.
5. **Origin-id encoding.** On the device, origin ids are "interleaved (hi, lo) u32 pairs, with a comptime fast path when every id fits in u32". The design specifies no encoding, so the following are undefined:
   - how a signed i64 compares as an unsigned pair;
   - what "fits in u32" means for negative ids.
6. **Entity and connectivity dimensions.**
   - `create_entities(dim)` and `create_connectivity(from, to)` do not say which dimensions and pairs exist.
   - Listing T1 requires `EntityBlock::vertices` in canonical orientation (A.6), but cell vertices are stored in reference order. Cells therefore cannot simply be returned as entity blocks of dimension tdim.
7. **Key emission layout.** Steps 1–2 give block b the emission range `[e_b, e_b + n_b · k_b)` and leave the index order inside it open.
   - TopologyKeys is vector-generic (CLAUDE.md, candidate list), with lanes over consecutive cells.
   - **Cell-major order** (`cell · k_b + i`) strides lanes by k_b.
   - **Sub-entity-major order** (`i · n_b + cell`) is contiguous, but vector-aligned only if `n_b` and `e_b` are multiples of the vector size. Vector indices count whole vectors (Listing S1, `vector_arg`).
8. **Missing type definitions.**
   - `PointSet`, which Listings B4b and B5 use, has no definition.
   - `FacetQuadRule` "holds point sets on each reference facet type", but `facet_normals` returns one `QuadField` per block, whose `num_points` must be one number per cell. The facets of a prism or pyramid have different point counts.
   - `QuadField` has no accessors for its chunks or sizes, which the 1-versus-7-chunks acceptance test and downstream crates need.
9. **The builder.**
   - "Mesh from host arrays, built with origin ids" has no listing.
   - Numbering is stated only for the parallel pipeline (stage 8: "vertices and nodes in order of first touching cell").
   - A node can be listed as a vertex of one cell and as a non-vertex node of another, which no rule covers.
10. **The M3/M9 boundary.**
    - M3's scope lists the factor kernels "X, J, det J, K, facet normals". M9's lists "normals, pseudo-inverse and facet normals".
    - The M1a compile tests, which M3 must run against the real `Mesh`, call `normals()` and `boundary()`. `boundary()` needs `EntityMap` and submesh extraction, which are M9 scope.
11. **Host copies.** CLAUDE.md allows host↔device copies "only for I/O, explicit to_host()/read_node(), the count read of run-length and unique-by-key, the debug-build index checks of gather and scatter, and the host-staged halo exchange". The design needs three copies that this list does not name:
    - `measure` "finished in f64 on the host" ("Precision policy") needs one device read per block;
    - mesh creation from host arrays uploads its input;
    - `QuadRule` creation and the first use of a tabulation upload points and tabulated values ("uploaded once per (element, point set)").
12. **Factor caching.** "Geometric factors" says factors are "cached by (rule id, factor flags)", but `QuadRule` has no id, the M3 scope does not list a factor cache, and the B4 methods return owned `BlockFields`.
13. **Vector sizes at tile width 1.** The last M3 acceptance bullet requires agreement "at every vector size from 1 to 16 and at tile widths 1 and 16".
    - Lanes are cells of one tile (Listing B5: "N divides the tile"), so at tile width 1 only N = 1 exists.
    - GPUs offer at most 4 lanes (wgpu) or the widest IO-optimized size.
    - "Testing and validation" reads "every larger power of two up to 16 that the device offers".

## Options

1. **Keep the design as written.**
   - **Does not compile or contradicts itself:** `DeviceGrid` in gridcl (problem 1), `gb.evector()` on `GeometryBlock<TD>` (problem 2), and `ConnectivityNotCreated(ConnKey)` in gridcl-device (problem 3).
   - **Cannot pass:** the 1-versus-7-chunks test, because injected limits do not reach other crates (problem 4).
   - **Settled ad hoc:** the remaining items would be decided during M3 and recorded only in the plan's Deviations log, although M4–M9 build on them.

   Not chosen.
2. **Decide every item in one ADR and apply it to the design** (detailed below). The public API grows by the items listed under Consequences, and three items move between crates: `DeviceGrid`, `CoordinateElement`/`GeometryBlock` and `ConnKey`. CLAUDE.md's host-copy rule gains three cases.
3. **Option 2, but with a wider crate graph:** gridcl-device depends on gridcl-dims and gridcl-refcell, so `GridError` can carry `CellType`, `NodeFamily` and `WrongCellDim` directly. Both crates are host-only and free of other dependencies.
   - It changes dependency rule text and the crate table, for one error type.
   - Not chosen: `&'static str` names follow `UnsupportedPrecision`, the precedent of ADR-0009.

Sub-options considered for single items:
- **Problem 2:** build every E-vector eagerly at mesh creation. B7 would stay literal, but every mesh would pay the memory and time, even when it never computes factors.
- **Problem 3:** add a `ConnectivityNotCreated { from: u8, to: u8 }` variant instead of moving `ConnKey`.
- **Problem 5:** store the raw words and reject negative ids.
- **Problem 10:** make `boundary()` panic with `unimplemented!`.

The maintainer chose the proposed sub-option for problems 2, 5 and 10 on 2026-10-21 (docs/plans/M3.md, questions 4, 8 and 5).

## Proposed decision

Option 2. Each item is decided as follows.

### Crate placement (problems 1 and 3)

- **gridcl-geometry:** `DeviceGrid` (Listing B7), `CoordinateElement<TD>` and `GeometryBlock<TD>` (Listing B3), and a `Geometry<F, TD, GD>` container (geometry blocks, nodes, node `IndexMap`, origin, E-vector cache). gridcl re-exports them.
- **gridcl-topology:** `CellBlock<TD>` (Listing B3), `EntityBlock` and `Connectivity` (Listing T1), and a `Topology<TD>` container (cell blocks, vertex `IndexMap` and origin ids, entity blocks, connectivity, permutation info).
- **gridcl-device:** `ConnKey` moves here (Listing T1's definition, unchanged), so that `GridError::ConnectivityNotCreated(ConnKey)` stays as Listing B7 writes it. gridcl-topology and gridcl re-export it.
- **gridcl:** `Mesh`, its aliases, `MeshBuilder`, `AnyMesh`, `AnyMeshRef`, `with_mesh!`, `EntityMap` and the prelude.

### gridcl-device (problem 4)

- **New `GridError` variants.** Cell types and node families are named by `&'static str` (`"Tetrahedron"`, `"Gll"`).
  - `UnsupportedNodeFamily { cell: &'static str, family: &'static str }` and `UnsupportedDegree { cell: &'static str, degree: u32 }`. gridcl-geometry maps `gridcl_refcell::RefCellError` onto them.
  - `WrongCellDim { found: &'static str, expected_tdim: usize }`.
  - `DimensionMismatch { expected: (usize, usize), found: (usize, usize) }` (Listing B6).
  - `EntitiesNotCreated(usize)`, `ConnectivityNotCreated(ConnKey)` and `UnsupportedConnectivity(ConnKey)`.
  - `DuplicateOriginId(i64)`, `IndexOutOfRange { index: usize, len: usize }` and `NodeRoleMismatch(usize)`.
  - `DuplicateCellBlock(&'static str)` and `CellNotInRule(&'static str)`.
  - `NotYetImplemented(&'static str)`: a method declared before the milestone that implements it (problem 10).
- **Made public:**
  - `pad_len(len) -> Result<usize>`;
  - `device_properties(&Client) -> Arc<DeviceProperties>`: the properties gridcl-device itself reads, honouring `testing::with_properties`. Kernel launchers in every crate read limits through it;
  - `debug_assert_bindings(&Client, n)`.

### gridcl-refcell (host, f64)

- `reference_facet_normals(cell: CellType) -> Vec<f64>`: the outward unit normal of every facet, in facet order, with `tdim(cell)` entries per facet.
- `facet_points_to_cell(cell, facet: usize, points: &[f64]) -> Vec<f64>`: maps points on the reference facet into cell coordinates through the facet's vertex list, with A.5's affine rule (v0 + Σ t_k (v_k − v0)).
- `canonical_order(entity: CellType, ids: &[i64]) -> Vec<usize>`: the A.6 storage slots, giving the local positions of the entity's vertices in canonical order.

### Origin ids on the device (problem 5)

- **Encoding.** Each id is stored as the u32 pair `(hi ^ 0x8000_0000, lo)`, where hi and lo are the high and low words of the i64. Unsigned lexicographic pair order then equals i64 order, so negative ids stay valid.
- **Fast path.** It applies when every hi word equals `0x8000_0000`, that is, when 0 ≤ id < 2^32. The mesh records this as a flag at creation, and kernels compare only the lo words.

### gridcl-topology (problems 6 and 7)

- **`CellBlock::new`.** `CellBlock::new(cell: TD::Cell, index_map, vertices, origin_ids, flags) -> CellBlock<TD>`, so that Listing B10's `CellBlock::<D2>::new(Cell3::Tetrahedron, …)` exists.
- **Dimensions.**
  - `create_entities(d)` accepts 0 ≤ d < tdim. Dimension 0 gives one `Point` block of the vertices.
  - Cells stay in `cell_blocks()` in reference order, so `entity_blocks(tdim)` returns `EntitiesNotCreated(tdim)`.
  - `create_connectivity` supports two kinds of pair, for 0 ≤ d < tdim:
    - `(tdim, d)`: `Downward`, filled by entity creation;
    - `(d, tdim)`: `Upward`, a `DeviceCsr`.
  - Two pairs are not stored: `d → 0` is `EntityBlock::vertices` and `tdim → 0` is `CellBlock::vertices`.
  - Every other pair returns `UnsupportedConnectivity`.
- **Key emission.**
  - **Index.** Block b emits sub-entity i of cell c at `e_b + i · n̂_b + c`, where `n̂_b` is `n_b` rounded up to `VECTOR_PAD`, and `e_b` is the prefix sum of `k_b · n̂_b`. The lanes of a vector are then aligned consecutive cells.
  - **Sentinels.** Padding positions write an all-ones key, which sorts last and forms one group. Entity creation drops that group from the entity count and from the decode.
  - **Limit.** The 2^32 limit applies to the padded total.
- **Steps 4–5** use `unique_by_key`: its `group` is the provisional id, and its count read is the one CLAUDE.md allows.
- **Step 9** fills each row of entity → cells in cell order with a stable radix sort, with offsets from `histogram` and `exclusive_scan`. The result is deterministic without atomics.

### gridcl-geometry (problems 2 and 8)

- **`PointSet<F, TD>`.**
  - Host data in reference coordinates: `cell()`, `len()`, `points() -> &[f64]` and `weights() -> &[f64]`.
  - `tabulate(&CoordinateElement<TD>) -> Result<DeviceBuffer<F>>` gives values `[q][k]`, and `tabulate_derivatives(…)` gives `[q][k][d]`.
  - Each tabulation is computed in f64, narrowed, uploaded on first use, and cached per element; calls return `share()`s.
- **`FacetQuadRule<F, TD>`.**
  - `gauss(client, cells, degree)`, `for_grid(grid, degree)` and `get(cell) -> Result<&FacetPointSet<F, TD>>`.
  - A `FacetPointSet` is one `PointSet` holding every facet point of the cell in cell coordinates, facet-major, with the facet's weights. It also provides `facet_offsets() -> &[usize]` and `reference_normal(q)`.
  - `facet_normals` therefore has Σ_f n_f points per cell.
- **`QuadField` additions:** `num_cells()`, `num_points()` and `chunks() -> &[QuadChunk<F>]`. `uninit` sizes its chunks from `device_properties(client).memory.max_page_size`.
- **`EVector<F>`.** The cached E-vector of one geometry block, with `num_cells()` and `chunks()`.
  - Layout: chunks of whole tiles, `[k][g][cell]`.
  - Padding: the cell count is padded to `VECTOR_PAD` with copies of the block's last cell.
  - An empty block has none.
- **`DeviceGrid` additions:**
  - `fn evector(&self, b: usize) -> Result<&EVector<Self::F>>`: built on first use and cached in a `OnceCell` inside `Geometry`. Listing B5 calls `g.evector(b)?` instead of `gb.evector()?`.
  - `fn origin(&self) -> <Self::GD as Dim>::Array<f64>`.
- **B7's rule** becomes: "Accessors create no topology: entities and connectivity are built only by the `&mut self` methods `create_entities` and `create_connectivity`. Derived geometry data (E-vectors) is built on first use and cached."
- **Block functions** are generic over `G: DeviceGrid`, as in Listing B5: `physical_points_block`, `jacobians_block`, `detj_block`, `pseudo_inverse_block`, `facet_normals_block`, `normals_block` and `measure_block` (Listing B11).

### gridcl (problems 9 and 10)

- **`MeshBuilder<F, TD, GD>`.**
  - `new(client)`.
  - `nodes(coords: &[f64], origin_ids: Option<&[i64]>)`: coordinates `[node][gdim]` in f64. Without ids, each id is the index.
  - `cell_block(element: CoordinateElement<TD>, nodes: &[u32], origin_ids: Option<&[i64]>)`: cell→node lists in A.5 order. A cell's vertices are its first V nodes, and vertex origin ids are the node origin ids of the corner nodes.
  - `build() -> Result<Mesh<F, TD, GD>>`.
- **Builder checks:**
  - `check_precision`;
  - lengths (`LengthMismatch`) and index ranges (`IndexOutOfRange`);
  - unique node and cell ids (`DuplicateOriginId`);
  - every node is either a vertex of all cells that list it or a vertex of none (`NodeRoleMismatch`);
  - at most one block per cell type (`DuplicateCellBlock`).

  Empty blocks are allowed and launch nothing.
- **Builder numbering** follows pipeline stage 8: vertices and nodes in order of the first cell that touches them, with blocks in call order.
- **Coordinates.** The origin is the centre of the bounding box (0 for an empty mesh). Coordinates are stored relative to it.
- **M3/M9 boundary.**
  - M3 implements every factor method of Listing B4, `normals` included. Its sign comes from the flags, which stay 0 until `orient`.
  - `boundary()` is declared with Listing B4's signature and returns `Err(NotYetImplemented("boundary: M9"))` until M9. `EntityMap` is an opaque placeholder until then.
  - `facet_submesh` and `orient` are declared in M9; they need `MeshTags` and `OrientationRule`.
- **Tags.** The `tags: TagRegistry` field of Listing B3 is added in M4, its first user. It is a private field.

### Rules (problems 11, 12 and 13)

- **Host copies.** CLAUDE.md and Listing H1 read: "Host<->device copies in library code only for I/O (including the upload of input arrays at mesh creation, and of quadrature points and tabulations when a rule is created or a tabulation is first cached), explicit to_host()/read_node(), the count read of run-length and unique-by-key, the per-block read that finishes `measure` in f64, the debug-build index checks of gather and scatter, and the host-staged halo exchange."
- **Factor caching** by (rule id, factor flags) is deferred to the first milestone that needs it. Until then, the B4 methods compute and return fresh `BlockFields`.
- **Vector sizes and tiles.** Factor kernels plan with `max_vector = tile`.
  - At tile width 1 they run at vector size 1.
  - At tile width 16 the invariance tests sweep 1 and every larger power of two up to 16 that the device offers.
  - The M3 acceptance bullet is read that way.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B3:**
    - `CellBlock::new`;
    - the `origin_ids` comment states the sign-flipped encoding;
    - `CoordinateElement` and `GeometryBlock` are marked "(gridcl-geometry, re-exported by gridcl)";
    - a note that `TagRegistry` arrives in M4.
  - **Listing B4:** a doc line on `boundary`, "returns `NotYetImplemented` until M9".
  - **Listing B4b:**
    - `PointSet`, `FacetQuadRule` and `FacetPointSet`;
    - the `QuadField` accessors;
    - chunk sizing through `device_properties`.
  - **Listing B5:** `g.evector(b)?`.
  - **Listing B7:**
    - "(gridcl-geometry, re-exported by gridcl)";
    - `evector` and `origin`;
    - the reworded rule in the paragraph after it.
  - **Listing T1:** `ConnKey` is marked "(gridcl-device, re-exported)".
  - **The paragraph after Listing S1:**
    - the M3 variants;
    - `pad_len`, `device_properties` and `debug_assert_bindings` are public.
  - **"Reference cells and conventions" (origin ids) and "Data layouts and index widths":** the encoding and the fast-path condition.
  - **"Entity creation":**
    - steps 1–2: the padded, sub-entity-major emission and the sentinel;
    - steps 4–5: `unique_by_key`;
    - step 9: the stable radix sort;
    - the dimension and pair rules.
  - **"Geometric factors":**
    - caching deferred;
    - the facet-major layout of facet normals.
  - **"Testing and validation", vector-size invariance:** factor kernels at tile width 1 run at size 1.
  - **Roadmap:**
    - M3 scope adds "normals" to the factor kernels and "`AnyMesh` and `DeviceGrid` (BLK-3)";
    - M3's last acceptance bullet reads "at every vector size from 1 up to 16 that the device offers and the tile holds";
    - M9 scope reads "boundary, facet and edge submeshes … ; the manifold tests of normals, pseudo-inverse and facet normals".
  - **Crate table:**
    - gridcl-geometry gains `DeviceGrid`;
    - gridcl-device gains `ConnKey`;
    - gridcl-refcell gains "reference facet normals, facet point maps, A.6 storage slots";
    - gridcl-testing gains gridcl-dims and gridcl-refcell;
    - gridcl-topology and gridcl-geometry gain the dev-dependency gridcl-testing.
  - **Listing H1:** the host-copy line, as above.
- **CLAUDE.md:** the same host-copy line.
- **Public API changes:**
  - **Additions:**
    - gridcl-device: `ConnKey` (moved), the variants, and the three helpers;
    - gridcl-refcell: three functions;
    - gridcl-topology: `CellBlock::new` and `Topology`;
    - gridcl-geometry: `PointSet`, `FacetPointSet`, `FacetQuadRule`, `EVector`, `Geometry`, the `QuadField` accessors, `DeviceGrid::{evector, origin}` and the block functions;
    - gridcl: `MeshBuilder` and `EntityMap`.
  - **Relocations:** `DeviceGrid`, `CoordinateElement` and `GeometryBlock` to gridcl-geometry, and `ConnKey` to gridcl-device, all re-exported where the design lists them.
  - **No existing signature changes.**
- **Acceptance tests affected:**
  - M3's last bullet is clarified for tile width 1 and for devices with fewer than 16 lanes, without weakening it: every size the device offers is still tested at tile width 16.
  - "`QuadField` identical with 1 and 7 chunks" uses `testing::with_properties` through `device_properties`.
  - No other acceptance test changes.
- **Later milestones affected:**
  - **M3s:** benchmarks TopologyKeys, PermutationInfo, EVector and GeomFactors in the forms fixed here, and adds the cell-fastest connectivity copy behind the policy.
  - **M4:** reads into `MeshBuilder`'s numbering rules, adds `TagRegistry`, and matches tag keys against entity blocks of dimension < tdim.
  - **M6:** extends the builder's numbering to the parallel pipeline, adds ownership to entity creation, and keeps the emission layout.
  - **M9:**
    - implements `boundary`, replacing `NotYetImplemented`;
    - adds `facet_submesh` and `orient`;
    - fills `EntityMap`;
    - tests normals on manifolds.
  - **The first milestone that needs factor caching** decides its key.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
