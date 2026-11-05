# ADR-0039: M9 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-11-05
- **Raised in:** M9 planning (docs/plans/M9.md, "Planned public API and rules" and questions 1–20)
- **Design sections affected:**
  - "Architecture": Listing B4 (`boundary`, `facet_submesh`, new `edge_submesh`, `cell_submesh`, `pull_back`); Listing B4b (pull-back items next to the fields); Listing B7 (no change of signature; the accessor rule is applied to submeshes);
  - "Device model and backends" / "Device containers": the `GridError` variant history;
  - "Reference cells and conventions": the origin-id bullet (cell origin ids of submeshes);
  - "Topology construction on the device" / "Ownership of vertices and entities": the `VertexSharers` bullet (`set_vertex_sharers` becomes fallible);
  - "Geometry": "Coordinate element and storage" (origin of a submesh), "Tabulation and quadrature" (device evaluation), "Manifold orientation", "Pull-back";
  - "Tags, regions and submeshes": the Submeshes bullet;
  - "Workspace layout": the crate table (gridcl-partition, gridcl-io, gridcl-bench), "Dependency rules" (the FFI note);
  - "Handing this design to Claude Code": the working-agreement copy list (also CLAUDE.md);
  - "Implementation roadmap": M9 scope and how its acceptance is read.

## Context

M9's roadmap section:

> **Scope:** boundary, facet and edge submeshes with `EntityMap` (`boundary()` replaces M3's `NotYetImplemented`); distributed manifold orientation (`orient`); the manifold tests of normals, pseudo-inverse and facet normals, whose kernels exist from M3 (ADR-0012); pull-back, generic over the vector size.

Acceptance: outward normals on the boundary of a centred ball; ∮ x·n = D_GEO · V to 10^−10; `orient(Outward)` recovers a randomly flipped sphere on 1–4 ranks; `NonManifold`, `InteriorFacet`, `NeedsReferencePoint` and `NonOrientable`; the pull-back recovers reference points to 10^−12 for p = 1–3 and "gives the same reference points at vector size 1 and at the widest size the device offers, to 10^−12".

The design gives the signatures of `boundary`, `facet_submesh`, `normals` and `orient` (Listing B4) and describes the rest in prose: submeshes "come with an `EntityMap`: child → parent indices on the device, parent → child as a sparse map. Ownership and ghosting are inherited from the parent" ("Tags, regions and submeshes"); orientation by parity label propagation in which "(label, parity) of ghost cells is exchanged" between rounds ("Manifold orientation"); a "Newton kernel … generic over the vector size: lanes that have converged keep their value through arithmetic masks" ("Pull-back"). Earlier ADRs leave M9 these items: `boundary` returns `NotYetImplemented`, `EntityMap` is a placeholder, and `facet_submesh` and `orient` are declared in M9 (ADR-0012 item 10); "M9: submeshes inherit ghosts through cloned `IndexMap`s" (ADR-0026, "Later milestones"); the pull-back "stays strictly bitwise" across vector sizes (ADR-0013); PullBack plans as its own kind (ADR-0015). Review follow-ups wait for "the next ADR": M7 minors 4–6, M8's `ndarray` entry, M3s B11, and M6's fallible `set_vertex_sharers` "when `Topology`'s API is next revised".

Planning read these against the code at main dcd6f31:

- **Stubs.** `Mesh::boundary` returns `NotYetImplemented("boundary: M9")` (`gridcl/src/mesh.rs:650-678`); `EntityMap` is `struct EntityMap { _private: () }`; `facet_submesh`, `orient`, `OrientationRule`, any pull-back code and the four error variants do not exist (`GridError`, `gridcl-device/src/error.rs:61-152`).
- **Assembly.** `Topology::new(client, cell_blocks, vertex_index_map, vertex_origin_ids: &[i64])` takes host origin ids; `set_vertex_sharers(VertexSharers)` is infallible and the stored sharers have no accessor; no API writes `CellBlock::flags` after creation (upload writes zeros, `gridcl/src/pipeline/upload.rs:27,93`). `Geometry::new` takes device nodes.
- **Facets in parallel.** No API counts the cells of a facet globally. Under `GhostMode::None` a process-boundary facet has one local cell on each of two ranks; with ghost cells the local rows over-count.
- **IndexMap.** No API builds an `IndexMap` from a subset of another; `IndexMap::with_ghosts` takes host lists. Node origin ids are not stored on a mesh (ADR-0035, M8 finding 7).
- **Bases.** Coordinate-element bases are tabulated on the host at fixed point sets (`PointSet::tabulate`, Listing B4b); gridcl-refcell's orthonormal polynomials are `pub(crate)` and `LagrangeElement`'s coefficients are private. The pull-back needs X(ξ) and J(ξ) at points that change every iteration.
- **Vectors.** `KernelKind::PullBack` exists at vector size 1 in every device class; the factor kernel selects per lane with `select_many` (`gridcl-geometry/src/layout.rs:380-394`).
- **Copies.** The CLAUDE.md copy list (ADR-0028) covers none of M9's reads.

## Problem

Each item matches the question of the same number in docs/plans/M9.md.

1. **`EntityMap`** has no listing.
2. **`OrientationRule`** has no listing; `AwayFrom(p)` and `Towards(p)` need a point, but `orient` is not generic over GD.
3. **Edge submeshes** ("tagged edges of 3D meshes") have no listing, and "3D" can mean D_TOP = 3 or D_GEO = 3.
4. **Cell submeshes** ("tagged cells (same D_TOP)") are in "Tags, regions and submeshes" but in no milestone's scope.
5. **Prerequisites.** `boundary(&self)` must find facets, but "accessors create no topology: entities and connectivity are built only by the `&mut self` methods" (Listing B7).
6. **Ownership.** "Inherited from the parent" cannot hold for vertices and nodes: a parent vertex owner may hold no child cell at that vertex (its owned cell is interior and the boundary facet's cell is on another rank). A cloned `IndexMap` describes every parent entity, so it serves no child that keeps a subset. Nodes have no origin ids to hash.
7. **Cell origin ids** of children are undefined: facets and edges have none, and cells "carry origin ids … used for output ordering and tests".
8. **Geometry and flags.** The design does not say in which vertex order a child cell is stored, how high-order child nodes are found, how "oriented so that its normal agrees with the outward facet normal of its single parent cell" is stored, what "oriented outward" means for the boundary of a surface in 3D, or what happens when one facet type has parents of different degrees ("every block of one cell type has one degree").
9. **Origin.** "The origin is the centre of the global bounding box, agreed … at mesh creation": a submesh's bounding box differs from its parent's.
10. **Blocks and tags.** Child blocks, empty children and tags on submeshes are unspecified.
11. **Boundary facets in parallel** and the errors of `boundary` and `facet_submesh` (finding "Facets in parallel").
12. **`orient`.** Under `GhostMode::None` there are no ghost cells to exchange; the design leaves open existing flags, the kernel layout under the six-buffer rule, how component sums are combined across ranks, ties, and error agreement.
13. **Pull-back API.** None exists; the candidate table asks for "points tiled like cells"; CLAUDE.md forbids branches on lane values, so a loop that stops when its lanes have converged needs a rule.
14. **Invariance tolerance.** The acceptance reads "to 10^−12"; ADR-0013 says the pull-back "stays strictly bitwise".
15. **Device bases.** Nothing evaluates a coordinate-element basis on the device (finding "Bases"); pyramid bases have a pole at ζ = 1.
16. **`set_vertex_sharers`** (M6 follow-up) should check its input, which changes its signature in "Ownership of vertices and entities".
17. **Hooks.** gridcl and gridcl-geometry need `Topology` internals that no stable API offers.
18. **Copies.** The new host↔device copies are not on the copy list.
19. **Manifold tests.** The roadmap names "the manifold tests of normals, pseudo-inverse and facet normals" without saying what they check; `facet_normals` exists only in codimension 0.
20. **Carried corrections.** M7 review minors 4–6, M8's `ndarray` entry and M3s B11 wait for an ADR; M3 follow-up m12 asks for an ADR on sealing `DeviceGrid` "before … the next `DeviceGrid` implementation"; the four new error variants have no fields in the design.

## Options

1. **Keep the design as written.** Implement only what the listings fix and decide the rest during M9. Items 6, 12 and 15 cannot be implemented as written (no inherited owner exists for some vertices; no ghost cells under `GhostMode::None`; no device basis), items 5 and 14 contradict other accepted text, and the remaining items would be settled ad hoc and recorded only in the plan, although M10 and downstream FEM/BEM crates build on them. Not chosen.
2. **Fix the open items now, item by item** (below). Adds the public items of items 1–4, 13 and 15–16, five hidden hooks (item 17), copy-list entries (item 18) and design text for every item; the M9 scope gains cell submeshes.
3. **Option 2 without the scope additions:** no `cell_submesh` (item 4), and a host-staged pull-back basis in place of item 15 (tabulate on the host at each iterate). The first leaves a designed feature without a milestone; the second copies every iterate to the host, against the copy rule, and does not vectorize. Not proposed; the maintainer chose items 4 and 15 during planning.

## Proposed decision

Option 2, item by item. Items 3, 4, 5, 10 (tags) and 15 record the maintainer's answers during planning (2026-11-05).

1. **`EntityMap`** (gridcl):
   ```rust
   pub struct EntityMap { /* parent_dim, child→parent, sparse parent→child, vertex and node maps */ }
   impl EntityMap {
       pub fn parent_dim(&self) -> usize;
       pub fn child_to_parent(&self) -> &DeviceBuffer<u32>;     // flat parent entity per local child cell, flat child order
       pub fn parent_to_child(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<u32>); // parent flat indices ascending, child cell of each
       pub fn vertex_to_parent(&self) -> &DeviceBuffer<u32>;    // parent local vertex per child local vertex
       pub fn node_to_parent(&self) -> &DeviceBuffer<u32>;      // parent geometry node per child geometry node
       pub fn try_clone(&self) -> Result<Self>;
   }
   ```
   Owned and ghost child cells, vertices and nodes are covered. For facet children the parent cell and local facet stay crate-private.
2. **`OrientationRule`** (gridcl-geometry, re-exported): `#[derive(Clone, Copy, Debug, PartialEq)] pub enum OrientationRule { Outward, AwayFrom([f64; 3]), Towards([f64; 3]) }`. Points are absolute coordinates; the first `gdim` entries are used.
3. **`edge_submesh`** exists for volume meshes only: `impl<F: Real, C: Comm> Mesh<F, D3, D3, C> { pub fn edge_submesh(&self, edges: &MeshTags) -> Result<(CurveMesh<F, C>, EntityMap)>; }`. Interior and boundary edges are allowed.
4. **`cell_submesh`** joins M9: `pub fn cell_submesh(&self, cells: &MeshTags) -> Result<(Self, EntityMap)>` on every `Mesh` with `TD: EmbeddedIn<GD>`. Child cells keep the parent's origin ids.
5. **Prerequisites.** Submesh methods take `&self` and create no topology. `boundary` and `facet_submesh` return `EntitiesNotCreated(tdim − 1)` until `create_entities(tdim − 1)` has run. `edge_submesh` needs `create_entities(1)` and `create_connectivity(1, 3)` (`EntitiesNotCreated`, `ConnectivityNotCreated`). `orient` takes `&mut self` and creates the (tdim − 1) entities and the (tdim − 1, tdim) connectivity it needs.
6. **Ownership and numbering of a child.**
   - Cells inherit the parent entity's owner and ghost status, including ghost child cells where the parent holds the entity as a ghost. There is no new ghost layer.
   - Child `VertexSharers` are the parent's rows restricted to the ranks holding the vertex in a child cell, with owned-child-cell flags, found by one exchange over the parent's sharing neighbourhood for shared vertices only.
   - Vertex owners follow the design's rule: `owner_hash(&[origin id]) mod n` among the owned-child-cell sharers.
   - A corner node takes its vertex's owner. Any other node takes the same rule with the key `[parent global node index]`, its candidates collected by the parent node owner over the node `IndexMap`'s neighbourhoods.
   - Owned entities keep the parent's local order; ghosts follow by (owner rank, child global index). Global numbers come by exclusive scan, and a ghost's child global index travels from the child owner to the parent owner (reverse) and on to every holder (forward). `IndexMap::with_ghosts` is used on more than one rank, `IndexMap::new` on one.
   - "Cloned `IndexMap`s" (ADR-0026) applies only where a child keeps every parent entity.
7. **Child cell origin ids** are partition-independent:
   - a facet child gets 8 · (parent cell origin id) + local facet index;
   - an edge child gets the minimum, over the cells containing the edge, of 12 · (cell origin id) + local edge index. Ranks reduce it by `Min` on the interleaved hi word and `forward`, then by `Min` on the lo word with non-matching hi words set to `u32::MAX` and `forward`;
   - a cell child keeps the parent's id.
   - Overflow of i64 gives `TooLarge`, agreed across ranks. Vertex origin ids are the parent's, as the design already says.
8. **Child vertex order, geometry and flags.**
   - Facet children take the parent cell's local facet vertex order and the facet's geometry nodes in the child's reference order. The closure node map comes from matching the child's `lagrange_points`, mapped into the parent cell, with the parent's. The child geometry is thus the restriction of the parent's, with the same degree and family.
   - Edge children take the canonical orientation (A.6). Interior nodes come from the first local cell in the edge's (1, 3) row, reversed where that cell's local edge runs the other way.
   - A codimension-1 child (of a volume or planar mesh) sets flag bit 0 where its geometric normal at the facet midpoint has a negative dot product with the parent cell's outward facet normal there.
   - Children of codimension 2 (the boundary of a surface in 3D, edge submeshes) have no normal, and their flags are 0.
   - A child block whose parent cells differ in degree or node family gives `UnsupportedDegree` or `UnsupportedNodeFamily` for the facet type, agreed across ranks.
9. **Origin.** A submesh keeps its parent's origin, so its coordinates are bitwise the parent's in every precision.
10. **Blocks and tags.**
    - A child has one cell block per parent entity block (or cell block) with a non-zero global child count, in the parent's order and the same on every rank. A rank may hold empty blocks, and a child with no cells is a valid empty mesh.
    - The child's `TagRegistry` holds the parent's physical and elementary tags of the dimension its cells come from, restricted and renumbered as child cell tags, with those name-table rows. Lower-dimensional tags are not carried.
11. **`boundary` and `facet_submesh`.**
    - A facet is on the boundary if it has exactly one owned cell globally: owned cells are counted per local facet, combined by `reverse(Add)` and returned by `forward` over the facet block's `IndexMap`. This works in every `GhostMode`.
    - Errors, all agreed across ranks:
      - more than two owned cells on any facet gives `NonManifold`;
      - a tagged facet with two gives `InteriorFacet`;
      - tags of another dimension give `DimensionMismatch`.
12. **`orient`.**
    - Existing flags are ignored and overwritten; ghost cells get their owner's flags.
    - Any facet with more than two owned cells globally gives `NonManifold`.
    - Parity per interior facet follows "Manifold orientation", step 2.
    - A neighbour table is built once: per cell and local facet, one packed record of the neighbour flat cell (or none) and the parity. The state is one record per flat cell (label hi, label lo, parity, pointer), double-buffered. A round kernel binds at most four buffers.
    - Propagation is min-label adoption with parity and pointer jumping, Jacobi-style, so it needs no atomics and is deterministic.
    - Between rounds, the state of the cells on both sides of every process-boundary facet is exchanged over the facet `IndexMap`: `reverse(Max)` of the single non-owner contribution, encoded + 1 with others 0, then `forward` of both sides. This works in every `GhostMode`, including `None`. Ghost cells take their owner's state by `forward`. A round ends with `comm.max(changed)`.
    - After convergence, any interior facet whose cells violate s_a · s_b = (−1)^parity gives `NonOrientable`. Process-boundary facets are checked through the same exchange.
    - Per component, owned cells contribute (1/D_GEO) ∮ x·n, or ∫ (x − p)·n, with Gauss degree 3p (the largest over blocks). Sums are formed per label on the device, finished in f64 on the host, and combined at rank `owner_hash(&[label]) mod P` in rank order.
    - Under `Outward`, an open component (one with a boundary facet) gives `NeedsReferencePoint`. A sum of exactly 0 flips nothing.
    - Errors are agreed across ranks. The kernels plan as `KernelKind::Scalar`.
13. **Pull-back iteration** (gridcl-geometry, re-exported, and gridcl):
    ```rust
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct PullBackOptions { pub tolerance: Option<f64>, pub max_iterations: u32 } // Default: None, 16
    pub struct PullBack<F: Real, TD: TopoDim> {
        pub reference: BlockFields<F, VecShape<TD>>,   // ξ per cell and point
        pub last_step: BlockFields<F, ScalarShape>,    // |Δξ|∞ of the last iteration
    }
    pub fn pull_back_block<F, TD, GD, G>(g: &G, b: usize, x: &QuadField<F, VecShape<GD>>, opts: &PullBackOptions)
        -> Result<(QuadField<F, VecShape<TD>>, QuadField<F, ScalarShape>)>
    where F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim, G: DeviceGrid<F = F, TD = TD, GD = GD>;
    impl Mesh<F, TD, GD, C> /* TD: EmbeddedIn<GD> */ {
        pub fn pull_back(&self, x: &BlockFields<F, VecShape<GD>>, opts: &PullBackOptions) -> Result<PullBack<F, TD>>;
    }
    ```
    - The input holds absolute physical points per cell and point, as `physical_points` returns them. Lanes are consecutive cells of a tile reading the E-vector ("points tiled like cells"), planned as `KernelKind::PullBack` with `max_vector = tile`.
    - Iteration starts at the reference cell's vertex centroid, with Δ = K(ξ) (x − X(ξ)) and K the pseudo-inverse. This is Newton in codimension 0 and Gauss–Newton otherwise.
    - A lane freezes once |Δ|∞ ≤ tolerance. Frozen lanes keep ξ through a lane select, so another lane's NaN or Inf never reaches them.
    - The loop always runs `max_iterations`. There is no early exit, so "no branches on lane values" holds as written.
    - Non-convergence shows in `last_step`, not as an error.
    - `tolerance: None` means 10^−13 in f64 and 10^−5 in f32.
14. **Invariance tolerance.** The pull-back is bitwise identical across vector sizes on the CPU runtime (ADR-0013). The acceptance's 10^−12 is the GPU tolerance in f64, with 10^−5 in f32.
15. **Device basis evaluation.**
    - The device evaluates the orthonormal basis of "Tabulation" (Legendre, Dubiner, orthonormal rationomials) and its first derivatives by the host's recurrences, in `#[cube]` helpers specialised at comptime by cell type and degree.
    - It combines them with the new `LagrangeElement::coefficients(&self) -> &[f64]` (gridcl-refcell; C row-major N × N, ψ_j = Σ_k C[k·N + j] φ_k). The coefficients are narrowed to F and uploaded once per geometry block on first use, then cached like the E-vector.
    - Pyramid ζ is clamped to at most 1 − 2^−10 by an arithmetic `min`.
    - The pull-back kernel binds five buffers.
16. **`Topology::set_vertex_sharers(&mut self, s: VertexSharers) -> Result<()>`** checks lengths, row order and the presence of the own rank, returning `LengthMismatch` or `IndexOutOfRange`.
17. **Hidden hooks** (Dependency rule 7):
    - `Topology::from_device_ids(client, cell_blocks, vertex_index_map, vertex_origin_ids: DeviceBuffer<u32>)` with interleaved ids;
    - `Topology::set_cell_flags(b, DeviceBuffer<u32>) -> Result<()>`;
    - `Topology::vertex_sharers() -> Option<&VertexSharers>`;
    - `gridcl_topology::facet_cell_counts(comm, topology) -> Result<FacetCells>`, giving global owned-cell counts and the parent cell and local facet of boundary facets;
    - `gridcl_geometry::orientation_flags(g, rule) -> Result<Vec<DeviceBuffer<u32>>>`.
18. **Copy list.** These copies are added:
    - submesh creation: the per-block count reads of compaction; the agreed `TooLarge`, `NonManifold` and `InteriorFacet` flags; the child ghost lists for `IndexMap::with_ghosts`; the child sharer rows of shared vertices and the node-owner candidates of shared nodes; the upload of the closure node maps; the child's vertex shared flags;
    - `orient`: the per-round change flag, the agreed error flags, the per-component (label, sum, open) records and the flip decisions back;
    - the pull-back: the upload of the expansion coefficients on first use.
19. **Manifold tests:**
    - normals have unit length, are orthogonal to J's columns, are flipped by bit 0, and point outward after `boundary()` and `orient`;
    - the pseudo-inverse satisfies K J = I, J K J = J and J K n = 0 on curved boundary submeshes;
    - the parent's `facet_normals` at boundary facets equal the boundary submesh's `normals` at the same points.
20. **Carried corrections and records.**
    - The crate table lists gridcl-partition's build dependency `scotch-sys-bindgen` (bindgen 0.65.1, feature `scotch`), gridcl-io's optional `ndarray` (with `hdf5`), and gridcl-bench's `gridcl-refcell` and dev-dependency `proptest`.
    - ADR-0029 item 5 is corrected: parmetis-sys links `parmetis` only, and the shared libparmetis resolves METIS itself.
    - The FFI rule allows a -sys crate's own tests and examples to call its declarations.
    - The `DeviceGrid` sealing trigger (M3 m12) has not fired, because submeshes are `Mesh`.
    - The four new `GridError` variants are unit variants, agreed across ranks (ADR-0026 item 15).

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B4:**
    - The `boundary` doc becomes: "All boundary facets as a mesh one dimension lower; facet entities must exist (`EntitiesNotCreated`). Codimension-1 children are oriented outward through their flags; codimension-2 children have flags 0 (ADR-0039)."
    - The fourth `impl` block adds `cell_submesh` and `pull_back`.
    - A new block `impl<F: Real, C: Comm> Mesh<F, D3, D3, C>` holds `edge_submesh`.
  - **Listing B4b:** add `PullBackOptions`, `PullBack` and `pull_back_block` after `BlockFields`.
  - **"Device containers"**, the variant history, adds: "M9 adds `NonManifold`, `InteriorFacet`, `NeedsReferencePoint` and `NonOrientable`, unit variants agreed across ranks (ADR-0039)."
  - **"Reference cells and conventions"**, origin ids, adds: "A submesh's cells take 8 · (parent cell id) + local facet index (facet children), the smallest 12 · (cell id) + local edge index over the cells containing the edge (edge children), or the parent's id (cell children) (ADR-0039)."
  - **"Ownership of vertices and entities"**: `set_vertex_sharers(&mut self, s: VertexSharers) -> Result<()>` checks its input (ADR-0039).
  - **"Coordinate element and storage"**, origin, adds: "A submesh keeps its parent's origin (ADR-0039)."
  - **"Tabulation and quadrature"** adds: "The pull-back evaluates the same orthonormal bases on the device by recurrence, with `LagrangeElement::coefficients` uploaded once per geometry block; pyramid ζ is clamped below 1 (ADR-0039)."
  - **"Manifold orientation"**:
    - Step 3: "Between rounds, (label, parity) of ghost cells is exchanged" becomes "Between rounds, the (label, parity) of the cells on both sides of every process-boundary facet is exchanged over the facet `IndexMap`, and ghost cells take their owner's; this works in every `GhostMode` (ADR-0039)."
    - Step 4 adds the per-label rendezvous and the zero-sum rule.
    - The closing paragraph adds the flag rule of item 8 and the codimension-2 note.
  - **"Pull-back"** adds item 13's iteration (vertex-centroid start, pseudo-inverse step, lane select, a fixed `max_iterations`, `last_step`) and item 14.
  - **"Tags, regions and submeshes"**, Submeshes:
    - "Ownership and ghosting are inherited from the parent" becomes "Cells inherit ownership and ghosting from the parent; vertices and nodes are re-owned among the ranks holding them in owned child cells (ADR-0039)".
    - Add `edge_submesh` (volume meshes), `cell_submesh`, the prerequisites, the origin, the child blocks and the tag rule.
  - **"Workspace layout"**: the gridcl-partition, gridcl-io and gridcl-bench rows as in item 20; the FFI note of item 20 in the working agreement.
  - **"Handing this design to Claude Code"**, the copy list (and CLAUDE.md): the entries of item 18.
  - **Roadmap M9:**
    - The scope reads "boundary, facet, edge and cell submeshes with `EntityMap` … (ADR-0039)".
    - The acceptance adds "read as in ADR-0039 item 14".
- **Public API changes:**
  - gridcl: `EntityMap` (methods), `Mesh::{facet_submesh, edge_submesh, cell_submesh, orient, pull_back}`, and `boundary` implemented.
  - gridcl-geometry: `OrientationRule`, `PullBackOptions`, `PullBack`, `pull_back_block`.
  - gridcl-refcell: `LagrangeElement::coefficients`.
  - gridcl-device: four `GridError` variants.
  - gridcl-topology: `set_vertex_sharers` returns `Result<()>`, the one existing signature that changes; its only caller is gridcl's upload stage.
  - Hidden: the five hooks of item 17.
- **Acceptance tests affected:**
  - The M9 bullets, with the vector-size bullet read as in item 14; their tests are those of docs/plans/M9.md.
  - `gridcl/tests/builder.rs::boundary_is_not_yet_implemented` and the `boundary` doctest are replaced. The trybuild `.stderr` files change only if rustc's text changes.
- **Later milestones affected:**
  - M10: the PullBack row of the vectorization re-run (ADR-0015, ADR-0018) measures the fixed-iteration kernel; an early exit or a convergence count can be revisited with that data. Submeshes are written through the existing writers (ADR-0035).

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
