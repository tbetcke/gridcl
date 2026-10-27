# ADR-0024: M5 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-27
- **Raised in:** M5 planning (docs/plans/M5.md, "Planned public API and rules", questions 2–13)
- **Design sections affected:**
  - "API walkthrough", Listing B13 (`VtkWriter`) and the sentence after it;
  - "Reference cells and conventions" (the node-families bullet);
  - "Import and export / VTK" (VTU, high-order pyramids, PVTU, VTU reading, test oracles);
  - "Testing and validation" (the I/O bullets);
  - "Workspace layout": the crate table (gridcl-io, gridcl, gridcl-geometry);
  - "Implementation roadmap": M5 (scope and the second acceptance bullet);
  - Appendix A.8 (the pyramid row's writer column).

## Context

M5 adds VTK output. The roadmap gives its scope and acceptance:

> **Scope:** VTU (version 2.2) with linear and Lagrange cells, cell and point data, the PVTU skeleton and the pyramid fallback.
>
> - VTK's Python reader (pinned, ≥ 9.2) reads every fixture;
> - the integrated volume or area of P2 and P3 meshes (`vtkIntegrateAttributes`) equals gridcl's `measure` to 10^−10 (f64);
> - meshio reads identical raw arrays;
> - a P2 pyramid mesh is written as linear cells with one warning.

The only API is Listing B13:

> ```rust
> VtkWriter::vtu("out/scalp").lagrange(true).write(&scalp)?;          // generic over TD, GD
> VtkWriter::vtu("out/scalp2").write_any(AnyMeshRef::Surface(&scalp))?; // erased path, no clone
> ```

"Import and export / VTK" adds appended raw or base64 binary with optional zlib, pyramids "written as vertices only (VTK type 14) with a warning, or as a subdivided output mesh", the PVTU with one piece per rank, and VTU reading of "lower priority … used for round-trip tests". "Reference cells and conventions" says writers always convert GLL to equispaced, by "an interpolation applied on the device", and ADR-0022 (item 9) moved that conversion to M5.

Planning read these sections against the code after M4. The relevant facts:

- **Crate graph.** `Mesh` and `AnyMesh` live in gridcl, which depends on gridcl-io. ADR-0022 resolved the same cycle for Gmsh by putting the host model in `gridcl_io::gmsh` and the mesh-level entry points in `gridcl::gmsh`.
- **No node-family conversion exists.**
  - `GmshWriter` returns `UnsupportedNodeFamily` for GLL geometry (`gridcl/src/gmsh.rs:333`).
  - `physical_points_block` (`gridcl-geometry/src/points_jac.rs:31`, the GeomFactors kernel) evaluates geometry at a `PointSet`.
  - `PointSet::new` is crate-private to gridcl-geometry.
- **No cell or point data concept** exists in the code or in any listing.
- **No compression or base64 code** is in the workspace. `flate2` is not in Cargo.lock.
- **Python oracles.**
  - docs/dev-environment.md records vtk 9.7.1 and meshio 5.3.5 in a `.venv`, but nothing pins them. `scripts/check-local-tools.sh` asks only for vtk ≥ 9.2 and any meshio.
  - No checkout currently has a `.venv`.
  - CI's cpu and vulkan jobs (`.github/workflows/ci.yml`) install no Python packages.
  - The remote CUDA targets have no VTK.
- **No VTK-written fixture exists**, though ADR-0008 asks for VTK-written prisms at p = 3, 5 and 6 in M5.

## Problem

1. **`VtkWriter` cannot live in gridcl-io.** It must copy a `Mesh` to the host and accept `AnyMeshRef`; gridcl-io cannot see either type.
2. **Listing B13 leaves the writer's options open.** It does not say:
   - whether `lagrange` defaults to true or false, or what false means;
   - how appended or base64 data and zlib are selected;
   - whether `vtu("out/scalp")` appends `.vtu`;
   - how a PVTU and its pieces are named.
3. **"Cell and point data" has no API.** No listing says:
   - which buffers can be attached, or of which types;
   - how point data is indexed, given that points are geometry nodes and not vertices;
   - how cell data is indexed over several cell blocks;
   - what happens with a wrong length or a repeated name.
4. **Points of linear output and point precision.** With `lagrange(false)`, or for a P2 pyramid written as type 14, only corner nodes are referenced. The design does not say whether the other nodes are dropped, which would remap point data, or kept. It also does not say whether f32 meshes write Float32 or Float64 points.
5. **The pyramid fallback offers two outputs** ("vertices only … or a subdivided output mesh"), and does not say whether the warning is per cell or per file.
6. **The integration bullet cannot hold on curved meshes.** VTK 9.7's `vtkIntegrateAttributes` integrates Lagrange cells piecewise linearly over sub-cells through their nodes. Measured with VTK 9.7.0 (Homebrew Python), on one cell each:

   | Cell (VTK type) | Strategy | VTK result | Exact |
   | --- | --- | --- | --- |
   | P2 triangle (69), one edge midpoint bulged from (0.5, 0.5) to (0.6, 0.6) | Linear | 0.600000 | 0.633333 |
   | same | Gaussian (`vtkIntegrationGaussianStrategy`) | 0.600000 (warns "Cell type (69) is not yet supported … falls back to Linear Strategy") | 0.633333 |
   | P2 quadrilateral (70), one edge bulged by 0.2 | both | 1.100000 | 1.133333 |
   | P2 curve (68) | both | 0 (warns "Odd number of points(3) encountered - skipping 1D Cell") | 2.957886 |
   | affine P2 triangle (69) | both | 0.5 | 0.5 |

   The Gaussian strategy also falls back for types 70, 71, 72 and 73. So for every curved P2 or P3 mesh, VTK's value differs from `measure` by the gap between the cells and their piecewise-linear interpolants, typically 10^−3 relative, not 10^−10. `vtkCell::EvaluateLocation` on a Lagrange cell is exact: on the curved P2 triangle at parametric (1/4, 1/4) it returns (0.275, 0.275), the P2 interpolant.
7. **GLL conversion has no API and no rule for shared nodes.** Each cell's interpolation gives a value for every node it contains. A node shared by several cells gets values that differ in the last bits (different summation order and edge orientation). "Bitwise deterministic" therefore needs a rule for which value a shared node keeps.
8. **"The PVTU skeleton" is undefined.** The design does not say how much of PVTU M5 delivers before M8's parallel I/O and `vtkGhostType`.
9. **VTU reading.** The design gives it "lower priority", but M5 needs a reader:
   - the ADR-0008 check against VTK-written fixtures needs to parse VTK's files in Rust;
   - Rust-side round-trip tests need it.

   The design does not say whether the reader builds meshes.
10. **zlib and base64** need code, or a crate the crate table does not list.
11. **Python oracles are unpinned and not always present.** "Pinned, ≥ 9.2" names no version. No rule says what a test does when the interpreter or the packages are missing: on CI, on remote CUDA targets, or on a fresh checkout. A silent skip would hide failures.
12. **"Every fixture" is ambiguous.** It could mean committed files or files written by the test.

## Options

1. **Keep the design as written.**
   - A `VtkWriter` in gridcl-io cannot compile (problem 1).
   - The integration test fails on every curved P2 or P3 mesh (problem 6).
   - The rest leaves T0's stubs without signatures to target.

   Not chosen.
2. **Fix the items below; keep the file level in gridcl-io and the mesh level in gridcl, as for Gmsh (ADR-0022); amend the integration bullet to affine meshes and add a curved check through `EvaluateLocation`** (proposed).
3. **Keep curved meshes in the integration bullet, but compare with the piecewise-linear interpolant.**
   - The test would compute the measure of the sub-cell subdivision VTK uses, which re-implements VTK's `Triangulate` for every Lagrange cell type and degree.
   - It tests VTK's subdivision rather than gridcl's output.

   Not chosen.
4. **Pyramids as a subdivided output mesh** (problem 5).
   - Subdividing a degree-p pyramid into linear tetrahedra and pyramids needs new subdivision tables and a second point set.
   - The maintainer chose vertices only for M5 (2026-10-27).
5. **A mesh-level `vtk::read`** (problem 9).
   - It would build `AnyMesh` from a VTU through `MeshBuilder`, with the inverse cell-type table, degree inference and the pyramid ambiguity (a linear type 14 cannot be told apart from a degraded P2 pyramid).
   - Nothing in M5 or M8 needs it.

   Not chosen.
6. **Skip Python oracle tests silently when the interpreter is missing** (problem 11). This is simple, but a missing `.venv` would turn every oracle test green. Not chosen.

## Proposed decision

Option 2. Each item answers the problem of the same number. The maintainer chose items 5 and 6 during M5 planning (2026-10-27).

1. **Crate split.**
   - `gridcl_io::vtk` holds the host side:
     - the cell-type table of A.8: `vtk_cell_type(cell: CellType, degree: u32, lagrange: bool) -> u8`, where p ≥ 2 pyramids give 14;
     - the host mesh description `VtkMesh { gdim: usize, nodes: Vec<f64> /* [node][gdim], origin added */, blocks: Vec<VtkCellBlock> }`, with `VtkCellBlock { cell: CellType, degree: u32, nodes: Vec<u32> /* A.5 order */ }`;
     - the file model:
       - `VtuFile { points: DataValues /* F32 or F64, [point][3] */, connectivity: Vec<i64>, offsets: Vec<i64>, types: Vec<u8>, point_data: Vec<DataArray>, cell_data: Vec<DataArray> }`;
       - `DataArray { name: String, components: usize, values: DataValues }`;
       - `enum DataValues { F32(Vec<f32>), F64(Vec<f64>), I32(Vec<i32>), U32(Vec<u32>), I64(Vec<i64>), U8(Vec<u8>) }`;
       - all fields `pub`;
     - `to_vtu(mesh: &VtkMesh, lagrange: bool, f32_points: bool) -> Result<(VtuFile, usize)>`, where the `usize` is the number of pyramids written as linear cells. Connectivity uses the inverse of `vtk_to_internal`;
     - `enum Binary { Appended, Base64 }` and `VtuFile::write_to(&self, w: &mut dyn Write, binary: Binary, zlib: bool) -> Result<()>`, which writes `type="UnstructuredGrid" version="2.2" byte_order="LittleEndian" header_type="UInt64"`;
     - `write_pvtu(w: &mut dyn Write, pieces: &[String], like: &VtuFile) -> Result<()>`;
     - `parse_vtu(bytes: &[u8]) -> Result<VtuFile>` (item 9).
   - `gridcl::vtk` holds `VtkWriter` (items 2–4, 8), which downloads a `Mesh`.
   - The prelude exports `VtkWriter` (Listing B13 uses it unqualified) and the module `vtk`.
2. **`VtkWriter<'a>`.**
   - `vtu(path: impl AsRef<Path>) -> Self` appends `.vtu` if the path has no extension.
   - `pvtu(path)` writes `<stem>.pvtu` and the pieces `<stem>_<rank>.vtu` beside it.
   - `.lagrange(bool)` defaults to true, following A.8's writer column ("3 for p = 1, else 68"). False writes every cell as its linear type, from its corner nodes.
   - `.base64(bool)` defaults to false: appended raw data. True selects inline base64.
   - `.zlib(bool)` defaults to false. True compresses each data array with zlib, in VTK's block layout.
   - `.write(&Mesh<F, TD, GD, C>) -> Result<()>` and `.write_any(AnyMeshRef<'_, F, C>) -> Result<()>`.
   - `.write` with more than one rank gives `NotYetImplemented("parallel VTU write (M8)")`.
   - Errors: `Io` for file errors; `LengthMismatch` (item 3); device errors.
3. **Point and cell data.**
   - `.point_data<T: VtkScalar>(self, name: &str, data: &'a DeviceBuffer<T>, components: usize) -> Self` and `.cell_data<T: VtkScalar>(…)` likewise.
   - `VtkScalar` is a sealed trait, implemented for f32, f64, i32 and u32.
   - Indexing and lengths:
     - point data is indexed by geometry node, with length `num_nodes · components`;
     - cell data is indexed by flat cell index over the cell blocks in block order (as for `MeshTags`, ADR-0022), with length `num_cells · components`;
     - `components ≥ 1`; any other length gives `LengthMismatch`.
   - A second array with the same name in the same section replaces the first.
   - The buffers are copied to the host at write time, as I/O.
   - The writer adds no arrays of its own. M8 adds `vtkGhostType`.
4. **Points.**
   - Points are always all geometry nodes, in node order, after item 7's conversion, with the origin added back in f64.
   - Linear output and the pyramid fallback leave the non-vertex nodes unreferenced. VTK and meshio accept unreferenced points, and point data needs no remapping.
   - Points are Float64 for f64 meshes, and Float32 for f32 meshes (narrowed after the origin is added).
   - Connectivity and offsets are Int64, and types are UInt8.
5. **Pyramid fallback.**
   - With `lagrange(true)`, pyramids of degree ≥ 2 are written as type 14 from their five vertices. The writer calls `log::warn!` once per written file, naming the number of pyramids and their degree.
   - With `lagrange(false)` every cell is linear by request, and no warning is logged.
   - No subdivision in M5. The design keeps it as a possible later option.
6. **Integration acceptance (curved meshes).**
   - The second M5 bullet applies to P2 and P3 meshes whose cells are affine images of the reference cell:
     - an affine map (rotation, shear, scaling, translation) of the box meshes of every cell type and of the mixed box;
     - planar triangle and quadrilateral meshes in 2D, and in 3D on a tilted plane.

     On these, VTK's piecewise-linear subdivision is exact, and the 10^−10 tolerance stands.
   - A new bullet checks curved geometry: on curved P2 and P3 meshes (sphere, mixed surface, ball, perturbed tetrahedron and prism boxes), VTK's `EvaluateLocation` at the Gauss points of each cell equals gridcl's `physical_points` to 10^−10 (f64). VTK's parametric coordinates coincide with the reference cells of A.1 for every cell VTK writes as a Lagrange cell.
7. **GLL → equispaced.**
   - `gridcl_geometry::equispaced_nodes<G: DeviceGrid>(grid: &G) -> Result<DeviceBuffer<G::F>>` returns an L-vector with the numbering, length and origin of `grid.nodes()`. Every node of a GLL block holds the equispaced interpolant; equispaced blocks are copied unchanged.
   - Per GLL block, the existing physical-points path evaluates the geometry at `lagrange_points(cell, p, Equispaced)`. A crate-private `PointSet` is built from those points, so no new kernel family is added.
   - A node shared by several cells takes the value from its first occurrence in flat (block, cell, local node) order. This is found by a stable radix sort of node indices, `unique_by_key` (its allowed count read) and a gather. The result is bitwise deterministic and uses no atomics.
   - Both writers use it. `GmshWriter` no longer returns `UnsupportedNodeFamily`.
   - The interpolation matrix is a tabulation uploaded once per call, under CLAUDE.md's existing exemption for tabulations.
8. **PVTU skeleton.**
   - `VtkWriter::pvtu` writes the `.pvtu` header, with `PPoints`, `PPointData` and `PCellData` declared from the piece's arrays, and one `<Piece Source=…>` per rank.
   - In M5 it requires one rank; more give `NotYetImplemented("parallel PVTU write (M8)")`.
   - M8 lifts the limit and adds `vtkGhostType`.
9. **VTU reading.**
   - `gridcl_io::vtk::parse_vtu` reads `UnstructuredGrid` files with ASCII, inline base64 or appended (raw or base64) data, with or without zlib, and UInt32 or UInt64 headers.
   - It returns the file model only. There is no mesh-level `vtk::read` in M5.
   - Errors are `GridError::Parse { format: "VTU", offset, message }`.
   - Header counts and lengths use checked arithmetic, and no allocation is sized from a count before the data is there.
10. **Dependencies.**
    - gridcl-io depends on `flate2` with `default-features = false, features = ["rust_backend"]`, a workspace dependency, so no C library is needed.
    - base64 is a crate-private encoder and decoder of about 50 lines, tested against RFC 4648's vectors.
    - The pyramid warning is logged in gridcl, which already depends on `log`. gridcl-io gains no `log` dependency.
11. **Python oracles.**
    - The pins are `vtk==9.7.1` and `meshio==5.3.5` in `.venv`. `scripts/check-local-tools.sh` checks them, and the oracle script `scripts/oracles/vtk_oracle.py` refuses other versions.
    - Tests find the interpreter through `GRIDCL_PYTHON`, else `<repo>/.venv/bin/python`.
    - If it is missing or lacks the pins, the test fails with the install command, unless `GRIDCL_SKIP_PYTHON=1` is set; the test then prints that it skipped.
    - Where the oracles run:
      - CI's cpu and vulkan jobs install the pins;
      - remote CUDA runs set `GRIDCL_SKIP_PYTHON=1`, because the oracles check host-side file content, and Rust tests cover the device paths (download, item 7).
12. **Fixtures.**
    - "Every fixture" means:
      - every file of the M5 output corpus, written by the test into a temporary directory and not committed. The corpus covers:
        - every cell type at p = 1–4, with pyramids at p = 1 plus the fallback at p = 2 and 3;
        - all six (TD, GD) combinations, and the mixed box;
        - appended and base64 data, each with and without zlib;
        - point and cell data of every `VtkScalar` type with 1, 3 and 9 components;
        - the PVTU;
      - the committed VTK-written reference cells.
    - The VTK-written reference cells:
      - one cell per Lagrange type 68–73 and degree p = 1–6, prisms at 3, 5 and 6 included (ADR-0008);
      - points from VTK's own parametric coordinates of its nodes;
      - written in ASCII by `vtkXMLUnstructuredGridWriter` through `scripts/fixtures/vtk/generate.py`, which can regenerate them byte-identically.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B13:** unchanged. A sentence after it reads: "`VtkWriter` (gridcl) writes Lagrange cells by default (`.lagrange(false)` writes linear cells), appended raw data unless `.base64(true)`, and zlib with `.zlib(true)`; `.point_data(name, &buffer, components)` and `.cell_data(…)` attach device buffers of f32, f64, i32 or u32, indexed by geometry node and by flat cell index in block order. `vtu` and `pvtu` require one rank until M8 (ADR-0024)."
  - **"Reference cells and conventions", node families:** replace "(Until M5, `GmshWriter` rejects GLL geometry with `UnsupportedNodeFamily`; ADR-0022.)" with "Both writers call `gridcl_geometry::equispaced_nodes`; a node shared by several cells takes its value from its first cell in flat order, so the result is deterministic (ADR-0024)."
  - **"Import and export / VTK":**
    - the VTU bullet adds: "gridcl-io holds the file model (`VtuFile`), the cell-type table, the writer and `parse_vtu`; `gridcl::vtk::VtkWriter` downloads meshes. Points are all geometry nodes (Float64, or Float32 for f32 meshes), also for linear output. zlib through `flate2` (`rust_backend`) (ADR-0024).";
    - the high-order-pyramids bullet reads: "… are written as vertices only (VTK type 14) with one warning per file (`log::warn!`). A subdivided output mesh is a possible later option (ADR-0024). …";
    - the PVTU bullet adds: "M5 writes the header and one piece on one rank; M8 adds more ranks and `vtkGhostType`.";
    - the VTU-reading bullet reads: "**VTU reading** (`parse_vtu`, file level; linear and Lagrange) has lower priority and is used for round-trip tests and the VTK-written fixtures; no mesh is built from a VTU (ADR-0024).";
    - the test-oracles bullet adds: "Pinned at `vtk==9.7.1` and `meshio==5.3.5`. `vtkIntegrateAttributes` integrates Lagrange cells piecewise linearly (VTK 9.7, both strategies), so it is exact only on affine cells; curved cells are checked through `EvaluateLocation`. A missing interpreter fails the test unless `GRIDCL_SKIP_PYTHON=1` (ADR-0024)."
  - **"Testing and validation", I/O:** "every VTU fixture read back by VTK's Python reader" adds "(the files the M5 tests write, and the committed VTK-written reference cells)".
  - **"Workspace layout":**
    - gridcl-io row: purpose "Gmsh reader and writer (file model and host mesh description), VTU/PVTU (file model, writer, `parse_vtu`), VTKHDF"; dependencies add `flate2`;
    - gridcl row: purpose adds "`VtkWriter`";
    - gridcl-geometry row: purpose adds "GLL → equispaced conversion (`equispaced_nodes`)".
  - **Roadmap M5:**
    - the second acceptance bullet reads "the integrated volume or area of affine P2 and P3 meshes (`vtkIntegrateAttributes`) equals gridcl's `measure` to 10^−10 (f64); on curved P2 and P3 meshes, VTK's `EvaluateLocation` at Gauss points equals `physical_points` to 10^−10 (f64) (ADR-0024)";
    - the scope adds "and the device GLL → equispaced conversion for both writers".
  - **A.8:** the pyramid row's writer column reads "14 for p = 1; for p ≥ 2 vertices only (14) with one warning per file (ADR-0024)".
- **Public API changes:**
  - gridcl-io: the `vtk` module of item 1.
  - gridcl-geometry: `equispaced_nodes`.
  - gridcl:
    - the `vtk` module (`VtkWriter`, `VtkScalar`);
    - prelude exports `VtkWriter` and `vtk`;
    - `GmshWriter::write` no longer returns `UnsupportedNodeFamily` (documented errors only).
  - gridcl-testing (unpublished): `HostMesh::mapped`.
  - No existing signature changes.
- **Acceptance tests affected:**
  - M5 bullet 2 is split into `vtk_integrated_measure_equals_measure` (affine meshes) and `vtk_evaluates_curved_cells_like_gridcl` (curved meshes).
  - Bullet 1 covers the corpus of item 12.
  - Bullet 3 compares meshio's arrays bitwise with `parse_vtu` and with the uploaded data.
  - Bullet 4 runs on the P2 pyramid box and the P2 mixed box.
  - M4's `gmsh_write_rejects_gll` becomes `gmsh_write_converts_gll`.
  - No M1a–M4 acceptance test changes.
- **Later milestones affected:**
  - M8:
    - lifts the one-rank limit of `vtu` and `pvtu`;
    - adds `vtkGhostType`;
    - writes VTKHDF with the same cell types and arrays;
    - "output files, after sorting by origin id" may need origin-id arrays, which M8 decides.
  - M10 may add pyramid subdivision.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
