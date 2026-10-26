# ADR-0022: M4 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-26
- **Raised in:** M4 planning (docs/plans/M4.md, "Planned public API and rules", questions 2–17, 19 and 20)
- **Design sections affected:**
  - "Architecture": Listing B3 (the `tags` field comment), Listing B4 (`tags`), and "API walkthrough", Listing B8 (`gmsh::read`, `GmshOptions`);
  - "Reference cells and conventions" (the node-families bullet: writers convert to equispaced);
  - "Device model and backends": the paragraph after Listing S1 (`GridError` variants);
  - "Import and export / Gmsh MSH 4.1" (mesh content, tags, geometric dimension, origin ids, precision);
  - "Tags, regions and submeshes" (the MeshTags and codimension bullets);
  - "Workspace layout": the crate table (gridcl-io, gridcl);
  - "Implementation roadmap": M4 (the first two acceptance bullets).

## Context

M4 adds gridcl's own Gmsh MSH 4.1 reader and writer. "Import and export / Gmsh MSH 4.1" gives the
file-level rules, and Listing B8 the entry point:

> ```rust
> let any: AnyMesh<f64, Mpi> = gmsh::read(&client, comm.clone(), "head.msh",
>     GmshOptions::default().partitioner(PtScotch::default()).ghosts(GhostMode::SharedFacet))?;
> ```

Lower-dimensional elements become MeshTags: "their sorted corner-node tags (origin ids) form keys;
after `create_entities(d)`, the keys are matched against the mesh's entities of that dimension by
a device sort and merge". "Tags, regions and submeshes" describes MeshTags as "(dimension, sorted
local entity indices in flat numbering, i32 values) on the device", with a host `TagRegistry`
mapping names to (dimension, value). ADR-0012 adds the private `tags: TagRegistry` field in M4.

Planning read these sections against the code of M1a–M3s. The relevant facts:

- **The crate graph.** `Mesh`, `MeshBuilder` and `AnyMesh` live in gridcl (gridcl/src/{mesh,builder,any}.rs). gridcl depends on gridcl-io, whose crate-table row says it holds the "Gmsh reader and writer".
- **`MeshBuilder`** (gridcl/src/builder.rs) takes `nodes(coords, Option<&[i64]>)` and `cell_block(CoordinateElement<TD>, &[u32] /* A.5 order */, Option<&[i64]>)`, and returns a `Serial` mesh. It keeps only the origin ids of vertices (`Topology::vertex_origin_ids`); `Geometry` stores no node ids. Default ids are indices from 0.
- **`GridError`** (gridcl-device/src/error.rs) has no I/O, parse, element-type or tag variant, and is `Clone`.
- **`SortKeys`** holds 1–4 u32 words per key. Entity keys are sorted local vertex indices; they are dropped after `create_entities`, and no public function maps a vertex tuple to an entity.
- **No tag types exist.** `TagRegistry`, `MeshTags`, `Codim` and `Mesh::tags` appear only in the design text; no listing gives their signatures.
- **No logging crate** is in the workspace.
- **gridcl-refcell** has `gmsh_to_internal(cell, degree)` for every cell and degree, and nothing for serendipity elements.
- **Gmsh** requires node and element tags ≥ 1, and lets one elementary entity belong to several physical groups and one name be used in several dimensions.

## Problem

1. **Dependency cycle.** A `gridcl_io::gmsh::read` that returns `AnyMesh` or calls `MeshBuilder` needs gridcl-io → gridcl, but gridcl → gridcl-io already exists.
2. **No error variants.** The design names `GridError::UnmatchedTagElement` but not its fields. File errors (I/O, malformed files, unsupported element types) and unknown tag names have no variant, and `std::io::Error` is not `Clone`.
3. **Tags have no API, and the described shape cannot hold Gmsh data.**
   - Nothing fixes the fields of `MeshTags`, the methods of `TagRegistry` (Listing B9 calls `tags().named(Codim::Bnd, "scalp")`), or `Codim`.
   - One value per entity cannot represent an element in two physical groups, and a name → (dimension, value) table cannot hold one name in two dimensions.
   - Selecting one group on the device needs its length on the host. A count read is not among CLAUDE.md's allowed copies outside I/O.
4. **Keys do not fit.** Origin ids are two u32 words on the device, so a quadrilateral key of origin ids needs 8 words and a triangle 6; `SortKeys` holds 4. Entity keys are not kept, and there is no lookup.
5. **Listing B8 uses types of later milestones.** `Mpi` reads need M8; `PtScotch` arrives in M7 and `GhostMode` in M6. Without a decision now, M4's `read` signature would change in M8.
6. **D_GEO and `GmshOptions::gdim` leave cases open.** The design does not say what a fixed `gdim` outside 1–3, below D_TOP or below the inferred D_GEO does, or whether "all nodes" means all nodes of the file or of the D_TOP elements.
7. **Non-vertex node tags are lost.** The writer cannot write back the tags of edge, face and interior nodes, and the round-trip bullet does not say what "identical" means for them.
8. **The round-trip tolerance has no reference.** "Coordinates equal to 10^−15 relative" cannot hold, per coordinate, for coordinates near 0, since the mesh stores x − origin.
9. **GLL meshes.** "Writers always convert to equispaced", by "an interpolation applied on the device", but no conversion kernel exists, and `MeshBuilder` accepts GLL elements.
10. **Writer layout.** The design does not fix:
    - the degree of written tag elements;
    - element tags for tag elements;
    - how origin ids < 1 are written;
    - how `$Entities` and node blocks are laid out;
    - the binary byte order.
11. **Serendipity upgrade.** The design does not say:
    - where the serendipity map lives;
    - how added nodes are numbered;
    - that a face-centre node shared by two cells must be created once.
12. **"Logs one warning per file"** needs a logging mechanism.
13. **Reader rules for cell blocks.** The builder allows at most one block per cell type. The design does not say:
    - what happens when D_TOP has one cell type in two degrees;
    - the block order;
    - what happens with a file without cells;
    - how partitioned files are read on one rank.
14. **"Correct node counts" is ambiguous.** The builder drops nodes not used by any cell, and the serendipity upgrade adds nodes.
15. **The host copy of the match result.** Naming the element in `UnmatchedTagElement` needs the match result on the host.

## Options

1. **Keep the design as written.** `gmsh::read` in gridcl-io returning `AnyMesh` cannot compile (problem 1). The rest leaves the M4 tests and T0 stubs without signatures to target. Not chosen.
2. **Fix the items below; keep the reader and writer at the file level in gridcl-io, and the mesh level in gridcl** (proposed).
3. **Move `Mesh`, `MeshBuilder` and `AnyMesh` into a new crate below gridcl-io** (for example gridcl-mesh), so gridcl-io can build meshes itself.
   - This changes the crate table and the facade's role, and moves code that M3 finished.
   - Gmsh-specific mesh construction would be in gridcl-io, and M5's VTK writer would need the same crate.
   - Not chosen: option 2 meets the design's API without moving the mesh.
4. **Store node origin ids in `Geometry`** (problem 7). This changes Listing B7 and the builder, and costs device memory for every mesh. The maintainer chose fresh tags instead (2026-10-26).
5. **One `MeshTags` per physical group** instead of one per dimension (problem 3). Group lengths would then be known without a count read, but values would be redundant, and "Tags, regions and submeshes" describes per-dimension values. Not chosen.
6. **Tag elements at the cell degree** (problem 10). This needs the closure nodes of each facet or edge, oriented, from a neighbouring cell. The maintainer chose linear tag elements for M4 (2026-10-26).
7. **The GLL conversion kernel in M4** (problem 9). The maintainer chose to reject GLL meshes in M4's writer and add the conversion in M5, whose VTK writer needs it too (2026-10-26).

## Proposed decision

Option 2. Each item answers the problem of the same number. The maintainer chose items 7, 9 and
the tag-element degree in item 10 during M4 planning (2026-10-26).

1. **Crate split.**
   - `gridcl_io::gmsh` holds the host side:
     - the file model `MshFile` (with `PhysicalName`, `Entities`, `NodeBlock`, `ElementBlock`, `PartitionedEntities`, `Periodic`; `pub` fields; tags `i64`; coordinates `[node][3]` in f64);
     - `parse(bytes: &[u8]) -> Result<MshFile>`, `read_file(path: &Path) -> Result<MshFile>`, `MshFile::write_to(&self, w: &mut dyn Write, encoding: Encoding) -> Result<()>` and `enum Encoding { Ascii, Binary }`;
     - the element table of A.7: `ElementType { id: i32, cell: CellType, degree: u32, num_nodes: usize, serendipity: bool }`, `element_type(id) -> Result<ElementType>` and `lagrange_element_type(cell, degree) -> Result<ElementType>`;
     - the host mesh description `GmshMesh` (with `GmshCellBlock` and `GmshTagGroup`), `to_gmsh_mesh(file: &MshFile, comm: &dyn CommDyn, gdim: Option<usize>) -> Result<GmshMesh>` and `from_gmsh_mesh(mesh: &GmshMesh) -> Result<MshFile>`.
   - `gridcl::gmsh` holds `read`, `GmshOptions` and `GmshWriter` (item 5), which build a `Mesh` through `MeshBuilder` or download one.
   - The prelude exports the module `gmsh` and `GmshOptions`.
   - gridcl-io gains direct dependencies on gridcl-dims, gridcl-refcell and gridcl-device. These are already transitive, so the crate graph gains no edge. Its dev-dependencies are `mshio = "=0.4.2"` (the test oracle) and `serde_json`.
   - The parser checks header counts with checked arithmetic and sizes no allocation from a count before the data is there. Unknown sections are skipped, parametric node coordinates are read and dropped, and `$Periodic` is stored, not applied.
2. **`GridError` variants**, each with a `Display` line:
   - `UnmatchedTagElement { dim: usize, tag: i64 }`, where `tag` is the Gmsh element tag;
   - `Io(String)`: the path and the `std::io::Error` text, so `GridError` stays `Clone`;
   - `Parse { format: &'static str, offset: u64, message: String }`, with a byte offset in ASCII and binary files alike. M5's VTU reader reuses it.
   - `UnsupportedElementType { format: &'static str, type_id: i32 }`;
   - `UnknownTag { name: String, dim: usize }`.
3. **Tags** (gridcl):
   - `pub struct MeshTags { pub dim: usize, pub indices: DeviceBuffer<u32>, pub values: DeviceBuffer<i32> }`.
     - Entries are sorted by (index, value). An index repeats only with distinct values, which happens when an element is in several physical groups.
     - Cells (dim = tdim) use flat numbering over the cell blocks in block order, like entity blocks.
   - `pub enum Codim { Vol, Bnd, BBnd, BBBnd }` with `Codim::dim(self, tdim: usize) -> Option<usize>`.
   - `TagRegistry` (`Default` is empty) holds per dimension one physical and one elementary `MeshTags`, the name table, and each group's size, recorded on the host at read time:
     - `names() -> impl Iterator<Item = (&str, usize, i32)>`;
     - `lookup(dim, name) -> Option<i32>`;
     - `physical(dim) -> Option<&MeshTags>` and `elementary(dim) -> Option<&MeshTags>`;
     - `group_size(dim, value) -> Option<usize>`;
     - `group(dim, value) -> Result<MeshTags>` and `named(codim, name) -> Result<MeshTags>`. Each selects one group on the device into a buffer of the recorded size, so it needs no count read. An unknown name or a name absent in that dimension gives `UnknownTag`.
   - `Mesh::tags(&self) -> &TagRegistry` reads the private `tags` field of Listing B3. `Mesh::try_clone` deep-clones the tags.
   - Ghost consistency by forward scatter remains M6's.
4. **Matching.**
   - The reader maps corner node tags to local vertex indices on the host. A corner node that is not a mesh vertex gives `UnmatchedTagElement`.
   - The matching itself is `Topology::match_entities(&self, dim: usize, keys: &SortKeys) -> Result<DeviceBuffer<u32>>` (gridcl-topology). Keys are sorted local vertex indices of 1–4 words; the word count selects the entity type.
   - The steps:
     - a sort kernel rebuilds entity keys from `EntityBlock::vertices`;
     - the entity keys go before the tag keys, and one stable `radix_sort` sorts both;
     - `unique_by_key` groups them, with its allowed count read;
     - one kernel gives each tag key the flat index of its group's entity, or `u32::MAX` if the group has none.
   - The new kernels plan with `KernelKind::Scalar` (ADR-0015).
   - Physical groups of D_TOP elements tag cells directly, by the builder's cell numbering.
5. **Entry points.**
   - `pub fn read<F: Real, C: Comm>(client: &Client, comm: C, path: impl AsRef<Path>, options: GmshOptions) -> Result<AnyMesh<F, C>>`. In M4, `comm.size() > 1` gives `NotYetImplemented("parallel Gmsh read (M8)")`. Building a mesh for a `C` other than `Serial` goes through a crate-private `MeshBuilder::build_with_comm`.
   - `GmshOptions` implements `Default` and has private fields. In M4 it has only `gdim(self, usize) -> Self`. M8 adds `partitioner` and `ghosts` as further builder methods, so M4's calls stay valid.
   - `GmshWriter::new(path)`, `.binary(bool)` (ASCII by default), `.write(&Mesh<F, TD, GD, C>)` and `.write_any(AnyMeshRef<'_, F, C>)`. One rank only until M8.
6. **D_GEO.**
   - The inference runs over all nodes of the file, with the design's absolute rule as written, so a plane at z = c ≠ 0 is 3D.
   - `GmshOptions::gdim` must be 1–3 and at least D_TOP. Fixing it below the inferred D_GEO would drop non-zero coordinates, so it gives `DimensionMismatch`. Fixing it above raises the embedding.
7. **Node tags.**
   - The mesh keeps no tags for non-vertex nodes.
   - The writer writes vertex origin ids as node tags and numbers the other nodes after the largest vertex id, in node order.
   - "Identical topology" in the round trip means:
     - cell types and degrees;
     - cell origin ids and cell vertex origin ids;
     - names;
     - physical and elementary tags keyed by origin-id tuples.
   - Coordinates are compared per cell, in internal node order.
8. **Tolerance.** "Relative" in M4's round-trip bullet is relative to the largest absolute coordinate of the mesh. f32 meshes round-trip to 10^−5.
9. **GLL.** Until M5, `GmshWriter` returns `UnsupportedNodeFamily` for GLL geometry. M5 adds the device conversion to equispaced and both writers use it. The reader always produces equispaced meshes.
10. **Writer layout.**
    - **Tag elements** are linear: their corner nodes from `EntityBlock::vertices`, in canonical A.6 orientation. The file's original boundary orientation is not kept.
    - **Element tags:** cell origin ids for cells; tag elements are numbered after the largest cell id.
    - **Tags below 1:** if any node tag or element tag would be below 1, all tags of that kind are shifted by one constant so that the smallest is 1.
    - **`$Entities`:**
      - one node block, in the first D_TOP entity;
      - entities rebuilt from the elementary tags, one per (elementary tag, set of physical tags);
      - each entity gets a bounding box from its nodes, dimension-0 entities get their point's coordinates, and boundary lists are empty;
      - a mesh without tags gets one entity with tag 1.
    - **Coordinates:** always three per node.
    - **Binary:** always little-endian with an 8-byte `size_t`.
    - **Reader side:** node and element tags ≤ 0 give `Parse`.
11. **Serendipity.**
    - `gridcl_refcell::serendipity_upgrade(cell: CellType) -> Result<SerendipityUpgrade, RefCellError>` gives, in f64 on the host, the weights of the missing complete-P2 nodes over the file's 8, 20, 15 or 13 nodes (Gmsh order). The weights come from Gmsh's serendipity map evaluated at the missing nodes. The serendipity bases themselves are crate-private.
    - An added face-centre node is created once per face, keyed by the face's sorted corner node tags. It reuses the node of a complete-P2 neighbour where there is one, and otherwise takes a fresh tag above the file's largest node tag, in order of first use.
    - The upgraded cells are degree 2 in the cell type's block.
12. **Logging.** gridcl depends on `log = "0.4"` (a workspace dependency). `gmsh::read` into an f32 mesh calls `log::warn!` once per file. M5's pyramid warning uses the same crate.
13. **Reader rules.**
    - Cell blocks are in `CellType` order, with cells in file order within a block.
    - One cell type in two degrees in D_TOP gives `DuplicateCellBlock`.
    - A file without elements of dimension ≥ 1 gives `Parse`.
    - Duplicate node tags and duplicate element tags give `DuplicateOriginId`, checked over the whole file before building.
    - A partitioned file read on one rank ignores the partition, and takes elementary tags from the parent entities in `$PartitionedEntities`.
14. **Node counts.** "Correct node counts" means:
    - the parsed file has Gmsh's node count;
    - the mesh has the number of distinct nodes of the D_TOP elements;
    - for a serendipity file, the mesh has the count of the same mesh written as complete P2.
15. **Host copy.** `gmsh::read` copies the match result to the host to name the first unmatched element. This is an I/O copy under CLAUDE.md's existing exemption, so CLAUDE.md does not change.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B3:** the `tags` comment reads `// physical and elementary MeshTags per dimension, names (ADR-0012, ADR-0022)`.
  - **Listing B4:** the first `impl` block adds `/// Physical and elementary tags and the name table (empty unless read from a file).` with `pub fn tags(&self) -> &TagRegistry;`.
  - **Listing B8:**
    - The `gmsh::read` call keeps its form.
    - A sentence after the listing reads: "`gmsh::read` is generic over `C: Comm`. Until M8 it requires one rank, and `GmshOptions` offers only `gdim`; `partitioner` and `ghosts` arrive with M7 and M6–M8 (ADR-0022)."
  - **"Reference cells and conventions", node families:** append "(Until M5, `GmshWriter` rejects GLL geometry with `UnsupportedNodeFamily`; ADR-0022.)"
  - **The paragraph after Listing S1:** append "M4 adds `UnmatchedTagElement { dim: usize, tag: i64 }`, `Io(String)`, `Parse { format: &'static str, offset: u64, message: String }`, `UnsupportedElementType { format: &'static str, type_id: i32 }` and `UnknownTag { name: String, dim: usize }` (ADR-0022)."
  - **"Gmsh MSH 4.1":**
    - "Own reader and writer" adds: "gridcl-io parses and writes files through a host model (`MshFile`, `GmshMesh`); `gridcl::gmsh` builds and downloads meshes (ADR-0022)."
    - "Mesh content":
      - the matching sub-bullet reads "… corner-node tags are mapped to local vertex indices, and the keys are matched against the mesh's entities of that dimension by a device sort and merge (`Topology::match_entities`) …";
      - it adds the block-order, mixed-degree and empty-file rules of item 13.
    - "Tags" adds: "An element may be in several physical groups; a name may be used in several dimensions."
    - "Geometric dimension" adds: "The reduction covers all nodes of the file. A fixed `gdim` must be 1–3, at least D_TOP and at least the inferred D_GEO; otherwise `GridError::DimensionMismatch`."
    - "Origin ids" adds: "Tags must be ≥ 1. The writer writes vertex origin ids as node tags, numbers the other nodes after the largest vertex id, and shifts tags below 1 (ADR-0022)."
    - A new bullet, **Writer**, gives the layout of item 10.
    - A new bullet, **Serendipity nodes**, gives item 11.
    - "Precision" reads "… logs one warning per file (`log::warn!`)".
  - **"Tags, regions and submeshes":**
    - The MeshTags bullet reads "(dimension, local entity indices in flat numbering sorted by (index, value), i32 values) on the device … An index repeats only with distinct values. `TagRegistry` holds a physical and an elementary `MeshTags` per dimension, the name table (name, dimension) → value and each group's size; `named` and `group` select one group on the device (ADR-0022)."
    - The codimension bullet adds `Codim::dim(tdim) -> Option<usize>`.
  - **"Workspace layout":**
    - the gridcl-io row reads "Gmsh reader and writer (file model and host mesh description), VTU/PVTU, VTKHDF | dims, refcell, device, topology, geometry, parallel, partition; hdf5-metno (optional); dev: mshio, serde_json";
    - the gridcl row's purpose adds "`gmsh::read` and `GmshWriter`", and its dependencies read "all of the above; log".
  - **Roadmap M4:**
    - acceptance bullet 1 reads "read → write → read gives identical topology (cell types and degrees, cell and vertex origin ids, names and tags keyed by origin ids), and coordinates equal to 10^−15 relative to the largest absolute coordinate (f64; 10^−5 in f32)";
    - bullet 2 adds "(file: Gmsh's node count; mesh: the nodes of D_TOP elements)".
- **Public API changes:**
  - gridcl-device: the five `GridError` variants.
  - gridcl-refcell: `serendipity_upgrade`, `SerendipityUpgrade`.
  - gridcl-topology: `Topology::match_entities`.
  - gridcl-io: the `gmsh` module of item 1.
  - gridcl:
    - the `gmsh` module (`read`, `GmshOptions`, `GmshWriter`);
    - `MeshTags`, `TagRegistry`, `Codim` and `Mesh::tags`;
    - prelude exports.
  - None of these changes an existing signature.
- **Acceptance tests affected:**
  - M4 bullets 1 and 2 read as in items 7, 8 and 14 (`read_write_read_round_trip`, `corpus_reads_with_gmsh_node_counts`, `corpus_builds_meshes_with_gmsh_node_counts`).
  - `tag_counts_equal_physical_group_sizes` compares the matched entities with Gmsh's per-group corner-node sets, not only the counts.
  - No M1a–M3s test changes.
- **Later milestones affected:**
  - M5:
    - adds the device GLL → equispaced conversion and lifts the `GmshWriter` restriction;
    - reuses `GridError::Parse` for VTU reading;
    - uses `log` for the pyramid warning.
  - M6 makes ghost tag values consistent by forward scatter.
  - M8:
    - lifts the one-rank limit of `read` and `GmshWriter`;
    - adds `GmshOptions::partitioner` and `ghosts`;
    - distributes the key matching (keys hashed to ranks), reusing `match_entities` on each rank;
    - honours the partition data of partitioned files.
  - M9's `facet_submesh(&MeshTags)` and `orient` use the tag types of item 3.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
