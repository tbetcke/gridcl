# ADR-0035: M8 public items and rules the design leaves open

- **Status:** Accepted
- **Date:** 2026-11-02
- **Raised in:** M8 planning (docs/plans/M8.md, "Planned public API and rules" and questions 1–16)
- **Design sections affected:**
  - "Architecture": Listing B4 (a hidden test accessor);
  - "API walkthrough": the sentence after Listing B8; the sentence after Listing B13;
  - "Distributed mesh model" / "Parallel creation pipeline": stages 1 and 3 of the stage table;
  - "Partitioning": the Block row of the backend table, the bullet "Default choice";
  - "Import and export" / "Gmsh MSH 4.1": the bullets "Parallel read" and "Parallel write"; "VTK": the PVTU and VTKHDF bullets, "Test oracles";
  - "Tags, regions and submeshes": the MeshTags bullet;
  - "Testing and validation": the I/O bullet;
  - "Workspace layout": the crate table (gridcl-io, gridcl, gridcl-testing, gridcl-mpi-tests), "Cargo features", "CI matrix" (the mpi rows);
  - "Implementation roadmap": M8 acceptance (how the four bullets are read).

## Context

M8's roadmap section:

> **Scope:** chunked binary Gmsh reading, partitioned writing, PVTU with `vtkGhostType`, and VTKHDF (hdf5-metno with `mpio`).

Acceptance: the global mesh, compared by origin ids, is identical for 1–8 ranks; reading a 10^7-cell binary file on 8 ranks of one Kathleen node keeps peak resident memory per rank below twice the rank's final mesh size plus 256 MiB (getrusage); VTKHDF read back by VTK's Python reader equals the PVTU content; after VTK removes ghost cells, every cell appears exactly once.

The design adds: "Binary files are read in byte-range chunks per rank. Block headers are scanned first … ASCII files are read on rank 0 and scattered" and "Parallel write: gather to rank 0, or one partitioned file per rank with $PartitionedEntities" ("Gmsh MSH 4.1"); stage 1 gives rank r "cells [rN/P, (r+1)N/P) with origin ids, types and tags, and an independent chunk of nodes"; Block "also honours partition data stored in Gmsh files" ("Partitioning"); PVTU ghost cells "carry a `vtkGhostType` cell array (value 1 = duplicate cell)", and VTKHDF is "a single file with per-rank offsets, the same cell types and the same ghost array" ("VTK"). Earlier ADRs leave M8 these items: `GmshOptions::partitioner` and `ghosts` (ADR-0022 item 5), the distributed key matching and the partition data (ADR-0022, "Later milestones"), the reader's default `CellOrdering` (ADR-0026 item 6), and origin-id arrays in VTK output (ADR-0024, "Later milestones").

Planning read these against the code at main e3107ed:

- **Parser.** `gridcl_io::gmsh::read_file` (`gridcl-io/src/gmsh/parse.rs:572`) reads the whole file with `std::fs::read` and parses it in one forward pass. Binary records have fixed sizes (`size_t` of 4 or 8 bytes, either byte order), so headers can be scanned by seeking.
- **Conversion.** `to_gmsh_mesh` (`gmsh/convert.rs:421`) checks duplicates and missing nodes (`Parse`) on the whole file. `from_gmsh_mesh` (:721) computes the largest vertex id and the shift of tags below 1 per call, and never writes `$PartitionedEntities`.
- **`gmsh::read`** (`gridcl/src/gmsh.rs:221`) refuses more than one rank, builds with `nodes` and `cell_block`, and tags cells by flat input index. Its test `gmsh_read_unmatched_tag_element_and_io` expects the Gmsh element tag in `UnmatchedTagElement` (ADR-0022 item 4). `GmshOptions` derives `Clone, Debug, Default`.
- **Builder.** Lower-dimensional physical tags are matched by rendezvous after upload (`pipeline/tags.rs::match_lower`); their `UnmatchedTagElement.tag` is the entry's global input position (ADR-0026 item 3). Cell tags use a keyed rendezvous in stage 1. There are no elementary tags, per-cell tag values or fixed destinations. Duplicate cell and node ids are already found (`check_cell_ids`, the stage-6 directory).
- **Partitioners.** `PtScotch` and `ParMetis` return `NeedsMpiCommunicator` when `comm.raw()` is `None`, which it is for `ThreadComm` and `Serial`.
- **Writers.** `gmsh.rs::download` and `vtk.rs::download` take every local cell (`0..num_local`), ghosts included, without marking them. `gridcl_io::vtk::write_pvtu(w, pieces, like)` hard-codes `GhostLevel="0"`. Node origin ids are not stored on the mesh; only vertex origin ids are.
- **HDF5.** hdf5-metno 0.15.1 (MSRV 1.85.1; hdf5-metno-sys 0.12.5 knows HDF5 1.8.5–1.14.6 and 2.0–2.2.0) has a feature `mpio = ["dep:mpi-sys", "hdf5-sys/mpio"]` with `mpi-sys = "0.2"`, the version rsmpi 0.8.2 uses, and a safe `FileAccessBuilder::mpio(comm, info)`. The local Mac has Homebrew's `hdf5-mpi` 2.2.0 (parallel); Ubuntu 24.04 ships `libhdf5-{mpich,openmpi}-dev` 1.10.10.
- **Oracles.** `scripts/oracles/vtk_oracle.py` reads VTU and PVTU; it has no VTKHDF reader and no ghost removal.
- **Measurement.** `getrusage` is a libc call; CLAUDE.md allows FFI calls only in gridcl-parallel, gridcl-device and gridcl-partition. `scripts/mpi-test.sh` builds debug test binaries.

## Problem

Each item matches the question of the same number in docs/plans/M8.md.

1. **Default cell order.** ADR-0026 item 6: "M8's parallel reader chooses its own default". M4 requires file order within a block on one rank.
2. **Reader options.** Listing B8 calls `.partitioner(PtScotch::default()).ghosts(GhostMode::SharedFacet)` without signatures or defaults. The design's default ("PT-Scotch when the `ptscotch` feature is enabled") fails with `NeedsMpiCommunicator` on `ThreadComm`, so a gridcl build with `ptscotch` could not read a file in the in-process tests. Holding a `Partitioner` in `GmshOptions` while keeping `Clone` needs a shared pointer, which drops `Send` and `Sync`.
3. **getrusage.** The memory bullet names getrusage; the FFI rule leaves no crate for a test-side call.
4. **Origin ids in VTK output.** Bullet 4 has to identify cells after VTK removes ghosts, and "output files, after sorting by origin id" ("Testing and validation") needs an origin-id array. ADR-0024 left both to M8.
5. **VTKHDF.** The design gives no constructor, file version, compression rule, or behaviour for communicators without an MPI handle (`Serial`, `ThreadComm`), with which mpio cannot run.
6. **gridcl-io items.** Chunked reading needs public file-level items in gridcl-io, as ADR-0022 item 1 and ADR-0024 item 1 listed for M4 and M5. `write_pvtu` needs the ghost level, which changes its signature.
7. **Serendipity files.** Upgrading types 16–19 needs the face nodes of neighbouring elements and fresh tags "in order of first use" (ADR-0022 item 11). Both are global, so byte-range chunks cannot do it locally. The design names only ASCII files for the rank-0 path.
8. **Elementary tags.** ADR-0026 item 3: "Only physical tags; elementary tags remain the Gmsh reader's". A distributed read must move elementary tags with their cells and entities, just like physical tags. Matching 10^7 cell tags through a keyed rendezvous would also add a second all-to-all of every cell to the memory peak.
9. **Error forms.** Going through the builder would change `UnmatchedTagElement.tag` from the Gmsh element tag (ADR-0022) to a global input position (ADR-0026). A node missing from `$Nodes` would become stage 6's `IndexOutOfRange` instead of `Parse`. Duplicate tags between tag elements, and between tag elements and cells, would not be found at all.
10. **Partition data.** "Block also honours partition data" does not say when, or what happens when the file's partition count differs from the rank count.
11. **Partitioned writing.** The design gives no file names, no `$PartitionedEntities` layout, and no numbering of non-vertex nodes and tag elements that agrees across ranks. ADR-0022 item 7 numbers non-vertex nodes "after the largest vertex id, in node order", and `from_gmsh_mesh` computes that bound per rank.
12. **VTK writers on many ranks.** Open points:
    - what `vtu` writes on more than one rank;
    - whether one-rank PVTU output gains `vtkGhostType`, which would change M5's files;
    - what `GhostLevel` the header declares.

    A gather, a PVTU piece or a collective HDF5 call can fail on one rank and leave the others waiting or returning `Ok`. That breaks ADR-0026 item 15 (agreed errors).
13. **Memory bullet.** The design does not define "final mesh size", the 10^7-cell file, the partitioner, the ghost mode or the build profile. Debug builds add host copies (the index checks of gather and scatter), and METIS on rank 0 gathers the whole graph on one rank.
14. **Bullet 1.** "The global mesh" lists no quantities. Running 60 corpus files at every rank count from 1 to 8, in two ghost modes, both precisions and with every partitioner would exceed CI time.
15. **Bullet 3.** "Equals the PVTU content" sets no comparison or tolerance.
16. **Features.** It is open whether `hdf5` needs MPI and a parallel HDF5. The crate table lists no test-side dependencies: a getrusage wrapper, the oracle's `serde_json` in gridcl-testing, and a file generator binary.

## Options

1. **Keep the design as written.**
   - The ASCII-only rank-0 rule would leave serendipity binary files without a parallel read (7).
   - The design's default partitioner would fail on `ThreadComm` whenever `ptscotch` is on (2).
   - Error forms would change silently (9).
   - The memory, VTKHDF and output-invariance bullets would have no test definition (3, 4, 13–15).

   Not chosen.
2. **Fix the open items as below (proposed).** One option per question, with these alternatives considered:
   - **1:** `Hilbert` when `size() > 1`. Locality by default, but a mesh's cell order would depend on the rank count, and the parallel default would differ from `MeshBuilder`'s `Input`. Not proposed.
   - **3:** a crate-private `gridcl_device::testing::peak_resident_bytes()` with `libc` on every unix (today: macOS only). This keeps the call inside an FFI-permitted crate, but adds a library-crate dependency and a test hook for one test. Not proposed.
   - **4:** always write `OriginId` on more than one rank. M5's one-rank files would differ from multi-rank output, so the invariance comparison would need special handling; opt-in keeps every output explicit. Not proposed.
   - **7:** a distributed serendipity upgrade: face keys by rendezvous, and fresh tags by a global sort of first-use positions. This is a large algorithm for a legacy element family that the corpus has only as small files. Not proposed.
   - **8:** a public `MeshBuilder::elementary_tags`. Nothing outside the reader needs it. Not proposed.
3. **Defer VTKHDF to M10.** This would leave bullet 3 untestable in M8. Not proposed.

## Proposed decision

Option 2, item by item.

1. **Default order.** `GmshOptions::ordering(self, o: CellOrdering) -> Self`, default `CellOrdering::Input` on every rank count. `Hilbert` and `ReverseCuthillMcKee` are opt-in, as in `MeshBuilder`. On one rank the order is file order within a block, as M4 requires.
2. **Reader options.**
   - `GmshOptions::partitioner(self, p: impl Partitioner + 'static) -> Self` and `GmshOptions::ghosts(self, mode: GhostMode) -> Self`, the latter defaulting to `GhostMode::None`.
   - `GmshOptions` stays `Clone + Debug + Default`. It holds the partitioner as `Rc<dyn Partitioner>`, and its manual `Debug` prints `name()`. It is therefore not `Send` or `Sync`, like `Mesh`.
   - **Default partitioner** when `comm.size() > 1` and none is set:
     1. `PtScotch` if the `ptscotch` feature is on and `comm.raw()` is `Some`;
     2. else `Metis` if `metis` is on;
     3. else `Block`, with one `log::warn!`.

     One rank uses no partitioner.
   - **Warnings** (Block fallback, f32 narrowing, the pyramid fallback) are logged once, by rank 0.
3. **getrusage.** gridcl-mpi-tests (unpublished) calls `nix::sys::resource::getrusage(UsageWho::RUSAGE_SELF)`. `nix` is a dependency only on `cfg(unix)`, with feature `resource`. `ru_maxrss` is in KiB on Linux and in bytes on macOS. No gridcl code calls FFI, and the FFI rule is unchanged.
4. **Origin ids in VTK.** `VtkWriter::origin_ids(self, yes: bool) -> Self`, default false. True adds the cell data `OriginId` (Int64), the cell origin ids in the writer's cell order. Point data is unchanged, because the mesh keeps no node origin ids. The writers add no other array, apart from `vtkGhostType` (item 12).
5. **VTKHDF.**
   - `VtkWriter::vtkhdf(path: impl AsRef<Path>) -> Self` (feature `hdf5`) writes `<stem>.vtkhdf`, appending the extension if the path has none.
   - It is an `UnstructuredGrid` with one partition per rank:
     - per-partition `NumberOfPoints`, `NumberOfCells` and `NumberOfConnectivityIds`;
     - `Points`, `Connectivity` (partition-local point ids), `Offsets`, `Types`;
     - groups `CellData` and `PointData`;
     - the format version that VTK 9.7.1's own `vtkHDFWriter` writes, and file-format bounds with low = v18, so that the HDF5 inside the VTK wheel reads the file.
   - Each partition holds what that rank's PVTU piece holds (item 12), including `vtkGhostType` and, if requested, `OriginId`.
   - With `comm.raw()` = `Some` (features `mpi` and `hdf5`), all ranks write collectively through the `mpio` driver; a rank with an empty partition still takes part in every collective call. Otherwise (`Serial`, `ThreadComm`), the partitions are gathered and rank 0 writes with HDF5's default driver.
   - Datasets and attributes are the same on both paths.
   - `.base64` has no effect. `.zlib(true)` gives `NotYetImplemented("compressed VTKHDF")`.
6. **gridcl-io items.**
   - `gridcl_io::gmsh::MshIndex` holds the format, `size_t` width, byte order, the small sections, and the node and element block headers with byte offsets. It has:
     - `scan(path: &Path) -> Result<MshIndex>`, which reads headers and small sections only, by seeking;
     - `is_binary(&self) -> bool`;
     - `num_nodes(&self) -> usize` and `num_elements(&self, dim: usize) -> usize`;
     - `read_nodes(&self, range: Range<usize>) -> Result<NodeChunk>`, in file order over all node blocks;
     - `read_elements(&self, dim: usize, range: Range<usize>) -> Result<ElementChunk>`, in file order within that dimension.
   - `pub struct NodeChunk { pub tags: Vec<i64>, pub coords: Vec<[f64; 3]> }` and `pub struct ElementChunk { pub blocks: Vec<ElementBlock> }`, which reuses the existing model type cut to the range.
   - `gridcl_io::vtk::write_pvtu(w: &mut dyn Write, pieces: &[String], like: &VtuFile, ghost_level: u32) -> Result<()>`. This changes ADR-0024's signature: it gains `ghost_level`.
   - `#[cfg(feature = "hdf5")] gridcl_io::vtk::write_vtkhdf(path: &Path, piece: &VtuFile, comm: &dyn CommDyn) -> Result<()>`, which is collective (item 5).
   - Other conversion helpers that gridcl needs from gridcl-io are `#[doc(hidden)] pub`, like ADR-0028's test accessors.
7. **Rank-0 read.** ASCII files and binary files that contain serendipity elements (types 16–19) are read whole on rank 0. Rank 0 upgrades them as in ADR-0022 item 11, then sends the same chunks the binary path produces. Every other binary file is read in chunks (item 14's stage-1 text).
8. **Elementary tags and per-cell values.** These are crate-private builder inputs, used by the reader:
   - D_TOP physical and elementary values travel with their cells from stage 1, with no keyed rendezvous;
   - lower-dimensional elementary tags take the `match_lower` path beside the physical ones.

   `MeshBuilder`'s public API is unchanged: `physical_tags` and `tag_names` only. Group sizes stay local counts (ADR-0026).
9. **Serial error forms.**
   - The crate-private tag inputs carry the Gmsh element tag as their error id. An unmatched entry gives `UnmatchedTagElement { dim, tag }` with the tag of the unmatched entry at the lowest input position, agreed on every rank. On one rank this is today's result.
   - A node that an element references but `$Nodes` lacks gives `Parse`, as in M4. The reader maps stage 6's `IndexOutOfRange` back to it.
   - Element tags share one space over all dimensions. Tag-element tags are checked against each other and against cell ids by a crate-private rendezvous in gridcl; a repeat gives `DuplicateOriginId`. Cell ids and node ids keep their existing checks.
   - Every error is agreed (ADR-0026 item 15).
10. **Partition data.** If the partitioner is `Block` (set explicitly, or the item-2 fallback) and the file's `numPartitions` equals `comm.size()`, each D_TOP element goes to rank `partition − 1`. Stages 2 and 3 are then skipped. Otherwise the partition data is ignored. Ghost entities in the file are always ignored, because gridcl builds its own ghosts by `GhostMode`.
11. **Parallel Gmsh writing.** Writes are collective and end with an agreed status.
    - **Default (gather).** Owned cells, the nodes they use and owned tag entities are gathered to rank 0, which writes one file: cells by (rank, owned order), nodes in global node order.
    - **Node tags.**
      - Vertex nodes get their origin ids.
      - Every other node gets M + 1 + k, where M is the largest vertex origin id over all ranks and k is the node's position among non-vertex nodes in global node order. k comes from an exscan over owned nodes and a forward scatter to ghosts.
    - **Tag-element tags** start after the largest cell id over all ranks and are numbered by an exscan over owned tag entities.
    - **Tags below 1.** The shift uses minima over all ranks.
    - On one rank all of this is ADR-0022 item 7's rule.
    - **`GmshWriter::partitioned(self, yes: bool) -> Self`**, default false. True makes rank r write `<stem>_<r+1>.msh`, Gmsh's split-file naming, containing:
      - its owned cells, the nodes they use and its owned tag entities;
      - `$Entities` with the parent entities;
      - `$PartitionedEntities` with `numPartitions` = P, no ghost entities, and one partitioned entity per (elementary tag, set of physical tags), whose parent is that entity and whose partition list is [r+1].

      Each file reads serially as that rank's part (ADR-0022 item 13).
12. **VTK writing on several ranks.** Writes are collective and end with an agreed status.
    - **`vtu`.** Owned cells are gathered to rank 0, which writes one file: points in global node order, no ghost array. Point data and cell data are selected the same way.
    - **`pvtu`.** Every rank writes its piece with all local cells (owned and ghost) and all local nodes.
      - With more than one rank the piece gets the cell data `vtkGhostType` (UInt8: 0 for owned, 1 = `DUPLICATECELL` for ghost). One-rank output is unchanged from M5.
      - Rank 0 writes the header with `GhostLevel` = 1 if any rank has ghost cells, else 0.
13. **Memory bullet.**
    - **Final mesh size** is `Mesh::footprint_bytes(&self) -> usize`, a `#[doc(hidden)]` test accessor. It sums the byte lengths of every device buffer the mesh holds (topology, entities, connectivity, geometry, tags) and its index maps' host vectors, measured when `gmsh::read` returns.
    - **Peak** is item 3's `ru_maxrss`.
    - **File.** `cube_box(Tetrahedron, 119, 1)`, 10 110 954 cells, with one volume group and six side physical groups. It is written in binary by the gridcl-testing binary `gen-gmsh-box` into the remote tree's `target/` (Lustre; Kathleen's nodes are diskless), in a separate process before the measured run.
    - **Read and run.** `gmsh::read` with `Block` and `GhostMode::None`, a release build (`GRIDCL_MPI_PROFILE=release` in `scripts/mpi-test.sh`), on 8 ranks of one Kathleen node.
    - PT-Scotch's peak is logged, not asserted. METIS on rank 0 gathers the whole graph by design and is not measured.
14. **Bullet 1.**
    - **"The global mesh"** is gathered to rank 0 by cell origin id:
      - cell type and degree;
      - vertex origin ids in cell order;
      - every node coordinate of the cell in internal order, bitwise;
      - physical and elementary tags keyed by sorted vertex-origin-id tuples;
      - the name table.

      It must equal the one-rank read.
    - **Corpus and ranks.** The whole corpus (ASCII and binary) runs at 1, 2, 3, 4 and 8 ranks or threads. A subset runs at 5–7: one file per cell type, the partitioned files, a serendipity file, and a generated multi-block binary box.
    - **Options.** Ghost modes `None` and `SharedFacet`. Every enabled partitioner on MPI; `Block` (and `Metis` with `metis`) on `ThreadComm`.
    - 5–8 ranks run on one Kathleen node (ADR-0016).
15. **Bullet 3.** VTK's `vtkHDFReader` output and its `vtkXMLPUnstructuredGridReader` output, read with ghost level 1, are equal array by array and bitwise: points, connectivity, offsets, types, every cell and point array (including `vtkGhostType` and `OriginId`), and the per-partition counts.
16. **Features and dependencies.**
    - gridcl-io: `hdf5 = ["dep:hdf5-metno"]` and a new `mpi = ["gridcl-parallel/mpi", "hdf5-metno?/mpio"]`, which gridcl's `mpi` forwards. `hdf5` alone builds with HDF5 1.10.10–2.2.0; `hdf5` with `mpi` needs a parallel HDF5 (`HDF5_DIR`).
    - `hdf5-metno = "=0.15.1"` goes in `[workspace.dependencies]`. It is a safe wrapper; the MPI handle comes from `CommDyn::raw()`.
    - Test-only additions:
      - gridcl-testing gains `serde_json` (the Python oracle helper moves there) and the binary `gen-gmsh-box`;
      - gridcl-mpi-tests gains `nix` (item 3) and a feature `hdf5`;
      - gridcl-io gains the dev-dependency gridcl-testing.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B4:** after the first `impl` block's `tags` line, add `// #[doc(hidden)] pub fn footprint_bytes(&self) -> usize;  test accessor: bytes of the mesh's buffers (ADR-0035)`.
  - **The sentence after Listing B8** reads: "`gmsh::read` is generic over `C: Comm`. `GmshOptions` has `gdim`, `partitioner`, `ghosts` (default `GhostMode::None`) and `ordering` (default `CellOrdering::Input`); without a partitioner, a read on more than one rank uses the default choice of "Partitioning" (ADR-0022, ADR-0035)."
  - **The sentence after Listing B13:** replace "`vtu` and `pvtu` require one rank until M8 (ADR-0024)." with "On more than one rank, `vtu` gathers the owned cells to rank 0, and `pvtu` and `vtkhdf` (feature `hdf5`) write one piece per rank with a `vtkGhostType` cell array; `.origin_ids(true)` adds the cell array `OriginId`. Writes are collective (ADR-0024, ADR-0035)."
  - **"Parallel creation pipeline", stage table:**
    - stage 1 adds: "binary files are scanned by block headers and read in byte-range chunks; ASCII files and files with serendipity elements are read on rank 0 and scattered in the same chunks (ADR-0035)";
    - stage 3 adds: "skipped when Block honours a file's partition data (ADR-0035)".
  - **"Partitioning":**
    - Block row: "Also honours partition data stored in Gmsh files" becomes "Also honours partition data stored in Gmsh files whose partition count equals the rank count (ADR-0035)";
    - "Default choice" reads "… PT-Scotch when the `ptscotch` feature is enabled and the communicator has an MPI handle, otherwise METIS on rank 0 when `metis` is enabled, otherwise Block with a warning (ADR-0035)".
  - **"Gmsh MSH 4.1":**
    - "Parallel read" adds: "Files with serendipity elements are also read on rank 0. Errors keep their serial form (`UnmatchedTagElement` names the Gmsh element tag). Elementary tags travel with their cells and entities like physical tags (ADR-0035)."
    - "Parallel write" adds: "`GmshWriter::partitioned(true)` writes `<stem>_<rank+1>.msh`; non-vertex node tags and tag-element tags are numbered globally, so that every file and the gathered file agree, and one rank gives the serial numbering (ADR-0035)."
  - **"VTK":**
    - PVTU: "M5 writes the header and one piece on one rank; M8 adds more ranks and `vtkGhostType`." becomes "`vtkGhostType` is written with more than one rank; the header's `GhostLevel` is 1 if any rank has ghost cells (ADR-0035)."
    - VTKHDF adds: "`VtkWriter::vtkhdf`; one partition per rank with the PVTU piece's arrays; collective through `mpio` with an MPI communicator, otherwise gathered to rank 0; no compression (ADR-0035)."
    - Test oracles add: "VTKHDF files are compared with the PVTU through VTK's `vtkHDFReader` (ADR-0035)."
  - **"Tags, regions and submeshes", MeshTags:** add "The Gmsh reader also passes elementary tags through the same matching (ADR-0035)."
  - **"Testing and validation", I/O:** add "VTKHDF and PVTU output of the same mesh read back identically by VTK's Python reader."
  - **"Workspace layout":**
    - gridcl-io: dependencies read "…; flate2; hdf5-metno (optional, `mpio` with feature `mpi`); dev: mshio, serde_json, gridcl-testing";
    - gridcl-testing: purpose adds "the Python oracle helper and `gen-gmsh-box`"; dependencies add "serde_json";
    - gridcl-mpi-tests: dependencies add "nix (unix; getrusage)".
  - **"Cargo features":** "`hdf5` builds the VTKHDF writer with any HDF5 1.10.10–2.2.0; with `mpi` it enables hdf5-metno's `mpio` and needs a parallel HDF5 (`HDF5_DIR`) (ADR-0035)."
  - **"CI matrix":** the mpi-mpich and mpi-openmpi rows read "gridcl-mpi-tests on 1–4 ranks with the Python oracles, and the VTKHDF targets with Ubuntu's parallel HDF5 (`--oversubscribe` for Open MPI)".
  - **Roadmap M8:** acceptance adds "read as in ADR-0035 (bullet 1: item 14; bullet 2: item 13; bullet 3: item 15)".
- **Public API changes:**
  - gridcl: `GmshOptions::{partitioner, ghosts, ordering}` (`GmshOptions` loses `Send` and `Sync`), `GmshWriter::partitioned`, `VtkWriter::{origin_ids, vtkhdf}`, and hidden `Mesh::footprint_bytes`.
  - gridcl-io: `MshIndex`, `NodeChunk`, `ElementChunk`, `write_vtkhdf`, and the extra `ghost_level` parameter of `write_pvtu`, the one existing signature that changes.
  - The `NotYetImplemented("… (M8)")` errors of `gmsh::read`, `GmshWriter`, `vtu` and `pvtu` go away.
- **Acceptance tests affected:**
  - The M8 bullets read as in items 13–15. Their tests are those of docs/plans/M8.md.
  - `gmsh_read_needs_one_rank` and `vtk_write_needs_one_rank` are replaced by multi-rank tests.
  - The M4 and M5 tests are unchanged, apart from the callers of `write_pvtu`.
- **Later milestones affected:**
  - M9: submeshes may be written in parallel through the same writers.
  - M10: VTKHDF compression, a distributed serendipity upgrade and `Hilbert` as the parallel default can be revisited with benchmark data.

## Decision record

Accepted by the maintainer on 2026-11-02, as proposed (option 2, items 1–16), without conditions. Applied to docs/design.md in `adr: 0035 apply`.
