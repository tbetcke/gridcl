# ADR-0026: M6 public items and rules the design leaves open

- **Status:** Proposed
- **Date:** 2026-10-28
- **Raised in:** M6 planning (docs/plans/M6.md, "Planned public API and rules" and questions 1–17)
- **Design sections affected:**
  - "Architecture": Listing B3 (`Mesh` is no longer `Send`);
  - "API walkthrough": Listing B8 (`Mpi::world`, `local_gpu_index`);
  - "Device model and backends": the `GridError` paragraph; Listing S1 (`DeviceBuffer::write_host`); Listing S1b (`pack`, `unpack`);
  - "Topology construction on the device": "Ownership of vertices and entities" (`VertexSharers`, the hash, ghost order);
  - "Distributed mesh model":
    - Listing S2c;
    - "Transports" (HostStaged);
    - "Parallel creation pipeline": the stage table and the sentence before it;
  - "Partitioning": Listing S3 (`Block`);
  - "Tags, regions and submeshes" (the MeshTags bullet);
  - "Workspace layout": the crate table (gridcl-device, gridcl, gridcl-testing);
  - "Implementation roadmap": M6 (first acceptance bullet), M7 (scope), M10 (scope);
  - "Handing this design to Claude Code": Listing H1, the host↔device copy rule. CLAUDE.md changes the same way.

## Context

M6 is the parallel core. The roadmap gives its scope:

> - `Mpi` and the `mpi::ffi` wrappers (distributed graph, neighbourhood all-to-all-v);
> - `IndexMap` with ghosts, and `Scatterer` (host-staged, deterministic reverse); its pack and unpack kernels take their shape from `LaunchPlan` and are generic over the vector size (`KernelKind::Pack`);
> - pipeline stages 1–10 from host arrays with the block partitioner, ghost modes, ownership rules, global numbering and locality ordering.

Its first acceptance bullet requires "global entity counts, total volume and tag counts are identical to one rank" on 1–4 ranks.

The public items are Listing S2c (`CommDyn`, `IndexMap`, `Scatterer`), Listing S3 (`DistGraph`, `Partitioner`) and Listing B8. Earlier ADRs left the rest to M6:

- ADR-0009: ghosts, `scatterer` and the device field of `IndexMap`; construction of `Neighbourhood`; `MessageTooLarge`; reverse on `gather` and `segmented_reduce`;
- ADR-0011: the rank-0 identity of `Mpi::exscan_bytes`;
- ADR-0012: parallel numbering and entity ownership;
- ADR-0015: Pack and Scalar plans;
- ADR-0022: `GhostMode`, and ghost tag consistency "by forward scatter".

Planning read these against the code after M5:

- **Builder.**
  - `MeshBuilder` (`gridcl/src/builder.rs:118`) builds only `Serial` meshes. Its generic path `build_with_comm` (l.185) is crate-private and numbers serially.
  - `cell_block` takes u32 indices into the builder's own node array (l.150–157).
  - The origin is the centre of the local bounding box (l.259–270).
  - There is no tag input; only the serial Gmsh reader fills `TagRegistry`.
- **gridcl-parallel.**
  - `IndexMap` has only `new` (`index_map.rs:58`).
  - `Neighbourhood` holds two rank lists and nothing else (`comm.rs:276`).
  - The crate does not depend on cubecl, so it cannot name `Client` or `CubeElement`.
  - `CommExt` has `sum`, `max` and `exscan_sum` only.
- **gridcl-device.**
  - No pack or unpack kernel exists. `KernelKind::Pack` exists, at vector size 1 for every class.
  - `DeviceBuffer` cannot upload into an existing buffer, and nothing calls `Client::write`.
  - `MessageTooLarge` is absent.
- **gridcl-topology.**
  - Entity creation has no ownership step: `renumber` (`renumber.rs:125`) treats every entity as owned, and every map comes from `IndexMap::new`.
  - `try_clone` asserts ghost-free maps (`topology.rs:15`, `geometry.rs:157`).
- **gridcl-partition** is an empty stub.

The maintainer answered four scoping questions on 2026-10-28:

- tags for cells and lower dimensions, keyed by corner origin ids;
- M6 builds the full facet dual graph;
- `build()` goes through the same pipeline with `Serial`;
- gridcl-testing gets an in-process, thread-backed communicator (`ThreadComm`), which needs no ADR.

The design-reviewer's critique of the draft plan found problems 1 and 4.

## Problem

Each problem has the number of the item that answers it.

1. **No public API builds a distributed mesh from host arrays.** Listing B8 reads files, which is M8. `cell_block`'s u32 node indices cannot reference a node held in another rank's chunk. If every rank passes the nodes its own cells use, shared nodes appear on several ranks and fail the duplicate check. The pipeline's stage 1 assumes cells reference nodes by origin id.
2. **M6 and M7 overlap.** M6's "pipeline stages 1–10" include stage 2, the dual graph, while M7's scope lists "the distributed dual graph". The block partitioner needs only `vtxdist`, but ghost detection (`SharedFacet`) and reverse Cuthill–McKee need the same facet matching.
3. **"Tag counts" have no input.** No listing gives tags with host arrays.
4. **`Scatterer` cannot exchange.**
   - `forward` and `reverse` take no communicator (Listing S2c, B13), so the exchange must go through the `IndexMap`'s `Neighbourhood`, which holds only rank lists.
   - Under MPI a graph communicator could be stored, but a non-MPI communicator (the thread-backed test communicator) has nothing to store.
   - `create_entities` holds only `&dyn CommDyn`, which cannot be cloned into a map.
5. **Constructors are missing.** The listings give no constructor for `IndexMap` with ghosts, `Neighbourhood` or `Mpi` (Listing B8 calls `Mpi::world(&universe)`). `IndexMap` is not `Clone`, and the M3 follow-up on copying ghosted maps is open.
6. **Stage 7 conflicts with the serial cell order.** Stage 7 orders owned cells "along a Hilbert curve", and "Locality ordering happens before any numbering". `build()` must keep the input order, because ADR-0012 numbers serial meshes "by first touching cell, blocks in call order", ADR-0022 keeps "cells in file order within a block", and M3–M5 tests compare per-cell results in input order. Routing `build()` through stage 7 as written would reorder every serial mesh.
7. **Stage 6 has no well-defined source.** It fetches coordinates "from the stage-1 chunk owners". With arbitrary builder chunks, the chunk owner of a node id cannot be computed.
8. **No type carries the sharer sets.** Entity ownership needs per-vertex sharer sets (stage 8), and gridcl-topology's public API has no way to receive them.
9. **The hash is unspecified.** "hash(origin id) mod (number of candidates)" and "hash(key)" name no function. The owners must be identical on every platform and in M7 and M8.
10. **Ghost tag consistency.** The MeshTags bullet says "Ghost values are made consistent by a forward scatter". `MeshTags` is sparse: an index may repeat with distinct values. A forward scatter of a dense array cannot carry several values per index.
11. **"Total volume identical to one rank" cannot be bitwise.** `measure` sums owned cells per block in a fixed order, and the partition changes which cells each rank sums.
12. **gridcl-parallel needs cubecl types.** `Scatterer::forward<T: CubeElement>` and `IndexMap::scatterer(&Client, …)` need them, but the crate table gives gridcl-parallel no cubecl dependency. Dependency rule 1 allows cubecl only in crates that define or launch kernels.
13. **No upload into an existing buffer.** HostStaged "uploads the receive buffer" with `Client::write`, and `DeviceBuffer` has no method for that. Listing S1b names no pack or unpack primitive, although the crate table places "pack/unpack kernels" in gridcl-device.
14. **The copy rule does not cover entity ownership.** "Ownership of vertices and entities" runs the boundary-entity exchange "on the host". CLAUDE.md's list of allowed host↔device copies (Listing H1) does not name it.
15. **Errors inside collectives can deadlock.** If one rank finds an input error, such as a duplicate origin id, and returns, the others wait forever in the next collective. No `GridError` variant covers an error raised on another rank.
16. **Smaller gaps:**
    - the order of ghost vertices and nodes ("owned-first … in order of first touching cell" leaves ghosts open);
    - the variants of `ReverseOp` (only `Add` and `Insert` appear) and what they do with the owner's own value;
    - `SharedVertex { layers: 0 }`;
    - how cell blocks are agreed when a rank has no cells of some type;
    - `ScatterNumeric`'s bound and `Transport`'s visibility.
17. **`local_gpu_index` belongs to no milestone.** Listing B8 uses it, and dev-environment.md calls it a follow-up.

## Options

1. **Keep the design as written.**
   - Several M6 pieces cannot be built:
     - problem 1: the builder cannot express distributed input;
     - problem 4: the scatterer has nothing to exchange through;
     - problem 12: gridcl-parallel cannot name its own argument types;
     - problem 15: collective errors deadlock.
   - Routing `build()` through stage 7 breaks the M3–M5 suites (problem 6).
   - The volume bullet fails (problem 11).

   Not chosen.
2. **Fix the items below** (proposed).
3. **Put the graph communicator in `Neighbourhood` only, with no `Rc<dyn CommDyn>`** (problem 4). This keeps `CommDyn` unchanged and keeps `IndexMap` `Send` without the `mpi` feature. But the scatterer then works only on `Mpi`, the in-process multi-rank tests cannot cover it, and `Send` changes with a Cargo feature, which features must not do. Not chosen.
4. **A separate distributed builder type** (problem 1), for example `DistMeshBuilder`. This keeps `MeshBuilder` serial, but needs two numbering implementations that must agree, which the maintainer rejected (2026-10-28). Not chosen.
5. **Deliver ghost tags by a forward scatter of a dense first-value array** (problem 10). This loses repeated indices, and needs a second pass for them. Not chosen.

## Proposed decision

Option 2. Each item answers the problem of the same number. The maintainer chose items 1–3 and the single pipeline on 2026-10-28.

1. **Distributed builder (gridcl).**
   - `MeshBuilder<F, TD, GD>` gains these methods:
     - `cell_block_by_ids(self, element: CoordinateElement<TD>, node_ids: &[i64], origin_ids: Option<&[i64]>) -> Self`, whose cells reference nodes by node origin id;
     - `ghosts(self, mode: GhostMode) -> Self`, default `GhostMode::None`;
     - `partitioner(self, p: impl Partitioner + 'static) -> Self`, default `Block`;
     - `ordering(self, o: CellOrdering) -> Self`, default `CellOrdering::Input` (item 6);
     - `physical_tags` and `tag_names` (item 3);
     - `build_on<C: Comm>(self, comm: C) -> Result<Mesh<F, TD, GD, C>>`, which is collective.
   - `build()` is `build_on(Serial)`, and `build_with_comm` becomes the pipeline. The Gmsh reader keeps its vertex map.
   - `pub enum GhostMode { None, SharedFacet, SharedVertex { layers: usize } }` and `CellOrdering` live in gridcl. Both are `Clone, Copy, Debug, PartialEq, Eq`, and both are in the prelude.
   - **Input contract of `build_on`:**
     - Each rank passes a chunk of nodes, and every node appears on exactly one rank. A duplicate gives `DuplicateOriginId`.
     - Each rank passes a chunk of cells. With `cell_block` the u32 indices refer to the rank's own nodes and are translated to ids on that rank; this is always valid on one rank. With `cell_block_by_ids` a cell may reference any rank's nodes.
     - Default origin ids are global construction indices: an exscan of the counts plus the local index. This is today's serial default.
     - Nodes that no cell uses are dropped. `NodeRoleMismatch` is checked globally (item 7).
     - Cell blocks are agreed as in item 16.
2. **Stage 2 in M6.**
   - M6 builds the full facet dual graph: facet keys (sorted vertex origin ids) are hashed to ranks by `owner_hash` and matched, which gives `DistGraph { vtxdist, xadj, adjncy, vwgt: None }`.
   - Ghost detection and reverse Cuthill–McKee use the same rendezvous.
   - `pub struct Block;` (gridcl-partition; `Clone, Copy, Debug, Default`) implements `Partitioner`. It sends global graph vertex i to rank ⌊i · nparts / N⌋, where N = `vtxdist[nranks]`.
   - M7 keeps the vertex weights, the external partitioners and the partition-quality acceptance.
3. **Tags from host arrays.**
   - `MeshBuilder::physical_tags(self, entity: CellType, corners: &[i64], values: &[i32]) -> Self` gives one entry per entity, with `entity.num_vertices()` corner origin ids per entry. This is Gmsh's key, so a tag of any dimension, cells included, is matched by its sorted corner ids.
   - `MeshBuilder::tag_names(self, names: &[(&str, usize, i32)]) -> Self` gives (name, dimension, value).
   - Cell tags are matched to their cells in stage 1, by a rendezvous on the sorted corner key, and travel with the cell through migration and ghosting.
   - Tags of lower dimension are matched after `create_entities(dim)`, which the build calls for every tagged dimension, as the Gmsh reader does. Matching works like this:
     1. the key goes to its rendezvous rank;
     2. that rank forwards the tag to every rank that holds all its corner vertices, found through the stage-8 vertex directory;
     3. each of those ranks matches the tag with `Topology::match_entities`.
   - An entry that matches nothing gives `UnmatchedTagElement { dim, tag }` on every rank (item 15), where `tag` is the entry's global input position: an exscan of the entry counts plus the local index.
   - Only physical tags; elementary tags remain the Gmsh reader's.
   - Group sizes stay local counts, ghosts included, as in ADR-0022.
4. **`Neighbourhood` holds its communicator.**
   - `CommDyn` gains `fn duplicate(&self) -> Rc<dyn CommDyn>`, a cheap, object-safe clone that refers to the same communicator: for `Mpi` an `Rc` clone, not `MPI_Comm_dup`.
   - `Neighbourhood` holds `Option<Rc<dyn CommDyn>>`, which is `None` for `Neighbourhood::empty()`, plus, under `mpi`, an `Rc` of the graph communicator.
   - `Scatterer` exchanges through `nb.comm().neighbour_all_to_all_v_bytes(nb, …)`.
   - Consequence: `IndexMap`, `Topology`, `Geometry` and every `Mesh` are not `Send`, whatever the communicator and with or without `mpi`. Nothing in the workspace requires `Send`. Listing B3's sentence "`Mesh<…, Mpi>` is deliberately not `Send`" is generalised.
5. **Constructors and `Clone`.**
   - `impl Mpi { pub fn world(universe: &mpi::environment::Universe) -> Mpi; pub fn new(comm: mpi::topology::SimpleCommunicator) -> Mpi; }`, under feature `mpi`.
   - `Neighbourhood::new(comm: &dyn CommDyn, sources: Vec<usize>, destinations: Vec<usize>) -> Result<Neighbourhood>` is collective. When `comm.raw()` is `Some`, it creates the graph communicator with `MPI_Dist_graph_create_adjacent` (unweighted, `reorder = 0`). Otherwise it keeps the rank lists. `Serial` accepts only empty lists.
   - `IndexMap::with_ghosts(comm: &dyn CommDyn, num_owned: usize, ghosts: Vec<i64>, ghost_owners: Vec<i32>) -> Result<IndexMap>` is collective:
     - ghosts request their indices from their owners, and owners range-check them (`IndexOutOfRange`, agreed);
     - destinations are discovered by `all_to_all_v`;
     - the map keeps a forward and a reverse (transposed) `Neighbourhood`, and host send lists ordered by (neighbour rank, ghost position on the receiver).
   - `IndexMap: Clone`. A clone shares the neighbourhoods and does not copy the device cache.
   - `CommExt` gains:
     - `min`;
     - `all_gather<T: CommScalar>(&self, x: T) -> Vec<T>`;
     - `all_to_all_v<T: CommScalar>(&self, send: &[T], send_counts: &[usize]) -> Result<(Vec<T>, Vec<usize>)>`, which exchanges the counts first;
     - `neighbour_all_to_all_v<T: CommScalar>(&self, nb: &Neighbourhood, send: &[T], send_counts: &[usize]) -> Result<(Vec<T>, Vec<usize>)>`.
   - gridcl re-exports `Mpi` (feature `mpi`), `Neighbourhood`, `DType` and `CommScalar`.
6. **Cell ordering.**
   - `pub enum CellOrdering { Input, Hilbert, ReverseCuthillMcKee }`, with default `Input`. Stage 7 applies the chosen order to owned cells:
     - `Input`: source rank, then input position;
     - `Hilbert`: vertex centroids, quantised to 2^21 per axis in the global bounding box, a 64-bit Hilbert key, ties broken by cell origin id;
     - `ReverseCuthillMcKee`: on the local facet graph of owned cells, from a minimum-degree cell, ties broken by origin id.
   - Ghost cells always follow, ordered by (owner rank, origin id).
   - Locality ordering is therefore opt-in. M8's parallel reader chooses its own default. Ordering still happens before any numbering.
7. **Stage 6 by rendezvous.**
   - Every node (origin id, coordinates, vertex or non-vertex role) is posted to rank `owner_hash(&[id]) mod P`. Ranks fetch the nodes of their owned and ghost cells from there.
   - The directory also finds duplicate node ids and role mismatches. The global origin is agreed from `CommExt::min` and `CommExt::max` of the bounding box.
8. **Sharer sets.**
   - `pub struct VertexSharers { pub offsets: Vec<usize>, pub ranks: Vec<i32>, pub owned_cell: Vec<bool> }` (gridcl-topology) is a host CSR over local vertices:
     - row v lists every rank that holds vertex v, this one included, in ascending order;
     - `owned_cell` says whether that rank holds v in an owned cell.
   - `Topology::set_vertex_sharers(&mut self, s: VertexSharers)` stores it, uploads one shared flag per vertex, and derives the sharing neighbourhood: every rank that shares a vertex with this one.
   - A topology without sharers, which is every serial mesh, treats every vertex as unshared.
9. **`owner_hash`.**
   - `pub fn owner_hash(key: &[i64]) -> u64` (gridcl-parallel) starts at h = 0x6772_6964_636C_0006 and, for each word w, sets h = mix(h ^ (w as u64)).
   - mix is SplitMix64's finaliser: z += 0x9E37_79B9_7F4A_7C15; z = (z ^ (z >> 30)) · 0xBF58_476D_1CE4_E5B9; z = (z ^ (z >> 27)) · 0x94D0_49BB_1331_11EB; z ^ (z >> 31), with wrapping arithmetic.
   - It is used as follows:
     - the vertex and node owner is the candidate at position `owner_hash(&[origin id]) mod n`;
     - the entity owner is at `owner_hash(sorted origin ids) mod n`;
     - the rendezvous rank of any key is `owner_hash(key) mod P`.
   - Golden values are tested. M7 and M8 use the same function.
10. **Ghost tags by construction.** Every rank that holds an entity, owner or ghost, receives the tag entries of its key (item 3), so ghost values equal the owner's values without a forward scatter. A multi-rank test checks this. ADR-0022's consequence "M6 makes ghost tag values consistent by forward scatter" is replaced.
11. **Volume to tolerance.** The first M6 bullet compares total volume to the precision tolerances, 10^−12 relative in f64 and 10^−5 in f32. Every other quantity in that bullet is compared exactly.
12. **cubecl types through gridcl-device.**
    - gridcl-device re-exports `cubecl::prelude::{Client, CubeElement}`, and gridcl-parallel uses them from there. The crate graph does not change.
    - gridcl-parallel defines no kernel. Every device step of `Scatterer` is a gridcl-device primitive (item 13).
13. **Upload and pack primitives (gridcl-device).**
    - `DeviceBuffer::write_host(&mut self, data: &[T]) -> Result<()>` overwrites the buffer through `Client::write`. A length different from `len()` gives `LengthMismatch`.
    - Listing S1b gains:
      ```rust
      /// out[k·block .. (k+1)·block] = src[idx[k]·block .. (idx[k]+1)·block], as u32 words.
      pub fn pack<T: CubeElement>(src: &DeviceBuffer<T>, idx: &DeviceBuffer<u32>, block: usize,
                                  out: &mut DeviceBuffer<T>) -> Result<()>;
      /// dst[idx[k]·block ..] = packed[k·block ..]; indices distinct (debug check, as scatter).
      pub fn unpack<T: CubeElement>(packed: &DeviceBuffer<T>, idx: &DeviceBuffer<u32>, block: usize,
                                    dst: &mut DeviceBuffer<T>) -> Result<()>;
      ```
      Both:
      - plan as `KernelKind::Pack` and are generic over the vector size, with lanes over output elements (pack) or input elements (unpack);
      - assert at compile time that `size_of::<T>()` is a multiple of 4, as `copy` does;
      - never launch on an empty index list.
    - `Scatterer::forward` is `pack`, then `to_host`, then the neighbourhood exchange, then `write_host`, then `unpack`.
    - `Scatterer::reverse` runs these steps:
      1. pack the ghost values and exchange them;
      2. build [owner values `gather`ed by `rev_targets` | received values, `unpack`ed];
      3. `gather` that buffer by `rev_order`;
      4. `segmented_reduce` over `rev_segments`;
      5. `scatter` the result to the targets.
14. **Host↔device copies.** CLAUDE.md and Listing H1 add three items to the allowed copies:
    - "the boundary-entity ownership exchange of entity creation (the candidate count and keys to the host; owners and global numbers back)";
    - "the upload of an IndexMap's send and receive index lists when its scatterer is first created";
    - "the vertex shared flags at mesh creation".

    Tag matching at mesh creation counts as I/O, as in the Gmsh reader.
15. **Agreed errors.**
    - `GridError` gains `RemoteFailure { rank: usize }` and `MessageTooLarge { count: usize }`.
    - Every pipeline stage, and every collective constructor (`IndexMap::with_ghosts`, `Neighbourhood::new`, `create_entities` on more than one rank), ends with an agreed status: an allreduce of the lowest failing rank.
    - A rank with a local error returns that error. Every other rank returns `RemoteFailure { rank }`.
    - `MessageTooLarge` is returned when an MPI count or displacement exceeds `i32::MAX`.
16. **Smaller rules.**
    - **Ghost order:** ghost vertices and nodes are ordered by (owner rank, global index), like ghost entities. Owned ones keep "first touching cell" order.
    - **`ReverseOp`:** `pub enum ReverseOp { Add, Min, Max, Insert }`.
      - `Add`, `Min` and `Max` combine the owner's value with every ghost contribution in a fixed order.
      - `Insert` replaces the owner's value with the contribution of the lowest neighbour rank, and leaves it unchanged without contributions.
      - Ghost values are unchanged by `reverse`.
      - Empty segments use the ADR-0011 identities.
    - **`ScatterNumeric`:** `pub trait ScatterNumeric: DeviceScalar {}`, implemented for f32, f64, i32 and u32.
    - **`Transport`:** `pub enum Transport { HostStaged, GpuAware }` and `Scatterer::transport(&self) -> Transport`. M6 always chooses `HostStaged`; M10 adds `GpuAware`.
    - **`SharedVertex { layers: 0 }`** behaves as `None`. `SharedVertex { layers: k }` adds k rounds of vertex neighbours. `SharedFacet` adds the cells of other ranks that share a facet with an owned cell.
    - **Cell blocks:** the union over ranks, in order of first declaration by rank (rank 0's list first, then new elements from rank 1, …). A rank without cells of a block gets it empty. One cell type in two degrees gives `DuplicateCellBlock`.
17. **`local_gpu_index`** moves to M10, together with the GPU-aware tests, the first that need one GPU per rank. Until then every rank opens GPU 0.

## Consequences

- **Text to change in docs/design.md:**
  - **Listing B3:** the sentence after it reads "… `Mesh` is deliberately not `Send` (its index maps hold the communicator, ADR-0026): gridcl runs one host thread per rank …".
  - **Listing B8:** add the comment "`local_gpu_index` arrives with M10 (ADR-0026)".
  - **The `GridError` paragraph:** add "M6 adds `MessageTooLarge { count: usize }` and `RemoteFailure { rank: usize }`; collective operations agree on errors, and ranks without a local error return `RemoteFailure` with the lowest failing rank (ADR-0026)."
  - **Listing S1:** add `write_host`.
  - **Listing S1b:** add `pack` and `unpack` as in item 13.
  - **The paragraph after S1b:** add "gridcl-device re-exports `Client` and `CubeElement` for gridcl-parallel".
  - **"Ownership of vertices and entities":**
    - name `owner_hash` and its definition (item 9);
    - add the ghost order of item 16;
    - add `VertexSharers` and `Topology::set_vertex_sharers` (item 8).
  - **Listing S2c:**
    - `CommDyn::duplicate`;
    - the `CommExt` additions;
    - `Mpi::{world, new}`;
    - `Neighbourhood::new`, and the comment "holds the parent communicator and, under `mpi`, the graph communicator";
    - `IndexMap::with_ghosts` and `Clone`;
    - `ReverseOp`, `ScatterNumeric` and `Transport` as in item 16;
    - `owner_hash`.
  - **"Transports", HostStaged:** "`Client::write`" becomes "`DeviceBuffer::write_host` (`Client::write`)".
  - **"Parallel creation pipeline":**
    - the sentence "Locality ordering happens before any numbering" adds "; it is chosen by `CellOrdering`, default `Input` (ADR-0026)";
    - the stage table changes:
      - stage 1: "Builder chunks (`MeshBuilder::build_on`) or file chunks; cells reference nodes by origin id; global duplicate checks and cell-tag matching by rendezvous";
      - stage 2: adds "(M6; M7 adds vertex weights)";
      - stage 6: "from the rendezvous directory `owner_hash(id) mod P`";
      - stage 7: "Owned cells in `CellOrdering`: input order (default), Hilbert curve of their centroids, or reverse Cuthill–McKee; ghost cells after, ordered by (owner rank, origin id)";
      - stage 8: adds "ghosts by (owner rank, global index)".
  - **Listing S3:** add `pub struct Block;`.
  - **"Tags, regions and submeshes":** "Ghost values are made consistent by a forward scatter" becomes "Every holder of an entity, owner or ghost, receives its tags by key matching, so ghost values equal the owner's (ADR-0026). `MeshBuilder::physical_tags` and `tag_names` give tags with host arrays."
  - **Crate table:**
    - gridcl-device purpose adds "re-exports of `Client` and `CubeElement`";
    - gridcl purpose adds "distributed builder (`build_on`, `GhostMode`, `CellOrdering`)";
    - gridcl-testing purpose adds "`ThreadComm`, an in-process multi-rank communicator".
  - **Roadmap:**
    - M6's first bullet reads "global entity counts and tag counts are identical to one rank, and total volume agrees to 10^−12 (f64) and 10^−5 (f32) relative (ADR-0026)";
    - M7's scope reads "vertex weights on M6's distributed dual graph, ptscotch-sys …";
    - M10's scope adds "`local_gpu_index`".
  - **Listing H1:** the host↔device copy rule adds the three items of item 14.
- **CLAUDE.md:** the same change to the host↔device copy rule.
- **Public API changes:** the items above, in gridcl-device, gridcl-parallel, gridcl-partition, gridcl-topology and gridcl. The only existing signatures that change are these:
  - `CommDyn` gains a required method, so every implementor changes: `Serial`, the test fakes and `TwoRanks`;
  - `IndexMap`, `Topology`, `Geometry` and `Mesh` lose `Send`.
- **Acceptance tests affected:**
  - M6 bullet 1 as in item 11;
  - no test of M0–M5 changes. `build()` must reproduce today's numbering, and T12 of the M6 plan checks this with the full suite.
- **Later milestones affected:**
  - M7: weights on M6's graph; `owner_hash` reused.
  - M8: parallel Gmsh reading feeds stage 1 with file chunks and chooses its default `CellOrdering`. `GmshOptions::ghosts` takes `GhostMode`. Tags from files use item 3's matching.
  - M9: submeshes inherit ghosts through cloned `IndexMap`s.
  - M10: `GpuAware` transport and `local_gpu_index`.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
