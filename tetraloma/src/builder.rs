// Design: "Parallel creation pipeline"; ADR-0012, ADR-0026.
//! `MeshBuilder`: a `Mesh` from host arrays with origin ids, serial or distributed, through the
//! parallel creation pipeline.

use std::collections::HashMap;

use cubecl::prelude::Client;
use tetraloma_device::{GridError, Real, Result, check_precision};
use tetraloma_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use tetraloma_geometry::{CoordinateElement, NodeFamily};
use tetraloma_parallel::{Comm, CommDyn, CommExt, Serial};
use tetraloma_partition::{Block, DistGraph, Partitioner};
use tetraloma_refcell::RefCellError;

use crate::Mesh;
use crate::pipeline::dual_graph::{check_destinations, dual_graph, partition, vertices_only};
use crate::pipeline::ghosts::ghost_cells;
use crate::pipeline::input::{self, Input, NodeRefs, Read, TagInput, TagKind};
use crate::pipeline::migrate::migrate;
use crate::pipeline::nodes::fetch_nodes;
use crate::pipeline::number::number;
use crate::pipeline::order::order_cells;
use crate::pipeline::tags::{TagInputs, cell_elementary, cell_tags, registry};
use crate::pipeline::upload::upload;
use crate::pipeline::{BlockSpec, agree, log_memory};

/// The name of a cell type in errors.
pub(crate) fn cell_name(cell: CellType) -> &'static str {
    match cell {
        CellType::Point => "Point",
        CellType::Interval => "Interval",
        CellType::Triangle => "Triangle",
        CellType::Quadrilateral => "Quadrilateral",
        CellType::Tetrahedron => "Tetrahedron",
        CellType::Hexahedron => "Hexahedron",
        CellType::Prism => "Prism",
        CellType::Pyramid => "Pyramid",
    }
}

/// `GridError` of an element tetraloma-refcell does not offer.
pub(crate) fn element_error(e: RefCellError) -> GridError {
    match e {
        RefCellError::UnsupportedNodeFamily { cell, family } => GridError::UnsupportedNodeFamily {
            cell: cell_name(cell),
            family: match family {
                NodeFamily::Equispaced => "Equispaced",
                NodeFamily::Gll => "Gll",
            },
        },
        RefCellError::UnsupportedDegree { cell, degree } => GridError::UnsupportedDegree {
            cell: cell_name(cell),
            degree,
        },
        other => unreachable!("lagrange_points returned {other}"),
    }
}

/// `ids`, or `0..n` offset by `first`, after checking the length.
pub(crate) fn ids_or_index(ids: Option<&[i64]>, n: usize, first: usize) -> Result<Vec<i64>> {
    match ids {
        Some(ids) if ids.len() != n => Err(GridError::LengthMismatch {
            expected: n,
            found: ids.len(),
        }),
        Some(ids) => Ok(ids.to_vec()),
        None => (first..first + n)
            .map(|i| i64::try_from(i).map_err(|_| GridError::TooLarge))
            .collect(),
    }
}

// Design: "Distributed mesh model"; ADR-0026.
/// The ghost cells each rank holds besides its owned cells: copies of cells that other ranks own.
///
/// Ghost cells let a rank compute on the neighbourhood of its owned cells without asking other
/// ranks. They come after the owned cells in each cell block. [Ghost modes] describes the modes,
/// and [Ownership and ghosts] explains ownership.
///
/// [Ghost modes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#ghost-modes
/// [Ownership and ghosts]: https://tetracl.github.io/tetraloma/concepts/meshes.html#ownership-and-ghosts
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GhostMode {
    /// No ghost cells.
    None,
    /// The cells of other ranks that share a facet with an owned cell.
    SharedFacet,
    /// `layers` rounds of cells of other ranks that share a vertex; `layers: 0` is `None`.
    SharedVertex {
        /// The number of rounds.
        layers: usize,
    },
}

// Design: "Parallel creation pipeline", stage 7; ADR-0026, item 6.
/// The order of the owned cells of each rank. Ghost cells follow, ordered by owner rank and then by
/// origin id.
///
/// The order is the order in which kernels visit the cells, and vertices and nodes are numbered in
/// order of the first cell that uses them. Ties are broken by origin id, so every ordering is
/// deterministic. [Cell ordering] describes the orderings.
///
/// [Cell ordering]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#cell-ordering
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellOrdering {
    /// Source rank, then input position (the default).
    Input,
    /// A Hilbert curve of the vertex centroids, ties broken by cell origin id.
    Hilbert,
    /// Reverse Cuthill–McKee on the local facet graph, from a minimum-degree cell, ties broken
    /// by origin id.
    ReverseCuthillMcKee,
}

/// A built mesh and the local vertex index of each input node (`u32::MAX` for nodes that are
/// no cell's vertex).
pub(crate) type BuiltMesh<F, TD, GD, C> = (Mesh<F, TD, GD, C>, Vec<u32>);

/// One `cell_block` or `cell_block_by_ids` call: element, cell → node lists and optional cell
/// ids; the tag values of its cells from [`tagged_cell_block`](MeshBuilder::tagged_cell_block).
struct BlockInput<TD: TopoDim> {
    element: CoordinateElement<TD>,
    nodes: NodeRefs,
    origin_ids: Option<Vec<i64>>,
    physical: Vec<(u32, i32)>,
    elementary: Option<Vec<i32>>,
}

// Design: "API walkthrough", "Parallel creation pipeline"; ADR-0012, ADR-0026.
/// Builds a [`Mesh`] from arrays on the host.
///
/// - [`nodes`](MeshBuilder::nodes) sets the node coordinates, `[node][gdim]` in f64, with optional
///   origin ids (by default a node's global index: its index plus the number of nodes on lower
///   ranks).
/// - [`cell_block`](MeshBuilder::cell_block) adds a block of cells of one cell type, one row of
///   node indices per cell in tetraloma's node order, with optional cell origin ids (by default the
///   cell's global index over all blocks, in call order). A cell's vertices are its first nodes,
///   and a vertex's origin id is its node's. [`cell_block_by_ids`](MeshBuilder::cell_block_by_ids)
///   names the nodes by origin id instead.
/// - [`build`](MeshBuilder::build) builds a serial mesh, and [`build_on`](MeshBuilder::build_on) a
///   distributed one.
///
/// Both check the input, partition and order the cells, number the vertices and nodes in order of
/// the first cell that uses them (blocks in the order of their first declaration), store the
/// coordinates relative to the centre of the global bounding box, and upload the mesh to the
/// device. Nodes that no cell uses are dropped. [Building a mesh] shows the builder step by step,
/// and [Lagrange nodes] gives the node order of every cell type.
///
/// [Building a mesh]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html
/// [Lagrange nodes]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-ordering
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // Two triangles of degree 1 that make up the unit square.
/// let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
/// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
/// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
///     .nodes(&coords, Some(&[10, 11, 12, 13]))
///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
///     .build()?;
/// assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 2);
/// assert_eq!(mesh.origin(), [0.5, 0.5]);
/// assert_eq!(mesh.read_node(3)?, [1.0, 1.0]);
/// # Ok::<(), GridError>(())
/// ```
pub struct MeshBuilder<F: Real, TD: TopoDim, GD: Dim> {
    client: Client,
    coords: Vec<f64>,
    node_ids: Option<Vec<i64>>,
    blocks: Vec<BlockInput<TD>>,
    tags: Vec<TagInput>,
    tag_names: Vec<(String, usize, i32)>,
    element_ids: Vec<i64>,
    node_refs: Vec<i64>,
    destinations: Option<Vec<i32>>,
    ghosts: GhostMode,
    partitioner: Box<dyn Partitioner>,
    ordering: CellOrdering,
    _f: core::marker::PhantomData<(F, GD)>,
}

impl<F, TD, GD> MeshBuilder<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    /// An empty builder for a mesh on the device of `client`.
    ///
    /// The type parameters fix the mesh: the scalar `F` and the dimensions `TD` and `GD`. Rust
    /// often infers them from the type of the built mesh. [From arrays] shows the steps that
    /// follow.
    ///
    /// [From arrays]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#from-arrays
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let builder = MeshBuilder::<f32, D1, D1>::new(&client);
    /// // The interval [0, 2] as two cells.
    /// let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh = builder.nodes(&[0.0, 1.0, 2.0], None).cell_block(seg, &[0, 1, 1, 2], None).build()?;
    /// assert!((mesh.measure(1)? - 2.0).abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn new(client: &Client) -> Self {
        MeshBuilder {
            client: client.clone(),
            coords: Vec::new(),
            node_ids: None,
            blocks: Vec::new(),
            tags: Vec::new(),
            tag_names: Vec::new(),
            element_ids: Vec::new(),
            node_refs: Vec::new(),
            destinations: None,
            ghosts: GhostMode::None,
            partitioner: Box::new(Block),
            ordering: CellOrdering::Input,
            _f: core::marker::PhantomData,
        }
    }

    /// Sets the node coordinates, `[node][gdim]` in f64, and optionally one origin id per node.
    ///
    /// The coordinates are f64 whatever the scalar `F` of the mesh. Without origin ids, a node's id
    /// is its global index: its index plus the number of nodes on lower ranks. A second call
    /// replaces the first. [From arrays] explains the arguments, and [Origin ids] what the ids are
    /// for.
    ///
    /// [From arrays]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#from-arrays
    /// [Origin ids]: https://tetracl.github.io/tetraloma/concepts/meshes.html#origin-ids
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // One triangle, corners (0, 0), (2, 0) and (0, 2), node origin ids 10, 20 and 30.
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&[0.0, 0.0, 2.0, 0.0, 0.0, 2.0], Some(&[10, 20, 30]))
    ///     .cell_block(tri, &[0, 1, 2], None)
    ///     .build()?;
    /// // The mesh stores coordinates relative to the centre of the bounding box; `read_node` adds
    /// // it back.
    /// assert_eq!(mesh.origin(), [1.0, 1.0]);
    /// assert_eq!(mesh.read_node(1)?, [2.0, 0.0]);
    ///
    /// // Five coordinates are not a whole number of nodes in 2D.
    /// let r = MeshBuilder::<f32, D2, D2>::new(&client)
    ///     .nodes(&[0.0, 0.0, 2.0, 0.0, 0.0], None)
    ///     .cell_block(tri, &[0, 1, 2], None)
    ///     .build();
    /// assert!(matches!(r, Err(GridError::LengthMismatch { expected: 6, found: 5 })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn nodes(mut self, coords: &[f64], origin_ids: Option<&[i64]>) -> Self {
        self.coords = coords.to_vec();
        self.node_ids = origin_ids.map(<[i64]>::to_vec);
        self
    }

    /// Adds a block of cells of one cell type: `nodes` holds one row of node indices per cell, as
    /// many as the element has nodes, in tetraloma's node order; `origin_ids` optionally holds one
    /// id per cell.
    ///
    /// The element fixes the cell type, the polynomial degree of the geometry and its node family.
    /// Each cell type has at most one block, and the blocks keep the order of the calls. [Several
    /// cell types] and [Higher-degree geometry] show it, and [Lagrange nodes] gives the node order
    /// of every cell type.
    ///
    /// [Several cell types]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#several-cell-types
    /// [Higher-degree geometry]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#higher-degree-geometry
    /// [Lagrange nodes]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-ordering
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // The rectangle [0, 2] × [0, 1]: a quadrilateral on [0, 1] × [0, 1] and two triangles.
    /// let quad = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 1, family: NodeFamily::Equispaced };
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, ..quad };
    /// let coords = [0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&coords, None)
    ///     // The quadrilateral's corners in tensor order: (0, 0), (1, 0), (0, 1), (1, 1).
    ///     .cell_block(quad, &[0, 1, 3, 4], None)
    ///     .cell_block(tri, &[1, 2, 5, 1, 5, 4], None)
    ///     .build()?;
    /// assert_eq!(mesh.cell_blocks().len(), 2);
    /// assert_eq!(mesh.cell_blocks()[1].cell, Cell2::Triangle);
    /// assert!((mesh.measure(1)? - 2.0).abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn cell_block(
        mut self,
        element: CoordinateElement<TD>,
        nodes: &[u32],
        origin_ids: Option<&[i64]>,
    ) -> Self {
        self.blocks.push(BlockInput {
            element,
            nodes: NodeRefs::Indices(nodes.to_vec()),
            origin_ids: origin_ids.map(<[i64]>::to_vec),
            physical: Vec::new(),
            elementary: None,
        });
        self
    }

    // ADR-0026 (the input contract of a distributed build).
    /// Adds a block of cells that name their nodes by origin id: `node_ids` holds one row of node
    /// origin ids per cell, in tetraloma's node order; `origin_ids` optionally holds one id per
    /// cell.
    ///
    /// On a distributed build a cell may name nodes that another rank passed, so each rank can pass
    /// any share of the nodes and cells. [Building on a communicator] shows it on several
    /// processes.
    ///
    /// [Building on a communicator]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#building-on-a-communicator
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // The interval [0, 2] as two cells; the nodes have the origin ids 10, 11 and 12.
    /// let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: Mesh<f32, D1, D1> = MeshBuilder::new(&client)
    ///     .nodes(&[0.0, 1.0, 2.0], Some(&[10, 11, 12]))
    ///     .cell_block_by_ids(seg, &[10, 11, 11, 12], None)
    ///     .build()?;
    /// assert!((mesh.measure(1)? - 2.0).abs() < 1e-6);
    ///
    /// // No node has the id 12 here.
    /// let r = MeshBuilder::<f32, D1, D1>::new(&client)
    ///     .nodes(&[0.0, 1.0], Some(&[10, 11]))
    ///     .cell_block_by_ids(seg, &[10, 12], None)
    ///     .build();
    /// assert!(matches!(r, Err(GridError::IndexOutOfRange { .. })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn cell_block_by_ids(
        mut self,
        element: CoordinateElement<TD>,
        node_ids: &[i64],
        origin_ids: Option<&[i64]>,
    ) -> Self {
        self.blocks.push(BlockInput {
            element,
            nodes: NodeRefs::Ids(node_ids.to_vec()),
            origin_ids: origin_ids.map(<[i64]>::to_vec),
            physical: Vec::new(),
            elementary: None,
        });
        self
    }

    /// The Gmsh reader's cell block (ADR-0035, item 8): [`cell_block_by_ids`] with cell ids,
    /// whose cells carry the physical values `physical` ((cell index in the block, value)
    /// entries) and the elementary tags `elementary` (one per cell) from stage 1, without a
    /// keyed rendezvous. Either every block of every rank has elementary tags or none has.
    ///
    /// [`cell_block_by_ids`]: MeshBuilder::cell_block_by_ids
    pub(crate) fn tagged_cell_block(
        mut self,
        element: CoordinateElement<TD>,
        node_ids: Vec<i64>,
        origin_ids: Vec<i64>,
        physical: Vec<(u32, i32)>,
        elementary: Vec<i32>,
    ) -> Self {
        self.blocks.push(BlockInput {
            element,
            nodes: NodeRefs::Ids(node_ids),
            origin_ids: Some(origin_ids),
            physical,
            elementary: Some(elementary),
        });
        self
    }

    /// Tags of the Gmsh reader of dimension below the cells' (ADR-0035, items 8 and 9): as
    /// [`physical_tags`](MeshBuilder::physical_tags), of `kind`, and with `error_ids` (one per
    /// entry, the Gmsh element tags) as the `tag` of `UnmatchedTagElement`.
    pub(crate) fn tag_entries(
        mut self,
        kind: TagKind,
        entity: CellType,
        corners: Vec<i64>,
        values: Vec<i32>,
        error_ids: Vec<i64>,
    ) -> Self {
        self.tags.push(TagInput {
            entity,
            corners,
            values,
            kind,
            error_ids: Some(error_ids),
        });
        self
    }

    /// Ids that share the cell ids' space (the Gmsh reader's tag-element tags; ADR-0035,
    /// item 9): an id given twice, or also as a cell id, is `DuplicateOriginId`.
    pub(crate) fn element_ids(mut self, ids: Vec<i64>) -> Self {
        self.element_ids.extend(ids);
        self
    }

    /// Node ids that must be in some rank's nodes (the Gmsh reader's tag-element corners;
    /// ADR-0035, item 9): an id no rank passed is `IndexOutOfRange`, as a cell's node id.
    pub(crate) fn node_refs(mut self, ids: Vec<i64>) -> Self {
        self.node_refs.extend(ids);
        self
    }

    /// The destination rank of every input cell, blocks in call order (the partition data of a
    /// Gmsh file; ADR-0035, item 10): stages 2 and 3 are skipped. If any rank gives them,
    /// every rank must, one per cell.
    pub(crate) fn destinations(mut self, dest: Vec<i32>) -> Self {
        self.destinations = Some(dest);
        self
    }

    /// Sets the ghost cells each rank holds besides its owned cells; the default is
    /// [`GhostMode::None`].
    ///
    /// On one rank there are no ghost cells. [Ghost modes] describes the modes.
    ///
    /// [Ghost modes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#ghost-modes
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    /// // ThreadComm: a test communicator that runs each rank as a thread of this process; a
    /// // program passes an `Mpi` communicator instead.
    /// use tetraloma_testing::ThreadComm;
    ///
    /// // Two ranks, one interval each, sharing the node with id 1.
    /// let ghosts = ThreadComm::run(2, |comm| -> Result<usize> {
    ///     // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    ///     # let client = tetraloma_testing::test_client();
    ///     let r = comm.rank() as i64;
    ///     let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    ///     let (coords, ids): (&[f64], &[i64]) = if r == 0 { (&[0.0, 1.0], &[0, 1]) } else { (&[2.0], &[2]) };
    ///     let mesh: Mesh<f32, D1, D1, ThreadComm> = MeshBuilder::new(&client)
    ///         .nodes(coords, Some(ids))
    ///         .cell_block_by_ids(seg, &[r, r + 1], None)
    ///         .ghosts(GhostMode::SharedFacet)
    ///         .build_on(comm)?;
    ///     // The ghost cell does not count twice.
    ///     assert!((mesh.measure(1)? - 2.0).abs() < 1e-6);
    ///     Ok(mesh.cell_blocks()[0].index_map.num_ghosts())
    /// });
    /// // Each rank holds the other rank's cell as a ghost.
    /// assert!(ghosts.iter().all(|g| g.as_ref().ok() == Some(&1)));
    /// ```
    pub fn ghosts(mut self, mode: GhostMode) -> Self {
        self.ghosts = mode;
        self
    }

    // Design: "Partitioning"; ADR-0029, ADR-0033.
    /// Sets the partitioner that assigns the cells to ranks on a distributed build; the default is
    /// [`Block`].
    ///
    /// On one rank the partitioner is not called. The external partitioners need features: `Metis`
    /// and `Scotch` partition on rank 0, and `PtScotch` and `ParMetis` on all ranks through the
    /// communicator's MPI handle (their features imply `mpi`). The features `metis` and `parmetis`
    /// exclude each other. [Partitioners] compares them, and [Cargo features] lists their features
    /// and system libraries.
    ///
    /// [Partitioners]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
    /// [Cargo features]: https://tetracl.github.io/tetraloma/features.html#partitioners
    ///
    /// # Examples
    ///
    /// METIS on rank 0, with the feature `metis`:
    ///
    /// ```
    /// # #[cfg(feature = "metis")]
    /// # fn main() {
    /// use tetraloma::prelude::*;
    /// // ThreadComm: a test communicator that runs each rank as a thread of this process; a
    /// // program passes an `Mpi` communicator instead.
    /// use tetraloma_testing::ThreadComm;
    ///
    /// // Two ranks; rank 0 passes the four intervals of [0, 4], rank 1 none.
    /// let lengths = ThreadComm::run(2, |comm| {
    ///     // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    ///     # let client = tetraloma_testing::test_client();
    ///     let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    ///     let (coords, ids, cells): (&[f64], &[i64], &[i64]) = if comm.rank() == 0 {
    ///         (&[0.0, 1.0, 2.0, 3.0, 4.0], &[0, 1, 2, 3, 4], &[0, 1, 1, 2, 2, 3, 3, 4])
    ///     } else {
    ///         (&[], &[], &[])
    ///     };
    ///     let mesh: Mesh<f32, D1, D1, ThreadComm> = MeshBuilder::new(&client)
    ///         .nodes(coords, Some(ids))
    ///         .cell_block_by_ids(element, cells, None)
    ///         .partitioner(Metis::default().seed(7))
    ///         .build_on(comm)?;
    ///     mesh.measure(1)
    /// });
    /// assert_eq!(lengths[0].as_ref().ok(), Some(&4.0));
    /// assert_eq!(lengths[1].as_ref().ok(), Some(&4.0));
    /// # }
    /// # #[cfg(not(feature = "metis"))]
    /// # fn main() {}
    /// ```
    ///
    /// PT-Scotch on MPI, with the feature `ptscotch`, launched by `mpirun`:
    ///
    /// ```no_run
    /// # #[cfg(feature = "ptscotch")]
    /// # fn main() -> tetraloma::Result<()> {
    /// use tetraloma::prelude::*;
    ///
    /// let universe = mpi::initialize().expect("MPI is not initialised yet");
    /// let comm = Mpi::world(&universe);
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // Rank r passes the intervals [2r, 2r + 1] and [2r + 1, 2r + 2], and their nodes from 2r on
    /// // (the last rank also the final node).
    /// let (r, n) = (comm.rank() as i64, comm.size() as i64);
    /// let ids: Vec<i64> = (2 * r..if r == n - 1 { 2 * n + 1 } else { 2 * r + 2 }).collect();
    /// let coords: Vec<f64> = ids.iter().map(|&i| i as f64).collect();
    /// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: Mesh<f64, D1, D1, Mpi> = MeshBuilder::new(&client)
    ///     .nodes(&coords, Some(&ids))
    ///     .cell_block_by_ids(element, &[2 * r, 2 * r + 1, 2 * r + 1, 2 * r + 2], None)
    ///     .partitioner(PtScotch::default().seed(7))
    ///     .build_on(comm)?;
    /// assert_eq!(mesh.measure(1)?, 2.0 * n as f64);
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "ptscotch"))]
    /// # fn main() {}
    /// ```
    pub fn partitioner(mut self, p: impl Partitioner + 'static) -> Self {
        self.partitioner = Box::new(p);
        self
    }

    /// Sets the order of each rank's owned cells; the default is [`CellOrdering::Input`].
    ///
    /// Vertices and nodes are numbered in order of the first cell that uses them, so the ordering
    /// changes their local order too. It does not change which cells a rank owns. [Cell ordering]
    /// describes the orderings.
    ///
    /// [Cell ordering]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#cell-ordering
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // The interval [0, 4] as four cells, listed out of order.
    /// let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let coords = [0.0, 1.0, 2.0, 3.0, 4.0];
    /// let cells: [u32; 8] = [2, 3, 0, 1, 3, 4, 1, 2];
    /// let build = |ordering| -> Result<Mesh<f32, D1, D1>> {
    ///     MeshBuilder::new(&client)
    ///         .nodes(&coords, None)
    ///         .cell_block(seg, &cells, None)
    ///         .ordering(ordering)
    ///         .build()
    /// };
    /// // Whether each cell shares a vertex with the next one; the copy to the host is for
    /// // inspection.
    /// let chained = |mesh: &Mesh<f32, D1, D1>| -> Result<bool> {
    ///     let v = mesh.cell_blocks()[0].vertices.data.to_host()?;
    ///     Ok(v.chunks_exact(2).zip(v.chunks_exact(2).skip(1)).all(|(a, b)| a.iter().any(|x| b.contains(x))))
    /// };
    /// // The input order jumps; reverse Cuthill–McKee walks along the line.
    /// assert!(!chained(&build(CellOrdering::Input)?)?);
    /// assert!(chained(&build(CellOrdering::ReverseCuthillMcKee)?)?);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn ordering(mut self, o: CellOrdering) -> Self {
        self.ordering = o;
        self
    }

    // ADR-0026 (tags of a distributed build).
    /// Adds physical tags to entities of type `entity`: one entry per entity, with
    /// `entity.num_vertices()` corner origin ids in `corners` and one value in `values`.
    ///
    /// Entries match entities by their sorted corner ids, cells and lower-dimensional entities
    /// alike, and the build creates the entities of every tagged dimension below the cells'. Every
    /// rank that holds a tagged entity, as owner or ghost, gets its values, and an entity in
    /// several groups gets each value once. [Physical groups] shows how to tag a mesh and select
    /// the tagged entities.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&coords, Some(&[10, 11, 12, 13]))
    ///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    ///     // The second triangle in group 5; the edges (10, 11) and (13, 11) in group 1.
    ///     .physical_tags(CellType::Triangle, &[13, 12, 11], &[5])
    ///     .physical_tags(CellType::Interval, &[10, 11, 13, 11], &[1, 1])
    ///     .tag_names(&[("wall", 1, 1), ("domain", 2, 5)])
    ///     .build()?;
    /// assert_eq!(mesh.tags().named(Codim::Vol, "domain")?.indices.to_host()?, [1]);
    /// assert_eq!(mesh.tags().group_size(1, 1), Some(2));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn physical_tags(mut self, entity: CellType, corners: &[i64], values: &[i32]) -> Self {
        self.tags.push(TagInput::physical(
            entity,
            corners.to_vec(),
            values.to_vec(),
        ));
        self
    }

    /// Adds entries to the table of tag names: (name, dimension, value) each, stored in the mesh's
    /// [`TagRegistry`](crate::TagRegistry).
    ///
    /// The ranks agree on one table: rank 0's entries, then each further rank's entries that no
    /// lower rank gave. [Physical groups] shows how to select tagged entities by name.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&coords, Some(&[10, 11, 12, 13]))
    ///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    ///     // The edge from node 10 to node 11 in group 1 of dimension 1, named "bottom".
    ///     .physical_tags(CellType::Interval, &[10, 11], &[1])
    ///     .tag_names(&[("bottom", 1, 1)])
    ///     .build()?;
    /// assert_eq!(mesh.tags().names().collect::<Vec<_>>(), [("bottom", 1, 1)]);
    /// assert_eq!(mesh.tags().lookup(1, "bottom"), Some(1));
    /// assert_eq!(mesh.tags().named(Codim::Bnd, "bottom")?.indices.len(), 1);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn tag_names(mut self, names: &[(&str, usize, i32)]) -> Self {
        self.tag_names.extend(
            names
                .iter()
                .map(|&(name, dim, value)| (name.to_owned(), dim, value)),
        );
        self
    }

    // Design: "Parallel creation pipeline", stages 1–9; ADR-0026 (the input contract).
    /// Builds a distributed mesh on the communicator `comm`. Collective: every rank passes its
    /// share of the nodes, cells and tags, and gets its part of the mesh.
    ///
    /// Which rank passes what is up to you, as long as the origin ids are unique over all ranks.
    /// The build checks the shares, partitions the cells with the
    /// [`partitioner`](MeshBuilder::partitioner), moves each cell to its rank, adds the
    /// [`ghosts`](MeshBuilder::ghosts), orders the cells by [`ordering`](MeshBuilder::ordering),
    /// numbers cells, vertices and nodes, and uploads the mesh. On one rank the partitioner is not
    /// called. [Building on a communicator] walks through it.
    ///
    /// [Building on a communicator]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#building-on-a-communicator
    ///
    /// # Errors
    ///
    /// Agreed over the ranks: a rank with a local error returns it, and every other rank returns
    /// `RemoteFailure` with the lowest failing rank. The errors of [`build`](MeshBuilder::build);
    /// `IndexOutOfRange` for a node id of `cell_block_by_ids` that no rank passed; errors of the
    /// partitioner and of the exchanges. `UnmatchedTagElement` for a tag entry of dimension below
    /// the cells' is the same on every rank: the entry of lowest global input position.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    /// // ThreadComm: a test communicator that runs each rank as a thread of this process; a
    /// // program passes an `Mpi` communicator instead.
    /// use tetraloma_testing::ThreadComm;
    ///
    /// // Two ranks, one interval each, sharing the node with id 1.
    /// let lengths = ThreadComm::run(2, |comm| {
    ///     // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    ///     # let client = tetraloma_testing::test_client();
    ///     let r = comm.rank() as i64;
    ///     let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    ///     // Rank 0 passes the nodes 0 and 1, rank 1 the node 2.
    ///     let (coords, ids): (&[f64], &[i64]) = if r == 0 { (&[0.0, 1.0], &[0, 1]) } else { (&[3.0], &[2]) };
    ///     let mesh: Mesh<f32, D1, D1, ThreadComm> = MeshBuilder::new(&client)
    ///         .nodes(coords, Some(ids))
    ///         .cell_block_by_ids(element, &[r, r + 1], None)
    ///         .build_on(comm)?;
    ///     assert_eq!(mesh.origin(), [1.5]);
    ///     mesh.measure(1)
    /// });
    /// assert_eq!(lengths[0].as_ref().ok(), Some(&3.0));
    /// assert_eq!(lengths[1].as_ref().ok(), Some(&3.0));
    /// ```
    pub fn build_on<C: Comm>(self, comm: C) -> Result<Mesh<F, TD, GD, C>> {
        Ok(self.build_with_comm(comm)?.0)
    }

    /// The dual graph that [`build_on`](MeshBuilder::build_on) gives the partitioner (stages
    /// 1–2, host only): a test accessor outside the stable API (ADR-0029, item 15). Collective.
    /// Unlike `build_on`, it builds the graph on one rank too.
    ///
    /// # Errors
    ///
    /// The input errors of `build_on`, agreed over the ranks, and errors of the exchanges.
    #[doc(hidden)]
    pub fn dual_graph_on<C: Comm>(self, comm: C) -> Result<DistGraph> {
        let read = input::read(&comm, &self.input())?;
        dual_graph(&comm, &read.blocks, &read.cells)
    }

    /// Checks the input, numbers the cells, vertices and nodes, and uploads the mesh:
    /// [`build_on`](MeshBuilder::build_on) with the one-rank [`Serial`] communicator.
    ///
    /// [Errors] shows the common input errors.
    ///
    /// [Errors]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#errors
    ///
    /// # Errors
    ///
    /// - `UnsupportedPrecision` if the device lacks `F`;
    /// - `LengthMismatch` if a length is not a multiple of its row length or an id list does not
    ///   match its rows; `IndexOutOfRange` for a node index beyond the nodes;
    /// - `DuplicateOriginId` for repeated node or cell ids; `NodeRoleMismatch` if a node is a
    ///   vertex of one cell and a non-vertex node of another; `DuplicateCellBlock` for a second
    ///   block of one cell type; `UnsupportedNodeFamily` or `UnsupportedDegree` for an element that
    ///   tetraloma does not offer;
    /// - `UnmatchedTagElement` for a tag entry that matches no cell or entity, `tag` being its
    ///   global input position (entries counted over the `physical_tags` calls in call order, ranks
    ///   in order);
    /// - allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&coords, None)
    ///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    ///     .build()?;
    /// assert_eq!(mesh.cell_blocks()[0].index_map.num_owned(), 2);
    ///
    /// // A second block of triangles.
    /// let r = MeshBuilder::<f32, D2, D2>::new(&client)
    ///     .nodes(&coords, None)
    ///     .cell_block(tri, &[0, 1, 2], None)
    ///     .cell_block(tri, &[3, 2, 1], None)
    ///     .build();
    /// assert!(matches!(r, Err(GridError::DuplicateCellBlock(_))));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn build(self) -> Result<Mesh<F, TD, GD>> {
        self.build_on(Serial)
    }

    /// The input of stage 1.
    fn input(&self) -> Input<'_> {
        Input {
            tdim: TD::N,
            gdim: GD::N,
            coords: &self.coords,
            node_ids: self.node_ids.as_deref(),
            blocks: self
                .blocks
                .iter()
                .map(|b| input::BlockInput {
                    spec: BlockSpec {
                        cell: b.element.cell.into(),
                        degree: b.element.degree,
                        family: b.element.family,
                    },
                    nodes: &b.nodes,
                    origin_ids: b.origin_ids.as_deref(),
                    physical: &b.physical,
                    elementary: b.elementary.as_deref(),
                })
                .collect(),
            tags: &self.tags,
            element_ids: &self.element_ids,
        }
    }

    /// Frees the input arrays, which stage 1 has copied.
    fn release_input(&mut self) {
        self.coords = Vec::new();
        self.node_ids = None;
        self.blocks = Vec::new();
        self.tags = Vec::new();
        self.element_ids = Vec::new();
    }

    /// [`build_on`](MeshBuilder::build_on), also returning the local vertex index of each node
    /// of this rank's input chunk (`u32::MAX` for nodes that are no local cell's vertex); the
    /// Gmsh reader matches tags through it (ADR-0022, item 5).
    pub(crate) fn build_with_comm<C: Comm>(mut self, comm: C) -> Result<BuiltMesh<F, TD, GD, C>> {
        let client = self.client.clone();
        let client = &client;
        agree(&comm, check_precision::<F>(client))?;
        // Stages 1–8. Every intermediate is dropped once its last stage has run (the M8
        // memory bound, docs/plans/M8.md T5b).
        let Read {
            blocks,
            nodes: chunk,
            cells,
            lower_tags,
        } = input::read(&comm, &self.input())?;
        self.release_input();
        log_memory(client, "stage 1 (input)");
        // Fixed destinations replace stages 2–3; a rank without them has none.
        let fixed = (&comm as &dyn CommDyn).max(u32::from(self.destinations.is_some())) == 1;
        let owned = if fixed {
            let dest = self.destinations.take().unwrap_or_default();
            agree(&comm, check_destinations(&dest, cells.len(), comm.size()))?;
            if comm.size() == 1 {
                cells
            } else {
                migrate(&comm, cells, &dest)?
            }
        } else if comm.size() == 1 {
            cells
        } else {
            // `Block` reads only the vertex distribution, so its graph has no edges.
            let graph = if self.partitioner.name() == Block.name() {
                vertices_only(&comm, cells.len())
            } else {
                dual_graph(&comm, &blocks, &cells)?
            };
            let dest = partition(&comm, self.partitioner.as_ref(), &graph)?;
            drop(graph);
            migrate(&comm, cells, &dest)?
        };
        log_memory(client, "stages 2-4 (partition, migrate)");
        let local = ghost_cells(&comm, &blocks, owned, self.ghosts)?;
        log_memory(client, "stage 5 (ghosts)");
        let refs = std::mem::take(&mut self.node_refs);
        let nodes = fetch_nodes(&comm, &blocks, GD::N, &chunk, &local, &refs)?;
        drop(refs);
        log_memory(client, "stage 6 (nodes)");
        let local = order_cells(&blocks, local, &nodes, self.ordering);
        let mut numbered = number(&comm, &blocks, local, &nodes)?;
        log_memory(client, "stages 7-8 (order, number)");

        let vertex_local: HashMap<i64, u32> = numbered
            .vertex_ids
            .iter()
            .enumerate()
            .map(|(v, &id)| (id, v as u32))
            .collect();
        let vertex_of_node = chunk
            .ids
            .iter()
            .map(|id| vertex_local.get(id).copied().unwrap_or(u32::MAX))
            .collect();
        drop(chunk);
        let cells = agree(&comm, cell_tags(&numbered.blocks))?;
        let cell_elementary = agree(&comm, cell_elementary(&numbered.blocks))?;
        let directory = std::mem::take(&mut numbered.directory);
        // Stage 9, then the tags.
        let (mut topology, geometry) = upload(client, &comm, &blocks, &nodes, numbered)?;
        drop(nodes);
        log_memory(client, "stage 9 (upload)");
        let tags = TagInputs {
            cells,
            cell_elementary,
            lower: &lower_tags,
            names: &self.tag_names,
            directory: &directory,
            vertex_local: &vertex_local,
        };
        let tags = registry(client, &comm, &mut topology, tags)?;
        log_memory(client, "stage 10 (tags)");
        let mut mesh = Mesh::from_parts(client.clone(), comm, topology, geometry);
        mesh.set_tags(tags);
        Ok((mesh, vertex_of_node))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use tetraloma_dims::{Cell2, Cell3, D2, D3};
    use tetraloma_geometry::DeviceGrid;
    use tetraloma_testing::{HostChunk, HostMesh, ThreadComm};
    use tetraloma_topology::decode_origin_ids;

    /// Physical and elementary values by (dimension, sorted vertex origin ids).
    type Values = BTreeMap<(usize, Vec<i64>), (Vec<i32>, Vec<i32>)>;

    /// The reader's physical values of the cell with origin id `id`.
    fn cell_physical(id: i64) -> Vec<i32> {
        if id % 3 == 0 { vec![1, 4] } else { vec![2] }
    }

    /// The reader's elementary tag of the cell with origin id `id`.
    fn cell_elementary(id: i64) -> i32 {
        10 + id.rem_euclid(4) as i32
    }

    /// The reader's elementary tag of the lower-dimensional entity with sorted corners `key`.
    fn key_elementary(key: &[i64]) -> i32 {
        20 + key.iter().sum::<i64>().rem_euclid(5) as i32
    }

    /// Sorted copy of `v`.
    fn sorted(v: &[i64]) -> Vec<i64> {
        let mut v = v.to_vec();
        v.sort_unstable();
        v
    }

    /// The builder the Gmsh reader would make of `chunk`: cells with their values through
    /// `tagged_cell_block`, the tags below the cell dimension as physical and elementary
    /// `tag_entries` with error ids from 10^6 on, which are also element ids. Collective.
    fn reader_builder<TD, GD>(
        client: &Client,
        comm: &dyn CommDyn,
        chunk: &HostChunk,
    ) -> MeshBuilder<f32, TD, GD>
    where
        TD: TopoDim + EmbeddedIn<GD>,
        TD::Cell: TryFrom<CellType>,
        GD: Dim,
    {
        let mut b = MeshBuilder::new(client).nodes(&chunk.nodes, Some(&chunk.node_ids));
        for block in &chunk.blocks {
            let Ok(cell) = TD::Cell::try_from(block.cell) else {
                unreachable!("cell blocks have the cell dimension");
            };
            let element = CoordinateElement::<TD> {
                cell,
                degree: block.degree,
                family: NodeFamily::Equispaced,
            };
            let mut physical = Vec::new();
            for (c, &id) in block.cell_ids.iter().enumerate() {
                physical.extend(cell_physical(id).into_iter().map(|v| (c as u32, v)));
            }
            let elementary = block
                .cell_ids
                .iter()
                .map(|&id| cell_elementary(id))
                .collect();
            b = b.tagged_cell_block(
                element,
                block.node_ids.clone(),
                block.cell_ids.clone(),
                physical,
                elementary,
            );
        }
        let lower: Vec<_> = chunk
            .tags
            .iter()
            .filter(|t| tetraloma_refcell::tdim(t.entity) < TD::N)
            .collect();
        let n: usize = lower.iter().map(|t| t.values.len()).sum();
        let mut next = 1_000_000 + comm.exscan_sum(n as i64);
        for t in lower {
            let ids: Vec<i64> = (next..next + t.values.len() as i64).collect();
            next += t.values.len() as i64;
            let nv = tetraloma_refcell::vertices(t.entity).len();
            let elementary = t
                .corners
                .chunks_exact(nv)
                .map(|c| key_elementary(&sorted(c)))
                .collect();
            b = b
                .tag_entries(
                    TagKind::Elementary,
                    t.entity,
                    t.corners.clone(),
                    elementary,
                    ids.clone(),
                )
                .tag_entries(
                    TagKind::Physical,
                    t.entity,
                    t.corners.clone(),
                    t.values.clone(),
                    ids.clone(),
                )
                .element_ids(ids);
        }
        b
    }

    /// The reader's values of `h`: cells by `cell_physical` and `cell_elementary`, lower
    /// entities by the host tags and `key_elementary`.
    fn reference(h: &HostMesh) -> Values {
        let whole = h.chunk(0, 1);
        let mut out = Values::new();
        for block in &whole.blocks {
            let k = tetraloma_refcell::num_lagrange_nodes(block.cell, block.degree);
            let nv = tetraloma_refcell::vertices(block.cell).len();
            let dim = tetraloma_refcell::tdim(block.cell);
            for (row, &id) in block.node_ids.chunks_exact(k).zip(&block.cell_ids) {
                let values = (cell_physical(id), vec![cell_elementary(id)]);
                out.insert((dim, sorted(&row[..nv])), values);
            }
        }
        for t in &whole.tags {
            let dim = tetraloma_refcell::tdim(t.entity);
            if dim
                == h.blocks
                    .iter()
                    .map(|b| tetraloma_refcell::tdim(b.cell))
                    .max()
                    .unwrap()
            {
                continue;
            }
            let nv = tetraloma_refcell::vertices(t.entity).len();
            for (corners, &v) in t.corners.chunks_exact(nv).zip(&t.values) {
                let key = sorted(corners);
                let e = key_elementary(&key);
                let (physical, elementary) = out.entry((dim, key)).or_default();
                physical.push(v);
                elementary.push(e);
                physical.sort_unstable();
                physical.dedup();
                elementary.dedup();
            }
        }
        out
    }

    /// The values of every local cell and tagged entity of `mesh`, owned or ghost.
    fn local_values<TD, GD, C>(mesh: &mut Mesh<f32, TD, GD, C>) -> Values
    where
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let ids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
        let mut keys: BTreeMap<(usize, u32), Vec<i64>> = BTreeMap::new();
        let mut first = 0;
        for b in mesh.cell_blocks() {
            let stride = b.vertices.stride as usize;
            for (c, row) in b
                .vertices
                .data
                .to_host()
                .unwrap()
                .chunks_exact(stride)
                .enumerate()
            {
                let key = sorted(&row.iter().map(|&v| ids[v as usize]).collect::<Vec<_>>());
                keys.insert((TD::N, (first + c) as u32), key);
            }
            first += b.index_map.num_local();
        }
        for d in 0..TD::N {
            if mesh.tags().physical(d).is_none() && mesh.tags().elementary(d).is_none() {
                continue;
            }
            for b in mesh.entity_blocks(d).unwrap() {
                let stride = b.vertices.stride as usize;
                let data = b.vertices.data.to_host().unwrap();
                for (e, row) in data.chunks_exact(stride).enumerate() {
                    let key = sorted(&row.iter().map(|&v| ids[v as usize]).collect::<Vec<_>>());
                    keys.insert((d, b.offset + e as u32), key);
                }
            }
        }
        let mut out = Values::new();
        for d in 0..=TD::N {
            for (kind, tags) in [(0, mesh.tags().physical(d)), (1, mesh.tags().elementary(d))] {
                let Some(t) = tags else { continue };
                let indices = t.indices.to_host().unwrap();
                for (i, v) in indices.into_iter().zip(t.values.to_host().unwrap()) {
                    let entry = out.entry((d, keys[&(d, i)].clone())).or_default();
                    if kind == 0 {
                        &mut entry.0
                    } else {
                        &mut entry.1
                    }
                    .push(v);
                }
            }
        }
        out
    }

    /// The sorted origin ids of the cells `mesh` owns.
    fn owned_ids<TD, GD, C>(mesh: &Mesh<f32, TD, GD, C>) -> Vec<i64>
    where
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
        C: Comm,
    {
        let mut owned = Vec::new();
        for b in mesh.cell_blocks() {
            let ids = decode_origin_ids(&b.origin_ids.to_host().unwrap());
            owned.extend_from_slice(&ids[..b.index_map.num_owned()]);
        }
        owned.sort_unstable();
        owned
    }

    /// The destination of cell `id` on `p` ranks.
    fn destination(id: i64, p: usize) -> i32 {
        (id * 7 + 3).rem_euclid(p as i64) as i32
    }

    /// On 1–4 ranks, with partitioner or fixed destinations and in two ghost modes, every local
    /// cell and tagged entity has the physical and elementary values of the one-rank build,
    /// which equal the reader's input; fixed destinations give each rank exactly its cells.
    fn reader_hooks<TD, GD>(name: &str, h: &HostMesh)
    where
        TD: TopoDim + EmbeddedIn<GD>,
        TD::Cell: TryFrom<CellType>,
        GD: Dim,
    {
        let want = reference(h);
        let serial = ThreadComm::run(1, |comm| {
            let client = tetraloma_testing::test_client();
            let chunk = h.chunk(0, 1);
            let mut mesh = reader_builder::<TD, GD>(&client, &comm, &chunk)
                .build_on(comm.clone())
                .unwrap();
            local_values(&mut mesh)
        })
        .remove(0);
        assert_eq!(serial, want, "{name}");
        for p in 1..=4 {
            for fixed in [false, true] {
                for mode in [GhostMode::None, GhostMode::SharedFacet] {
                    ThreadComm::run(p, |comm| {
                        let client = tetraloma_testing::test_client();
                        let chunk = h.chunk(comm.rank(), p);
                        let mut b = reader_builder::<TD, GD>(&client, &comm, &chunk).ghosts(mode);
                        if fixed {
                            let dest = chunk
                                .blocks
                                .iter()
                                .flat_map(|b| b.cell_ids.iter().map(|&id| destination(id, p)))
                                .collect();
                            b = b.destinations(dest);
                        }
                        let mut mesh = b.build_on(comm.clone()).unwrap();
                        let at = format!("{name}, {p} ranks, fixed {fixed}, {mode:?}");
                        for (key, values) in local_values(&mut mesh) {
                            assert_eq!(Some(&values), serial.get(&key), "{at}: {key:?}");
                        }
                        if fixed {
                            let mut mine: Vec<i64> = h
                                .cell_origin_ids()
                                .concat()
                                .into_iter()
                                .filter(|&id| destination(id, p) == comm.rank() as i32)
                                .collect();
                            mine.sort_unstable();
                            assert_eq!(owned_ids(&mesh), mine, "{at}");
                        }
                    });
                }
            }
        }
    }

    #[test]
    fn reader_hooks_equal_one_rank() {
        reader_hooks::<D2, D2>(
            "triangles",
            &tetraloma_testing::square_box(Cell2::Triangle, 3, 1),
        );
        reader_hooks::<D3, D3>("mixed", &tetraloma_testing::mixed_box(2, 1));
        reader_hooks::<D3, D3>(
            "tetrahedra p2, shuffled ids",
            &tetraloma_testing::cube_box(Cell3::Tetrahedron, 2, 2).with_shuffled_ids(5),
        );
    }

    /// Agreed errors of the hooks: an unmatched entry names its error id (the lowest position
    /// over the ranks below the cells; at the cell dimension, from the key's rendezvous rank); a repeated
    /// element id, or one that is also a cell id; destinations not one per cell.
    #[test]
    fn reader_hook_errors() {
        let h = tetraloma_testing::square_box(Cell2::Quadrilateral, 3, 1);
        let build = |p: usize,
                     edit: &(
                          dyn Fn(usize, MeshBuilder<f32, D2, D2>) -> MeshBuilder<f32, D2, D2> + Sync
                      )| {
            ThreadComm::run(p, |comm| {
                let client = tetraloma_testing::test_client();
                let chunk = h.chunk(comm.rank(), p);
                let b = reader_builder::<D2, D2>(&client, &comm, &chunk);
                edit(comm.rank(), b).build_on(comm.clone()).map(|_| ())
            })
        };
        let bad_edge = |r: usize, b: MeshBuilder<f32, D2, D2>, kind: TagKind| {
            if r == 0 {
                return b;
            }
            let id = h.chunk(r, 3).node_ids[0];
            let error = 5000 + r as i64;
            b.tag_entries(
                kind,
                CellType::Interval,
                vec![id, 99_999],
                vec![1],
                vec![error],
            )
        };
        for kind in [TagKind::Physical, TagKind::Elementary] {
            let out = build(3, &|r, b| bad_edge(r, b, kind));
            for res in &out {
                assert!(
                    matches!(
                        res,
                        Err(GridError::UnmatchedTagElement { dim: 1, tag: 5001 })
                    ),
                    "{kind:?}: {out:?}"
                );
            }
        }

        // A cell-dimension entry matching no cell, on rank 1.
        let out = build(2, &|r, b| {
            if r == 1 {
                b.tag_entries(
                    TagKind::Physical,
                    CellType::Quadrilateral,
                    vec![99_996, 99_997, 99_998, 99_999],
                    vec![3],
                    vec![777],
                )
            } else {
                b
            }
        });
        let failing = out
            .iter()
            .position(|r| matches!(r, Err(GridError::UnmatchedTagElement { dim: 2, tag: 777 })))
            .expect("the key's rendezvous rank fails");
        for (r, res) in out.iter().enumerate().filter(|&(r, _)| r != failing) {
            assert!(
                matches!(res, Err(GridError::RemoteFailure { rank }) if *rank == failing),
                "rank {r}: {out:?}"
            );
        }

        // Element ids: a cell id of rank 0 given on rank 1, and one id on two ranks.
        let cell = h.chunk(0, 2).blocks[0].cell_ids[0];
        for ids in [[cell, -1], [-7, -7]] {
            let out = build(2, &|r, b| b.element_ids(vec![ids[r]]));
            let failing: Vec<_> = out
                .iter()
                .filter(|r| matches!(r, Err(GridError::DuplicateOriginId(_))))
                .collect();
            assert_eq!(failing.len(), 1, "{out:?}");
            assert!(matches!(failing[0], Err(GridError::DuplicateOriginId(id)) if *id == ids[0]));
            assert!(out.iter().all(|r| matches!(
                r,
                Err(GridError::DuplicateOriginId(_) | GridError::RemoteFailure { .. })
            )));
        }

        // Destinations: too few on rank 1, none on rank 0, or out of range.
        let out = build(2, &|r, b| if r == 1 { b.destinations(vec![0]) } else { b });
        assert!(
            matches!(out[0], Err(GridError::LengthMismatch { found: 0, .. })),
            "{out:?}"
        );
        assert!(
            matches!(out[1], Err(GridError::LengthMismatch { found: 1, .. })),
            "{out:?}"
        );
        let out = build(2, &|r, b| {
            let n = h.chunk(r, 2).blocks.iter().map(|b| b.cell_ids.len()).sum();
            b.destinations(vec![2; n])
        });
        assert!(
            out.iter()
                .all(|r| matches!(r, Err(GridError::IndexOutOfRange { index: 2, len: 2 }))),
            "{out:?}"
        );
    }
}
