//! `MeshBuilder` (ADR-0012, ADR-0026): a `Mesh` from host arrays with origin ids, serial or
//! distributed, through the parallel creation pipeline.

use std::collections::HashMap;

use cubecl::prelude::Client;
use gridcl_device::{GridError, Real, Result, check_precision};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, NodeFamily};
use gridcl_parallel::{Comm, Serial};
use gridcl_partition::{Block, DistGraph, Partitioner};
use gridcl_refcell::RefCellError;

use crate::Mesh;
use crate::pipeline::dual_graph::{dual_graph, partition};
use crate::pipeline::ghosts::ghost_cells;
use crate::pipeline::input::{self, Input, NodeRefs, Read, TagInput};
use crate::pipeline::migrate::migrate;
use crate::pipeline::nodes::fetch_nodes;
use crate::pipeline::number::number;
use crate::pipeline::order::order_cells;
use crate::pipeline::tags::{TagInputs, cell_tags, registry};
use crate::pipeline::upload::upload;
use crate::pipeline::{BlockSpec, agree};

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

/// `GridError` of an element gridcl-refcell does not offer.
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

/// The ghost cells each rank holds besides its owned cells (docs/design.md, "Distributed mesh
/// model"; ADR-0026).
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

/// The order of the owned cells of each rank (pipeline stage 7; ADR-0026). Ghost cells follow,
/// ordered by (owner rank, origin id).
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
/// ids.
struct BlockInput<TD: TopoDim> {
    element: CoordinateElement<TD>,
    nodes: NodeRefs,
    origin_ids: Option<Vec<i64>>,
}

/// Builds a [`Mesh`] from host arrays (ADR-0012, ADR-0026).
///
/// - [`nodes`](MeshBuilder::nodes): coordinates `[node][gdim]` in f64, with optional origin ids
///   (default: the node's global index, an exclusive scan of the ranks' node counts plus its
///   index).
/// - [`cell_block`](MeshBuilder::cell_block): cell → node lists in A.5 order, one block per
///   cell type, with optional cell origin ids (default: the cell's global index over all
///   blocks, in call order). A cell's vertices are its first V nodes, and a vertex's origin id
///   is its node's. [`cell_block_by_ids`](MeshBuilder::cell_block_by_ids) references the nodes
///   by origin id instead.
/// - [`build`](MeshBuilder::build) builds a serial mesh, [`build_on`](MeshBuilder::build_on) a
///   distributed one, both through the parallel creation pipeline: they check the input,
///   partition and order the cells, number vertices and nodes in order of the first cell that
///   touches them (blocks in order of first declaration; pipeline stage 8), store coordinates
///   relative to the centre of the global bounding box, and upload. Nodes no cell uses are
///   dropped.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// // Two P1 triangles of the unit square.
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
    /// An empty builder on `client`.
    pub fn new(client: &Client) -> Self {
        MeshBuilder {
            client: client.clone(),
            coords: Vec::new(),
            node_ids: None,
            blocks: Vec::new(),
            tags: Vec::new(),
            tag_names: Vec::new(),
            ghosts: GhostMode::None,
            partitioner: Box::new(Block),
            ordering: CellOrdering::Input,
            _f: core::marker::PhantomData,
        }
    }

    /// The node coordinates, `[node][gdim]` in f64, and optionally their origin ids.
    pub fn nodes(mut self, coords: &[f64], origin_ids: Option<&[i64]>) -> Self {
        self.coords = coords.to_vec();
        self.node_ids = origin_ids.map(<[i64]>::to_vec);
        self
    }

    /// A cell block: `nodes` holds one row of the element's node count per cell, indices into
    /// the node array, in A.5 order; `origin_ids` optionally one id per cell.
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
        });
        self
    }

    /// A cell block whose cells reference nodes by node origin id: `node_ids` holds one row of
    /// the element's node count per cell, in A.5 order; `origin_ids` optionally one id per
    /// cell. A cell may reference the nodes of any rank (ADR-0026).
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
        });
        self
    }

    /// The ghost cells of each rank; default [`GhostMode::None`].
    pub fn ghosts(mut self, mode: GhostMode) -> Self {
        self.ghosts = mode;
        self
    }

    /// The partitioner of pipeline stage 3; default [`Block`].
    pub fn partitioner(mut self, p: impl Partitioner + 'static) -> Self {
        self.partitioner = Box::new(p);
        self
    }

    /// The order of the owned cells; default [`CellOrdering::Input`].
    pub fn ordering(mut self, o: CellOrdering) -> Self {
        self.ordering = o;
        self
    }

    /// Physical tags of entities of type `entity`: one entry per entity, with
    /// `entity.num_vertices()` corner origin ids in `corners` and one value in `values`.
    /// Entries are matched by their sorted corner ids, cells and lower-dimensional entities
    /// alike; the build creates the entities of every tagged dimension below the cells'. Every
    /// holder of an entity, owner or ghost, gets its values, and an entity in several groups
    /// gets each value once (ADR-0026).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
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
        self.tags.push(TagInput {
            entity,
            corners: corners.to_vec(),
            values: values.to_vec(),
        });
        self
    }

    /// The tag name table: (name, dimension, value), in the mesh's
    /// [`TagRegistry`](crate::TagRegistry). The tables of all ranks are agreed: rank 0's
    /// entries, then each further rank's entries that no lower rank gave.
    pub fn tag_names(mut self, names: &[(&str, usize, i32)]) -> Self {
        self.tag_names.extend(
            names
                .iter()
                .map(|&(name, dim, value)| (name.to_owned(), dim, value)),
        );
        self
    }

    /// Builds the distributed mesh on `comm` through the parallel creation pipeline (stages
    /// 1–9; docs/design.md, "Parallel creation pipeline"). Collective: every rank passes its
    /// chunk of nodes, cells and tags, under the input contract of ADR-0026. On one rank the
    /// partitioner is not called.
    ///
    /// # Errors
    ///
    /// Agreed over the ranks: a rank with a local error returns it, every other rank
    /// `RemoteFailure` with the lowest failing rank. The errors of [`build`](MeshBuilder::build);
    /// `IndexOutOfRange` for a node id of `cell_block_by_ids` that no rank passed; errors of the
    /// partitioner and the exchanges. `UnmatchedTagElement` for a tag entry of dimension below
    /// the cells' is the same on every rank: the entry of lowest global input position.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    /// use gridcl_testing::ThreadComm;
    ///
    /// // Two ranks, one P1 interval each, sharing the node with id 1.
    /// let lengths = ThreadComm::run(2, |comm| {
    ///     let client = gridcl_testing::test_client();
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

    /// Checks, numbers and uploads the mesh: [`build_on`](MeshBuilder::build_on) with the
    /// one-rank [`Serial`] communicator.
    ///
    /// # Errors
    ///
    /// - `UnsupportedPrecision` if the device lacks `F`;
    /// - `LengthMismatch` if a length is not a multiple of its row length or an id list does
    ///   not match its rows; `IndexOutOfRange` for a node index beyond the nodes;
    /// - `DuplicateOriginId` for repeated node or cell ids; `NodeRoleMismatch` if a node is a
    ///   vertex of one cell and a non-vertex node of another; `DuplicateCellBlock` for a second
    ///   block of one cell type; `UnsupportedNodeFamily` or `UnsupportedDegree` for an element
    ///   gridcl-refcell does not offer;
    /// - `UnmatchedTagElement` for a tag entry that matches no cell or entity, `tag` being its
    ///   global input position (entries counted over the `physical_tags` calls in call order,
    ///   ranks in order);
    /// - allocation and device errors.
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
                })
                .collect(),
            tags: &self.tags,
        }
    }

    /// [`build_on`](MeshBuilder::build_on), also returning the local vertex index of each node
    /// of this rank's input chunk (`u32::MAX` for nodes that are no local cell's vertex); the
    /// Gmsh reader matches tags through it (ADR-0022, item 5).
    pub(crate) fn build_with_comm<C: Comm>(self, comm: C) -> Result<BuiltMesh<F, TD, GD, C>> {
        let client = &self.client;
        agree(&comm, check_precision::<F>(client))?;
        // Stages 1–8.
        let Read {
            blocks,
            nodes: chunk,
            cells,
            lower_tags,
        } = input::read(&comm, &self.input())?;
        let owned = if comm.size() == 1 {
            cells
        } else {
            let graph = dual_graph(&comm, &blocks, &cells)?;
            let dest = partition(&comm, self.partitioner.as_ref(), &graph)?;
            migrate(&comm, &cells, &dest)?
        };
        let local = ghost_cells(&comm, &blocks, owned, self.ghosts)?;
        let nodes = fetch_nodes(&comm, &blocks, GD::N, &chunk, &local)?;
        let local = order_cells(&blocks, local, &nodes, self.ordering);
        let mut numbered = number(&comm, &blocks, &local, &nodes)?;

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
        let cells = agree(&comm, cell_tags(&numbered.blocks))?;
        let directory = std::mem::take(&mut numbered.directory);
        // Stage 9, then the tags.
        let (mut topology, geometry) = upload(client, &comm, &blocks, &nodes, numbered)?;
        let tags = TagInputs {
            cells,
            lower: &lower_tags,
            names: &self.tag_names,
            directory: &directory,
            vertex_local: &vertex_local,
        };
        let tags = registry(client, &comm, &mut topology, tags)?;
        let mut mesh = Mesh::from_parts(client.clone(), comm, topology, geometry);
        mesh.set_tags(tags);
        Ok((mesh, vertex_of_node))
    }
}
