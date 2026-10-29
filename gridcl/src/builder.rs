//! `MeshBuilder` (ADR-0012): a serial `Mesh` from host arrays with origin ids.

use std::collections::HashSet;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Real, Result, check_precision};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
use gridcl_parallel::{Comm, IndexMap, Serial};
use gridcl_partition::{Block, Partitioner};
use gridcl_refcell::RefCellError;
use gridcl_topology::{CellBlock, Topology, encode_origin_ids};

use crate::Mesh;
use crate::pipeline::input::{NodeRefs, TagInput};

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

/// Fails with `DuplicateOriginId` at the first repeated id.
fn check_unique(ids: impl IntoIterator<Item = i64>) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            return Err(GridError::DuplicateOriginId(id));
        }
    }
    Ok(())
}

/// A u32 index of `i`.
fn index_u32(i: usize) -> Result<u32> {
    u32::try_from(i).map_err(|_| GridError::TooLarge)
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

/// Builds a serial [`Mesh`] from host arrays (ADR-0012).
///
/// - [`nodes`](MeshBuilder::nodes): coordinates `[node][gdim]` in f64, with optional origin ids
///   (default: the node's index).
/// - [`cell_block`](MeshBuilder::cell_block): cell → node lists in A.5 order, one block per
///   cell type, with optional cell origin ids (default: the cell's index over all blocks, in
///   call order). A cell's vertices are its first V nodes, and a vertex's origin id is its
///   node's.
/// - [`build`](MeshBuilder::build) checks the input, numbers vertices and nodes in order of the
///   first cell that touches them (blocks in call order; pipeline stage 8), stores coordinates
///   relative to the centre of the bounding box, and uploads. Nodes no cell uses are dropped.
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
    /// Entries are matched by their sorted corner ids (ADR-0026).
    pub fn physical_tags(mut self, entity: CellType, corners: &[i64], values: &[i32]) -> Self {
        self.tags.push(TagInput {
            entity,
            corners: corners.to_vec(),
            values: values.to_vec(),
        });
        self
    }

    /// The tag name table: (name, dimension, value).
    pub fn tag_names(mut self, names: &[(&str, usize, i32)]) -> Self {
        self.tag_names.extend(
            names
                .iter()
                .map(|&(name, dim, value)| (name.to_owned(), dim, value)),
        );
        self
    }

    /// Builds the distributed mesh on `comm` through the parallel creation pipeline.
    /// Collective; every rank passes its chunk of nodes, cells and tags (ADR-0026).
    pub fn build_on<C: Comm>(self, comm: C) -> Result<Mesh<F, TD, GD, C>> {
        let _ = comm;
        todo!("M6 T12: MeshBuilder::build_on")
    }

    /// Checks, numbers and uploads the mesh.
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
    /// - allocation and device errors.
    pub fn build(self) -> Result<Mesh<F, TD, GD>> {
        Ok(self.build_with_comm(Serial)?.0)
    }

    /// [`build`](MeshBuilder::build) on the one-rank communicator `comm` (ADR-0022, item 5),
    /// also returning the local vertex index of each input node (`u32::MAX` for nodes that are
    /// no cell's vertex).
    pub(crate) fn build_with_comm<C: Comm>(self, comm: C) -> Result<BuiltMesh<F, TD, GD, C>> {
        let client = &self.client;
        check_precision::<F>(client)?;
        let gd = GD::N;
        if !self.coords.len().is_multiple_of(gd) {
            return Err(GridError::LengthMismatch {
                expected: self.coords.len().next_multiple_of(gd),
                found: self.coords.len(),
            });
        }
        let num_input = self.coords.len() / gd;
        let node_ids = ids_or_index(self.node_ids.as_deref(), num_input, 0)?;
        // Cells by node id need the pipeline (M6 T12).
        let rows: Vec<&[u32]> = self
            .blocks
            .iter()
            .map(|b| match &b.nodes {
                NodeRefs::Indices(v) => Ok(v.as_slice()),
                NodeRefs::Ids(_) => Err(GridError::NotYetImplemented("cell_block_by_ids: M6 T12")),
            })
            .collect::<Result<_>>()?;
        check_unique(node_ids.iter().copied())?;

        // Per block: element checks, lengths, index ranges and cell ids.
        let mut cells_seen = HashSet::new();
        let mut cell_ids = Vec::with_capacity(self.blocks.len());
        let mut first_cell = 0;
        for (b, nodes) in self.blocks.iter().zip(&rows) {
            let cell: CellType = b.element.cell.into();
            if !cells_seen.insert(cell) {
                return Err(GridError::DuplicateCellBlock(cell_name(cell)));
            }
            gridcl_refcell::lagrange_points(cell, b.element.degree, b.element.family)
                .map_err(element_error)?;
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            if !nodes.len().is_multiple_of(k) {
                return Err(GridError::LengthMismatch {
                    expected: nodes.len().next_multiple_of(k),
                    found: nodes.len(),
                });
            }
            if let Some(&bad) = nodes.iter().find(|&&n| n as usize >= num_input) {
                return Err(GridError::IndexOutOfRange {
                    index: bad as usize,
                    len: num_input,
                });
            }
            let n = nodes.len() / k;
            cell_ids.push(ids_or_index(b.origin_ids.as_deref(), n, first_cell)?);
            first_cell += n;
        }
        check_unique(cell_ids.iter().flatten().copied())?;

        // Node roles, and numbering in order of the first touching cell (pipeline stage 8).
        const NONE: u32 = u32::MAX;
        let mut role: Vec<Option<bool>> = vec![None; num_input];
        let mut node_number = vec![NONE; num_input];
        let mut vertex_number = vec![NONE; num_input];
        let (mut nodes_used, mut vertices_used) = (Vec::new(), Vec::new());
        for (b, nodes) in self.blocks.iter().zip(&rows) {
            let cell: CellType = b.element.cell.into();
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            let nv = gridcl_refcell::vertices(cell).len();
            for row in nodes.chunks_exact(k) {
                for (j, &node) in row.iter().enumerate() {
                    let node = node as usize;
                    let is_vertex = j < nv;
                    match role[node] {
                        Some(r) if r != is_vertex => return Err(GridError::NodeRoleMismatch(node)),
                        _ => role[node] = Some(is_vertex),
                    }
                    if node_number[node] == NONE {
                        node_number[node] = index_u32(nodes_used.len())?;
                        nodes_used.push(node);
                    }
                    if is_vertex && vertex_number[node] == NONE {
                        vertex_number[node] = index_u32(vertices_used.len())?;
                        vertices_used.push(node);
                    }
                }
            }
        }

        // The origin: the centre of the bounding box of the mesh's nodes.
        let mut origin = GD::Array::<f64>::default();
        for g in 0..gd {
            let coord = nodes_used.iter().map(|&n| self.coords[n * gd + g]);
            let lo = coord.clone().fold(f64::INFINITY, f64::min);
            let hi = coord.fold(f64::NEG_INFINITY, f64::max);
            origin.as_mut()[g] = if nodes_used.is_empty() {
                0.0
            } else {
                0.5 * (lo + hi)
            };
        }
        let relative: Vec<F> = nodes_used
            .iter()
            .flat_map(|&n| (0..gd).map(move |g| (n, g)))
            .map(|(n, g)| F::narrow(self.coords[n * gd + g] - origin.as_ref()[g]))
            .collect();
        let vertex_ids: Vec<i64> = vertices_used.iter().map(|&n| node_ids[n]).collect();

        let mut cell_blocks = Vec::with_capacity(self.blocks.len());
        let mut geometry_blocks = Vec::with_capacity(self.blocks.len());
        for ((b, nodes), ids) in self.blocks.iter().zip(&rows).zip(&cell_ids) {
            let cell: CellType = b.element.cell.into();
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            let nv = gridcl_refcell::vertices(cell).len();
            let n = ids.len();
            let dofmap: Vec<u32> = nodes.iter().map(|&x| node_number[x as usize]).collect();
            let verts: Vec<u32> = nodes
                .chunks_exact(k)
                .flat_map(|row| row[..nv].iter().map(|&x| vertex_number[x as usize]))
                .collect();
            cell_blocks.push(CellBlock::new(
                b.element.cell,
                IndexMap::new(&comm, n)?,
                DeviceStrided {
                    data: DeviceBuffer::from_host(client, &verts)?,
                    stride: index_u32(nv)?,
                },
                DeviceBuffer::from_host(client, &encode_origin_ids(ids))?,
                DeviceBuffer::zeros(client, n)?,
            ));
            geometry_blocks.push(GeometryBlock {
                element: b.element,
                dofmap: DeviceStrided {
                    data: DeviceBuffer::from_host(client, &dofmap)?,
                    stride: index_u32(k)?,
                },
            });
        }
        let topology = Topology::new(
            client,
            cell_blocks,
            IndexMap::new(&comm, vertices_used.len())?,
            &vertex_ids,
        )?;
        let geometry = Geometry::new(
            geometry_blocks,
            DeviceBuffer::from_host(client, &relative)?,
            IndexMap::new(&comm, nodes_used.len())?,
            origin,
        )?;
        Ok((
            Mesh::from_parts(client.clone(), comm, topology, geometry),
            vertex_number,
        ))
    }
}
