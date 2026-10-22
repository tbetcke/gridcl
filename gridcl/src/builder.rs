//! `MeshBuilder` (ADR-0012): a serial `Mesh` from host arrays with origin ids.

use std::collections::HashSet;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Real, Result, check_precision};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
use gridcl_parallel::{IndexMap, Serial};
use gridcl_refcell::RefCellError;
use gridcl_topology::{CellBlock, Topology, encode_origin_ids};

use crate::Mesh;

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
fn element_error(e: RefCellError) -> GridError {
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
fn ids_or_index(ids: Option<&[i64]>, n: usize, first: usize) -> Result<Vec<i64>> {
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

/// One `cell_block` call: element, cell → node lists and optional cell ids.
struct BlockInput<TD: TopoDim> {
    element: CoordinateElement<TD>,
    nodes: Vec<u32>,
    origin_ids: Option<Vec<i64>>,
}

/// Builds a serial [`Mesh`] from host arrays (ADR-0012).
///
/// - [`nodes`](MeshBuilder::nodes): coordinates [node][gdim] in f64, with optional origin ids
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
            _f: core::marker::PhantomData,
        }
    }

    /// The node coordinates, [node][gdim] in f64, and optionally their origin ids.
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
            nodes: nodes.to_vec(),
            origin_ids: origin_ids.map(<[i64]>::to_vec),
        });
        self
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
        check_unique(node_ids.iter().copied())?;

        // Per block: element checks, lengths, index ranges and cell ids.
        let mut cells_seen = HashSet::new();
        let mut cell_ids = Vec::with_capacity(self.blocks.len());
        let mut first_cell = 0;
        for b in &self.blocks {
            let cell: CellType = b.element.cell.into();
            if !cells_seen.insert(cell) {
                return Err(GridError::DuplicateCellBlock(cell_name(cell)));
            }
            gridcl_refcell::lagrange_points(cell, b.element.degree, b.element.family)
                .map_err(element_error)?;
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            if !b.nodes.len().is_multiple_of(k) {
                return Err(GridError::LengthMismatch {
                    expected: b.nodes.len().next_multiple_of(k),
                    found: b.nodes.len(),
                });
            }
            if let Some(&bad) = b.nodes.iter().find(|&&n| n as usize >= num_input) {
                return Err(GridError::IndexOutOfRange {
                    index: bad as usize,
                    len: num_input,
                });
            }
            let n = b.nodes.len() / k;
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
        for b in &self.blocks {
            let cell: CellType = b.element.cell.into();
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            let nv = gridcl_refcell::vertices(cell).len();
            for row in b.nodes.chunks_exact(k) {
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
        for (b, ids) in self.blocks.iter().zip(&cell_ids) {
            let cell: CellType = b.element.cell.into();
            let k = gridcl_refcell::num_lagrange_nodes(cell, b.element.degree);
            let nv = gridcl_refcell::vertices(cell).len();
            let n = ids.len();
            let dofmap: Vec<u32> = b.nodes.iter().map(|&x| node_number[x as usize]).collect();
            let verts: Vec<u32> = b
                .nodes
                .chunks_exact(k)
                .flat_map(|row| row[..nv].iter().map(|&x| vertex_number[x as usize]))
                .collect();
            cell_blocks.push(CellBlock::new(
                b.element.cell,
                IndexMap::new(&Serial, n)?,
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
            IndexMap::new(&Serial, vertices_used.len())?,
            &vertex_ids,
        )?;
        let geometry = Geometry::new(
            geometry_blocks,
            DeviceBuffer::from_host(client, &relative)?,
            IndexMap::new(&Serial, nodes_used.len())?,
            origin,
        )?;
        Ok(Mesh::from_parts(client.clone(), Serial, topology, geometry))
    }
}
