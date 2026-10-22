//! `MeshBuilder` (ADR-0012): a serial `Mesh` from host arrays with origin ids.

use cubecl::prelude::Client;
use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::CoordinateElement;

use crate::Mesh;

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
///   cell type, with optional cell origin ids (default: the cell's index in the block). A cell's
///   vertices are its first V nodes, and a vertex's origin id is its node's.
/// - [`build`](MeshBuilder::build) checks the input, numbers vertices and nodes in order of the
///   first cell that touches them (blocks in call order; pipeline stage 8), stores coordinates
///   relative to the centre of the bounding box, and uploads.
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
        let _ = (&self.client, &self.coords, &self.node_ids);
        let _ = self
            .blocks
            .iter()
            .map(|b| (&b.element, &b.nodes, &b.origin_ids));
        let _ = Mesh::<F, TD, GD>::from_parts;
        todo!()
    }
}
