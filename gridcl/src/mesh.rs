//! `Mesh` (Listings B3, B4 and B7).

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, Real, Result};
use gridcl_dims::{Codim0, Codim1, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{
    BlockFields, DeviceGrid, EVector, FacetQuadRule, Geometry, GeometryBlock, JacShape, PinvShape,
    QuadRule, ScalarShape, VecShape,
};
use gridcl_parallel::{Comm, CommDyn, IndexMap, Serial};
use gridcl_topology::{CellBlock, Connectivity, EntityBlock, Topology};

use crate::EntityMap;

/// A mesh, generic over scalar, dimensions and communicator, not over the backend
/// (Listing B3). Constructors and methods require `TD: EmbeddedIn<GD>`. Not `Clone`: see
/// [`try_clone`](Mesh::try_clone).
pub struct Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial> {
    client: Client,
    comm: C,
    topology: Topology<TD>,
    geometry: Geometry<F, TD, GD>,
}

/// A curve in 3D.
pub type CurveMesh<F, C = Serial> = Mesh<F, D1, D3, C>;
/// A planar mesh.
pub type PlanarMesh<F, C = Serial> = Mesh<F, D2, D2, C>;
/// A surface in 3D.
pub type SurfaceMesh<F, C = Serial> = Mesh<F, D2, D3, C>;
/// A volume mesh.
pub type VolumeMesh<F, C = Serial> = Mesh<F, D3, D3, C>;

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    /// The topological dimension.
    pub const TDIM: usize = TD::N;
    /// The geometric dimension.
    pub const GDIM: usize = GD::N;

    /// A mesh from its parts. `MeshBuilder` is the usual way to make one.
    pub(crate) fn from_parts(
        client: Client,
        comm: C,
        topology: Topology<TD>,
        geometry: Geometry<F, TD, GD>,
    ) -> Self {
        Mesh {
            client,
            comm,
            topology,
            geometry,
        }
    }

    /// The topological dimension.
    pub fn tdim(&self) -> usize {
        TD::N
    }

    /// The geometric dimension.
    pub fn gdim(&self) -> usize {
        GD::N
    }

    /// The runtime-erased client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// A deep copy of every device array.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        todo!()
    }

    /// Physical points at the points of `rule`, one field per cell block.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if `rule` lacks a cell type of the mesh; allocation and device errors.
    pub fn physical_points(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        let _ = rule;
        todo!()
    }

    /// Jacobians at the points of `rule`: per point GD × TD, column-major.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, JacShape<TD, GD>>> {
        let _ = rule;
        todo!()
    }

    /// det J, or sqrt(det(JᵀJ)) when TD < GD.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn detj(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, ScalarShape>> {
        let _ = rule;
        todo!()
    }

    /// K = (JᵀJ)⁻¹Jᵀ; equals J⁻¹ in codimension 0. Needed for surface gradients in BEM.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn pseudo_inverse_jacobians(
        &self,
        rule: &QuadRule<F, TD>,
    ) -> Result<BlockFields<F, PinvShape<TD, GD>>> {
        let _ = rule;
        todo!()
    }

    /// Sum over owned cells on all ranks, finished in f64 on the host, with Gauss rules of
    /// `degree`.
    ///
    /// # Errors
    ///
    /// Allocation and device errors.
    pub fn measure(&self, degree: u32) -> Result<f64> {
        let _ = degree;
        todo!()
    }

    /// One node copied to the host, with the origin added back. Tests and debugging only: one
    /// device read and one sync per call.
    ///
    /// # Errors
    ///
    /// `IndexOutOfRange` if `i` is not a local node; device errors.
    pub fn read_node(&self, i: usize) -> Result<GD::Array<F>> {
        let _ = i;
        todo!()
    }

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and the connectivity
    /// (tdim, dim).
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim; allocation and device errors.
    pub fn create_entities(&mut self, dim: usize) -> Result<()> {
        let _ = dim;
        todo!()
    }

    /// Creates the connectivity `from` → `to` and the entities it needs.
    ///
    /// # Errors
    ///
    /// `UnsupportedConnectivity` for pairs gridcl does not store; errors of
    /// [`create_entities`](Mesh::create_entities).
    pub fn create_connectivity(&mut self, from: usize, to: usize) -> Result<()> {
        let _ = (from, to);
        todo!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + Codim0<GD>,
    GD: Dim,
    C: Comm,
{
    /// J⁻¹ at the points of `rule`.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn inverse_jacobians(
        &self,
        rule: &QuadRule<F, TD>,
    ) -> Result<BlockFields<F, PinvShape<TD, GD>>> {
        let _ = rule;
        todo!()
    }

    /// Outward unit normals of every local facet of every cell, at facet quadrature points
    /// (facet-major).
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn facet_normals(
        &self,
        rule: &FacetQuadRule<F, TD>,
    ) -> Result<BlockFields<F, VecShape<GD>>> {
        let _ = rule;
        todo!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    /// Unit normals; sign from the stored manifold orientation.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    pub fn normals(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        let _ = rule;
        todo!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    /// All boundary facets as a mesh one dimension lower, oriented outward.
    ///
    /// # Errors
    ///
    /// Returns `GridError::NotYetImplemented` until M9 (ADR-0012).
    #[allow(clippy::type_complexity)] // Listing B4's signature
    pub fn boundary(&self) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)> {
        todo!()
    }
}

impl<F, TD, GD, C> DeviceGrid for Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    type F = F;
    type TD = TD;
    type GD = GD;

    fn client(&self) -> &Client {
        &self.client
    }
    fn comm(&self) -> &dyn CommDyn {
        &self.comm
    }
    fn cell_blocks(&self) -> &[CellBlock<TD>] {
        self.topology.cell_blocks()
    }
    fn geometry_block(&self, b: usize) -> &GeometryBlock<TD> {
        &self.geometry.blocks()[b]
    }
    fn nodes(&self) -> &DeviceBuffer<F> {
        self.geometry.nodes()
    }
    fn origin(&self) -> GD::Array<f64> {
        self.geometry.origin()
    }
    fn evector(&self, b: usize) -> Result<&EVector<F>> {
        self.geometry.evector(b)
    }
    fn node_index_map(&self) -> &IndexMap {
        self.geometry.node_index_map()
    }
    fn vertex_index_map(&self) -> &IndexMap {
        self.topology.vertex_index_map()
    }
    fn vertex_origin_ids(&self) -> &DeviceBuffer<u32> {
        self.topology.vertex_origin_ids()
    }
    fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]> {
        self.topology.entity_blocks(dim)
    }
    fn connectivity(&self, key: ConnKey) -> Result<&Connectivity> {
        self.topology.connectivity(key)
    }
    fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>> {
        self.topology.permutation_info(b)
    }
}
