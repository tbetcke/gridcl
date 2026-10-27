//! `Mesh` (Listings B3, B4 and B7).

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, GridError, Real, Result};
use gridcl_dims::{Codim0, Codim1, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{
    BlockFields, DeviceGrid, EVector, FacetQuadRule, FieldShape, Geometry, GeometryBlock, JacShape,
    PinvShape, QuadField, QuadRule, ScalarShape, VecShape, detj_block, facet_normals_block,
    jacobians_block, measure_block, normals_block, physical_points_block, pseudo_inverse_block,
};
use gridcl_parallel::{Comm, CommDyn, CommExt, IndexMap, Serial};
use gridcl_topology::{CellBlock, Connectivity, EntityBlock, Topology};

use crate::{EntityMap, TagRegistry};

/// A mesh, generic over scalar, dimensions and communicator, not over the backend
/// (Listing B3). Constructors and methods require `TD: EmbeddedIn<GD>`. Not `Clone`: see
/// [`try_clone`](Mesh::try_clone).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh: Mesh<f32, D2, D2> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
/// assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 4);
/// assert!((mesh.measure(1)? - 1.0).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
pub struct Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial> {
    client: Client,
    comm: C,
    topology: Topology<TD>,
    geometry: Geometry<F, TD, GD>,
    /// Physical and elementary MeshTags per dimension, names (ADR-0012, ADR-0022).
    tags: TagRegistry,
}

/// A curve in 3D.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// assert_eq!((CurveMesh::<f32>::TDIM, CurveMesh::<f32>::GDIM), (1, 3));
/// ```
pub type CurveMesh<F, C = Serial> = Mesh<F, D1, D3, C>;
/// A planar mesh.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build(&client)?;
/// assert_eq!((mesh.tdim(), mesh.gdim()), (2, 2));
/// # Ok::<(), GridError>(())
/// ```
pub type PlanarMesh<F, C = Serial> = Mesh<F, D2, D2, C>;
/// A surface in 3D.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh: SurfaceMesh<f32> = gridcl_testing::sphere(0, 1).build(&client)?;
/// assert_eq!((mesh.tdim(), mesh.gdim()), (2, 3));
/// # Ok::<(), GridError>(())
/// ```
pub type SurfaceMesh<F, C = Serial> = Mesh<F, D2, D3, C>;
/// A volume mesh.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mesh: VolumeMesh<f32> = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 1).build(&client)?;
/// assert_eq!((mesh.tdim(), mesh.gdim()), (3, 3));
/// # Ok::<(), GridError>(())
/// ```
pub type VolumeMesh<F, C = Serial> = Mesh<F, D3, D3, C>;

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    /// The topological dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert_eq!(SurfaceMesh::<f32>::TDIM, 2);
    /// ```
    pub const TDIM: usize = TD::N;
    /// The geometric dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert_eq!(SurfaceMesh::<f32>::GDIM, 3);
    /// ```
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
            tags: TagRegistry::default(),
        }
    }

    /// The cell type of block `b`.
    fn cell(&self, b: usize) -> TD::Cell {
        self.topology.cell_blocks()[b].cell
    }

    /// One field per cell block.
    fn per_block<S: FieldShape>(
        &self,
        f: impl Fn(usize) -> Result<QuadField<F, S>>,
    ) -> Result<BlockFields<F, S>> {
        let blocks = (0..self.topology.cell_blocks().len())
            .map(f)
            .collect::<Result<_>>()?;
        Ok(BlockFields { blocks })
    }

    /// The topological dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: SurfaceMesh<f32> = gridcl_testing::sphere(0, 1).build(&client)?;
    /// assert_eq!(mesh.tdim(), 2);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn tdim(&self) -> usize {
        TD::N
    }

    /// The geometric dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: SurfaceMesh<f32> = gridcl_testing::sphere(0, 1).build(&client)?;
    /// assert_eq!(mesh.gdim(), 3);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn gdim(&self) -> usize {
        GD::N
    }

    /// The runtime-erased client.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
    /// // Device arrays for this mesh go on its client.
    /// let values = DeviceBuffer::from_host(mesh.client(), &[1.0f32, 2.0])?;
    /// assert_eq!(values.len(), 2);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Physical and elementary tags and the name table (empty unless read from a file).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// // A builder mesh has no tags.
    /// assert!(mesh.tags().physical(1).is_none());
    /// assert_eq!(mesh.tags().names().count(), 0);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn tags(&self) -> &TagRegistry {
        &self.tags
    }

    /// Replaces the tags (the Gmsh reader).
    pub(crate) fn set_tags(&mut self, tags: TagRegistry) {
        self.tags = tags;
    }

    /// The topology (the Gmsh reader matches tag keys against it).
    pub(crate) fn topology(&self) -> &Topology<TD> {
        &self.topology
    }

    /// A deep copy of every device array, tags included.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let copy = mesh.try_clone()?;
    /// assert_eq!(copy.nodes().to_host()?, mesh.nodes().to_host()?);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn try_clone(&self) -> Result<Self> {
        Ok(Mesh {
            client: self.client.clone(),
            comm: self.comm.clone(),
            topology: self.topology.try_clone(&self.comm)?,
            geometry: self.geometry.try_clone(&self.comm)?,
            tags: self.tags.try_clone()?,
        })
    }

    /// Physical points at the points of `rule`, one field per cell block.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if `rule` lacks a cell type of the mesh; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let x = mesh.physical_points(&rule)?;
    /// // The one Gauss point of the unit square is its centre.
    /// let [x0, y0] = x.blocks[0].to_host_cell_major()?[0];
    /// assert!((x0 - 0.5).abs() < 1e-6 && (y0 - 0.5).abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn physical_points(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        self.per_block(|b| physical_points_block(self, b, rule.get(self.cell(b))?))
    }

    /// Jacobians at the points of `rule`: per point GD × TD, column-major.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// // Squares of side 1/2: J = I / 2, column-major.
    /// for j in mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()? {
    ///     assert!((j[0][0] - 0.5).abs() < 1e-6 && (j[1][1] - 0.5).abs() < 1e-6);
    ///     assert!(j[0][1].abs() < 1e-6 && j[1][0].abs() < 1e-6);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, JacShape<TD, GD>>> {
        self.per_block(|b| jacobians_block(self, b, rule.get(self.cell(b))?))
    }

    /// det J, or sqrt(det(JᵀJ)) when TD < GD.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
    /// assert!(detj.iter().all(|d| (d - 0.25).abs() < 1e-6));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn detj(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, ScalarShape>> {
        self.per_block(|b| detj_block(self, b, rule.get(self.cell(b))?))
    }

    /// K = (JᵀJ)⁻¹Jᵀ; equals J⁻¹ in codimension 0. Needed for surface gradients in BEM.
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: SurfaceMesh<f32> = gridcl_testing::sphere(0, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let j = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
    /// let k = mesh.pseudo_inverse_jacobians(&rule)?.blocks[0].to_host_cell_major()?;
    /// // K J = I (2 × 2) on every face; K is 2 × 3 and J is 3 × 2, both column-major.
    /// for (k, j) in k.iter().zip(&j) {
    ///     for a in 0..2 {
    ///         for b in 0..2 {
    ///             let kj: f32 = (0..3).map(|g| k[g][a] * j[b][g]).sum();
    ///             assert!((kj - if a == b { 1.0 } else { 0.0 }).abs() < 1e-5);
    ///         }
    ///     }
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn pseudo_inverse_jacobians(
        &self,
        rule: &QuadRule<F, TD>,
    ) -> Result<BlockFields<F, PinvShape<TD, GD>>> {
        self.per_block(|b| pseudo_inverse_block(self, b, rule.get(self.cell(b))?))
    }

    /// Sum over owned cells on all ranks, finished in f64 on the host, with Gauss rules of
    /// `degree`.
    ///
    /// # Errors
    ///
    /// Allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 1).build::<f32, D3, D3>(&client)?;
    /// assert!((mesh.measure(1)? - 1.0).abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn measure(&self, degree: u32) -> Result<f64> {
        let rule = QuadRule::for_grid(self, degree)?;
        let mut local = 0.0;
        for b in 0..self.topology.cell_blocks().len() {
            local += measure_block(self, b, rule.get(self.cell(b))?)?;
        }
        Ok(self.comm.sum(local))
    }

    /// One node copied to the host, with the origin added back. Tests and debugging only: one
    /// device read and one sync per call.
    ///
    /// # Errors
    ///
    /// `IndexOutOfRange` if `i` is not a local node; device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
    /// // The four corners of the unit square.
    /// for i in 0..4 {
    ///     let x = mesh.read_node(i)?;
    ///     assert!(x.iter().all(|c| *c == 0.0 || *c == 1.0));
    /// }
    /// assert!(matches!(mesh.read_node(4), Err(GridError::IndexOutOfRange { index: 4, len: 4 })));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn read_node(&self, i: usize) -> Result<GD::Array<F>> {
        let num_nodes = self.geometry.node_index_map().num_local();
        if i >= num_nodes {
            return Err(GridError::IndexOutOfRange {
                index: i,
                len: num_nodes,
            });
        }
        let nodes = self.geometry.nodes().to_host()?;
        let origin = self.geometry.origin();
        let mut x = GD::Array::<F>::default();
        for (g, out) in x.as_mut().iter_mut().enumerate() {
            *out = F::narrow(nodes[i * GD::N + g].widen() + origin.as_ref()[g]);
        }
        Ok(x)
    }

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and the connectivity
    /// (tdim, dim).
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?;
    /// mesh.create_entities(1)?;
    /// mesh.create_entities(2)?;
    /// // One cube: 12 edges and 6 faces.
    /// assert_eq!(mesh.entity_blocks(1)?[0].index_map.num_local(), 12);
    /// assert_eq!(mesh.entity_blocks(2)?[0].index_map.num_local(), 6);
    /// // Cells are in `cell_blocks()`, not entity blocks.
    /// assert!(mesh.create_entities(3).is_err());
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn create_entities(&mut self, dim: usize) -> Result<()> {
        self.topology.create_entities(&self.comm, dim)
    }

    /// Creates the connectivity `from` → `to` and the entities it needs.
    ///
    /// # Errors
    ///
    /// `UnsupportedConnectivity` for pairs gridcl does not store; errors of
    /// [`create_entities`](Mesh::create_entities).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?;
    /// mesh.create_connectivity(1, 3)?;
    /// let Connectivity::Upward(csr) = mesh.connectivity(ConnKey { from: 1, to: 3 })? else {
    ///     unreachable!("edge -> cell is upward")
    /// };
    /// // 12 edges with one cell each.
    /// assert_eq!((csr.offsets.len(), csr.indices.len()), (13, 12));
    /// // The entities it needed exist now.
    /// assert_eq!(mesh.entity_blocks(1)?[0].index_map.num_local(), 12);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn create_connectivity(&mut self, from: usize, to: usize) -> Result<()> {
        self.topology.create_connectivity(&self.comm, from, to)
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// // J = I / 2, so J⁻¹ = 2 I.
    /// for k in mesh.inverse_jacobians(&rule)?.blocks[0].to_host_cell_major()? {
    ///     assert!((k[0][0] - 2.0).abs() < 1e-5 && (k[1][1] - 2.0).abs() < 1e-5);
    ///     assert!(k[0][1].abs() < 1e-5 && k[1][0].abs() < 1e-5);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn inverse_jacobians(
        &self,
        rule: &QuadRule<F, TD>,
    ) -> Result<BlockFields<F, PinvShape<TD, GD>>> {
        self.pseudo_inverse_jacobians(rule)
    }

    /// Outward unit normals of every local facet of every cell, at facet quadrature points
    /// (facet-major).
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::square_box(Cell2::Quadrilateral, 1, 1).build::<f32, D2, D2>(&client)?;
    /// let rule = FacetQuadRule::for_grid(&mesh, 1)?;
    /// let n = mesh.facet_normals(&rule)?.blocks[0].to_host_cell_major()?;
    /// // One point per side: four outward unit normals that sum to zero.
    /// assert_eq!(n.len(), 4);
    /// assert!(n.iter().all(|v| ((v[0] * v[0] + v[1] * v[1]) - 1.0).abs() < 1e-6));
    /// let sum = n.iter().fold([0.0f32; 2], |s, v| [s[0] + v[0], s[1] + v[1]]);
    /// assert!(sum[0].abs() < 1e-6 && sum[1].abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn facet_normals(
        &self,
        rule: &FacetQuadRule<F, TD>,
    ) -> Result<BlockFields<F, VecShape<GD>>> {
        self.per_block(|b| facet_normals_block(self, b, rule.get(self.cell(b))?))
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh: SurfaceMesh<f32> = gridcl_testing::sphere(0, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
    /// assert_eq!(n.len(), 20);
    /// assert!(n.iter().all(|v| (v.iter().map(|c| c * c).sum::<f32>() - 1.0).abs() < 1e-5));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn normals(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        self.per_block(|b| normals_block(self, b, rule.get(self.cell(b))?))
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?;
    /// assert!(matches!(mesh.boundary(), Err(GridError::NotYetImplemented(_))));
    /// # Ok::<(), GridError>(())
    /// ```
    #[allow(clippy::type_complexity)] // Listing B4's signature
    pub fn boundary(&self) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)> {
        Err(GridError::NotYetImplemented("boundary: M9"))
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
