// Design: "Architecture", "API walkthrough"; Listings B3, B4 and B7.
//! `Mesh`: the mesh type, its aliases and its methods.

use cubecl::prelude::Client;
use tetraloma_device::{ConnKey, DeviceBuffer, GridError, Real, Result};
use tetraloma_dims::{Codim0, Codim1, D1, D2, D3, Dim, EmbeddedIn, TopoDim};
use tetraloma_geometry::{
    BlockFields, DeviceGrid, EVector, FacetQuadRule, FieldShape, Geometry, GeometryBlock, JacShape,
    NodeFamily, OrientationRule, PinvShape, PullBack, PullBackOptions, QuadField, QuadRule,
    ScalarShape, VecShape, detj_block, facet_normals_block, jacobians_block, measure_block,
    normals_block, physical_points_block, pseudo_inverse_block,
};
use tetraloma_parallel::{Comm, CommDyn, CommExt, IndexMap, Serial};
use tetraloma_topology::{CellBlock, Connectivity, EntityBlock, Topology};

use crate::{EntityMap, MeshTags, TagRegistry};

// Design: "Architecture", "Distributed mesh model"; Listing B3; ADR-0012, ADR-0026.
/// A mesh on a compute device: its cells, vertices, geometry nodes and tags.
///
/// The type parameters are the scalar type `F` (`f32` or `f64`), the topological dimension `TD` and
/// the geometric dimension `GD` (the markers [`D1`](crate::D1), [`D2`](crate::D2) and
/// [`D3`](crate::D3)), and the communicator `C`, [`Serial`](crate::Serial) by default. The device's
/// runtime is not a type parameter: the mesh holds a runtime-erased CubeCL client, so one program
/// type serves every device. Every method needs `TD: EmbeddedIn<GD>`, that is TD ≤ GD; some exist
/// only for particular dimensions, as [Methods by dimension] lists. A mesh is not `Clone`, because
/// a copy duplicates every device array: [`try_clone`](Mesh::try_clone) makes one explicitly.
///
/// [The mesh type] explains the type parameters, [Building a mesh] shows how to make a mesh from
/// arrays, and [Distributed meshes] how to make one on many processes.
///
/// [Methods by dimension]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#methods-by-dimension
/// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
/// [Building a mesh]: https://tetracl.github.io/tetraloma/tutorials/build-mesh.html
/// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // The unit square as one quadrilateral of degree 1, its corners in tensor order.
/// let quad = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 1, family: NodeFamily::Equispaced };
/// let mesh: Mesh<f32, D2, D2> = MeshBuilder::new(&client)
///     .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], None)
///     .cell_block(quad, &[0, 1, 2, 3], None)
///     .build()?;
/// assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 1);
/// assert!((mesh.measure(1)? - 1.0).abs() < 1e-6);
/// # Ok::<(), GridError>(())
/// ```
///
/// A distributed mesh (feature `mpi`, run under `mpirun`): every rank passes its share of the input
/// to [`MeshBuilder::build_on`](crate::MeshBuilder::build_on), which partitions it with
/// [`Block`](crate::Block) and adds one layer of ghost cells across facets.
///
/// ```no_run
/// # #[cfg(feature = "mpi")] {
/// use tetraloma::prelude::*;
///
/// let universe = mpi::initialize().expect("MPI is not initialised yet");
/// let comm = Mpi::world(&universe);
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // builder: a `MeshBuilder` with this rank's share of the unit cube in 6 · 8³ tetrahedra.
/// # let builder = tetraloma_testing::cube_box(Cell3::Tetrahedron, 8, 1)
/// #     .chunk(comm.rank(), comm.size())
/// #     .builder::<f32, D3, D3>(&client)?;
/// let mesh: VolumeMesh<f32, Mpi> = builder
///     .ghosts(GhostMode::SharedFacet)
///     .build_on(comm.clone())?;
/// let owned: usize = mesh.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
/// assert_eq!(comm.sum(owned as u64), 6 * 8 * 8 * 8);
/// assert!((mesh.measure(1)? - 1.0).abs() < 1e-5); // collective: sums the owned cells
/// # }
/// # Ok::<(), tetraloma::GridError>(())
/// ```
///
/// Dropping a distributed mesh is collective: it frees the MPI communicators of its index maps (see
/// [`IndexMap`](crate::IndexMap)), so every rank drops its meshes in the same order.
pub struct Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial> {
    pub(crate) client: Client,
    pub(crate) comm: C,
    pub(crate) topology: Topology<TD>,
    pub(crate) geometry: Geometry<F, TD, GD>,
    /// Physical and elementary MeshTags per dimension, names (ADR-0012, ADR-0022).
    pub(crate) tags: TagRegistry,
}

// Design: "Architecture", Listing B3.
/// A curve in three-dimensional space: a mesh of intervals, [`Mesh<F, D1, D3, C>`](Mesh).
///
/// [The mesh type] lists the named mesh types.
///
/// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// assert_eq!((CurveMesh::<f32>::TDIM, CurveMesh::<f32>::GDIM), (1, 3));
/// ```
pub type CurveMesh<F, C = Serial> = Mesh<F, D1, D3, C>;
// Design: "Architecture", Listing B3.
/// A planar mesh: triangles and quadrilaterals in the plane, [`Mesh<F, D2, D2, C>`](Mesh).
///
/// [The mesh type] lists the named mesh types.
///
/// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // The unit square as two triangles.
/// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
/// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
///     .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], None)
///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
///     .build()?;
/// assert_eq!((mesh.tdim(), mesh.gdim()), (2, 2));
/// # Ok::<(), GridError>(())
/// ```
pub type PlanarMesh<F, C = Serial> = Mesh<F, D2, D2, C>;
// Design: "Architecture", Listing B3.
/// A surface in three-dimensional space: triangles and quadrilaterals in space,
/// [`Mesh<F, D2, D3, C>`](Mesh).
///
/// [The mesh type] lists the named mesh types.
///
/// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // One triangle in space.
/// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
/// let mesh: SurfaceMesh<f32> = MeshBuilder::new(&client)
///     .nodes(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], None)
///     .cell_block(tri, &[0, 1, 2], None)
///     .build()?;
/// assert_eq!((mesh.tdim(), mesh.gdim()), (2, 3));
/// # Ok::<(), GridError>(())
/// ```
pub type SurfaceMesh<F, C = Serial> = Mesh<F, D2, D3, C>;
// Design: "Architecture", Listing B3.
/// A volume mesh: tetrahedra, hexahedra, prisms and pyramids in space,
/// [`Mesh<F, D3, D3, C>`](Mesh).
///
/// [The mesh type] lists the named mesh types.
///
/// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // mesh: a `VolumeMesh<f32>`, the unit cube as one hexahedron; the tutorial "Building a mesh"
/// // shows how to make a mesh.
/// # let mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build(&client)?;
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
    /// use tetraloma::prelude::*;
    ///
    /// assert_eq!(SurfaceMesh::<f32>::TDIM, 2);
    /// ```
    pub const TDIM: usize = TD::N;
    /// The geometric dimension.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
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
        let tags = TagRegistry::new(&client, TD::N, Vec::new());
        Mesh {
            client,
            comm,
            topology,
            geometry,
            tags,
        }
    }

    /// The geometry nodes `[node][GD]` as the writers write them: converted to equispaced
    /// nodes on the device if a block is GLL (`equispaced_nodes`), downloaded, and with the
    /// origin added in f64 (ADR-0024, item 7).
    pub(crate) fn equispaced_host_nodes(&self) -> Result<Vec<f64>> {
        let blocks = self.geometry.blocks();
        let nodes = if blocks.iter().any(|b| b.element.family == NodeFamily::Gll) {
            tetraloma_geometry::equispaced_nodes(self)?.to_host()?
        } else {
            self.nodes().to_host()?
        };
        let origin = self.origin();
        let origin = origin.as_ref();
        Ok(nodes
            .iter()
            .enumerate()
            .map(|(i, x)| x.widen() + origin[i % GD::N])
            .collect())
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

    /// The topological dimension TD as a number: 1 for curves, 2 for planar and surface meshes, 3
    /// for volume meshes.
    ///
    /// [The mesh type] explains the dimensions.
    ///
    /// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the icosahedron (20 flat triangles with their corners on the
    /// // unit sphere); the tutorial "Building a mesh" shows how to make a mesh.
    /// # let mesh: SurfaceMesh<f32> = tetraloma_testing::sphere(0, 1).build(&client)?;
    /// assert_eq!(mesh.tdim(), 2);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn tdim(&self) -> usize {
        TD::N
    }

    /// The geometric dimension GD as a number: the number of coordinates of a point.
    ///
    /// [The mesh type] explains the dimensions.
    ///
    /// [The mesh type]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#the-mesh-type
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the icosahedron (20 flat triangles with their corners on the
    /// // unit sphere); the tutorial "Building a mesh" shows how to make a mesh.
    /// # let mesh: SurfaceMesh<f32> = tetraloma_testing::sphere(0, 1).build(&client)?;
    /// assert_eq!(mesh.gdim(), 3);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn gdim(&self) -> usize {
        GD::N
    }

    /// The CubeCL client of the device that holds the mesh, with its runtime type erased.
    ///
    /// Device arrays that you pass to the mesh's methods live on this client. [Compute devices]
    /// shows how to open a client.
    ///
    /// [Compute devices]: https://tetracl.github.io/tetraloma/devices.html
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square as one quadrilateral; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build(&client)?;
    /// // Device arrays for this mesh go on its client.
    /// let values = DeviceBuffer::from_host(mesh.client(), &[1.0f32, 2.0])?;
    /// assert_eq!(values.len(), 2);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The mesh's tags: the physical and elementary tags of its cells and of lower-dimensional
    /// entities, and the table of tag names.
    ///
    /// A mesh read from a Gmsh file carries the file's tags; a mesh from
    /// [`MeshBuilder`](crate::MeshBuilder) carries the tags you give it, and none otherwise.
    /// [Physical groups] shows how to select tagged cells and facets.
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
    /// // mesh: a `PlanarMesh<f32>`, the unit square as two triangles; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build(&client)?;
    /// // A builder mesh without tags.
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

    /// A deep copy of the mesh: every device array, tags included, is copied on the device.
    ///
    /// A mesh is not `Clone`, so a copy of its device memory is always explicit. The copy of a
    /// distributed mesh shares the neighbourhoods of its index maps with the original, so it is not
    /// collective. [Meshes and topology] describes the arrays a mesh holds.
    ///
    /// [Meshes and topology]: https://tetracl.github.io/tetraloma/concepts/meshes.html
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the copies.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square in 2 × 2 quadrilaterals; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
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

    /// The physical points x = X(ξ) of the reference points ξ of `rule` in every cell, one field
    /// per cell block.
    ///
    /// X is the map from the reference cell to the cell. Each field holds GD absolute coordinates
    /// per cell and point, with the mesh's origin added back. [Points] shows how to read them, and
    /// [The reference-to-physical map] defines X.
    ///
    /// [Points]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#points
    /// [The reference-to-physical map]: https://tetracl.github.io/tetraloma/concepts/reference-cells.html#the-reference-to-physical-map
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if `rule` has no points for a cell type of the mesh; allocation and device
    /// errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square as one quadrilateral; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let x = mesh.physical_points(&rule)?;
    /// // The one Gauss point of the unit square is its centre; the copy to the host is for
    /// // inspection.
    /// let [x0, y0] = x.blocks[0].to_host_cell_major()?[0];
    /// assert!((x0 - 0.5).abs() < 1e-6 && (y0 - 0.5).abs() < 1e-6);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn physical_points(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        self.per_block(|b| physical_points_block(self, b, rule.get(self.cell(b))?))
    }

    /// The Jacobians J = ∂x/∂ξ at the points of `rule`: per cell and point a GD × TD matrix, stored
    /// column-major, one field per cell block.
    ///
    /// On the host, entry `[c][r]` of a point is row r of column c. [Jacobians] shows the layout on
    /// a mesh, and [The Jacobian] defines J.
    ///
    /// [Jacobians]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#jacobians
    /// [The Jacobian]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-jacobian
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square in 2 × 2 quadrilaterals; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// // Squares of side 1/2: J = I / 2.
    /// for j in mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()? {
    ///     assert!((j[0][0] - 0.5).abs() < 1e-6 && (j[1][1] - 0.5).abs() < 1e-6);
    ///     assert!(j[0][1].abs() < 1e-6 && j[1][0].abs() < 1e-6);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn jacobians(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, JacShape<TD, GD>>> {
        self.per_block(|b| jacobians_block(self, b, rule.get(self.cell(b))?))
    }

    /// The determinants det J at the points of `rule`, or the pseudo-determinants √det(JᵀJ) where
    /// TD < GD, one field per cell block.
    ///
    /// det J turns a reference quadrature weight into a physical one. [Determinants] shows it on a
    /// mesh, and [The determinant] defines both cases.
    ///
    /// [Determinants]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#determinants
    /// [The determinant]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-determinant
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square in 2 × 2 quadrilaterals; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// // Squares of side 1/2 have det J = 1/4.
    /// let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
    /// assert!(detj.iter().all(|d| (d - 0.25).abs() < 1e-6));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn detj(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, ScalarShape>> {
        self.per_block(|b| detj_block(self, b, rule.get(self.cell(b))?))
    }

    /// The pseudo-inverses K = (JᵀJ)⁻¹Jᵀ at the points of `rule`: per cell and point a TD × GD
    /// matrix, stored column-major, one field per cell block.
    ///
    /// K J is the identity, and K equals J⁻¹ when TD = GD. On a surface, Kᵀ turns a reference
    /// gradient into the surface gradient, as boundary element methods need. [Pseudo-inverses]
    /// shows it on a mesh, and [The pseudo-inverse] derives it.
    ///
    /// [Pseudo-inverses]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#pseudo-inverses
    /// [The pseudo-inverse]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-pseudo-inverse
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the icosahedron (20 flat triangles with their corners on the
    /// // unit sphere); the tutorial "Building a mesh" shows how to make a mesh.
    /// # let mesh: SurfaceMesh<f32> = tetraloma_testing::sphere(0, 1).build(&client)?;
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

    /// The measure of the mesh (its length, area or volume), with Gauss rules of degree `degree`:
    /// the sum of det J · w over the owned cells of all ranks.
    ///
    /// Each block's sum is computed on the device and finished in f64 on the host, and every rank
    /// gets the same total. Collective on a distributed mesh. [Measures] shows it, and [Measure]
    /// defines it.
    ///
    /// [Measures]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#measures
    /// [Measure]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#measure
    ///
    /// # Errors
    ///
    /// The errors of [`QuadRule::for_grid`](crate::QuadRule::for_grid) for `degree`; allocation and
    /// device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `VolumeMesh<f32>`, the unit cube in 48 tetrahedra; the tutorial "Building a mesh"
    /// // shows how to make a mesh.
    /// # let mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Tetrahedron, 2, 1).build(&client)?;
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

    /// Geometry node `i` copied to the host, with the origin added back: its GD absolute
    /// coordinates.
    ///
    /// It copies the whole node array to read one node, so it is for tests and debugging only.
    /// [Reading data back to the host] shows how to read data in bulk.
    ///
    /// [Reading data back to the host]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#reading-data-back-to-the-host
    ///
    /// # Errors
    ///
    /// `IndexOutOfRange` if `i` is not a local node; device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square as one quadrilateral; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build(&client)?;
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

    /// Creates the entities of dimension `dim` (0 ≤ dim < tdim) and the connectivity (tdim, dim)
    /// from the cells to them.
    ///
    /// An entity is a vertex, edge or face that the cells containing it share. [Cells and entities]
    /// explains them, and [Entities] shows how to read them.
    ///
    /// Collective: on more than one rank, every rank calls it with the same `dim`, and the ranks
    /// agree on the owner of each entity on a process boundary.
    ///
    /// [Cells and entities]: https://tetracl.github.io/tetraloma/concepts/meshes.html#cells-and-entities
    /// [Entities]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#entities
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` for dim ≥ tdim, because the cells are in the cell blocks;
    /// allocation, device and communication errors; `RemoteFailure` if another rank failed.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `VolumeMesh<f32>`, the unit cube as one hexahedron; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mut mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build(&client)?;
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

    /// Creates the connectivity `from` → `to`, and the entities it needs.
    ///
    /// [Connectivity] explains downward and upward connectivity, and [the tutorial] shows how to
    /// read it. Collective, as [`create_entities`](Mesh::create_entities).
    ///
    /// [Connectivity]: https://tetracl.github.io/tetraloma/concepts/meshes.html#connectivity
    /// [the tutorial]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#connectivity
    ///
    /// # Errors
    ///
    /// `UnsupportedConnectivity` for a pair tetraloma does not store; the errors of
    /// [`create_entities`](Mesh::create_entities).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `VolumeMesh<f32>`, the unit cube as one hexahedron; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mut mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build(&client)?;
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

    // Design: "Tags, regions and submeshes"; ADR-0039, item 4.
    /// The tagged cells as a mesh of the same dimensions, with the map to the parent's cells.
    ///
    /// `cells` lists cells of this mesh (`cells.dim` = tdim), for example a group from
    /// [`TagRegistry::named`](crate::TagRegistry::named). Child cells keep their owner, ghost
    /// status, origin id, flags and coordinate element. The ranks that hold a vertex or node in
    /// owned child cells agree on its new owner. The child keeps the parent's origin, has one cell
    /// block per parent block with tagged cells on some rank, and carries the parent's cell tags.
    /// [Regions] shows it on a mesh with two regions. Collective.
    ///
    /// [Regions]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#regions
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `cells.dim` is not tdim; `TooLarge` if a size or index does not fit
    /// its type; allocation, device and communication errors; `RemoteFailure` if another rank
    /// failed.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `VolumeMesh<f32>`, the unit cube in 2 × 2 × 2 hexahedra; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 2, 1).build(&client)?;
    /// // Every other cell of the eight.
    /// let cells = MeshTags {
    ///     dim: 3,
    ///     indices: DeviceBuffer::from_host(&client, &[0u32, 2, 4, 6])?,
    ///     values: DeviceBuffer::from_host(&client, &[1i32; 4])?,
    /// };
    /// let (half, map) = mesh.cell_submesh(&cells)?;
    /// assert_eq!(map.child_to_parent().to_host()?, [0, 2, 4, 6]);
    /// assert!((half.measure(2)? - 0.5).abs() < 1e-5);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn cell_submesh(&self, cells: &MeshTags) -> Result<(Self, EntityMap)> {
        crate::submesh::cell_submesh(self, cells)
    }

    // Design: "Geometry", the pull-back; ADR-0039, item 13.
    /// The reference points ξ of the physical points `x`, by Newton iteration (Gauss–Newton where
    /// TD < GD), with the size |Δξ|∞ of each point's last step.
    ///
    /// `x` holds absolute physical points per cell and point of every block, in the layout that
    /// [`physical_points`](Mesh::physical_points) returns; each cell's points are pulled back
    /// through that cell's map. The result's `reference` holds the points ξ in the same layout, and
    /// its `last_step` is above the tolerance where a point has not converged.
    /// [`PullBackOptions`](crate::PullBackOptions) sets the tolerance and the number of iterations;
    /// [`pull_back_block`](crate::pull_back_block) does the same for one block. [Pulling back
    /// points] shows it, and [The pull-back] explains the iteration and when it fails.
    ///
    /// [Pulling back points]: https://tetracl.github.io/tetraloma/tutorials/pull-back.html
    /// [The pull-back]: https://tetracl.github.io/tetraloma/concepts/pull-back.html
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `x` does not have one field per cell block, with one cell per local
    /// cell; the errors of a block's coordinate element; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the unit sphere in 80 curved triangles of degree 2; the
    /// // tutorial "Building a mesh" shows how to make a mesh.
    /// # let mesh: SurfaceMesh<f32> = tetraloma_testing::sphere(1, 2).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 4)?;
    /// let x = mesh.physical_points(&rule)?;
    /// let pb = mesh.pull_back(&x, &PullBackOptions::default())?;
    /// // The Gauss points come back on the curved triangles.
    /// let pts = rule.get(Cell2::Triangle)?;
    /// let xi = pb.reference.blocks[0].to_host_cell_major()?;
    /// for (i, p) in xi.iter().enumerate() {
    ///     let q = i % pts.len();
    ///     assert!((f64::from(p[0]) - pts.points()[2 * q]).abs() < 1e-5);
    ///     assert!((f64::from(p[1]) - pts.points()[2 * q + 1]).abs() < 1e-5);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn pull_back(
        &self,
        x: &BlockFields<F, VecShape<GD>>,
        opts: &PullBackOptions,
    ) -> Result<PullBack<F, TD>> {
        crate::pull_back::pull_back(self, x, opts)
    }

    /// The bytes this rank holds for the mesh: every device buffer (topology, entities,
    /// connectivity, geometry, tags), padding included, plus the host vectors of its index
    /// maps. The "final mesh size" of the M8 memory acceptance test (ADR-0035, rule 10).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// let client = tetraloma_testing::test_client();
    /// let mut mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Tetrahedron, 1, 2).build(&client)?;
    /// let before = mesh.footprint_bytes();
    /// assert!(before > mesh.nodes().footprint_bytes());
    /// // Created entities count too.
    /// mesh.create_entities(1)?;
    /// assert!(mesh.footprint_bytes() > before);
    /// # Ok::<(), GridError>(())
    /// ```
    #[doc(hidden)]
    pub fn footprint_bytes(&self) -> usize {
        self.topology.footprint_bytes()
            + self.geometry.footprint_bytes()
            + self.tags.footprint_bytes()
    }

    /// The mesh's topology, for tetraloma-topology's tests of the hooks that take a
    /// `&Topology` (`facet_cell_counts`). Not part of the stable API (ADR-0038; test:
    /// tetraloma-topology's `facet_cells`).
    #[doc(hidden)]
    pub fn topology(&self) -> &Topology<TD> {
        &self.topology
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + Codim0<GD>,
    GD: Dim,
    C: Comm,
{
    /// The inverse Jacobians J⁻¹ at the points of `rule`, for a mesh whose cells fill their space
    /// (TD = GD): the values of [`pseudo_inverse_jacobians`](Mesh::pseudo_inverse_jacobians).
    ///
    /// [Pseudo-inverses] shows both, and [Methods by dimension] explains why this method exists
    /// only for TD = GD.
    ///
    /// [Pseudo-inverses]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#pseudo-inverses
    /// [Methods by dimension]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#methods-by-dimension
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square in 2 × 2 quadrilaterals; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
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

    /// The outward unit normals of every facet of every cell at the points of the facet rule
    /// `rule`, one field per cell block, facet-major: per cell, the points of facet 0, then those
    /// of facet 1, and so on.
    ///
    /// It exists for meshes whose cells fill their space (TD = GD). [Facet normals] defines them,
    /// and [Normals] shows them on a mesh.
    ///
    /// [Facet normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#facet-normals
    /// [Normals]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#normals
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `PlanarMesh<f32>`, the unit square as one quadrilateral; the tutorial "Building a
    /// // mesh" shows how to make a mesh.
    /// # let mesh: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 1, 1).build(&client)?;
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
    /// The unit normals at the points of `rule` of a mesh one dimension lower than its space (TD =
    /// GD − 1), one field per cell block.
    ///
    /// Each cell's normal takes its sign from the cell's stored orientation, which
    /// [`orient`](Mesh::orient) makes consistent. [Normals] shows them on a mesh, and [the concept]
    /// defines them.
    ///
    /// [Normals]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#normals
    /// [the concept]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#normals
    ///
    /// # Errors
    ///
    /// As [`physical_points`](Mesh::physical_points).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the icosahedron (20 flat triangles with their corners on the
    /// // unit sphere); the tutorial "Building a mesh" shows how to make a mesh.
    /// # let mesh: SurfaceMesh<f32> = tetraloma_testing::sphere(0, 1).build(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
    /// assert_eq!(n.len(), 20);
    /// assert!(n.iter().all(|v| (v.iter().map(|c| c * c).sum::<f32>() - 1.0).abs() < 1e-5));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn normals(&self, rule: &QuadRule<F, TD>) -> Result<BlockFields<F, VecShape<GD>>> {
        self.per_block(|b| normals_block(self, b, rule.get(self.cell(b))?))
    }

    // Design: "Manifold orientation"; ADR-0039, item 2.
    /// Orients a mesh one dimension lower than its space consistently: writes bit 0 of every cell's
    /// flags, so that the normals of neighbouring cells agree, and ignores the flags the cells had.
    ///
    /// `rule` chooses the direction of each connected component: `Outward` from the sign of (1/GD)
    /// ∮ x·n, `AwayFrom(p)` and `Towards(p)` from the sign of ∫ (x − p)·n, each computed with a
    /// Gauss rule of degree 3p for geometry of degree p. A sum of exactly 0 flips nothing. `orient`
    /// creates the facet entities and the (tdim − 1, tdim) connectivity it needs. [Manifold
    /// orientation] explains the method, and [Orientation] shows it. Collective.
    ///
    /// [Manifold orientation]: https://tetracl.github.io/tetraloma/concepts/orientation.html#manifold-orientation
    /// [Orientation]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#orientation
    ///
    /// # Errors
    ///
    /// Agreed across ranks: `NonManifold` if a facet has more than two cells, `NonOrientable` if a
    /// component cannot be oriented consistently, `NeedsReferencePoint` for `Outward` on a
    /// component with boundary facets. Allocation, device and communication errors; `RemoteFailure`
    /// if another rank failed.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `SurfaceMesh<f32>`, the icosahedron with half of its cells flipped; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let host = tetraloma_testing::sphere(0, 1).with_flipped_cells(1, 0.5);
    /// # let mut mesh: SurfaceMesh<f32> = host.build(&client)?;
    /// mesh.orient(OrientationRule::Outward)?;
    /// let rule = QuadRule::for_grid(&mesh, 1)?;
    /// let x = mesh.physical_points(&rule)?.blocks[0].to_host_cell_major()?;
    /// let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
    /// // Every normal points away from the centre.
    /// for (x, n) in x.iter().zip(&n) {
    ///     assert!(x.iter().zip(n).map(|(a, b)| a * b).sum::<f32>() > 0.0);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn orient(&mut self, rule: OrientationRule) -> Result<()> {
        crate::orient::orient(self, rule)
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
    // Design: "Tags, regions and submeshes"; ADR-0039.
    /// The boundary facets as a mesh one dimension lower, with the map to the parent's facets.
    ///
    /// A facet is on the boundary if exactly one owned cell over all ranks contains it. The facet
    /// entities must exist: call `create_entities(tdim − 1)` first. A child one dimension lower
    /// than its space (the surface of a volume mesh, the boundary curve of a planar mesh) is
    /// oriented outward through its flags; a child two dimensions lower (the boundary of a surface
    /// in 3D) has flags 0. Child cells keep their facet's owner and ghost status, and the ranks
    /// that hold a vertex or node in owned child cells agree on its new owner. The child keeps the
    /// parent's origin and vertex origin ids, and the parent's facet tags become its cell tags.
    /// [Boundary meshes] shows it. Collective.
    ///
    /// [Boundary meshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#boundary-meshes
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(tdim − 1)` before `create_entities(tdim − 1)`. Agreed across ranks:
    /// `NonManifold` if a facet has more than two cells; `TooLarge` if a child cell's origin id, 8
    /// · c + f for facet f of the parent cell with origin id c, does not fit i64;
    /// `UnsupportedDegree` or `UnsupportedNodeFamily` if the parent cells of one facet type differ
    /// in degree or node family. Allocation, device and communication errors; `RemoteFailure` if
    /// another rank failed.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // volume: a `VolumeMesh<f32>`, the unit cube in 2 × 2 × 2 hexahedra; the tutorial "Building
    /// // a mesh" shows how to make a mesh.
    /// # let mut volume: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 2, 1).build(&client)?;
    /// volume.create_entities(2)?;
    /// let (mut surface, map) = volume.boundary()?;
    /// // Six faces of 2 × 2 quadrilaterals, each with one parent facet.
    /// assert_eq!(surface.cell_blocks()[0].index_map.num_local(), 24);
    /// assert_eq!(map.child_to_parent().len(), 24);
    /// assert!((surface.measure(1)? - 6.0).abs() < 1e-5);
    /// // `volume.boundary().boundary()` compiles; a closed surface has an empty boundary.
    /// surface.create_entities(1)?;
    /// let (curve, _) = surface.boundary()?;
    /// assert!(curve.cell_blocks().is_empty());
    /// # Ok::<(), GridError>(())
    /// ```
    #[allow(clippy::type_complexity)] // Listing B4's signature
    pub fn boundary(&self) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)> {
        crate::submesh::boundary(self)
    }

    // Design: "Tags, regions and submeshes", Listing B9; ADR-0039.
    /// The tagged boundary facets as a mesh one dimension lower, with the map to the parent's
    /// facets: [`boundary`](Mesh::boundary) for a subset of the boundary facets.
    ///
    /// `facets` lists facets of this mesh (`facets.dim` = tdim − 1), for example a group from
    /// [`TagRegistry::named`](crate::TagRegistry::named) with [`Codim::Bnd`](crate::Codim::Bnd).
    /// [Surface meshes] shows it. Collective.
    ///
    /// [Surface meshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#surface-meshes
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `facets.dim` is not tdim − 1; agreed across ranks: `InteriorFacet` if
    /// a tagged facet has two cells and `NonManifold` if a facet has more; the errors of
    /// [`boundary`](Mesh::boundary).
    ///
    /// # Example
    ///
    /// The surface of a box whose boundary facets carry physical tags:
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // head: a `VolumeMesh<f32>`, the unit cube in 48 tetrahedra of degree 2, whose boundary
    /// // facets carry the tags 100 and 101 alternately, named "boundary-100" and "boundary-101";
    /// // the tutorial "Tags and submeshes" shows how to tag a mesh.
    /// # let mut head: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Tetrahedron, 2, 2)
    /// #     .chunk(0, 1)
    /// #     .builder::<f32, D3, D3>(&client)?
    /// #     .build()?;
    /// head.create_entities(2)?;
    /// let scalp_tag = head.tags().named(Codim::Bnd, "boundary-100")?; // boundary facets only
    /// let (scalp, emap): (SurfaceMesh<f32>, EntityMap) = head.facet_submesh(&scalp_tag)?;
    /// // Each facet has one parent cell; normals inherit outward orientation from it.
    /// assert_eq!(emap.child_to_parent().len(), scalp_tag.indices.len());
    ///
    /// let rule = QuadRule::for_grid(&scalp, 6)?; // one point set per cell type present
    /// let n = scalp.normals(&rule)?; // compiles: D2 is Codim1 in D3
    /// let j = scalp.jacobians(&rule)?; // per point two columns of three
    /// let k = scalp.pseudo_inverse_jacobians(&rule)?; // for surface gradients
    /// let area = scalp.measure(6)?; // the pseudo-determinant, chosen at compile time
    /// assert_eq!((n.blocks.len(), j.blocks.len(), k.blocks.len()), (1, 1, 1));
    /// assert!(area > 0.0 && area < 6.0);
    /// // The child carries the tagged facets as cell tags.
    /// assert_eq!(scalp.tags().group(2, 100)?.indices.len(), scalp_tag.indices.len());
    /// # Ok::<(), GridError>(())
    /// ```
    #[allow(clippy::type_complexity)] // Listing B4's signature
    pub fn facet_submesh(
        &self,
        facets: &MeshTags,
    ) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)> {
        crate::submesh::facet_submesh(self, facets)
    }
}

impl<F: Real, C: Comm> Mesh<F, D3, D3, C> {
    // Design: "Tags, regions and submeshes"; ADR-0039, items 3 and 7.
    /// The tagged edges, interior or boundary, as a curve mesh in 3D, with the map to the parent's
    /// edges.
    ///
    /// `edges` lists edges of this mesh (`edges.dim` = 1). The edges and the edge-to-cell
    /// connectivity must exist: call `create_entities(1)` and `create_connectivity(1, 3)` first.
    /// Child cells run along their edge from its lower vertex origin id to its higher one, and have
    /// flags 0. A child cell's origin id is the smallest 12 · c + e over the cells that contain its
    /// edge, for edge e of the cell with origin id c. Its geometry is the parent's along the edge,
    /// with the parent's degree and node family. The parent's edge tags become its cell tags.
    /// [Boundary meshes] shows it. Collective.
    ///
    /// [Boundary meshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#boundary-meshes
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `edges.dim` is not 1; `EntitiesNotCreated(1)`,
    /// `ConnectivityNotCreated`; agreed across ranks: `TooLarge` if a child origin id does not fit
    /// i64, `UnsupportedDegree` or `UnsupportedNodeFamily` if the cells that give the child nodes
    /// differ in degree or node family. Allocation, device and communication errors;
    /// `RemoteFailure` if another rank failed.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // mesh: a `VolumeMesh<f32>`, the unit cube as one hexahedron of degree 2; the tutorial
    /// // "Building a mesh" shows how to make a mesh.
    /// # let mut mesh: VolumeMesh<f32> = tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 2).build(&client)?;
    /// mesh.create_connectivity(1, 3)?;
    /// // All twelve edges of the unit cube.
    /// let edges = MeshTags {
    ///     dim: 1,
    ///     indices: DeviceBuffer::from_host(&client, &(0..12).collect::<Vec<u32>>())?,
    ///     values: DeviceBuffer::from_host(&client, &[1i32; 12])?,
    /// };
    /// let (frame, map) = mesh.edge_submesh(&edges)?;
    /// assert_eq!(map.parent_dim(), 1);
    /// assert_eq!(frame.vertex_index_map().num_local(), 8);
    /// assert!((frame.measure(4)? - 12.0).abs() < 1e-4);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn edge_submesh(&self, edges: &MeshTags) -> Result<(CurveMesh<F, C>, EntityMap)> {
        crate::submesh::edge_submesh(self, edges)
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

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use cubecl::prelude::Client;
    use tetraloma_device::VECTOR_PAD;

    /// Two tagged triangles; every array has at most `VECTOR_PAD` entries, so each takes one
    /// padded block.
    fn two_triangles<F: Real>(client: &Client) -> Mesh<F, D2, D2> {
        let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let tri = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        MeshBuilder::new(client)
            .nodes(&coords, Some(&[10, 11, 12, 13]))
            .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
            .physical_tags(CellType::Triangle, &[13, 12, 11], &[5])
            .physical_tags(CellType::Interval, &[10, 11, 13, 11], &[1, 1])
            .build()
            .expect("two triangles")
    }

    fn footprint_counts_every_buffer<F: Real>(client: &Client) {
        let block = |bytes: usize| VECTOR_PAD * bytes;
        let mut mesh = two_triangles::<F>(client);
        // Serial index maps hold no host vectors.
        let topology = 3 * block(4) // cell vertices, cell origin ids, flags
            + block(4) // vertex origin ids
            + block(4) // permutation info
            + block(4) // vertex shared flags
            + 2 * block(4); // edges (the tagged edges create them): vertices, cell -> edge
        let geometry = block(4) + 2 * block(size_of::<F>()); // dofmap; nodes, origin
        let tags = 4 * block(4); // indices and values of dimensions 2 and 1
        assert_eq!(mesh.footprint_bytes(), topology + geometry + tags);
        // Edge -> cell: 5 + 1 offsets and 6 indices.
        mesh.create_connectivity(1, 2).expect("edge -> cell");
        assert_eq!(
            mesh.footprint_bytes(),
            topology + geometry + tags + 2 * block(4)
        );
    }

    #[test]
    fn footprint_counts_every_buffer_f32() {
        footprint_counts_every_buffer::<f32>(&tetraloma_testing::test_client());
    }

    #[test]
    fn footprint_counts_every_buffer_f64() {
        let client = tetraloma_testing::test_client();
        if tetraloma_testing::supports::<f64>(&client) {
            footprint_counts_every_buffer::<f64>(&client);
        }
    }
}
