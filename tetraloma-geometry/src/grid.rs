//! `DeviceGrid` (Listing B7; tetraloma-geometry, re-exported by tetraloma; ADR-0012).

use cubecl::prelude::Client;
use tetraloma_device::{ConnKey, DeviceBuffer, Real, Result};
use tetraloma_dims::{Dim, EmbeddedIn, TopoDim};
use tetraloma_parallel::{CommDyn, IndexMap};
use tetraloma_topology::{CellBlock, Connectivity, EntityBlock};

use crate::{EVector, GeometryBlock};

// Design: "Geometry", "Generic algorithms over any grid", Listing B7; ADR-0012.
/// The device view of a grid that finite and boundary element codes program against:
/// its client and communicator, its cell blocks with their geometry, its nodes, entities and
/// connectivity.
///
/// `Mesh` and its submeshes implement it; the geometric factor functions of this crate
/// ([`jacobians_block`](crate::jacobians_block) and the others) and code of your own can take
/// any `G: DeviceGrid`. [Writing kernels] shows a kernel and a generic function over it.
///
/// The accessors create no topology: edges, faces and connectivity exist only after the `&mut
/// self` methods `create_entities` and `create_connectivity` of `Mesh` have built them. The
/// E-vectors are built on first use and cached.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// /// The local cells of any grid.
/// fn num_cells(g: &impl DeviceGrid) -> usize {
///     g.cell_blocks().iter().map(|b| b.index_map.num_local()).sum()
/// }
///
/// // `mesh`: the unit square in 2 × 2 squares of two triangles each, on the test device; the
/// // tutorial "Building a mesh" shows how to make one.
/// # let client = tetraloma_testing::test_client();
/// # let mut mesh = tetraloma_testing::square_box(Cell2::Triangle, 2, 1).build::<f32, D2, D2>(&client)?;
/// assert_eq!(num_cells(&mesh), 8);
/// // Accessors create no topology.
/// assert!(mesh.entity_blocks(1).is_err());
/// mesh.create_entities(1)?;
/// // 9 vertices and 8 triangles: 16 edges.
/// assert_eq!(mesh.entity_blocks(1)?[0].index_map.num_local(), 16);
/// # Ok::<(), GridError>(())
/// ```
///
/// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#devicegrid
pub trait DeviceGrid {
    /// The scalar of the geometry.
    type F: Real;
    /// The topological dimension.
    type TD: TopoDim + EmbeddedIn<Self::GD>;
    /// The geometric dimension.
    type GD: Dim;

    /// The CubeCL client of the device that holds the grid's arrays.
    ///
    /// Kernels that read the grid launch on it; see [Writing kernels] and [Compute devices].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// // Arrays for the grid's kernels go on the grid's client.
    /// let ones = DeviceBuffer::from_host(DeviceGrid::client(&mesh), &[1.0f32; 4])?;
    /// assert_eq!(ones.len(), 4);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#devicegrid
    /// [Compute devices]: https://tetracl.github.io/tetraloma/devices.html
    fn client(&self) -> &Client;
    /// The communicator of the grid: `Serial` for a mesh on one process.
    ///
    /// Collective operations on the grid run over it; see [Building on a communicator].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// assert_eq!((mesh.comm().rank(), mesh.comm().size()), (0, 1));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Building on a communicator]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#building-on-a-communicator
    fn comm(&self) -> &dyn CommDyn;
    /// The cell blocks, one per cell type, in creation order.
    ///
    /// Each holds the cell type, the index map of the block's cells, their vertices, origin ids
    /// and orientation flags; see [Blocks].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let blocks = mesh.cell_blocks();
    /// assert_eq!(blocks.len(), 1);
    /// assert_eq!(blocks[0].cell, Cell2::Quadrilateral);
    /// assert_eq!(blocks[0].index_map.num_local(), 4);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
    fn cell_blocks(&self) -> &[CellBlock<Self::TD>];
    /// The geometry of cell block `b`: its coordinate element and dofmap.
    ///
    /// See [Blocks] and [Writing kernels].
    ///
    /// # Panics
    ///
    /// If there is no block `b`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let gb = mesh.geometry_block(0);
    /// assert_eq!((gb.element.cell, gb.element.degree), (Cell2::Quadrilateral, 1));
    /// assert_eq!(gb.num_cells(), 4);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
    /// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#geometryblock
    fn geometry_block(&self, b: usize) -> &GeometryBlock<Self::TD>;
    /// The node coordinates on the device: `gdim` values per local node, node after node,
    /// relative to the [`origin`](DeviceGrid::origin).
    ///
    /// See [Reading data back to the host] and, for why the coordinates are relative,
    /// [Tolerances].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// // 3 × 3 nodes. The copy to the host is for inspection.
    /// let x = mesh.nodes().to_host()?;
    /// assert_eq!(x.len(), 9 * 2);
    /// // The origin is the centre (1/2, 1/2), so every coordinate is -1/2, 0 or 1/2.
    /// assert!(x.iter().all(|c| [-0.5, 0.0, 0.5].contains(c)));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Reading data back to the host]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#reading-data-back-to-the-host
    /// [Tolerances]: https://tetracl.github.io/tetraloma/concepts/precision.html#tolerances
    fn nodes(&self) -> &DeviceBuffer<Self::F>;
    /// The origin of the node coordinates, in f64: the centre of the global bounding box for a
    /// mesh from `MeshBuilder` or a reader, the same on every rank.
    ///
    /// `physical_points` and `read_node` add it back; see [Tolerances].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// assert_eq!(mesh.origin(), [0.5, 0.5]);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Tolerances]: https://tetracl.github.io/tetraloma/concepts/precision.html#tolerances
    fn origin(&self) -> <Self::GD as Dim>::Array<f64>;
    /// The E-vector of block `b`, the node coordinates gathered per cell: built on the device on
    /// first use, then cached.
    ///
    /// See [`EVector`] and [Writing kernels].
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the first build.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let xe = mesh.evector(0)?;
    /// assert_eq!(xe.num_cells(), 4);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#geometryblock
    fn evector(&self, b: usize) -> Result<&EVector<Self::F>>;
    /// The index map of the nodes: the owned nodes first, then the ghosts.
    ///
    /// See [Index maps].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let map = mesh.node_index_map();
    /// assert_eq!((map.num_owned(), map.num_ghosts()), (9, 0));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Index maps]: https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#index-maps
    fn node_index_map(&self) -> &IndexMap;
    /// The index map of the vertices: the owned vertices first, then the ghosts.
    ///
    /// On a mesh of degree 1 the vertices are the nodes; on a curved mesh they are the nodes at
    /// the cells' corners. See [Ownership and ghosts].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
    /// // 9 vertices; on a mesh of degree 2 there would be 25 nodes.
    /// assert_eq!(mesh.vertex_index_map().num_local(), 9);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Ownership and ghosts]: https://tetracl.github.io/tetraloma/concepts/meshes.html#ownership-and-ghosts
    fn vertex_index_map(&self) -> &IndexMap;
    /// The origin id of every local vertex on the device, as two `u32` words per vertex: the
    /// high word of the id with its top bit flipped, then the low word.
    ///
    /// With the bit flipped, comparing the word pairs as unsigned numbers gives the order of
    /// the signed ids. See [Origin ids].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // The unit square as two triangles, with node origin ids 10 to 13.
    /// let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let mesh: PlanarMesh<f32> = MeshBuilder::new(&client)
    ///     .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], Some(&[10, 11, 12, 13]))
    ///     .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    ///     .build()?;
    /// // Decode the words; the copy to the host is for inspection.
    /// let words = mesh.vertex_origin_ids().to_host()?;
    /// let mut ids: Vec<i64> = words
    ///     .chunks_exact(2)
    ///     .map(|w| (i64::from((w[0] ^ 0x8000_0000) as i32) << 32) | i64::from(w[1]))
    ///     .collect();
    /// ids.sort();
    /// assert_eq!(ids, [10, 11, 12, 13]);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Origin ids]: https://tetracl.github.io/tetraloma/concepts/meshes.html#origin-ids
    fn vertex_origin_ids(&self) -> &DeviceBuffer<u32>;
    /// The entity blocks of dimension `dim` (edges for 1, faces for 2), one per entity type.
    ///
    /// See [Entities].
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` until `create_entities(dim)` has run, and always for dim =
    /// tdim: the cells are in [`cell_blocks`](DeviceGrid::cell_blocks).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in two triangles, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mut mesh = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// assert!(matches!(mesh.entity_blocks(1), Err(GridError::EntitiesNotCreated(1))));
    /// mesh.create_entities(1)?;
    /// // Two triangles share one of their 6 edges.
    /// assert_eq!(mesh.entity_blocks(1)?[0].index_map.num_local(), 5);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Entities]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#entities
    fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]>;
    /// The connectivity from dimension `key.from` to dimension `key.to`: downward (cell to
    /// edges, cell to faces) or upward (entity to cells).
    ///
    /// See [Connectivity].
    ///
    /// # Errors
    ///
    /// `ConnectivityNotCreated(key)` until `create_connectivity` has run;
    /// `UnsupportedConnectivity(key)` for pairs other than (tdim, d) and (d, tdim).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in two triangles, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mut mesh = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// let key = ConnKey { from: 2, to: 1 };
    /// assert!(matches!(mesh.connectivity(key), Err(GridError::ConnectivityNotCreated(_))));
    /// mesh.create_connectivity(2, 1)?;
    /// // Cell to edges: one row of 3 edges per triangle.
    /// let Connectivity::Downward(rows) = mesh.connectivity(key)? else { unreachable!() };
    /// assert_eq!(rows[0].data.len(), 2 * 3);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Connectivity]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#connectivity
    fn connectivity(&self, key: ConnKey) -> Result<&Connectivity>;
    /// The orientation word of every cell of block `b`, one `u32` per local cell, or `None` for
    /// a block of intervals, which have no orientation bits.
    ///
    /// The bits say how each edge and face of a cell is oriented relative to its reference
    /// orientation; [Orientation bits] gives the layout and [Permutation bits] the meaning.
    ///
    /// # Panics
    ///
    /// If there is no block `b`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in two triangles, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build::<f32, D2, D2>(&client)?;
    /// // The copy to the host is for inspection: one word per triangle, edge bits only.
    /// let words = mesh.permutation_info(0).expect("triangles have bits").to_host()?;
    /// assert_eq!(words.len(), 2);
    /// assert!(words.iter().all(|w| w >> 3 == 0));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Orientation bits]: https://tetracl.github.io/tetraloma/elements/orientation-bits.html#storage
    /// [Permutation bits]: https://tetracl.github.io/tetraloma/concepts/orientation.html#permutation-bits
    fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>>;
}
