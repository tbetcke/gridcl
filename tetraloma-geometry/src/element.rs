//! `CoordinateElement` and `GeometryBlock` (Listing B3; tetraloma-geometry, re-exported by tetraloma).

use tetraloma_device::{DeviceStrided, Result};
use tetraloma_dims::CellType;
use tetraloma_dims::TopoDim;
use tetraloma_refcell::NodeFamily;

// Design: "Geometry", Listing B3.
/// The coordinate element of a cell block: its cell type, the polynomial degree p of its
/// geometry, and the node family of its Lagrange nodes.
///
/// A block of degree 1 has straight edges and flat faces; a block of degree p ≥ 2 is curved,
/// with the nodes of [Lagrange nodes] in each cell. [Coordinate elements] explains the map
/// that the element defines.
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_geometry::{CoordinateElement, NodeFamily};
///
/// // Quadratic quadrilaterals with Gauss–Lobatto–Legendre nodes.
/// let e = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 2, family: NodeFamily::Gll };
/// assert_eq!(e.degree, 2);
/// ```
///
/// [Lagrange nodes]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-ordering
/// [Coordinate elements]: https://tetracl.github.io/tetraloma/concepts/geometry.html#coordinate-elements
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoordinateElement<TD: TopoDim> {
    /// The cell type.
    pub cell: TD::Cell,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// The node family.
    pub family: NodeFamily,
}

// Design: "Geometry", Listing B3.
/// The geometry of one cell block: its coordinate element and its dofmap, which lists the
/// nodes of each cell.
///
/// [`DeviceGrid::geometry_block`](crate::DeviceGrid::geometry_block) returns it for a block of
/// a mesh; [Blocks] shows it in use and [Writing kernels] reads it in a kernel.
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceStrided};
/// use tetraloma_dims::{Cell1, D1};
/// use tetraloma_geometry::{CoordinateElement, GeometryBlock, NodeFamily};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // Two quadratic intervals: nodes (0, 2, 1) and (2, 4, 3), endpoints first.
/// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 2, family: NodeFamily::Equispaced };
/// let dofmap = DeviceBuffer::from_host(&client, &[0u32, 2, 1, 2, 4, 3]).unwrap();
/// let block = GeometryBlock { element, dofmap: DeviceStrided { data: dofmap, stride: 3 } };
/// assert_eq!((block.nodes_per_cell(), block.num_cells()), (3, 2));
/// ```
///
/// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
/// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#geometryblock
pub struct GeometryBlock<TD: TopoDim> {
    /// The coordinate element.
    pub element: CoordinateElement<TD>,
    /// The local node indices of each cell, one row of `nodes_per_cell()` entries per cell, in
    /// the node ordering of the coordinate element.
    pub dofmap: DeviceStrided<u32>,
}

impl<TD: TopoDim> GeometryBlock<TD> {
    /// A deep copy: the dofmap is copied into a new device buffer.
    ///
    /// See [Blocks] for what the block holds.
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the copy.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 triangles, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Triangle, 2, 1).build::<f32, D2, D2>(&client)?;
    /// let block = mesh.geometry_block(0);
    /// let copy = block.try_clone()?;
    /// assert_eq!(copy.element, block.element);
    /// assert_eq!(copy.dofmap.data.to_host()?, block.dofmap.data.to_host()?);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
    pub fn try_clone(&self) -> Result<Self> {
        Ok(GeometryBlock {
            element: self.element,
            dofmap: DeviceStrided {
                data: self.dofmap.data.deep_clone()?,
                stride: self.dofmap.stride,
            },
        })
    }

    /// The number of nodes per cell of the coordinate element: the stride of the dofmap.
    ///
    /// [Node counts] lists it for every cell type and degree.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 quadratic triangles, on the test device; the
    /// // tutorial "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Triangle, 2, 2).build::<f32, D2, D2>(&client)?;
    /// // A quadratic triangle has 6 nodes: 3 vertices and 3 edge midpoints.
    /// assert_eq!(mesh.geometry_block(0).nodes_per_cell(), 6);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Node counts]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-counts
    pub fn nodes_per_cell(&self) -> usize {
        let cell: CellType = self.element.cell.into();
        tetraloma_refcell::num_lagrange_nodes(cell, self.element.degree)
    }

    /// The number of cells: the rows of the dofmap.
    ///
    /// On a mesh it equals the local (owned and ghost) cells of the block; see [Blocks].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit square in 2 × 2 triangles, on the test device; the tutorial
    /// // "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::square_box(Cell2::Triangle, 2, 1).build::<f32, D2, D2>(&client)?;
    /// assert_eq!(mesh.geometry_block(0).num_cells(), 8);
    /// assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 8);
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Blocks]: https://tetracl.github.io/tetraloma/tutorials/iterate.html#blocks
    pub fn num_cells(&self) -> usize {
        self.dofmap.data.len() / self.nodes_per_cell()
    }
}
