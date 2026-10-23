//! `CoordinateElement` and `GeometryBlock` (Listing B3; gridcl-geometry, re-exported by gridcl).

use gridcl_device::{DeviceStrided, Result};
use gridcl_dims::CellType;
use gridcl_dims::TopoDim;
use gridcl_refcell::NodeFamily;

/// The coordinate element of a geometry block: cell type, degree p and node family.
///
/// # Example
///
/// ```
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_geometry::{CoordinateElement, NodeFamily};
///
/// let e = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 2, family: NodeFamily::Gll };
/// assert_eq!(e.degree, 2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoordinateElement<TD: TopoDim> {
    /// The cell type.
    pub cell: TD::Cell,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// The node family.
    pub family: NodeFamily,
}

/// The geometry of one cell block: its coordinate element and its dofmap into the node array.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceStrided};
/// use gridcl_dims::{Cell1, D1};
/// use gridcl_geometry::{CoordinateElement, GeometryBlock, NodeFamily};
///
/// let client = gridcl_testing::test_client();
/// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 2, family: NodeFamily::Equispaced };
/// let dofmap = DeviceBuffer::from_host(&client, &[0u32, 2, 1, 2, 4, 3]).unwrap();
/// let block = GeometryBlock { element, dofmap: DeviceStrided { data: dofmap, stride: 3 } };
/// assert_eq!((block.nodes_per_cell(), block.num_cells()), (3, 2));
/// let copy = block.try_clone().unwrap();
/// assert_eq!(copy.dofmap.data.to_host().unwrap(), [0, 2, 1, 2, 4, 3]);
/// ```
pub struct GeometryBlock<TD: TopoDim> {
    /// The coordinate element.
    pub element: CoordinateElement<TD>,
    /// cells × nodes-per-cell, into the node array, in A.5 order.
    pub dofmap: DeviceStrided<u32>,
}

impl<TD: TopoDim> GeometryBlock<TD> {
    /// A deep copy of the dofmap.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        Ok(GeometryBlock {
            element: self.element,
            dofmap: DeviceStrided {
                data: self.dofmap.data.deep_clone()?,
                stride: self.dofmap.stride,
            },
        })
    }

    /// Nodes per cell of the coordinate element.
    pub fn nodes_per_cell(&self) -> usize {
        let cell: CellType = self.element.cell.into();
        gridcl_refcell::num_lagrange_nodes(cell, self.element.degree)
    }

    /// The number of cells: dofmap rows.
    pub fn num_cells(&self) -> usize {
        self.dofmap.data.len() / self.nodes_per_cell()
    }
}
