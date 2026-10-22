//! `CoordinateElement` and `GeometryBlock` (Listing B3; gridcl-geometry, re-exported by gridcl).

use gridcl_device::{DeviceStrided, Result};
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
        todo!()
    }
}
