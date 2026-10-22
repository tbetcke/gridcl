//! `Geometry<F, TD, GD>` (ADR-0012): geometry blocks, the node L-vector, the node `IndexMap`,
//! the origin and the E-vector cache.

use core::cell::OnceCell;

use gridcl_device::{DeviceBuffer, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::IndexMap;

use crate::{EVector, GeometryBlock};

/// The geometry of a mesh (ADR-0012).
pub struct Geometry<F: Real, TD: TopoDim, GD: Dim> {
    blocks: Vec<GeometryBlock<TD>>,
    nodes: DeviceBuffer<F>,
    node_index_map: IndexMap,
    origin: GD::Array<f64>,
    evectors: Vec<OnceCell<EVector<F>>>,
}

impl<F, TD, GD> Geometry<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    /// The geometry of `blocks` over `nodes` ([node][gdim], relative to `origin`).
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `nodes` does not hold `gdim` values per local node.
    pub fn new(
        blocks: Vec<GeometryBlock<TD>>,
        nodes: DeviceBuffer<F>,
        node_index_map: IndexMap,
        origin: GD::Array<f64>,
    ) -> Result<Self> {
        let _ = (&blocks, &nodes, &node_index_map, &origin);
        let _: &Vec<OnceCell<EVector<F>>> = &Vec::new();
        todo!()
    }

    /// The geometry blocks, one per cell block.
    pub fn blocks(&self) -> &[GeometryBlock<TD>] {
        &self.blocks
    }

    /// The L-vector [node][gdim], relative to [`origin`](Geometry::origin).
    pub fn nodes(&self) -> &DeviceBuffer<F> {
        &self.nodes
    }

    /// The [owned | ghost] nodes.
    pub fn node_index_map(&self) -> &IndexMap {
        &self.node_index_map
    }

    /// The centre of the bounding box, in f64.
    pub fn origin(&self) -> GD::Array<f64> {
        self.origin
    }

    /// The E-vector of block `b`, built on first use and cached.
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the first build.
    pub fn evector(&self, b: usize) -> Result<&EVector<F>> {
        let _ = (b, &self.evectors);
        todo!()
    }

    /// A deep copy of every device array; the E-vector cache starts empty.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        todo!()
    }
}
