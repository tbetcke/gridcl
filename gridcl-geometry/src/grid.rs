//! `DeviceGrid` (Listing B7; gridcl-geometry, re-exported by gridcl; ADR-0012).

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::{CommDyn, IndexMap};
use gridcl_topology::{CellBlock, Connectivity, EntityBlock};

use crate::{EVector, GeometryBlock};

/// What FEM and BEM crates program against (Listing B7). Implemented by `Mesh` and by
/// submeshes; later by extruded or structured grids.
///
/// Accessors create no topology: entities and connectivity are built only by the `&mut self`
/// methods `create_entities` and `create_connectivity`. Derived geometry data (E-vectors) is
/// built on first use and cached (ADR-0012).
pub trait DeviceGrid {
    /// The scalar of the geometry.
    type F: Real;
    /// The topological dimension.
    type TD: TopoDim + EmbeddedIn<Self::GD>;
    /// The geometric dimension.
    type GD: Dim;

    /// The runtime-erased client.
    fn client(&self) -> &Client;
    /// The communicator (Listing S2c).
    fn comm(&self) -> &dyn CommDyn;
    /// The cell blocks.
    fn cell_blocks(&self) -> &[CellBlock<Self::TD>];
    /// The geometry of cell block `b`.
    fn geometry_block(&self, b: usize) -> &GeometryBlock<Self::TD>;
    /// The L-vector [node][gdim], relative to [`origin`](DeviceGrid::origin).
    fn nodes(&self) -> &DeviceBuffer<Self::F>;
    /// Centre of the global bounding box; `physical_points` adds it back (ADR-0012).
    fn origin(&self) -> <Self::GD as Dim>::Array<f64>;
    /// E-vector of geometry block `b`, built on first use and cached (ADR-0012).
    fn evector(&self, b: usize) -> Result<&EVector<Self::F>>;
    /// The [owned | ghost] nodes.
    fn node_index_map(&self) -> &IndexMap;
    /// The [owned | ghost] vertices.
    fn vertex_index_map(&self) -> &IndexMap;
    /// Interleaved (hi ^ 0x8000_0000, lo) origin id per local vertex.
    fn vertex_origin_ids(&self) -> &DeviceBuffer<u32>;
    /// `Err(GridError::EntitiesNotCreated(dim))` until `create_entities(dim)` has run, and
    /// always for dim = tdim: cells are in `cell_blocks()` (ADR-0012).
    fn entity_blocks(&self, dim: usize) -> Result<&[EntityBlock]>;
    /// `Err(GridError::ConnectivityNotCreated(key))` until `create_connectivity` has run;
    /// `Err(GridError::UnsupportedConnectivity(key))` for pairs other than (tdim, d) and
    /// (d, tdim).
    fn connectivity(&self, key: ConnKey) -> Result<&Connectivity>;
    /// Per-cell orientation bits (A.6); `None` for interval cells, which have none.
    fn permutation_info(&self, b: usize) -> Option<&DeviceBuffer<u32>>;
}
