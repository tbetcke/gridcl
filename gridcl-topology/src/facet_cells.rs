//! Facet classification for submeshes and `orient` (ADR-0039, items 11 and 17): the number of
//! owned cells of every local facet over all ranks, which works in every `GhostMode`.

use gridcl_device::{DeviceBuffer, Result};
use gridcl_dims::TopoDim;
use gridcl_parallel::CommDyn;

use crate::Topology;

/// The cells of every local facet (entities of dimension tdim − 1), in flat facet order over
/// the facet blocks. For gridcl's submeshes and `orient`; not part of the stable API.
#[doc(hidden)]
pub struct FacetCells {
    /// The number of owned cells of each local facet, summed over all ranks: 1 on the
    /// boundary, 2 inside a manifold, more where the mesh is not a manifold.
    pub counts: DeviceBuffer<u32>,
    /// For a facet with one cell globally: the flat local index of that cell, which every
    /// holder of the facet holds; `u32::MAX` for other facets.
    pub cell: DeviceBuffer<u32>,
    /// For a facet with one cell globally: its local facet index in that cell; `u32::MAX` for
    /// other facets.
    pub local_facet: DeviceBuffer<u32>,
}

/// The [`FacetCells`] of `topology`, whose facet entities exist: owned cells are counted per
/// local facet over their downward (tdim, tdim − 1) rows, combined by a reverse `Add` and
/// returned by a forward scatter over each facet block's `IndexMap` (ADR-0039, item 11).
/// Collective. Not part of the stable API (tests: gridcl-topology's `facet_cells`).
///
/// # Errors
///
/// `EntitiesNotCreated(tdim − 1)` before `create_entities(tdim − 1)`; allocation, device and
/// communication errors.
#[doc(hidden)]
pub fn facet_cell_counts<TD: TopoDim>(
    _comm: &dyn CommDyn,
    _topology: &Topology<TD>,
) -> Result<FacetCells> {
    todo!("facet_cell_counts: M9 T2")
}
