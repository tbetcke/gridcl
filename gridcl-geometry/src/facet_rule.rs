//! `FacetPointSet` and `FacetQuadRule` (Listing B4b; ADR-0012).

use cubecl::prelude::Client;
use gridcl_device::{Real, Result};
use gridcl_dims::TopoDim;

use crate::{DeviceGrid, PointSet};

/// Points on every facet of a cell, in cell coordinates, facet-major (ADR-0012). A cell's
/// facets may differ in point count (prism, pyramid), so a facet field has Σ_f n_f points per
/// cell.
pub struct FacetPointSet<F: Real, TD: TopoDim> {
    points: PointSet<F, TD>,
    facet_offsets: Vec<usize>,
    normals: Vec<f64>,
}

impl<F: Real, TD: TopoDim> FacetPointSet<F, TD> {
    /// All facet points as one point set, with the facets' weights.
    pub fn points(&self) -> &PointSet<F, TD> {
        &self.points
    }

    /// Where each facet's points start; facets + 1 entries.
    pub fn facet_offsets(&self) -> &[usize] {
        &self.facet_offsets
    }

    /// The outward reference normal at point `q` (`tdim` entries).
    pub fn reference_normal(&self, q: usize) -> &[f64] {
        let _ = (q, &self.normals);
        todo!()
    }
}

/// Facet point sets for each cell type of a dimension.
pub struct FacetQuadRule<F: Real, TD: TopoDim> {
    sets: Vec<FacetPointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> FacetQuadRule<F, TD> {
    /// Gauss rules of `degree` on every facet of the given cell types.
    ///
    /// # Errors
    ///
    /// Device errors.
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        let _ = (client, cells, degree);
        todo!()
    }

    /// Facet rules for every cell type present in `grid`.
    ///
    /// # Errors
    ///
    /// As [`gauss`](FacetQuadRule::gauss).
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let _ = (grid, degree);
        todo!()
    }

    /// The facet point set of `cell`.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has none.
    pub fn get(&self, cell: TD::Cell) -> Result<&FacetPointSet<F, TD>> {
        let _ = (cell, &self.sets);
        todo!()
    }
}
