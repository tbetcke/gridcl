//! `QuadRule` (Listing B4b).

use cubecl::prelude::Client;
use gridcl_device::{Real, Result};
use gridcl_dims::TopoDim;

use crate::{DeviceGrid, PointSet};

/// One point set per cell type of a dimension, so one rule serves mixed-cell meshes.
pub struct QuadRule<F: Real, TD: TopoDim> {
    sets: Vec<PointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> QuadRule<F, TD> {
    /// Gauss rules exact for polynomials of `degree`, for the given cell types.
    ///
    /// # Errors
    ///
    /// Device errors.
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        let _ = (client, cells, degree);
        todo!()
    }

    /// Gauss rules for every cell type present in `grid`.
    ///
    /// # Errors
    ///
    /// As [`gauss`](QuadRule::gauss).
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let _ = (grid, degree);
        todo!()
    }

    /// The point set of `cell`.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has none.
    pub fn get(&self, cell: TD::Cell) -> Result<&PointSet<F, TD>> {
        let _ = (cell, &self.sets);
        todo!()
    }
}
