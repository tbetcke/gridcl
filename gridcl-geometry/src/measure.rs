//! `measure_block` (Listing B11).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::{DeviceGrid, PointSet};

/// The measure of the owned cells of block `b`: per-cell sums of det J · w on the device, a
/// device reduction, one read, finished in f64 on this rank (Listing B11).
///
/// # Errors
///
/// Allocation and device errors.
pub fn measure_block<F, TD, GD, G>(g: &G, b: usize, pts: &PointSet<F, TD>) -> Result<f64>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    let _ = (g, b, pts);
    todo!()
}
