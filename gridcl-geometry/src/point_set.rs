//! `PointSet` (Listing B4b; ADR-0012).

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, Real, Result};
use gridcl_dims::TopoDim;

use crate::CoordinateElement;

/// Points and weights on one reference cell (host, f64), with device tabulations of coordinate
/// elements: computed in f64, narrowed to `F`, uploaded on first use and cached per element.
/// The tabulate calls return `share()`s of the cached buffers.
pub struct PointSet<F: Real, TD: TopoDim> {
    cell: TD::Cell,
    points: Vec<f64>,
    weights: Vec<f64>,
    client: Client,
    _f: core::marker::PhantomData<F>,
}

impl<F: Real, TD: TopoDim> PointSet<F, TD> {
    /// The cell type.
    pub fn cell(&self) -> TD::Cell {
        self.cell
    }

    /// The number of points.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the set has no points.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Reference coordinates, [q][tdim].
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// One weight per point.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The client the tabulations live on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Basis values of `e` at the points, [q][k].
    ///
    /// # Errors
    ///
    /// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element gridcl-refcell does not
    /// offer; device errors.
    pub fn tabulate(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        let _ = e;
        todo!()
    }

    /// Reference derivatives of the basis of `e` at the points, [q][k][d].
    ///
    /// # Errors
    ///
    /// As [`tabulate`](PointSet::tabulate).
    pub fn tabulate_derivatives(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        let _ = e;
        todo!()
    }
}
