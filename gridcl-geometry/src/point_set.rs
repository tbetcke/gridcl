//! `PointSet` (Listing B4b; ADR-0012).

use core::cell::RefCell;
use std::collections::HashMap;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, Real, Result};
use gridcl_dims::{CellType, TopoDim};
use gridcl_refcell::LagrangeElement;

use crate::CoordinateElement;
use crate::names::element_error;

/// Which table of an element a cache entry holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Table {
    /// Values, [q][k].
    Values,
    /// Reference derivatives, [q][k][d].
    Derivatives,
}

/// The uploaded tables of a point set, per element and table.
type TableCache<F, TD> = HashMap<(CoordinateElement<TD>, Table), DeviceBuffer<F>>;

/// Points and weights on one reference cell (host, f64), with device tabulations of coordinate
/// elements: computed in f64, narrowed to `F`, uploaded on first use and cached per element.
/// The tabulate calls return `share()`s of the cached buffers.
///
/// # Example
///
/// ```
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_geometry::QuadRule;
///
/// let client = gridcl_testing::test_client();
/// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 2).unwrap();
/// let ps = rule.get(Cell2::Triangle).unwrap();
/// assert_eq!(ps.cell(), Cell2::Triangle);
/// assert_eq!(ps.points().len(), 2 * ps.len());
/// ```
pub struct PointSet<F: Real, TD: TopoDim> {
    cell: TD::Cell,
    points: Vec<f64>,
    weights: Vec<f64>,
    client: Client,
    /// The weights narrowed to `F`, uploaded with the rule.
    weights_device: DeviceBuffer<F>,
    /// Uploaded tabulations, per element and table.
    cache: RefCell<TableCache<F, TD>>,
}

impl<F: Real, TD: TopoDim> PointSet<F, TD> {
    /// A point set of `cell` with points [q][tdim] and one weight per point.
    pub(crate) fn new(
        client: &Client,
        cell: TD::Cell,
        points: Vec<f64>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        debug_assert_eq!(points.len(), weights.len() * TD::N);
        let narrowed: Vec<F> = weights.iter().map(|&w| F::narrow(w)).collect();
        Ok(Self {
            cell,
            points,
            weights,
            client: client.clone(),
            weights_device: DeviceBuffer::from_host(client, &narrowed)?,
            cache: RefCell::new(HashMap::new()),
        })
    }

    /// The weights on the device, narrowed to `F` (uploaded when the rule was created).
    pub(crate) fn weights_device(&self) -> &DeviceBuffer<F> {
        &self.weights_device
    }

    /// The cell type.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Prism], 1).unwrap();
    /// assert_eq!(rule.get(Cell3::Prism).unwrap().cell(), Cell3::Prism);
    /// ```
    pub fn cell(&self) -> TD::Cell {
        self.cell
    }

    /// The number of points.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell1, D1};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// // Gauss–Jacobi with 2 points is exact for degree 3.
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 3).unwrap();
    /// assert_eq!(rule.get(Cell1::Interval).unwrap().len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the set has no points.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell1, D1};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// assert!(!rule.get(Cell1::Interval).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Reference coordinates, [q][tdim].
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, D2};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 3).unwrap();
    /// let ps = rule.get(Cell2::Quadrilateral).unwrap();
    /// assert!(ps.points().iter().all(|&x| (0.0..=1.0).contains(&x)));
    /// ```
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// One weight per point.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Tetrahedron], 2).unwrap();
    /// let volume: f64 = rule.get(Cell3::Tetrahedron).unwrap().weights().iter().sum();
    /// assert!((volume - 1.0 / 6.0).abs() < 1e-14);
    /// ```
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The client the tabulations live on.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell1, D1};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// let ps = rule.get(Cell1::Interval).unwrap();
    /// assert_eq!(ps.client().name(), client.name());
    /// ```
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Basis values of `e` at the points, [q][k].
    ///
    /// # Errors
    ///
    /// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element gridcl-refcell does not
    /// offer; device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, D2};
    /// use gridcl_geometry::{CoordinateElement, NodeFamily, QuadRule};
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 2).unwrap();
    /// let ps = rule.get(Cell2::Triangle).unwrap();
    /// let e = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// let phi = ps.tabulate(&e).unwrap().to_host().unwrap();
    /// assert_eq!(phi.len(), ps.len() * 3);
    /// // The basis is a partition of unity at every point.
    /// for row in phi.chunks(3) {
    ///     assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    /// }
    /// ```
    pub fn tabulate(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        self.cached(e, Table::Values)
    }

    /// Reference derivatives of the basis of `e` at the points, [q][k][d].
    ///
    /// # Errors
    ///
    /// As [`tabulate`](PointSet::tabulate).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell2, D2};
    /// use gridcl_geometry::{CoordinateElement, NodeFamily, QuadRule};
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 3).unwrap();
    /// let ps = rule.get(Cell2::Quadrilateral).unwrap();
    /// let e = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 2, family: NodeFamily::Gll };
    /// let dphi = ps.tabulate_derivatives(&e).unwrap().to_host().unwrap();
    /// assert_eq!(dphi.len(), ps.len() * 9 * 2);
    /// ```
    pub fn tabulate_derivatives(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        self.cached(e, Table::Derivatives)
    }

    /// A share of the cached table of `e`, computed and uploaded on first use.
    fn cached(&self, e: &CoordinateElement<TD>, table: Table) -> Result<DeviceBuffer<F>> {
        let key = (*e, table);
        if let Some(buffer) = self.cache.borrow().get(&key) {
            return Ok(buffer.share());
        }
        let host = self.compute(e, table)?;
        let buffer = DeviceBuffer::from_host(&self.client, &host)?;
        let share = buffer.share();
        self.cache.borrow_mut().insert(key, buffer);
        Ok(share)
    }

    /// The table of `e` at the points, in f64, narrowed to `F`.
    fn compute(&self, e: &CoordinateElement<TD>, table: Table) -> Result<Vec<F>> {
        let cell: CellType = e.cell.into();
        let element = LagrangeElement::new(cell, e.degree, e.family).map_err(element_error)?;
        let nderivs = match table {
            Table::Values => 0,
            Table::Derivatives => 1,
        };
        let tab = element
            .tabulate(&self.points, nderivs)
            .map_err(element_error)?;
        let num_points = self.len();
        let k_count = element.dim();
        let tdim = TD::N;
        let out = match table {
            Table::Values => (0..num_points)
                .flat_map(|q| (0..k_count).map(move |k| (q, k)))
                .map(|(q, k)| F::narrow(tab.get(0, q, k)))
                .collect(),
            Table::Derivatives => (0..num_points)
                .flat_map(|q| (0..k_count).map(move |k| (q, k)))
                .flat_map(|(q, k)| (0..tdim).map(move |d| (q, k, d)))
                .map(|(q, k, d)| F::narrow(tab.get(1 + d, q, k)))
                .collect(),
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use gridcl_dims::{Cell2, D2};
    use gridcl_refcell::NodeFamily;

    use super::*;

    #[test]
    fn an_empty_point_set_tabulates_to_empty_buffers() {
        let client = gridcl_testing::test_client();
        let ps =
            PointSet::<f32, D2>::new(&client, Cell2::Triangle, Vec::new(), Vec::new()).unwrap();
        assert!(ps.is_empty());
        let e = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 2,
            family: NodeFamily::Equispaced,
        };
        assert_eq!(ps.tabulate(&e).unwrap().len(), 0);
        assert_eq!(ps.tabulate_derivatives(&e).unwrap().len(), 0);
    }
}
