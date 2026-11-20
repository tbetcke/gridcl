//! `PointSet` (Listing B4b; ADR-0012).

use core::cell::RefCell;
use std::collections::HashMap;

use cubecl::prelude::Client;
use tetraloma_device::{DeviceBuffer, Real, Result};
use tetraloma_dims::{CellType, TopoDim};
use tetraloma_refcell::LagrangeElement;

use crate::CoordinateElement;
use crate::names::element_error;
use crate::sumfact::TensorFactors;

/// Which table of an element a cache entry holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Table {
    /// Values, `[q][k]`.
    Values,
    /// Reference derivatives, `[q][k][d]`.
    Derivatives,
    /// The table of the sum-factorised kernels (`TensorFactors::table`).
    Tensor,
}

/// The uploaded tables of a point set, per element and table.
type TableCache<F, TD> = HashMap<(CoordinateElement<TD>, Table), DeviceBuffer<F>>;

// Design: "Geometry", "Tabulation and quadrature", Listing B4b; ADR-0012.
/// Points and weights on one reference cell, with the tabulations of coordinate elements at
/// the points.
///
/// The points and weights stay on the host in f64. A tabulation, the values or first
/// derivatives of an element's basis functions at the points, is computed on the host in f64,
/// narrowed to `F` and uploaded the first time it is asked for; the point set keeps it, so
/// later calls with the same element reuse it. [`QuadRule::get`](crate::QuadRule::get) and
/// [`FacetPointSet::points`](crate::FacetPointSet::points) return point sets. See
/// [Quadrature rules] and [Tabulation].
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_geometry::QuadRule;
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 2).unwrap();
/// let ps = rule.get(Cell2::Triangle).unwrap();
/// assert_eq!(ps.cell(), Cell2::Triangle);
/// // Two reference coordinates per point.
/// assert_eq!(ps.points().len(), 2 * ps.len());
/// ```
///
/// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
/// [Tabulation]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#tabulation
pub struct PointSet<F: Real, TD: TopoDim> {
    cell: TD::Cell,
    points: Vec<f64>,
    weights: Vec<f64>,
    client: Client,
    /// The weights narrowed to `F`, uploaded with the rule.
    weights_device: DeviceBuffer<F>,
    /// Uploaded tabulations, per element and table.
    cache: RefCell<TableCache<F, TD>>,
    /// The tensor structure of the points, recorded by the rules that build tensor products
    /// (ADR-0042, item 5); `None` for every other point set.
    tensor: Option<TensorFactors>,
}

impl<F: Real, TD: TopoDim> PointSet<F, TD> {
    /// A point set of `cell` with points `[q][tdim]` and one weight per point.
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
            tensor: None,
        })
    }

    /// A point set of `cell` with points `[q][tdim]` and zero weights, made on the device
    /// without an upload: an interpolation lattice (`equispaced_nodes`), whose weights no
    /// kernel reads.
    pub(crate) fn lattice(client: &Client, cell: TD::Cell, points: Vec<f64>) -> Result<Self> {
        debug_assert!(points.len().is_multiple_of(TD::N));
        let num_points = points.len() / TD::N;
        Ok(Self {
            cell,
            points,
            weights: vec![0.0; num_points],
            client: client.clone(),
            weights_device: DeviceBuffer::zeros(client, num_points)?,
            cache: RefCell::new(HashMap::new()),
            tensor: None,
        })
    }

    /// The point set with the tensor structure `factors` recorded.
    ///
    /// # Panics
    ///
    /// If `factors` do not describe exactly the points of the set.
    pub(crate) fn with_tensor(mut self, factors: TensorFactors) -> Self {
        assert!(
            factors.points(TD::N, self.len()).as_deref() == Some(self.points.as_slice()),
            "tensor factors that do not describe the points"
        );
        self.tensor = Some(factors);
        self
    }

    /// The tensor structure of the points, if a rule recorded one.
    pub(crate) fn tensor(&self) -> Option<&TensorFactors> {
        self.tensor.as_ref()
    }

    /// The table of the sum-factorised kernels for `e`, `[TensorFactors::table]` narrowed to
    /// `F`, uploaded on first use and cached like the tabulations.
    ///
    /// # Errors
    ///
    /// As [`tabulate`](PointSet::tabulate), and `TooLarge` for a table whose integers reach
    /// 2^24.
    ///
    /// # Panics
    ///
    /// If the set records no tensor structure.
    pub(crate) fn tabulate_tensor(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        self.cached(e, Table::Tensor)
    }

    /// The weights on the device, narrowed to `F` (uploaded when the rule was created).
    pub(crate) fn weights_device(&self) -> &DeviceBuffer<F> {
        &self.weights_device
    }

    /// The reference cell of the points.
    ///
    /// See [Quadrature rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Prism], 1).unwrap();
    /// assert_eq!(rule.get(Cell3::Prism).unwrap().cell(), Cell3::Prism);
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn cell(&self) -> TD::Cell {
        self.cell
    }

    /// The number of points.
    ///
    /// A Gauss rule of degree k has ⌊k / 2⌋ + 1 points per direction; see [Quadrature rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell1, D1};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // Gauss–Legendre with 2 points is exact for degree 3.
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 3).unwrap();
    /// assert_eq!(rule.get(Cell1::Interval).unwrap().len(), 2);
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the set has no points.
    ///
    /// Every Gauss rule has at least one point; see [Quadrature rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell1, D1};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// assert!(!rule.get(Cell1::Interval).unwrap().is_empty());
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// The reference coordinates ξ of the points, in f64: `tdim` values per point, point
    /// after point.
    ///
    /// All points of a Gauss rule lie inside the reference cell; see [Quadrature rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 3).unwrap();
    /// let ps = rule.get(Cell2::Quadrilateral).unwrap();
    /// assert!(ps.points().iter().all(|&x| (0.0..=1.0).contains(&x)));
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn points(&self) -> &[f64] {
        &self.points
    }

    /// The weights w_q, one per point, in f64.
    ///
    /// The weights of a Gauss rule are positive and add up to the volume of the reference
    /// cell; see [Quadrature rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Tetrahedron], 2).unwrap();
    /// let volume: f64 = rule.get(Cell3::Tetrahedron).unwrap().weights().iter().sum();
    /// assert!((volume - 1.0 / 6.0).abs() < 1e-14);
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// The client that holds the uploaded weights and tabulations.
    ///
    /// See [Tabulation].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell1, D1};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// let ps = rule.get(Cell1::Interval).unwrap();
    /// assert_eq!(ps.client().name(), client.name());
    /// ```
    ///
    /// [Tabulation]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#tabulation
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The values φ_k(ξ_q) of the basis functions of `e` at the points, on the device:
    /// point-major `[q][k]`, narrowed to `F`.
    ///
    /// The first call for an element computes the table in f64 and uploads it; later calls
    /// return a share of the same buffer. See [Tabulation].
    ///
    /// # Errors
    ///
    /// `UnsupportedNodeFamily` or `UnsupportedDegree` for an element that tetraloma does not
    /// offer; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_geometry::{CoordinateElement, NodeFamily, QuadRule};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 2).unwrap();
    /// let ps = rule.get(Cell2::Triangle).unwrap();
    /// let e = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
    /// // A copy to the host, for inspection.
    /// let phi = ps.tabulate(&e).unwrap().to_host().unwrap();
    /// assert_eq!(phi.len(), ps.len() * 3);
    /// // The basis is a partition of unity at every point.
    /// for row in phi.chunks(3) {
    ///     assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// [Tabulation]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#tabulation
    pub fn tabulate(&self, e: &CoordinateElement<TD>) -> Result<DeviceBuffer<F>> {
        self.cached(e, Table::Values)
    }

    /// The first derivatives ∂φ_k/∂ξ_d(ξ_q) of the basis functions of `e` at the points, on
    /// the device: `[q][k][d]`, narrowed to `F`.
    ///
    /// The first call for an element computes the table in f64 and uploads it; later calls
    /// return a share of the same buffer. See [Tabulation].
    ///
    /// # Errors
    ///
    /// As [`tabulate`](PointSet::tabulate).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_geometry::{CoordinateElement, NodeFamily, QuadRule};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 3).unwrap();
    /// let ps = rule.get(Cell2::Quadrilateral).unwrap();
    /// let e = CoordinateElement::<D2> { cell: Cell2::Quadrilateral, degree: 2, family: NodeFamily::Gll };
    /// // A copy to the host, for inspection: 9 basis functions, 2 derivatives each.
    /// let dphi = ps.tabulate_derivatives(&e).unwrap().to_host().unwrap();
    /// assert_eq!(dphi.len(), ps.len() * 9 * 2);
    /// // The basis functions add up to 1, so their derivatives add up to 0.
    /// for q in 0..ps.len() {
    ///     for d in 0..2 {
    ///         let sum: f32 = (0..9).map(|k| dphi[(q * 9 + k) * 2 + d]).sum();
    ///         assert!(sum.abs() < 1e-5);
    ///     }
    /// }
    /// ```
    ///
    /// [Tabulation]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#tabulation
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
        if table == Table::Tensor {
            let factors = self.tensor.as_ref().expect("a tensor point set");
            let host = factors.table(cell, e.degree, e.family)?;
            return Ok(host.into_iter().map(F::narrow).collect());
        }
        let element = LagrangeElement::new(cell, e.degree, e.family).map_err(element_error)?;
        let nderivs = match table {
            Table::Values => 0,
            Table::Derivatives | Table::Tensor => 1,
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
            Table::Derivatives | Table::Tensor => (0..num_points)
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
    use tetraloma_dims::{Cell2, D2};
    use tetraloma_refcell::NodeFamily;

    use super::*;

    #[test]
    fn an_empty_point_set_tabulates_to_empty_buffers() {
        let client = tetraloma_testing::test_client();
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
