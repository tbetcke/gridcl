//! `QuadRule` (Listing B4b).

use cubecl::prelude::Client;
use tetraloma_device::{GridError, Real, Result, check_precision};
use tetraloma_dims::{CellType, TopoDim};
use tetraloma_refcell::gauss_rule;

use crate::names::cell_name;
use crate::sumfact::TensorFactors;
use crate::{DeviceGrid, PointSet};

/// The distinct cell types of `cells`, in order of first appearance.
pub(crate) fn distinct<C: PartialEq + Copy>(cells: impl IntoIterator<Item = C>) -> Vec<C> {
    let mut out = Vec::new();
    for c in cells {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

// Design: "Geometry", "Tabulation and quadrature", Listing B4b.
/// A quadrature rule on the reference cells of one dimension: one [`PointSet`] per cell type,
/// so that one rule serves a mesh with several cell types.
///
/// The rules are Gauss rules: Gauss–Legendre tensor products on intervals, quadrilaterals and
/// hexahedra, collapsed Gauss–Jacobi rules on triangles, tetrahedra and pyramids, and a
/// triangle rule times an interval rule on prisms. A rule of degree k integrates polynomials
/// of degree k exactly. [Quadrature rules] gives the details, and [Rules] shows how to use
/// them on a mesh.
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_geometry::QuadRule;
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle, Cell2::Quadrilateral], 3).unwrap();
/// // The weights add up to the area 1/2 of the reference triangle.
/// let area: f64 = rule.get(Cell2::Triangle).unwrap().weights().iter().sum();
/// assert!((area - 0.5).abs() < 1e-14);
/// ```
///
/// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
/// [Rules]: https://tetracl.github.io/tetraloma/tutorials/quadrature.html#rules
pub struct QuadRule<F: Real, TD: TopoDim> {
    sets: Vec<PointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> QuadRule<F, TD> {
    /// Gauss rules exact for polynomials of `degree`, for the given cell types.
    ///
    /// Each rule has m = ⌊degree / 2⌋ + 1 points per direction. A repeated cell type gets one
    /// point set; the sets keep the order of first appearance. The points and weights stay on
    /// the host in f64; the weights are also uploaded to `client`, narrowed to `F`. See
    /// [Quadrature rules].
    ///
    /// # Errors
    ///
    /// `UnsupportedPrecision` if the client does not support `F`; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 3).unwrap();
    /// // 2 Gauss points per direction.
    /// assert_eq!(rule.get(Cell3::Hexahedron).unwrap().len(), 8);
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        check_precision::<F>(client)?;
        let sets = distinct(cells.iter().copied())
            .into_iter()
            .map(|cell| {
                let cell_type: CellType = cell.into();
                let rule = gauss_rule(cell_type, degree);
                let set = PointSet::new(client, cell, rule.points, rule.weights)?;
                // Tensor Gauss on quadrilaterals and hexahedra records its 1D factor, the
                // interval rule of the same degree, and the prism's rule its triangle and
                // interval factors (ADR-0042, item 5).
                let line = || gauss_rule(CellType::Interval, degree).points;
                Ok(match cell_type {
                    CellType::Quadrilateral | CellType::Hexahedron => {
                        set.with_tensor(TensorFactors::cube(TD::N, &line()))
                    }
                    CellType::Prism => {
                        let triangle = gauss_rule(CellType::Triangle, degree).points;
                        set.with_tensor(TensorFactors::prism(&triangle, &line()))
                    }
                    _ => set,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { sets })
    }

    /// Gauss rules of `degree` for every cell type of `grid`'s cell blocks, on the grid's
    /// client.
    ///
    /// This is the usual way to make a rule for a mesh; see [Rules].
    ///
    /// # Errors
    ///
    /// As [`gauss`](QuadRule::gauss).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // `mesh`: the unit cube in pyramids and tetrahedra, one block per cell type, on the
    /// // test device; the tutorial "Building a mesh" shows how to make one.
    /// # let client = tetraloma_testing::test_client();
    /// # let mesh = tetraloma_testing::mixed_box(1, 1).build::<f32, D3, D3>(&client)?;
    /// let rule = QuadRule::for_grid(&mesh, 2)?;
    /// for block in mesh.cell_blocks() {
    ///     assert_eq!(rule.get(block.cell)?.cell(), block.cell);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Rules]: https://tetracl.github.io/tetraloma/tutorials/quadrature.html#rules
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let cells: Vec<TD::Cell> = grid.cell_blocks().iter().map(|b| b.cell).collect();
        Self::gauss(grid.client(), &cells, degree)
    }

    /// The point set of `cell`.
    ///
    /// See [Quadrature rules].
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has no point set for `cell`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::GridError;
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::QuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 1).unwrap();
    /// assert_eq!(rule.get(Cell3::Hexahedron).unwrap().len(), 1);
    /// assert!(matches!(
    ///     rule.get(Cell3::Tetrahedron),
    ///     Err(GridError::CellNotInRule("Tetrahedron"))
    /// ));
    /// ```
    ///
    /// [Quadrature rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#quadrature-rules
    pub fn get(&self, cell: TD::Cell) -> Result<&PointSet<F, TD>> {
        self.sets
            .iter()
            .find(|s| s.cell() == cell)
            .ok_or_else(|| GridError::CellNotInRule(cell_name(cell.into())))
    }
}
