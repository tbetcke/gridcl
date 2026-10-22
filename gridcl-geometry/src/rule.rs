//! `QuadRule` (Listing B4b).

use cubecl::prelude::Client;
use gridcl_device::{GridError, Real, Result, check_precision};
use gridcl_dims::TopoDim;
use gridcl_refcell::gauss_rule;

use crate::names::cell_name;
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

/// One point set per cell type of a dimension, so one rule serves mixed-cell meshes.
///
/// # Example
///
/// ```
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_geometry::QuadRule;
///
/// let client = gridcl_testing::test_client();
/// let rule = QuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle, Cell2::Quadrilateral], 3).unwrap();
/// let area: f64 = rule.get(Cell2::Triangle).unwrap().weights().iter().sum();
/// assert!((area - 0.5).abs() < 1e-14);
/// ```
pub struct QuadRule<F: Real, TD: TopoDim> {
    sets: Vec<PointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> QuadRule<F, TD> {
    /// Gauss rules exact for polynomials of `degree`, for the given cell types.
    ///
    /// Repeated cell types get one point set; the sets keep the order of first appearance.
    ///
    /// # Errors
    ///
    /// Device errors: `UnsupportedPrecision` if the client does not support `F`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 3).unwrap();
    /// // 2 Gauss points per direction.
    /// assert_eq!(rule.get(Cell3::Hexahedron).unwrap().len(), 8);
    /// ```
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        check_precision::<F>(client)?;
        let sets = distinct(cells.iter().copied())
            .into_iter()
            .map(|cell| {
                let rule = gauss_rule(cell.into(), degree);
                PointSet::new(client, cell, rule.points, rule.weights)
            })
            .collect::<Result<_>>()?;
        Ok(Self { sets })
    }

    /// Gauss rules for every cell type present in `grid`.
    ///
    /// # Errors
    ///
    /// As [`gauss`](QuadRule::gauss).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::mixed_box(1, 1).build::<f32, D3, D3>(&client).unwrap();
    /// let rule = QuadRule::for_grid(&mesh, 2).unwrap();
    /// for block in mesh.cell_blocks() {
    ///     assert_eq!(rule.get(block.cell).unwrap().cell(), block.cell);
    /// }
    /// ```
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let cells: Vec<TD::Cell> = grid.cell_blocks().iter().map(|b| b.cell).collect();
        Self::gauss(grid.client(), &cells, degree)
    }

    /// The point set of `cell`.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has none.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::GridError;
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::QuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = QuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 1).unwrap();
    /// assert!(matches!(
    ///     rule.get(Cell3::Tetrahedron),
    ///     Err(GridError::CellNotInRule("Tetrahedron"))
    /// ));
    /// ```
    pub fn get(&self, cell: TD::Cell) -> Result<&PointSet<F, TD>> {
        self.sets
            .iter()
            .find(|s| s.cell() == cell)
            .ok_or_else(|| GridError::CellNotInRule(cell_name(cell.into())))
    }
}
