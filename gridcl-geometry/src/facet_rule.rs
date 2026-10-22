//! `FacetPointSet` and `FacetQuadRule` (Listing B4b; ADR-0012).

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, GridError, Real, Result, check_precision};
use gridcl_dims::{CellType, TopoDim};
use gridcl_refcell::{facet_points_to_cell, gauss_rule, reference_facet_normals, sub_entity_types};

use crate::names::cell_name;
use crate::rule::distinct;
use crate::{DeviceGrid, PointSet};

/// Points on every facet of a cell, in cell coordinates, facet-major (ADR-0012). A cell's
/// facets may differ in point count (prism, pyramid), so a facet field has Σ_f n_f points per
/// cell.
///
/// # Example
///
/// ```
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_geometry::FacetQuadRule;
///
/// let client = gridcl_testing::test_client();
/// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 3).unwrap();
/// let fps = rule.get(Cell2::Triangle).unwrap();
/// // Two Gauss points on each of the three edges.
/// assert_eq!(fps.facet_offsets(), &[0, 2, 4, 6]);
/// assert_eq!(fps.points().len(), 6);
/// ```
pub struct FacetPointSet<F: Real, TD: TopoDim> {
    points: PointSet<F, TD>,
    facet_offsets: Vec<usize>,
    /// The outward reference normal of each point's facet, [q][tdim].
    normals: Vec<f64>,
    /// `normals` narrowed to `F`, uploaded with the rule.
    normals_device: DeviceBuffer<F>,
}

impl<F: Real, TD: TopoDim> FacetPointSet<F, TD> {
    /// The facet point set of `cell`: Gauss rules of `degree` on its facets, mapped into cell
    /// coordinates, facet by facet.
    fn gauss(client: &Client, cell: TD::Cell, degree: u32) -> Result<Self> {
        let cell_type: CellType = cell.into();
        let td = TD::N;
        let facet_types = sub_entity_types(cell_type, td - 1);
        let facet_normals = reference_facet_normals(cell_type);
        let mut points = Vec::new();
        let mut weights = Vec::new();
        let mut normals = Vec::new();
        let mut facet_offsets = Vec::with_capacity(facet_types.len() + 1);
        facet_offsets.push(0);
        for (f, &facet_type) in facet_types.iter().enumerate() {
            let rule = gauss_rule(facet_type, degree);
            points.extend(facet_points_to_cell(cell_type, f, &rule.points));
            let normal = &facet_normals[f * td..(f + 1) * td];
            for _ in &rule.weights {
                normals.extend_from_slice(normal);
            }
            weights.extend(rule.weights);
            facet_offsets.push(weights.len());
        }
        let narrowed: Vec<F> = normals.iter().map(|&x| F::narrow(x)).collect();
        Ok(Self {
            points: PointSet::new(client, cell, points, weights)?,
            facet_offsets,
            normals_device: DeviceBuffer::from_host(client, &narrowed)?,
            normals,
        })
    }

    /// The reference normals on the device, [q][tdim], narrowed to `F` (uploaded when the
    /// rule was created).
    pub(crate) fn normals_device(&self) -> &DeviceBuffer<F> {
        &self.normals_device
    }

    /// All facet points as one point set, with the facets' weights.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::FacetQuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 1).unwrap();
    /// let fps = rule.get(Cell3::Hexahedron).unwrap();
    /// // The six unit faces have total area 6.
    /// let area: f64 = fps.points().weights().iter().sum();
    /// assert!((area - 6.0).abs() < 1e-14);
    /// ```
    pub fn points(&self) -> &PointSet<F, TD> {
        &self.points
    }

    /// Where each facet's points start; facets + 1 entries.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::FacetQuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Prism], 1).unwrap();
    /// let offsets = rule.get(Cell3::Prism).unwrap().facet_offsets();
    /// // Two triangles and three quadrilaterals.
    /// assert_eq!(offsets.len(), 6);
    /// assert_eq!(*offsets.last().unwrap(), rule.get(Cell3::Prism).unwrap().points().len());
    /// ```
    pub fn facet_offsets(&self) -> &[usize] {
        &self.facet_offsets
    }

    /// The outward reference normal at point `q` (`tdim` entries).
    ///
    /// # Panics
    ///
    /// If `q` is not a point of the set.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell1, D1};
    /// use gridcl_geometry::FacetQuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// let fps = rule.get(Cell1::Interval).unwrap();
    /// // The facets of the interval are its vertices 0 and 1.
    /// assert_eq!(fps.points().points(), &[0.0, 1.0]);
    /// assert_eq!(fps.reference_normal(0), &[-1.0]);
    /// assert_eq!(fps.reference_normal(1), &[1.0]);
    /// ```
    pub fn reference_normal(&self, q: usize) -> &[f64] {
        let td = TD::N;
        &self.normals[q * td..(q + 1) * td]
    }
}

/// Facet point sets for each cell type of a dimension.
///
/// # Example
///
/// ```
/// use gridcl_dims::{Cell2, D2};
/// use gridcl_geometry::FacetQuadRule;
///
/// let client = gridcl_testing::test_client();
/// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 1).unwrap();
/// let fps = rule.get(Cell2::Quadrilateral).unwrap();
/// assert_eq!(fps.facet_offsets().len(), 5);
/// ```
pub struct FacetQuadRule<F: Real, TD: TopoDim> {
    sets: Vec<FacetPointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> FacetQuadRule<F, TD> {
    /// Gauss rules of `degree` on every facet of the given cell types.
    ///
    /// Repeated cell types get one facet point set; the sets keep the order of first
    /// appearance.
    ///
    /// # Errors
    ///
    /// Device errors: `UnsupportedPrecision` if the client does not support `F`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::{Cell3, D3};
    /// use gridcl_geometry::FacetQuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Tetrahedron], 2).unwrap();
    /// let fps = rule.get(Cell3::Tetrahedron).unwrap();
    /// assert_eq!(fps.facet_offsets().len(), 5);
    /// ```
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        check_precision::<F>(client)?;
        let sets = distinct(cells.iter().copied())
            .into_iter()
            .map(|cell| FacetPointSet::gauss(client, cell, degree))
            .collect::<Result<_>>()?;
        Ok(Self { sets })
    }

    /// Facet rules for every cell type present in `grid`.
    ///
    /// # Errors
    ///
    /// As [`gauss`](FacetQuadRule::gauss).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mesh = gridcl_testing::mixed_box(1, 1).build::<f32, D3, D3>(&client).unwrap();
    /// let rule = FacetQuadRule::for_grid(&mesh, 2).unwrap();
    /// for block in mesh.cell_blocks() {
    ///     assert!(rule.get(block.cell).is_ok());
    /// }
    /// ```
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let cells: Vec<TD::Cell> = grid.cell_blocks().iter().map(|b| b.cell).collect();
        Self::gauss(grid.client(), &cells, degree)
    }

    /// The facet point set of `cell`.
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has none.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::GridError;
    /// use gridcl_dims::{Cell2, D2};
    /// use gridcl_geometry::FacetQuadRule;
    ///
    /// let client = gridcl_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 1).unwrap();
    /// assert!(matches!(
    ///     rule.get(Cell2::Quadrilateral),
    ///     Err(GridError::CellNotInRule("Quadrilateral"))
    /// ));
    /// ```
    pub fn get(&self, cell: TD::Cell) -> Result<&FacetPointSet<F, TD>> {
        self.sets
            .iter()
            .find(|s| s.points.cell() == cell)
            .ok_or_else(|| GridError::CellNotInRule(cell_name(cell.into())))
    }
}
