//! `FacetPointSet` and `FacetQuadRule` (Listing B4b; ADR-0012).

use cubecl::prelude::Client;
use tetraloma_device::{DeviceBuffer, GridError, Real, Result, check_precision};
use tetraloma_dims::{CellType, TopoDim};
use tetraloma_refcell::{
    facet_points_to_cell, gauss_rule, reference_facet_normals, sub_entity_types,
};

use crate::names::cell_name;
use crate::rule::distinct;
use crate::sumfact::TensorFactors;
use crate::{DeviceGrid, PointSet};

// Design: "Geometry", "Tabulation and quadrature", Listing B4b; ADR-0012.
/// Quadrature points on every facet of one reference cell, in the coordinates of the cell,
/// facet-major: the points of facet 0 first, then those of facet 1, and so on.
///
/// The facets of a prism or pyramid have different types, so their point counts differ; a
/// field on facet points has Σ_f n_f points per cell, with n_f the point count of facet f. The
/// weights are those of each facet's own reference cell. [Facet rules] explains the layout.
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_geometry::FacetQuadRule;
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 3).unwrap();
/// let fps = rule.get(Cell2::Triangle).unwrap();
/// // Two Gauss points on each of the three edges.
/// assert_eq!(fps.facet_offsets(), &[0, 2, 4, 6]);
/// assert_eq!(fps.points().len(), 6);
/// ```
///
/// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
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
        let set = PointSet::new(client, cell, points, weights)?;
        // The facets of quadrilaterals, hexahedra and prisms carry tensor Gauss rules (on the
        // prism, triangle × z or edge × z), each mapped along cell axes: the set records one
        // tensor block per facet (ADR-0042, item 5).
        let set = match cell_type {
            CellType::Quadrilateral | CellType::Hexahedron | CellType::Prism => {
                set.with_tensor(TensorFactors::facets(cell_type, degree))
            }
            _ => set,
        };
        Ok(Self {
            points: set,
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

    /// All facet points as one point set, in cell coordinates, with the weights of the facet
    /// rules.
    ///
    /// The weights add up to the volume of each facet's reference cell (1 on an edge, 1/2 on
    /// a triangle), whatever the facet's size in the cell; see [Facet rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::FacetQuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Hexahedron], 1).unwrap();
    /// let fps = rule.get(Cell3::Hexahedron).unwrap();
    /// // The six unit faces have total area 6.
    /// let area: f64 = fps.points().weights().iter().sum();
    /// assert!((area - 6.0).abs() < 1e-14);
    /// ```
    ///
    /// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
    pub fn points(&self) -> &PointSet<F, TD> {
        &self.points
    }

    /// Where each facet's points start: the points of facet f are
    /// `facet_offsets()[f]..facet_offsets()[f + 1]`, one entry per facet plus one.
    ///
    /// See [Facet rules].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::FacetQuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Prism], 1).unwrap();
    /// let offsets = rule.get(Cell3::Prism).unwrap().facet_offsets();
    /// // Two triangles and three quadrilaterals.
    /// assert_eq!(offsets.len(), 6);
    /// assert_eq!(*offsets.last().unwrap(), rule.get(Cell3::Prism).unwrap().points().len());
    /// ```
    ///
    /// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
    pub fn facet_offsets(&self) -> &[usize] {
        &self.facet_offsets
    }

    /// The outward unit normal n̂ of the reference cell on the facet of point `q`: `tdim`
    /// entries, in f64.
    ///
    /// The facet normal kernels map it to the physical normal n = Kᵀ n̂ / |Kᵀ n̂|; see
    /// [Facet normals].
    ///
    /// # Panics
    ///
    /// If `q` is not a point of the set.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell1, D1};
    /// use tetraloma_geometry::FacetQuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D1>::gauss(&client, &[Cell1::Interval], 1).unwrap();
    /// let fps = rule.get(Cell1::Interval).unwrap();
    /// // The facets of the interval are its vertices 0 and 1.
    /// assert_eq!(fps.points().points(), &[0.0, 1.0]);
    /// assert_eq!(fps.reference_normal(0), &[-1.0]);
    /// assert_eq!(fps.reference_normal(1), &[1.0]);
    /// ```
    ///
    /// [Facet normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#facet-normals
    pub fn reference_normal(&self, q: usize) -> &[f64] {
        let td = TD::N;
        &self.normals[q * td..(q + 1) * td]
    }
}

// Design: "Geometry", "Tabulation and quadrature", Listing B4b.
/// A quadrature rule on the facets of the reference cells of one dimension: one
/// [`FacetPointSet`] per cell type.
///
/// Each facet gets the Gauss rule of its own cell type, mapped into the coordinates of the
/// cell. Use it for boundary integrals, for fluxes between cells and for facet normals
/// ([Facet rules], [Facet normals]).
///
/// # Example
///
/// ```
/// use tetraloma_dims::{Cell2, D2};
/// use tetraloma_geometry::FacetQuadRule;
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Quadrilateral], 1).unwrap();
/// let fps = rule.get(Cell2::Quadrilateral).unwrap();
/// // One point on each of the four edges.
/// assert_eq!(fps.facet_offsets(), &[0, 1, 2, 3, 4]);
/// ```
///
/// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
/// [Facet normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#facet-normals
pub struct FacetQuadRule<F: Real, TD: TopoDim> {
    sets: Vec<FacetPointSet<F, TD>>,
}

impl<F: Real, TD: TopoDim> FacetQuadRule<F, TD> {
    /// Gauss rules of `degree` on every facet of the given cell types.
    ///
    /// A repeated cell type gets one facet point set; the sets keep the order of first
    /// appearance. The points, weights and reference normals stay on the host in f64; the
    /// weights and normals are also uploaded to `client`, narrowed to `F`. See [Facet rules].
    ///
    /// # Errors
    ///
    /// `UnsupportedPrecision` if the client does not support `F`; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_dims::{Cell3, D3};
    /// use tetraloma_geometry::FacetQuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D3>::gauss(&client, &[Cell3::Tetrahedron], 2).unwrap();
    /// let fps = rule.get(Cell3::Tetrahedron).unwrap();
    /// // Four triangular faces with 4 points each.
    /// assert_eq!(fps.facet_offsets(), &[0, 4, 8, 12, 16]);
    /// ```
    ///
    /// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
    pub fn gauss(client: &Client, cells: &[TD::Cell], degree: u32) -> Result<Self> {
        check_precision::<F>(client)?;
        let sets = distinct(cells.iter().copied())
            .into_iter()
            .map(|cell| FacetPointSet::gauss(client, cell, degree))
            .collect::<Result<_>>()?;
        Ok(Self { sets })
    }

    /// Facet rules of `degree` for every cell type of `grid`'s cell blocks, on the grid's
    /// client.
    ///
    /// See [Facet rules] and [Facet normals].
    ///
    /// # Errors
    ///
    /// As [`gauss`](FacetQuadRule::gauss).
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
    /// let rule = FacetQuadRule::for_grid(&mesh, 2)?;
    /// for block in mesh.cell_blocks() {
    ///     assert!(rule.get(block.cell).is_ok());
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
    /// [Facet normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#facet-normals
    pub fn for_grid<G: DeviceGrid<F = F, TD = TD>>(grid: &G, degree: u32) -> Result<Self> {
        let cells: Vec<TD::Cell> = grid.cell_blocks().iter().map(|b| b.cell).collect();
        Self::gauss(grid.client(), &cells, degree)
    }

    /// The facet point set of `cell`.
    ///
    /// See [Facet rules].
    ///
    /// # Errors
    ///
    /// `CellNotInRule` if the rule has no facet point set for `cell`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::GridError;
    /// use tetraloma_dims::{Cell2, D2};
    /// use tetraloma_geometry::FacetQuadRule;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let rule = FacetQuadRule::<f32, D2>::gauss(&client, &[Cell2::Triangle], 1).unwrap();
    /// assert!(rule.get(Cell2::Triangle).is_ok());
    /// assert!(matches!(
    ///     rule.get(Cell2::Quadrilateral),
    ///     Err(GridError::CellNotInRule("Quadrilateral"))
    /// ));
    /// ```
    ///
    /// [Facet rules]: https://tetracl.github.io/tetraloma/concepts/quadrature.html#facet-rules
    pub fn get(&self, cell: TD::Cell) -> Result<&FacetPointSet<F, TD>> {
        self.sets
            .iter()
            .find(|s| s.points.cell() == cell)
            .ok_or_else(|| GridError::CellNotInRule(cell_name(cell.into())))
    }
}
