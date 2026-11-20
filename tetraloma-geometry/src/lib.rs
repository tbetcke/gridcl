// Design: "Geometry", "Geometric factors", "Pull-back", Listings B3, B4b, B5 and B7; ADR-0012.
//! Geometry of tetraloma meshes: coordinate elements, quadrature rules, and the device kernels
//! that compute geometric factors and pull points back.
//!
//! Most programs use this crate through the facade `tetraloma`, which re-exports its public
//! items and calls them from methods of `Mesh` such as `jacobians` and `detj`. Use it directly
//! when you write code that works on any grid through the [`DeviceGrid`] trait.
//!
//! - [`CoordinateElement`] and [`GeometryBlock`] describe the Lagrange geometry of a cell block
//!   ([Lagrange geometry]).
//! - [`QuadRule`], [`FacetQuadRule`] and [`PointSet`] hold quadrature points and weights on the
//!   reference cells, and tabulate coordinate elements at them ([Quadrature]).
//! - [`physical_points_block`], [`jacobians_block`], [`detj_block`], [`pseudo_inverse_block`],
//!   [`normals_block`], [`facet_normals_block`] and [`measure_block`] compute the geometric
//!   factors of one cell block on the device ([Geometric factors]); they return a
//!   [`QuadField`], and the methods of `Mesh` collect one per block in [`BlockFields`].
//! - [`pull_back_block`] finds the reference points of physical points ([The pull-back]).
//! - [`OrientationRule`] says how `Mesh::orient` orients a curve or surface ([Orientation]).
//!
//! Every kernel reads the node coordinates of a block gathered per cell, its [`EVector`]
//! ([`DeviceGrid::evector`]), and writes a [`QuadField`] in tiles of consecutive cells, so that
//! the lanes of a vector instruction work on neighbouring cells.
//!
//! # Example
//!
//! ```
//! use tetraloma::prelude::*;
//! use tetraloma_geometry::detj_block;
//!
//! // `mesh`: the unit square in 2 × 2 quadrilaterals, on the test device; the tutorial
//! // "Building a mesh" shows how to make one.
//! # let client = tetraloma_testing::test_client();
//! # let mesh = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build::<f32, D2, D2>(&client)?;
//! let rule = QuadRule::for_grid(&mesh, 2)?;
//! let detj = detj_block(&mesh, 0, rule.get(Cell2::Quadrilateral)?)?;
//! // Squares of side 1/2 over the unit reference square: det J = 1/4 at every point.
//! assert!(detj.to_host_cell_major()?.iter().all(|d| (d - 0.25).abs() < 1e-5));
//! # Ok::<(), GridError>(())
//! ```
//!
//! [Lagrange geometry]: https://tetracl.github.io/tetraloma/concepts/geometry.html
//! [Quadrature]: https://tetracl.github.io/tetraloma/concepts/quadrature.html
//! [Geometric factors]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html
//! [The pull-back]: https://tetracl.github.io/tetraloma/concepts/pull-back.html
//! [Orientation]: https://tetracl.github.io/tetraloma/concepts/orientation.html

mod basis_eval;
#[doc(hidden)]
#[cfg(feature = "bench")]
pub mod bench;
mod cooperative;
mod detj_pinv;
mod element;
mod equispaced;
mod evector;
mod facet_rule;
mod facets;
mod field;
mod geometry;
mod grid;
mod layout;
mod layout_loops;
mod measure;
mod names;
mod orient;
mod point_set;
mod points_jac;
mod pull_back;
mod rule;
mod shape;
mod shared;
mod strategy;
mod sumfact;
#[doc(hidden)]
#[cfg(feature = "testing")]
pub mod testing;

pub use detj_pinv::{detj_block, pseudo_inverse_block};
pub use element::{CoordinateElement, GeometryBlock};
pub use equispaced::equispaced_nodes;
pub use evector::EVector;
pub use facet_rule::{FacetPointSet, FacetQuadRule};
pub use facets::{facet_normals_block, normals_block};
pub use field::{BlockFields, QuadChunk, QuadField};
pub use geometry::Geometry;
pub use grid::DeviceGrid;
pub use layout::GeomLayout;
pub use measure::measure_block;
pub use orient::OrientationRule;
#[doc(hidden)]
pub use orient::{
    OrientationComponents, orientation_components, orientation_flags, orientation_neighbours,
};
pub use point_set::PointSet;
pub use points_jac::{jacobians_block, physical_points_block};
pub use pull_back::{PullBack, PullBackOptions, pull_back_block};
pub use rule::QuadRule;
pub use shape::{FieldShape, JacShape, PinvShape, ScalarShape, VecShape};
pub use tetraloma_refcell::NodeFamily;
