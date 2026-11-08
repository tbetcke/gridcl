//! Coordinate elements, geometry blocks, `DeviceGrid`, point sets and quadrature rules,
//! E-vectors, the tiled `QuadField` and the geometric factor kernels (docs/design.md,
//! "Geometry", Listings B3, B4b, B5 and B7; ADR-0012).
//!
//! Factor kernels read the cached E-vector of a geometry block ([`DeviceGrid::evector`]) and
//! write tiled [`QuadField`]s, with the lanes of a vector over consecutive cells of a tile.

mod basis_eval;
#[doc(hidden)]
#[cfg(feature = "bench")]
pub mod bench;
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
mod measure;
mod names;
mod orient;
mod point_set;
mod points_jac;
mod pull_back;
mod rule;
mod shape;
mod strategy;
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
pub use gridcl_refcell::NodeFamily;
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
