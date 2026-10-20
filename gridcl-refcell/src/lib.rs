//! Reference cells, sub-entity tables, Lagrange layouts, quadrature and orientation bits.
//!
//! gridcl's own definitions, following the DefElement convention (docs/design.md, "Reference
//! cells and conventions" and Appendix A). They are checked against ndelement by
//! gridcl-conformance in `conformance/`.
//!
//! Everything here is host code in f64 ("Precision policy"). Point arrays are flat and
//! node-major: point `i` occupies `points[i * tdim..(i + 1) * tdim]`, with `tdim` the
//! topological dimension of the cell ([`tdim`]).
//!
//! - [`vertices`], [`sub_entities`], [`sub_entity_types`]: tables A.1–A.3.
//! - [`num_lagrange_nodes`], [`lagrange_points`], [`lagrange_entity_nodes`]: A.4 and A.5.
//! - [`gauss_rule`]: quadrature on every cell.
//! - [`LagrangeElement`]: tabulation of Lagrange bases with an orthonormal basis and dense LU.
//! - [`orientation_bits`]: the permutation info of A.6, from origin ids.
//! - [`gmsh_to_internal`], [`vtk_to_internal`]: node maps between files and gridcl (A.9).

mod cell;
mod error;
mod generator;
mod jacobi;
mod lagrange;
mod lu;
mod node_maps;
mod orientation;
mod polynomials;
mod quadrature;
mod tabulation;

pub use cell::{num_sub_entities, sub_entities, sub_entity_types, tdim, vertices};
pub use error::RefCellError;
pub use lagrange::{
    NodeFamily, lagrange_entity_nodes, lagrange_points, num_interior_lagrange_nodes,
    num_lagrange_nodes,
};
pub use node_maps::{gmsh_to_internal, vtk_to_internal};
pub use orientation::orientation_bits;
pub use quadrature::{QuadratureRule, gauss_rule};
pub use tabulation::{LagrangeElement, Tabulation};
