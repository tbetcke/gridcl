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
//! - [`orientation_bits`], [`canonical_order`]: the permutation info and storage slots of A.6,
//!   from origin ids.
//! - [`reference_facet_normals`], [`facet_points_to_cell`]: facet normals and facet point maps.
//! - [`gmsh_to_internal`], [`vtk_to_internal`]: node maps between files and gridcl (A.9).
//! - [`serendipity_upgrade`]: the complete-P2 nodes of Gmsh's serendipity elements.

mod cell;
mod error;
mod facet;
mod generator;
mod jacobi;
mod lagrange;
mod lu;
mod node_maps;
mod orientation;
mod polynomials;
mod quadrature;
mod serendipity;
mod tabulation;

pub use cell::{num_sub_entities, sub_entities, sub_entity_types, tdim, vertices};
pub use error::RefCellError;
pub use facet::{facet_points_to_cell, reference_facet_normals};
pub use lagrange::{
    NodeFamily, lagrange_entity_nodes, lagrange_points, num_interior_lagrange_nodes,
    num_lagrange_nodes,
};
pub use node_maps::{gmsh_to_internal, vtk_to_internal};
pub use orientation::{canonical_order, orientation_bits};
pub use quadrature::{QuadratureRule, gauss_rule};
pub use serendipity::{SerendipityUpgrade, serendipity_upgrade};
pub use tabulation::{LagrangeElement, Tabulation};
