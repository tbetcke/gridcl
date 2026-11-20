// Design: "Reference cells and conventions", "Precision policy", Appendix A (tables A.1–A.9).
// The definitions are checked against ndelement by tetraloma-conformance in `conformance/`.
//! Reference cells, sub-entity tables, Lagrange node layouts, quadrature rules, tabulation and
//! orientation bits: the cell definitions every other tetraloma crate builds on.
//!
//! The definitions follow the DefElement conventions, which ndelement and Symfem share. The book's
//! [Element definitions] part gives every table of this crate, and the concepts chapters
//! [Reference cells and maps] and [Quadrature] explain what they are for.
//!
//! Everything here is host code in f64, with no device. Point arrays are flat and node-major:
//! point `i` occupies `points[i * tdim..(i + 1) * tdim]`, with `tdim` the topological dimension
//! of the cell ([`tdim`]).
//!
//! - [`vertices`], [`sub_entities`], [`sub_entity_types`], [`num_sub_entities`]: the reference
//!   cells and the numbering of their edges and faces.
//! - [`num_lagrange_nodes`], [`num_interior_lagrange_nodes`], [`lagrange_points`],
//!   [`lagrange_entity_nodes`]: the Lagrange nodes of each degree, in tetraloma's order.
//! - [`gauss_rule`]: a Gauss quadrature rule on every cell.
//! - [`LagrangeElement`]: the values and first derivatives of a Lagrange basis.
//! - [`orientation_bits`], [`canonical_order`]: the permutation bits of a cell and the stored
//!   vertex order of an entity, from origin ids.
//! - [`reference_facet_normals`], [`facet_points_to_cell`]: facet normals and facet point maps.
//! - [`gmsh_to_internal`], [`vtk_to_internal`]: node maps between file formats and tetraloma.
//! - [`serendipity_upgrade`]: the complete quadratic nodes of Gmsh's serendipity elements.
//!
//! # Example
//!
//! ```
//! use tetraloma_dims::CellType;
//! use tetraloma_refcell::{gauss_rule, num_lagrange_nodes, sub_entities};
//!
//! // The reference triangle has three edges, a quadratic triangle six nodes, and the weights
//! // of a Gauss rule sum to the triangle's area 1/2.
//! assert_eq!(sub_entities(CellType::Triangle, 1).len(), 3);
//! assert_eq!(num_lagrange_nodes(CellType::Triangle, 2), 6);
//! let rule = gauss_rule(CellType::Triangle, 2);
//! assert!((rule.weights.iter().sum::<f64>() - 0.5).abs() < 1e-15);
//! ```
//!
//! [Element definitions]: https://tetracl.github.io/tetraloma/elements/reference-cells.html
//! [Reference cells and maps]: https://tetracl.github.io/tetraloma/concepts/reference-cells.html
//! [Quadrature]: https://tetracl.github.io/tetraloma/concepts/quadrature.html

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
