// Design: "Dimension markers", "Cell families per topological dimension".
//! Dimension markers, cell types and the traits that relate them: the type layer of
//! tetraloma.
//!
//! A mesh is typed by the pair (TD, GD) of its topological and geometric dimension, each one of
//! the markers [`D0`]–[`D3`]. [`EmbeddedIn`] admits the six valid pairs (TD ≤ GD), [`Codim0`]
//! and [`Codim1`] select the pairs on which inverse Jacobians and normals exist, and
//! [`TopoDim`] names the cell family ([`Cell1`], [`Cell2`], [`Cell3`]) of a topological
//! dimension. The crate has no dependencies; the facade crate `tetraloma` re-exports its markers,
//! cell types and traits.
//!
//! The book explains why the dimensions are types, and which methods each pair has, in
//! [Typed dimensions].
//!
//! # Example
//!
//! ```
//! use tetraloma_dims::{Cell2, CellType, D2, D3, Dim, TopoDim};
//!
//! // A surface mesh has TD = D2 and GD = D3; its cells are triangles or quadrilaterals.
//! let cell: <D2 as TopoDim>::Cell = Cell2::Triangle;
//! assert_eq!(CellType::from(cell), CellType::Triangle);
//! assert_eq!((D2::N, D3::N), (2, 3));
//! ```
//!
//! [Typed dimensions]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html

mod cell;
mod dim;

pub use cell::{Cell1, Cell2, Cell3, CellType, TopoDim, WrongCellDim};
pub use dim::{Codim0, Codim1, D0, D1, D2, D3, Dim, EmbeddedIn};
