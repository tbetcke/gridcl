//! Dimension markers, `EmbeddedIn`/`Codim`, `CellType`, `Cell1`–`Cell3` and `TopoDim`.
//!
//! The type layer of gridcl (docs/design.md, "Dimension markers" and "Cell families per
//! topological dimension"). It has no dependencies.
//!
//! A mesh is typed by the pair (TD, GD) of its topological and geometric dimension, each one of
//! the markers [`D0`]–[`D3`]. [`EmbeddedIn`] admits the six valid pairs, [`Codim0`] and
//! [`Codim1`] select the pairs on which inverse Jacobians and normals exist, and [`TopoDim`]
//! names the cell family ([`Cell1`], [`Cell2`], [`Cell3`]) of a topological dimension.

mod cell;
mod dim;

pub use cell::{Cell1, Cell2, Cell3, CellType, TopoDim, WrongCellDim};
pub use dim::{Codim0, Codim1, D0, D1, D2, D3, Dim, EmbeddedIn};
