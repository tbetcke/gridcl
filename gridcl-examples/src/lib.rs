//! The examples of gridcl's API walkthrough (docs/design.md, Listings B8, B9 and B13;
//! ADR-0060 item 6).
//!
//! Each example under `examples/` is a thin `main` that picks its device as Listing B8 does and
//! calls the `run` function of one module here; `tests/examples.rs` runs every `run` on the test
//! device.
//!
//! - [`read_gmsh`]: read a Gmsh file and fix the dimensions once (Listing B8);
//! - [`surface_submesh`]: a boundary surface mesh, its normals, Jacobians and area (Listing B9);
//! - [`geometric_factors`]: geometric factors at p = 1, 4 and 8 on a curved sphere;
//! - [`halo_exchange`]: `forward` and `reverse` on the node index map (Listing B13);
//! - [`vtk_output`]: VTU output (Listing B13).

pub mod geometric_factors;
pub mod halo_exchange;
pub mod read_gmsh;
pub mod surface_submesh;
pub mod vtk_output;
