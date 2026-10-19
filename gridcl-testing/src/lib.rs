//! Test support for gridcl: the test device chosen by `GRIDCL_TEST_DEVICE`, test meshes and
//! tolerance helpers.
//!
//! This crate is not published. It is the only library crate that enables CubeCL runtime
//! features, which its own features forward (docs/design.md, "Cargo features").

use cubecl::prelude::Client;

/// The client of the device named by the `GRIDCL_TEST_DEVICE` environment variable.
pub fn test_client() -> Client {
    todo!()
}
