//! Compile-fail and compile-pass tests of the type layer (docs/design.md, Listing B10), against
//! the real `Mesh`.
//!
//! The `.stderr` baselines are generated with `TRYBUILD=overwrite` and reviewed; they depend on
//! the pinned rustc (ADR-0003) and quote the line `pub fn new(` of gridcl-topology's
//! src/cell_block.rs and `pub struct D0;` and `pub struct D3;` of gridcl-dims' src/dim.rs (all
//! without line numbers): editing any of them, or the `Mesh` impl bounds of gridcl/src/mesh.rs,
//! means regenerating and reviewing the baselines. The first run builds gridcl and cubecl in
//! target/tests/trybuild (about two minutes on the M3 Max).

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/*.rs");
    t.pass("tests/ui/pass/*.rs");
}
