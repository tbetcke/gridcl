//! Compile-fail and compile-pass tests of the type layer (docs/design.md, Listing B10).
//!
//! The cases use a test-only stub `Mesh` (tests/ui/stub/mesh.rs) with the impl bounds of
//! Listing B4. The `.stderr` baselines are generated with `TRYBUILD=overwrite` and reviewed;
//! they depend on the pinned rustc (ADR-0003).

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/fail/*.rs");
    t.pass("tests/ui/pass/*.rs");
}
