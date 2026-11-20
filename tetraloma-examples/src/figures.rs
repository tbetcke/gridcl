//! The figures of the book's element definitions, generated from tetraloma-refcell's tables: an
//! SVG of each reference cell with its vertex, edge and face numbers, and the Lagrange node
//! orderings for p = 1, 2, 3 on the 2D cells and p = 1, 2 on the 3D cells.
//!
//! The binary `figures` writes them to `docs/user/src/figures/`; the test
//! `figures_are_generated` checks that the committed files equal [`svg_files`].

/// Every figure: its file name in `docs/user/src/figures/` and its SVG text.
///
/// # Example
///
/// ```no_run
/// for (name, svg) in tetraloma_examples::figures::svg_files() {
///     assert!(name.ends_with(".svg") && svg.starts_with("<svg"));
/// }
/// ```
pub fn svg_files() -> Vec<(String, String)> {
    todo!()
}
