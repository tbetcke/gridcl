//! The mixed triangle–quadrilateral surface: a cubed sphere with some quadrilaterals split.

use super::HostMesh;

/// The unit sphere as a cubed sphere of 6 · 4^refinements quadrilaterals of degree `p`, with
/// every third quadrilateral split into two triangles, and every node projected onto the sphere
/// (gdim 3). One `Quadrilateral` and one `Triangle` block.
///
/// # Panics
///
/// If `p` is 0.
pub fn mixed_surface(refinements: u32, p: u32) -> HostMesh {
    let _ = (refinements, p);
    todo!()
}
