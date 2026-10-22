//! Curved sphere surfaces from icosahedral refinement, nodes projected onto the unit sphere.

use super::HostMesh;

/// The unit sphere as 20 · 4^refinements triangles of degree `p`: the icosahedron refined
/// `refinements` times, with every node projected onto the sphere (gdim 3).
///
/// # Panics
///
/// If `p` is 0.
pub fn sphere(refinements: u32, p: u32) -> HostMesh {
    let _ = (refinements, p);
    todo!()
}
