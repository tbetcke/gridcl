//! Curved balls: a hexahedral cube mapped smoothly onto the unit ball.

use super::HostMesh;

/// The unit ball as (2 · 2^refinements)³ hexahedra of degree `p`: the cube [−1, 1]³ mapped onto
/// the ball by a smooth radial map, with the boundary nodes on the sphere (gdim 3).
///
/// # Panics
///
/// If `p` is 0.
pub fn ball(refinements: u32, p: u32) -> HostMesh {
    let _ = (refinements, p);
    todo!()
}
