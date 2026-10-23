//! Curved balls: a hexahedral cube mapped smoothly onto the unit ball.
//!
//! The cube [−1, 1]³ is split into (2 · 2^refinements)³ hexahedra on the integer lattice of
//! resolution 2^refinements · p per unit length; the nodes are then mapped by the analytic
//! cube-to-ball map
//!
//!   x₀ = ξ₀ · sqrt(1 − ξ₁²/2 − ξ₂²/2 + ξ₁²ξ₂²/3), and cyclically for x₁ and x₂,
//!
//! for which |x|² = 1 − (1 − ξ₀²)(1 − ξ₁²)(1 − ξ₂²): it sends the cube's boundary onto the unit
//! sphere, and |x| grows along every ray. Its det J is positive in the open cube and vanishes
//! only on the cube's edges, as for every C¹ map of the cube onto the ball.
//!
//! A radial map x = ξ/|ξ| · sqrt(1 − Π(1 − ξᵢ²)) has the same |x|, but compresses the
//! neighbourhood of each cube corner cubically along the ray; its degree-2 interpolants then
//! have det J < 0 in the corner cells at every refinement. The map above does not.

use gridcl_dims::CellType;

use super::HostMesh;
use super::lattice::Assembler;

/// The analytic map of the cube [−1, 1]³ onto the unit ball.
fn cube_to_ball(xi: [f64; 3]) -> [f64; 3] {
    let [a, b, c] = xi.map(|v| v * v);
    let stretch = |b: f64, c: f64| (1.0 - b / 2.0 - c / 2.0 + b * c / 3.0).sqrt();
    [
        xi[0] * stretch(b, c),
        xi[1] * stretch(c, a),
        xi[2] * stretch(a, b),
    ]
}

/// The unit ball as (2 · 2^refinements)³ hexahedra of degree `p`: the cube [−1, 1]³ mapped onto
/// the ball by a smooth map, with the boundary nodes on the sphere (gdim 3).
///
/// The map is x₀ = ξ₀ · sqrt(1 − ξ₁²/2 − ξ₂²/2 + ξ₁²ξ₂²/3) and cyclically; it is not radial,
/// because no smooth radial map keeps det J > 0 for p = 2 near the cube's corners. det J > 0
/// inside the cube.
///
/// # Panics
///
/// If `p` is 0.
///
/// # Example
///
/// ```
/// let h = gridcl_testing::ball(0, 1);
/// assert_eq!((h.gdim, h.num_cells(), h.num_nodes()), (3, 8, 27));
/// // 26 of the 27 nodes lie on the unit sphere; the centre stays at the origin.
/// let on_sphere = h
///     .nodes
///     .chunks(3)
///     .filter(|x| (x.iter().map(|c| c * c).sum::<f64>() - 1.0).abs() < 1e-14)
///     .count();
/// assert_eq!(on_sphere, 26);
/// ```
pub fn ball(refinements: u32, p: u32) -> HostMesh {
    assert!(p > 0, "a ball needs degree p ≥ 1");
    let n = 1i64
        .checked_shl(refinements)
        .and_then(|n| n.checked_mul(2))
        .filter(|&n| n > 0)
        .expect("too many refinements");
    let step = i64::from(p);
    let half = (n / 2).checked_mul(step).expect("lattice overflows");
    let mut mesh = Assembler::new(3, half);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let corners: [[i64; 3]; 8] = core::array::from_fn(|v| {
                    let bit = |b: usize| ((v >> b) & 1) as i64;
                    [
                        -half + step * (i + bit(0)),
                        -half + step * (j + bit(1)),
                        -half + step * (k + bit(2)),
                    ]
                });
                mesh.cell(CellType::Hexahedron, p, &corners);
            }
        }
    }
    let scale = half as f64;
    mesh.finish_with(|key| cube_to_ball(key.map(|c| c as f64 / scale)))
}
