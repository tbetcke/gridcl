//! The mixed triangle–quadrilateral surface: a cubed sphere with some quadrilaterals split.
//!
//! The faces of the cube [−1, 1]³ are split into 2^refinements × 2^refinements flat squares on
//! the integer lattice of resolution 2^refinements · p per unit length, so nodes shared between
//! faces have equal lattice keys. Every node is then projected radially onto the unit sphere.

use gridcl_dims::CellType;

use super::HostMesh;
use super::lattice::Assembler;
use super::sphere::project;

/// The unit sphere as a cubed sphere of 6 · 4^refinements quadrilaterals of degree `p`, with
/// every third quadrilateral split into two triangles, and every node projected onto the sphere
/// (gdim 3). One `Quadrilateral` and one `Triangle` block.
///
/// Squares are generated face by face (faces x = ±1, then y = ±1, then z = ±1, each in
/// lattice order); square q is split when q mod 3 = 2, along the diagonal from its first to
/// its last vertex. Every cell is oriented with J0 × J1 pointing away from the origin.
///
/// # Panics
///
/// If `p` is 0.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
///
/// let h = gridcl_testing::mixed_surface(0, 1);
/// let blocks: Vec<_> = h.blocks.iter().map(|b| (b.cell, b.num_cells())).collect();
/// assert_eq!(blocks, [(CellType::Triangle, 4), (CellType::Quadrilateral, 4)]);
/// assert_eq!(h.num_nodes(), 8);
/// ```
pub fn mixed_surface(refinements: u32, p: u32) -> HostMesh {
    assert!(p > 0, "a surface needs degree p ≥ 1");
    let n = 1i64
        .checked_shl(refinements)
        .filter(|&n| n > 0)
        .expect("too many refinements");
    let step = 2 * i64::from(p);
    let half = n.checked_mul(i64::from(p)).expect("lattice overflows");
    let mut mesh = Assembler::new(3, half);
    let mut square = 0usize;
    for axis in 0..3 {
        for sign in [-1i64, 1] {
            // Tangent axes u, v with e_u × e_v = sign · e_axis (outward).
            let (mut u, mut v) = ((axis + 1) % 3, (axis + 2) % 3);
            if sign < 0 {
                core::mem::swap(&mut u, &mut v);
            }
            let at = |i: i64, j: i64| {
                let mut key = [0; 3];
                key[axis] = sign * half;
                key[u] = -half + step * i;
                key[v] = -half + step * j;
                key
            };
            for j in 0..n {
                for i in 0..n {
                    let q = [at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1)];
                    if square % 3 == 2 {
                        mesh.cell(CellType::Triangle, p, &[q[0], q[1], q[3]]);
                        mesh.cell(CellType::Triangle, p, &[q[0], q[3], q[2]]);
                    } else {
                        mesh.cell(CellType::Quadrilateral, p, &q);
                    }
                    square += 1;
                }
            }
        }
    }
    let scale = half as f64;
    mesh.finish_with(|k| project(k.map(|c| c as f64 / scale)))
}
