//! Structured boxes of every cell type, the mixed tetrahedron/pyramid/hexahedron box and the
//! P1 triangle strip.

use gridcl_dims::{Cell2, Cell3};

use super::HostMesh;

/// The unit interval [0, 1] with `n` cells of degree `p` (gdim 1).
///
/// # Panics
///
/// If `n` or `p` is 0.
pub fn interval_box(n: usize, p: u32) -> HostMesh {
    let _ = (n, p);
    todo!()
}

/// The unit square with n × n quadrilaterals of degree `p`, or 2n² triangles, each square split
/// along its diagonal (gdim 2).
///
/// # Panics
///
/// If `n` or `p` is 0.
pub fn square_box(cell: Cell2, n: usize, p: u32) -> HostMesh {
    let _ = (cell, n, p);
    todo!()
}

/// The unit cube with n³ hexahedra of degree `p`: 6n³ tetrahedra (Kuhn split), 2n³ prisms
/// (each hexahedron split along a diagonal of its z-faces) or 6n³ pyramids (each hexahedron
/// split into six around its centre) (gdim 3).
///
/// # Panics
///
/// If `n` or `p` is 0.
pub fn cube_box(cell: Cell3, n: usize, p: u32) -> HostMesh {
    let _ = (cell, n, p);
    todo!()
}

/// A conforming tetrahedron/pyramid/hexahedron box on the unit cube with n³ hexahedral
/// subcells of degree `p`: subcells with an even index sum are split into six pyramids around
/// their centre, and those pyramids whose base lies on the face x = 0 into two tetrahedra; the
/// other subcells stay hexahedra.
///
/// # Panics
///
/// If `n` or `p` is 0.
pub fn mixed_box(n: usize, p: u32) -> HostMesh {
    let _ = (n, p);
    todo!()
}

/// `num_cells` P1 triangles in a strip along the x axis, [0, ⌈num_cells/2⌉] × [0, 1] (gdim 2).
pub fn triangle_strip(num_cells: usize) -> HostMesh {
    let _ = num_cells;
    todo!()
}
