//! Test meshes with another node family (M9h, T0a): the same geometry, interpolated at the
//! nodes of another 1D point family.

use gridcl::NodeFamily;
use gridcl_refcell::{LagrangeElement, lagrange_points};

use super::HostMesh;

impl HostMesh {
    /// The mesh with every block's coordinate element moved to the node family `family`, and the
    /// same geometry.
    ///
    /// Each cell's map x(ξ) = Σ_j x_j ψ_j(ξ) is a polynomial of the block's degree in either
    /// family, so it is evaluated at the nodes of the new family; a node shared by several cells
    /// takes the value of the first cell that holds it (the cells agree there up to rounding,
    /// since their maps agree on shared sub-entities). Node and cell numbering, origin ids and
    /// the A.5 order are kept.
    ///
    /// # Panics
    ///
    /// If a block's cell does not offer `family` (GLL is offered on intervals, quadrilaterals
    /// and hexahedra only): a mesh that mixes them would give a shared facet two node layouts.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::NodeFamily;
    ///
    /// let h = gridcl_testing::interval_box(1, 4);
    /// let g = h.with_node_family(NodeFamily::Gll);
    /// assert_eq!(g.blocks[0].family, NodeFamily::Gll);
    /// // The vertices stay; the middle node of the GLL points of degree 4 is the midpoint.
    /// assert_eq!((g.nodes[0], g.nodes[1]), (0.0, 1.0));
    /// assert!((g.nodes[3] - 0.5).abs() < 1e-15);
    /// // The other interior nodes move off the equispaced positions 1/4 and 3/4.
    /// assert!((g.nodes[2] - 0.25).abs() > 0.05);
    /// ```
    pub fn with_node_family(&self, family: NodeFamily) -> HostMesh {
        let gd = self.gdim;
        let mut out = self.clone();
        let mut done = vec![false; self.num_nodes()];
        for (block, new_block) in self.blocks.iter().zip(&mut out.blocks) {
            let (cell, degree) = (block.cell, block.degree);
            let points = lagrange_points(cell, degree, family).unwrap_or_else(|e| {
                panic!("with_node_family({family:?}): {cell:?} cells of degree {degree}: {e}")
            });
            new_block.family = family;
            if block.family == family {
                continue;
            }
            let old = LagrangeElement::new(cell, degree, block.family).expect("element");
            // tab.get(0, i, j) = ψ_j^old at the new node i.
            let tab = old.tabulate(&points, 0).expect("tabulate");
            let k = block.nodes_per_cell();
            for row in block.cells.chunks_exact(k) {
                for (i, &node) in row.iter().enumerate() {
                    let node = node as usize;
                    if done[node] {
                        continue;
                    }
                    done[node] = true;
                    for g in 0..gd {
                        out.nodes[node * gd + g] = row
                            .iter()
                            .enumerate()
                            .map(|(j, &n)| tab.get(0, i, j) * self.nodes[n as usize * gd + g])
                            .sum();
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use gridcl::NodeFamily;
    use gridcl_dims::{Cell2, Cell3};

    use crate::reference::host;
    use crate::{HostMesh, cube_box, interval_box, square_box, triangle_strip};

    /// X and J of `a` (equispaced) and `b` at the Gauss points of degree 2p agree to the f64
    /// tolerance scaled by `a`'s amplification (ADR-0042, item 2), relative to the largest value
    /// of each.
    fn same_geometry(a: &HostMesh, b: &HostMesh, what: &str) {
        use crate::reference::{Amplification, rel_err, tolerance, tolerance_scale};
        for blk in 0..a.blocks.len() {
            let block = &a.blocks[blk];
            let rule = gridcl_refcell::gauss_rule(block.cell, 2 * block.degree);
            let pts = &rule.points;
            let scale = |kind| tolerance_scale(block.cell, block.degree, block.family, pts, kind);
            let checks = [
                (
                    "X",
                    host::points(a, blk, pts),
                    host::points(b, blk, pts),
                    scale(Amplification::Values),
                ),
                (
                    "J",
                    host::jacobians(a, blk, pts),
                    host::jacobians(b, blk, pts),
                    scale(Amplification::Derivatives),
                ),
            ];
            for (name, want, got, scale) in checks {
                let err = rel_err(&got, &want);
                let tol = tolerance::<f64>(scale);
                assert!(
                    err <= tol,
                    "{what}, block {blk}: {name} differs by {err:e} > {tol:e}"
                );
            }
        }
    }

    #[test]
    fn gll_meshes_reproduce_their_equispaced_geometry() {
        let degrees = if cfg!(debug_assertions) { 1..=5 } else { 1..=8 };
        for p in degrees {
            let meshes = [
                ("interval box", interval_box(3, p).perturbed(0.05)),
                (
                    "quadrilateral box",
                    square_box(Cell2::Quadrilateral, 2, p).perturbed(0.05),
                ),
                (
                    "quadrilateral surface",
                    square_box(Cell2::Quadrilateral, 2, p)
                        .mapped(3, |x| vec![x[0], x[1], 0.3 * (x[0] * x[1] * 3.0).sin()]),
                ),
                (
                    "hexahedron box",
                    cube_box(Cell3::Hexahedron, 2, p).perturbed(0.05),
                ),
            ];
            for (what, h) in meshes {
                let g = h.with_node_family(NodeFamily::Gll);
                assert!(g.blocks.iter().all(|b| b.family == NodeFamily::Gll));
                assert_eq!(g.node_ids, h.node_ids);
                assert!(
                    g.blocks
                        .iter()
                        .zip(&h.blocks)
                        .all(|(a, b)| a.cells == b.cells)
                );
                let what = format!("{what}, p = {p}");
                same_geometry(&h, &g, &what);
                // And back.
                same_geometry(&h, &g.with_node_family(NodeFamily::Equispaced), &what);
            }
        }
    }

    #[test]
    fn gll_moves_interior_nodes_from_degree_three() {
        for p in 1..=4 {
            let h = cube_box(Cell3::Hexahedron, 1, p);
            let g = h.with_node_family(NodeFamily::Gll);
            let moved = h
                .nodes
                .iter()
                .zip(&g.nodes)
                .filter(|(a, b)| (*a - *b).abs() > 1e-14)
                .count();
            if p <= 2 {
                assert_eq!(moved, 0, "p = {p}");
            } else {
                assert!(moved > 0, "p = {p}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "Triangle")]
    fn gll_is_rejected_on_triangles() {
        triangle_strip(2).with_node_family(NodeFamily::Gll);
    }
}
