//! M3 acceptance: Jacobians on a mixed triangle–quadrilateral surface mesh use one `QuadRule`
//! and match a host reference to 1e-12 (and 1e-5 in f32).

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;

fn check<F: Real>(client: &Client) {
    for p in [1, 2, 3] {
        let h = gridcl_testing::mixed_surface(1, p);
        let mesh = build::<F, D2, D3>(&h, client);
        let cells: Vec<Cell2> = mesh.cell_blocks().iter().map(|b| b.cell).collect();
        assert!(cells.contains(&Cell2::Triangle) && cells.contains(&Cell2::Quadrilateral));
        // One rule serves both cell types.
        let rule = QuadRule::for_grid(&mesh, 2 * p + 1).expect("rule");
        let j = mesh.jacobians(&rule).expect("jacobians");
        assert_eq!(j.blocks.len(), mesh.cell_blocks().len());
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let want = host_jacobians(&h, b, pts.points());
            let got: Vec<f64> = flat(&j.blocks[b]).into_iter().map(Real::widen).collect();
            let err = rel_max_err(&got, &want);
            assert!(
                err <= tol::<F>(),
                "p = {p}, {:?} block, {}: relative max error {err:e}",
                block.cell,
                name::<F>()
            );
        }
    }
}

#[test]
fn mixed_surface_jacobians_match_host() {
    for_each_precision!(client(), check);
}
