//! M3 acceptance: factor kernels on E-vectors agree with a scalar L-vector reference kernel at
//! every vector size from 1 up to 16 that the device offers and the tile holds, and at tile
//! widths 1 and 16, bitwise on the CPU runtime and within the tolerances on GPUs.
//!
//! X, J, det J and K on every cell type and every (TD, GD) pair, facet normals in codimension 0
//! and normals in codimension 1, at p = 2 with 0, 1, 15, 16 and 17 cells; J and det J also on
//! 10^6 + 3 P1 triangles. Factor kernels plan with max_vector = tile, so tile width 1 runs at
//! size 1 (ADR-0012); each run asserts its size through `record_plans`.

mod common;

use common::reference::{self, RefFactor};
use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{
    FieldShape, QuadField, detj_block, facet_normals_block, jacobians_block, normals_block,
    physical_points_block, pseudo_inverse_block,
};
use gridcl_device::KernelKind;
use gridcl_device::testing::with_quad_tile;
use gridcl_testing::HostMesh;

const P: u32 = 2;
const RULE_DEGREE: u32 = 3;

/// Runs `compute` at every vector size the device offers within `tile`, for tile widths 1 and
/// 16, and compares each result with `want`.
fn sweep<F: Real, S: FieldShape>(
    client: &Client,
    what: &str,
    num_cells: usize,
    want: &[F],
    compute: impl Fn() -> QuadField<F, S>,
) {
    for tile in [1, 16] {
        for n in vector_sizes(client, size_of::<F>(), tile) {
            let field = with_quad_tile(tile, || {
                at_vector_size(n, KernelKind::GeomFactors, num_cells > 0, &compute)
            });
            assert_eq!(field.tile(), tile);
            assert_eq!(field.num_cells(), num_cells);
            assert_matches(
                &flat(&field),
                want,
                &format!("{what}, tile {tile}, vector size {n}"),
            );
        }
    }
}

/// X, J, det J and K of every block of `h`.
fn common_factors<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    client: &Client,
    what: &str,
    h: &HostMesh,
) {
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, RULE_DEGREE).expect("rule");
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let pts = rule.get(block.cell).expect("point set");
        let cells = block.index_map.num_local();
        let what = format!("{what} ({:?}, {cells} cells, {})", block.cell, name::<F>());
        let want = reference::points(&mesh, b, pts);
        sweep(client, &format!("X of {what}"), cells, &want, || {
            physical_points_block(&mesh, b, pts).expect("X")
        });
        let want = reference::factor(&mesh, b, pts, RefFactor::Jacobian);
        sweep(client, &format!("J of {what}"), cells, &want, || {
            jacobians_block(&mesh, b, pts).expect("J")
        });
        let want = reference::factor(&mesh, b, pts, RefFactor::DetJ);
        sweep(client, &format!("det J of {what}"), cells, &want, || {
            detj_block(&mesh, b, pts).expect("det J")
        });
        let want = reference::factor(&mesh, b, pts, RefFactor::Pinv);
        sweep(client, &format!("K of {what}"), cells, &want, || {
            pseudo_inverse_block(&mesh, b, pts).expect("K")
        });
    }
}

/// Facet normals of every block of the codimension-0 mesh `h`.
fn facet_factors<F: Real, TD: TopoDim + Codim0<TD>>(client: &Client, what: &str, h: &HostMesh) {
    let mesh = build::<F, TD, TD>(h, client);
    let rule = FacetQuadRule::for_grid(&mesh, RULE_DEGREE).expect("facet rule");
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let fps = rule.get(block.cell).expect("facet point set");
        let cells = block.index_map.num_local();
        let want = reference::facet_normals(&mesh, b, fps);
        sweep(
            client,
            &format!("facet normals of {what} ({cells} cells, {})", name::<F>()),
            cells,
            &want,
            || facet_normals_block(&mesh, b, fps).expect("facet normals"),
        );
    }
}

/// Normals of every block of the codimension-1 mesh `h`.
fn normal_factors<F: Real, TD: TopoDim + Codim1<GD>, GD: Dim>(
    client: &Client,
    what: &str,
    h: &HostMesh,
) {
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, RULE_DEGREE).expect("rule");
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let pts = rule.get(block.cell).expect("point set");
        let cells = block.index_map.num_local();
        let want = reference::factor(&mesh, b, pts, RefFactor::Normals);
        sweep(
            client,
            &format!("normals of {what} ({cells} cells, {})", name::<F>()),
            cells,
            &want,
            || normals_block(&mesh, b, pts).expect("normals"),
        );
    }
}

fn check<F: Real>(client: &Client) {
    use BoxKind::*;
    // Boxes with at least 17 cells of each type.
    let boxes = [
        (Interval, 17),
        (Triangle, 3),
        (Quadrilateral, 5),
        (Tetrahedron, 2),
        (Hexahedron, 3),
        (Prism, 3),
        (Pyramid, 2),
    ];
    for (kind, n) in boxes {
        let full = kind.host(n, P).perturbed(0.05);
        for cells in INVARIANCE_CELLS {
            let h = full.first_cells(cells);
            let what = format!("{kind:?} box");
            match kind.tdim() {
                1 => {
                    common_factors::<F, D1, D1>(client, &what, &h);
                    facet_factors::<F, D1>(client, &what, &h);
                    common_factors::<F, D1, D2>(client, "curve in 2D", &lift_curve(&h, 2));
                    common_factors::<F, D1, D3>(client, "curve in 3D", &lift_curve(&h, 3));
                    normal_factors::<F, D1, D2>(client, "curve in 2D", &lift_curve(&h, 2));
                }
                2 => {
                    common_factors::<F, D2, D2>(client, &what, &h);
                    facet_factors::<F, D2>(client, &what, &h);
                }
                _ => {
                    common_factors::<F, D3, D3>(client, &what, &h);
                    facet_factors::<F, D3>(client, &what, &h);
                }
            }
        }
    }
    let surface = gridcl_testing::mixed_surface(1, P);
    for cells in INVARIANCE_CELLS {
        let h = surface.first_cells(cells);
        common_factors::<F, D2, D3>(client, "mixed surface", &h);
        normal_factors::<F, D2, D3>(client, "mixed surface", &h);
    }
    // J and det J on 10^6 + 3 P1 triangles.
    let strip = gridcl_testing::triangle_strip(LARGE_CELLS);
    let mesh = build::<F, D2, D2>(&strip, client);
    let rule = QuadRule::for_grid(&mesh, 2).expect("rule");
    let pts = rule.get(Cell2::Triangle).expect("point set");
    let want = reference::factor(&mesh, 0, pts, RefFactor::Jacobian);
    sweep(
        client,
        "J of the triangle strip",
        LARGE_CELLS,
        &want,
        || jacobians_block(&mesh, 0, pts).expect("J"),
    );
    let want = reference::factor(&mesh, 0, pts, RefFactor::DetJ);
    sweep(
        client,
        "det J of the triangle strip",
        LARGE_CELLS,
        &want,
        || detj_block(&mesh, 0, pts).expect("det J"),
    );
}

#[test]
fn factor_kernels_match_lvector_reference() {
    for_each_precision!(client(), check);
}
