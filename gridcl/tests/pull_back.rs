//! M9 acceptance of the pull-back (docs/plans/M9.md, acceptance coverage; ADR-0039, items 13
//! and 14): it recovers the reference points of Gauss points on every cell type and (TD, GD)
//! pair, and gives the same reference points at every vector size, bitwise on the CPU runtime.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{physical_points_block, pull_back_block};
use gridcl_device::KernelKind;
use gridcl_device::testing::with_quad_tile;
use gridcl_testing::HostMesh;

/// The default tolerance of the pull-back in `F` (ADR-0039, item 13).
fn default_tolerance<F: Real>() -> f64 {
    if size_of::<F>() == 8 { 1e-13 } else { 1e-5 }
}

/// The pull-back of the Gauss points of degree 2p of `h` recovers them to the tolerance of `F`,
/// with the last step within the default tolerance.
fn recovers<F, TD, GD>(client: &Client, what: &str, h: &HostMesh, p: u32)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let what = format!("{what} p={p} ({})", name::<F>());
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, 2 * p).expect("rule");
    let x = mesh.physical_points(&rule).expect("X");
    let pb = mesh
        .pull_back(&x, &PullBackOptions::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let td = TD::N;
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let pts = rule.get(block.cell).expect("point set");
        let nq = pts.len();
        let xi = flat(&pb.reference.blocks[b]);
        let step = flat(&pb.last_step.blocks[b]);
        assert_eq!(
            xi.len(),
            block.index_map.num_local() * nq * td,
            "{what}: length"
        );
        for (i, (xi, step)) in xi.chunks_exact(td).zip(&step).enumerate() {
            let want = &pts.points()[(i % nq) * td..(i % nq + 1) * td];
            for (t, (got, want)) in xi.iter().zip(want).enumerate() {
                let err = (got.widen() - want).abs();
                assert!(
                    err <= tol::<F>(),
                    "{what}: {:?} cell {}, point {}, ξ_{t} = {} vs {want} (error {err:e})",
                    block.cell,
                    i / nq,
                    i % nq,
                    got.widen()
                );
            }
            assert!(
                step.widen() <= default_tolerance::<F>(),
                "{what}: last step {} at entry {i}",
                step.widen()
            );
        }
    }
}

fn recover_all<F: Real>(client: &Client) {
    let affine = |x: &[f64]| vec![x[0] + 0.3 * x[1] + 1.5, x[1] - 0.25, 1.2 * x[2] + 0.7];
    for p in 1..=3 {
        let interval = gridcl_testing::interval_box(5, p);
        recovers::<F, D1, D1>(client, "perturbed intervals", &interval.perturbed(0.05), p);
        recovers::<F, D1, D2>(client, "curve in 2D", &lift_curve(&interval, 2), p);
        recovers::<F, D1, D3>(client, "curve in 3D", &lift_curve(&interval, 3), p);
        for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
            let h = gridcl_testing::square_box(cell, 3, p).perturbed(0.05);
            recovers::<F, D2, D2>(client, &format!("perturbed {cell:?} square"), &h, p);
        }
        recovers::<F, D2, D3>(client, "sphere", &gridcl_testing::sphere(1, p), p);
        recovers::<F, D2, D3>(
            client,
            "mixed surface",
            &gridcl_testing::mixed_surface(1, p),
            p,
        );
        for kind in [
            BoxKind::Tetrahedron,
            BoxKind::Hexahedron,
            BoxKind::Prism,
            BoxKind::Mixed,
        ] {
            let h = kind.host(2, p).perturbed(0.05);
            recovers::<F, D3, D3>(client, &format!("perturbed {kind:?} box"), &h, p);
        }
        let h = BoxKind::Pyramid.host(2, p).mapped(3, affine);
        recovers::<F, D3, D3>(client, "affine pyramid box", &h, p);
        recovers::<F, D3, D3>(client, "ball", &gridcl_testing::ball(0, p), p);
    }
}

#[test]
fn recovers_reference_points() {
    for_each_precision!(client(), recover_all);
}

/// The pull-back of every block of `h`, at vector size 1 and tile 1 and at every vector size
/// the device offers within tiles 1 and 16, gives the same ξ and last step: bitwise on the CPU
/// runtime, within the tolerance of `F` on GPUs (ADR-0039, item 14). Each run asserts its vector
/// size through `record_plans`.
fn same_at_every_size<F, TD, GD>(client: &Client, what: &str, h: &HostMesh, p: u32)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, 2 * p).expect("rule");
    let opts = PullBackOptions::default();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let pts = rule.get(block.cell).expect("point set");
        let cells = block.index_map.num_local();
        let what = format!("{what} ({:?}, {cells} cells, {})", block.cell, name::<F>());
        let run = |tile: usize, n: usize| {
            with_quad_tile(tile, || {
                let x = physical_points_block(&mesh, b, pts).expect("X");
                let (xi, step) = at_vector_size(n, KernelKind::PullBack, cells > 0, || {
                    pull_back_block(&mesh, b, &x, &opts).expect("pull-back")
                });
                assert_eq!((xi.tile(), step.tile()), (tile, tile));
                (flat(&xi), flat(&step))
            })
        };
        let (want_xi, want_step) = run(1, 1);
        for tile in [1, 16] {
            for n in vector_sizes(client, size_of::<F>(), tile) {
                let (xi, step) = run(tile, n);
                let at = format!("{what}, tile {tile}, vector size {n}");
                assert_matches(&xi, &want_xi, &format!("ξ of {at}"));
                assert_matches(&step, &want_step, &format!("last step of {at}"));
            }
        }
    }
}

fn invariance<F: Real>(client: &Client) {
    const P: u32 = 2;
    let meshes = [
        (
            "perturbed quadrilaterals",
            gridcl_testing::square_box(Cell2::Quadrilateral, 5, P).perturbed(0.05),
        ),
        (
            "perturbed hexahedra",
            gridcl_testing::cube_box(Cell3::Hexahedron, 3, P).perturbed(0.05),
        ),
        ("mixed surface", gridcl_testing::mixed_surface(1, P)),
        (
            "curve in 3D",
            lift_curve(&gridcl_testing::interval_box(17, P), 3),
        ),
    ];
    for (what, full) in &meshes {
        for cells in INVARIANCE_CELLS {
            let h = full.first_cells(cells);
            match (full.blocks[0].cell, full.gdim) {
                (CellType::Quadrilateral, 2) => {
                    same_at_every_size::<F, D2, D2>(client, what, &h, P)
                }
                (CellType::Hexahedron, _) => same_at_every_size::<F, D3, D3>(client, what, &h, P),
                (CellType::Interval, _) => same_at_every_size::<F, D1, D3>(client, what, &h, P),
                _ => same_at_every_size::<F, D2, D3>(client, what, &h, P),
            }
        }
    }
    // 10^6 + 3 P1 triangles, in release builds.
    if !cfg!(debug_assertions) {
        let strip = gridcl_testing::triangle_strip(LARGE_CELLS);
        same_at_every_size::<F, D2, D2>(client, "triangle strip", &strip, 1);
    }
}

#[test]
fn vector_size_invariance() {
    for_each_precision!(client(), invariance);
}
