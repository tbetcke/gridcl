//! M9 acceptance of the pull-back (docs/plans/M9.md, acceptance coverage; ADR-0039, items 13
//! and 14): it recovers the reference points of Gauss points on every cell type and (TD, GD)
//! pair, and gives the same reference points at every vector size, bitwise on the CPU runtime.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{physical_points_block, pull_back_block};
use gridcl_device::KernelKind;
use gridcl_device::testing::{with_properties, with_quad_tile};
use gridcl_testing::HostMesh;

/// The default tolerance of the pull-back in `F` (ADR-0039, item 13).
fn default_tolerance<F: Real>() -> f64 {
    if size_of::<F>() == 8 { 1e-13 } else { 1e-5 }
}

/// Asserts the last steps `got` match `want`: as `assert_matches` on the CPU runtime, and
/// within the default tolerance, absolute, elsewhere. A converged step is rounding noise of
/// the order of ε, so its relative error between vector sizes says nothing (ADR-0039, item 14
/// sets the GPU tolerance for ξ).
fn assert_steps_match<F: Real>(got: &[F], want: &[F], what: &str) {
    if gridcl_testing::bitwise_device() {
        return assert_matches(got, want, what);
    }
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        let (a, b) = (a.widen(), b.widen());
        assert!(
            (a - b).abs() <= default_tolerance::<F>(),
            "{what}: entry {i}: {a} vs {b}"
        );
    }
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
/// runtime; on GPUs ξ within the tolerance of `F` (ADR-0039, item 14) and the last step within
/// the default tolerance, absolute. Each run asserts its vector size through `record_plans`.
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
                assert_steps_match(&step, &want_step, &format!("last step of {at}"));
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
    // 10^6 + 3 P1 triangles, in release builds. The strip is folded into a triangle wave of
    // period 16 in x, so every cell stays a unit cell near the origin: unfolded, its
    // coordinates reach 5 · 10^5 and in f32 the points and nodes carry errors of 10^−2 cell
    // widths, which the GPU's operation order turns into different ξ at each vector size.
    if !cfg!(debug_assertions) {
        let strip = gridcl_testing::triangle_strip(LARGE_CELLS)
            .mapped(2, |x| vec![8.0 - (x[0] % 16.0 - 8.0).abs(), x[1]]);
        same_at_every_size::<F, D2, D2>(client, "folded triangle strip", &strip, 1);
    }
}

#[test]
fn vector_size_invariance() {
    for_each_precision!(client(), invariance);
}

/// The vertex centroid of the reference cell.
fn reference_centroid(cell: CellType) -> Vec<f64> {
    let v = gridcl_refcell::vertices(cell);
    (0..v[0].len())
        .map(|d| v.iter().map(|p| p[d]).sum::<f64>() / v.len() as f64)
        .collect()
}

/// The box [−0.1, 1.1] that confines the pull-back's iterates in every coordinate (ADR-0046),
/// with the margins rounded to f32 as the kernels hold them.
const BOX: [f64; 2] = [-0.1f32 as f64, 1.1f32 as f64];

/// One affine cell `h` (degree 1), and the same cell stretched by `stretch` about its vertex
/// centroid: the stretched cell's Gauss points pull back on the first cell to
/// ξ_c + stretch (ξ_g − ξ_c), ξ_c the reference centroid, in one Newton step, where that point
/// lies in the box [`BOX`]^TD (ADR-0046); some of them lie outside the reference cell. A point
/// beyond the box stays on its boundary, the clamp of its preimage, and does not converge.
/// Returns the numbers of converged points outside the reference cell and of points beyond the
/// box.
fn outside<F, TD, GD>(client: &Client, what: &str, h: &HostMesh, stretch: f64) -> (usize, usize)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let what = format!("{what} ({}), stretched by {stretch}", name::<F>());
    let cell = h.blocks[0].cell;
    assert_eq!(
        h.blocks[0].cells.len(),
        h.blocks[0].nodes_per_cell(),
        "{what}: one cell"
    );
    let num_vertices = gridcl_refcell::vertices(cell).len();
    let c: Vec<f64> = (0..h.gdim)
        .map(|g| {
            h.blocks[0].cells[..num_vertices]
                .iter()
                .map(|&n| h.nodes[n as usize * h.gdim + g])
                .sum::<f64>()
                / num_vertices as f64
        })
        .collect();
    let stretched = h.mapped(h.gdim, |x| {
        x.iter()
            .zip(&c)
            .map(|(x, c)| c + stretch * (x - c))
            .collect()
    });
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, 2).expect("rule");
    let x = build::<F, TD, GD>(&stretched, client)
        .physical_points(&rule)
        .expect("X");
    let pb = mesh
        .pull_back(&x, &PullBackOptions::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let pts = rule.get(mesh.cell_blocks()[0].cell).expect("point set");
    let xi_c = reference_centroid(cell);
    let td = TD::N;
    let (mut outside_cell, mut beyond_box) = (0, 0);
    for (q, (xi, step)) in flat(&pb.reference.blocks[0])
        .chunks_exact(td)
        .zip(flat(&pb.last_step.blocks[0]))
        .enumerate()
    {
        let want: Vec<f64> = (0..td)
            .map(|d| xi_c[d] + stretch * (pts.points()[q * td + d] - xi_c[d]))
            .collect();
        let in_box = want.iter().all(|&w| (BOX[0]..=BOX[1]).contains(&w));
        for (d, (got, want)) in xi.iter().zip(&want).enumerate() {
            let want = want.clamp(BOX[0], BOX[1]);
            let err = (got.widen() - want).abs();
            assert!(
                err <= 4.0 * tol::<F>(),
                "{what}: point {q}, ξ_{d} = {} vs {want} (error {err:e})",
                got.widen()
            );
        }
        if in_box {
            assert!(
                step.widen() <= default_tolerance::<F>(),
                "{what}: last step {} at point {q}",
                step.widen()
            );
            if want.iter().any(|&w| !(0.0..=1.0).contains(&w)) {
                outside_cell += 1;
            }
        } else {
            assert!(
                step.widen() > default_tolerance::<F>(),
                "{what}: point {q} beyond the box converged (last step {})",
                step.widen()
            );
            beyond_box += 1;
        }
    }
    (outside_cell, beyond_box)
}

fn outside_all<F: Real>(client: &Client) {
    let (mut outside_cell, mut beyond_box) = (0, 0);
    for stretch in [1.8, 3.0] {
        let mut add = |(a, b): (usize, usize)| {
            outside_cell += a;
            beyond_box += b;
        };
        add(outside::<F, D1, D1>(
            client,
            "interval",
            &gridcl_testing::interval_box(3, 1).first_cells(1),
            stretch,
        ));
        for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
            let h = gridcl_testing::square_box(cell, 2, 1).first_cells(1);
            add(outside::<F, D2, D2>(
                client,
                &format!("{cell:?}"),
                &h,
                stretch,
            ));
        }
        for kind in [BoxKind::Tetrahedron, BoxKind::Hexahedron, BoxKind::Prism] {
            let h = kind.host(2, 1).first_cells(1);
            add(outside::<F, D3, D3>(
                client,
                &format!("{kind:?}"),
                &h,
                stretch,
            ));
        }
    }
    assert!(outside_cell > 0, "no converged point outside the cell");
    assert!(beyond_box > 0, "no point beyond the box");
}

#[test]
fn outside_points_of_affine_cells_converge_outside_the_reference_cell() {
    for_each_precision!(client(), outside_all);
}

/// On curved cells, one iteration leaves the points that are not reached in one step with a
/// last step above the tolerance; no iteration leaves the centroid with a last step of
/// `F::MAX`.
fn few_iterations<F: Real>(client: &Client) {
    let mesh = build::<F, D3, D3>(&gridcl_testing::ball(0, 2), client);
    let rule = QuadRule::for_grid(&mesh, 4).expect("rule");
    let x = mesh.physical_points(&rule).expect("X");
    let converged = mesh
        .pull_back(&x, &PullBackOptions::default())
        .expect("pull-back");
    let one = PullBackOptions {
        max_iterations: 1,
        ..PullBackOptions::default()
    };
    let one = mesh.pull_back(&x, &one).expect("pull-back");
    let none = PullBackOptions {
        max_iterations: 0,
        ..PullBackOptions::default()
    };
    let none = mesh.pull_back(&x, &none).expect("pull-back");
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let what = format!("{:?} ({})", block.cell, name::<F>());
        let want = flat(&converged.reference.blocks[b]);
        let got = flat(&one.reference.blocks[b]);
        let step = flat(&one.last_step.blocks[b]);
        let mut off = 0;
        for (i, s) in step.iter().enumerate() {
            let (w, g) = (&want[i * 3..i * 3 + 3], &got[i * 3..i * 3 + 3]);
            if w.iter()
                .zip(g)
                .any(|(w, g)| (w.widen() - g.widen()).abs() > 1e-3)
            {
                off += 1;
                assert!(
                    s.widen() > default_tolerance::<F>(),
                    "{what}: entry {i} is not converged, but its last step is {}",
                    s.widen()
                );
            }
        }
        assert!(off > 0, "{what}: every point converged in one step");
        let centroid = reference_centroid(block.cell.into());
        for xi in flat(&none.reference.blocks[b]).as_chunks::<3>().0 {
            for (got, want) in xi.iter().zip(&centroid) {
                assert!((got.widen() - want).abs() < 1e-6, "{what}: start {xi:?}");
            }
        }
        let max = if size_of::<F>() == 4 {
            f64::from(f32::MAX)
        } else {
            f64::MAX
        };
        assert!(
            flat(&none.last_step.blocks[b])
                .iter()
                .all(|s| s.widen() == max),
            "{what}: the last step of no iteration is F::MAX"
        );
    }
}

#[test]
fn last_step_shows_non_convergence() {
    for_each_precision!(client(), few_iterations);
}

/// Inputs with the wrong number of blocks or cells give `LengthMismatch`.
#[test]
fn wrong_lengths_are_rejected() {
    let client = client();
    let mesh = build::<f32, D2, D2>(&gridcl_testing::square_box(Cell2::Triangle, 2, 1), &client);
    let other = build::<f32, D2, D2>(&gridcl_testing::square_box(Cell2::Triangle, 3, 1), &client);
    let rule = QuadRule::for_grid(&mesh, 2).expect("rule");
    let x = other.physical_points(&rule).expect("X");
    let opts = PullBackOptions::default();
    assert!(matches!(
        mesh.pull_back(&x, &opts),
        Err(GridError::LengthMismatch {
            expected: 8,
            found: 18
        })
    ));
    let none = BlockFields { blocks: vec![] };
    assert!(matches!(
        mesh.pull_back(&none, &opts),
        Err(GridError::LengthMismatch {
            expected: 1,
            found: 0
        })
    ));
}

/// ξ, the last step and the chunk counts of the E-vector, the input and both outputs, on a
/// fresh P2 sphere (320 cells, 4 points per cell). The mesh is built outside `limit`, and its
/// E-vector, the input and the outputs inside it.
#[allow(clippy::type_complexity)]
fn chunked_run<F: Real>(client: &Client, limit: Option<u64>) -> (Vec<F>, Vec<F>, [usize; 4]) {
    let mesh = build::<F, D2, D3>(&gridcl_testing::sphere(2, 2), client);
    let mut props = client.properties().clone();
    if let Some(limit) = limit {
        props.memory.max_page_size = limit;
    }
    with_properties(props, || {
        let rule = QuadRule::for_grid(&mesh, 2).expect("rule");
        let x = mesh.physical_points(&rule).expect("X");
        let pb = mesh
            .pull_back(&x, &PullBackOptions::default())
            .expect("pull-back");
        let counts = [
            mesh.evector(0).expect("E-vector").chunks().len(),
            x.blocks[0].chunks().len(),
            pb.reference.blocks[0].chunks().len(),
            pb.last_step.blocks[0].chunks().len(),
        ];
        (
            flat(&pb.reference.blocks[0]),
            flat(&pb.last_step.blocks[0]),
            counts,
        )
    })
}

/// With an injected allocation limit of 1024 elements, the E-vector (18 values per cell), the
/// input (12), ξ (8) and the last step (4) split into 7, 4, 3 and 2 chunks, so launch ranges
/// cut at every chunk start of the four; the results equal the unchunked ones bitwise.
fn chunks<F: Real>(client: &Client) {
    let (xi, step, counts) = chunked_run::<F>(client, None);
    assert_eq!(counts, [1; 4], "without a limit");
    let limit = (1024 * size_of::<F>()) as u64;
    let (xi_c, step_c, counts) = chunked_run::<F>(client, Some(limit));
    assert_eq!(
        counts,
        [7, 4, 3, 2],
        "{}: chunks with the limit",
        name::<F>()
    );
    assert_bits(&xi_c, &xi, "ξ, chunked vs one chunk");
    assert_bits(&step_c, &step, "last step, chunked vs one chunk");
}

#[test]
fn identical_across_chunkings() {
    for_each_precision!(client(), chunks);
}
