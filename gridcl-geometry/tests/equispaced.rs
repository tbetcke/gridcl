//! M5 T5: `equispaced_nodes`. GLL interval, quadrilateral and hexahedron meshes at p = 2–4,
//! affine and curved, give the host interpolant (the GLL element tabulated at the equispaced
//! points, applied to each cell's nodes) in f32 and f64; equispaced blocks come back bitwise
//! unchanged; the result is bitwise reproducible, and the same at tile widths 1 and 16 (bitwise
//! on the CPU runtime, within the tolerance on GPUs).

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_device::testing::with_quad_tile;
use gridcl_geometry::equispaced_nodes;
use gridcl_refcell::{LagrangeElement, lagrange_points};
use gridcl_testing::{HostMesh, TOL_F32, TOL_F64};

fn tol<F: Real>() -> f64 {
    if size_of::<F>() == 4 {
        TOL_F32
    } else {
        TOL_F64
    }
}

/// `h` with the nodes of its equispaced lattice moved to the GLL lattice: every coordinate
/// k / (n·p) becomes (k div p + gll[k mod p]) / n, so each axis-aligned cell holds its GLL
/// nodes in A.5 order and the cells stay affine.
fn gll_lattice(h: &HostMesh, n: usize, p: u32) -> HostMesh {
    let mut line = lagrange_points(CellType::Interval, p, NodeFamily::Gll).unwrap();
    line.sort_by(f64::total_cmp);
    let (n, p) = (n as f64, p as usize);
    let nodes = h
        .nodes
        .iter()
        .map(|&x| {
            let k = (x * n * p as f64).round() as usize;
            let (cell, local) = (k / p, k % p);
            (cell as f64 + line[local]) / n
        })
        .collect();
    HostMesh { nodes, ..h.clone() }
}

/// `h` (on the unit box) bent by a smooth map that keeps every coordinate's range: a curved
/// mesh. With `gdim` above `h.gdim`, the extra coordinates are smooth functions of the others.
fn curved(h: &HostMesh, gdim: usize) -> HostMesh {
    let pi = std::f64::consts::PI;
    h.mapped(gdim, |x| {
        let s: f64 = x.iter().map(|&xi| (pi * xi).sin()).product();
        let mut y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(g, &xi)| xi + 0.05 * (1.0 + 0.3 * g as f64) * s)
            .collect();
        while y.len() < gdim {
            y.push(0.2 * (pi * x[0]).cos() * s);
        }
        y
    })
}

/// The mesh of `h` with `family` on the blocks whose cell `gll` accepts, and equispaced
/// elements on the others.
fn build<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    client: &Client,
    h: &HostMesh,
    gll: impl Fn(CellType) -> bool,
) -> Mesh<F, TD, GD> {
    assert_eq!(h.gdim, GD::N);
    let mut builder = MeshBuilder::<F, TD, GD>::new(client).nodes(&h.nodes, None);
    for block in &h.blocks {
        let element = CoordinateElement::<TD> {
            cell: TD::Cell::try_from(block.cell).unwrap(),
            degree: block.degree,
            family: if gll(block.cell) {
                NodeFamily::Gll
            } else {
                NodeFamily::Equispaced
            },
        };
        builder = builder.cell_block(element, &block.cells, None);
    }
    builder.build().unwrap()
}

/// The result of `equispaced_nodes` as f64.
fn converted<G: DeviceGrid>(mesh: &G) -> Vec<f64> {
    let out = equispaced_nodes(mesh).unwrap();
    assert_eq!(out.len(), mesh.nodes().len());
    out.to_host().unwrap().iter().map(|x| x.widen()).collect()
}

/// Every occurrence of every node of a GLL block of `mesh` holds the host interpolant to the
/// tolerance (relative to the largest coordinate); returns which nodes GLL blocks touch.
fn check_interpolant<G: DeviceGrid>(mesh: &G, got: &[f64], what: &str) -> Vec<bool> {
    let gd = <G::GD as Dim>::N;
    let x: Vec<f64> = mesh
        .nodes()
        .to_host()
        .unwrap()
        .iter()
        .map(|v| v.widen())
        .collect();
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    let mut touched = vec![false; x.len() / gd];
    let mut worst = 0.0f64;
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let gb = mesh.geometry_block(b);
        if gb.element.family != NodeFamily::Gll {
            continue;
        }
        let cell: CellType = gb.element.cell.into();
        let p = gb.element.degree;
        let element = LagrangeElement::new(cell, p, NodeFamily::Gll).unwrap();
        let lattice = lagrange_points(cell, p, NodeFamily::Equispaced).unwrap();
        let tab = element.tabulate(&lattice, 0).unwrap();
        let k = gb.nodes_per_cell();
        let dofmap = gb.dofmap.data.to_host().unwrap();
        for c in 0..block.index_map.num_local() {
            let row = &dofmap[c * k..(c + 1) * k];
            for (i, &node) in row.iter().enumerate() {
                let node = node as usize;
                touched[node] = true;
                for g in 0..gd {
                    let want: f64 = (0..k)
                        .map(|j| tab.get(0, i, j) * x[row[j] as usize * gd + g])
                        .sum();
                    worst = worst.max((got[node * gd + g] - want).abs() / scale);
                }
            }
        }
    }
    assert!(
        worst <= tol::<G::F>(),
        "{what}: relative error {worst:e} > {:e}",
        tol::<G::F>()
    );
    touched
}

fn bits(x: &[f64]) -> Vec<u64> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The interpolant on a GLL mesh of `h`, and the same result twice and at tile widths 1
/// and 16.
fn check_gll<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    client: &Client,
    h: &HostMesh,
    what: &str,
) {
    let what = format!(
        "{what}, {}",
        if size_of::<F>() == 4 { "f32" } else { "f64" }
    );
    let mesh = build::<F, TD, GD>(client, h, |_| true);
    let got = converted(&mesh);
    let touched = check_interpolant(&mesh, &got, &what);
    assert!(
        touched.iter().all(|&t| t),
        "{what}: every node is a GLL node"
    );
    let nodes: Vec<f64> = mesh
        .nodes()
        .to_host()
        .unwrap()
        .iter()
        .map(|x| x.widen())
        .collect();
    // At p = 2 the GLL points are the equispaced ones.
    if mesh.geometry_block(0).element.degree > 2 {
        assert_ne!(bits(&got), bits(&nodes), "{what}: the GLL nodes moved");
    }
    assert_eq!(
        bits(&converted(&mesh)),
        bits(&got),
        "{what}: a second run differs"
    );
    let narrow = with_quad_tile(1, || converted(&mesh));
    let wide = with_quad_tile(16, || converted(&mesh));
    if gridcl_testing::bitwise_device() {
        assert_eq!(bits(&narrow), bits(&got), "{what}: tile 1 differs");
        assert_eq!(bits(&wide), bits(&got), "{what}: tile 16 differs");
    } else {
        check_interpolant(&mesh, &narrow, &format!("{what}, tile 1"));
        check_interpolant(&mesh, &wide, &format!("{what}, tile 16"));
    }
}

/// Affine and curved GLL meshes of the boxes at p = 2–4, in f32 and, where the device has
/// it, f64.
fn gll_meshes<F: Real>(client: &Client) {
    for p in 2..=4 {
        let interval = gll_lattice(&gridcl_testing::interval_box(3, p), 3, p);
        check_gll::<F, D1, D1>(client, &interval, &format!("affine interval, p = {p}"));
        check_gll::<F, D1, D1>(
            client,
            &curved(&interval, 1),
            &format!("curved interval, p = {p}"),
        );
        check_gll::<F, D1, D2>(
            client,
            &curved(&interval, 2),
            &format!("curved interval in 2D, p = {p}"),
        );

        let square = gll_lattice(
            &gridcl_testing::square_box(Cell2::Quadrilateral, 2, p),
            2,
            p,
        );
        check_gll::<F, D2, D2>(client, &square, &format!("affine quadrilaterals, p = {p}"));
        check_gll::<F, D2, D2>(
            client,
            &curved(&square, 2),
            &format!("curved quadrilaterals, p = {p}"),
        );
        check_gll::<F, D2, D3>(
            client,
            &curved(&square, 3),
            &format!("curved quadrilaterals in 3D, p = {p}"),
        );

        let cube = gll_lattice(&gridcl_testing::cube_box(Cell3::Hexahedron, 2, p), 2, p);
        check_gll::<F, D3, D3>(client, &cube, &format!("affine hexahedra, p = {p}"));
        check_gll::<F, D3, D3>(
            client,
            &curved(&cube, 3),
            &format!("curved hexahedra, p = {p}"),
        );
    }
}

#[test]
fn equispaced_gll_meshes_give_the_interpolant_f32() {
    gll_meshes::<f32>(&gridcl_testing::test_client());
}

#[test]
fn equispaced_gll_meshes_give_the_interpolant_f64() {
    let client = gridcl_testing::test_client();
    if gridcl_testing::supports::<f64>(&client) {
        gll_meshes::<f64>(&client);
    }
}

/// Equispaced meshes, every cell type, come back bitwise unchanged.
fn equispaced_unchanged<F: Real>(client: &Client) {
    let check = |got: Vec<f64>, nodes: Vec<f64>, what: &str| {
        assert_eq!(bits(&got), bits(&nodes), "{what}");
    };
    let nodes = |b: &DeviceBuffer<F>| -> Vec<f64> {
        b.to_host().unwrap().iter().map(|x| x.widen()).collect()
    };
    let h = curved(&gridcl_testing::interval_box(3, 3), 1);
    let m = build::<F, D1, D1>(client, &h, |_| false);
    check(converted(&m), nodes(m.nodes()), "intervals");
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        let h = curved(&gridcl_testing::square_box(cell, 2, 3), 2);
        let m = build::<F, D2, D2>(client, &h, |_| false);
        check(converted(&m), nodes(m.nodes()), &format!("{cell:?}"));
    }
    let h = gridcl_testing::mixed_box(2, 2).perturbed(0.05);
    let m = build::<F, D3, D3>(client, &h, |_| false);
    check(converted(&m), nodes(m.nodes()), "mixed box");
}

#[test]
fn equispaced_meshes_come_back_unchanged() {
    let client = gridcl_testing::test_client();
    equispaced_unchanged::<f32>(&client);
    if gridcl_testing::supports::<f64>(&client) {
        equispaced_unchanged::<f64>(&client);
    }
}

/// A mixed mesh with GLL hexahedra and equispaced tetrahedra and pyramids: the nodes of the
/// hexahedra hold the interpolant, all other nodes are unchanged.
#[test]
fn equispaced_converts_only_gll_blocks() {
    let client = gridcl_testing::test_client();
    let h = gridcl_testing::mixed_box(2, 3).perturbed(0.05);
    let mesh = build::<f32, D3, D3>(&client, &h, |c| c == CellType::Hexahedron);
    assert!(
        mesh.cell_blocks()
            .iter()
            .any(|b| b.cell == Cell3::Hexahedron && b.index_map.num_local() > 0)
    );
    let got = converted(&mesh);
    let touched = check_interpolant(&mesh, &got, "GLL hexahedra of the mixed box");
    let nodes = mesh.nodes().to_host().unwrap();
    let mut untouched = 0;
    for (node, t) in touched.iter().enumerate() {
        if !t {
            untouched += 1;
            for g in 0..3 {
                assert_eq!(
                    got[node * 3 + g].to_bits(),
                    f64::from(nodes[node * 3 + g]).to_bits(),
                    "node {node} of an equispaced block changed"
                );
            }
        }
    }
    assert!(untouched > 0);
    assert_eq!(bits(&converted(&mesh)), bits(&got), "a second run differs");
}
