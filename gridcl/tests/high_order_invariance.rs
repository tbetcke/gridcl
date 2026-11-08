//! M9h acceptance, bullet 5 (docs/plans/M9h.md, acceptance coverage; ADR-0042, items 2 and 8):
//! at p = 1–8 (pyramids p ≤ 3), the E-vector, X, J, det J, K, normals, facet normals and the
//! pull-back are invariant under the vector size, the tile width and a second run:
//!
//! - **vector sizes:** every size the device offers up to 16 at tile width 16, against size 1,
//!   for 0, 1, 15, 16 and 17 cells of every cell type and (TD, GD) pair; and at p = 3–8 for
//!   16k + 3 cells with a Gauss rule of degree 1, the mesh built outside
//!   `testing::with_properties` and the calls inside it with `max_page_size` a third of the
//!   E-vector or of the J field, whichever is smaller, so that the E-vector, J and K span at
//!   least three chunks (quadrilaterals; hexahedra in release builds);
//! - **tiles:** tile widths 1 and 16;
//! - **determinism:** two builds and runs give bitwise identical arrays, on every device.
//!
//! Comparisons are bitwise on the CPU runtime (up to the sign of zero) and within the tolerance
//! of the precision scaled by max(1, A / A_ref) on GPUs; the pull-back's last step is compared
//! on the CPU runtime only (on GPUs a converged step is rounding noise, see pull_back.rs). Every
//! check runs under the degree policy and every strategy ([`for_each_strategy!`]); each run
//! builds its own E-vector on a copy of the mesh. p ≥ 5 runs in release builds only.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{
    detj_block, facet_normals_block, jacobians_block, normals_block, physical_points_block,
    pseudo_inverse_block, pull_back_block,
};
use gridcl_device::KernelKind;
use gridcl_device::testing::{with_properties, with_quad_tile};
use gridcl_testing::HostMesh;
use gridcl_testing::reference::{self, field_f64};

/// The cells of the large case at p = 3–8 (ADR-0042, item 8).
const LARGE_HIGH_ORDER_CELLS: usize = 16 * 1024 + 3;

/// One compared array: a name, the values widened to f64 and the tolerance scale on GPUs.
struct Quantity {
    name: String,
    values: Vec<f64>,
    scale: f64,
    /// Compared on bitwise devices only (the pull-back's last step).
    bitwise_only: bool,
}

/// Asserts `got` equals `want`, quantity by quantity: [`assert_invariant`], or bitwise on every
/// device if `bitwise`.
fn compare<F: Real>(got: &[Quantity], want: &[Quantity], bitwise: bool, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: quantity counts");
    for (g, w) in got.iter().zip(want) {
        let at = format!("{} of {what}", w.name);
        if bitwise {
            let g: Vec<u64> = g.values.iter().map(|x| x.to_bits()).collect();
            let w: Vec<u64> = w.values.iter().map(|x| x.to_bits()).collect();
            assert!(g == w, "{at}: two runs differ");
        } else if !w.bitwise_only || gridcl_testing::bitwise_device() {
            assert_invariant::<F>(&g.values, &w.values, w.scale, &at);
        }
    }
}

/// The rule degree of the invariance meshes: twice the highest degree.
fn rule_degree(h: &HostMesh) -> u32 {
    2 * h.blocks.iter().map(|b| b.degree).max().unwrap_or(1)
}

/// The GPU tolerance scales of a mesh at the Gauss points of `degree`: X, J and the pull-back
/// of each block; all 1 on bitwise devices, where they are not used.
struct Scales {
    x: f64,
    j: f64,
    pull_back: Vec<f64>,
}

impl Scales {
    fn of<F, TD, GD>(mesh: &Mesh<F, TD, GD>, h: &HostMesh, degree: u32) -> Scales
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let blocks = mesh.cell_blocks().len();
        if gridcl_testing::bitwise_device() {
            return Scales {
                x: 1.0,
                j: 1.0,
                pull_back: vec![1.0; blocks],
            };
        }
        let gauss = |cell| gridcl_refcell::gauss_rule(cell, degree).points;
        Scales {
            x: block_scale(h, gauss, Amplification::Values),
            j: block_scale(h, gauss, Amplification::Derivatives),
            pull_back: (0..blocks)
                .map(|b| {
                    let cell = mesh.cell_blocks()[b].cell.into();
                    reference::pull_back_scale(mesh, b, &gauss(cell)).expect("scale")
                })
                .collect(),
        }
    }
}

/// The E-vector, X, J, det J, K, ξ and the last step of every block of a copy of `mesh` with an
/// empty E-vector cache.
fn common_quantities<F, TD, GD>(mesh: &Mesh<F, TD, GD>, degree: u32, s: &Scales) -> Vec<Quantity>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let mesh = mesh.try_clone().expect("try_clone");
    let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
    let opts = PullBackOptions::default();
    let mut out = Vec::new();
    let mut push = |name: String, values: Vec<f64>, scale: f64, bitwise_only: bool| {
        out.push(Quantity {
            name,
            values,
            scale,
            bitwise_only,
        })
    };
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let pts = rule.get(block.cell).expect("point set");
        let e: Vec<f64> = mesh
            .evector(b)
            .expect("E-vector")
            .chunks()
            .iter()
            .flat_map(|c| c.data.to_host().expect("read"))
            .map(Real::widen)
            .collect();
        push(format!("E-vector of block {b}"), e, s.x, false);
        let x = physical_points_block(&mesh, b, pts).expect("X");
        push(
            format!("X of block {b}"),
            field_f64(&x).unwrap(),
            s.x,
            false,
        );
        let j = jacobians_block(&mesh, b, pts).expect("J");
        push(
            format!("J of block {b}"),
            field_f64(&j).unwrap(),
            s.j,
            false,
        );
        let d = detj_block(&mesh, b, pts).expect("det J");
        push(
            format!("det J of block {b}"),
            field_f64(&d).unwrap(),
            s.j,
            false,
        );
        let k = pseudo_inverse_block(&mesh, b, pts).expect("K");
        push(
            format!("K of block {b}"),
            field_f64(&k).unwrap(),
            s.j,
            false,
        );
        let (xi, step) = pull_back_block(&mesh, b, &x, &opts).expect("pull-back");
        let pb = s.pull_back[b];
        push(
            format!("ξ of block {b}"),
            field_f64(&xi).unwrap(),
            pb,
            false,
        );
        push(
            format!("last step of block {b}"),
            field_f64(&step).unwrap(),
            pb,
            true,
        );
    }
    out
}

/// Facet normals of every block of a copy of the codimension-0 `mesh`.
fn facet_quantities<F: Real, TD: TopoDim + Codim0<TD>>(
    mesh: &Mesh<F, TD, TD>,
    degree: u32,
    scale: f64,
) -> Vec<Quantity> {
    let mesh = mesh.try_clone().expect("try_clone");
    let rule = FacetQuadRule::for_grid(&mesh, degree).expect("facet rule");
    (0..mesh.cell_blocks().len())
        .map(|b| {
            let fps = rule.get(mesh.cell_blocks()[b].cell).expect("facet points");
            let n = facet_normals_block(&mesh, b, fps).expect("facet normals");
            Quantity {
                name: format!("facet normals of block {b}"),
                values: field_f64(&n).unwrap(),
                scale,
                bitwise_only: false,
            }
        })
        .collect()
}

/// Normals of every block of a copy of the codimension-1 `mesh`.
fn normal_quantities<F, TD, GD>(mesh: &Mesh<F, TD, GD>, degree: u32, scale: f64) -> Vec<Quantity>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let mesh = mesh.try_clone().expect("try_clone");
    let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
    (0..mesh.cell_blocks().len())
        .map(|b| {
            let pts = rule.get(mesh.cell_blocks()[b].cell).expect("point set");
            let n = normals_block(&mesh, b, pts).expect("normals");
            Quantity {
                name: format!("normals of block {b}"),
                values: field_f64(&n).unwrap(),
                scale,
                bitwise_only: false,
            }
        })
        .collect()
}

/// Runs `compute` at tile 16 and every vector size the device offers, under every strategy, and
/// compares each with vector size 1. Each run asserts its vector size through `record_plans`.
fn sweep_vector_sizes<F: Real>(
    client: &Client,
    what: &str,
    cells: usize,
    compute: impl Fn() -> Vec<Quantity>,
) {
    let at = |n: usize| {
        with_quad_tile(16, || {
            at_vector_size(n, KernelKind::GeomFactors, cells > 0, &compute)
        })
    };
    for_each_strategy!(|run| {
        let want = at(1);
        for n in vector_sizes(client, size_of::<F>(), 16).into_iter().skip(1) {
            let what = format!("{what} ({cells} cells), vector size {n}, under {run}");
            compare::<F>(&at(n), &want, false, &what);
        }
    });
}

/// Rule 8's small cases: [`INVARIANCE_CELLS`] of every mesh.
struct VectorSizes<'a> {
    client: &'a Client,
}

impl MeshVisitor for VectorSizes<'_> {
    fn common<F, TD, GD>(&mut self, what: &str, full: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let degree = rule_degree(full);
        for cells in INVARIANCE_CELLS {
            let h = full.first_cells(cells);
            let mesh = build::<F, TD, GD>(&h, self.client);
            let s = Scales::of(&mesh, &h, degree);
            sweep_vector_sizes::<F>(self.client, what, cells, || {
                common_quantities(&mesh, degree, &s)
            });
        }
    }

    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, what: &str, full: &HostMesh) {
        let degree = rule_degree(full);
        for cells in INVARIANCE_CELLS {
            let h = full.first_cells(cells);
            let mesh = build::<F, TD, TD>(&h, self.client);
            let s = Scales::of(&mesh, &h, degree);
            sweep_vector_sizes::<F>(self.client, what, cells, || {
                facet_quantities(&mesh, degree, s.j)
            });
        }
    }

    fn normals<F, TD, GD>(&mut self, what: &str, full: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
        let degree = rule_degree(full);
        for cells in INVARIANCE_CELLS {
            let h = full.first_cells(cells);
            let mesh = build::<F, TD, GD>(&h, self.client);
            let s = Scales::of(&mesh, &h, degree);
            sweep_vector_sizes::<F>(self.client, what, cells, || {
                normal_quantities(&mesh, degree, s.j)
            });
        }
    }
}

/// Rule 8's large case at degree `p`: [`LARGE_HIGH_ORDER_CELLS`] cells of `full` with a Gauss
/// rule of degree 1. The mesh is built outside `with_properties`; the calls run inside it with
/// `max_page_size` a third of the E-vector or of the J field, whichever is smaller, which must
/// exceed the tabulation and one E-vector tile.
fn large<F, TD, GD>(client: &Client, what: &str, full: &HostMesh, p: u32)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let h = full.first_cells(LARGE_HIGH_ORDER_CELLS);
    assert_eq!(h.num_cells(), LARGE_HIGH_ORDER_CELLS, "{what}: cells");
    let mesh = build::<F, TD, GD>(&h, client);
    let s = Scales::of(&mesh, &h, 1);
    let k = h.blocks[0].nodes_per_cell();
    let elem = size_of::<F>();
    let cells = LARGE_HIGH_ORDER_CELLS.next_multiple_of(PAD);
    let evector_bytes = cells * k * GD::N * elem;
    let jacobian_bytes = cells * TD::N * GD::N * elem;
    let limit = evector_bytes.min(jacobian_bytes) / 3;
    let rule = gridcl_refcell::gauss_rule(h.blocks[0].cell, 1);
    let tabulation_bytes = (1 + TD::N) * rule.weights.len() * k * elem;
    let tile_bytes = 16 * k * GD::N * elem;
    assert!(
        limit > tabulation_bytes.max(tile_bytes),
        "{what} p={p}: limit {limit} B below the tabulation ({tabulation_bytes} B) or one \
         E-vector tile ({tile_bytes} B)"
    );
    let mut props = client.properties().clone();
    props.memory.max_page_size = limit as u64;
    with_properties(props, || {
        let fresh = mesh.try_clone().expect("try_clone");
        let chunks = fresh.evector(0).expect("E-vector").chunks().len();
        assert!(
            chunks >= 3,
            "{what} p={p}: the E-vector spans {chunks} chunks"
        );
        let rule = QuadRule::for_grid(&fresh, 1).expect("rule");
        let pts = rule.get(fresh.cell_blocks()[0].cell).expect("point set");
        let chunks = jacobians_block(&fresh, 0, pts).expect("J").chunks().len();
        assert!(chunks >= 3, "{what} p={p}: J spans {chunks} chunks");
        sweep_vector_sizes::<F>(client, &format!("{what} p={p}"), h.num_cells(), || {
            common_quantities(&mesh, 1, &s)
        });
    });
}

fn vector_sizes_all<F: Real>(client: &Client) {
    for p in degrees(false) {
        visit_high_order_meshes::<F>(MeshSet::Invariance, p, &mut VectorSizes { client });
        if p >= 3 {
            let quads = gridcl_testing::square_box(Cell2::Quadrilateral, 129, p).perturbed(0.05);
            large::<F, D2, D2>(client, "large quadrilateral box", &quads, p);
            if !cfg!(debug_assertions) {
                let hexes = gridcl_testing::cube_box(Cell3::Hexahedron, 26, p).perturbed(0.05);
                large::<F, D3, D3>(client, "large hexahedron box", &hexes, p);
            }
        }
    }
}

#[test]
fn vector_size_invariance() {
    for_each_precision!(client(), vector_sizes_all);
}

/// Tile widths 1 and 16 give the same results (rule 8).
struct Tiles<'a> {
    client: &'a Client,
}

/// Runs `compute` at tile widths 1 and 16 under every strategy and compares them.
fn sweep_tiles<F: Real>(what: &str, compute: impl Fn() -> Vec<Quantity>) {
    for_each_strategy!(|run| {
        let want = with_quad_tile(1, &compute);
        let got = with_quad_tile(16, &compute);
        compare::<F>(
            &got,
            &want,
            false,
            &format!("{what}, tile 16 vs 1, under {run}"),
        );
    });
}

impl MeshVisitor for Tiles<'_> {
    fn common<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let degree = rule_degree(h);
        let mesh = build::<F, TD, GD>(h, self.client);
        let s = Scales::of(&mesh, h, degree);
        sweep_tiles::<F>(what, || common_quantities(&mesh, degree, &s));
    }

    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, what: &str, h: &HostMesh) {
        let degree = rule_degree(h);
        let mesh = build::<F, TD, TD>(h, self.client);
        let s = Scales::of(&mesh, h, degree);
        sweep_tiles::<F>(what, || facet_quantities(&mesh, degree, s.j));
    }

    fn normals<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
        let degree = rule_degree(h);
        let mesh = build::<F, TD, GD>(h, self.client);
        let s = Scales::of(&mesh, h, degree);
        sweep_tiles::<F>(what, || normal_quantities(&mesh, degree, s.j));
    }
}

fn tiles_all<F: Real>(client: &Client) {
    for p in degrees(false) {
        visit_high_order_meshes::<F>(MeshSet::Invariance, p, &mut Tiles { client });
    }
}

#[test]
fn tile_invariance() {
    for_each_precision!(client(), tiles_all);
}

/// Two builds and runs give bitwise identical arrays on every device ("Determinism").
struct Twice<'a> {
    client: &'a Client,
}

/// Runs `compute` twice under every strategy, each on its own build, and compares bitwise.
fn sweep_twice<F: Real>(what: &str, compute: impl Fn() -> Vec<Quantity>) {
    for_each_strategy!(|run| {
        compare::<F>(&compute(), &compute(), true, &format!("{what} under {run}"));
    });
}

impl MeshVisitor for Twice<'_> {
    fn common<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let degree = rule_degree(h);
        let s = Scales {
            x: 1.0,
            j: 1.0,
            pull_back: vec![1.0; h.blocks.len()],
        };
        sweep_twice::<F>(what, || {
            common_quantities(&build::<F, TD, GD>(h, self.client), degree, &s)
        });
    }

    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, what: &str, h: &HostMesh) {
        let degree = rule_degree(h);
        sweep_twice::<F>(what, || {
            facet_quantities(&build::<F, TD, TD>(h, self.client), degree, 1.0)
        });
    }

    fn normals<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
        let degree = rule_degree(h);
        sweep_twice::<F>(what, || {
            normal_quantities(&build::<F, TD, GD>(h, self.client), degree, 1.0)
        });
    }
}

fn twice_all<F: Real>(client: &Client) {
    for p in degrees(false) {
        visit_high_order_meshes::<F>(MeshSet::Accuracy, p, &mut Twice { client });
    }
}

#[test]
fn two_runs_are_bitwise_identical() {
    for_each_precision!(client(), twice_all);
}
