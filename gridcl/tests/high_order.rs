//! M9h acceptance, bullet 1 (docs/plans/M9h.md, acceptance coverage; ADR-0042, items 2 and 3):
//! on every cell type and (TD, GD) pair, at p = 1–8 (pyramids p ≤ 3) and with both node
//! families where offered, X, J, det J, K, normals and facet normals from the device match the
//! f64 host reference (`gridcl_testing::reference`, gridcl-refcell's tabulation at the stored
//! nodes), and the pull-back recovers the reference points, to the tolerance of the precision
//! scaled by max(1, A / A_ref): the amplification factor of X for points, of J for J and the
//! quantities derived from it, and of the coefficients times κ(J) for the pull-back.
//!
//! Every check runs under the degree policy and under every strategy forced
//! ([`for_each_strategy!`]; under `GRIDCL_GEOM_STRATEGY` only that setting). The references are
//! computed once per mesh and precision; each run uses a copy of the mesh with an empty E-vector
//! cache, so the E-vector is built under the run's strategy too. p ≥ 5 runs in release builds only ([`max_degree`]).

use crate::common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{
    detj_block, facet_normals_block, jacobians_block, normals_block, physical_points_block,
    pseudo_inverse_block,
};
use gridcl_testing::HostMesh;
use gridcl_testing::reference::{self, field_f64};

/// The rule degree of a mesh: twice its highest degree.
fn rule_degree(h: &HostMesh) -> u32 {
    2 * h.blocks.iter().map(|b| b.degree).max().unwrap_or(1)
}

/// The quadrature points of `cell` at `degree`, for the tolerance scales.
fn gauss(degree: u32) -> impl Fn(CellType) -> Vec<f64> {
    move |cell| gridcl_refcell::gauss_rule(cell, degree).points
}

/// A copy of `mesh` with an empty E-vector cache, so each strategy run builds its own E-vector
/// (and pull-back coefficients).
fn fresh<F, TD, GD>(mesh: &Mesh<F, TD, GD>) -> Mesh<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    mesh.try_clone().expect("try_clone")
}

/// Factors of every block against the host reference.
struct Factors<'a> {
    client: &'a Client,
}

impl MeshVisitor for Factors<'_> {
    fn common<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let mesh = build::<F, TD, GD>(h, self.client);
        let degree = rule_degree(h);
        let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
        let sx = block_scale(h, gauss(degree), Amplification::Values);
        let sj = block_scale(h, gauss(degree), Amplification::Derivatives);
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let p = pts.points();
            let x_ref = reference::points(&mesh, b, p).expect("X");
            let j_ref = reference::jacobians(&mesh, b, p).expect("J");
            let d_ref = reference::detj(&mesh, b, p).expect("det J");
            let k_ref = reference::pseudo_inverses(&mesh, b, p).expect("K");
            for_each_strategy!(|run| {
                let mesh = fresh(&mesh);
                let at = format!("{what}, block {b}, under {run}");
                let x = field_f64(&physical_points_block(&mesh, b, pts).expect("X")).unwrap();
                assert_scaled::<F>(&x, &x_ref, sx, &format!("X of {at}"));
                let j = field_f64(&jacobians_block(&mesh, b, pts).expect("J")).unwrap();
                assert_scaled::<F>(&j, &j_ref, sj, &format!("J of {at}"));
                let d = field_f64(&detj_block(&mesh, b, pts).expect("det J")).unwrap();
                assert_scaled::<F>(&d, &d_ref, sj, &format!("det J of {at}"));
                let k = field_f64(&pseudo_inverse_block(&mesh, b, pts).expect("K")).unwrap();
                assert_scaled::<F>(&k, &k_ref, sj, &format!("K of {at}"));
            });
        }
    }

    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, what: &str, h: &HostMesh) {
        let mesh = build::<F, TD, TD>(h, self.client);
        let degree = rule_degree(h);
        let rule = FacetQuadRule::for_grid(&mesh, degree).expect("facet rule");
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let fps = rule.get(block.cell).expect("facet points");
            let want = reference::facet_normals(&mesh, b, fps).expect("facet normals");
            let scale = block_scale(
                h,
                |_| fps.points().points().to_vec(),
                Amplification::Derivatives,
            );
            for_each_strategy!(|run| {
                let mesh = fresh(&mesh);
                let n = field_f64(&facet_normals_block(&mesh, b, fps).expect("n")).unwrap();
                let at = format!("facet normals of {what}, block {b}, under {run}");
                assert_scaled::<F>(&n, &want, scale, &at);
            });
        }
    }

    fn normals<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
        let mesh = build::<F, TD, GD>(h, self.client);
        let degree = rule_degree(h);
        let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
        let scale = block_scale(h, gauss(degree), Amplification::Derivatives);
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let want = reference::normals(&mesh, b, pts.points()).expect("normals");
            for_each_strategy!(|run| {
                let mesh = fresh(&mesh);
                let n = field_f64(&normals_block(&mesh, b, pts).expect("n")).unwrap();
                let at = format!("normals of {what}, block {b}, under {run}");
                assert_scaled::<F>(&n, &want, scale, &at);
            });
        }
    }
}

fn factors<F: Real>(client: &Client) {
    for p in degrees(false) {
        visit_high_order_meshes::<F>(MeshSet::Accuracy, p, &mut Factors { client });
    }
}

#[test]
fn factors_match_host_reference() {
    for_each_precision!(client(), factors);
}

/// The pull-back of the device's X at the Gauss points of degree 2p recovers those points.
struct PullBacks<'a> {
    client: &'a Client,
}

impl MeshVisitor for PullBacks<'_> {
    fn common<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let mesh = build::<F, TD, GD>(h, self.client);
        let rule = QuadRule::for_grid(&mesh, rule_degree(h)).expect("rule");
        let td = TD::N;
        let mut wants = Vec::new();
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let nq = pts.len();
            let want: Vec<f64> = (0..block.index_map.num_local() * nq)
                .flat_map(|i| pts.points()[(i % nq) * td..(i % nq + 1) * td].to_vec())
                .collect();
            let scale = reference::pull_back_scale(&mesh, b, pts.points()).expect("scale");
            wants.push((want, scale));
        }
        for_each_strategy!(|run| {
            let mesh = fresh(&mesh);
            let x = mesh.physical_points(&rule).expect("X");
            let pb = mesh
                .pull_back(&x, &PullBackOptions::default())
                .unwrap_or_else(|e| panic!("{what} under {run}: {e}"));
            for (b, (want, scale)) in wants.iter().enumerate() {
                let xi = field_f64(&pb.reference.blocks[b]).unwrap();
                let at = format!("ξ of {what}, block {b}, under {run}");
                assert_scaled::<F>(&xi, want, *scale, &at);
            }
        });
    }

    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, _: &str, _: &HostMesh) {}

    fn normals<F, TD, GD>(&mut self, _: &str, _: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
    }
}

fn pull_backs<F: Real>(client: &Client) {
    for p in degrees(false) {
        visit_high_order_meshes::<F>(MeshSet::Accuracy, p, &mut PullBacks { client });
    }
}

#[test]
fn pull_back_recovers_reference_points() {
    for_each_precision!(client(), pull_backs);
}
