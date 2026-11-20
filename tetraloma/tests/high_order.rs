//! M9h acceptance, bullet 1 (docs/plans/M9h.md, acceptance coverage; ADR-0042, items 2 and 3):
//! on every cell type and (TD, GD) pair, at p = 1–8 (pyramids p ≤ 3) and with both node
//! families where offered, X, J, det J, K, normals and facet normals from the device match the
//! f64 host reference (`tetraloma_testing::reference`, tetraloma-refcell's tabulation at the stored
//! nodes), and the pull-back recovers the reference points, to the tolerance of the precision
//! scaled by max(1, A / A_ref): the amplification factor of X for points, of J for J, det J and
//! the normals, of J times κ_F(J) / TD for K, and of the coefficients times κ_F(J) / TD for the
//! pull-back (ADR-0044); for all but X also by max(1, n / n_ref) and max(1, R) (ADR-0046).
//!
//! Every check runs under the degree policy and under every strategy forced
//! ([`for_each_strategy!`]; under `TETRALOMA_GEOM_STRATEGY` only that setting). The references are
//! computed once per mesh and precision; each run uses a copy of the mesh with an empty E-vector
//! cache, so the E-vector is built under the run's strategy too. One test per degree ([`degree_tests!`]); p ≥ 5 runs in release builds only.

use crate::common;

use common::*;
use cubecl::prelude::Client;
use tetraloma::prelude::*;
use tetraloma::{
    detj_block, facet_normals_block, jacobians_block, normals_block, physical_points_block,
    pseudo_inverse_block,
};
use tetraloma_testing::HostMesh;
use tetraloma_testing::reference::{self, field_f64};

/// The rule degree of a mesh: twice its highest degree.
fn rule_degree(h: &HostMesh) -> u32 {
    2 * h.blocks.iter().map(|b| b.degree).max().unwrap_or(1)
}

/// The quadrature points of `cell` at `degree`, for the tolerance scales.
fn gauss(degree: u32) -> impl Fn(CellType) -> Vec<f64> {
    move |cell| tetraloma_refcell::gauss_rule(cell, degree).points
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
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let p = pts.points();
            let sj = reference::jacobian_scale(&mesh, b, p).expect("J scale");
            let x_ref = reference::points(&mesh, b, p).expect("X");
            let j_ref = reference::jacobians(&mesh, b, p).expect("J");
            let d_ref = reference::detj(&mesh, b, p).expect("det J");
            let k_ref = reference::pseudo_inverses(&mesh, b, p).expect("K");
            let sk = reference::pinv_scale(&mesh, b, p).expect("K scale");
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
                assert_scaled::<F>(&k, &k_ref, sk, &format!("K of {at}"));
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
            let scale = reference::jacobian_scale(&mesh, b, fps.points().points())
                .expect("facet normal scale");
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
        for (b, block) in mesh.cell_blocks().iter().enumerate() {
            let pts = rule.get(block.cell).expect("point set");
            let want = reference::normals(&mesh, b, pts.points()).expect("normals");
            let scale = reference::jacobian_scale(&mesh, b, pts.points()).expect("normal scale");
            for_each_strategy!(|run| {
                let mesh = fresh(&mesh);
                let n = field_f64(&normals_block(&mesh, b, pts).expect("n")).unwrap();
                let at = format!("normals of {what}, block {b}, under {run}");
                assert_scaled::<F>(&n, &want, scale, &at);
            });
        }
    }
}

fn factors<F: Real>(client: &Client, p: u32) {
    visit_high_order_meshes::<F>(MeshSet::Accuracy, p, &mut Factors { client });
}

fn factors_at(p: u32) {
    for_each_precision!(client(), factors, p);
}

crate::degree_tests!(factors_match_host_reference, factors_at);

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

fn pull_backs<F: Real>(client: &Client, p: u32) {
    visit_high_order_meshes::<F>(MeshSet::Accuracy, p, &mut PullBacks { client });
}

fn pull_backs_at(p: u32) {
    for_each_precision!(client(), pull_backs, p);
}

crate::degree_tests!(pull_back_recovers_reference_points, pull_backs_at);

/// The policy's open-ended last ranges at p = 9 and 10 (docs/design.md, "Geometric factors":
/// the last range "resolves to a strategy known to compile at p = 9–10"; ADR-0042, item 3):
/// on every cell type but pyramids, in each node family the cell offers, and for every
/// (TD, GD) pair, every family launches under the policy of the device's class: unperturbed
/// boxes (codimension 0, with the facet normals), lifted curves in 2D and 3D, the sphere and
/// the mixed surface (codimension 1 with the normals; the kernels' comptime variants depend
/// on GD). The launches cover exactly the policy's families, none runs `direct`, every result
/// is finite and det J is positive. No accuracy is asserted: rule 2's tolerances are defined
/// to p = 8.
fn open_ended<F: Real>(client: &Client, p: u32) {
    use tetraloma_geometry::testing::{
        Strategy, StrategyFamily, StrategyOverride, record_strategies,
    };
    use tetraloma_testing::{cube_box, interval_box, mixed_surface, sphere, square_box};

    type Fields = Vec<(&'static str, Vec<f64>)>;

    /// A field of every block, flattened.
    macro_rules! all {
        ($field:expr) => {
            $field
                .blocks
                .iter()
                .map(|b| field_f64(b).expect("to_host"))
                .collect::<Vec<_>>()
                .concat()
        };
    }

    /// X, J, det J, K and the pull-back's ξ of `mesh`, then `more` (the normals of its
    /// codimension), under the policy; checked as the function's doc says.
    fn launch<F, TD, GD>(
        client: &Client,
        what: &str,
        h: &HostMesh,
        normals: StrategyFamily,
        more: impl Fn(&Mesh<F, TD, GD>, &QuadRule<F, TD>, u32) -> Fields,
    ) where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        use StrategyFamily::*;
        let mesh = build::<F, TD, GD>(h, client);
        let degree = rule_degree(h);
        let rule = QuadRule::for_grid(&mesh, degree).expect("rule");
        // The scoped hook comes before `TETRALOMA_GEOM_STRATEGY`, so this is the policy always.
        let (fields, ran) = record_strategies(|| {
            with_every_family(StrategyOverride::Policy, || {
                let x = mesh.physical_points(&rule).expect("X");
                let pb = mesh
                    .pull_back(&x, &PullBackOptions::default())
                    .unwrap_or_else(|e| panic!("pull-back of {what}: {e}"));
                let mut fields: Fields = vec![
                    ("det J", all!(mesh.detj(&rule).expect("det J"))),
                    ("X", all!(x)),
                    ("J", all!(mesh.jacobians(&rule).expect("J"))),
                    ("K", all!(mesh.pseudo_inverse_jacobians(&rule).expect("K"))),
                    ("ξ", all!(pb.reference)),
                ];
                fields.extend(more(&mesh, &rule, degree));
                fields
            })
        });
        let at = format!("{what} ({})", name::<F>());
        let mut families: Vec<StrategyFamily> = ran.iter().map(|&(f, _)| f).collect();
        families.sort();
        families.dedup();
        let mut want = vec![EVector, Points, Jacobian, PullBack];
        if normals != EVector {
            want.push(normals);
        }
        want.sort();
        assert_eq!(families, want, "{at}: the families launched ({ran:?})");
        assert!(
            ran.iter().all(|&(_, s)| s != Strategy::Direct),
            "{at}: the open-ended range ran direct: {ran:?}"
        );
        for (quantity, values) in &fields {
            assert!(!values.is_empty(), "{quantity} of {at}: empty");
            assert!(
                values.iter().all(|v| v.is_finite()),
                "{quantity} of {at}: not finite (under {ran:?})"
            );
        }
        assert_eq!(fields[0].0, "det J");
        assert!(
            fields[0].1.iter().all(|&d| d > 0.0),
            "det J of {at}: not positive (under {ran:?})"
        );
    }

    /// Codimension 0, with the facet normals.
    fn volume<F: Real, TD: TopoDim + Codim0<TD>>(client: &Client, what: &str, h: &HostMesh) {
        launch::<F, TD, TD>(
            client,
            what,
            h,
            StrategyFamily::FacetNormals,
            |mesh, _, degree| {
                let facets = FacetQuadRule::for_grid(mesh, degree).expect("facet rule");
                vec![(
                    "facet normals",
                    all!(mesh.facet_normals(&facets).expect("facet normals")),
                )]
            },
        );
    }

    /// Codimension 1, with the normals.
    fn surface<F, TD, GD>(client: &Client, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD> + EmbeddedIn<GD>,
        GD: Dim,
    {
        launch::<F, TD, GD>(client, what, h, StrategyFamily::Normals, |mesh, rule, _| {
            vec![("normals", all!(mesh.normals(rule).expect("normals")))]
        });
    }

    let named = |what: &str, f: NodeFamily| format!("{what} p={p} {f:?}");
    for f in node_families(&[CellType::Interval]) {
        let h = interval_box(3, p).with_node_family(f);
        volume::<F, D1>(client, &named("intervals", f), &h);
        surface::<F, D1, D2>(client, &named("curve in 2D", f), &lift_curve(&h, 2));
        // Codimension 2 has no normals: `EVector` stands for "no further family".
        launch::<F, D1, D3>(
            client,
            &named("curve in 3D", f),
            &lift_curve(&h, 3),
            StrategyFamily::EVector,
            |_, _, _| Vec::new(),
        );
    }
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        for f in node_families(&[cell.into()]) {
            let h = square_box(cell, 2, p).with_node_family(f);
            volume::<F, D2>(client, &named(&format!("{cell:?} box"), f), &h);
        }
    }
    for (what, h) in [
        ("sphere", sphere(0, p)),
        ("mixed surface", mixed_surface(1, p)),
    ] {
        surface::<F, D2, D3>(client, &named(what, NodeFamily::Equispaced), &h);
    }
    for cell in [Cell3::Tetrahedron, Cell3::Hexahedron, Cell3::Prism] {
        for f in node_families(&[cell.into()]) {
            let h = cube_box(cell, 1, p).with_node_family(f);
            volume::<F, D3>(client, &named(&format!("{cell:?} box"), f), &h);
        }
    }
}

/// One test per degree above the measured ones, in release builds only (as p ≥ 5): Metal, CUDA
/// and the CPU runtime run them; hosted CI does not (scripts/ci-tetraloma-shard.sh).
mod open_ended_ranges_launch {
    use super::*;

    #[test]
    #[cfg_attr(debug_assertions, ignore = "p ≥ 5 runs in release builds only")]
    fn p9() {
        for_each_precision!(client(), open_ended, 9);
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "p ≥ 5 runs in release builds only")]
    fn p10() {
        for_each_precision!(client(), open_ended, 10);
    }
}
