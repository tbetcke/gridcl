//! Strategy dispatch (M9h T4; ADR-0042, item 3): every launcher of a degree-dependent family
//! asks the degree policy, and `record_strategies` shows what ran. `loops` runs for every
//! family (T6a, T6b); `sumfact` for the factors on tensor rules (T7a) and facet rules (T7b),
//! and on prisms (T7c), and for the pull-back on quadrilaterals, hexahedra and prisms (T8);
//! `shared` for every family with a table that fits the shared memory (T9); `cooperative` for
//! every family but the E-vector (T10a, T10b).

use gridcl::prelude::*;
use gridcl::{FacetQuadRule, FieldShape, PullBackOptions, VecShape};
use gridcl_geometry::testing::{
    Strategy, StrategyFamily, StrategyOverride, record_strategies, with_strategy,
};

/// Every family on `mesh` with the Gauss rule of degree 2: E-vector, the factors, `normals`
/// (the facet normals of codimension 0 or the normals of codimension 1) and the pull-back.
fn every_family<TD, GD>(
    mesh: &Mesh<f32, TD, GD>,
    normals: impl Fn(&Mesh<f32, TD, GD>, &QuadRule<f32, TD>) -> Result<()>,
) -> Result<()>
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let rule = QuadRule::for_grid(mesh, 2)?;
    let x = mesh.physical_points(&rule)?;
    mesh.jacobians(&rule)?;
    mesh.detj(&rule)?;
    normals(mesh, &rule)?;
    mesh.pull_back(&x, &PullBackOptions::default())?;
    Ok(())
}

/// The families that ran, each once, in order of first launch.
fn families(ran: &[(StrategyFamily, Strategy)]) -> Vec<StrategyFamily> {
    let mut out = Vec::new();
    for (f, _) in ran {
        if !out.contains(f) {
            out.push(*f);
        }
    }
    out
}

#[test]
fn record_strategies_shows_direct() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let sphere = gridcl_testing::sphere(0, 2)
        .build::<f32, D2, D3>(&client)
        .unwrap();
    let ((), ran) = record_strategies(|| {
        with_every(StrategyOverride::Policy, || {
            every_family(&square, |m, _| {
                m.facet_normals(&FacetQuadRule::for_grid(m, 2)?).map(drop)
            })
            .unwrap();
            every_family(&sphere, |m, rule| m.normals(rule).map(drop)).unwrap();
        })
    });
    // The policy is `direct` everywhere until T11b; the E-vector is built at the first factor.
    assert!(ran.iter().all(|(_, s)| *s == Strategy::Direct), "{ran:?}");
    assert_eq!(
        families(&ran),
        [Points, EVector, Jacobian, FacetNormals, PullBack, Normals]
    );
    // The E-vector is cached: one build per mesh.
    assert_eq!(ran.iter().filter(|(f, _)| *f == EVector).count(), 2);
}

#[test]
fn forced_direct_matches_the_policy_bitwise() {
    let client = gridcl_testing::test_client();
    let run = |s| {
        let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 2)
            .perturbed(0.05)
            .build::<f32, D3, D3>(&client)
            .unwrap();
        let rule = QuadRule::for_grid(&mesh, 4).unwrap();
        with_every(s, || {
            let j = mesh.jacobians(&rule).unwrap();
            j.blocks[0].to_host_cell_major().unwrap()
        })
    };
    let policy = run(StrategyOverride::Policy);
    let direct = run(StrategyOverride::Fixed(Strategy::Direct));
    let bits = |v: &Vec<[[f32; 3]; 3]>| -> Vec<u32> {
        v.iter().flatten().flatten().map(|x| x.to_bits()).collect()
    };
    assert_eq!(bits(&policy), bits(&direct));
}

#[test]
fn forced_loops_runs_loops() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let triangle = gridcl_testing::square_box(Cell2::Triangle, 2, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    // Sum factorisation does not apply to a triangle: its fallback `loops` runs.
    for s in [Strategy::Loops, Strategy::SumFactorised] {
        let mesh = triangle.try_clone().unwrap();
        let rule = QuadRule::for_grid(&mesh, 2).unwrap();
        let ((), ran) = record_strategies(|| {
            with_strategy(EVector, StrategyOverride::Fixed(s), || {
                with_strategy(Points, StrategyOverride::Fixed(s), || {
                    with_strategy(Jacobian, StrategyOverride::Fixed(s), || {
                        with_strategy(PullBack, StrategyOverride::Fixed(s), || {
                            let x = mesh.physical_points(&rule).unwrap();
                            mesh.jacobians(&rule).unwrap();
                            mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
                        })
                    })
                })
            })
        });
        assert_eq!(
            ran,
            [
                (Points, Strategy::Loops),
                (EVector, Strategy::Loops),
                (Jacobian, Strategy::Loops),
                (PullBack, Strategy::Loops)
            ],
            "{s:?}"
        );
    }
}

/// X and J on `mesh` with every factor family and the E-vector forced to `s`, cell-major.
fn points_and_jacobians<F: Real>(
    mesh: &Mesh<F, D3, D3>,
    rule: &QuadRule<F, D3>,
    s: Strategy,
) -> (Vec<f64>, Vec<f64>, Vec<(StrategyFamily, Strategy)>) {
    use StrategyFamily::*;
    let mesh = mesh.try_clone().unwrap();
    let s = StrategyOverride::Fixed(s);
    let ((x, j), ran) = record_strategies(|| {
        with_strategy(EVector, s, || {
            with_strategy(Points, s, || {
                with_strategy(Jacobian, s, || {
                    let x = mesh.physical_points(rule).unwrap();
                    let j = mesh.jacobians(rule).unwrap();
                    (
                        x.blocks[0].to_host_cell_major().unwrap(),
                        j.blocks[0].to_host_cell_major().unwrap(),
                    )
                })
            })
        })
    });
    (
        x.into_iter().flatten().map(F::widen).collect(),
        j.into_iter().flatten().flatten().map(F::widen).collect(),
        ran,
    )
}

#[test]
fn forced_sumfact_runs_on_tensor_rules() {
    let client = gridcl_testing::test_client();
    sumfact_on_tensor_rules::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        sumfact_on_tensor_rules::<f64>(&client, 1e-13);
    }
}

/// Sum-factorised X and J of perturbed hexahedra (both node families) and prisms at p = 3
/// against `loops`, relative to the largest entry, within `tol`.
fn sumfact_on_tensor_rules<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    for (cell, family) in [
        (Cell3::Hexahedron, NodeFamily::Equispaced),
        (Cell3::Hexahedron, NodeFamily::Gll),
        (Cell3::Prism, NodeFamily::Equispaced),
    ] {
        let mesh = gridcl_testing::cube_box(cell, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let rule = QuadRule::for_grid(&mesh, 6).unwrap();
        let (x, j, ran) = points_and_jacobians(&mesh, &rule, Strategy::SumFactorised);
        assert_eq!(
            ran,
            [
                (Points, Strategy::SumFactorised),
                // The E-vector is a gather: sum factorisation falls back to `loops`.
                (EVector, Strategy::Loops),
                (Jacobian, Strategy::SumFactorised)
            ]
        );
        let (x_loops, j_loops, _) = points_and_jacobians(&mesh, &rule, Strategy::Loops);
        for (what, got, want) in [("X", &x, &x_loops), ("J", &j, &j_loops)] {
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() <= tol * scale,
                    "{what} {cell:?} {family:?}: {g} vs {w}"
                );
            }
        }
    }
}

/// The values of every block of `field`, concatenated.
fn blocks_f64<F: Real, S: FieldShape>(field: &BlockFields<F, S>) -> Vec<f64> {
    field
        .blocks
        .iter()
        .flat_map(|b| gridcl_testing::reference::field_f64(b).unwrap())
        .collect()
}

/// Named quantities, each flattened.
type Quantities = Vec<(&'static str, Vec<f64>)>;

/// `f` on a fresh copy of `mesh` with every family forced to `s`, and what ran.
fn forced<F, TD, GD>(
    mesh: &Mesh<F, TD, GD>,
    s: Strategy,
    f: impl FnOnce(&Mesh<F, TD, GD>) -> Quantities,
) -> (Quantities, Vec<(StrategyFamily, Strategy)>)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let mesh = mesh.try_clone().unwrap();
    record_strategies(|| with_every(StrategyOverride::Fixed(s), || f(&mesh)))
}

/// The quantities of `f` under `sumfact` against `loops`, relative to each quantity's largest
/// entry, within `tol`; `sumfact` runs for each of `families`, `loops` for everything else.
fn sumfact_against_loops<F, TD, GD>(
    what: &str,
    mesh: &Mesh<F, TD, GD>,
    families: &[StrategyFamily],
    tol: f64,
    f: impl Fn(&Mesh<F, TD, GD>) -> Quantities,
) where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let (got, ran) = forced(mesh, Strategy::SumFactorised, &f);
    for family in families {
        assert!(
            ran.contains(&(*family, Strategy::SumFactorised)),
            "{what}: {family:?} not sum-factorised: {ran:?}"
        );
    }
    assert!(
        ran.iter()
            .all(|(fam, s)| *s == Strategy::Loops || families.contains(fam)),
        "{what}: {ran:?}"
    );
    let (want, _) = forced(mesh, Strategy::Loops, &f);
    for ((name, got), (_, want)) in got.iter().zip(&want) {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol * scale, "{name} of {what}: {g} vs {w}");
        }
    }
}

#[test]
fn forced_sumfact_runs_for_derived_factors_and_facet_rules() {
    let client = gridcl_testing::test_client();
    derived_factors_and_facets::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        derived_factors_and_facets::<f64>(&client, 1e-13);
    }
}

/// det J, K and the facet normals of perturbed quadrilaterals, hexahedra and prisms, and det J,
/// K and the normals of the mixed surface (its quadrilateral block), at p = 3, under `sumfact`
/// against `loops`.
fn derived_factors_and_facets<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    fn codim0<F: Real, TD: TopoDim + Codim0<TD>>(mesh: &Mesh<F, TD, TD>) -> Quantities {
        let rule = QuadRule::for_grid(mesh, 6).unwrap();
        let facets = FacetQuadRule::for_grid(mesh, 6).unwrap();
        vec![
            ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
            (
                "K",
                blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
            ),
            (
                "facet normals",
                blocks_f64(&mesh.facet_normals(&facets).unwrap()),
            ),
        ]
    }
    for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
        let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D2, D2>(client)
            .unwrap();
        let cube = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let factors = [Jacobian, FacetNormals];
        sumfact_against_loops(
            &format!("square {family:?}"),
            &square,
            &factors,
            tol,
            codim0,
        );
        sumfact_against_loops(&format!("cube {family:?}"), &cube, &factors, tol, codim0);
    }
    let prisms = gridcl_testing::cube_box(Cell3::Prism, 2, 3)
        .perturbed(0.05)
        .build::<F, D3, D3>(client)
        .unwrap();
    sumfact_against_loops("prisms", &prisms, &[Jacobian, FacetNormals], tol, codim0);
    let surface = gridcl_testing::mixed_surface(1, 3)
        .build::<F, D2, D3>(client)
        .unwrap();
    sumfact_against_loops(
        "mixed surface",
        &surface,
        &[Jacobian, Normals],
        tol,
        |mesh| {
            let rule = QuadRule::for_grid(mesh, 6).unwrap();
            vec![
                ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
                (
                    "K",
                    blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
                ),
                ("normals", blocks_f64(&mesh.normals(&rule).unwrap())),
            ]
        },
    );
}

#[test]
fn forced_sumfact_pull_back_agrees_with_loops() {
    let client = gridcl_testing::test_client();
    sumfact_pull_backs::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        sumfact_pull_backs::<f64>(&client, 1e-12);
    }
}

/// ξ and the last step of perturbed quadrilaterals, hexahedra (both node families), prisms and
/// the mixed surface (its quadrilateral block, codimension 1) at p = 3, at the Gauss points of
/// degree 6, under `sumfact` against `loops`; every point converges under both.
fn sumfact_pull_backs<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    fn pull_back<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
        mesh: &Mesh<F, TD, GD>,
    ) -> Quantities {
        let x = mesh
            .physical_points(&QuadRule::for_grid(mesh, 6).unwrap())
            .unwrap();
        let pb = mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
        let steps = blocks_f64(&pb.last_step);
        let converged = if size_of::<F>() == 8 { 1e-13 } else { 1e-5 };
        assert!(steps.iter().all(|&s| s <= converged), "{steps:?}");
        vec![("ξ", blocks_f64(&pb.reference))]
    }
    for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
        let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D2, D2>(client)
            .unwrap();
        let cube = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let what = format!("square {family:?}");
        sumfact_against_loops(&what, &square, &[Points, PullBack], tol, pull_back);
        let what = format!("cube {family:?}");
        sumfact_against_loops(&what, &cube, &[Points, PullBack], tol, pull_back);
    }
    let prisms = gridcl_testing::cube_box(Cell3::Prism, 2, 3)
        .perturbed(0.05)
        .build::<F, D3, D3>(client)
        .unwrap();
    sumfact_against_loops("prisms", &prisms, &[Points, PullBack], tol, pull_back);
    let surface = gridcl_testing::mixed_surface(1, 3)
        .build::<F, D2, D3>(client)
        .unwrap();
    sumfact_against_loops(
        "mixed surface",
        &surface,
        &[Points, PullBack],
        tol,
        pull_back,
    );
}

#[test]
fn sumfact_falls_back_to_loops_without_tensor_points() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 2)
        .with_node_family(NodeFamily::Gll)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    let sumfact = StrategyOverride::Fixed(Strategy::SumFactorised);
    let ((), ran) = record_strategies(|| {
        with_strategy(EVector, sumfact, || {
            with_strategy(Points, sumfact, || {
                // The equispaced lattice records no factors.
                gridcl_geometry::equispaced_nodes(&mesh).unwrap();
            })
        })
    });
    assert_eq!(ran, [(Points, Strategy::Loops), (EVector, Strategy::Loops)]);
}

#[test]
fn overrides_nest_per_family() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 2).unwrap();
    // The E-vector is built here, so the records below hold the factors alone.
    mesh.physical_points(&rule).unwrap();
    let ran = |f: &dyn Fn()| record_strategies(f).1;
    let points = || {
        mesh.physical_points(&rule).unwrap();
    };
    let jacobians = || {
        mesh.jacobians(&rule).unwrap();
    };
    let fixed = StrategyOverride::Fixed;
    with_strategy(Points, fixed(Strategy::Loops), || {
        with_strategy(Jacobian, fixed(Strategy::Cooperative), || {
            // Another family's override does not reach the points.
            assert_eq!(ran(&points), [(Points, Strategy::Loops)]);
            assert_eq!(ran(&jacobians), [(Jacobian, Strategy::Cooperative)]);
            // The innermost override wins, and the outer one is back afterwards.
            with_strategy(Jacobian, fixed(Strategy::Direct), || {
                assert_eq!(ran(&jacobians), [(Jacobian, Strategy::Direct)]);
            });
            assert_eq!(ran(&jacobians), [(Jacobian, Strategy::Cooperative)]);
        });
        assert_eq!(ran(&points), [(Points, Strategy::Loops)]);
    });
}

#[test]
fn records_nest_and_survive_panics() {
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 2).unwrap();
    let ((inner, panicked), outer) = record_strategies(|| {
        let ((), inner) = record_strategies(|| {
            mesh.physical_points(&rule).unwrap();
        });
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_strategy(
                StrategyFamily::Jacobian,
                StrategyOverride::Fixed(Strategy::Cooperative),
                || {
                    let _ = record_strategies(|| {
                        mesh.jacobians(&rule).unwrap();
                        panic!("inside");
                    });
                },
            )
        }))
        .is_err();
        (inner, panicked)
    });
    assert!(panicked);
    assert_eq!(
        families(&inner),
        [StrategyFamily::Points, StrategyFamily::EVector]
    );
    // The outer record sees the inner launches, also those of the scope that panicked.
    assert_eq!(
        families(&outer),
        [
            StrategyFamily::Points,
            StrategyFamily::EVector,
            StrategyFamily::Jacobian
        ]
    );
    assert!(outer.contains(&(StrategyFamily::Jacobian, Strategy::Cooperative)));
    // The hook of a scope that panicked is gone: the enclosing one is back.
    with_strategy(
        StrategyFamily::Jacobian,
        StrategyOverride::Fixed(Strategy::Loops),
        || {
            let panicked = std::panic::catch_unwind(|| {
                with_strategy(
                    StrategyFamily::Jacobian,
                    StrategyOverride::Fixed(Strategy::Cooperative),
                    || panic!("inside"),
                )
            });
            assert!(panicked.is_err());
            let ((), ran) = record_strategies(|| {
                mesh.jacobians(&rule).unwrap();
            });
            assert_eq!(ran, [(StrategyFamily::Jacobian, Strategy::Loops)]);
        },
    );
}

#[test]
fn forced_shared_matches_the_kernels_it_stages_for() {
    let client = gridcl_testing::test_client();
    shared_against_kernels::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        shared_against_kernels::<f64>(&client, 1e-12);
    }
}

/// X, J, det J, K, the facet normals and the pull-back's ξ under `shared` against the kernels
/// whose table it stages: `sumfact` on perturbed hexahedra (both node families) and prisms at
/// the Gauss points of degree 6, `loops` on perturbed tetrahedra, at p = 3. Every family but
/// the E-vector runs `shared`. The arithmetic is theirs, so the results are bitwise equal on
/// the CPU runtime, and within `tol` of the largest entry elsewhere.
fn shared_against_kernels<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    fn quantities<F: Real>(mesh: &Mesh<F, D3, D3>) -> Quantities {
        let rule = QuadRule::for_grid(mesh, 6).unwrap();
        let facets = FacetQuadRule::for_grid(mesh, 6).unwrap();
        let x = mesh.physical_points(&rule).unwrap();
        let pb = mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
        vec![
            ("X", blocks_f64(&x)),
            ("J", blocks_f64(&mesh.jacobians(&rule).unwrap())),
            ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
            (
                "K",
                blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
            ),
            (
                "facet normals",
                blocks_f64(&mesh.facet_normals(&facets).unwrap()),
            ),
            ("ξ", blocks_f64(&pb.reference)),
        ]
    }
    let bitwise = gridcl_testing::bitwise_device();
    for (cell, family, staged_for) in [
        (
            Cell3::Hexahedron,
            NodeFamily::Equispaced,
            Strategy::SumFactorised,
        ),
        (Cell3::Hexahedron, NodeFamily::Gll, Strategy::SumFactorised),
        (
            Cell3::Prism,
            NodeFamily::Equispaced,
            Strategy::SumFactorised,
        ),
        (Cell3::Tetrahedron, NodeFamily::Equispaced, Strategy::Loops),
    ] {
        let mesh = gridcl_testing::cube_box(cell, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let what = format!("{} {cell:?} {family:?}", std::any::type_name::<F>());
        let (got, ran) = forced(&mesh, Strategy::Shared, quantities);
        for want in [Points, Jacobian, FacetNormals, PullBack] {
            assert!(ran.contains(&(want, Strategy::Shared)), "{what}: {ran:?}");
        }
        assert!(
            ran.iter()
                .all(|r| r.1 == Strategy::Shared || *r == (EVector, Strategy::Loops)),
            "{what}: {ran:?}"
        );
        let (want, _) = forced(&mesh, staged_for, quantities);
        for ((name, got), (_, want)) in got.iter().zip(&want) {
            assert_eq!(got.len(), want.len());
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (g, w) in got.iter().zip(want) {
                if bitwise {
                    assert_eq!(g.to_bits(), w.to_bits(), "{name} of {what}: {g} vs {w}");
                } else {
                    assert!((g - w).abs() <= tol * scale, "{name} of {what}: {g} vs {w}");
                }
            }
        }
    }
}

/// Where the table exceeds the shared memory per cube, `shared` falls back to `loops`: every
/// family on hexahedra (the tensor table) and tetrahedra (the full tabulation) at p = 2 with
/// 64 bytes of shared memory, against the device's own limit, under which they run `shared`.
#[test]
fn shared_falls_back_to_loops_above_the_limit() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    for cell in [Cell3::Hexahedron, Cell3::Tetrahedron] {
        let mesh = gridcl_testing::cube_box(cell, 1, 2)
            .build::<f32, D3, D3>(&client)
            .unwrap();
        let rule = QuadRule::for_grid(&mesh, 4).unwrap();
        let run = |mesh: &Mesh<f32, D3, D3>| {
            let x = mesh.physical_points(&rule).unwrap();
            mesh.jacobians(&rule).unwrap();
            mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
            vec![]
        };
        let (_, ran) = forced(&mesh, Strategy::Shared, run);
        let want = |s| {
            vec![
                (Points, s),
                (EVector, Strategy::Loops),
                (Jacobian, s),
                (PullBack, s),
            ]
        };
        assert_eq!(ran, want(Strategy::Shared), "{cell:?}");
        let mut props = (*gridcl_device::device_properties(&client)).clone();
        props.hardware.max_shared_memory_size = 64;
        let (_, ran) =
            gridcl_device::testing::with_properties(props, || forced(&mesh, Strategy::Shared, run));
        assert_eq!(ran, want(Strategy::Loops), "{cell:?}");
    }
}

/// Runs `f` with every family overridden by `s`.
fn with_every<R>(s: StrategyOverride, f: impl FnOnce() -> R) -> R {
    StrategyFamily::ALL
        .into_iter()
        .rev()
        .fold(Box::new(f) as Box<dyn FnOnce() -> R>, |f, family| {
            Box::new(move || with_strategy(family, s, f))
        })()
}

#[test]
fn forced_cooperative_is_independent_of_units_per_group() {
    let client = gridcl_testing::test_client();
    cooperative_units::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        cooperative_units::<f64>(&client, 1e-12);
    }
}

/// X, J, det J, K, the facet normals and ξ of the pull-back of X of perturbed hexahedra (both
/// node families; 64 nodes, so L = 32 leaves) and tetrahedra (20 nodes, L = 16), and the
/// normals and ξ of the sphere (10 nodes, L = 8), at p = 3 and the Gauss points of degree 6
/// (every point converges), under `cooperative` with 1, 2 and
/// L units per group, L at most the device's own units per cube, through injected GPU-class
/// properties with `units` units per cube (so the CPU runtime runs groups of several units
/// too), at vector sizes 1 and the widest: bitwise equal to one unit at vector size 1 on the
/// CPU runtime, and within `tol` of the largest entry elsewhere; and within `tol` of `loops`
/// everywhere (another summation order). Every family but the E-vector runs `cooperative`, the
/// E-vector `loops`.
fn cooperative_units<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    fn reference<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
        mesh: &Mesh<F, TD, GD>,
        x: &BlockFields<F, VecShape<GD>>,
    ) -> Vec<f64> {
        let pb = mesh.pull_back(x, &PullBackOptions::default()).unwrap();
        let steps = blocks_f64(&pb.last_step);
        let converged = if size_of::<F>() == 8 { 1e-13 } else { 1e-5 };
        assert!(steps.iter().all(|&s| s <= converged), "{steps:?}");
        blocks_f64(&pb.reference)
    }
    fn volume<F: Real>(mesh: &Mesh<F, D3, D3>) -> Quantities {
        let rule = QuadRule::for_grid(mesh, 6).unwrap();
        let facets = FacetQuadRule::for_grid(mesh, 6).unwrap();
        let x = mesh.physical_points(&rule).unwrap();
        vec![
            ("X", blocks_f64(&x)),
            ("J", blocks_f64(&mesh.jacobians(&rule).unwrap())),
            ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
            (
                "K",
                blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
            ),
            (
                "facet normals",
                blocks_f64(&mesh.facet_normals(&facets).unwrap()),
            ),
            ("ξ", reference(mesh, &x)),
        ]
    }
    fn surface<F: Real>(mesh: &Mesh<F, D2, D3>) -> Quantities {
        let rule = QuadRule::for_grid(mesh, 6).unwrap();
        let x = mesh.physical_points(&rule).unwrap();
        vec![
            ("X", blocks_f64(&x)),
            ("J", blocks_f64(&mesh.jacobians(&rule).unwrap())),
            ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
            (
                "K",
                blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
            ),
            ("normals", blocks_f64(&mesh.normals(&rule).unwrap())),
            ("ξ", reference(mesh, &x)),
        ]
    }
    let volume_mesh = |cell, family| {
        gridcl_testing::cube_box(cell, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap()
    };
    let name = std::any::type_name::<F>();
    for (cell, family, leaves) in [
        (Cell3::Hexahedron, NodeFamily::Equispaced, 32),
        (Cell3::Hexahedron, NodeFamily::Gll, 32),
        (Cell3::Tetrahedron, NodeFamily::Equispaced, 16),
    ] {
        let what = format!("{name} {cell:?} {family:?}");
        let mesh = volume_mesh(cell, family);
        units_against_one(client, &what, &mesh, leaves, tol, volume);
    }
    let sphere = gridcl_testing::sphere(1, 3)
        .build::<F, D2, D3>(client)
        .unwrap();
    units_against_one(client, &format!("{name} sphere"), &sphere, 8, tol, surface);
}

/// The quantities of `f` under `cooperative` with 1, 2 and `leaves` units per group (at most
/// the device's own) at vector sizes 1 and the widest against one unit at vector size 1, and
/// against `loops` (see [`cooperative_units`]).
fn units_against_one<F, TD, GD>(
    client: &cubecl::prelude::Client,
    what: &str,
    mesh: &Mesh<F, TD, GD>,
    leaves: u32,
    tol: f64,
    f: impl Fn(&Mesh<F, TD, GD>) -> Quantities,
) where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    use StrategyFamily::*;
    use gridcl_device::testing::{record_plans, with_properties, with_vector_size};
    use gridcl_device::{KernelKind, VectorSizeOverride};
    let run = |units: u32, vector: VectorSizeOverride| {
        let mut props = (*gridcl_device::device_properties(client)).clone();
        // A GPU class, whose cooperative launches ask for L units per group (CPUs ask for
        // one), and cubes of `units` units.
        props.hardware.num_cpu_cores = None;
        props.hardware.max_units_per_cube = units;
        props.hardware.max_cube_dim.0 = units;
        let ((got, ran), plans) = with_properties(props, || {
            with_vector_size(vector, || {
                record_plans(|| forced(mesh, Strategy::Cooperative, &f))
            })
        });
        assert!(
            ran.iter()
                .all(|r| *r == (EVector, Strategy::Loops) || r.1 == Strategy::Cooperative),
            "{what}: {ran:?}"
        );
        let upg = |kind| -> Vec<u32> {
            plans
                .iter()
                .filter(|(k, _)| *k == kind)
                .map(|(_, p)| p.units_per_group)
                .collect()
        };
        let want = units.min(leaves);
        let factors = upg(KernelKind::GeomFactors);
        assert!(!factors.is_empty(), "{what}: no factor plan");
        assert!(
            factors.iter().all(|&u| u == want),
            "{what}, {units} units: {factors:?}"
        );
        // The pull-back's set-up is `loops`' (one unit per group), its Newton steps cooperate.
        let pull_back = upg(KernelKind::PullBack);
        assert!(
            pull_back.contains(&want) && pull_back.iter().all(|&u| u == 1 || u == want),
            "{what}, {units} units: pull-back {pull_back:?}"
        );
        got
    };
    let bitwise = gridcl_testing::bitwise_device();
    let compare = |at: &str, got: &Quantities, want: &Quantities, exact: bool| {
        for ((name, got), (_, want)) in got.iter().zip(want) {
            assert_eq!(got.len(), want.len(), "{name} of {at}");
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (g, w) in got.iter().zip(want) {
                if exact {
                    assert_eq!(g.to_bits(), w.to_bits(), "{name} of {at}: {g} vs {w}");
                } else {
                    assert!((g - w).abs() <= tol * scale, "{name} of {at}: {g} vs {w}");
                }
            }
        }
    };
    // At most the units the device runs in one cube (a power of two): on the CPU runtime a cube of more units
    // than cores spins at each `sync_cube` until the OS schedules all of them (C21).
    let own = {
        let hw = &gridcl_device::device_properties(client).hardware;
        1 << hw.max_units_per_cube.min(hw.max_cube_dim.0).ilog2()
    };
    let one = run(1, VectorSizeOverride::Scalar);
    for units in [1, 2, leaves.min(own)] {
        for vector in [VectorSizeOverride::Scalar, VectorSizeOverride::Widest] {
            let at = format!("{what}, {units} units per group, {vector:?}");
            compare(&at, &run(units, vector), &one, bitwise);
        }
    }
    let (loops, _) = forced(mesh, Strategy::Loops, &f);
    compare(&format!("{what} against loops"), &one, &loops, false);
}
